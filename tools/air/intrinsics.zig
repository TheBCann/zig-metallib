//! Intrinsic table for the assembler: which `llvm.*` calls Metal's runtime
//! compiler takes as they are, which must be renamed to their `air.*`
//! equivalent, and which have to be expanded into plain instructions.
//!
//! Everything here is a Metal runtime fact (DESIGN.md D5-12a/12b); the
//! frontend (`xcrun metal`) accepts every form below, only pipeline
//! creation tells them apart:
//!
//!   * `llvm.sin/cos/tan/log10/pow/powi/minimum/maximum/nearbyint/roundeven`
//!     abort MTLCompilerService ("unable to legalize instruction") at every
//!     width; `llvm.exp10` and `llvm.ldexp` fail with "Undefined symbols".
//!     Their `air.fast_*` spellings all build (research/math-builtins/
//!     RESULTS_TABLE.tsv, _results/r02.txt findings 3/5/6, r11 finding 13).
//!     `llvm.nearbyint`/`llvm.roundeven` become `llvm.rint`, which passes.
//!     `llvm.powi.<T>.i32` has no AIR form; it becomes `air.fast_pow.<T>`
//!     with the exponent converted by `sitofp`. `llvm.ldexp.<T>.i32` keeps
//!     its integer parameter but Apple spells it `air.fast_ldexp.<T>`.
//!   * Every `llvm.vector.reduce.*` crashes the GPU backend (r11 finding
//!     15, dumps/air_acceptance.txt reduce_*); a scalar chain of
//!     `extractelement` + the operation passes (shuffle_reduce_manual_fadd),
//!     as do `air.any.vNi1` / `air.all.vNi1` for bool vectors
//!     (air_any_v4i1_control). An op with no expansion here (`fmaximum`,
//!     `fminimum`, ...) is refused: the `llvm.*` name would either crash
//!     the legalizer or fail with "Undefined symbols" (r02 finding 1).
//!   * `bitcast <N x i1> to iN` (Zig's `@bitCast` of a bool vector and the
//!     form LLVM folds `@reduce(.Or)` into) crashes the backend
//!     (bool_vec_bitcast_i4_zext); `zext <N x i1> to <N x i32>` followed
//!     by per-lane integer ops passes (bool_vec_zext_v4i32_reduce_manual).
//!     The reverse `bitcast iN to <N x i1>` (`@as(@Vector(N, bool),
//!     @bitCast(uN))`) crashes the same way (review-a2/h_bitcast_rev.ll);
//!     it becomes splat + `lshr` by the lane index + `and 1` + `icmp ne 0`
//!     on `<N x i32>`, all of which pass.
//!   * Accepted unchanged: sqrt exp exp2 log log2 fabs floor ceil trunc
//!     round rint fma fmuladd minnum maxnum copysign ctlz cttz ctpop bswap
//!     bitreverse umin umax smin smax abs *.sat *.with.overflow assume trap
//!     memcpy memset memmove (r11 finding 14).
//!   * `llvm.memcpy/memmove/memset.pX.pY` carry the operand address spaces
//!     in their name; after constant relocation the suffix must be rewritten
//!     to match (DESIGN.md D1).
//!   * The texture intrinsics (`air.sample_texture_2d.v4f32`, ...) have one
//!     canonical signature each (`texture_intrinsics`, from Apple's own
//!     declarations in research/_results/r03.json ir_patterns 0-3, 11-14):
//!     the sample/read family returns `{ <4 x float>, i8 }` (colour plus
//!     sparse-residency status) and takes the sampler as `ptr addrspace(2)`.
//!     Zig can spell neither (extern structs may not hold vectors, nvptx
//!     has no constant address space), so a Zig declaration returns the
//!     bare vector and passes `ptr`; the assembler and the splice replace
//!     the declaration by the canonical one and give every call an
//!     `extractvalue 0` (DESIGN.md D5-12c). Metal tolerates the bare
//!     declaration (h07_v4f32_only passes pipeline creation) but the
//!     canonical form is what Apple emits and what the pixel checks run.

const std = @import("std");
const Allocator = std.mem.Allocator;
const Builder = std.zig.llvm.Builder;
const Type = Builder.Type;
const Value = Builder.Value;
const WipFunction = Builder.WipFunction;

pub const Error = error{ OutOfMemory, Unsupported };

/// `llvm.vector.reduce.<op>` operations. fadd/fmul take a start operand.
pub const ReduceOp = enum { fadd, fmul, fmax, fmin, add, mul, @"and", @"or", xor, umax, umin, smax, smin };

pub const Rename = struct {
    /// The callee to declare and call instead (allocated by `classify`).
    name: []const u8,
    /// `llvm.powi`: the i32 exponent becomes the float type of the base.
    exponent_to_float: bool = false,
};

pub const Lowering = union(enum) {
    /// Call the intrinsic as declared.
    pass,
    /// Call another function instead.
    rename: Rename,
    /// Expand into a scalar chain.
    reduce: ReduceOp,
    /// No accepted form exists: the assembler reports the line with this
    /// hint (error.Unsupported).
    refuse: []const u8,

    pub fn deinit(l: Lowering, gpa: Allocator) void {
        switch (l) {
            .rename => |r| gpa.free(r.name),
            else => {},
        }
    }
};

const RenameRule = struct {
    /// Segment after `llvm.`.
    op: []const u8,
    /// Replacement for `llvm.<op>`.
    to: []const u8,
    /// Keep only the first suffix segment (`f32.i32` -> `f32`).
    drop_int_suffix: bool = false,
    exponent_to_float: bool = false,
};

const rename_rules = [_]RenameRule{
    .{ .op = "sin", .to = "air.fast_sin" },
    .{ .op = "cos", .to = "air.fast_cos" },
    .{ .op = "tan", .to = "air.fast_tan" },
    .{ .op = "log10", .to = "air.fast_log10" },
    .{ .op = "exp10", .to = "air.fast_exp10" },
    .{ .op = "pow", .to = "air.fast_pow" },
    .{ .op = "powi", .to = "air.fast_pow", .drop_int_suffix = true, .exponent_to_float = true },
    .{ .op = "ldexp", .to = "air.fast_ldexp", .drop_int_suffix = true },
    .{ .op = "nearbyint", .to = "llvm.rint" },
    .{ .op = "roundeven", .to = "llvm.rint" },
    .{ .op = "minimum", .to = "air.fast_fmin" },
    .{ .op = "maximum", .to = "air.fast_fmax" },
};

/// Decide how a call to `name` is lowered. Non-intrinsics pass through.
pub fn classify(gpa: Allocator, name: []const u8) Allocator.Error!Lowering {
    const prefix = "llvm.";
    if (!std.mem.startsWith(u8, name, prefix)) return .pass;
    const rest = name[prefix.len..];
    if (std.mem.startsWith(u8, rest, "vector.reduce.")) {
        const after = rest["vector.reduce.".len..];
        const op_end = std.mem.findScalar(u8, after, '.') orelse after.len;
        if (std.meta.stringToEnum(ReduceOp, after[0..op_end])) |op| return .{ .reduce = op };
        return .{ .refuse = "this llvm.vector.reduce.* op has no scalar expansion here and every llvm.vector.reduce.* crashes Metal's compiler; reduce the lanes with extractelement + scalar ops" };
    }
    const op_end = std.mem.findScalar(u8, rest, '.') orelse return .pass;
    const op = rest[0..op_end];
    var suffix = rest[op_end + 1 ..];
    for (rename_rules) |rule| {
        if (!std.mem.eql(u8, rule.op, op)) continue;
        if (rule.drop_int_suffix) {
            if (std.mem.findScalar(u8, suffix, '.')) |dot| suffix = suffix[0..dot];
        }
        const new_name = try std.mem.concat(gpa, u8, &.{ rule.to, ".", suffix });
        return .{ .rename = .{ .name = new_name, .exponent_to_float = rule.exponent_to_float } };
    }
    return .pass;
}

pub fn isMemIntrinsic(name: []const u8) bool {
    return std.mem.startsWith(u8, name, "llvm.memcpy.") or
        std.mem.startsWith(u8, name, "llvm.memmove.") or
        std.mem.startsWith(u8, name, "llvm.memset.");
}

/// Whether Apple's compiler marks `name` `convergent`: the threadgroup and
/// SIMD-group barriers and every SIMD-group / quad-group collective
/// (`air.simd_sum.f32`, `air.simd_shuffle_down.f32`, `air.simd_any`, ...;
/// research/_results/r10.json claim 2: `declare float @air.simd_sum.f32(float)
/// #6`, `attributes #6 = { convergent ... }`, and research/verify-compute-kernels/
/// v1.air.ll). Zig cannot put the attribute on an `extern fn`, so LLVM's
/// jump threading may duplicate a call into divergent blocks, where a
/// barrier desynchronises the threadgroup and a `simd_sum` only sums the
/// lanes that took the same arm (review-b/r4/helper/check.log: `out[0]=1
/// out[1]=33 want 528 1552`). The assembler refuses such call sites
/// (`FunctionState.convergentCallAllowed`); the table here decides which
/// callees it inspects.
pub fn isConvergent(name: []const u8) bool {
    return std.mem.eql(u8, name, "air.wg.barrier") or
        std.mem.eql(u8, name, "air.simdgroup.barrier") or
        std.mem.startsWith(u8, name, "air.simd_") or
        std.mem.startsWith(u8, name, "air.quad_");
}

/// `llvm.memcpy.p0.p0.i64` with pointer operands in address spaces
/// `spaces` (in operand order) -> `llvm.memcpy.p0.p2.i64`.
pub fn memIntrinsicName(gpa: Allocator, name: []const u8, spaces: []const u32) Allocator.Error![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(gpa);
    var it = std.mem.splitScalar(u8, name, '.');
    var k: usize = 0;
    var first = true;
    while (it.next()) |seg| {
        if (!first) try out.append(gpa, '.');
        first = false;
        if (seg.len >= 2 and seg[0] == 'p' and allDigits(seg[1..]) and k < spaces.len) {
            try out.print(gpa, "p{d}", .{spaces[k]});
            k += 1;
        } else try out.appendSlice(gpa, seg);
    }
    return out.toOwnedSlice(gpa);
}

fn allDigits(s: []const u8) bool {
    if (s.len == 0) return false;
    for (s) |ch| if (!std.ascii.isDigit(ch)) return false;
    return true;
}

/// Function type for a renamed intrinsic: identical to the `llvm.*`
/// declaration except that `powi`'s exponent takes the base's type.
pub fn renamedFnType(b: *Builder, rename: Rename, llvm_ty: Type) Error!Type {
    if (!rename.exponent_to_float) return llvm_ty;
    const params = llvm_ty.functionParameters(b);
    if (params.len != 2) return error.Unsupported;
    const new_params = [_]Type{ params[0], params[0] };
    return b.fnType(llvm_ty.functionReturn(b), &new_params, .normal);
}

/// Arguments for a renamed intrinsic call: `powi`'s exponent is converted
/// with `sitofp` (and splat when the base is a vector).
pub fn renamedArgs(b: *Builder, wip: *WipFunction, rename: Rename, args: []const Value, out: []Value) Error!void {
    std.debug.assert(out.len == args.len);
    @memcpy(out, args);
    if (!rename.exponent_to_float) return;
    if (args.len != 2) return error.Unsupported;
    const base_ty = args[0].typeOfWip(wip);
    const scalar = base_ty.scalarType(b);
    if (!scalar.isFloatingPoint() or !args[1].typeOfWip(wip).isInteger(b)) return error.Unsupported;
    const as_float = try wip.cast(.sitofp, args[1], scalar, "");
    out[1] = if (base_ty == scalar) as_float else try splat(b, wip, as_float, base_ty);
}

/// `insertelement` + `shufflevector` broadcast of a scalar.
fn splat(b: *Builder, wip: *WipFunction, scalar: Value, vec_ty: Type) Error!Value {
    const one = try b.vectorType(.normal, 1, scalar.typeOfWip(wip));
    const poison = (try b.poisonConst(one)).toValue();
    const seed = try wip.insertElement(poison, scalar, (try b.intConst(.i64, 0)).toValue(), "");
    const mask_ty = try b.vectorType(.normal, vec_ty.vectorLen(b), .i32);
    return wip.shuffleVector(seed, poison, (try b.zeroInitConst(mask_ty)).toValue(), "");
}

/// Expand `llvm.vector.reduce.<op>` into `extractelement` + scalar ops.
/// `declarer.declare(name, fn_ty)` must return the callee `Value` for a
/// function declared on demand (`llvm.maxnum.<T>`, `air.any.vNi1`, ...).
pub fn lowerReduce(b: *Builder, wip: *WipFunction, declarer: anytype, op: ReduceOp, args: []const Value) Error!Value {
    const has_start = op == .fadd or op == .fmul;
    if (args.len != @as(usize, if (has_start) 2 else 1)) return error.Unsupported;
    const vec = args[args.len - 1];
    const vec_ty = vec.typeOfWip(wip);
    if (!vec_ty.isVector(b)) return error.Unsupported;
    const elem_ty = vec_ty.childType(b);
    const n = vec_ty.vectorLen(b);
    if (n == 0) return error.Unsupported;
    const is_float = elem_ty.isFloatingPoint();
    const is_int = elem_ty.isInteger(b);

    // Bool vectors: `or` is "any lane set", `and` is "all lanes set".
    if (elem_ty == .i1 and (op == .@"or" or op == .@"and")) {
        var name_buf: [32]u8 = undefined;
        const name = std.fmt.bufPrint(&name_buf, "air.{s}.v{d}i1", .{ if (op == .@"or") "any" else "all", n }) catch unreachable;
        const fn_ty = try b.fnType(.i1, &.{vec_ty}, .normal);
        const callee = try declarer.declare(name, fn_ty);
        return wip.call(.normal, .ccc, .none, fn_ty, callee, &.{vec}, "");
    }

    switch (op) {
        .fadd, .fmul, .fmax, .fmin => if (!is_float) return error.Unsupported,
        else => if (!is_int) return error.Unsupported,
    }
    if (has_start and args[0].typeOfWip(wip) != elem_ty) return error.Unsupported;

    // fmax/fmin chain through llvm.maxnum/minnum, which Metal accepts.
    var minmax_ty: Type = .none;
    var minmax_callee: Value = undefined;
    if (op == .fmax or op == .fmin) {
        const suffix: []const u8 = switch (elem_ty) {
            .half => "f16",
            .float => "f32",
            .double => "f64",
            else => return error.Unsupported,
        };
        var name_buf: [32]u8 = undefined;
        const name = std.fmt.bufPrint(&name_buf, "llvm.{s}.{s}", .{ if (op == .fmax) "maxnum" else "minnum", suffix }) catch unreachable;
        minmax_ty = try b.fnType(elem_ty, &.{ elem_ty, elem_ty }, .normal);
        minmax_callee = try declarer.declare(name, minmax_ty);
    }

    var acc: Value = undefined;
    var i: u32 = 0;
    if (has_start) {
        acc = args[0];
    } else {
        acc = try wip.extractElement(vec, (try b.intConst(.i64, 0)).toValue(), "");
        i = 1;
    }
    while (i < n) : (i += 1) {
        const lane = try wip.extractElement(vec, (try b.intConst(.i64, i)).toValue(), "");
        acc = switch (op) {
            .fadd => try wip.bin(.fadd, acc, lane, ""),
            .fmul => try wip.bin(.fmul, acc, lane, ""),
            .add => try wip.bin(.add, acc, lane, ""),
            .mul => try wip.bin(.mul, acc, lane, ""),
            .@"and" => try wip.bin(.@"and", acc, lane, ""),
            .@"or" => try wip.bin(.@"or", acc, lane, ""),
            .xor => try wip.bin(.xor, acc, lane, ""),
            .fmax, .fmin => try wip.call(.normal, .ccc, .none, minmax_ty, minmax_callee, &.{ acc, lane }, ""),
            .umax, .umin, .smax, .smin => blk: {
                const cond: Builder.IntegerCondition = switch (op) {
                    .umax => .ugt,
                    .umin => .ult,
                    .smax => .sgt,
                    .smin => .slt,
                    else => unreachable,
                };
                const keep = try wip.icmp(cond, acc, lane, "");
                break :blk try wip.select(.normal, keep, acc, lane, "");
            },
        };
    }
    return acc;
}

/// `bitcast <N x i1> %v to iN` without the bitcast: widen the lanes to
/// i32, shift lane i to bit i, `or` them together, narrow to `dest_ty`.
pub fn lowerBoolVectorBitcast(b: *Builder, wip: *WipFunction, vec: Value, dest_ty: Type) Error!Value {
    const vec_ty = vec.typeOfWip(wip);
    if (!vec_ty.isVector(b) or vec_ty.childType(b) != .i1 or !dest_ty.isInteger(b)) return error.Unsupported;
    const n = vec_ty.vectorLen(b);
    if (n == 0 or n > 32) return error.Unsupported;
    const wide_ty = try b.vectorType(.normal, n, .i32);
    const wide = try wip.cast(.zext, vec, wide_ty, "");
    var acc = try wip.extractElement(wide, (try b.intConst(.i64, 0)).toValue(), "");
    var i: u32 = 1;
    while (i < n) : (i += 1) {
        const lane = try wip.extractElement(wide, (try b.intConst(.i64, i)).toValue(), "");
        const shifted = try wip.bin(.shl, lane, (try b.intConst(.i32, i)).toValue(), "");
        acc = try wip.bin(.@"or", acc, shifted, "");
    }
    if (dest_ty == .i32) return acc;
    if (dest_ty.scalarBits(b) < 32) return wip.cast(.trunc, acc, dest_ty, "");
    return wip.cast(.zext, acc, dest_ty, "");
}

/// `bitcast iN %v to <N x i1>` without the bitcast: bring the integer to
/// i32, splat it, shift lane i right by i, mask bit 0 and compare with
/// zero, so lane i holds bit i of the source.
pub fn lowerIntToBoolVector(b: *Builder, wip: *WipFunction, int: Value, vec_ty: Type) Error!Value {
    const int_ty = int.typeOfWip(wip);
    if (!vec_ty.isVector(b) or vec_ty.childType(b) != .i1 or int_ty.isVector(b) or !int_ty.isInteger(b)) return error.Unsupported;
    const n = vec_ty.vectorLen(b);
    if (n == 0 or n > 32) return error.Unsupported;
    const bits = int_ty.scalarBits(b);
    const wide = if (bits == 32) int else if (bits < 32) try wip.cast(.zext, int, .i32, "") else try wip.cast(.trunc, int, .i32, "");
    const wide_ty = try b.vectorType(.normal, n, .i32);
    const lanes = try splat(b, wip, wide, wide_ty);
    var shift_consts: [32]Builder.Constant = undefined;
    for (shift_consts[0..n], 0..) |*sc, i| sc.* = try b.intConst(.i32, i);
    const shifts = (try b.vectorConst(wide_ty, shift_consts[0..n])).toValue();
    const shifted = try wip.bin(.lshr, lanes, shifts, "");
    const ones = (try b.splatConst(wide_ty, try b.intConst(.i32, 1))).toValue();
    const masked = try wip.bin(.@"and", shifted, ones, "");
    return wip.icmp(.ne, masked, (try b.zeroInitConst(wide_ty)).toValue(), "");
}

// ── Texture intrinsics (DESIGN.md D5-12c) ────────────────────────────────────

/// The canonical (Apple) signature of a texture intrinsic, as LLVM IR
/// text so the splice can print it and the assembler can parse it into
/// Builder types.
pub const TextureIntrinsic = struct {
    name: []const u8,
    ret: []const u8,
    params: []const []const u8,
    /// Position of the `ptr addrspace(2)` sampler operand; a constant
    /// global passed there is a constexpr sampler that must be listed in
    /// `!air.sampler_states`.
    sampler_param: ?usize = null,
    /// The canonical return is `{ <colour>, i8 }`: a Zig declaration that
    /// returns the bare colour vector is replaced and every call site gets
    /// `extractvalue 0` bound to its result name.
    wraps_status: bool = false,
};

const tex_ptr = "ptr addrspace(1)";
const sampler_ptr = "ptr addrspace(2)";
const v4f32_status = "{ <4 x float>, i8 }";

/// research/_results/r03.json ir_patterns 0 (sample), 5 (sample_grad),
/// 8 (gather), 9 (read + get_read_sampler), 10 (get_width/get_height),
/// 11 (write); opaque-pointer forms proven in texture-sampling/h01, h03,
/// h05 and by `zig build check` on the sample shader.
pub const texture_intrinsics = [_]TextureIntrinsic{
    .{ .name = "air.sample_texture_2d.v4f32", .ret = v4f32_status, .params = &.{ tex_ptr, sampler_ptr, "<2 x float>", "i1", "<2 x i32>", "i1", "float", "float", "i32" }, .sampler_param = 1, .wraps_status = true },
    .{ .name = "air.sample_texture_2d_grad.v4f32", .ret = v4f32_status, .params = &.{ tex_ptr, sampler_ptr, "<2 x float>", "<2 x float>", "<2 x float>", "float", "i1", "<2 x i32>", "i32" }, .sampler_param = 1, .wraps_status = true },
    .{ .name = "air.gather_texture_2d.v4f32", .ret = v4f32_status, .params = &.{ tex_ptr, sampler_ptr, "<2 x float>", "i1", "<2 x i32>", "i32", "i32" }, .sampler_param = 1, .wraps_status = true },
    .{ .name = "air.read_texture_2d.v4f32", .ret = v4f32_status, .params = &.{ tex_ptr, sampler_ptr, "<2 x i32>", "<2 x i32>", "i32", "i32" }, .sampler_param = 1, .wraps_status = true },
    .{ .name = "air.get_read_sampler", .ret = sampler_ptr, .params = &.{} },
    .{ .name = "air.write_texture_2d.v4f32", .ret = "void", .params = &.{ tex_ptr, "<2 x i32>", "<4 x float>", "i32", "i32" } },
    .{ .name = "air.get_width_texture_2d", .ret = "i32", .params = &.{ tex_ptr, "i32" } },
    .{ .name = "air.get_height_texture_2d", .ret = "i32", .params = &.{ tex_ptr, "i32" } },
};

pub fn textureIntrinsic(name: []const u8) ?*const TextureIntrinsic {
    for (&texture_intrinsics) |*ti| if (std.mem.eql(u8, ti.name, name)) return ti;
    return null;
}

/// `declare RET @name(P1, P2, ...)` for the canonical signature.
pub fn textureDeclareText(gpa: Allocator, ti: *const TextureIntrinsic) Allocator.Error![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(gpa);
    try out.print(gpa, "declare {s} @{s}(", .{ ti.ret, ti.name });
    for (ti.params, 0..) |p, i| {
        if (i > 0) try out.appendSlice(gpa, ", ");
        try out.appendSlice(gpa, p);
    }
    try out.append(gpa, ')');
    return out.toOwnedSlice(gpa);
}

// ── Tests ───────────────────────────────────────────────────────────────────

test "texture intrinsic table: canonical signatures, sampler positions, declare text" {
    const gpa = std.testing.allocator;
    const sample = textureIntrinsic("air.sample_texture_2d.v4f32").?;
    try std.testing.expectEqual(9, sample.params.len);
    try std.testing.expectEqual(1, sample.sampler_param.?);
    try std.testing.expect(sample.wraps_status);
    const text = try textureDeclareText(gpa, sample);
    defer gpa.free(text);
    try std.testing.expectEqualStrings("declare { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)", text);
    const read = textureIntrinsic("air.read_texture_2d.v4f32").?;
    try std.testing.expectEqualStrings("ptr addrspace(2)", read.params[1]);
    const rs = textureIntrinsic("air.get_read_sampler").?;
    try std.testing.expectEqualStrings("ptr addrspace(2)", rs.ret);
    try std.testing.expect(rs.sampler_param == null and !rs.wraps_status);
    const write = textureIntrinsic("air.write_texture_2d.v4f32").?;
    try std.testing.expectEqualStrings("void", write.ret);
    try std.testing.expectEqual(5, write.params.len);
    const wtext = try textureDeclareText(gpa, write);
    defer gpa.free(wtext);
    try std.testing.expectEqualStrings("declare void @air.write_texture_2d.v4f32(ptr addrspace(1), <2 x i32>, <4 x float>, i32, i32)", wtext);
    try std.testing.expectEqualStrings("i32", textureIntrinsic("air.get_height_texture_2d").?.ret);
    try std.testing.expect(textureIntrinsic("air.sample_texture_2d.v4f16") == null);
    try std.testing.expect(textureIntrinsic("air.discard_fragment") == null);
    try std.testing.expect(textureIntrinsic("llvm.sqrt.f32") == null);
}

test "rename table: every suffix, powi/ldexp suffix handling, pass-through" {
    const gpa = std.testing.allocator;
    const cases = [_]struct { in: []const u8, out: []const u8 }{
        .{ .in = "llvm.sin.f32", .out = "air.fast_sin.f32" },
        .{ .in = "llvm.sin.f16", .out = "air.fast_sin.f16" },
        .{ .in = "llvm.cos.v4f32", .out = "air.fast_cos.v4f32" },
        .{ .in = "llvm.tan.v3f32", .out = "air.fast_tan.v3f32" },
        .{ .in = "llvm.log10.v2f32", .out = "air.fast_log10.v2f32" },
        .{ .in = "llvm.exp10.v2f16", .out = "air.fast_exp10.v2f16" },
        .{ .in = "llvm.pow.v4f16", .out = "air.fast_pow.v4f16" },
        .{ .in = "llvm.powi.f32.i32", .out = "air.fast_pow.f32" },
        .{ .in = "llvm.powi.v4f32.i32", .out = "air.fast_pow.v4f32" },
        .{ .in = "llvm.ldexp.f32.i32", .out = "air.fast_ldexp.f32" },
        .{ .in = "llvm.nearbyint.f32", .out = "llvm.rint.f32" },
        .{ .in = "llvm.roundeven.v4f32", .out = "llvm.rint.v4f32" },
        .{ .in = "llvm.minimum.f32", .out = "air.fast_fmin.f32" },
        .{ .in = "llvm.maximum.v4f16", .out = "air.fast_fmax.v4f16" },
    };
    for (cases) |cs| {
        const l = try classify(gpa, cs.in);
        defer l.deinit(gpa);
        try std.testing.expectEqualStrings(cs.out, l.rename.name);
        try std.testing.expectEqual(std.mem.startsWith(u8, cs.in, "llvm.powi."), l.rename.exponent_to_float);
    }
    const untouched = [_][]const u8{
        "llvm.sqrt.f32",         "llvm.exp.f32",       "llvm.exp2.v4f32",             "llvm.log.f16",     "llvm.log2.f32",
        "llvm.fabs.v3f32",       "llvm.floor.f32",     "llvm.ceil.f32",               "llvm.trunc.f32",   "llvm.round.f32",
        "llvm.rint.f32",         "llvm.fma.v4f32",     "llvm.fmuladd.f32",            "llvm.minnum.f32",  "llvm.maxnum.v4f16",
        "llvm.copysign.f32",     "llvm.ctlz.i32",      "llvm.cttz.i32",               "llvm.ctpop.i32",   "llvm.bswap.i32",
        "llvm.bitreverse.i32",   "llvm.umin.i32",      "llvm.umax.i64",               "llvm.smin.i32",    "llvm.smax.i32",
        "llvm.abs.i32",          "llvm.uadd.sat.i32",  "llvm.sadd.with.overflow.i32", "llvm.assume",      "llvm.trap",
        "llvm.memcpy.p0.p1.i64", "llvm.memset.p0.i64", "llvm.memmove.p0.p0.i64",      "air.fast_sin.f32", "my.helper",
    };
    for (untouched) |name| {
        const l = try classify(gpa, name);
        defer l.deinit(gpa);
        try std.testing.expect(l == .pass);
    }
    // A reduce op without an expansion is refused, never passed through
    // (the llvm.* name crashes or fails to link at pipeline creation).
    for ([_][]const u8{ "llvm.vector.reduce.fmaximum.v4f32", "llvm.vector.reduce.fminimum.v2f16", "llvm.vector.reduce.bogus.v4f32" }) |name| {
        const l = try classify(gpa, name);
        defer l.deinit(gpa);
        try std.testing.expect(l == .refuse);
    }
    for ([_][]const u8{ "fadd", "fmul", "fmax", "fmin", "add", "mul", "and", "or", "xor", "umax", "umin", "smax", "smin" }, 0..) |op, i| {
        var buf: [64]u8 = undefined;
        const name = try std.fmt.bufPrint(&buf, "llvm.vector.reduce.{s}.v4i32", .{op});
        const l = try classify(gpa, name);
        try std.testing.expectEqual(@as(ReduceOp, @fromBackingInt(@intCast(i))), l.reduce);
    }
}

test "memcpy suffix rewrite follows the operand address spaces" {
    const gpa = std.testing.allocator;
    const a = try memIntrinsicName(gpa, "llvm.memcpy.p0.p0.i64", &.{ 0, 2 });
    defer gpa.free(a);
    try std.testing.expectEqualStrings("llvm.memcpy.p0.p2.i64", a);
    const m = try memIntrinsicName(gpa, "llvm.memset.p0.i64", &.{3});
    defer gpa.free(m);
    try std.testing.expectEqualStrings("llvm.memset.p3.i64", m);
    const mv = try memIntrinsicName(gpa, "llvm.memmove.p1.p1.i32", &.{ 1, 1 });
    defer gpa.free(mv);
    try std.testing.expectEqualStrings("llvm.memmove.p1.p1.i32", mv);
    try std.testing.expect(isMemIntrinsic("llvm.memcpy.p0.p0.i64"));
    try std.testing.expect(!isMemIntrinsic("llvm.sin.f32"));
}

test "convergent table: barriers and every SIMD-group collective, nothing else" {
    for ([_][]const u8{
        "air.wg.barrier",       "air.simdgroup.barrier",     "air.simd_sum.f32",                  "air.simd_max.f32",
        "air.simd_min.f32",     "air.simd_product.f32",      "air.simd_prefix_inclusive_sum.f32", "air.simd_broadcast.f32",
        "air.simd_shuffle.f32", "air.simd_shuffle_down.f32", "air.simd_shuffle_up.f32",           "air.simd_shuffle_xor.f32",
        "air.simd_sum.u.i32",   "air.simd_any",              "air.simd_all",                      "air.simd_ballot",
        "air.quad_sum.f32",
    }) |name| try std.testing.expect(isConvergent(name));
    for ([_][]const u8{
        "air.atomic.global.add.u.i32", "air.atomic.local.add.u.i32",  "air.atomic.fence", "air.fast_sin.f32",
        "llvm.sqrt.f32",               "air.sample_texture_2d.v4f32", "my_shader.syncTg", "air.simdgroup_barrier",
    }) |name| try std.testing.expect(!isConvergent(name));
}

/// Test harness: one function with the given parameter types, whose body
/// the test fills through `lower`, rendered by the Builder's printer.
const TestDeclarer = struct {
    b: *Builder,
    pub fn declare(d: *TestDeclarer, name: []const u8, fn_ty: Type) Error!Value {
        const func = try d.b.addFunction(fn_ty, try d.b.strtabString(name), .default);
        return func.ptr(d.b).global.toConst().toValue();
    }
};

/// A type by shape; `Type` handles are interned per Builder, so the tests
/// describe types and build them inside the harness's Builder. `len == 0`
/// is the scalar itself; `bits != 0` makes an integer of that width.
const Shape = struct { len: u32 = 0, elem: Type = .void, bits: u24 = 0 };

fn shapeType(b: *Builder, s: Shape) !Type {
    const elem = if (s.bits != 0) try b.intType(s.bits) else s.elem;
    return if (s.len == 0) elem else b.vectorType(.normal, s.len, elem);
}

/// Builds `f(<params>)` whose body is `lower(...)` and returns the
/// Builder's own rendering of the module.
fn testLower(gpa: Allocator, comptime lower: anytype, ctx: anytype, ret: Shape, params: []const Shape) ![]u8 {
    var b = try Builder.init(.{ .allocator = gpa, .strip = true, .name = "intrinsics-test" });
    defer b.deinit();
    var param_types: [4]Type = undefined;
    for (params, 0..) |p, i| param_types[i] = try shapeType(&b, p);
    const fn_ty = try b.fnType(try shapeType(&b, ret), param_types[0..params.len], .normal);
    const func = try b.addFunction(fn_ty, try b.strtabString("f"), .default);
    var wip = try WipFunction.init(&b, .{ .function = func, .strip = true });
    defer wip.deinit();
    const entry = try wip.block(0, "entry");
    wip.cursor = .{ .block = entry };
    var declarer = TestDeclarer{ .b = &b };
    const result = try lower(&b, &wip, &declarer, ctx);
    _ = try wip.ret(result);
    try wip.finish();
    var aw: std.Io.Writer.Allocating = .init(gpa);
    defer aw.deinit();
    try b.print(&aw.writer);
    return aw.toOwnedSlice();
}

fn countOccurrences(hay: []const u8, needle: []const u8) usize {
    var n: usize = 0;
    var i: usize = 0;
    while (std.mem.findPos(u8, hay, i, needle)) |p| : (i = p + needle.len) n += 1;
    return n;
}

const ReduceCase = struct { op: ReduceOp, has_start: bool };
fn reduceBody(b: *Builder, wip: *WipFunction, d: *TestDeclarer, ctx: ReduceCase) Error!Value {
    if (ctx.has_start) return lowerReduce(b, wip, d, ctx.op, &.{ wip.arg(0), wip.arg(1) });
    return lowerReduce(b, wip, d, ctx.op, &.{wip.arg(0)});
}

test "reduce lowering: fadd with start operand, fmax via maxnum, integer min via icmp+select, bool any/all" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();

    // fadd(start, <4 x float>): four lanes, four fadds, no llvm.vector.reduce.
    const fadd = try testLower(gpa, reduceBody, ReduceCase{ .op = .fadd, .has_start = true }, .{ .elem = .float }, &.{ .{ .elem = .float }, .{ .len = 4, .elem = .float } });
    try std.testing.expectEqual(4, countOccurrences(fadd, "extractelement <4 x float>"));
    try std.testing.expectEqual(4, countOccurrences(fadd, "fadd float"));
    try std.testing.expect(std.mem.find(u8, fadd, "vector.reduce") == null);

    // fmax(<3 x float>): lane 0 seeds, two maxnum calls, declared on demand.
    const fmax = try testLower(gpa, reduceBody, ReduceCase{ .op = .fmax, .has_start = false }, .{ .elem = .float }, &.{.{ .len = 3, .elem = .float }});
    try std.testing.expectEqual(3, countOccurrences(fmax, "extractelement <3 x float>"));
    try std.testing.expectEqual(2, countOccurrences(fmax, "call float @llvm.maxnum.f32("));
    try std.testing.expect(std.mem.find(u8, fmax, "declare float @llvm.maxnum.f32(float %0, float %1)") != null);

    // fmin(<4 x half>) uses the half suffix.
    const fmin = try testLower(gpa, reduceBody, ReduceCase{ .op = .fmin, .has_start = false }, .{ .elem = .half }, &.{.{ .len = 4, .elem = .half }});
    try std.testing.expectEqual(3, countOccurrences(fmin, "call half @llvm.minnum.f16("));

    // umin(<4 x i32>): icmp ult + select per remaining lane.
    const umin = try testLower(gpa, reduceBody, ReduceCase{ .op = .umin, .has_start = false }, .{ .elem = .i32 }, &.{.{ .len = 4, .elem = .i32 }});
    try std.testing.expectEqual(4, countOccurrences(umin, "extractelement <4 x i32>"));
    try std.testing.expectEqual(3, countOccurrences(umin, "icmp ult i32"));
    try std.testing.expectEqual(3, countOccurrences(umin, "select i1"));
    const smax = try testLower(gpa, reduceBody, ReduceCase{ .op = .smax, .has_start = false }, .{ .elem = .i32 }, &.{.{ .len = 2, .elem = .i32 }});
    try std.testing.expectEqual(1, countOccurrences(smax, "icmp sgt i32"));

    // add/xor chains.
    const add = try testLower(gpa, reduceBody, ReduceCase{ .op = .add, .has_start = false }, .{ .elem = .i32 }, &.{.{ .len = 4, .elem = .i32 }});
    try std.testing.expectEqual(3, countOccurrences(add, "add i32"));
    const xor = try testLower(gpa, reduceBody, ReduceCase{ .op = .xor, .has_start = false }, .{ .elem = .i1 }, &.{.{ .len = 4, .elem = .i1 }});
    try std.testing.expectEqual(3, countOccurrences(xor, "xor i1"));

    // or/and of <N x i1> become air.any / air.all.
    const any = try testLower(gpa, reduceBody, ReduceCase{ .op = .@"or", .has_start = false }, .{ .elem = .i1 }, &.{.{ .len = 4, .elem = .i1 }});
    try std.testing.expect(std.mem.find(u8, any, "declare i1 @air.any.v4i1(<4 x i1> %0)") != null);
    try std.testing.expect(std.mem.find(u8, any, "call i1 @air.any.v4i1(<4 x i1> %0)") != null);
    try std.testing.expectEqual(0, countOccurrences(any, "extractelement"));
    const all = try testLower(gpa, reduceBody, ReduceCase{ .op = .@"and", .has_start = false }, .{ .elem = .i1 }, &.{.{ .len = 3, .elem = .i1 }});
    try std.testing.expect(std.mem.find(u8, all, "call i1 @air.all.v3i1(<3 x i1> %0)") != null);

    // Wrong shapes are refused, never asserted.
    try std.testing.expectError(error.Unsupported, testLower(gpa, reduceBody, ReduceCase{ .op = .fadd, .has_start = false }, .{ .elem = .float }, &.{.{ .len = 4, .elem = .float }}));
    try std.testing.expectError(error.Unsupported, testLower(gpa, reduceBody, ReduceCase{ .op = .add, .has_start = false }, .{ .elem = .float }, &.{.{ .len = 4, .elem = .float }}));
    try std.testing.expectError(error.Unsupported, testLower(gpa, reduceBody, ReduceCase{ .op = .fmax, .has_start = false }, .{ .elem = .i32 }, &.{.{ .len = 4, .elem = .i32 }}));
}

const RenameCase = struct { name: []const u8 };
fn renameBody(b: *Builder, wip: *WipFunction, d: *TestDeclarer, ctx: RenameCase) Error!Value {
    const l = try classify(b.gpa, ctx.name);
    defer l.deinit(b.gpa);
    const base_ty = wip.arg(0).typeOfWip(wip);
    const llvm_ty = try b.fnType(base_ty, &.{ base_ty, .i32 }, .normal);
    const new_ty = try renamedFnType(b, l.rename, llvm_ty);
    var args: [2]Value = undefined;
    try renamedArgs(b, wip, l.rename, &.{ wip.arg(0), wip.arg(1) }, &args);
    const callee = try d.declare(l.rename.name, new_ty);
    return wip.call(.normal, .ccc, .none, new_ty, callee, &args, "");
}

test "powi becomes air.fast_pow with a sitofp exponent (splat for vectors); ldexp keeps its i32" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();
    const scalar = try testLower(gpa, renameBody, RenameCase{ .name = "llvm.powi.f32.i32" }, .{ .elem = .float }, &.{ .{ .elem = .float }, .{ .elem = .i32 } });
    try std.testing.expect(std.mem.find(u8, scalar, "declare float @air.fast_pow.f32(float %0, float %1)") != null);
    try std.testing.expect(std.mem.find(u8, scalar, "sitofp i32 %1 to float") != null);
    const vector = try testLower(gpa, renameBody, RenameCase{ .name = "llvm.powi.v4f32.i32" }, .{ .len = 4, .elem = .float }, &.{ .{ .len = 4, .elem = .float }, .{ .elem = .i32 } });
    try std.testing.expect(std.mem.find(u8, vector, "declare <4 x float> @air.fast_pow.v4f32(<4 x float> %0, <4 x float> %1)") != null);
    try std.testing.expect(std.mem.find(u8, vector, "shufflevector <1 x float>") != null);
    const ldexp = try testLower(gpa, renameBody, RenameCase{ .name = "llvm.ldexp.f32.i32" }, .{ .elem = .float }, &.{ .{ .elem = .float }, .{ .elem = .i32 } });
    try std.testing.expect(std.mem.find(u8, ldexp, "declare float @air.fast_ldexp.f32(float %0, i32 %1)") != null);
    try std.testing.expect(std.mem.find(u8, ldexp, "sitofp") == null);
}

fn bitcastBody(b: *Builder, wip: *WipFunction, d: *TestDeclarer, dest_bits: u24) Error!Value {
    _ = d;
    return lowerBoolVectorBitcast(b, wip, wip.arg(0), try b.intType(dest_bits));
}

test "bool-vector bitcast packs lanes with zext/shl/or and narrows to the target width" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();
    const text = try testLower(gpa, bitcastBody, @as(u24, 4), .{ .bits = 4 }, &.{.{ .len = 4, .elem = .i1 }});
    try std.testing.expect(std.mem.find(u8, text, "bitcast") == null);
    try std.testing.expect(std.mem.find(u8, text, "zext <4 x i1> %0 to <4 x i32>") != null);
    try std.testing.expectEqual(4, countOccurrences(text, "extractelement <4 x i32>"));
    try std.testing.expectEqual(3, countOccurrences(text, "shl i32"));
    try std.testing.expectEqual(3, countOccurrences(text, "or i32"));
    try std.testing.expect(std.mem.find(u8, text, "trunc i32 %") != null);
    try std.testing.expect(std.mem.find(u8, text, "to i4") != null);
}

fn intToBoolBody(b: *Builder, wip: *WipFunction, d: *TestDeclarer, lanes: u32) Error!Value {
    _ = d;
    return lowerIntToBoolVector(b, wip, wip.arg(0), try b.vectorType(.normal, lanes, .i1));
}

test "int-to-bool-vector bitcast becomes splat + lshr + and + icmp on <N x i32>" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();
    // i4 -> <4 x i1>: zext to i32 first.
    const text = try testLower(gpa, intToBoolBody, @as(u32, 4), .{ .len = 4, .elem = .i1 }, &.{.{ .bits = 4 }});
    try std.testing.expect(std.mem.find(u8, text, "bitcast") == null);
    try std.testing.expect(std.mem.find(u8, text, "zext i4 %0 to i32") != null);
    try std.testing.expect(std.mem.find(u8, text, "shufflevector <1 x i32>") != null);
    try std.testing.expect(std.mem.find(u8, text, "lshr <4 x i32> %") != null);
    try std.testing.expect(std.mem.find(u8, text, "<i32 0, i32 1, i32 2, i32 3>") != null);
    try std.testing.expect(std.mem.find(u8, text, "and <4 x i32> %") != null);
    try std.testing.expect(std.mem.find(u8, text, "icmp ne <4 x i32> %") != null);
    try std.testing.expect(std.mem.find(u8, text, "ret <4 x i1> %") != null);
    // i32 -> <3 x i1>: no widening; i64 -> <2 x i1>: narrowed to i32.
    const from32 = try testLower(gpa, intToBoolBody, @as(u32, 3), .{ .len = 3, .elem = .i1 }, &.{.{ .elem = .i32 }});
    try std.testing.expect(std.mem.find(u8, from32, "zext") == null);
    try std.testing.expect(std.mem.find(u8, from32, "icmp ne <3 x i32> %") != null);
    const from64 = try testLower(gpa, intToBoolBody, @as(u32, 2), .{ .len = 2, .elem = .i1 }, &.{.{ .elem = .i64 }});
    try std.testing.expect(std.mem.find(u8, from64, "trunc i64 %0 to i32") != null);
    // Wrong shapes are refused: float source, more than 32 lanes.
    try std.testing.expectError(error.Unsupported, testLower(gpa, intToBoolBody, @as(u32, 4), .{ .len = 4, .elem = .i1 }, &.{.{ .elem = .float }}));
    try std.testing.expectError(error.Unsupported, testLower(gpa, intToBoolBody, @as(u32, 64), .{ .len = 64, .elem = .i1 }, &.{.{ .elem = .i64 }}));
}
