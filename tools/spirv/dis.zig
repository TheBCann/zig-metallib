//! SPIR-V disassembler: text in the style of Khronos' spirv-dis from a
//! reader.Module. It serves `zig build spirv-dis`, the tests' round trips
//! (asm.zig assembles this output back to the identical binary), and every
//! refusal message of the lowering, which quotes the offending instruction
//! through `instructionText`.
//!
//! Format, one instruction per line:
//!   `%N = OpXxx operands` for instructions with a result id, `OpXxx operands`
//!   otherwise. Ids print as `%N`, or as `%name` under `Options.names` when
//!   the id carries an OpName that is unique in the module and spells a valid
//!   identifier after sanitising (`[A-Za-z0-9_.]`, not digits only; other
//!   bytes become `_`). Value enums print by their Khronos name (`GLCompute`,
//!   `StorageBuffer`, `Aligned 4`, `BuiltIn GlobalInvocationId`), bit enums
//!   as `Volatile|Aligned 16` (`None` when zero), strings in double quotes
//!   with `\"` and `\\` escapes and raw newlines (as spirv-dis does for the
//!   multi-line OpSource strings naga and Zig emit), extended instructions by
//!   their GLSL.std.450 / OpenCL.std name (`Sqrt`, `s_min`) when the set is
//!   known, OpSpecConstantOp's opcode without its `Op` prefix (`IAdd`).
//!   Context-dependent numbers follow their type: signed/unsigned decimal for
//!   integers (64-bit from two words), floats as shortest round-trip decimals
//!   (`1.5`, `-340282350000000000000000000000000000000`; exponent form once
//!   |x| >= 1e16 or < 1e-5), `inf`/`-inf`, `nan`/`-nan` for the canonical
//!   quiet NaN of either sign, and raw bits (`0x7fc00001`) for any other NaN
//!   so nothing is lost; a float type with an FPEncoding operand
//!   (`OpTypeFloat 16 BFloat16KHR`, the float8 encodings) prints raw bits
//!   (`0x3f80` is bfloat16 1.0), never an IEEE reading of them; a signed integer narrower than its word prints its
//!   value whether the producer sign-extended the word or not (SPIR-V 2.2.1;
//!   asm.zig writes it back sign-extended); OpSpecConstantOp operands follow
//!   the named opcode's grammar (`CompositeExtract %v 0` has a literal index);
//!   OpSwitch case literals follow the selector's type. The header is five
//!   comment lines (`; SPIR-V`, `; Version: 1.0`, `; Generator: 0x00290900`,
//!   `; Bound: 95`, `; Schema: 0`), which asm.zig reads back so the bound and
//!   generator survive a round trip.
//! Verified against the grammar-driven reference disassembler used while
//! designing this (scratch spvdis.py) on the Zig binaries in testdata/.

const std = @import("std");
const reader = @import("reader.zig");
const spec = reader.spec;
const Module = reader.Module;
const Instruction = reader.Instruction;
const Id = reader.Id;

pub const Options = struct {
    /// Print `%name` for ids with a unique, valid OpName (see the module doc).
    names: bool = false,
    /// Print the five header comment lines.
    header: bool = true,
};

/// Names chosen for ids under `Options.names`.
pub const NameTable = struct {
    map: std.AutoHashMapUnmanaged(Id, []const u8) = .empty,

    pub fn build(a: std.mem.Allocator, m: *const Module) !NameTable {
        var t: NameTable = .{};
        // Sanitise every OpName; keep the ones that are unique.
        var count: std.StringHashMapUnmanaged(u32) = .empty;
        var chosen: std.AutoHashMapUnmanaged(Id, []const u8) = .empty;
        var it = m.names.iterator();
        while (it.next()) |e| {
            const s = try sanitize(a, e.value_ptr.*) orelse continue;
            const gop = try count.getOrPut(a, s);
            if (!gop.found_existing) gop.value_ptr.* = 0;
            gop.value_ptr.* += 1;
            try chosen.put(a, e.key_ptr.*, s);
        }
        var it2 = chosen.iterator();
        while (it2.next()) |e| {
            if (count.get(e.value_ptr.*).? == 1) try t.map.put(a, e.key_ptr.*, e.value_ptr.*);
        }
        return t;
    }

    /// `[A-Za-z0-9_.]` only, other bytes to `_`; null when empty or digits only
    /// (which would read as a numeric id).
    fn sanitize(a: std.mem.Allocator, s: []const u8) !?[]const u8 {
        if (s.len == 0) return null;
        const out = try a.alloc(u8, s.len);
        var all_digits = true;
        for (s, 0..) |c, i| {
            const ok = std.ascii.isAlphanumeric(c) or c == '_' or c == '.';
            out[i] = if (ok) c else '_';
            if (!std.ascii.isDigit(c)) all_digits = false;
        }
        if (all_digits) return null;
        return out;
    }
};

/// The whole module as text. Caller frees the result with `gpa`.
pub fn disassemble(gpa: std.mem.Allocator, m: *const Module, opts: Options) ![]u8 {
    var out = std.Io.Writer.Allocating.init(gpa);
    errdefer out.deinit();
    try write(&out.writer, gpa, m, opts);
    return out.toOwnedSlice();
}

pub fn write(w: *std.Io.Writer, gpa: std.mem.Allocator, m: *const Module, opts: Options) !void {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const names: ?NameTable = if (opts.names) try NameTable.build(arena.allocator(), m) else null;
    if (opts.header) {
        try w.print("; SPIR-V\n; Version: {d}.{d}\n; Generator: 0x{x:0>8}\n; Bound: {d}\n; Schema: {d}\n", .{ m.version_major, m.version_minor, m.generator, m.bound, m.schema });
    }
    for (m.instructions) |inst| {
        try writeInstruction(w, m, inst, if (names) |*n| n else null);
        try w.writeByte('\n');
    }
}

/// One instruction as a line of text (no newline), with names resolved
/// through `names` when given.
pub fn writeInstruction(w: *std.Io.Writer, m: *const Module, inst: Instruction, names: ?*const NameTable) !void {
    if (inst.result) |id| {
        try writeId(w, id, names);
        try w.writeAll(" = ");
    }
    try w.writeAll(@tagName(inst.op));
    if (inst.result_type) |t| {
        try w.writeByte(' ');
        try writeId(w, t, names);
    }
    const switch_type: ?reader.Type = if (inst.op == .OpSwitch and inst.operands.len > 0)
        (if (inst.operands[0].asId()) |sel| (if (m.resultTypeOf(sel)) |t| m.typeOf(t) else null) else null)
    else
        null;
    var i: usize = 0;
    while (i < inst.operands.len) : (i += 1) {
        const o = inst.operands[i];
        try w.writeByte(' ');
        switch (o.value) {
            .id => |id| try writeId(w, id, names),
            .string => |s| try writeString(w, s),
            .literal, .literal64 => {
                const v = o.asInt().?;
                switch (o.kind) {
                    .literal_context_dependent_number => try writeNumber(w, m, inst.result_type, v),
                    .literal_ext_inst_integer => {
                        if (inst.ext_set) |set| {
                            if (extInstName(set, @intCast(v))) |n| {
                                try w.writeAll(n);
                                continue;
                            }
                        }
                        try w.print("{d}", .{v});
                    },
                    .literal_spec_constant_op_integer => {
                        const op: ?spec.Opcode = if (std.math.cast(u16, v)) |n| std.enums.fromInt(spec.Opcode, n) else null;
                        if (op) |sop| {
                            try w.writeAll(@tagName(sop)["Op".len..]);
                        } else try w.print("{d}", .{v});
                    },
                    .literal_integer => {
                        if (inst.op == .OpSwitch) {
                            // Case literals follow the selector's type.
                            if (switch_type) |t| {
                                if (t == .int and t.int.signed) {
                                    try writeSigned(w, v, t.int.width);
                                    continue;
                                }
                            }
                        }
                        try w.print("{d}", .{v});
                    },
                    else => switch (o.kind.category()) {
                        .value_enum => {
                            if (reader.enumerant(o.kind, @intCast(v))) |e| try w.writeAll(e.name) else try w.print("{d}", .{v});
                        },
                        .bit_enum => try writeBitEnum(w, o.kind, @intCast(v)),
                        else => try w.print("{d}", .{v}),
                    },
                }
            },
        }
    }
}

/// `instruction` as one line, ids named when possible; for diagnostics.
pub fn instructionText(gpa: std.mem.Allocator, m: *const Module, inst: Instruction) ![]u8 {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const names = try NameTable.build(arena.allocator(), m);
    var out = std.Io.Writer.Allocating.init(gpa);
    errdefer out.deinit();
    try writeInstruction(&out.writer, m, inst, &names);
    return out.toOwnedSlice();
}

fn writeId(w: *std.Io.Writer, id: Id, names: ?*const NameTable) !void {
    if (names) |n| {
        if (n.map.get(id)) |s| {
            try w.writeByte('%');
            try w.writeAll(s);
            return;
        }
    }
    try w.print("%{d}", .{id});
}

pub fn writeString(w: *std.Io.Writer, s: []const u8) !void {
    try w.writeByte('"');
    for (s) |c| switch (c) {
        '"' => try w.writeAll("\\\""),
        '\\' => try w.writeAll("\\\\"),
        else => try w.writeByte(c),
    };
    try w.writeByte('"');
}

fn writeBitEnum(w: *std.Io.Writer, kind: spec.OperandKind, v: u32) !void {
    if (v == 0) {
        // Zig's generator drops the zero-valued enumerants (None, Relaxed);
        // Khronos spells every one of them `None` except MemorySemantics'
        // alias `Relaxed`, and spirv-dis prints `None`.
        if (reader.enumerant(kind, 0)) |e| try w.writeAll(e.name) else try w.writeAll("None");
        return;
    }
    var first = true;
    var covered: u32 = 0;
    for (reader.enumerants(kind)) |e| {
        if (e.value != 0 and (v & e.value) == e.value and (covered & e.value) != e.value) {
            if (!first) try w.writeByte('|');
            first = false;
            try w.writeAll(e.name);
            covered |= e.value;
        }
    }
    if (covered != v) {
        // Unknown bits: fall back to the number so nothing is lost.
        if (!first) try w.writeByte('|');
        try w.print("0x{x}", .{v & ~covered});
    }
}

/// `v` as a signed `width`-bit value: only the low `width` bits count, so a
/// producer that sign-extended a narrow literal to its word and one that did
/// not print the same (SPIR-V 2.2.1).
fn writeSigned(w: *std.Io.Writer, v: u64, width: u32) !void {
    if (width == 0 or width > 64) return w.print("{d}", .{v});
    const shift: u6 = @intCast(64 - width);
    const x: i64 = @as(i64, @bitCast(v << shift)) >> shift;
    try w.print("{d}", .{x});
}

/// A context-dependent number by the (scalar) type `type_id` names.
pub fn writeNumber(w: *std.Io.Writer, m: *const Module, type_id: ?Id, v: u64) !void {
    const t: ?reader.Type = if (type_id) |id| m.typeOf(id) else null;
    if (t) |ty| switch (ty) {
        .int => |i| {
            if (i.signed) return writeSigned(w, v, i.width);
            return w.print("{d}", .{v});
        },
        .float => |f| {
            // A non-IEEE encoding (bfloat16, float8) prints its raw bits:
            // decoding them as a half would print the wrong value.
            if (f.encoding != null) return w.print("0x{x}", .{reader.truncateBits(v, f.width)});
            return writeFloat(w, v, f.width);
        },
        else => {},
    };
    try w.print("{d}", .{v});
}

pub fn writeFloat(w: *std.Io.Writer, bits: u64, width: u32) !void {
    switch (width) {
        16 => {
            const b: u16 = @truncate(bits);
            const x: f16 = @bitCast(b);
            if (std.math.isNan(x)) {
                if (b == 0x7e00) return w.writeAll("nan");
                if (b == 0xfe00) return w.writeAll("-nan");
                return w.print("0x{x}", .{b});
            }
            return writeFiniteFloat(w, @floatCast(x));
        },
        32 => {
            const b: u32 = @truncate(bits);
            const x: f32 = @bitCast(b);
            if (std.math.isNan(x)) {
                if (b == 0x7fc00000) return w.writeAll("nan");
                if (b == 0xffc00000) return w.writeAll("-nan");
                return w.print("0x{x}", .{b});
            }
            if (std.math.isInf(x)) return w.writeAll(if (x < 0) "-inf" else "inf");
            const a = @abs(x);
            if (a != 0 and (a >= 1e16 or a < 1e-5)) return w.print("{e}", .{x});
            return w.print("{d}", .{x});
        },
        64 => {
            const x: f64 = @bitCast(bits);
            if (std.math.isNan(x)) {
                if (bits == 0x7ff8000000000000) return w.writeAll("nan");
                if (bits == 0xfff8000000000000) return w.writeAll("-nan");
                return w.print("0x{x}", .{bits});
            }
            return writeFiniteFloat(w, x);
        },
        else => return w.print("{d}", .{bits}),
    }
}

fn writeFiniteFloat(w: *std.Io.Writer, x: f64) !void {
    if (std.math.isInf(x)) return w.writeAll(if (x < 0) "-inf" else "inf");
    const a = @abs(x);
    if (a != 0 and (a >= 1e16 or a < 1e-5)) return w.print("{e}", .{x});
    return w.print("{d}", .{x});
}

/// The name of extended instruction `number` in `set`, or null.
pub fn extInstName(set: spec.InstructionSet, number: u32) ?[]const u8 {
    return switch (set) {
        .core => null,
        .@"GLSL.std.450" => if (std.enums.fromInt(spec.GlslOpcode, std.math.cast(u16, number) orelse return null)) |op| @tagName(op) else null,
        .@"OpenCL.std" => if (std.enums.fromInt(spec.OpenClOpcode, std.math.cast(u16, number) orelse return null)) |op| @tagName(op) else null,
    };
}

// ───────────────────────────── tests ─────────────────────────────

const testing = std.testing;

fn has(text: []const u8, needle: []const u8) bool {
    return std.mem.indexOf(u8, text, needle) != null;
}

test "every operand kind prints in spirv-dis style" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var b = try reader.WordBuilder.init(a, 40);
    b.words.items[2] = 0x00290900;
    try b.inst(.OpCapability, &.{@backingInt(spec.Capability.shader)});
    try b.instStr(.OpExtInstImport, &.{1}, "GLSL.std.450");
    try b.inst(.OpMemoryModel, &.{ 0, 1 });
    try b.instStr(.OpEntryPoint, &.{ 5, 10 }, "main");
    try b.inst(.OpExecutionMode, &.{ 10, 17, 64, 1, 1 });
    try b.instStr(.OpString, &.{13}, "k.zig");
    try b.instStr(.OpSource, &.{ 12, 2304, 13 }, "line one\n\"two\"\\");
    try b.instStr(.OpName, &.{10}, "main");
    try b.instStr(.OpName, &.{3}, "u32");
    try b.instStr(.OpName, &.{4}, "i64");
    try b.instStr(.OpName, &.{5}, "f32");
    try b.inst(.OpDecorate, &.{ 21, 11, 28 });
    try b.inst(.OpMemberDecorate, &.{ 20, 0, 35, 16 });
    try b.inst(.OpTypeVoid, &.{2});
    try b.inst(.OpTypeInt, &.{ 3, 32, 0 });
    try b.inst(.OpTypeInt, &.{ 4, 64, 1 });
    try b.inst(.OpTypeFloat, &.{ 5, 32 });
    try b.inst(.OpTypeFloat, &.{ 6, 64 });
    try b.inst(.OpTypeFloat, &.{ 26, 16 });
    try b.inst(.OpConstant, &.{ 3, 7, 4294967295 });
    try b.inst(.OpConstant, &.{ 4, 8, 0xFFFFFFFE, 0xFFFFFFFF });
    try b.inst(.OpConstant, &.{ 5, 9, @bitCast(@as(f32, 1.5)) });
    const d: u64 = @bitCast(@as(f64, -0.25));
    try b.inst(.OpConstant, &.{ 6, 11, @truncate(d), @truncate(d >> 32) });
    try b.inst(.OpConstant, &.{ 5, 27, 0x7fc00001 }); // a NaN with a payload
    try b.inst(.OpConstant, &.{ 5, 28, 0x7f800000 }); // inf
    try b.inst(.OpConstant, &.{ 5, 29, @bitCast(@as(f32, -3.4028235e38)) });
    try b.inst(.OpConstant, &.{ 5, 30, @bitCast(@as(f32, 1e-7)) });
    try b.inst(.OpConstant, &.{ 26, 31, @as(u16, @bitCast(@as(f16, 0.5))) });
    try b.inst(.OpConstant, &.{ 5, 33, 0xffc00000 }); // -nan f32
    try b.inst(.OpConstant, &.{ 26, 34, 0xfe00 }); // -nan f16
    try b.inst(.OpConstant, &.{ 6, 35, 0, 0xfff80000 }); // -nan f64
    try b.inst(.OpTypeInt, &.{ 36, 8, 1 });
    try b.inst(.OpConstant, &.{ 36, 37, 0xffffffff }); // i8 -1, sign-extended word (Zig's form)
    try b.inst(.OpConstant, &.{ 36, 38, 0xff }); // i8 -1, bare
    try b.inst(.OpTypeFunction, &.{ 17, 2 });
    try b.inst(.OpTypePointer, &.{ 16, 7, 3 });
    try b.inst(.OpSpecConstantOp, &.{ 3, 32, @backingInt(spec.Opcode.OpIAdd), 7, 7 });
    try b.inst(.OpFunction, &.{ 2, 10, 0, 17 });
    try b.inst(.OpLabel, &.{18});
    try b.inst(.OpVariable, &.{ 16, 19, 7 });
    try b.inst(.OpStore, &.{ 19, 7, 2, 4 });
    try b.inst(.OpLoad, &.{ 3, 22, 19, 3, 4 });
    try b.inst(.OpExtInst, &.{ 5, 23, 1, 31, 9 });
    try b.inst(.OpSelectionMerge, &.{ 24, 0 });
    try b.inst(.OpSwitch, &.{ 7, 24, 5, 24 });
    try b.inst(.OpLabel, &.{24});
    try b.inst(.OpReturn, &.{});
    try b.inst(.OpFunctionEnd, &.{});
    var m = try reader.read(testing.allocator, try b.bytes(), null);
    defer m.deinit();

    const text = try disassemble(testing.allocator, &m, .{});
    defer testing.allocator.free(text);
    try testing.expect(has(text, "; SPIR-V\n; Version: 1.0\n; Generator: 0x00290900\n; Bound: 40\n; Schema: 0\n"));
    try testing.expect(has(text, "\nOpCapability Shader\n"));
    try testing.expect(has(text, "\n%1 = OpExtInstImport \"GLSL.std.450\"\n"));
    try testing.expect(has(text, "\nOpMemoryModel Logical GLSL450\n"));
    try testing.expect(has(text, "\nOpEntryPoint GLCompute %10 \"main\"\n"));
    try testing.expect(has(text, "\nOpExecutionMode %10 LocalSize 64 1 1\n"));
    try testing.expect(has(text, "\n%13 = OpString \"k.zig\"\nOpSource Zig 2304 %13 \"line one\n\\\"two\\\"\\\\\"\n"));
    try testing.expect(has(text, "\nOpDecorate %21 BuiltIn GlobalInvocationId\n"));
    try testing.expect(has(text, "\nOpMemberDecorate %20 0 Offset 16\n"));
    try testing.expect(has(text, "\n%3 = OpTypeInt 32 0\n"));
    try testing.expect(has(text, "\n%7 = OpConstant %3 4294967295\n"));
    try testing.expect(has(text, "\n%8 = OpConstant %4 -2\n"));
    try testing.expect(has(text, "\n%9 = OpConstant %5 1.5\n"));
    try testing.expect(has(text, "\n%11 = OpConstant %6 -0.25\n"));
    try testing.expect(has(text, "\n%27 = OpConstant %5 0x7fc00001\n"));
    try testing.expect(has(text, "\n%28 = OpConstant %5 inf\n"));
    try testing.expect(has(text, "\n%29 = OpConstant %5 -3.4028235e38\n"));
    try testing.expect(has(text, "\n%30 = OpConstant %5 1e-7\n"));
    try testing.expect(has(text, "\n%31 = OpConstant %26 0.5\n"));
    try testing.expect(has(text, "\n%33 = OpConstant %5 -nan\n"));
    try testing.expect(has(text, "\n%34 = OpConstant %26 -nan\n"));
    try testing.expect(has(text, "\n%35 = OpConstant %6 -nan\n"));
    try testing.expect(has(text, "\n%37 = OpConstant %36 -1\n"));
    try testing.expect(has(text, "\n%38 = OpConstant %36 -1\n"));
    try testing.expect(has(text, "\n%32 = OpSpecConstantOp %3 IAdd %7 %7\n"));
    try testing.expect(has(text, "\n%10 = OpFunction %2 None %17\n"));
    try testing.expect(has(text, "\n%19 = OpVariable %16 Function\n"));
    try testing.expect(has(text, "\nOpStore %19 %7 Aligned 4\n"));
    try testing.expect(has(text, "\n%22 = OpLoad %3 %19 Volatile|Aligned 4\n"));
    try testing.expect(has(text, "\n%23 = OpExtInst %5 %1 Sqrt %9\n"));
    try testing.expect(has(text, "\nOpSelectionMerge %24 None\n"));
    try testing.expect(has(text, "\nOpSwitch %7 %24 5 %24\n"));
    try testing.expect(has(text, "\nOpFunctionEnd\n"));

    // Names mode resolves unique OpNames.
    const named = try disassemble(testing.allocator, &m, .{ .names = true, .header = false });
    defer testing.allocator.free(named);
    try testing.expect(has(named, "%main = OpFunction %2 None %17\n"));
    try testing.expect(has(named, "%8 = OpConstant %i64 -2\n"));
    try testing.expect(has(named, "OpName %main \"main\"\n"));
    try testing.expect(!has(named, "; SPIR-V"));

    // One instruction for a diagnostic.
    const line = try instructionText(testing.allocator, &m, m.instructions[m.instructions.len - 4]);
    defer testing.allocator.free(line);
    try testing.expectEqualStrings("OpSwitch %7 %24 5 %24", line);
}

test "duplicate and unusable OpNames fall back to numeric ids" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var b = try reader.WordBuilder.init(a, 10);
    try b.instStr(.OpName, &.{3}, "x");
    try b.instStr(.OpName, &.{4}, "x");
    try b.instStr(.OpName, &.{5}, "77");
    try b.instStr(.OpName, &.{6}, "a b");
    try b.instStr(.OpName, &.{7}, "");
    try b.inst(.OpTypeInt, &.{ 3, 32, 0 });
    try b.inst(.OpTypeInt, &.{ 4, 32, 1 });
    try b.inst(.OpTypeInt, &.{ 5, 8, 0 });
    try b.inst(.OpTypeInt, &.{ 6, 16, 0 });
    try b.inst(.OpTypeInt, &.{ 7, 64, 0 });
    var m = try reader.read(testing.allocator, try b.bytes(), null);
    defer m.deinit();
    const named = try disassemble(testing.allocator, &m, .{ .names = true, .header = false });
    defer testing.allocator.free(named);
    try testing.expect(has(named, "%3 = OpTypeInt 32 0\n"));
    try testing.expect(has(named, "%4 = OpTypeInt 32 1\n"));
    try testing.expect(has(named, "%5 = OpTypeInt 8 0\n"));
    try testing.expect(has(named, "%a_b = OpTypeInt 16 0\n"));
    try testing.expect(has(named, "%7 = OpTypeInt 64 0\n"));
}

test "the Zig binaries disassemble to the reference text" {
    // Spot checks against the reference disassembler's output recorded in
    // the design (scratch spvdis.py on the same files).
    const desc = @embedFile("testdata/desc.spv");
    var m = try reader.read(testing.allocator, desc, null);
    defer m.deinit();
    const text = try disassemble(testing.allocator, &m, .{ .names = true });
    defer testing.allocator.free(text);
    try testing.expect(has(text, "OpEntryPoint GLCompute %desc.crosscheckKernel \"crosscheckKernel\" %global_invocation_id\n"));
    try testing.expect(has(text, "OpExecutionMode %desc.crosscheckKernel LocalSize 64 1 1\n"));
    try testing.expect(has(text, "OpMemoryModel PhysicalStorageBuffer64 GLSL450\n"));
    try testing.expect(has(text, "OpDecorate %in0 NonWritable\n"));
    try testing.expect(has(text, "%out = OpVariable %140 StorageBuffer\n"));
    try testing.expect(has(text, "%gid = OpLoad %u32 %11 Aligned 4\n"));
    try testing.expect(has(text, "%36 = OpPhi %u32 %32 %30 %34 %35\n"));
    try testing.expect(has(text, "%96 = OpFunctionCall %i32 %desc.body %76 %93\n"));

    const cf = @embedFile("testdata/cf.spv");
    var mc = try reader.read(testing.allocator, cf, null);
    defer mc.deinit();
    const ct = try disassemble(testing.allocator, &mc, .{ .names = true });
    defer testing.allocator.free(ct);
    try testing.expect(has(ct, "OpSwitch %166 %175 0 %172 1 %173 2 %174\n"));
    try testing.expect(has(ct, "%335 = OpConstant %f32 1.5\n"));
    try testing.expect(has(ct, "OpDecorate %555 ArrayStride 16384\n"));
    try testing.expect(has(ct, "%tile = OpVariable %549 Workgroup\n"));
    try testing.expect(has(ct, "OpLoopMerge %142 %143 None\n"));
    try testing.expect(has(ct, " = OpExtInst %f32 %224 Sqrt "));
    try testing.expect(has(ct, " = OpExtInst %f32 %224 Fma "));
    // Two parameters both named "x": numeric.
    try testing.expect(has(ct, "%cf.body = OpFunction %i32 None %538\n%"));
    try testing.expect(!has(ct, "%x = OpFunctionParameter"));

    const ko = @embedFile("testdata/k-spirv64-opencl.spv");
    var mo = try reader.read(testing.allocator, ko, null);
    defer mo.deinit();
    const ot = try disassemble(testing.allocator, &mo, .{ .names = true });
    defer testing.allocator.free(ot);
    try testing.expect(has(ot, "OpMemoryModel Physical64 OpenCL\n"));
    try testing.expect(has(ot, "OpEntryPoint Kernel %k.kernel \"kernel\"\n"));
    try testing.expect(has(ot, "%5 = OpTypePointer CrossWorkgroup %u32\n"));
    try testing.expect(has(ot, "%54 = OpInBoundsPtrAccessChain %5 %51 %47\n"));
}

test "constants of an FPEncoding float print raw bits, IEEE halves print values" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var b = try reader.WordBuilder.init(a, 10);
    try b.inst(.OpTypeFloat, &.{ 1, 16, @backingInt(spec.FPEncoding.b_float16khr) });
    try b.inst(.OpTypeFloat, &.{ 2, 16 });
    try b.inst(.OpTypeFloat, &.{ 3, 8, @backingInt(spec.FPEncoding.float8e5m2ext) });
    try b.inst(.OpConstant, &.{ 1, 4, 0x3f80 }); // bfloat16 1.0 (a half 1.875)
    try b.inst(.OpConstant, &.{ 2, 5, 0x3c00 }); // half 1.0
    try b.inst(.OpConstant, &.{ 3, 6, 0x3c }); // float8 E5M2 1.0
    var m = try reader.read(testing.allocator, try b.bytes(), null);
    defer m.deinit();
    const text = try disassemble(testing.allocator, &m, .{ .header = false });
    defer testing.allocator.free(text);
    try testing.expect(has(text, "%1 = OpTypeFloat 16 BFloat16KHR\n"));
    try testing.expect(has(text, "%3 = OpTypeFloat 8 Float8E5M2EXT\n"));
    try testing.expect(has(text, "%4 = OpConstant %1 0x3f80\n"));
    try testing.expect(has(text, "%5 = OpConstant %2 1\n"));
    try testing.expect(has(text, "%6 = OpConstant %3 0x3c\n"));
}
