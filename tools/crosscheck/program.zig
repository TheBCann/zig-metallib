//! Program model for the differential test bench: the syntax tree of a generated
//! body, its random generation from a seed, and its printing as Zig source.
//! tools/crosscheck/gen.zig is the generator CLI; tools/crosscheck/reduce.zig
//! shrinks a failing program with the same tree.
//!
//! Given `--seed=<N>`, writes a `crosscheck_body.zig` whose `body(a, b)` is a
//! random i32 program: locals, assignments, `if`, bounded `for` loops, `switch`,
//! guarded `break`/`continue` and a final expression. The SAME file is compiled
//! two ways by build.zig: into
//! the GPU kernel (`src/engine/crosscheck_harness.zig` -> air-splice ->
//! .metallib, run on the Metal device) and into the host driver
//! (`tools/crosscheck/driver.zig`, run on the CPU). The driver feeds both
//! identical inputs and compares; any difference is a codegen bug in this
//! project, in Apple's compiler, or in our understanding of the IR.
//!
//! Every generated expression is defined for all 2^64 input pairs, so a
//! GPU/CPU mismatch is never undefined behaviour diverging between targets:
//!   * arithmetic wraps (`+% -% *%`) or saturates (`+| -| *|`);
//!   * shift amounts are masked to 5 bits, and `<<` wraps (no nsw/nuw);
//!   * every loop is `for` over a range of at most 8 iterations, so it ends;
//!   * `break`/`continue` conditions are runtime values, never comptime-known,
//!     or the statements after them would be unreachable code (an error);
//!   * division and remainder go through helpers that return the dividend
//!     for a zero divisor and for minInt / -1. `@rem`/`@mod` need the minInt
//!     guard too: Zig's Debug build defines @rem(minInt, -1) as 0, but its
//!     release lowering is a plain LLVM `srem`, which is undefined there.
//! The host driver is a Debug build, so its safety checks double as a UB
//! detector for the generator: a slip would panic on the CPU (bucket ERROR)
//! instead of silently diverging.
//!

const std = @import("std");

pub const Op = enum {
    // leaves
    a,
    b,
    constant,
    // unary
    bnot, // ~x
    neg, // 0 -% x (defined at minInt, unlike -x)
    popcount, // @popCount
    clz, // @clz
    ctz, // @ctz
    bswap, // @byteSwap
    bitrev, // @bitReverse
    abs, // @abs (u32 result, bit-cast back)
    // binary, written inline
    add, // +%
    sub, // -%
    mul, // *%
    band, // &
    bor, // |
    bxor, // ^
    add_sat, // +|
    sub_sat, // -|
    mul_sat, // *|
    min_s, // @min
    max_s, // @max
    // binary, through a helper emitted in the file
    shl, // x << (y & 31)
    ashr, // x >> (y & 31), arithmetic
    lshr, // logical, via u32
    div_trunc, // @divTrunc, guarded
    div_floor, // @divFloor, guarded
    rem_s, // @rem, guarded
    mod_s, // @mod, guarded
    div_u, // u32 /, guarded
    rem_u, // u32 %, guarded
    min_u, // @min on u32
    max_u, // @max on u32
    ov_add, // @addWithOverflow overflow bit
    ov_sub, // @subWithOverflow overflow bit
    ov_mul, // @mulWithOverflow overflow bit
    // integer widths, through helpers: unary truncate-then-extend
    t8s, // i8, sign-extended back
    t8u, // u8, zero-extended back
    t16s, // i16
    t16u, // u16
    // integer widths, binary
    add8, // i8 +%, sign-extended back
    mul16, // u16 *%, zero-extended back
    mulhi_s, // high 32 bits of the exact i64 product
    mulhi_u, // high 32 bits of the exact u64 product
    avg64, // (x + y) >> 1 computed in i64, so it cannot overflow
    shl64hi, // high 32 bits of a u64 shift left by (y & 63)
    // ternary
    select, // if (cond) x else y, cond = one or two comparisons
};

pub const leaves = [_]Op{ .a, .b, .constant };
pub const unary = [_]Op{ .bnot, .neg, .popcount, .clz, .ctz, .bswap, .bitrev, .abs, .t8s, .t8u, .t16s, .t16u };
// Known findings are left out of the menus so new sweeps surface new problems
// rather than rediscovering these (docs/crosscheck.md "Findings"; reproducers
// in tools/crosscheck/known/): `.mul_sat` (`*|`) crashes Metal's compiler,
// `.ov_add` / `.ov_sub` overflow bits come back wrong. Add them back here to
// re-test, e.g. on a newer macOS.
pub const binary_inline = [_]Op{ .add, .sub, .mul, .band, .bor, .bxor, .add_sat, .sub_sat, .min_s, .max_s };
pub const binary_helper = [_]Op{ .shl, .ashr, .lshr, .div_trunc, .div_floor, .rem_s, .mod_s, .div_u, .rem_u, .min_u, .max_u, .ov_mul, .add8, .mul16, .mulhi_s, .mulhi_u, .avg64, .shl64hi };

/// Constants that tend to expose codegen edge cases: identities, sign
/// boundaries, single bits, all-ones masks, and shift amounts near 31/32.
pub const constants = [_]i32{
    0,                              1,                              -1,
    2,                              -2,                             3,
    7,                              31,                             32,
    33,                             255,                            256,
    -256,                           0x7fffffff,                     -0x80000000,
    0x55555555,                     @bitCast(@as(u32, 0xaaaaaaaa)), 0x0000ffff,
    @bitCast(@as(u32, 0xffff0000)), 0x40000000,                     0x3fffffff,
};

/// Helpers every generated file carries. All are total; `inline` so neither
/// side's IR gains a call (Zig inlines them before LLVM sees the code).
/// Unreferenced ones are never analysed, so they cost nothing.
pub const helpers =
    \\const min_i32: i32 = -2147483648;
    \\inline fn u(x: i32) u32 {
    \\    return @bitCast(x);
    \\}
    \\inline fn s(x: u32) i32 {
    \\    return @bitCast(x);
    \\}
    \\inline fn sh(x: i32) u5 {
    \\    return @truncate(u(x));
    \\}
    \\inline fn shl(x: i32, y: i32) i32 {
    \\    return x << sh(y);
    \\}
    \\inline fn ashr(x: i32, y: i32) i32 {
    \\    return x >> sh(y);
    \\}
    \\inline fn lshr(x: i32, y: i32) i32 {
    \\    return s(u(x) >> sh(y));
    \\}
    \\inline fn divTrunc(x: i32, y: i32) i32 {
    \\    return if (y == 0 or (x == min_i32 and y == -1)) x else @divTrunc(x, y);
    \\}
    \\inline fn divFloor(x: i32, y: i32) i32 {
    \\    return if (y == 0 or (x == min_i32 and y == -1)) x else @divFloor(x, y);
    \\}
    \\inline fn remS(x: i32, y: i32) i32 {
    \\    return if (y == 0 or (x == min_i32 and y == -1)) x else @rem(x, y);
    \\}
    \\inline fn modS(x: i32, y: i32) i32 {
    \\    return if (y == 0 or (x == min_i32 and y == -1)) x else @mod(x, y);
    \\}
    \\inline fn divU(x: i32, y: i32) i32 {
    \\    return if (y == 0) x else s(u(x) / u(y));
    \\}
    \\inline fn remU(x: i32, y: i32) i32 {
    \\    return if (y == 0) x else s(u(x) % u(y));
    \\}
    \\inline fn minU(x: i32, y: i32) i32 {
    \\    return s(@min(u(x), u(y)));
    \\}
    \\inline fn maxU(x: i32, y: i32) i32 {
    \\    return s(@max(u(x), u(y)));
    \\}
    \\inline fn minS(x: i32, y: i32) i32 {
    \\    return @min(x, y);
    \\}
    \\inline fn maxS(x: i32, y: i32) i32 {
    \\    return @max(x, y);
    \\}
    \\inline fn ovAdd(x: i32, y: i32) i32 {
    \\    return @addWithOverflow(x, y)[1];
    \\}
    \\inline fn ovSub(x: i32, y: i32) i32 {
    \\    return @subWithOverflow(x, y)[1];
    \\}
    \\inline fn ovMul(x: i32, y: i32) i32 {
    \\    return @mulWithOverflow(x, y)[1];
    \\}
    \\inline fn t8s(x: i32) i32 {
    \\    return @as(i8, @truncate(x));
    \\}
    \\inline fn t8u(x: i32) i32 {
    \\    return @as(u8, @truncate(u(x)));
    \\}
    \\inline fn t16s(x: i32) i32 {
    \\    return @as(i16, @truncate(x));
    \\}
    \\inline fn t16u(x: i32) i32 {
    \\    return @as(u16, @truncate(u(x)));
    \\}
    \\inline fn add8(x: i32, y: i32) i32 {
    \\    return @as(i8, @truncate(x)) +% @as(i8, @truncate(y));
    \\}
    \\inline fn mul16(x: i32, y: i32) i32 {
    \\    return @as(u16, @truncate(u(x))) *% @as(u16, @truncate(u(y)));
    \\}
    \\inline fn mulhiS(x: i32, y: i32) i32 {
    \\    return @truncate((@as(i64, x) * @as(i64, y)) >> 32);
    \\}
    \\inline fn mulhiU(x: i32, y: i32) i32 {
    \\    return s(@truncate((@as(u64, u(x)) * @as(u64, u(y))) >> 32));
    \\}
    \\inline fn avg64(x: i32, y: i32) i32 {
    \\    return @truncate((@as(i64, x) + @as(i64, y)) >> 1);
    \\}
    \\inline fn shl64hi(x: i32, y: i32) i32 {
    \\    return s(@truncate((@as(u64, u(x)) << @as(u6, @truncate(u(y)))) >> 32));
    \\}
    \\
;

/// Appended only when the program has float nodes, so integer programs keep
/// their exact text.
pub const float_helpers =
    \\// Floats (docs/crosscheck-float-policy.md): the GPU flushes single-precision
    \\// denormals on input and output and canonicalises NaNs, so every float
    \\// value is flushed here too (a no-op on the GPU), which also keeps a
    \\// multiply and an add from being contracted, and NaNs leave as one bit
    \\// pattern. The flush tests the exponent bits rather than comparing, so
    \\// LLVM does not turn it into llvm.is.fpclass. Inputs are quieted: Metal's
    \\// min/max treat a signalling NaN as quiet (return the other operand),
    \\// the CPU returns NaN. IEEE leaves the zero returned by min/max
    \\// unspecified: + 0.0 turns -0.0 into +0.0 and changes nothing else.
    \\inline fn ftz(x: f32) f32 {
    \\    const b: u32 = @bitCast(x);
    \\    return if ((b & 0x7f800000) == 0) @bitCast(b & 0x80000000) else x;
    \\}
    \\inline fn fc(b: u32) f32 {
    \\    return @bitCast(b);
    \\}
    \\inline fn fin(x: i32) f32 {
    \\    const v: f32 = @bitCast(x);
    \\    return if (v != v) fc(0x7fc00000) else ftz(v);
    \\}
    \\inline fn bits(x: f32) i32 {
    \\    return if (x != x) 0x7fc00000 else @bitCast(x);
    \\}
    \\inline fn toInt(x: f32) i32 {
    \\    if (x != x) return 0;
    \\    return @intFromFloat(@min(@max(x, -2147483648.0), 2147483520.0));
    \\}
    \\inline fn fromInt(x: i32) f32 {
    \\    return ftz(@floatFromInt(x));
    \\}
    \\inline fn fadd(x: f32, y: f32) f32 {
    \\    return ftz(x + y);
    \\}
    \\inline fn fsub(x: f32, y: f32) f32 {
    \\    return ftz(x - y);
    \\}
    \\inline fn fmul(x: f32, y: f32) f32 {
    \\    return ftz(x * y);
    \\}
    \\inline fn fdiv(x: f32, y: f32) f32 {
    \\    return ftz(x / y);
    \\}
    \\inline fn fmin(x: f32, y: f32) f32 {
    \\    return @min(x, y) + 0.0;
    \\}
    \\inline fn fmax(x: f32, y: f32) f32 {
    \\    return @max(x, y) + 0.0;
    \\}
    \\inline fn ffma(x: f32, y: f32, z: f32) f32 {
    \\    return ftz(@mulAdd(f32, x, y, z));
    \\}
    \\inline fn fneg(x: f32) f32 {
    \\    return -x;
    \\}
    \\inline fn fabs(x: f32) f32 {
    \\    return @abs(x);
    \\}
    \\inline fn fsqrt(x: f32) f32 {
    \\    return ftz(@sqrt(x));
    \\}
    \\inline fn ffloor(x: f32) f32 {
    \\    return @floor(x);
    \\}
    \\inline fn fceil(x: f32) f32 {
    \\    return @ceil(x);
    \\}
    \\inline fn ftrunc(x: f32) f32 {
    \\    return @trunc(x);
    \\}
    \\
;

// ── Syntax tree ──────────────────────────────────────────────────────────────
// The generator builds a tree first and prints it afterwards, so the post-passes
// (every local assigned and read; which inputs were used) can walk it, and so a
// reducer can shrink a failing program structurally.

pub const max_vars = 4;
/// Block nesting depth: a construct inside a construct inside a construct.
pub const max_depth = 3;

pub const Cmp = enum { eq, ne, lt, le, gt, ge };

/// Float operators (docs/crosscheck-float-policy.md). Every one goes through
/// a helper that flushes denormals, which is also the contraction barrier.
pub const FOp = enum { neg, abs, sqrt, floor, ceil, trunc, add, sub, mul, div, min, max };
const funary = [_]FOp{ .neg, .abs, .sqrt, .floor, .ceil, .trunc };
const fbinary = [_]FOp{ .add, .sub, .mul, .div, .min, .max };

/// The static type of an expression node.
pub const Ty = enum { int, float };

/// Float constants as bit patterns, printed exactly: zeros of both signs,
/// small integers, 0.5, 0.1, 1/3, FLT_MIN, FLT_MAX, infinity, 2^24 (where
/// integers stop being exact) and the conversion limits.
const fconsts = [_]u32{
    0x00000000, 0x80000000, 0x3f800000, 0xbf800000, 0x3f000000, 0x40000000,
    0x40400000, 0x3dcccccd, 0x3eaaaaab, 0x00800000, 0x7f7fffff, 0x7f800000,
    0x4b800000, 0x4f000000, 0xcf000000, 0x41200000,
};
pub const cmps = [_]Cmp{ .eq, .ne, .lt, .le, .gt, .ge };

pub const Expr = union(enum) {
    a,
    b,
    v: u8,
    constant: i32,
    un: struct { op: Op, x: *Expr },
    bin: struct { op: Op, x: *Expr, y: *Expr },
    select: struct { c: *Cond, x: *Expr, y: *Expr },
    // Floats (docs/crosscheck-float-policy.md). Inputs and locals carry float
    // values as i32 bit patterns: a float subtree starts from `fin` (bits ->
    // flushed f32) or `fromint`, and ends in `fbits` (canonical bits) or
    // `ftoi` (guarded conversion). `fconst` holds the bit pattern.
    fin: *Expr,
    fconst: u32,
    fun: struct { op: FOp, x: *Expr },
    fbin: struct { op: FOp, x: *Expr, y: *Expr },
    ffma: struct { x: *Expr, y: *Expr, z: *Expr },
    fsel: struct { c: *Cond, x: *Expr, y: *Expr },
    fromint: *Expr,
    fbits: *Expr,
    ftoi: *Expr,
};

/// Whether an expression depends on an input (`a` or `b`). Locals do not
/// count: LLVM constant-folds one that is only ever assigned constants.
pub fn inputDependent(e: *const Expr) bool {
    return switch (e.*) {
        .a, .b => true,
        .v, .constant, .fconst => false,
        .un => |n| inputDependent(n.x),
        .fun => |n| inputDependent(n.x),
        .bin => |n| inputDependent(n.x) or inputDependent(n.y),
        .fbin => |n| inputDependent(n.x) or inputDependent(n.y),
        .ffma => |n| inputDependent(n.x) or inputDependent(n.y) or inputDependent(n.z),
        .select => |n| condInputDependent(n.c) or inputDependent(n.x) or inputDependent(n.y),
        .fsel => |n| condInputDependent(n.c) or inputDependent(n.x) or inputDependent(n.y),
        .fin, .fromint, .fbits, .ftoi => |x| inputDependent(x),
    };
}

fn condInputDependent(c: *const Cond) bool {
    return inputDependent(c.x) or inputDependent(c.y) or (if (c.rest) |r| condInputDependent(r.c) else false);
}

pub fn exprType(e: *const Expr) Ty {
    return switch (e.*) {
        .a, .b, .v, .constant, .un, .bin, .select, .fbits, .ftoi => .int,
        .fin, .fconst, .fun, .fbin, .ffma, .fsel, .fromint => .float,
    };
}

pub const Cond = struct {
    cmp: Cmp,
    /// Compare the operands as u32 (through the `u` helper).
    unsigned: bool,
    /// The operands are float expressions, compared as f32 (IEEE: every
    /// comparison with a NaN is false except `!=`).
    float: bool = false,
    x: *Expr,
    y: *Expr,
    /// An optional second comparison joined by `and` / `or`.
    rest: ?struct { is_and: bool, c: *Cond } = null,
};

pub const Bound = union(enum) {
    constant: u8,
    /// `u(expr) & 7`, evaluated once when the loop starts.
    runtime: *Expr,
};

pub const Stmt = union(enum) {
    assign: struct { v: u8, e: *Expr },
    if_: struct { c: *Cond, then: []Stmt, els: ?[]Stmt },
    /// `for (0..bound) |iN|`. With `idx`, the body starts with
    /// `v = v +% @as(i32, @intCast(kN))` so the capture is used (Zig rejects an
    /// unused capture); without it the capture is `_`.
    for_: struct { bound: Bound, idx: ?u8, body: []Stmt },
    /// `switch (u(sel) & 3)` with arms 0, 1, 2 and else.
    switch_: struct { sel: *Expr, arms: [4][]Stmt },
    /// `if (c) break;`, only inside a loop. `c` is runtime-known (see `expr`).
    brk: *Cond,
    cont: *Cond,
};

pub const Program = struct {
    nvars: u8,
    inits: []*Expr,
    stmts: []Stmt,
    ret: *Expr,
};

/// What a tree references: which inputs, and which locals are assigned / read.
pub const Usage = struct {
    a: bool = false,
    b: bool = false,
    assigned: [max_vars]bool = @splat(false),
    read: [max_vars]bool = @splat(false),
    /// Any float node: the printer then includes the float helpers.
    floats: bool = false,

    pub fn expr(u: *Usage, e: *const Expr) void {
        switch (e.*) {
            .a => u.a = true,
            .b => u.b = true,
            .v => |i| u.read[i] = true,
            .constant => {},
            .un => |n| u.expr(n.x),
            .bin => |n| {
                u.expr(n.x);
                u.expr(n.y);
            },
            .select => |n| {
                u.cond(n.c);
                u.expr(n.x);
                u.expr(n.y);
            },
            .fsel => |n| {
                u.cond(n.c);
                u.expr(n.x);
                u.expr(n.y);
            },
            .fin, .fromint, .fbits, .ftoi => |x| {
                u.floats = true;
                u.expr(x);
            },
            .fconst => u.floats = true,
            .fun => |n| {
                u.floats = true;
                u.expr(n.x);
            },
            .fbin => |n| {
                u.floats = true;
                u.expr(n.x);
                u.expr(n.y);
            },
            .ffma => |n| {
                u.floats = true;
                u.expr(n.x);
                u.expr(n.y);
                u.expr(n.z);
            },
        }
    }

    pub fn cond(u: *Usage, c: *const Cond) void {
        u.expr(c.x);
        u.expr(c.y);
        if (c.rest) |r| u.cond(r.c);
    }

    pub fn stmts(u: *Usage, list: []const Stmt) void {
        for (list) |*s| switch (s.*) {
            .assign => |n| {
                u.assigned[n.v] = true;
                u.expr(n.e);
            },
            .if_ => |n| {
                u.cond(n.c);
                u.stmts(n.then);
                if (n.els) |e| u.stmts(e);
            },
            .for_ => |n| {
                switch (n.bound) {
                    .constant => {},
                    .runtime => |e| u.expr(e),
                }
                if (n.idx) |v| {
                    u.assigned[v] = true;
                    u.read[v] = true;
                }
                u.stmts(n.body);
            },
            .switch_ => |n| {
                u.expr(n.sel);
                for (n.arms) |arm| u.stmts(arm);
            },
            .brk, .cont => |c| u.cond(c),
        };
    }

    pub fn program(u: *Usage, p: *const Program) void {
        for (p.inits) |e| u.expr(e);
        u.stmts(p.stmts);
        u.expr(p.ret);
    }
};

pub const Gen = struct {
    gpa: std.mem.Allocator,
    r: std.Random,
    nvars: u8,
    /// Locals a leaf may name: grows while the initialisers are generated, so
    /// `v2`'s initialiser can use `v0` and `v1` but not itself or `v3`.
    nvars_visible: u8 = 0,
    /// Expression nodes left before only leaves are emitted.
    expr_budget: u32,
    /// Statements left before blocks come out empty.
    stmt_budget: u32,
    /// Also emit the operations recorded as findings (docs/crosscheck.md), to
    /// re-test them or to give the reducer a failing program to shrink.
    include_known: bool = false,
    /// Also emit float subtrees (docs/crosscheck-float-policy.md). Off, the
    /// random draws are exactly those of the integer generator, so every
    /// integer seed keeps its program.
    floats: bool = false,

    pub fn pick(g: *Gen, comptime set: []const Op) Op {
        return set[g.r.uintLessThan(usize, set.len)];
    }

    const binary_inline_known = binary_inline ++ [_]Op{.mul_sat};
    const binary_helper_known = binary_helper ++ [_]Op{ .ov_add, .ov_sub };

    pub fn pickInline(g: *Gen) Op {
        return if (g.include_known) g.pick(&binary_inline_known) else g.pick(&binary_inline);
    }

    pub fn pickHelper(g: *Gen) Op {
        return if (g.include_known) g.pick(&binary_helper_known) else g.pick(&binary_helper);
    }

    fn pickF(g: *Gen, comptime set: []const FOp) FOp {
        return set[g.r.uintLessThan(usize, set.len)];
    }

    /// A float leaf: an input or local reinterpreted as a flushed f32, or (when
    /// a comptime-known value is allowed) a constant.
    fn fleaf(g: *Gen, rt: bool) anyerror!*Expr {
        if (!rt and g.r.uintLessThan(u8, 3) == 0) return g.new(.{ .fconst = fconsts[g.r.uintLessThan(usize, fconsts.len)] });
        const e = try g.leaf(true);
        return g.new(.{ .fin = e });
    }

    /// A float expression. `rt` as for `expr`.
    fn fexpr(g: *Gen, depth: u32, rt: bool) anyerror!*Expr {
        if (depth == 0 or g.expr_budget == 0) return g.fleaf(rt);
        g.expr_budget -= 1;
        const d = depth - 1;
        switch (g.r.uintLessThan(u8, 20)) {
            0...3 => return g.fleaf(rt),
            4...7 => {
                const op = g.pickF(&funary);
                const x = try g.fexpr(d, rt);
                return g.new(.{ .fun = .{ .op = op, .x = x } });
            },
            8...13 => {
                const op = g.pickF(&fbinary);
                const x = try g.fexpr(d, rt);
                const y = try g.fexpr(d, rt);
                return g.new(.{ .fbin = .{ .op = op, .x = x, .y = y } });
            },
            14, 15 => {
                var x = try g.fexpr(d, rt);
                const y = try g.fexpr(d, rt);
                const z = try g.fexpr(d, rt);
                // Apple's compiler folds an fma whose multiplicands are both
                // constants and flushes a denormal product, where the hardware
                // fma keeps it (docs/crosscheck.md, "Findings"). LLVM folds a
                // constant-initialised local too, so one multiplicand must
                // depend on an input. No random draw is involved.
                if (!inputDependent(x) and !inputDependent(y)) x = try g.new(.{ .fin = try g.new(.a) });
                return g.new(.{ .ffma = .{ .x = x, .y = y, .z = z } });
            },
            16, 17 => {
                const c = try g.cond(d -| 1, true);
                const x = try g.fexpr(d, rt);
                const y = try g.fexpr(d, rt);
                return g.new(.{ .fsel = .{ .c = c, .x = x, .y = y } });
            },
            else => {
                const e = try g.expr(d, rt);
                return g.new(.{ .fromint = e });
            },
        }
    }

    pub fn new(g: *Gen, e: Expr) !*Expr {
        const p = try g.gpa.create(Expr);
        p.* = e;
        return p;
    }

    /// `rt`: the leaf must be a runtime value (an input or a local), never a
    /// constant. An expression whose leaves are all runtime values is never
    /// comptime-known, which `break`/`continue` conditions require.
    pub fn leaf(g: *Gen, rt: bool) !*Expr {
        const n_choices: u32 = 2 + @as(u32, g.nvars_visible) + @as(u32, if (rt) 0 else 1);
        const k = g.r.uintLessThan(u32, n_choices);
        if (k == 0) return g.new(.a);
        if (k == 1) return g.new(.b);
        if (k < 2 + @as(u32, g.nvars_visible)) return g.new(.{ .v = @intCast(k - 2) });
        return g.new(.{ .constant = constants[g.r.uintLessThan(usize, constants.len)] });
    }

    pub fn expr(g: *Gen, depth: u32, rt: bool) anyerror!*Expr {
        if (depth == 0 or g.expr_budget == 0) return g.leaf(rt);
        g.expr_budget -= 1;
        const d = depth - 1;
        // A float subtree, back to an integer as canonical bits or a guarded
        // conversion. Only draws when floats are on, so integer seeds are
        // unchanged.
        if (g.floats and g.r.uintLessThan(u8, 10) < 3) {
            const fe = try g.fexpr(d, rt);
            return g.new(if (g.r.boolean()) .{ .fbits = fe } else .{ .ftoi = fe });
        }
        // Operands are generated into locals first: the order of the random
        // draws must not depend on how a struct literal is evaluated.
        switch (g.r.uintLessThan(u8, 20)) {
            0...4 => return g.leaf(rt),
            5...8 => {
                const x = try g.expr(d, rt);
                return g.new(.{ .un = .{ .op = g.pick(&unary), .x = x } });
            },
            9...12 => {
                const op = g.pickInline();
                const x = try g.expr(d, rt);
                const y = try g.expr(d, rt);
                return g.new(.{ .bin = .{ .op = op, .x = x, .y = y } });
            },
            13...17 => {
                const op = g.pickHelper();
                const x = try g.expr(d, rt);
                const y = try g.expr(d, rt);
                return g.new(.{ .bin = .{ .op = op, .x = x, .y = y } });
            },
            else => {
                const c = try g.cond(d -| 1, true);
                const x = try g.expr(d, rt);
                const y = try g.expr(d, rt);
                return g.new(.{ .select = .{ .c = c, .x = x, .y = y } });
            },
        }
    }

    /// One comparison, signed or unsigned, both operands runtime-known;
    /// `allow_rest` may add a second one joined by `and` / `or`.
    pub fn cond(g: *Gen, depth: u32, allow_rest: bool) anyerror!*Cond {
        const float = g.floats and g.r.boolean();
        const x = if (float) try g.fexpr(depth, true) else try g.expr(depth, true);
        const y = if (float) try g.fexpr(depth, true) else try g.expr(depth, true);
        const c = try g.gpa.create(Cond);
        c.* = .{
            .cmp = cmps[g.r.uintLessThan(usize, cmps.len)],
            .unsigned = if (float) false else g.r.boolean(),
            .float = float,
            .x = x,
            .y = y,
        };
        if (allow_rest and g.r.uintLessThan(u8, 10) < 3) {
            const is_and = g.r.boolean();
            c.rest = .{ .is_and = is_and, .c = try g.cond(depth, false) };
        }
        return c;
    }

    pub fn assign(g: *Gen) anyerror!Stmt {
        const v = g.r.uintLessThan(u8, g.nvars);
        const e = try g.expr(g.r.intRangeAtMost(u32, 1, 3), false);
        return .{ .assign = .{ .v = v, .e = e } };
    }

    pub fn block(g: *Gen, depth: u32, in_loop: bool, min: u32, max: u32) anyerror![]Stmt {
        var list: std.ArrayList(Stmt) = .empty;
        const n = g.r.intRangeAtMost(u32, min, max);
        var i: u32 = 0;
        while (i < n and g.stmt_budget > 0) : (i += 1) {
            g.stmt_budget -= 1;
            try list.append(g.gpa, try g.stmt(depth, in_loop));
        }
        return list.toOwnedSlice(g.gpa);
    }

    pub fn stmt(g: *Gen, depth: u32, in_loop: bool) anyerror!Stmt {
        const roll = g.r.uintLessThan(u8, 20);
        if (depth == 0 or roll < 9) return g.assign();
        if (roll < 13) {
            const c = try g.cond(2, true);
            const then = try g.block(depth - 1, in_loop, 1, 3);
            const els: ?[]Stmt = if (g.r.boolean()) try g.block(depth - 1, in_loop, 1, 2) else null;
            return .{ .if_ = .{ .c = c, .then = then, .els = els } };
        }
        if (roll < 16) {
            const bound: Bound = if (g.r.boolean())
                .{ .constant = g.r.intRangeAtMost(u8, 1, 6) }
            else
                .{ .runtime = try g.expr(2, true) };
            const idx: ?u8 = if (g.r.boolean()) g.r.uintLessThan(u8, g.nvars) else null;
            const body = try g.block(depth - 1, true, 1, 3);
            return .{ .for_ = .{ .bound = bound, .idx = idx, .body = body } };
        }
        if (roll < 18) {
            const sel = try g.expr(2, true);
            var arms: [4][]Stmt = undefined;
            for (&arms) |*arm| arm.* = try g.block(depth - 1, in_loop, 0, 2);
            return .{ .switch_ = .{ .sel = sel, .arms = arms } };
        }
        if (in_loop) {
            const c = try g.cond(1, true);
            return if (g.r.boolean()) .{ .brk = c } else .{ .cont = c };
        }
        return g.assign();
    }

    pub fn program(g: *Gen) anyerror!Program {
        var inits: std.ArrayList(*Expr) = .empty;
        g.nvars_visible = 0;
        for (0..g.nvars) |_| {
            try inits.append(g.gpa, try g.expr(g.r.intRangeAtMost(u32, 0, 2), false));
            g.nvars_visible += 1;
        }
        const stmts = try g.block(max_depth, false, 3, 7);
        const ret = try g.expr(g.r.intRangeAtMost(u32, 2, 4), false);
        var p = Program{ .nvars = g.nvars, .inits = try inits.toOwnedSlice(g.gpa), .stmts = stmts, .ret = ret };
        try g.fixups(&p);
        return p;
    }

    /// Zig rejects a `var` that is never assigned ("never mutated") and a local
    /// that is never read ("unused"). Give every local one assignment and fold
    /// every unread one into the result.
    pub fn fixups(g: *Gen, p: *Program) anyerror!void {
        var u: Usage = .{};
        u.program(p);
        var list: std.ArrayList(Stmt) = .empty;
        try list.appendSlice(g.gpa, p.stmts);
        for (0..p.nvars) |i| if (!u.assigned[i]) {
            const e = try g.expr(1, false);
            try list.append(g.gpa, .{ .assign = .{ .v = @intCast(i), .e = e } });
        };
        p.stmts = try list.toOwnedSlice(g.gpa);
        var after: Usage = .{};
        after.program(p);
        for (0..p.nvars) |i| if (!after.read[i]) {
            const v = try g.new(.{ .v = @intCast(i) });
            p.ret = try g.new(.{ .bin = .{ .op = .bxor, .x = p.ret, .y = v } });
        };
    }
};

// ── Printer ──────────────────────────────────────────────────────────────────

pub const Printer = struct {
    w: *std.Io.Writer,
    indent: u32 = 1,
    /// Names the loop capture `k1`, `k2`, ... so nested loops never shadow
    /// each other (`i1`, `i2` would be Zig primitive types).
    loop_depth: u32 = 0,

    pub fn pad(p: *Printer) !void {
        for (0..p.indent) |_| try p.w.writeAll("    ");
    }

    pub fn expr(p: *Printer, e: *const Expr) anyerror!void {
        const w = p.w;
        switch (e.*) {
            .a => try w.writeAll("a"),
            .b => try w.writeAll("b"),
            .v => |i| try w.print("v{d}", .{i}),
            .constant => |c| try w.print("@as(i32, {d})", .{c}),
            .un => |n| {
                const open: []const u8 = switch (n.op) {
                    .bnot => "~(",
                    .neg => "(0 -% (",
                    .popcount => "@as(i32, @popCount(",
                    .clz => "@as(i32, @clz(",
                    .ctz => "@as(i32, @ctz(",
                    .bswap => "@byteSwap(",
                    .bitrev => "@bitReverse(",
                    .abs => "s(@abs(",
                    .t8s => "t8s(",
                    .t8u => "t8u(",
                    .t16s => "t16s(",
                    .t16u => "t16u(",
                    else => unreachable,
                };
                const close: []const u8 = switch (n.op) {
                    .neg, .popcount, .clz, .ctz, .abs => "))",
                    else => ")",
                };
                try w.writeAll(open);
                try p.expr(n.x);
                try w.writeAll(close);
            },
            .bin => |n| switch (n.op) {
                .add, .sub, .mul, .band, .bor, .bxor, .add_sat, .sub_sat, .mul_sat => {
                    const sym: []const u8 = switch (n.op) {
                        .add => "+%",
                        .sub => "-%",
                        .mul => "*%",
                        .band => "&",
                        .bor => "|",
                        .bxor => "^",
                        .add_sat => "+|",
                        .sub_sat => "-|",
                        .mul_sat => "*|",
                        else => unreachable,
                    };
                    try w.writeAll("(");
                    try p.expr(n.x);
                    try w.print(" {s} ", .{sym});
                    try p.expr(n.y);
                    try w.writeAll(")");
                },
                else => {
                    const name: []const u8 = switch (n.op) {
                        // Through i32 helpers: a bare `@min`/`@max` with a
                        // comptime-known operand gets a narrowed result type
                        // (`@max(x, 0)` is u31), which `@byteSwap` rejects and
                        // which would run every operator above it at that width.
                        .min_s => "minS",
                        .max_s => "maxS",
                        .shl => "shl",
                        .ashr => "ashr",
                        .lshr => "lshr",
                        .div_trunc => "divTrunc",
                        .div_floor => "divFloor",
                        .rem_s => "remS",
                        .mod_s => "modS",
                        .div_u => "divU",
                        .rem_u => "remU",
                        .min_u => "minU",
                        .max_u => "maxU",
                        .ov_add => "ovAdd",
                        .ov_sub => "ovSub",
                        .ov_mul => "ovMul",
                        .add8 => "add8",
                        .mul16 => "mul16",
                        .mulhi_s => "mulhiS",
                        .mulhi_u => "mulhiU",
                        .avg64 => "avg64",
                        .shl64hi => "shl64hi",
                        else => unreachable,
                    };
                    try w.print("{s}(", .{name});
                    try p.expr(n.x);
                    try w.writeAll(", ");
                    try p.expr(n.y);
                    try w.writeAll(")");
                },
            },
            .select => |n| try p.selectLike(n.c, n.x, n.y),
            .fsel => |n| try p.selectLike(n.c, n.x, n.y),
            .fin => |x| {
                try w.writeAll("fin(");
                try p.expr(x);
                try w.writeAll(")");
            },
            .fconst => |bits| try w.print("fc(0x{X:0>8})", .{bits}),
            .fun => |n| {
                try w.print("{s}(", .{switch (n.op) {
                    .neg => "fneg",
                    .abs => "fabs",
                    .sqrt => "fsqrt",
                    .floor => "ffloor",
                    .ceil => "fceil",
                    .trunc => "ftrunc",
                    else => unreachable,
                }});
                try p.expr(n.x);
                try w.writeAll(")");
            },
            .fbin => |n| {
                try w.print("{s}(", .{switch (n.op) {
                    .add => "fadd",
                    .sub => "fsub",
                    .mul => "fmul",
                    .div => "fdiv",
                    .min => "fmin",
                    .max => "fmax",
                    else => unreachable,
                }});
                try p.expr(n.x);
                try w.writeAll(", ");
                try p.expr(n.y);
                try w.writeAll(")");
            },
            .ffma => |n| {
                try w.writeAll("ffma(");
                try p.expr(n.x);
                try w.writeAll(", ");
                try p.expr(n.y);
                try w.writeAll(", ");
                try p.expr(n.z);
                try w.writeAll(")");
            },
            .fromint => |x| {
                try w.writeAll("fromInt(");
                try p.expr(x);
                try w.writeAll(")");
            },
            .fbits => |x| {
                try w.writeAll("bits(");
                try p.expr(x);
                try w.writeAll(")");
            },
            .ftoi => |x| {
                try w.writeAll("toInt(");
                try p.expr(x);
                try w.writeAll(")");
            },
        }
    }

    fn selectLike(p: *Printer, c: *const Cond, x: *const Expr, y: *const Expr) anyerror!void {
        try p.w.writeAll("(if ");
        try p.cond(c);
        try p.w.writeAll(" ");
        try p.expr(x);
        try p.w.writeAll(" else ");
        try p.expr(y);
        try p.w.writeAll(")");
    }

    pub fn cmp(p: *Printer, c: *const Cond) anyerror!void {
        const sym: []const u8 = switch (c.cmp) {
            .eq => "==",
            .ne => "!=",
            .lt => "<",
            .le => "<=",
            .gt => ">",
            .ge => ">=",
        };
        // Float operands compare as f32 as they are; integer operands as i32
        // or, through `u`, as u32.
        const wrap = c.unsigned and !c.float;
        try p.w.writeAll("(");
        if (wrap) try p.w.writeAll("u(");
        try p.expr(c.x);
        if (wrap) try p.w.writeAll(")");
        try p.w.print(" {s} ", .{sym});
        if (wrap) try p.w.writeAll("u(");
        try p.expr(c.y);
        if (wrap) try p.w.writeAll(")");
        try p.w.writeAll(")");
    }

    /// Always parenthesised, so it reads as `if (...)` / `(if (...) x else y)`.
    pub fn cond(p: *Printer, c: *const Cond) anyerror!void {
        try p.w.writeAll("(");
        try p.cmp(c);
        if (c.rest) |r| {
            try p.w.writeAll(if (r.is_and) " and " else " or ");
            try p.cmp(r.c);
        }
        try p.w.writeAll(")");
    }

    pub fn block(p: *Printer, list: []const Stmt) anyerror!void {
        p.indent += 1;
        for (list) |*s| try p.stmt(s);
        p.indent -= 1;
    }

    pub fn stmt(p: *Printer, s: *const Stmt) anyerror!void {
        const w = p.w;
        switch (s.*) {
            .assign => |n| {
                try p.pad();
                try w.print("v{d} = ", .{n.v});
                try p.expr(n.e);
                try w.writeAll(";\n");
            },
            .if_ => |n| {
                try p.pad();
                try w.writeAll("if ");
                try p.cond(n.c);
                try w.writeAll(" {\n");
                try p.block(n.then);
                if (n.els) |e| {
                    try p.pad();
                    try w.writeAll("} else {\n");
                    try p.block(e);
                }
                try p.pad();
                try w.writeAll("}\n");
            },
            .for_ => |n| {
                try p.pad();
                try w.writeAll("for (0..");
                switch (n.bound) {
                    .constant => |k| try w.print("{d}", .{k}),
                    .runtime => |e| {
                        try w.writeAll("@as(usize, u(");
                        try p.expr(e);
                        try w.writeAll(") & 7)");
                    },
                }
                p.loop_depth += 1;
                if (n.idx != null) try w.print(") |k{d}| {{\n", .{p.loop_depth}) else try w.writeAll(") |_| {\n");
                p.indent += 1;
                if (n.idx) |v| {
                    try p.pad();
                    try w.print("v{d} = v{d} +% @as(i32, @intCast(k{d}));\n", .{ v, v, p.loop_depth });
                }
                for (n.body) |*b| try p.stmt(b);
                p.indent -= 1;
                p.loop_depth -= 1;
                try p.pad();
                try w.writeAll("}\n");
            },
            .switch_ => |n| {
                try p.pad();
                try w.writeAll("switch (u(");
                try p.expr(n.sel);
                try w.writeAll(") & 3) {\n");
                p.indent += 1;
                for (n.arms, 0..) |arm, k| {
                    try p.pad();
                    if (k < 3) try w.print("{d} => {{\n", .{k}) else try w.writeAll("else => {\n");
                    try p.block(arm);
                    try p.pad();
                    try w.writeAll("},\n");
                }
                p.indent -= 1;
                try p.pad();
                try w.writeAll("}\n");
            },
            .brk => |c| {
                try p.pad();
                try w.writeAll("if ");
                try p.cond(c);
                try w.writeAll(" break;\n");
            },
            .cont => |c| {
                try p.pad();
                try w.writeAll("if ");
                try p.cond(c);
                try w.writeAll(" continue;\n");
            },
        }
    }

    pub fn program(p: *Printer, prog: *const Program, usage: Usage) anyerror!void {
        try p.w.writeAll("pub fn body(a: i32, b: i32) i32 {\n");
        // A parameter the program never referenced must be discarded, or Zig
        // rejects it as unused.
        if (!usage.a) try p.w.writeAll("    _ = a;\n");
        if (!usage.b) try p.w.writeAll("    _ = b;\n");
        for (prog.inits, 0..) |e, i| {
            try p.pad();
            // A local that is never assigned must be `const` (Zig rejects a
            // `var` that is never mutated); one that is never read either is
            // discarded. Generated programs always assign and read every
            // local (Gen.fixups); the reducer's candidates may not.
            try p.w.print("{s} v{d}: i32 = ", .{ if (usage.assigned[i]) "var" else "const", i });
            try p.expr(e);
            try p.w.writeAll(";\n");
            if (!usage.assigned[i] and !usage.read[i]) {
                try p.pad();
                try p.w.print("_ = v{d};\n", .{i});
            }
        }
        for (prog.stmts) |*s| try p.stmt(s);
        try p.pad();
        try p.w.writeAll("return ");
        try p.expr(prog.ret);
        try p.w.writeAll(";\n}\n");
    }
};

pub fn countStmts(list: []const Stmt) u32 {
    var n: u32 = 0;
    for (list) |*s| {
        n += 1;
        switch (s.*) {
            .if_ => |x| {
                n += countStmts(x.then);
                if (x.els) |e| n += countStmts(e);
            },
            .for_ => |x| n += countStmts(x.body),
            .switch_ => |x| for (x.arms) |arm| {
                n += countStmts(arm);
            },
            else => {},
        }
    }
    return n;
}

pub const Options = struct {
    include_known: bool = false,
    floats: bool = false,
};

/// The program a seed denotes. Everything about it, including the number of
/// locals and the budgets, is derived from the seed, so a seed is an exact
/// reproducer.
pub fn generate(gpa: std.mem.Allocator, seed: u64, opts: Options) !Program {
    var prng: std.Random.DefaultPrng = .init(seed);
    const r = prng.random();
    var g: Gen = .{
        .gpa = gpa,
        .r = r,
        .nvars = r.intRangeAtMost(u8, 2, max_vars),
        // Bounded so a program stays readable when a seed has to be reduced.
        .expr_budget = 80,
        .stmt_budget = 40,
        .include_known = opts.include_known,
        .floats = opts.floats,
    };
    return g.program();
}

/// The complete body file: `header` (comment lines and `pub const seed`),
/// the helpers, then `body`.
pub fn render(gpa: std.mem.Allocator, prog: *const Program, header: []const u8) ![]u8 {
    var usage: Usage = .{};
    usage.program(prog);
    var w: std.Io.Writer.Allocating = .init(gpa);
    try w.writer.writeAll(header);
    try w.writer.writeAll(helpers);
    if (usage.floats) try w.writer.writeAll(float_helpers);
    try w.writer.writeAll("\n");
    var printer: Printer = .{ .w = &w.writer };
    try printer.program(prog, usage);
    return w.toOwnedSlice();
}

fn exprNodes(e: *const Expr) u32 {
    return switch (e.*) {
        .a, .b, .v, .constant, .fconst => 1,
        .un => |n| 1 + exprNodes(n.x),
        .fun => |n| 1 + exprNodes(n.x),
        .bin => |n| 1 + exprNodes(n.x) + exprNodes(n.y),
        .fbin => |n| 1 + exprNodes(n.x) + exprNodes(n.y),
        .ffma => |n| 1 + exprNodes(n.x) + exprNodes(n.y) + exprNodes(n.z),
        .select => |n| 1 + condNodes(n.c) + exprNodes(n.x) + exprNodes(n.y),
        .fsel => |n| 1 + condNodes(n.c) + exprNodes(n.x) + exprNodes(n.y),
        .fin, .fromint, .fbits, .ftoi => |x| 1 + exprNodes(x),
    };
}

fn condNodes(c: *const Cond) u32 {
    return exprNodes(c.x) + exprNodes(c.y) + (if (c.rest) |r| condNodes(r.c) else 0);
}

fn stmtNodes(list: []const Stmt) u32 {
    var n: u32 = 0;
    for (list) |*s| switch (s.*) {
        .assign => |x| n += exprNodes(x.e),
        .if_ => |x| {
            n += condNodes(x.c) + stmtNodes(x.then);
            if (x.els) |e| n += stmtNodes(e);
        },
        .for_ => |x| {
            switch (x.bound) {
                .constant => {},
                .runtime => |e| n += exprNodes(e),
            }
            n += stmtNodes(x.body);
        },
        .switch_ => |x| {
            n += exprNodes(x.sel);
            for (x.arms) |arm| n += stmtNodes(arm);
        },
        .brk, .cont => |c| n += condNodes(c),
    };
    return n;
}

/// Expression nodes in the whole program: the reducer's size measure.
pub fn countNodes(prog: *const Program) u32 {
    var n: u32 = 0;
    for (prog.inits) |e| n += exprNodes(e);
    return n + stmtNodes(prog.stmts) + exprNodes(prog.ret);
}
