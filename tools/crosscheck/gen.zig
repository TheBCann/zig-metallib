//! Generator for the differential test bench.
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
//! Usage: crosscheck-gen --seed=<u64> <out.zig>

const std = @import("std");

const Op = enum {
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

const leaves = [_]Op{ .a, .b, .constant };
const unary = [_]Op{ .bnot, .neg, .popcount, .clz, .ctz, .bswap, .bitrev, .abs, .t8s, .t8u, .t16s, .t16u };
// Known findings are left out of the menus so new sweeps surface new problems
// rather than rediscovering these (docs/crosscheck.md "Findings"; reproducers
// in tools/crosscheck/known/): `.mul_sat` (`*|`) crashes Metal's compiler,
// `.ov_add` / `.ov_sub` overflow bits come back wrong. Add them back here to
// re-test, e.g. on a newer macOS.
const binary_inline = [_]Op{ .add, .sub, .mul, .band, .bor, .bxor, .add_sat, .sub_sat, .min_s, .max_s };
const binary_helper = [_]Op{ .shl, .ashr, .lshr, .div_trunc, .div_floor, .rem_s, .mod_s, .div_u, .rem_u, .min_u, .max_u, .ov_mul, .add8, .mul16, .mulhi_s, .mulhi_u, .avg64, .shl64hi };

/// Constants that tend to expose codegen edge cases: identities, sign
/// boundaries, single bits, all-ones masks, and shift amounts near 31/32.
const constants = [_]i32{
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
const helpers =
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

// ── Syntax tree ──────────────────────────────────────────────────────────────
// The generator builds a tree first and prints it afterwards, so the post-passes
// (every local assigned and read; which inputs were used) can walk it, and so a
// reducer can shrink a failing program structurally.

const max_vars = 4;
/// Block nesting depth: a construct inside a construct inside a construct.
const max_depth = 3;

const Cmp = enum { eq, ne, lt, le, gt, ge };
const cmps = [_]Cmp{ .eq, .ne, .lt, .le, .gt, .ge };

const Expr = union(enum) {
    a,
    b,
    v: u8,
    constant: i32,
    un: struct { op: Op, x: *Expr },
    bin: struct { op: Op, x: *Expr, y: *Expr },
    select: struct { c: *Cond, x: *Expr, y: *Expr },
};

const Cond = struct {
    cmp: Cmp,
    /// Compare the operands as u32 (through the `u` helper).
    unsigned: bool,
    x: *Expr,
    y: *Expr,
    /// An optional second comparison joined by `and` / `or`.
    rest: ?struct { is_and: bool, c: *Cond } = null,
};

const Bound = union(enum) {
    constant: u8,
    /// `u(expr) & 7`, evaluated once when the loop starts.
    runtime: *Expr,
};

const Stmt = union(enum) {
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

const Program = struct {
    nvars: u8,
    inits: []*Expr,
    stmts: []Stmt,
    ret: *Expr,
};

/// What a tree references: which inputs, and which locals are assigned / read.
const Usage = struct {
    a: bool = false,
    b: bool = false,
    assigned: [max_vars]bool = @splat(false),
    read: [max_vars]bool = @splat(false),

    fn expr(u: *Usage, e: *const Expr) void {
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
        }
    }

    fn cond(u: *Usage, c: *const Cond) void {
        u.expr(c.x);
        u.expr(c.y);
        if (c.rest) |r| u.cond(r.c);
    }

    fn stmts(u: *Usage, list: []const Stmt) void {
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

    fn program(u: *Usage, p: *const Program) void {
        for (p.inits) |e| u.expr(e);
        u.stmts(p.stmts);
        u.expr(p.ret);
    }
};

const Gen = struct {
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

    fn pick(g: *Gen, comptime set: []const Op) Op {
        return set[g.r.uintLessThan(usize, set.len)];
    }

    fn new(g: *Gen, e: Expr) !*Expr {
        const p = try g.gpa.create(Expr);
        p.* = e;
        return p;
    }

    /// `rt`: the leaf must be a runtime value (an input or a local), never a
    /// constant. An expression whose leaves are all runtime values is never
    /// comptime-known, which `break`/`continue` conditions require.
    fn leaf(g: *Gen, rt: bool) !*Expr {
        const n_choices: u32 = 2 + @as(u32, g.nvars_visible) + @as(u32, if (rt) 0 else 1);
        const k = g.r.uintLessThan(u32, n_choices);
        if (k == 0) return g.new(.a);
        if (k == 1) return g.new(.b);
        if (k < 2 + @as(u32, g.nvars_visible)) return g.new(.{ .v = @intCast(k - 2) });
        return g.new(.{ .constant = constants[g.r.uintLessThan(usize, constants.len)] });
    }

    fn expr(g: *Gen, depth: u32, rt: bool) anyerror!*Expr {
        if (depth == 0 or g.expr_budget == 0) return g.leaf(rt);
        g.expr_budget -= 1;
        const d = depth - 1;
        // Operands are generated into locals first: the order of the random
        // draws must not depend on how a struct literal is evaluated.
        switch (g.r.uintLessThan(u8, 20)) {
            0...4 => return g.leaf(rt),
            5...8 => {
                const x = try g.expr(d, rt);
                return g.new(.{ .un = .{ .op = g.pick(&unary), .x = x } });
            },
            9...12 => {
                const op = g.pick(&binary_inline);
                const x = try g.expr(d, rt);
                const y = try g.expr(d, rt);
                return g.new(.{ .bin = .{ .op = op, .x = x, .y = y } });
            },
            13...17 => {
                const op = g.pick(&binary_helper);
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
    fn cond(g: *Gen, depth: u32, allow_rest: bool) anyerror!*Cond {
        const x = try g.expr(depth, true);
        const y = try g.expr(depth, true);
        const c = try g.gpa.create(Cond);
        c.* = .{
            .cmp = cmps[g.r.uintLessThan(usize, cmps.len)],
            .unsigned = g.r.boolean(),
            .x = x,
            .y = y,
        };
        if (allow_rest and g.r.uintLessThan(u8, 10) < 3) {
            const is_and = g.r.boolean();
            c.rest = .{ .is_and = is_and, .c = try g.cond(depth, false) };
        }
        return c;
    }

    fn assign(g: *Gen) anyerror!Stmt {
        const v = g.r.uintLessThan(u8, g.nvars);
        const e = try g.expr(g.r.intRangeAtMost(u32, 1, 3), false);
        return .{ .assign = .{ .v = v, .e = e } };
    }

    fn block(g: *Gen, depth: u32, in_loop: bool, min: u32, max: u32) anyerror![]Stmt {
        var list: std.ArrayList(Stmt) = .empty;
        const n = g.r.intRangeAtMost(u32, min, max);
        var i: u32 = 0;
        while (i < n and g.stmt_budget > 0) : (i += 1) {
            g.stmt_budget -= 1;
            try list.append(g.gpa, try g.stmt(depth, in_loop));
        }
        return list.toOwnedSlice(g.gpa);
    }

    fn stmt(g: *Gen, depth: u32, in_loop: bool) anyerror!Stmt {
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

    fn program(g: *Gen) anyerror!Program {
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
    fn fixups(g: *Gen, p: *Program) anyerror!void {
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

const Printer = struct {
    w: *std.Io.Writer,
    indent: u32 = 1,
    /// Names the loop capture `k1`, `k2`, ... so nested loops never shadow
    /// each other (`i1`, `i2` would be Zig primitive types).
    loop_depth: u32 = 0,

    fn pad(p: *Printer) !void {
        for (0..p.indent) |_| try p.w.writeAll("    ");
    }

    fn expr(p: *Printer, e: *const Expr) anyerror!void {
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
                        .min_s => "@min",
                        .max_s => "@max",
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
            .select => |n| {
                try w.writeAll("(if ");
                try p.cond(n.c);
                try w.writeAll(" ");
                try p.expr(n.x);
                try w.writeAll(" else ");
                try p.expr(n.y);
                try w.writeAll(")");
            },
        }
    }

    fn cmp(p: *Printer, c: *const Cond) anyerror!void {
        const sym: []const u8 = switch (c.cmp) {
            .eq => "==",
            .ne => "!=",
            .lt => "<",
            .le => "<=",
            .gt => ">",
            .ge => ">=",
        };
        try p.w.writeAll("(");
        if (c.unsigned) try p.w.writeAll("u(");
        try p.expr(c.x);
        if (c.unsigned) try p.w.writeAll(")");
        try p.w.print(" {s} ", .{sym});
        if (c.unsigned) try p.w.writeAll("u(");
        try p.expr(c.y);
        if (c.unsigned) try p.w.writeAll(")");
        try p.w.writeAll(")");
    }

    /// Always parenthesised, so it reads as `if (...)` / `(if (...) x else y)`.
    fn cond(p: *Printer, c: *const Cond) anyerror!void {
        try p.w.writeAll("(");
        try p.cmp(c);
        if (c.rest) |r| {
            try p.w.writeAll(if (r.is_and) " and " else " or ");
            try p.cmp(r.c);
        }
        try p.w.writeAll(")");
    }

    fn block(p: *Printer, list: []const Stmt) anyerror!void {
        p.indent += 1;
        for (list) |*s| try p.stmt(s);
        p.indent -= 1;
    }

    fn stmt(p: *Printer, s: *const Stmt) anyerror!void {
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

    fn program(p: *Printer, prog: *const Program, usage: Usage) anyerror!void {
        try p.w.writeAll("pub fn body(a: i32, b: i32) i32 {\n");
        // A parameter the program never referenced must be discarded, or Zig
        // rejects it as unused.
        if (!usage.a) try p.w.writeAll("    _ = a;\n");
        if (!usage.b) try p.w.writeAll("    _ = b;\n");
        for (prog.inits, 0..) |e, i| {
            try p.pad();
            try p.w.print("var v{d}: i32 = ", .{i});
            try p.expr(e);
            try p.w.writeAll(";\n");
        }
        for (prog.stmts) |*s| try p.stmt(s);
        try p.pad();
        try p.w.writeAll("return ");
        try p.expr(prog.ret);
        try p.w.writeAll(";\n}\n");
    }
};

fn countStmts(list: []const Stmt) u32 {
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

pub fn main(init: std.process.Init) !void {
    const arena = init.arena.allocator();
    const io = init.io;

    const args = try init.minimal.args.toSlice(arena);
    var seed: ?u64 = null;
    var out_path: ?[]const u8 = null;
    for (args[1..]) |arg| {
        if (std.mem.startsWith(u8, arg, "--seed=")) {
            seed = std.fmt.parseInt(u64, arg["--seed=".len..], 10) catch {
                std.debug.print("crosscheck-gen: bad --seed value '{s}'\n", .{arg["--seed=".len..]});
                std.process.exit(2);
            };
        } else if (std.mem.startsWith(u8, arg, "-")) {
            std.debug.print("crosscheck-gen: unknown option '{s}'\n", .{arg});
            std.process.exit(2);
        } else if (out_path == null) {
            out_path = arg;
        } else {
            std.debug.print("crosscheck-gen: unexpected extra argument '{s}'\n", .{arg});
            std.process.exit(2);
        }
    }
    const usage_text = "usage: crosscheck-gen --seed=<u64> <out.zig>\n";
    const sd = seed orelse {
        std.debug.print(usage_text, .{});
        std.process.exit(2);
    };
    const path = out_path orelse {
        std.debug.print(usage_text, .{});
        std.process.exit(2);
    };

    var prng: std.Random.DefaultPrng = .init(sd);
    const r = prng.random();
    var g: Gen = .{
        .gpa = arena,
        .r = r,
        .nvars = r.intRangeAtMost(u8, 2, max_vars),
        // Bounded so a program stays readable when a seed has to be reduced.
        .expr_budget = 80,
        .stmt_budget = 40,
    };
    const prog = try g.program();
    var usage: Usage = .{};
    usage.program(&prog);

    var w: std.Io.Writer.Allocating = .init(arena);
    try w.writer.print(
        \\// Generated by tools/crosscheck/gen.zig — do not edit.
        \\// seed={d} locals={d} statements={d}
        \\//
        \\// body(a, b) is a random total i32 program. Compiled for the GPU
        \\// (src/engine/crosscheck_harness.zig) and the CPU
        \\// (tools/crosscheck/driver.zig); the driver feeds both the same inputs
        \\// and compares.
        \\pub const seed: u64 = {d};
        \\
        \\
    , .{ sd, prog.nvars, countStmts(prog.stmts), sd });
    try w.writer.writeAll(helpers);
    try w.writer.writeAll("\n");
    var printer: Printer = .{ .w = &w.writer };
    try printer.program(&prog, usage);

    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = w.written() });
}
