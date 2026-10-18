//! Generator for the differential test bench.
//!
//! Given `--seed=<N>`, writes a `crosscheck_body.zig` whose `body(a, b)` is a
//! random i32 expression. The SAME file is compiled two ways by build.zig: into
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

/// Which inputs the expression referenced, so the caller can discard the
/// others (`_ = a;`): an unused parameter is a compile error.
const Used = struct { a: bool = false, b: bool = false };

const Gen = struct {
    w: *std.Io.Writer,
    r: std.Random,
    used: Used = .{},
    /// Nodes left before only leaves may be emitted; keeps programs small even
    /// though a select has four subtrees.
    budget: u32,

    fn pick(g: *Gen, comptime set: []const Op) Op {
        return set[g.r.uintLessThan(usize, set.len)];
    }

    fn expr(g: *Gen, depth: u32) std.Io.Writer.Error!void {
        const op: Op = if (depth == 0 or g.budget == 0) g.pick(&leaves) else blk: {
            g.budget -= 1;
            break :blk switch (g.r.uintLessThan(u8, 20)) {
                0...4 => g.pick(&leaves),
                5...8 => g.pick(&unary),
                9...12 => g.pick(&binary_inline),
                13...17 => g.pick(&binary_helper),
                else => .select,
            };
        };
        const w = g.w;
        const d = depth -| 1;
        switch (op) {
            .a => {
                g.used.a = true;
                try w.writeAll("a");
            },
            .b => {
                g.used.b = true;
                try w.writeAll("b");
            },
            .constant => try w.print("@as(i32, {d})", .{constants[g.r.uintLessThan(usize, constants.len)]}),
            .bnot => try g.wrap1("~(", d, ")"),
            .neg => try g.wrap1("(0 -% (", d, "))"),
            .popcount => try g.wrap1("@as(i32, @popCount(", d, "))"),
            .clz => try g.wrap1("@as(i32, @clz(", d, "))"),
            .ctz => try g.wrap1("@as(i32, @ctz(", d, "))"),
            .bswap => try g.wrap1("@byteSwap(", d, ")"),
            .bitrev => try g.wrap1("@bitReverse(", d, ")"),
            .abs => try g.wrap1("s(@abs(", d, "))"),
            .add, .sub, .mul, .band, .bor, .bxor, .add_sat, .sub_sat, .mul_sat => {
                const sym = switch (op) {
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
                try g.expr(d);
                try w.print(" {s} ", .{sym});
                try g.expr(d);
                try w.writeAll(")");
            },
            .min_s => try g.call2("@min", d),
            .max_s => try g.call2("@max", d),
            .shl => try g.call2("shl", d),
            .ashr => try g.call2("ashr", d),
            .lshr => try g.call2("lshr", d),
            .div_trunc => try g.call2("divTrunc", d),
            .div_floor => try g.call2("divFloor", d),
            .rem_s => try g.call2("remS", d),
            .mod_s => try g.call2("modS", d),
            .div_u => try g.call2("divU", d),
            .rem_u => try g.call2("remU", d),
            .min_u => try g.call2("minU", d),
            .max_u => try g.call2("maxU", d),
            .ov_add => try g.call2("ovAdd", d),
            .ov_sub => try g.call2("ovSub", d),
            .ov_mul => try g.call2("ovMul", d),
            .t8s => try g.wrap1("t8s(", d, ")"),
            .t8u => try g.wrap1("t8u(", d, ")"),
            .t16s => try g.wrap1("t16s(", d, ")"),
            .t16u => try g.wrap1("t16u(", d, ")"),
            .add8 => try g.call2("add8", d),
            .mul16 => try g.call2("mul16", d),
            .mulhi_s => try g.call2("mulhiS", d),
            .mulhi_u => try g.call2("mulhiU", d),
            .avg64 => try g.call2("avg64", d),
            .shl64hi => try g.call2("shl64hi", d),
            .select => {
                // The comparison operands are kept shallower than the arms.
                try w.writeAll("(if (");
                try g.cond(d -| 1);
                if (g.r.uintLessThan(u8, 10) < 3) {
                    try w.writeAll(if (g.r.boolean()) " and " else " or ");
                    try g.cond(d -| 1);
                }
                try w.writeAll(") ");
                try g.expr(d);
                try w.writeAll(" else ");
                try g.expr(d);
                try w.writeAll(")");
            },
        }
    }

    fn wrap1(g: *Gen, open: []const u8, d: u32, close: []const u8) std.Io.Writer.Error!void {
        try g.w.writeAll(open);
        try g.expr(d);
        try g.w.writeAll(close);
    }

    fn call2(g: *Gen, name: []const u8, d: u32) std.Io.Writer.Error!void {
        try g.w.print("{s}(", .{name});
        try g.expr(d);
        try g.w.writeAll(", ");
        try g.expr(d);
        try g.w.writeAll(")");
    }

    /// One comparison, signed or unsigned (operands bit-cast to u32).
    fn cond(g: *Gen, d: u32) std.Io.Writer.Error!void {
        const ops = [_][]const u8{ "==", "!=", "<", "<=", ">", ">=" };
        const cmp = ops[g.r.uintLessThan(usize, ops.len)];
        const unsigned = g.r.boolean();
        try g.w.writeAll(if (unsigned) "(u(" else "(");
        try g.expr(d);
        try g.w.print("{s} {s} ", .{ if (unsigned) ")" else "", cmp });
        try g.w.writeAll(if (unsigned) "u(" else "");
        try g.expr(d);
        try g.w.writeAll(if (unsigned) "))" else ")");
    }
};

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
    const usage = "usage: crosscheck-gen --seed=<u64> <out.zig>\n";
    const sd = seed orelse {
        std.debug.print(usage, .{});
        std.process.exit(2);
    };
    const path = out_path orelse {
        std.debug.print(usage, .{});
        std.process.exit(2);
    };

    var prng: std.Random.DefaultPrng = .init(sd);
    const r = prng.random();
    // Depth 3..6 and at most 40 interior nodes: several operators nest, and
    // the generated source stays readable when a seed has to be reduced.
    const depth = r.intRangeAtMost(u32, 3, 6);

    var expr: std.Io.Writer.Allocating = .init(arena);
    var g: Gen = .{ .w = &expr.writer, .r = r, .budget = 40 };
    try g.expr(depth);

    var w: std.Io.Writer.Allocating = .init(arena);
    try w.writer.print(
        \\// Generated by tools/crosscheck/gen.zig — do not edit.
        \\// seed={d} depth={d}
        \\//
        \\// body(a, b) is a random total i32 expression. Compiled for the GPU
        \\// (src/engine/crosscheck_harness.zig) and the CPU
        \\// (tools/crosscheck/driver.zig); the driver feeds both the same inputs
        \\// and compares.
        \\pub const seed: u64 = {d};
        \\
        \\
    , .{ sd, depth, sd });
    try w.writer.writeAll(helpers);
    try w.writer.writeAll("\npub fn body(a: i32, b: i32) i32 {\n");
    // A parameter the expression never referenced must be discarded, or Zig
    // rejects it as unused.
    if (!g.used.a) try w.writer.writeAll("    _ = a;\n");
    if (!g.used.b) try w.writer.writeAll("    _ = b;\n");
    try w.writer.print("    return {s};\n}}\n", .{expr.written()});

    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = w.written() });
}
