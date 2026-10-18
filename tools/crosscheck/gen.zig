//! Generator for the differential test bench (milestone 1).
//!
//! Given `--seed=<N>`, writes a `crosscheck_body.zig` whose `body(a, b)` is a random
//! pure i32 expression. The SAME file is compiled two ways by build.zig: into
//! the GPU kernel (`src/engine/crosscheck_harness.zig` -> air-splice -> .metallib,
//! run on the Metal device) and into the host driver (`tools/crosscheck/driver.zig`,
//! run on the CPU). The driver feeds both identical inputs and compares; any
//! difference is a codegen bug in this project, in Apple's compiler, or in our
//! understanding of the IR.
//!
//! Milestone 1 uses only TOTAL operators (`+% -% *% & | ^ ~`): every generated
//! expression is defined for all 2^64 input pairs, so a GPU/CPU mismatch is
//! never Zig undefined behaviour diverging between targets. Shifts, division
//! and wider types are milestone 2, after their exact semantics are pinned.
//!
//! Usage: crosscheck-gen --seed=<u64> <out.zig>

const std = @import("std");

/// A node of the expression tree. Leaves reference the two inputs or a
/// constant; internal nodes apply one total operator.
const Op = enum {
    a,
    b,
    constant,
    add, // +%
    sub, // -%
    mul, // *%
    band, // &
    bor, // |
    bxor, // ^
    bnot, // ~ (unary)
    neg, // 0 -% x (unary, defined at minInt unlike -x)

    fn arity(op: Op) u8 {
        return switch (op) {
            .a, .b, .constant => 0,
            .bnot, .neg => 1,
            else => 2,
        };
    }
};

const leaves = [_]Op{ .a, .b, .constant };
const binary = [_]Op{ .add, .sub, .mul, .band, .bor, .bxor };
const unary = [_]Op{ .bnot, .neg };

/// A grab-bag of constants that tends to expose codegen edge cases: identities,
/// sign boundaries, single bits and all-ones masks.
const constants = [_]i32{
    0,                              1,           -1,                             2,
    -2,                             3,           7,                              255,
    256,                            -256,        0x7fffffff,                     -0x80000000,
    0x55555555,                     0x5555_5556, @bitCast(@as(u32, 0xaaaaaaaa)), 0x0000ffff,
    @bitCast(@as(u32, 0xffff0000)), 0x40000000,  0x3fffffff,                     1 << 30,
};

/// Which inputs the generated expression referenced, so the caller can discard
/// the others (`_ = a;`): a Zig function parameter that is never used is a
/// compile error, and a constant-only or single-input body is common.
const Used = struct { a: bool = false, b: bool = false };

fn genExpr(w: *std.Io.Writer, r: std.Random, depth: u32, used: *Used) std.Io.Writer.Error!void {
    // At depth 0 force a leaf; otherwise pick any node, weighted so trees stay
    // finite (leaves are ~40% of the menu).
    const op: Op = if (depth == 0)
        leaves[r.uintLessThan(usize, leaves.len)]
    else switch (r.uintLessThan(u8, 10)) {
        0, 1, 2 => leaves[r.uintLessThan(usize, leaves.len)],
        3, 4 => unary[r.uintLessThan(usize, unary.len)],
        else => binary[r.uintLessThan(usize, binary.len)],
    };

    switch (op) {
        .a => {
            used.a = true;
            try w.writeAll("a");
        },
        .b => {
            used.b = true;
            try w.writeAll("b");
        },
        .constant => {
            const c = constants[r.uintLessThan(usize, constants.len)];
            // Print as a parenthesised i32 literal; negative values need the cast
            // so the expression stays i32 and never a comptime_int overflow.
            try w.print("@as(i32, {d})", .{c});
        },
        .bnot => {
            try w.writeAll("~(");
            try genExpr(w, r, depth - 1, used);
            try w.writeAll(")");
        },
        .neg => {
            try w.writeAll("(0 -% (");
            try genExpr(w, r, depth - 1, used);
            try w.writeAll("))");
        },
        else => {
            const sym = switch (op) {
                .add => "+%",
                .sub => "-%",
                .mul => "*%",
                .band => "&",
                .bor => "|",
                .bxor => "^",
                else => unreachable,
            };
            try w.writeAll("(");
            try genExpr(w, r, depth - 1, used);
            try w.print(" {s} ", .{sym});
            try genExpr(w, r, depth - 1, used);
            try w.writeAll(")");
        },
    }
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
    const s = seed orelse {
        std.debug.print("usage: crosscheck-gen --seed=<u64> <out.zig>\n", .{});
        std.process.exit(2);
    };
    const path = out_path orelse {
        std.debug.print("usage: crosscheck-gen --seed=<u64> <out.zig>\n", .{});
        std.process.exit(2);
    };

    var prng: std.Random.DefaultPrng = .init(s);
    const r = prng.random();
    // Expression depth 3..6: deep enough to nest several operators, bounded so
    // the generated source stays small.
    const depth = r.intRangeAtMost(u32, 3, 6);

    // Build the expression first, tracking which inputs it used.
    var expr: std.Io.Writer.Allocating = .init(arena);
    var used: Used = .{};
    try genExpr(&expr.writer, r, depth, &used);

    var w: std.Io.Writer.Allocating = .init(arena);
    try w.writer.print(
        \\// Generated by tools/crosscheck/gen.zig — do not edit.
        \\// seed={d} depth={d}
        \\//
        \\// body(a, b) is a random total i32 expression. Compiled for the GPU
        \\// (src/engine/crosscheck_harness.zig) and the CPU (tools/crosscheck/driver.zig);
        \\// the driver feeds both the same inputs and compares.
        \\pub const seed: u64 = {d};
        \\
        \\pub fn body(a: i32, b: i32) i32 {{
        \\    @setRuntimeSafety(false);
        \\
    , .{ s, depth, s });
    // A parameter the expression never referenced must be discarded, or Zig
    // rejects it as unused.
    if (!used.a) try w.writer.writeAll("    _ = a;\n");
    if (!used.b) try w.writer.writeAll("    _ = b;\n");
    try w.writer.print("    return {s};\n}}\n", .{expr.written()});

    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = w.written() });
}
