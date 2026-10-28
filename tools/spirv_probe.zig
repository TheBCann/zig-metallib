//! spirv-probe: one Metal library per SPIR-V kernel, for attributing
//! runtime-compiler verdicts to single instructions (docs/spirv.md,
//! "Runtime validation"); tools/spirv/run_probes.sh drives it and builds
//! a compute pipeline from every library with `metallib-check --kernels`.
//!
//!   spirv-probe [--fast-math=<bool>] probes <outdir>
//!       writes <outdir>/<probe>.spvasm and <probe>.metallib for every probe
//!       of tools/spirv/probes.zig (one module per instruction and type);
//!   spirv-probe [--fast-math=<bool>] corpus <file.spv|file.spvasm> <outdir>
//!       lowers each compute entry point of the file on its own
//!       (lower.Options.only_entry) into <outdir>/<file>.<entry>.metallib.
//!
//! Prints one line per kernel: `<name>\t<library>` or `<name>\tREFUSED\t<why>`
//! (a lowering refusal or an assembler error; the assembler's own line
//! message goes to stderr).

const std = @import("std");
const Io = std.Io;
const reader = @import("spirv/reader.zig");
const asm_ = @import("spirv/asm.zig");
const lower = @import("spirv/lower.zig");
const probes = @import("spirv/probes.zig");
const splice = @import("spirv_splice.zig");

const usage = "usage: spirv-probe [--fast-math=<bool>] probes <outdir> | corpus <file> <outdir>\n";

pub fn main(init: std.process.Init) !void {
    const arena = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(arena);
    var rest: []const [:0]const u8 = if (args.len > 0) args[1..] else &.{};
    var fast_math = false;
    if (rest.len > 0 and std.mem.startsWith(u8, rest[0], "--fast-math=")) {
        const v = rest[0]["--fast-math=".len..];
        if (std.mem.eql(u8, v, "true")) fast_math = true else if (!std.mem.eql(u8, v, "false")) {
            std.debug.print(usage, .{});
            std.process.exit(2);
        }
        rest = rest[1..];
    }
    var buf: [4096]u8 = undefined;
    var out: Io.File.Writer = .initStreaming(.stdout(), io, &buf);
    const w = &out.interface;
    const cwd = Io.Dir.cwd();
    if (rest.len == 2 and std.mem.eql(u8, rest[0], "probes")) {
        const dir = rest[1];
        for (try probes.all(arena)) |p| {
            var pa = std.heap.ArenaAllocator.init(init.gpa);
            defer pa.deinit();
            const a = pa.allocator();
            const text = try probes.moduleText(a, p);
            try cwd.writeFile(io, .{ .sub_path = try a.print("{s}/{s}.spvasm", .{ dir, p.name }), .data = text });
            const bytes = try asm_.assemble(a, text, null);
            var m = try reader.read(a, bytes, null);
            try one(a, io, w, &m, p.name, null, try a.print("{s}/{s}.metallib", .{ dir, p.name }), fast_math);
        }
    } else if (rest.len == 3 and std.mem.eql(u8, rest[0], "corpus")) {
        const path = rest[1];
        const dir = rest[2];
        const input = try cwd.readFileAlloc(io, path, arena, .unlimited);
        const bytes: []const u8 = if (splice.isBinary(input)) input else try asm_.assemble(arena, input, null);
        var m = try reader.read(arena, bytes, null);
        const base = std.fs.path.basename(path);
        const stem = base[0 .. std.mem.lastIndexOfScalar(u8, base, '.') orelse base.len];
        for (m.entry_points) |ep| {
            var pa = std.heap.ArenaAllocator.init(init.gpa);
            defer pa.deinit();
            const a = pa.allocator();
            try one(a, io, w, &m, try a.print("{s}:{s}", .{ stem, ep.name }), ep.name, try a.print("{s}/{s}.{s}.metallib", .{ dir, stem, ep.name }), fast_math);
        }
    } else {
        std.debug.print(usage, .{});
        std.process.exit(2);
    }
    try w.flush();
}

fn one(a: std.mem.Allocator, io: Io, w: *Io.Writer, m: *const reader.Module, name: []const u8, entry: ?[]const u8, lib: []const u8, fast_math: bool) !void {
    var diag: lower.Diagnostics = .{};
    const result = splice.splice(a, m, .{ .fast_math = fast_math, .only_entry = entry }, &diag) catch |err| {
        if (err == error.OutOfMemory) return err;
        const why = if (diag.message.len > 0) diag.message else @errorName(err);
        const flat = try std.mem.replaceOwned(u8, a, why, "\n", " ");
        try w.print("{s}\tREFUSED\t{s}\n", .{ name, flat });
        return;
    };
    try Io.Dir.cwd().writeFile(io, .{ .sub_path = lib, .data = result.image });
    try w.print("{s}\t{s}\n", .{ name, lib });
}
