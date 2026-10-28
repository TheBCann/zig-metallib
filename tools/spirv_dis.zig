//! spirv-dis: print a SPIR-V module as text (tools/spirv/dis.zig).
//!
//!   zig build spirv-dis -- [--numeric] [--no-header] <file.spv | file.spvasm>
//!
//! A binary input (magic 0x07230203 in either endianness) is decoded with
//! tools/spirv/reader.zig; anything else is taken as text, assembled with
//! tools/spirv/asm.zig and printed back, which is the quickest check that a
//! corpus file is well-formed. Ids print as `%name` where the module names
//! them uniquely (`--numeric` keeps `%N`), see dis.zig for the format.
//! Exit 1 with `spirv-dis: <file>: word N: <reason>` (or `line N:` for text)
//! when the input is malformed.
//!
//! This file is also the root of the `tools/spirv` test module: `zig build
//! test` runs the tests of reader.zig, dis.zig and asm.zig through it.

const std = @import("std");
const Io = std.Io;
const reader = @import("spirv/reader.zig");
const dis = @import("spirv/dis.zig");
const asm_ = @import("spirv/asm.zig");

pub fn main(init: std.process.Init) !void {
    const arena = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(arena);

    var names = true;
    var header = true;
    var path: ?[]const u8 = null;
    for (args[1..]) |arg| {
        if (std.mem.eql(u8, arg, "--numeric")) {
            names = false;
        } else if (std.mem.eql(u8, arg, "--no-header")) {
            header = false;
        } else if (std.mem.startsWith(u8, arg, "-")) {
            std.debug.print("spirv-dis: unknown option '{s}'\n", .{arg});
            std.process.exit(2);
        } else if (path == null) {
            path = arg;
        } else {
            std.debug.print("usage: spirv-dis [--numeric] [--no-header] <file.spv | file.spvasm>\n", .{});
            std.process.exit(2);
        }
    }
    const file = path orelse {
        std.debug.print("usage: spirv-dis [--numeric] [--no-header] <file.spv | file.spvasm>\n", .{});
        std.process.exit(2);
    };

    const input = try Io.Dir.cwd().readFileAlloc(io, file, arena, .unlimited);
    const bytes: []const u8 = if (isBinary(input)) input else blk: {
        var diag: asm_.Diagnostics = .{};
        break :blk asm_.assemble(arena, input, &diag) catch |err| {
            if (err == error.Syntax) {
                std.debug.print("spirv-dis: {s}: line {d}: {s}\n", .{ file, diag.line, diag.message });
                std.process.exit(1);
            }
            return err;
        };
    };

    var diag: reader.Diagnostics = .{};
    var module = reader.read(arena, bytes, &diag) catch |err| {
        if (err == error.Malformed) {
            std.debug.print("spirv-dis: {s}: word {d}: {s}\n", .{ file, diag.word_offset, diag.message });
            std.process.exit(1);
        }
        return err;
    };
    defer module.deinit();

    var buf: [8192]u8 = undefined;
    var out: Io.File.Writer = .initStreaming(.stdout(), io, &buf);
    try dis.write(&out.interface, arena, &module, .{ .names = names, .header = header });
    try out.interface.flush();
}

fn isBinary(input: []const u8) bool {
    if (input.len < 4) return false;
    const w = std.mem.readInt(u32, input[0..4], .little);
    return w == reader.spec.magic_number or w == @byteSwap(reader.spec.magic_number);
}

test {
    _ = reader;
    _ = dis;
    _ = asm_;
}
