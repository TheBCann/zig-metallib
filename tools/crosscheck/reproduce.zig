//! The command that reproduces one crosscheck run, as build.zig hands it to
//! tools/crosscheck/driver.zig (`--reproduce=`), which prints it on a
//! failure. Kept out of build.zig so `zig build test` can check it: a hint
//! that drops a flag sends the reader to a different kernel (without
//! `-Dcrosscheck-spirv=true` the same seed is compiled by the nvptx front end
//! and, for a generated body, printed from a different menu).
//!
//! The seed is always part of the command, because the driver's inputs come
//! from `-Dseed` even when the body is hand-written.

const std = @import("std");

/// The build options that select what a crosscheck run computes. Each one
/// that is set appears in the command, in this order.
pub const Options = struct {
    /// `-Dcrosscheck-known=true`: the generator emits the recorded findings.
    known: bool = false,
    /// `-Dcrosscheck-floats=true`: float subtrees and float edge inputs.
    floats: bool = false,
    /// `-Dcrosscheck-spirv=true`: the SPIR-V front end (docs/spirv.md).
    spirv: bool = false,
    /// `-Dspirv-zig-workarounds=false` (only with `spirv`): spirv-splice
    /// leaves Zig's SPIR-V backend bugs in place.
    spirv_zig_workarounds: bool = true,
    /// `-Dmetal-fast-math=true`: the library declares fast math.
    fast_math: bool = false,
    /// `-Dcrosscheck-body=<file>`: a hand-written or reduced body.
    body: ?[]const u8 = null,
    /// `-Dcrosscheck-metal=<file>`: the MSL twin (`crosscheck-msl`).
    metal: ?[]const u8 = null,
};

/// The `zig build <step> -Dseed=<seed> ...` command line for a run with
/// these options, allocated with `gpa`.
pub fn command(gpa: std.mem.Allocator, step: []const u8, seed: u64, o: Options) ![]u8 {
    var w: std.Io.Writer.Allocating = .init(gpa);
    errdefer w.deinit();
    try w.writer.print("zig build {s} -Dseed={d}", .{ step, seed });
    if (o.known) try w.writer.writeAll(" -Dcrosscheck-known=true");
    if (o.floats) try w.writer.writeAll(" -Dcrosscheck-floats=true");
    if (o.spirv) try w.writer.writeAll(" -Dcrosscheck-spirv=true");
    if (o.spirv and !o.spirv_zig_workarounds) try w.writer.writeAll(" -Dspirv-zig-workarounds=false");
    if (o.fast_math) try w.writer.writeAll(" -Dmetal-fast-math=true");
    if (o.body) |path| try w.writer.print(" -Dcrosscheck-body={s}", .{path});
    if (o.metal) |path| try w.writer.print(" -Dcrosscheck-metal={s}", .{path});
    return w.toOwnedSlice();
}

test "the reproduce command names the step, the seed and every option set" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try std.testing.expectEqualStrings("zig build crosscheck -Dseed=7", try command(a, "crosscheck", 7, .{}));
    try std.testing.expectEqualStrings(
        "zig build crosscheck -Dseed=7 -Dcrosscheck-spirv=true",
        try command(a, "crosscheck", 7, .{ .spirv = true }),
    );
    try std.testing.expectEqualStrings(
        "zig build crosscheck-apple -Dseed=3 -Dcrosscheck-known=true -Dcrosscheck-floats=true -Dcrosscheck-spirv=true -Dmetal-fast-math=true -Dcrosscheck-body=tools/crosscheck/known/hdiv_constant.zig",
        try command(a, "crosscheck-apple", 3, .{ .known = true, .floats = true, .spirv = true, .fast_math = true, .body = "tools/crosscheck/known/hdiv_constant.zig" }),
    );
    try std.testing.expectEqualStrings(
        "zig build crosscheck -Dseed=2 -Dcrosscheck-spirv=true -Dspirv-zig-workarounds=false",
        try command(a, "crosscheck", 2, .{ .spirv = true, .spirv_zig_workarounds = false }),
    );
    // The workaround switch only means something on the SPIR-V path.
    try std.testing.expectEqualStrings("zig build crosscheck -Dseed=2", try command(a, "crosscheck", 2, .{ .spirv_zig_workarounds = false }));
    try std.testing.expectEqualStrings(
        "zig build crosscheck-msl -Dseed=1 -Dcrosscheck-body=x.zig -Dcrosscheck-metal=x.metal",
        try command(a, "crosscheck-msl", 1, .{ .body = "x.zig", .metal = "x.metal" }),
    );
}
