//! The computation of the Zig control-flow kernel (tools/spirv/zig_kernels.zig
//! `control_flow`), shared with its CPU twin in tools/spirv/corpus_driver.zig:
//! the same source is compiled once by Zig's self-hosted SPIR-V backend (into
//! the kernel) and once for the host (the reference), as the crosscheck
//! bench does with its generated body.
//!
//! It keeps the shape of tools/spirv/testdata/cf.zig (a loop with a runtime
//! trip count, a four-way switch, an early `break`, labeled-break phis,
//! float arithmetic through GLSL.std.450 Floor / Fma and integer min / max),
//! but every float operation is exact or correctly rounded on both sides, so
//! the GPU must agree with the CPU bit for bit: integers are converted to
//! float only below 2^24 after a shift (no rounding in the conversion, and no
//! subnormal reinterpretation, which Metal flushes: cf.zig's `@bitCast` of
//! small integers to f32 makes subnormals), the products are IEEE-rounded
//! multiplies, `@floor` and `@mulAdd` are exact, and no `@sqrt` (Metal's is
//! not correctly rounded).

/// One output element from two inputs and the uniform `scale`.
pub fn body(a: i32, b: i32, scale: f32) i32 {
    var s: i32 = a;
    var i: u32 = 0;
    while (i < (@as(u32, @bitCast(b)) & 7)) : (i += 1) {
        s = switch (s & 3) {
            0 => s *% 3,
            1 => s +% @as(i32, @intCast(i)),
            2 => @intFromFloat(@floor(@as(f32, @floatFromInt(s >> 8)) * scale)),
            else => @intFromFloat(@mulAdd(f32, @floatFromInt(s >> 12), 0.5, @floatFromInt(b >> 12))),
        };
        if (s == 7) break;
    }
    const fa: f32 = @floatFromInt(a >> 8);
    const fs: f32 = @floatFromInt(s >> 8);
    return if (fs < fa) @min(s, b) else @max(s, b);
}
