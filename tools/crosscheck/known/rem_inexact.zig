//! Known finding: `@rem` on f32 is not the exact remainder.
//!
//! Worked around in this project since 2026-10-20: the assembler lowers `frem`
//! to a call to Apple's exact `air.fmod.f32` (tools/air/intrinsics.zig), so
//! `zig build crosscheck` on this file matches; `zig build crosscheck-apple`
//! (Apple's assembler, no workaround) still fails as described.
//!
//! fmod is exact by definition (IEEE 754 remainder, C fmod): for finite x and
//! y the result is representable and uniquely determined. Zig lowers `@rem`
//! to LLVM `frem`; the GPU returns results such as 0 where the answer is
//! 1.02e-10, 8 where it is 6.91, and -inf for x = 9.1e18, y = 1.7e-31, the
//! signature of computing x - y * trunc(x / y) with x / y overflowing. 1067
//! of the driver's 4096 inputs differ; the CPU side is exact. Persists through
//! Apple's assembler, so it is Metal's (Apple M3, macOS 26.3, Zig
//! 0.18.0-dev.253). See docs/crosscheck.md.
//!
//!   zig build crosscheck -Dcrosscheck-body=tools/crosscheck/known/rem_inexact.zig

pub const seed: u64 = 0;

inline fn f(x: i32) f32 {
    return @bitCast(x);
}
inline fn i(x: f32) i32 {
    return @bitCast(x);
}
inline fn ftz(x: f32) f32 {
    return if (@abs(x) < 1.17549435e-38) (if (x < 0 or (x == 0 and i(x) < 0)) -0.0 else 0.0) else x;
}
inline fn canon(x: f32) i32 {
    return if (x != x) 0x7fc00000 else i(x);
}

pub fn body(a: i32, b: i32) i32 {
    return canon(ftz(@rem(ftz(f(a)), ftz(f(b)))));
}
