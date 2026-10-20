//! Known finding: `@round` loses the sign of a negative zero result.
//!
//! IEEE 754 and C define round(-1.3e-17) as -0.0; Zig lowers `@round` to
//! `llvm.round.f32` and the GPU returns +0.0 for every input whose rounded
//! value is a negative zero (949 of the driver's 4096 inputs). `@floor`,
//! `@ceil` and `@trunc` keep the sign. The CPU side is exact. Persists through
//! Apple's assembler (`zig build crosscheck-apple`), so it is Metal's
//! (Apple M3, macOS 26.3, Zig 0.18.0-dev.253). See docs/crosscheck.md.
//!
//!   zig build crosscheck -Dcrosscheck-body=tools/crosscheck/known/round_negative_zero.zig

pub const seed: u64 = 0;

inline fn f(x: i32) f32 {
    return @bitCast(x);
}
inline fn i(x: f32) i32 {
    return @bitCast(x);
}
/// The GPU flushes single-precision denormals to zero; do the same on both
/// sides so that only the rounding is compared.
inline fn ftz(x: f32) f32 {
    return if (@abs(x) < 1.17549435e-38) (if (x < 0 or (x == 0 and i(x) < 0)) -0.0 else 0.0) else x;
}
inline fn canon(x: f32) i32 {
    return if (x != x) 0x7fc00000 else i(x);
}

pub fn body(a: i32, b: i32) i32 {
    _ = b;
    return canon(@round(ftz(f(a))));
}
