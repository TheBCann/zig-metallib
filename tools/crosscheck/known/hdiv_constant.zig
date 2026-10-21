//! Known finding: f16 division by a constant is not correctly rounded.
//!
//! `x / 33.0` in half precision differs from the correctly rounded quotient on
//! 10 of the driver's 4,096 inputs (by 3.0: 103, by 10.0: 96; always one f16
//! ulp). Division by a power of two, division by a run-time divisor, f32
//! division by the same constants and f16 multiplication by them are all exact,
//! so the compiler turns the constant division into a multiplication by the
//! reciprocal. Reproduces from Metal Shading Language (`half / 33.0h`,
//! -fno-fast-math) and through Apple's assembler. Found 2026-10-20 by the
//! first float sweep with half-precision nodes (seed 7, reduced). The
//! generator keeps f16 divisors input-dependent. Apple M3, macOS 26.3, Zig
//! 0.18.0-dev.253. See docs/crosscheck.md.
//!
//!   zig build crosscheck -Dcrosscheck-body=tools/crosscheck/known/hdiv_constant.zig
//!   zig build crosscheck-msl -Dcrosscheck-body=tools/crosscheck/known/hdiv_constant.zig -Dcrosscheck-metal=tools/crosscheck/known/hdiv_constant.metal

pub const seed: u64 = 0;

inline fn fc(b: u32) f32 {
    return @bitCast(b);
}
inline fn ftz(x: f32) f32 {
    const b: u32 = @bitCast(x);
    return if ((b & 0x7f800000) == 0) @bitCast(b & 0x80000000) else x;
}
inline fn fin(x: i32) f32 {
    const v: f32 = @bitCast(x);
    return if (v != v) fc(0x7fc00000) else ftz(v);
}
inline fn hin(x: f32) f16 {
    const h: f16 = @floatCast(x);
    return if (h != h) @bitCast(@as(u16, 0x7e00)) else h;
}
inline fn hout(x: f16) f32 {
    return if (x != x) fc(0x7fc00000) else @floatCast(x);
}
inline fn bits(x: f32) i32 {
    return if (x != x) 0x7fc00000 else @bitCast(x);
}

pub fn body(a: i32, b: i32) i32 {
    _ = b;
    return bits(hout(hin(fin(a)) / hin(fc(0x42040000))));
}
