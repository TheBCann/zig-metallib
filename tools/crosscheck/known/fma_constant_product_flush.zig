//! Known finding: a constant fma product is folded and flushed.
//!
//! `@mulAdd(f32, C1, C2, x)` with both multiplicands comptime-known and an
//! exact product that is a denormal (FLT_MIN * 0.5 = 2^-127): Apple's compiler
//! folds the product and flushes it to zero, so the fma degenerates to `x`,
//! while the hardware fma (one runtime multiplicand) keeps the exact
//! intermediate, as does the CPU. 223 of the driver's 4096 inputs differ (266
//! with -Dcrosscheck-floats=true, whose inputs include more small values).
//! Persists through Apple's assembler (`zig build crosscheck-apple`), so it is
//! Metal's (Apple M3, macOS 26.3, Zig 0.18.0-dev.253). See docs/crosscheck.md.
//!
//!   zig build crosscheck -Dcrosscheck-body=tools/crosscheck/known/fma_constant_product_flush.zig

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
inline fn bits(x: f32) i32 {
    return if (x != x) 0x7fc00000 else @bitCast(x);
}

pub fn body(a: i32, b: i32) i32 {
    _ = b;
    return bits(ftz(@mulAdd(f32, fc(0x00800000), fc(0x3F000000), fin(a))));
}
