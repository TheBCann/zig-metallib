//! Zig twin of fma_runtime_product.metal, the CPU reference for
//! `zig build crosscheck-msl -Dcrosscheck-body=<this> -Dcrosscheck-metal=<the .metal>`:
//! the fma of fma_constant_product_flush.zig with the first multiplicand taken
//! from the second input instead of a constant.
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
    return bits(ftz(@mulAdd(f32, fin(b), fc(0x3F000000), fin(a))));
}
