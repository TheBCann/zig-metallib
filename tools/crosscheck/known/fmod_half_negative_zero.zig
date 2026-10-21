//! Known finding: `fmod` in half precision returns `+0.0` where the result is `-0.0`.
//!
//! C's fmod gives the remainder the sign of x, so fmod(-20016, 2) is -0.0. Apple's
//! `air.fmod.f16`, what Metal Shading Language's `fmod(half, half)` compiles to
//! under -fno-fast-math, returns +0.0 for every negative x whose remainder is an
//! exact zero (single precision keeps the sign). 9 of the driver's 4,096 inputs
//! differ through Apple's front end (`zig build crosscheck-msl` with the .metal
//! twin). Found 2026-10-20 by the probe of this project's frem workaround, which
//! now adds a copysign for half, so this file matches through `zig build
//! crosscheck`. Apple M3, macOS 26.3, Zig 0.18.0-dev.253. See docs/crosscheck.md.
//!
//!   zig build crosscheck-msl -Dcrosscheck-body=tools/crosscheck/known/fmod_half_negative_zero.zig -Dcrosscheck-metal=tools/crosscheck/known/fmod_half_negative_zero.metal

pub const seed: u64 = 0;

/// x from the low half of a, y from the high half of b; NaN canonicalised.
pub fn body(a: i32, b: i32) i32 {
    const x: f16 = @bitCast(@as(u16, @truncate(@as(u32, @bitCast(a)))));
    const y: f16 = @bitCast(@as(u16, @truncate(@as(u32, @bitCast(b)) >> 16)));
    const r = @rem(x, y);
    const rb: u16 = if (r != r) 0x7e00 else @bitCast(r);
    return @as(i32, rb);
}
