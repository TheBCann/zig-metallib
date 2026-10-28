//! Known finding, in Zig's self-hosted SPIR-V backend (not Metal): widening
//! a signed integer into an i128 loses its sign.
//!
//! `@as(i128, s)` for `s: i16` must sign-extend. Zig's SPIR-V backend stores
//! an i128 as u64 limbs; `airIntCast` converts the source into limb 0 with
//! `buildConvert` (src/codegen/spirv/CodeGen.zig:6078 at f2f0d97a9b), which
//! picks the opcode by the RESULT's signedness (CodeGen.zig:3474) and so
//! emits `OpUConvert %u64 %i16`, a zero extension by the SPIR-V
//! specification, then fills the upper limbs from limb 0's top bit
//! (CodeGen.zig:6111-6131), which is now always 0. This is the mirror of
//! spirv_zig_unsigned_widen.zig (`OpSConvert %i32 %u16`). The nvptx build of
//! the same source is right. Found 2026-10-21 by a review of the SPIR-V
//! crosscheck work. spirv-splice undoes it for modules whose Generator is
//! Zig, on a widening OpUConvert from a signed-typed operand to an
//! unsigned-typed result whose result is sign-filled that way
//! (tools/spirv/lower.zig, "producer workarounds"), so the first command
//! matches; the second lowers by the specification alone and mismatches
//! wherever the i16 is negative. Apple M3, macOS 26.3, Zig 0.18.0-dev.253.
//! See docs/spirv.md.
//!
//!   zig build crosscheck -Dcrosscheck-spirv=true -Dcrosscheck-body=tools/crosscheck/known/spirv_zig_signed_widen.zig
//!   zig build crosscheck -Dcrosscheck-spirv=true -Dspirv-zig-workarounds=false -Dcrosscheck-body=tools/crosscheck/known/spirv_zig_signed_widen.zig

pub const seed: u64 = 0;

pub fn body(a: i32, b: i32) i32 {
    const s: i16 = @truncate(a);
    const w: i128 = s;
    const u: u16 = @truncate(@as(u32, @bitCast(b)));
    const x: i128 = u;
    return @as(i32, @truncate(w >> 64)) ^ @as(i32, @truncate(x >> 3)) ^ @as(i32, @truncate((w +% x) >> 100));
}
