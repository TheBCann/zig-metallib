//! Known finding, in Zig's self-hosted SPIR-V backend (not Metal): widening
//! an unsigned integer to a signed one sign-extends it.
//!
//! `@as(i32, x)` for `x: u16` (or u8) must zero-extend. Zig's SPIR-V backend
//! picks the conversion opcode by the RESULT's signedness (`buildConvert`,
//! src/codegen/spirv/CodeGen.zig:3474 at f2f0d97a9b), so it emits
//! `OpSConvert %i32 %u16`, which the SPIR-V specification defines as a sign
//! extension: 0xffff comes back as -1 instead of 65535. The nvptx build of
//! the same source is right. Found 2026-10-21 by the first SPIR-V crosscheck
//! sweep (seed 2: the t16u, t8u and mul16 helpers). spirv-splice undoes it
//! for modules whose Generator is Zig, on an OpSConvert that widens an
//! unsigned-typed operand to a signed-typed result (tools/spirv/lower.zig,
//! "producer workarounds"), so the first command matches; the second lowers
//! by the specification alone and mismatches on the inputs with bit 15 set.
//! Apple M3, macOS 26.3, Zig 0.18.0-dev.253. See docs/spirv.md.
//!
//!   zig build crosscheck -Dcrosscheck-spirv=true -Dcrosscheck-body=tools/crosscheck/known/spirv_zig_unsigned_widen.zig
//!   zig build crosscheck -Dcrosscheck-spirv=true -Dspirv-zig-workarounds=false -Dcrosscheck-body=tools/crosscheck/known/spirv_zig_unsigned_widen.zig

pub const seed: u64 = 0;

pub fn body(a: i32, b: i32) i32 {
    _ = b;
    return @as(u16, @truncate(@as(u32, @bitCast(a))));
}
