//! Known finding: saturating multiply crashes Metal's compiler.
//!
//! Worked around in this project since 2026-10-20: the assembler emits Apple's
//! `air.mad_sat.s.i32(a, b, 0)` for the intrinsic (tools/air/intrinsics.zig),
//! so `zig build crosscheck` on this file matches; `zig build crosscheck-apple`
//! (Apple's assembler, no workaround) still fails as described.
//!
//! Zig lowers `a *| b` on i32 to `llvm.smul.fix.sat.i32`. Building the
//! compute pipeline fails with MTLCompilerService's connection interrupted
//! (Apple M3, macOS 26.3, Zig 0.18.0-dev.253). See docs/crosscheck.md.
//!
//!   zig build crosscheck -Dcrosscheck-body=tools/crosscheck/known/mul_sat.zig

pub const seed: u64 = 0;

pub fn body(a: i32, b: i32) i32 {
    return a *| b;
}
