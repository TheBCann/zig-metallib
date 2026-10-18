//! Known finding: the overflow bit of a signed add comes back wrong.
//!
//! Zig lowers `@addWithOverflow(a, b)[1]` on i32 to the second result of
//! `llvm.sadd.with.overflow.i32`. The GPU disagrees with the CPU on about half
//! of random inputs (Apple M3, macOS 26.3, Zig 0.18.0-dev.253). See
//! docs/crosscheck.md.
//!
//!   zig build crosscheck -Dcrosscheck-body=tools/crosscheck/known/add_overflow_bit.zig

pub const seed: u64 = 0;

pub fn body(a: i32, b: i32) i32 {
    return @addWithOverflow(a, b)[1];
}
