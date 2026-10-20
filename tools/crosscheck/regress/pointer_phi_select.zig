//! Regression body for the assembler's pointer-phi typing (docs/crosscheck.md,
//! "In this project"). Must run and match:
//!
//!   zig build crosscheck -Dcrosscheck-body=tools/crosscheck/regress/pointer_phi_select.zig
//!
// a loop-carried optional pointer into a constant table, updated
// conditionally with no side effects, so LLVM forms a select in the latch and
// the header phi starts at null (the shape of the new assembler test).
pub const seed: u64 = 9001;

const tab = [8]i32{ 3, 1, 4, 1, 5, 9, 2, 6 };

inline fn u(x: i32) u32 {
    return @bitCast(x);
}

pub fn body(a: i32, b: i32) i32 {
    var best: ?*const i32 = null;
    const n: u32 = u(a) % 8;
    var i: u32 = 0;
    while (i <= n) : (i += 1) {
        if (tab[i] > (b & 7)) best = &tab[i];
    }
    return if (best) |p| p.* else -1;
}
