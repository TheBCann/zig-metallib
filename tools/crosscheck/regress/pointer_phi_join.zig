//! Regression body for the assembler's pointer-phi typing (docs/crosscheck.md,
//! "In this project"). Must run and match:
//!
//!   zig build crosscheck -Dcrosscheck-body=tools/crosscheck/regress/pointer_phi_join.zig
//!
// the same optional pointer, but the conditional update sits in a
// block with a non-speculatable call, so LLVM keeps the branch and joins with
// a phi: the header phi and the join phi then depend on each other.
pub const seed: u64 = 9002;

const tab = [8]i32{ 3, 1, 4, 1, 5, 9, 2, 6 };

inline fn u(x: i32) u32 {
    return @bitCast(x);
}

noinline fn tick(x: i32) i32 {
    return x *% 3 +% 1;
}

pub fn body(a: i32, b: i32) i32 {
    var best: ?*const i32 = null;
    var acc: i32 = 0;
    const n: u32 = u(a) % 8;
    var i: u32 = 0;
    while (i <= n) : (i += 1) {
        if (tab[i] > (b & 7)) {
            best = &tab[i];
            acc = tick(acc);
        }
    }
    return if (best) |p| p.* +% acc else acc;
}
