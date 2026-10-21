// Metal Shading Language twin of mul_sat.zig: a saturating multiply. MSL has
// no spelling that lowers to llvm.smul.fix.sat (the intrinsic that crashes the
// compiler), so this uses the library's madsat with a zero addend; it says what
// the saturating multiply should compute, and whether Apple's own path works.
#include <metal_stdlib>
using namespace metal;

// The same kernel shape as src/engine/crosscheck_harness.zig, so the bench's
// driver can run this library in place of the one built from the Zig body
// (`zig build crosscheck-msl -Dcrosscheck-body=<twin>.zig -Dcrosscheck-metal=<this file>`).
struct Params { uint count; };

static int body(int a, int b) {
    return madsat(a, b, 0);
}

kernel void crosscheckKernel(device const int* in0 [[buffer(0)]], device const int* in1 [[buffer(1)]],
                             device int* out [[buffer(2)]], constant Params& params [[buffer(3)]],
                             uint gid [[thread_position_in_grid]]) {
    if (gid >= params.count) return;
    out[gid] = body(in0[gid], in1[gid]);
}
