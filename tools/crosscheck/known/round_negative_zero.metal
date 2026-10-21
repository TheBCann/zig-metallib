// Metal Shading Language twin of round_negative_zero.zig: round() of a flushed
// input, NaN canonicalised. The library's round lowers to air.round.f32 under
// -fno-fast-math (air.fast_round.f32 under fast math), where Zig's @round is
// llvm.round.f32; the comparison tells whether both lose the sign of -0.0.
#include <metal_stdlib>
using namespace metal;

// The same kernel shape as src/engine/crosscheck_harness.zig, so the bench's
// driver can run this library in place of the one built from the Zig body
// (`zig build crosscheck-msl -Dcrosscheck-body=<twin>.zig -Dcrosscheck-metal=<this file>`).
struct Params { uint count; };

static float ftz(float x) { return (fabs(x) < 1.17549435e-38f) ? copysign(0.0f, x) : x; }
static int canon(float x) { return (x != x) ? 0x7fc00000 : as_type<int>(x); }

static int body(int a, int b) {
    return canon(round(ftz(as_type<float>(a))));
}

kernel void crosscheckKernel(device const int* in0 [[buffer(0)]], device const int* in1 [[buffer(1)]],
                             device int* out [[buffer(2)]], constant Params& params [[buffer(3)]],
                             uint gid [[thread_position_in_grid]]) {
    if (gid >= params.count) return;
    out[gid] = body(in0[gid], in1[gid]);
}
