// Metal Shading Language twin of hdiv_constant.zig: half-precision division by
// the constant 33.0, the result widened to float and returned as bits.
#include <metal_stdlib>
using namespace metal;

// The same kernel shape as src/engine/crosscheck_harness.zig, so the bench's
// driver can run this library in place of the one built from the Zig body
// (`zig build crosscheck-msl -Dcrosscheck-body=<twin>.zig -Dcrosscheck-metal=<this file>`).
struct Params { uint count; };

static float ftz(float x) { return (fabs(x) < 1.17549435e-38f) ? copysign(0.0f, x) : x; }
static float fin(int x) { float v = as_type<float>(x); return (v != v) ? as_type<float>(0x7fc00000) : ftz(v); }
static half hin(float x) { half h = (half)x; return (h != h) ? as_type<half>((ushort)0x7e00) : h; }
static float hout(half x) { return (x != x) ? as_type<float>(0x7fc00000) : (float)x; }
static int bits(float x) { return (x != x) ? 0x7fc00000 : as_type<int>(x); }

static int body(int a, int b) {
    return bits(hout(hin(fin(a)) / 33.0h));
}

kernel void crosscheckKernel(device const int* in0 [[buffer(0)]], device const int* in1 [[buffer(1)]],
                             device int* out [[buffer(2)]], constant Params& params [[buffer(3)]],
                             uint gid [[thread_position_in_grid]]) {
    if (gid >= params.count) return;
    out[gid] = body(in0[gid], in1[gid]);
}
