// Metal Shading Language twin of fma_constant_product_flush.zig: fma with two
// constant multiplicands whose exact product, FLT_MIN * 0.5 = 2^-127, is a
// denormal. The library's fma lowers to air.fma.f32, where Zig's @mulAdd is
// llvm.fma.f32.
#include <metal_stdlib>
using namespace metal;

// The same kernel shape as src/engine/crosscheck_harness.zig, so the bench's
// driver can run this library in place of the one built from the Zig body
// (`zig build crosscheck-msl -Dcrosscheck-body=<twin>.zig -Dcrosscheck-metal=<this file>`).
struct Params { uint count; };

static float ftz(float x) { return (fabs(x) < 1.17549435e-38f) ? copysign(0.0f, x) : x; }
static float fin(int x) { float v = as_type<float>(x); return (v != v) ? as_type<float>(0x7fc00000) : ftz(v); }
static int bits(float x) { return (x != x) ? 0x7fc00000 : as_type<int>(x); }

static int body(int a, int b) {
    return bits(ftz(fma(as_type<float>(0x00800000), as_type<float>(0x3F000000), fin(a))));
}

kernel void crosscheckKernel(device const int* in0 [[buffer(0)]], device const int* in1 [[buffer(1)]],
                             device int* out [[buffer(2)]], constant Params& params [[buffer(3)]],
                             uint gid [[thread_position_in_grid]]) {
    if (gid >= params.count) return;
    out[gid] = body(in0[gid], in1[gid]);
}
