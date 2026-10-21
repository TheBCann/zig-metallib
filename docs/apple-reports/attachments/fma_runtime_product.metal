// Control for fma_constant_product_flush.metal: the same fma, but the first
// multiplicand arrives in the second input buffer at run time instead of being
// a constant, so the compiler cannot fold the product. Pass FLT_MIN
// (0x00800000) as b to compute the same values as the constant version.
#include <metal_stdlib>
using namespace metal;

struct Params { uint count; };

static float ftz(float x) { return (fabs(x) < 1.17549435e-38f) ? copysign(0.0f, x) : x; }
static float fin(int x) { float v = as_type<float>(x); return (v != v) ? as_type<float>(0x7fc00000) : ftz(v); }
static int bits(float x) { return (x != x) ? 0x7fc00000 : as_type<int>(x); }

static int body(int a, int b) {
    return bits(ftz(fma(fin(b), as_type<float>(0x3F000000), fin(a))));
}

kernel void crosscheckKernel(device const int* in0 [[buffer(0)]], device const int* in1 [[buffer(1)]],
                             device int* out [[buffer(2)]], constant Params& params [[buffer(3)]],
                             uint gid [[thread_position_in_grid]]) {
    if (gid >= params.count) return;
    out[gid] = body(in0[gid], in1[gid]);
}
