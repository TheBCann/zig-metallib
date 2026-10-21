// Metal Shading Language twin of fmod_half_negative_zero.zig: fmod on half, the
// result's bit pattern with NaN canonicalised. The library's fmod on half lowers
// to air.fmod.f16 under -fno-fast-math.
#include <metal_stdlib>
using namespace metal;

// The same kernel shape as src/engine/crosscheck_harness.zig, so the bench's
// driver can run this library in place of the one built from the Zig body
// (`zig build crosscheck-msl -Dcrosscheck-body=<twin>.zig -Dcrosscheck-metal=<this file>`).
struct Params { uint count; };

static int body(int a, int b) {
    half x = as_type<half>((ushort)(a & 0xFFFF));
    half y = as_type<half>((ushort)(((uint)b) >> 16));
    half r = fmod(x, y);
    ushort rb = (r != r) ? (ushort)0x7e00 : as_type<ushort>(r);
    return (int)rb;
}

kernel void crosscheckKernel(device const int* in0 [[buffer(0)]], device const int* in1 [[buffer(1)]],
                             device int* out [[buffer(2)]], constant Params& params [[buffer(3)]],
                             uint gid [[thread_position_in_grid]]) {
    if (gid >= params.count) return;
    out[gid] = body(in0[gid], in1[gid]);
}
