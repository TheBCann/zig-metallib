// Metal Shading Language twin of add_overflow_bit.zig: the overflow bit of a
// signed add. clang's builtin lowers to llvm.sadd.with.overflow.i32, exactly
// what Zig emits for @addWithOverflow, so the two bodies are the same AIR.
#include <metal_stdlib>
using namespace metal;

// The same kernel shape as src/engine/crosscheck_harness.zig, so the bench's
// driver can run this library in place of the one built from the Zig body
// (`zig build crosscheck-msl -Dcrosscheck-body=<twin>.zig -Dcrosscheck-metal=<this file>`).
struct Params { uint count; };

static int body(int a, int b) {
    int r;
    return (int)__builtin_sadd_overflow(a, b, &r);
}

kernel void crosscheckKernel(device const int* in0 [[buffer(0)]], device const int* in1 [[buffer(1)]],
                             device int* out [[buffer(2)]], constant Params& params [[buffer(3)]],
                             uint gid [[thread_position_in_grid]]) {
    if (gid >= params.count) return;
    out[gid] = body(in0[gid], in1[gid]);
}
