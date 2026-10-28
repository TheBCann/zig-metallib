// Crosscheck harness in standard Vulkan compute form with explicit descriptors.
const spirv = @import("std").spirv;
const Buf = extern struct { data: [4096]i32 };
const in0 = @extern(*addrspace(.storage_buffer) const Buf, .{ .name = "in0", .decoration = .{ .descriptor = .{ .set = 0, .binding = 0 } } });
const in1 = @extern(*addrspace(.storage_buffer) const Buf, .{ .name = "in1", .decoration = .{ .descriptor = .{ .set = 0, .binding = 1 } } });
const out = @extern(*addrspace(.storage_buffer) Buf, .{ .name = "out", .decoration = .{ .descriptor = .{ .set = 0, .binding = 2 } } });
const Params = extern struct { count: u32 };
const params = @extern(*addrspace(.uniform) const Params, .{ .name = "params", .decoration = .{ .descriptor = .{ .set = 0, .binding = 3 } } });
fn body(a: i32, b: i32) i32 {
    return a +% b *% 3;
}
export fn crosscheckKernel() callconv(.{ .spirv_kernel = .{ .x = 64, .y = 1, .z = 1 } }) void {
    const gid = spirv.global_invocation_id[0];
    if (gid >= params.count) return;
    out.data[gid] = body(in0.data[gid], in1.data[gid]);
}
