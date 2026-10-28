extern var out8: [4096]i8 addrspace(.storage_buffer);
extern var out16: [4096]i16 addrspace(.storage_buffer);
extern var outu8: [4096]u8 addrspace(.storage_buffer);
export fn kernel() callconv(.{ .spirv_kernel = .{ .x = 64, .y = 1, .z = 1 } }) void {
    const gid = @workGroupId(0) * 64 + @workItemId(0);
    out8[gid] = -1;
    out16[gid] = -2;
    outu8[gid] = 200;
}
