extern var in0: [4096]i32 addrspace(.storage_buffer);
extern var in1: [4096]i32 addrspace(.storage_buffer);
extern var out: [4096]i32 addrspace(.storage_buffer);
extern const params: extern struct { count: u32, scale: f32 } addrspace(.uniform);
extern var tile: [64]i32 addrspace(.shared);
inline fn f(x: i32) f32 {
    return @bitCast(x);
}
inline fn fi(x: f32) i32 {
    return @bitCast(x);
}
fn body(a: i32, b: i32) i32 {
    var s: i32 = a;
    var i: u32 = 0;
    while (i < (@as(u32, @bitCast(b)) & 7)) : (i += 1) {
        s = switch (s & 3) {
            0 => s *% 3,
            1 => s +% @as(i32, @intCast(i)),
            2 => fi(@sqrt(@abs(f(s))) * params.scale),
            else => fi(@floor(f(s)) + @mulAdd(f32, f(a), f(b), 1.5)),
        };
        if (s == 7) break;
    }
    const t = if (f(s) < f(a)) @min(s, b) else @max(s, b);
    return t;
}
export fn kernel() callconv(.{ .spirv_kernel = .{ .x = 64, .y = 1, .z = 1 } }) void {
    const lid = @workItemId(0);
    const gid = @workGroupId(0) * 64 + lid;
    if (gid >= params.count) return;
    tile[lid] = in0[gid];
    out[gid] = body(tile[63 - lid], in1[gid]);
}
