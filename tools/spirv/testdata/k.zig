pub fn body(a: i32, b: i32) i32 {
    return a +% b *% 3;
}
export fn kernel(out: [*]addrspace(.global) i32, in_a: [*]addrspace(.global) const i32, in_b: [*]addrspace(.global) const i32, gid: u32) callconv(.kernel) void {
    out[gid] = body(in_a[gid], in_b[gid]);
}
