//! Stage-4 kernels written in Zig and compiled by Zig's self-hosted SPIR-V
//! backend (`zig build spirv-corpus` builds this file for `spirv64-vulkan`
//! with `use_llvm = false`, as `-Dcrosscheck-spirv=true` builds the
//! crosscheck harness), then run by tools/spirv/corpus_driver.zig against
//! known answers. They are the proof that SPIR-V from a real producer with
//! barriers, threadgroup memory and atomics runs correctly on Metal through
//! spirv-splice:
//!
//!   tg_reverse      out[gid] = in0[group * 64 + 63 - lid] through a
//!                   Workgroup array and `std.spirv.workgroupBarrier()`
//!                   (OpControlBarrier Workgroup Workgroup
//!                   AcquireRelease|WorkgroupMemory -> air.wg.barrier(2, 1));
//!   atomic_counter  out[gid] = atomicAdd(&counter.count, 1): the old values
//!                   are a permutation of 0..n-1 and the counter ends at n
//!                   (OpAtomicIAdd Device Relaxed -> atomicrmw add monotonic);
//!   control_flow    tools/testdata/cf.zig's kernel with the barrier it
//!                   lacked (cf.zig reads `tile[63 - lid]`, which another
//!                   thread wrote, without one: a data race, so the corpus
//!                   copy zig-control-flow.spvasm is only built, not run),
//!                   computing zig_kernels_body.zig's `body`, whose CPU twin
//!                   the driver compares every element with.
//!
//! The Zig SPIR-V backend has no atomics (`@atomicRmw` is not implemented
//! for SPIR-V at 0.18.0-dev.253), so `atomicAdd` is SPIR-V inline assembly,
//! the way std.spirv writes its barriers.
//!
//! Bindings (the driver binds them by the manifest spirv-splice prints):
//!   0 in0      StorageBuffer [4096]i32, NonWritable
//!   1 in1      StorageBuffer [4096]i32, NonWritable
//!   2 out      StorageBuffer [4096]i32
//!   3 params   Uniform { count: u32, scale: f32 }
//!   4 counter  StorageBuffer { count: u32 }
//! LocalSize 64 1 1; dispatched over 4096 threads.

const std = @import("std");
const spirv = std.spirv;
const kernels_body = @import("zig_kernels_body.zig");

pub const panic = std.debug.no_panic;

const n = 4096;
const Buf = extern struct { data: [n]i32 };
const Params = extern struct { count: u32, scale: f32 };
const Counter = extern struct { count: u32 };

const in0 = @extern(*addrspace(.storage_buffer) const Buf, .{ .name = "in0", .decoration = .{ .descriptor = .{ .set = 0, .binding = 0 } } });
const in1 = @extern(*addrspace(.storage_buffer) const Buf, .{ .name = "in1", .decoration = .{ .descriptor = .{ .set = 0, .binding = 1 } } });
const out = @extern(*addrspace(.storage_buffer) Buf, .{ .name = "out", .decoration = .{ .descriptor = .{ .set = 0, .binding = 2 } } });
const params = @extern(*addrspace(.uniform) const Params, .{ .name = "params", .decoration = .{ .descriptor = .{ .set = 0, .binding = 3 } } });
const counter = @extern(*addrspace(.storage_buffer) Counter, .{ .name = "counter", .decoration = .{ .descriptor = .{ .set = 0, .binding = 4 } } });

extern var tile: [64]i32 addrspace(.shared);

/// `atomicAdd(p, v)`: OpAtomicIAdd with Device scope and Relaxed semantics
/// (what glslang emits for GLSL's atomicAdd on a buffer); returns the old
/// value.
fn atomicAdd(p: *addrspace(.storage_buffer) u32, v: u32) u32 {
    return asm volatile (
        \\%ret = OpAtomicIAdd %u32 %p %scope %semantics %v
        : [ret] "" (-> u32),
        : [u32] "t" (u32),
          [p] "" (p),
          [scope] "" (@as(u32, @backingInt(spirv.Scope.device))),
          [semantics] "" (@as(u32, @bitCast(spirv.MemorySemantics.none))),
          [v] "" (v),
    );
}

export fn tg_reverse() callconv(.{ .spirv_kernel = .{ .x = 64, .y = 1, .z = 1 } }) void {
    const lid = spirv.local_invocation_id[0];
    const gid = spirv.global_invocation_id[0];
    tile[lid] = in0.data[gid];
    spirv.workgroupBarrier();
    out.data[gid] = tile[63 - lid];
}

export fn atomic_counter() callconv(.{ .spirv_kernel = .{ .x = 64, .y = 1, .z = 1 } }) void {
    const gid = spirv.global_invocation_id[0];
    out.data[gid] = @bitCast(atomicAdd(&counter.count, 1));
}

export fn control_flow() callconv(.{ .spirv_kernel = .{ .x = 64, .y = 1, .z = 1 } }) void {
    const lid = spirv.local_invocation_id[0];
    const gid = spirv.workgroup_id[0] * 64 + lid;
    tile[lid] = in0.data[gid];
    spirv.workgroupBarrier();
    if (gid >= params.count) return;
    out.data[gid] = kernels_body.body(tile[63 - lid], in1.data[gid], params.scale);
}
