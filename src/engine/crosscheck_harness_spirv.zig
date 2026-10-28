//! The crosscheck kernel for the SPIR-V path (`zig build crosscheck
//! -Dcrosscheck-spirv=true`, docs/spirv.md).
//!
//! The same computation as src/engine/crosscheck_harness.zig, out[gid] =
//! body(in0[gid], in1[gid]) for gid < count, written in the standard Vulkan
//! compute shape Zig's self-hosted SPIR-V backend emits for
//! `spirv64-vulkan` (`-fno-llvm`): the buffers are module-scope Block
//! variables with explicit descriptors instead of kernel parameters, the
//! thread id is the GlobalInvocationId builtin, and the entry point carries
//! its LocalSize. build.zig compiles this file with the generated
//! `crosscheck_body`, tools/spirv_splice.zig lowers the binary to a Metal
//! library, and the unchanged tools/crosscheck/driver.zig runs it, so the
//! bindings and the threadgroup size are the ones the driver already uses:
//!
//!   in0     StorageBuffer, Binding 0, NonWritable  -> [[buffer(0)]] device, read
//!   in1     StorageBuffer, Binding 1, NonWritable  -> [[buffer(1)]] device, read
//!   out     StorageBuffer, Binding 2               -> [[buffer(2)]] device, read_write
//!   params  Uniform,       Binding 3               -> [[buffer(3)]] constant (setBytes)
//!   gid     GlobalInvocationId.x                   -> thread_position_in_grid
//!   LocalSize 64 1 1                               (dispatchThreads 4096 x 64)
//!
//! The buffers are fixed-size Block structs (`extern struct { data: [4096]i32 }`):
//! a plain `[4096]i32 addrspace(.storage_buffer)` extern compiles too but
//! gets no Binding decoration, which spirv-splice would place at the lowest
//! free index instead (docs/spirv.md, "The manifest"). 4096 is the driver's
//! element count.

const std = @import("std");
const spirv = std.spirv;
const crosscheck_body = @import("crosscheck_body");

pub const panic = std.debug.no_panic;

/// The driver's element count (tools/crosscheck/driver.zig `n`).
const n = 4096;

/// One input or output buffer: a Block struct wrapping the array, as Vulkan
/// requires for a storage buffer.
const Buf = extern struct { data: [n]i32 };

/// `[[buffer(3)]]`, constant: how many threads do real work (the driver's
/// `Params`, byte for byte).
const Params = extern struct { count: u32 };

const in0 = @extern(*addrspace(.storage_buffer) const Buf, .{ .name = "in0", .decoration = .{ .descriptor = .{ .set = 0, .binding = 0 } } });
const in1 = @extern(*addrspace(.storage_buffer) const Buf, .{ .name = "in1", .decoration = .{ .descriptor = .{ .set = 0, .binding = 1 } } });
const out = @extern(*addrspace(.storage_buffer) Buf, .{ .name = "out", .decoration = .{ .descriptor = .{ .set = 0, .binding = 2 } } });
const params = @extern(*addrspace(.uniform) const Params, .{ .name = "params", .decoration = .{ .descriptor = .{ .set = 0, .binding = 3 } } });

/// out[gid] = body(in0[gid], in1[gid]) for gid < count.
export fn crosscheckKernel() callconv(.{ .spirv_kernel = .{ .x = 64, .y = 1, .z = 1 } }) void {
    const gid = spirv.global_invocation_id[0];
    if (gid >= params.count) return;
    out.data[gid] = crosscheck_body.body(in0.data[gid], in1.data[gid]);
}
