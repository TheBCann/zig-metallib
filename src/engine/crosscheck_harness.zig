//! Fixed GPU wrapper for the differential test bench (milestone 1).
//!
//! The kernel shape never changes: two `device const i32*` inputs, one
//! `device i32*` output, a `constant` count, and the thread id. Only
//! `crosscheck_body.body` — a generated total i32 expression — varies per seed
//! (tools/crosscheck/gen.zig). This file is compiled for nvptx64 like my_shader.zig;
//! air-splice reads `functions` here as its manifest.
//!
//! The same generated `crosscheck_body` is compiled for the host and run on the CPU
//! by tools/crosscheck/driver.zig, which compares the two. Keeping the GPU and CPU
//! sides a single Zig function is the whole point: a mismatch is a codegen
//! bug, not two implementations drifting.

const std = @import("std");
const builtin = @import("builtin");
pub const air = @import("air.zig");
pub const gpu = @import("gpu.zig");
const crosscheck_body = @import("crosscheck_body");

pub const panic = std.debug.no_panic;

/// `[[buffer(3)]]`, constant: how many threads do real work. The grid is
/// rounded up to a multiple of the threadgroup size, so the tail must bail.
pub const Params = struct {
    count: u32,
};

/// out[gid] = body(in0[gid], in1[gid]) for gid < count.
pub fn crosscheckKernel(
    in0: [*]addrspace(.global) const i32,
    in1: [*]addrspace(.global) const i32,
    out: [*]addrspace(.global) i32,
    params: *addrspace(.param) const Params,
    gid: u32,
) callconv(.kernel) void {
    if (gid >= params.count) return;
    out[gid] = crosscheck_body.body(in0[gid], in1[gid]);
}

pub const functions = [_]air.Function{
    .{
        .name = "crosscheckKernel",
        .stage = .kernel,
        .ret = void,
        .args = &.{
            .{ .buffer = .{ .index = 0, .T = i32, .name = "in0" } },
            .{ .buffer = .{ .index = 1, .T = i32, .name = "in1" } },
            .{ .buffer = .{ .index = 2, .T = i32, .name = "out", .access = .read_write } },
            .{ .buffer = .{ .index = 3, .T = Params, .name = "params", .space = .constant } },
            .{ .builtin = .{ .kind = .thread_position_in_grid, .T = u32, .name = "gid" } },
        },
    },
};

// Keep-alive: nothing here calls the kernel, so without a reference the GPU
// build would drop it. GPU-only, like my_shader.zig's exports.
comptime {
    if (builtin.target.cpu.arch == .nvptx64) {
        @export(&crosscheckKernel, .{ .name = "crosscheckKernel" });
    }
}
