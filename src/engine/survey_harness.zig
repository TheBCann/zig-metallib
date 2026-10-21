//! GPU kernels for the precision survey (docs/crosscheck-float-policy.md,
//! "Precision survey"): one kernel per transcendental function and variant,
//! `out[gid] = f(in[gid])` on f32. Three variants per function: what Zig's
//! builtin compiles to today (`llvm.exp.f32` passed through to Metal, or the
//! renamed `air.fast_sin.f32`; tools/air/intrinsics.zig), Apple's precise
//! `air.<f>.f32`, and Apple's `air.fast_<f>.f32`, the last two declared
//! directly by name, as src/engine/gpu.zig declares the SIMD intrinsics. The
//! survey then tells which one a builtin really is and what each costs in
//! ulps. Compiled for nvptx64 like every shader; the host survey
//! (tools/crosscheck/survey.zig) imports this file for the manifest only.

const std = @import("std");
const builtin = @import("builtin");
pub const air = @import("air.zig");
pub const gpu = @import("gpu.zig");

pub const panic = std.debug.no_panic;

/// `[[buffer(2)]]`, constant: how many threads do real work.
pub const Params = struct {
    count: u32,
};

// What `xcrun metal` emits for the MSL functions under -fno-fast-math and
// -ffast-math respectively.
extern fn @"air.exp.f32"(x: f32) f32;
extern fn @"air.fast_exp.f32"(x: f32) f32;
extern fn @"air.exp2.f32"(x: f32) f32;
extern fn @"air.fast_exp2.f32"(x: f32) f32;
extern fn @"air.log.f32"(x: f32) f32;
extern fn @"air.fast_log.f32"(x: f32) f32;
extern fn @"air.log2.f32"(x: f32) f32;
extern fn @"air.fast_log2.f32"(x: f32) f32;
extern fn @"air.log10.f32"(x: f32) f32;
extern fn @"air.fast_log10.f32"(x: f32) f32;
extern fn @"air.sin.f32"(x: f32) f32;
extern fn @"air.fast_sin.f32"(x: f32) f32;
extern fn @"air.cos.f32"(x: f32) f32;
extern fn @"air.fast_cos.f32"(x: f32) f32;
extern fn @"air.tan.f32"(x: f32) f32;
extern fn @"air.fast_tan.f32"(x: f32) f32;
extern fn @"air.sqrt.f32"(x: f32) f32;
extern fn @"air.fast_sqrt.f32"(x: f32) f32;

fn zigExp(x: f32) f32 {
    return @exp(x);
}
fn zigExp2(x: f32) f32 {
    return @exp2(x);
}
fn zigLog(x: f32) f32 {
    return @log(x);
}
fn zigLog2(x: f32) f32 {
    return @log2(x);
}
fn zigLog10(x: f32) f32 {
    return @log10(x);
}
fn zigSin(x: f32) f32 {
    return @sin(x);
}
fn zigCos(x: f32) f32 {
    return @cos(x);
}
fn zigTan(x: f32) f32 {
    return @tan(x);
}
fn zigSqrt(x: f32) f32 {
    return @sqrt(x);
}
fn airExp(x: f32) f32 {
    return @"air.exp.f32"(x);
}
fn fastExp(x: f32) f32 {
    return @"air.fast_exp.f32"(x);
}
fn airExp2(x: f32) f32 {
    return @"air.exp2.f32"(x);
}
fn fastExp2(x: f32) f32 {
    return @"air.fast_exp2.f32"(x);
}
fn airLog(x: f32) f32 {
    return @"air.log.f32"(x);
}
fn fastLog(x: f32) f32 {
    return @"air.fast_log.f32"(x);
}
fn airLog2(x: f32) f32 {
    return @"air.log2.f32"(x);
}
fn fastLog2(x: f32) f32 {
    return @"air.fast_log2.f32"(x);
}
fn airLog10(x: f32) f32 {
    return @"air.log10.f32"(x);
}
fn fastLog10(x: f32) f32 {
    return @"air.fast_log10.f32"(x);
}
fn airSin(x: f32) f32 {
    return @"air.sin.f32"(x);
}
fn fastSin(x: f32) f32 {
    return @"air.fast_sin.f32"(x);
}
fn airCos(x: f32) f32 {
    return @"air.cos.f32"(x);
}
fn fastCos(x: f32) f32 {
    return @"air.fast_cos.f32"(x);
}
fn airTan(x: f32) f32 {
    return @"air.tan.f32"(x);
}
fn fastTan(x: f32) f32 {
    return @"air.fast_tan.f32"(x);
}
fn airSqrt(x: f32) f32 {
    return @"air.sqrt.f32"(x);
}
fn fastSqrt(x: f32) f32 {
    return @"air.fast_sqrt.f32"(x);
}

pub const Variant = struct { name: [:0]const u8, f: fn (f32) f32 };

/// The kernels, by entry-point name. The host survey names them in its table
/// and checks each one exists here.
pub const kernels = [_]Variant{
    .{ .name = "expKernel", .f = zigExp },     .{ .name = "airExpKernel", .f = airExp },     .{ .name = "fastExpKernel", .f = fastExp },
    .{ .name = "exp2Kernel", .f = zigExp2 },   .{ .name = "airExp2Kernel", .f = airExp2 },   .{ .name = "fastExp2Kernel", .f = fastExp2 },
    .{ .name = "logKernel", .f = zigLog },     .{ .name = "airLogKernel", .f = airLog },     .{ .name = "fastLogKernel", .f = fastLog },
    .{ .name = "log2Kernel", .f = zigLog2 },   .{ .name = "airLog2Kernel", .f = airLog2 },   .{ .name = "fastLog2Kernel", .f = fastLog2 },
    .{ .name = "log10Kernel", .f = zigLog10 }, .{ .name = "airLog10Kernel", .f = airLog10 }, .{ .name = "fastLog10Kernel", .f = fastLog10 },
    .{ .name = "sinKernel", .f = zigSin },     .{ .name = "airSinKernel", .f = airSin },     .{ .name = "fastSinKernel", .f = fastSin },
    .{ .name = "cosKernel", .f = zigCos },     .{ .name = "airCosKernel", .f = airCos },     .{ .name = "fastCosKernel", .f = fastCos },
    .{ .name = "tanKernel", .f = zigTan },     .{ .name = "airTanKernel", .f = airTan },     .{ .name = "fastTanKernel", .f = fastTan },
    .{ .name = "sqrtKernel", .f = zigSqrt },   .{ .name = "airSqrtKernel", .f = airSqrt },   .{ .name = "fastSqrtKernel", .f = fastSqrt },
};

fn Kernel(comptime f: fn (f32) f32) type {
    return struct {
        fn kernel(
            in: [*]addrspace(.global) const f32,
            out: [*]addrspace(.global) f32,
            params: *addrspace(.param) const Params,
            gid: u32,
        ) callconv(.kernel) void {
            if (gid >= params.count) return;
            out[gid] = f(in[gid]);
        }
    };
}

pub const functions = blk: {
    var fs: [kernels.len]air.Function = undefined;
    for (kernels, 0..) |k, i| {
        fs[i] = .{
            .name = k.name,
            .stage = .kernel,
            .ret = void,
            .args = &.{
                .{ .buffer = .{ .index = 0, .T = f32, .name = "in" } },
                .{ .buffer = .{ .index = 1, .T = f32, .name = "out", .access = .read_write } },
                .{ .buffer = .{ .index = 2, .T = Params, .name = "params", .space = .constant } },
                .{ .builtin = .{ .kind = .thread_position_in_grid, .T = u32, .name = "gid" } },
            },
        };
    }
    break :blk fs;
};

// Keep-alive, GPU-only, as in crosscheck_harness.zig.
comptime {
    if (builtin.target.cpu.arch == .nvptx64) {
        for (kernels) |k| @export(&Kernel(k.f).kernel, .{ .name = k.name });
    }
}
