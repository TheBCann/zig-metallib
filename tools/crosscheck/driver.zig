//! Differential driver for the test bench (milestone 1).
//!
//! Loads the generated crosscheck kernel's .metallib, runs it on the Metal device,
//! runs the identical `crosscheck_body.body` on the CPU over the same inputs, and
//! compares element by element. A mismatch is a real finding: the GPU and CPU
//! evaluate one Zig function, so they must agree.
//!
//! Exit: 0 all match, 1 a mismatch was found, 77 no Metal device (SKIP), 2 bad
//! usage. The seed is reported so a finding can be reproduced with
//! `zig build crosscheck -Dseed=<N>`.
//!
//! Usage: crosscheck-driver --seed=<u64> [--body=<file>] [--reproduce=<command>] [--floats] <crosscheck.metallib>

const std = @import("std");
const builtin = @import("builtin");
const objc = @import("objc");
const crosscheck_body = @import("crosscheck_body");

extern "c" fn MTLCreateSystemDefaultDevice() ?objc.Object;

const MTLResourceStorageModeShared: u64 = 0;
const MTLFunctionTypeKernel: u64 = 3;
const MTLCommandBufferStatusError: u64 = 5;

const MTLSize = extern struct { width: u64, height: u64, depth: u64 };
fn size1d(len: u64) MTLSize {
    return .{ .width = len, .height = 1, .depth = 1 };
}

/// `count` must mirror src/engine/crosscheck_harness.zig's Params exactly (layout is
/// compared byte for byte by Metal's reflection).
const Params = extern struct { count: u32 };

/// Number of elements per run. A multiple of the threadgroup size so the grid
/// is exact.
const n: u32 = 4096;

/// Values that tend to break integer codegen: zero, identities, the two sign
/// boundaries, single bits and all-ones. The input stream is biased towards
/// these, with random i32 filling the rest.
const edges = [_]i32{
    0,                              1,                    -1,                             2,
    -2,                             std.math.maxInt(i32), std.math.minInt(i32),           std.math.maxInt(i32) - 1,
    std.math.minInt(i32) + 1,       0x55555555,           @bitCast(@as(u32, 0xaaaaaaaa)), 0x0000ffff,
    @bitCast(@as(u32, 0xffff0000)), 0x40000000,           -0x40000000,                    0x7f,
};

/// With `--floats`: the integer edges plus float bit patterns (both signs of
/// zero, ±1, 0.5, 0.1, ±inf, quiet and signalling NaN, FLT_MIN, FLT_MAX, the
/// largest denormal, 2^24 and the f32 -> i32 conversion limits).
const float_edges = edges ++ [_]i32{
    0x3f800000, @bitCast(@as(u32, 0xbf800000)), 0x3f000000,                     0x3dcccccd,
    0x7f800000, @bitCast(@as(u32, 0xff800000)), 0x7fc00000,                     0x7f800001,
    0x00800000, 0x7f7fffff,                     0x007fffff,                     0x4b800000,
    0x4b7fffff, 0x4f000000,                     @bitCast(@as(u32, 0xcf000000)), 0x40400000,
};

fn fillInputs(r: std.Random, buf: []i32, edge_list: []const i32) void {
    for (buf) |*v| {
        // ~40% edge values, the rest uniform random, so edge/edge, edge/random
        // and random/random pairs all occur.
        v.* = if (r.uintLessThan(u8, 10) < 4)
            edge_list[r.uintLessThan(usize, edge_list.len)]
        else
            r.int(i32);
    }
}

const SharedBuffer = struct {
    object: objc.Object,
    bytes: []u8,

    fn create(device: objc.Object, len: usize) !SharedBuffer {
        const object = objc.msgSend(?objc.Object, device, "newBufferWithLength:options:", .{ @as(u64, len), MTLResourceStorageModeShared }) orelse return error.NoBuffer;
        const ptr = objc.msgSend([*]u8, object, "contents", .{});
        return .{ .object = object, .bytes = ptr[0..len] };
    }

    fn items(self: SharedBuffer, comptime T: type) []T {
        return @alignCast(std.mem.bytesAsSlice(T, self.bytes));
    }
};

fn errorText(err: ?objc.Object) [*:0]const u8 {
    const e = err orelse return "(no NSError)";
    const desc = objc.msgSend(?objc.Object, e, "description", .{}) orelse return "(no description)";
    return objc.msgSend([*:0]const u8, desc, "UTF8String", .{});
}

pub fn main(init: std.process.Init) !void {
    const arena = init.arena.allocator();
    const io = init.io;

    const args = try init.minimal.args.toSlice(arena);
    var seed: ?u64 = null;
    var path: ?[]const u8 = null;
    // Set when build.zig ran a hand-written body (-Dcrosscheck-body), so the
    // messages name that file instead of a seed.
    var body_file: ?[]const u8 = null;
    // The exact command that reproduces this run, composed by build.zig.
    var reproduce: ?[]const u8 = null;
    // Float mode (docs/crosscheck-float-policy.md): more edge bit patterns.
    var floats = false;
    for (args[1..]) |arg| {
        if (std.mem.startsWith(u8, arg, "--seed=")) {
            seed = std.fmt.parseInt(u64, arg["--seed=".len..], 10) catch {
                std.debug.print("crosscheck-driver: bad --seed value\n", .{});
                std.process.exit(2);
            };
        } else if (std.mem.startsWith(u8, arg, "--body=")) {
            body_file = arg["--body=".len..];
        } else if (std.mem.startsWith(u8, arg, "--reproduce=")) {
            reproduce = arg["--reproduce=".len..];
        } else if (std.mem.eql(u8, arg, "--floats")) {
            floats = true;
        } else if (std.mem.startsWith(u8, arg, "-")) {
            std.debug.print("crosscheck-driver: unknown option '{s}'\n", .{arg});
            std.process.exit(2);
        } else if (path == null) {
            path = arg;
        } else {
            std.debug.print("crosscheck-driver: unexpected extra argument '{s}'\n", .{arg});
            std.process.exit(2);
        }
    }
    const s = seed orelse {
        std.debug.print("usage: crosscheck-driver --seed=<u64> [--body=<file>] [--reproduce=<command>] [--floats] <crosscheck.metallib>\n", .{});
        std.process.exit(2);
    };
    const lib_path = path orelse {
        std.debug.print("usage: crosscheck-driver --seed=<u64> [--body=<file>] [--reproduce=<command>] [--floats] <crosscheck.metallib>\n", .{});
        std.process.exit(2);
    };
    // `cpu-build` must read `debug`: that build's safety checks are the
    // undefined-behaviour detector (build.zig pins it).
    std.debug.print("info seed {d}  body.seed {d}  n {d}  cpu-build {t}\n", .{ s, crosscheck_body.seed, n, builtin.mode });

    // The input stream is derived from the seed but kept distinct from the
    // generator's own draws, so inputs and expression vary independently.
    var prng: std.Random.DefaultPrng = .init(s ^ 0x9e3779b97f4a7c15);
    const r = prng.random();
    const in0_host = try arena.alloc(i32, n);
    const in1_host = try arena.alloc(i32, n);
    const edge_list: []const i32 = if (floats) &float_edges else &edges;
    fillInputs(r, in0_host, edge_list);
    fillInputs(r, in1_host, edge_list);

    // CPU reference first: it needs no GPU, so a mismatch is attributable even
    // if the device later misbehaves.
    const want = try arena.alloc(i32, n);
    for (want, in0_host, in1_host) |*o, a, b| o.* = crosscheck_body.body(a, b);

    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, lib_path, arena, .unlimited);

    const pool = objc.objc_autoreleasePoolPush();
    defer objc.objc_autoreleasePoolPop(pool);

    const device = MTLCreateSystemDefaultDevice() orelse {
        std.debug.print("SKIP no Metal device available; nothing was checked\n", .{});
        std.process.exit(77);
    };
    const device_name = objc.msgSend(?objc.Object, device, "name", .{});
    std.debug.print("info device {s}\n", .{if (device_name) |nm| objc.msgSend([*:0]const u8, nm, "UTF8String", .{}) else "(unnamed)"});

    const data = objc.dispatch_data_create(bytes.ptr, bytes.len, null, null) orelse return error.DispatchDataFailed;
    defer objc.dispatch_release(data);
    var err: ?objc.Object = null;
    const library = objc.msgSend(?objc.Object, device, "newLibraryWithData:error:", .{ data, &err }) orelse {
        std.debug.print("FAIL newLibraryWithData: {s}\n", .{errorText(err)});
        std.process.exit(1);
    };
    const function = objc.msgSend(?objc.Object, library, "newFunctionWithName:", .{objc.createNSString("crosscheckKernel")}) orelse {
        std.debug.print("FAIL newFunctionWithName: crosscheckKernel not found\n", .{});
        std.process.exit(1);
    };
    if (objc.msgSend(u64, function, "functionType", .{}) != MTLFunctionTypeKernel) {
        std.debug.print("FAIL crosscheckKernel is not a kernel function\n", .{});
        std.process.exit(1);
    }
    const pso = objc.msgSend(?objc.Object, device, "newComputePipelineStateWithFunction:error:", .{ function, &err }) orelse {
        std.debug.print("FAIL newComputePipelineState crosscheckKernel: {s}\n", .{errorText(err)});
        std.process.exit(1);
    };

    // Device buffers, filled with the host inputs; output pre-poisoned so a
    // kernel that writes nothing is caught rather than reading as a match.
    const in0 = try SharedBuffer.create(device, n * 4);
    const in1 = try SharedBuffer.create(device, n * 4);
    const out = try SharedBuffer.create(device, n * 4);
    @memcpy(in0.items(i32), in0_host);
    @memcpy(in1.items(i32), in1_host);
    for (out.items(i32)) |*o| o.* = @bitCast(@as(u32, 0xdeadbeef));
    var params = Params{ .count = n };

    const queue = objc.msgSend(?objc.Object, device, "newCommandQueue", .{}) orelse return error.NoCommandQueue;
    const command_buffer = objc.msgSend(?objc.Object, queue, "commandBuffer", .{}) orelse return error.NoCommandBuffer;
    const encoder = objc.msgSend(?objc.Object, command_buffer, "computeCommandEncoder", .{}) orelse return error.NoEncoder;
    _ = objc.msgSend(void, encoder, "setComputePipelineState:", .{pso});
    _ = objc.msgSend(void, encoder, "setBuffer:offset:atIndex:", .{ in0.object, @as(u64, 0), @as(u64, 0) });
    _ = objc.msgSend(void, encoder, "setBuffer:offset:atIndex:", .{ in1.object, @as(u64, 0), @as(u64, 1) });
    _ = objc.msgSend(void, encoder, "setBuffer:offset:atIndex:", .{ out.object, @as(u64, 0), @as(u64, 2) });
    _ = objc.msgSend(void, encoder, "setBytes:length:atIndex:", .{ @as(*const anyopaque, @ptrCast(&params)), @as(u64, @sizeOf(Params)), @as(u64, 3) });
    _ = objc.msgSend(void, encoder, "dispatchThreads:threadsPerThreadgroup:", .{ size1d(n), size1d(64) });
    _ = objc.msgSend(void, encoder, "endEncoding", .{});
    _ = objc.msgSend(void, command_buffer, "commit", .{});
    _ = objc.msgSend(void, command_buffer, "waitUntilCompleted", .{});
    if (objc.msgSend(u64, command_buffer, "status", .{}) == MTLCommandBufferStatusError) {
        std.debug.print("FAIL command buffer: {s}\n", .{errorText(objc.msgSend(?objc.Object, command_buffer, "error", .{}))});
        std.process.exit(1);
    }

    var mismatches: u32 = 0;
    const got = out.items(i32);
    for (got, want, in0_host, in1_host, 0..) |g, wv, a, b, gid| {
        if (g != wv) {
            if (mismatches < 10) std.debug.print(
                "    MISMATCH [{d}] body({d}, {d}) = GPU {d} vs CPU {d}\n",
                .{ gid, a, b, g, wv },
            );
            mismatches += 1;
        }
    }
    if (body_file) |f| {
        if (mismatches == 0) {
            std.debug.print("ok  crosscheck body {s}: {d}/{d} match\n", .{ f, n, n });
            std.process.exit(0);
        }
        std.debug.print("FAIL crosscheck body {s}: {d}/{d} mismatched (reproduce: {s})\n", .{ f, mismatches, n, reproduce orelse "run it through zig build crosscheck" });
        std.process.exit(1);
    }
    if (mismatches == 0) {
        std.debug.print("ok  crosscheck seed {d}: {d}/{d} match\n", .{ s, n, n });
        std.process.exit(0);
    }
    std.debug.print("FAIL crosscheck seed {d}: {d}/{d} mismatched (reproduce: {s})\n", .{ s, mismatches, n, reproduce orelse "run it through zig build crosscheck" });
    std.process.exit(1);
}
