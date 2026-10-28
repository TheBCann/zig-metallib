//! spirv-probe-run: execute the probe kernels of tools/spirv/probes.zig on
//! the GPU and compare every lane with tools/spirv/reference.zig. Pipeline
//! creation (spirv-probe + metallib-check) proves Metal accepts a mapping;
//! this proves the lowered code computes what the SPIR-V instruction means.
//!
//!   spirv-probe-run [--fast-math=<bool>] [--only=<substring>] [--apple=<dir>]
//!
//! Then the whole-module tests of tools/spirv/exec.zig: fixed inputs, the
//! buffers bound at the manifest's indices, every checked output word
//! compared (`ok   exec <name>` / `FAIL exec <name>: ...`).
//!
//! Each probe is lowered and packed in process, its library loaded, 64
//! threads dispatched over generated inputs (edge values first, then random
//! bits; reference.prepare keeps them inside the instruction's defined
//! domain) and the stored results compared (under `--fast-math=true` only
//! lanes whose float inputs and result are finite, as fast math assumes). Prints `ok   <probe> (<n> lanes)`
//! or `FAIL <probe>: thread T lane L: inputs ... got ... want ...` per probe,
//! and a summary; exit 1 on any failure. Probes the reference does not model
//! print `skip`.
//!
//! `--apple=<dir>` runs the same checks on Apple's build of each module
//! instead of ours: spirv-splice's printed IR (`out.ll`, spelled for
//! Apple's front end by intrinsics.printForApple) is written to
//! `<dir>/apple.ll` and assembled with `xcrun metal -Xclang
//! -opaque-pointers` into `<dir>/apple.metallib` (both overwritten per
//! module). It is the attribution path: a module that passes ours and fails
//! Apple's, or the reverse, points at one of the two assemblers.

const std = @import("std");
const objc = @import("objc");
const reader = @import("spirv/reader.zig");
const asm_ = @import("spirv/asm.zig");
const lower = @import("spirv/lower.zig");
const probes = @import("spirv/probes.zig");
const reference = @import("spirv/reference.zig");
const exec = @import("spirv/exec.zig");
const splice = @import("spirv_splice.zig");

extern "c" fn MTLCreateSystemDefaultDevice() ?objc.Object;

const MTLResourceStorageModeShared: u64 = 0;
const MTLResourceUsageRead: u64 = 1;
const MTLResourceUsageWrite: u64 = 2;
const MTLCommandBufferStatusError: u64 = 5;
const MTLSize = extern struct { width: u64, height: u64, depth: u64 };
const threads = 64;

pub fn main(init: std.process.Init) !void {
    const arena = init.arena.allocator();
    const args = try init.minimal.args.toSlice(arena);
    var fast_math = false;
    var only: ?[]const u8 = null;
    var apple: ?Apple = null;
    for (args[1..]) |arg| {
        if (std.mem.eql(u8, arg, "--fast-math=true")) {
            fast_math = true;
        } else if (std.mem.eql(u8, arg, "--fast-math=false")) {
            fast_math = false;
        } else if (std.mem.startsWith(u8, arg, "--only=")) {
            only = arg["--only=".len..];
        } else if (std.mem.startsWith(u8, arg, "--apple=") and arg.len > "--apple=".len) {
            apple = .{ .io = init.io, .gpa = init.gpa, .dir = arg["--apple=".len..] };
        } else {
            std.debug.print("usage: spirv-probe-run [--fast-math=<bool>] [--only=<substring>] [--apple=<dir>]\n", .{});
            std.process.exit(2);
        }
    }
    const pool = objc.objc_autoreleasePoolPush();
    defer objc.objc_autoreleasePoolPop(pool);
    const device = MTLCreateSystemDefaultDevice() orelse {
        std.debug.print("SKIP no Metal device available\n", .{});
        std.process.exit(77);
    };
    const queue = objc.msgSend(?objc.Object, device, "newCommandQueue", .{}) orelse return error.NoQueue;

    var ok: usize = 0;
    var failed: usize = 0;
    var skipped: usize = 0;
    var lanes_checked: usize = 0;
    var rng = std.Random.DefaultPrng.init(0x5eed);
    for (try probes.all(arena)) |p| {
        if (only) |o| if (std.mem.find(u8, p.name, o) == null) continue;
        const check = reference.checkOf(p) orelse {
            skipped += 1;
            continue;
        };
        var pa = std.heap.ArenaAllocator.init(init.gpa);
        defer pa.deinit();
        const a = pa.allocator();
        const inner = objc.objc_autoreleasePoolPush();
        defer objc.objc_autoreleasePoolPop(inner);
        switch (try runProbe(a, device, queue, p, check, &rng, fast_math, apple)) {
            .ok => |n| {
                ok += 1;
                lanes_checked += n;
                std.debug.print("ok   {s} ({d} lanes)\n", .{ p.name, n });
            },
            .skip => skipped += 1,
            .fail => failed += 1,
        }
    }
    std.debug.print("{d} probes ok ({d} lanes), {d} failed, {d} not modelled\n", .{ ok, lanes_checked, failed, skipped });

    var exec_ok: usize = 0;
    var exec_failed: usize = 0;
    for (try exec.all(arena)) |t| {
        if (only) |o| if (std.mem.find(u8, t.name, o) == null) continue;
        if (fast_math and t.strict_only) {
            std.debug.print("skip exec {s} (checks a signed zero, which fast math may drop)\n", .{t.name});
            continue;
        }
        var pa = std.heap.ArenaAllocator.init(init.gpa);
        defer pa.deinit();
        const inner = objc.objc_autoreleasePoolPush();
        defer objc.objc_autoreleasePoolPop(inner);
        if (try runExec(pa.allocator(), device, queue, t, fast_math, apple)) {
            exec_ok += 1;
            std.debug.print("ok   exec {s}\n", .{t.name});
        } else exec_failed += 1;
    }
    std.debug.print("{d} execution tests ok, {d} failed\n", .{ exec_ok, exec_failed });
    if (failed > 0 or exec_failed > 0) std.process.exit(1);
}

/// One tools/spirv/exec.zig test: lower, build the pipeline, bind the
/// buffers at the manifest's indices, dispatch, compare.
/// `--apple`: where Apple's assembler builds each module.
const Apple = struct {
    io: std.Io,
    gpa: std.mem.Allocator,
    dir: []const u8,

    /// The library `xcrun metal -Xclang -opaque-pointers` builds from
    /// spirv-splice's printed IR, or null (the reason printed) when Apple's
    /// front end refuses it.
    fn image(ap: Apple, a: std.mem.Allocator, name: []const u8, text: []const u8) !?[]const u8 {
        const cwd = std.Io.Dir.cwd();
        const ll = try std.fs.path.join(a, &.{ ap.dir, "apple.ll" });
        const lib = try std.fs.path.join(a, &.{ ap.dir, "apple.metallib" });
        try cwd.writeFile(ap.io, .{ .sub_path = ll, .data = text });
        const r = try std.process.run(a, ap.io, .{ .argv = &.{ "xcrun", "metal", "-Xclang", "-opaque-pointers", ll, "-o", lib } });
        if (!r.term.success()) {
            std.debug.print("FAIL {s}: Apple's assembler: {s}\n", .{ name, r.stderr });
            return null;
        }
        return try cwd.readFileAlloc(ap.io, lib, a, .unlimited);
    }
};

fn runExec(a: std.mem.Allocator, device: objc.Object, queue: objc.Object, t: exec.Test, fast_math: bool, apple: ?Apple) !bool {
    const bytes = try asm_.assemble(a, t.text, null);
    var m = try reader.read(a, bytes, null);
    var diag: lower.Diagnostics = .{};
    const result = splice.splice(a, &m, .{ .fast_math = fast_math }, &diag) catch |err| {
        std.debug.print("FAIL exec {s}: lowering: {t} {s}\n", .{ t.name, err, diag.message });
        return false;
    };
    const image = if (apple) |ap| (try ap.image(a, t.name, result.text)) orelse return false else result.image;
    const data = objc.dispatch_data_create(image.ptr, image.len, null, null) orelse return error.DispatchData;
    defer objc.dispatch_release(data);
    var err: ?objc.Object = null;
    const library = objc.msgSend(?objc.Object, device, "newLibraryWithData:error:", .{ data, &err }) orelse {
        std.debug.print("FAIL exec {s}: newLibraryWithData\n", .{t.name});
        return false;
    };
    const name_z = try a.dupeSentinel(u8, t.name, 0);
    const function = objc.msgSend(?objc.Object, library, "newFunctionWithName:", .{objc.createNSString(name_z)}) orelse {
        std.debug.print("FAIL exec {s}: function not found\n", .{t.name});
        return false;
    };
    const pso = objc.msgSend(?objc.Object, device, "newComputePipelineStateWithFunction:error:", .{ function, &err }) orelse {
        std.debug.print("FAIL exec {s}: newComputePipelineState\n", .{t.name});
        return false;
    };
    for (result.kernels[0].params) |param| switch (param) {
        .buffer => |mb| {
            var bound = false;
            for (t.buffers) |b| bound = bound or b.index == mb.index;
            if (!bound) {
                std.debug.print("FAIL exec {s}: the test binds nothing at buffer {d}\n", .{ t.name, mb.index });
                return false;
            }
        },
        .builtin => {},
    };
    for (t.buffers) |b| {
        if (b.unbound) continue;
        var in_manifest = false;
        for (result.kernels[0].params) |param| switch (param) {
            .buffer => |mb| in_manifest = in_manifest or mb.index == b.index,
            .builtin => {},
        };
        if (!in_manifest) {
            std.debug.print("FAIL exec {s}: the manifest has no buffer {d}\n", .{ t.name, b.index });
            return false;
        }
    }
    const command_buffer = objc.msgSend(?objc.Object, queue, "commandBuffer", .{}) orelse return error.NoCommandBuffer;
    const encoder = objc.msgSend(?objc.Object, command_buffer, "computeCommandEncoder", .{}) orelse return error.NoEncoder;
    _ = objc.msgSend(void, encoder, "setComputePipelineState:", .{pso});
    const contents = try a.alloc([]u32, t.buffers.len);
    const bufs = try a.alloc(objc.Object, t.buffers.len);
    for (t.buffers, contents, bufs) |b, *c, *buf| {
        const len: u64 = @max(4, b.words.len * 4);
        buf.* = objc.msgSend(?objc.Object, device, "newBufferWithLength:options:", .{ len, MTLResourceStorageModeShared }) orelse return error.NoBuffer;
        c.* = @alignCast(std.mem.bytesAsSlice(u32, objc.msgSend([*]u8, buf.*, "contents", .{})[0..len]));
        @memset(c.*, 0);
        @memcpy(c.*[0..b.words.len], b.words);
        if (b.unbound) {
            // Reached only through an address: resident, not bound.
            _ = objc.msgSend(void, encoder, "useResource:usage:", .{ buf.*, MTLResourceUsageRead | MTLResourceUsageWrite });
        } else _ = objc.msgSend(void, encoder, "setBuffer:offset:atIndex:", .{ buf.*, @as(u64, 0), @as(u64, b.index) });
    }
    // Buffer addresses written into other buffers (GL_EXT_buffer_reference).
    for (t.buffers, contents) |b, c| for (b.addresses) |ad| {
        var addr: ?u64 = null;
        for (t.buffers, bufs) |ob, obuf| if (ob.index == ad.of) {
            addr = objc.msgSend(u64, obuf, "gpuAddress", .{});
        };
        const v = addr orelse return error.AddressOfUnknownBuffer;
        if (ad.word + 2 > c.len) return error.AddressPastBuffer;
        c[ad.word] = @truncate(v);
        c[ad.word + 1] = @truncate(v >> 32);
    };
    const grid = MTLSize{ .width = t.threads, .height = 1, .depth = 1 };
    const ls = result.kernels[0].local_size;
    const group = if (t.group_from_manifest) MTLSize{ .width = ls[0], .height = ls[1], .depth = ls[2] } else grid;
    _ = objc.msgSend(void, encoder, "dispatchThreads:threadsPerThreadgroup:", .{ grid, group });
    _ = objc.msgSend(void, encoder, "endEncoding", .{});
    _ = objc.msgSend(void, command_buffer, "commit", .{});
    _ = objc.msgSend(void, command_buffer, "waitUntilCompleted", .{});
    if (objc.msgSend(u64, command_buffer, "status", .{}) == MTLCommandBufferStatusError) {
        std.debug.print("FAIL exec {s}: command buffer error\n", .{t.name});
        return false;
    }
    for (t.buffers, contents) |b, c| {
        const want = b.expect orelse continue;
        if (b.expect_at + want.len > c.len) return error.ExpectPastBuffer;
        for (want, b.expect_at..) |w, i| if (c[i] != w) {
            std.debug.print("FAIL exec {s}: buffer {d} word {d}: got 0x{x:0>8} want 0x{x:0>8}\n", .{ t.name, b.index, i, c[i], w });
            return false;
        };
    }
    return true;
}

const Outcome = union(enum) { ok: usize, skip, fail };

fn runProbe(a: std.mem.Allocator, device: objc.Object, queue: objc.Object, p: probes.Probe, check: reference.Check, rng: *std.Random.DefaultPrng, fast_math: bool, apple: ?Apple) !Outcome {
    const text = try probes.moduleText(a, p);
    const bytes = try asm_.assemble(a, text, null);
    var m = try reader.read(a, bytes, null);
    var diag: lower.Diagnostics = .{};
    const result = splice.splice(a, &m, .{ .fast_math = fast_math }, &diag) catch |err| {
        std.debug.print("FAIL {s}: lowering: {t} {s}\n", .{ p.name, err, diag.message });
        return .fail;
    };

    // Inputs per thread, fixed up by the reference, and the expectations.
    const ins = try a.alloc(reference.Inputs, threads);
    const want = try a.alloc(reference.Expect, threads);
    for (ins, want, 0..) |*in, *w, t| {
        in.* = .{ .{ 0, 0, 0, 0 }, .{ 0, 0, 0, 0 }, .{ 0, 0, 0, 0 } };
        for (p.operands, 0..) |ty, op| {
            const e = reference.elemOf(ty);
            for (0..e.lanes) |l| in[op][l] = reference.inputBits(e, op, t, l, rng);
        }
        w.* = reference.prepare(p, in) orelse return .skip;
    }

    const image = if (apple) |ap| (try ap.image(a, p.name, result.text)) orelse return .fail else result.image;
    const data = objc.dispatch_data_create(image.ptr, image.len, null, null) orelse return error.DispatchData;
    defer objc.dispatch_release(data);
    var err: ?objc.Object = null;
    const library = objc.msgSend(?objc.Object, device, "newLibraryWithData:error:", .{ data, &err }) orelse {
        std.debug.print("FAIL {s}: newLibraryWithData\n", .{p.name});
        return .fail;
    };
    const name_z = try a.dupeSentinel(u8, p.name, 0);
    const function = objc.msgSend(?objc.Object, library, "newFunctionWithName:", .{objc.createNSString(name_z)}) orelse {
        std.debug.print("FAIL {s}: function not found\n", .{p.name});
        return .fail;
    };
    const pso = objc.msgSend(?objc.Object, device, "newComputePipelineStateWithFunction:error:", .{ function, &err }) orelse {
        std.debug.print("FAIL {s}: newComputePipelineState\n", .{p.name});
        return .fail;
    };

    const out_elem = reference.storeElem(p);
    var buffers: [4]objc.Object = undefined;
    var contents: [4][]u8 = undefined;
    const elems = [_]reference.Elem{
        if (p.operands.len > 0) reference.elemOf(p.operands[0]) else out_elem,
        if (p.operands.len > 1) reference.elemOf(p.operands[1]) else out_elem,
        if (p.operands.len > 2) reference.elemOf(p.operands[2]) else out_elem,
        out_elem,
    };
    for (&buffers, &contents, elems, 0..) |*buf, *c, e, i| {
        const len: u64 = threads * e.stride();
        buf.* = objc.msgSend(?objc.Object, device, "newBufferWithLength:options:", .{ len, MTLResourceStorageModeShared }) orelse return error.NoBuffer;
        c.* = objc.msgSend([*]u8, buf.*, "contents", .{})[0..len];
        @memset(c.*, 0);
        if (i < p.operands.len) {
            for (0..threads) |t| for (0..e.lanes) |l| writeLane(c.*, e, t, l, ins[t][i][l]);
        }
    }

    const command_buffer = objc.msgSend(?objc.Object, queue, "commandBuffer", .{}) orelse return error.NoCommandBuffer;
    const encoder = objc.msgSend(?objc.Object, command_buffer, "computeCommandEncoder", .{}) orelse return error.NoEncoder;
    _ = objc.msgSend(void, encoder, "setComputePipelineState:", .{pso});
    // Buffer indices from the manifest: operand i is binding i (Vulkan) or
    // parameter i (OpenCL); the result is binding 3 or the last parameter.
    for (result.kernels[0].params) |param| switch (param) {
        .buffer => |b| {
            const slot: usize = switch (b.source) {
                .binding => |d| if (d.binding == 3) 3 else d.binding,
                .parameter => |n| if (n == p.operands.len) 3 else n,
                else => return error.UnexpectedBuffer,
            };
            _ = objc.msgSend(void, encoder, "setBuffer:offset:atIndex:", .{ buffers[slot], @as(u64, 0), @as(u64, b.index) });
        },
        .builtin => {},
    };
    const grid = MTLSize{ .width = threads, .height = 1, .depth = 1 };
    _ = objc.msgSend(void, encoder, "dispatchThreads:threadsPerThreadgroup:", .{ grid, grid });
    _ = objc.msgSend(void, encoder, "endEncoding", .{});
    _ = objc.msgSend(void, command_buffer, "commit", .{});
    _ = objc.msgSend(void, command_buffer, "waitUntilCompleted", .{});
    if (objc.msgSend(u64, command_buffer, "status", .{}) == MTLCommandBufferStatusError) {
        std.debug.print("FAIL {s}: command buffer error\n", .{p.name});
        return .fail;
    }

    var checked: usize = 0;
    for (0..threads) |t| for (0..out_elem.lanes) |l| {
        if (!want[t].defined[l]) continue;
        // Fast math assumes finite values (Apple's air.fast_fmod(-0, inf) is
        // NaN): under --fast-math=true, lanes with a non-finite float input
        // or result are not compared.
        if (fast_math and !allFinite(p, ins[t], out_elem, want[t].lanes[l])) continue;
        const got = readLane(contents[3], out_elem, t, l);
        checked += 1;
        if (!reference.laneMatches(out_elem, check, want[t].lanes[l], got)) {
            std.debug.print("FAIL {s}: thread {d} lane {d}: inputs 0x{x} 0x{x} 0x{x}: got 0x{x} want 0x{x}\n", .{ p.name, t, l, ins[t][0][l], ins[t][1][l], ins[t][2][l], got, want[t].lanes[l] });
            return .fail;
        }
    };
    return .{ .ok = checked };
}

fn laneBytes(e: reference.Elem) usize {
    return @max(1, e.bits / 8);
}

fn writeLane(buf: []u8, e: reference.Elem, thread: usize, lane: usize, v: u64) void {
    const nb = laneBytes(e);
    const off = thread * e.stride() + lane * nb;
    for (0..nb) |k| buf[off + k] = @truncate(v >> @intCast(8 * k));
}

fn readLane(buf: []const u8, e: reference.Elem, thread: usize, lane: usize) u64 {
    const nb = laneBytes(e);
    const off = thread * e.stride() + lane * nb;
    var v: u64 = 0;
    for (0..nb) |k| v |= @as(u64, buf[off + k]) << @intCast(8 * k);
    return v;
}

fn finite(e: reference.Elem, v: u64) bool {
    if (e.kind != .float) return true;
    const x: f64 = if (e.bits == 16) @as(f16, @bitCast(@as(u16, @truncate(v)))) else @as(f32, @bitCast(@as(u32, @truncate(v))));
    return std.math.isFinite(x);
}

fn allFinite(p: probes.Probe, in: reference.Inputs, out: reference.Elem, want: u64) bool {
    if (!finite(out, want)) return false;
    for (p.operands, 0..) |ty, op| {
        const e = reference.elemOf(ty);
        for (0..e.lanes) |l| if (!finite(e, in[op][l])) return false;
    }
    return true;
}
