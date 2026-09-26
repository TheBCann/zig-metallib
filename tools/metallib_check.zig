//! metallib-check: load a .metallib into Metal and validate it the way the
//! app would, from the shader manifest (DESIGN.md D7):
//!   * the deployment target the header is stamped for (tools/air/target.zig)
//!     and the library's own VERS, printed as an `info` line, with a `note`
//!     when the target is a newer macOS major than the running one (Metal
//!     then rejects it with "Unsupported target triple"),
//!   * `newLibraryWithData:` and `newFunctionWithName:` for every entry,
//!   * a render pipeline for every (vertex, fragment) pair whose `stage_in`
//!     type is the vertex output type, with the colour / depth attachment
//!     formats the fragment's return type asks for (a `colorN` field sets
//!     attachment N, `depth` sets depthAttachmentPixelFormat),
//!   * a compute pipeline for every kernel (Metal's runtime compiler is the
//!     real gate: `xcrun metal` accepting the IR proves nothing),
//!   * dispatch self-tests, keyed by kernel name, that run the sample
//!     kernels with known inputs and verify the results read back from
//!     shared MTLBuffers,
//!   * an offscreen render of the instanced / MRT sample pair (two colour
//!     attachments + depth, one point, `baseInstance` 3) that reads one
//!     pixel per attachment back through a blit and compares it with what
//!     the shaders must produce (the discarded pixel included).
//! Prints one `ok  ...` line per check, `FAIL ...` and exit code 1 otherwise.
//! An `info device` line names the GPU the checks ran on (a CI runner's VM
//! reports a paravirtual device, not an Apple GPU), and every FAIL from a
//! Metal call that returns an NSError carries the whole error (domain, code,
//! userInfo), not only its one-line summary.
//! Without a Metal device (a CI runner, say) it prints `SKIP ...` and exits 77.
//! All Metal calls go through objc.msgSend; no C or Objective-C sources.
//!
//! With `--kernel=<name>` it skips the manifest and builds a compute pipeline
//! for one function of any library, e.g. a kernel Apple's compiler built: the
//! control that tells "this GPU cannot build pipelines" apart from "this
//! project's output is rejected" (.github/workflows/ci.yml).
//!
//! Usage: metallib-check [--kernel=<name>] <file.metallib>

const std = @import("std");
const objc = @import("objc");
const shader = @import("shader");
const air_target = @import("air/target.zig");
const container = @import("air/metallib.zig");

extern "c" fn MTLCreateSystemDefaultDevice() ?objc.Object;

// MTLPixelFormat.h
const MTLPixelFormatBGRA8Unorm: u64 = 80;
const MTLPixelFormatRGBA16Float: u64 = 115;
const MTLPixelFormatRGBA32Uint: u64 = 123;
const MTLPixelFormatRGBA32Sint: u64 = 124;
const MTLPixelFormatRGBA32Float: u64 = 125;
const MTLPixelFormatDepth32Float: u64 = 252;
/// MTLResourceStorageModeShared: CPU and GPU share the buffer memory.
const MTLResourceStorageModeShared: u64 = 0;
const MTLStorageModePrivate: u64 = 2;
const MTLTextureUsageRenderTarget: u64 = 4;
const MTLCommandBufferStatusError: u64 = 5;
const MTLLoadActionClear: u64 = 2;
const MTLStoreActionStore: u64 = 1;
const MTLCompareFunctionAlways: u64 = 7;
const MTLPrimitiveTypePoint: u64 = 0;
const MTLBlitOptionNone: u64 = 0;
const MTLBlitOptionDepthFromDepthStencil: u64 = 1;
const MTLPixelFormatRGBA8Unorm: u64 = 70;
const MTLTextureUsageShaderRead: u64 = 1;
const MTLTextureUsageShaderWrite: u64 = 2;
const MTLPrimitiveTypeTriangle: u64 = 3;
const MTLSamplerMinMagFilterNearest: u64 = 0;
const MTLSamplerAddressModeClampToEdge: u64 = 0;
const MTLFunctionTypeKernel: u64 = 3;

/// Passed by value through objc_msgSend; the comptime signature builder in
/// objc.zig gives the call the C ABI, which handles the 24-byte struct.
const MTLSize = extern struct { width: u64, height: u64, depth: u64 };
const MTLOrigin = extern struct { x: u64, y: u64, z: u64 };
/// Four doubles: a homogeneous float aggregate on arm64, passed in v0-v3.
const MTLClearColor = extern struct { red: f64, green: f64, blue: f64, alpha: f64 };

fn size1d(n: u64) MTLSize {
    return .{ .width = n, .height = 1, .depth = 1 };
}

var failed = false;

pub fn main(init: std.process.Init) !void {
    const arena = init.arena.allocator();
    const io = init.io;

    const args = try init.minimal.args.toSlice(arena);
    var kernel_only: ?[:0]const u8 = null;
    var path: []const u8 = undefined;
    if (args.len == 3 and std.mem.startsWith(u8, args[1], "--kernel=") and args[1].len > "--kernel=".len) {
        kernel_only = args[1]["--kernel=".len..];
        path = args[2];
    } else if (args.len == 2 and !std.mem.startsWith(u8, args[1], "-")) {
        path = args[1];
    } else {
        std.debug.print("usage: metallib-check [--kernel=<name>] <file.metallib>\n", .{});
        std.process.exit(2);
    }
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, arena, .unlimited);

    // Which deployment target the library is stamped for (tools/air/target.zig).
    const stamp = container.readStamp(bytes) orelse {
        if (std.mem.startsWith(u8, bytes, "MTLB")) {
            std.debug.print("FAIL {s} is truncated: {d} bytes, shorter than the 88-byte MTLB header\n", .{ path, bytes.len });
        } else {
            std.debug.print("FAIL {s} is not a .metallib (no MTLB header)\n", .{path});
        }
        std.process.exit(1);
    };
    const lib_target = stamp.macos;
    if (stamp.vers) |v| {
        std.debug.print("info library targets macOS {d}.{d}, compiled as AIR {d}.{d} / Metal {d}.{d}", .{ lib_target.major, lib_target.minor, v[0], v[1], v[2], v[3] });
    } else {
        std.debug.print("info library targets macOS {d}.{d} (no VERS tag found)", .{ lib_target.major, lib_target.minor });
    }
    if (air_target.fromHeader(stamp.container_minor, lib_target.major)) |p| {
        std.debug.print(" (profile {t}{s})\n", .{ p.name, if (p.verified) "" else ", unverified" });
    } else {
        std.debug.print(" (format 2.{d}: no known profile)\n", .{stamp.container_minor});
    }
    // Metal refuses a newer macOS major than the running one and accepts newer
    // minors. It decides from the bitcode triple, which this header mirrors,
    // so this is a note that explains the failure to come, not a verdict.
    if (hostMacos()) |host| {
        if (!lib_target.acceptedOn(host)) {
            std.debug.print("note library targets macOS {d}.{d} but this is macOS {d}.{d}: Metal rejects a newer macOS major (\"Unsupported target triple\"); rebuild for this macOS with -Dmetal-target\n", .{ lib_target.major, lib_target.minor, host.major, host.minor });
        }
    }

    const pool = objc.objc_autoreleasePoolPush();
    defer objc.objc_autoreleasePoolPop(pool);

    const device = MTLCreateSystemDefaultDevice() orelse {
        // A machine without a usable GPU (a CI runner, say) lands here. Exit 77,
        // the conventional "skipped" code, so a workflow can tell "not run"
        // from "failed".
        std.debug.print("SKIP no Metal device available; nothing was checked\n", .{});
        std.process.exit(77);
    };
    // Which GPU the checks run on: "Apple M3" on a Mac, a paravirtual device
    // inside a virtual machine such as a CI runner.
    const device_name = objc.msgSend(?objc.Object, device, "name", .{});
    std.debug.print("info device {s}\n", .{if (device_name) |n| objc.msgSend([*:0]const u8, n, "UTF8String", .{}) else "(unnamed)"});

    const data = objc.dispatch_data_create(bytes.ptr, bytes.len, null, null) orelse return error.DispatchDataFailed;
    defer objc.dispatch_release(data);

    var err: ?objc.Object = null;
    const library = objc.msgSend(?objc.Object, device, "newLibraryWithData:error:", .{ data, &err }) orelse {
        std.debug.print("FAIL newLibraryWithData: {s}\n", .{errorText(err)});
        std.process.exit(1);
    };
    std.debug.print("ok  newLibraryWithData ({d} bytes)\n", .{bytes.len});

    // Control mode: one compute pipeline from any library, no manifest.
    if (kernel_only) |name| {
        const ns_name = objc.createNSString(name) orelse {
            std.debug.print("FAIL --kernel={s}: not a valid UTF-8 function name\n", .{name});
            std.process.exit(2);
        };
        const function = objc.msgSend(?objc.Object, library, "newFunctionWithName:", .{ns_name}) orelse {
            std.debug.print("FAIL newFunctionWithName: {s} not found\n", .{name});
            std.process.exit(1);
        };
        std.debug.print("ok  function {s}\n", .{name});
        // Only a kernel makes a compute pipeline. A vertex function gets an
        // unrelated error and a fragment one crashes MTLCompilerService, which
        // would read as "unsupported IR" rather than "wrong function".
        const function_type = objc.msgSend(u64, function, "functionType", .{});
        if (function_type != MTLFunctionTypeKernel) {
            std.debug.print("FAIL {s} is not a kernel function (MTLFunctionType {d}; 1 is vertex, 2 fragment, 3 kernel)\n", .{ name, function_type });
            std.process.exit(1);
        }
        const pso = objc.msgSend(?objc.Object, device, "newComputePipelineStateWithFunction:error:", .{ function, &err }) orelse {
            std.debug.print("FAIL newComputePipelineState {s}: {s}\n", .{ name, errorText(err) });
            std.process.exit(1);
        };
        std.debug.print("ok  kernel {s} maxTotalThreadsPerThreadgroup={d}\n", .{ name, objc.msgSend(u64, pso, "maxTotalThreadsPerThreadgroup", .{}) });
        return;
    }

    // One MTLFunction per manifest entry, and a compute pipeline per kernel.
    var functions: [shader.functions.len]objc.Object = undefined;
    var pipelines: [shader.functions.len]?objc.Object = @splat(null);
    inline for (shader.functions, 0..) |f, i| {
        functions[i] = objc.msgSend(?objc.Object, library, "newFunctionWithName:", .{objc.createNSString(f.name)}) orelse {
            std.debug.print("FAIL newFunctionWithName: {s} not found\n", .{f.name});
            std.process.exit(1);
        };
        std.debug.print("ok  function {s}\n", .{f.name});
        if (f.stage == .kernel) {
            const pso = objc.msgSend(?objc.Object, device, "newComputePipelineStateWithFunction:error:", .{ functions[i], &err }) orelse {
                std.debug.print("FAIL newComputePipelineState {s}: {s}\n", .{ f.name, errorText(err) });
                std.process.exit(1);
            };
            pipelines[i] = pso;
            std.debug.print("ok  kernel {s} maxTotalThreadsPerThreadgroup={d} staticThreadgroupMemoryLength={d}\n", .{
                f.name,
                objc.msgSend(u64, pso, "maxTotalThreadsPerThreadgroup", .{}),
                objc.msgSend(u64, pso, "staticThreadgroupMemoryLength", .{}),
            });
        }
    }

    // Render pipelines: every vertex function with every fragment whose
    // stage_in type is its output type.
    inline for (shader.functions, 0..) |v, vi| {
        if (v.stage == .vertex) {
            inline for (shader.functions, 0..) |f, fi| {
                if (f.stage == .fragment and comptime stageInType(f) == v.ret) {
                    try checkRenderPipeline(device, v.name, functions[vi], f.name, functions[fi], f.ret);
                }
            }
        }
    }

    // Offscreen render self-test for the instanced / MRT sample pair.
    if (comptime findFunction("vertexShaderInstanced")) |vi| {
        if (comptime findFunction("fragmentShaderMRT")) |fi| try testInstancedMRT(device, functions[vi], functions[fi]);
    }

    // Textured renders: the sampling path is only proven by reading pixels
    // back, once with the shader's own constexpr sampler and once with a
    // sampler the host binds.
    if (comptime findFunction("vertexShader")) |vi| {
        if (comptime findFunction("fragmentShader")) |fi| {
            try testTexturedRender(device, functions[vi], functions[fi], "vertexShader + fragmentShader (constexpr sampler)", false);
        }
        if (comptime findFunction("fragmentShaderBoundSampler")) |fi| {
            try testTexturedRender(device, functions[vi], functions[fi], "vertexShader + fragmentShaderBoundSampler (bound sampler)", true);
        }
    }

    // Dispatch self-tests for the sample kernels that are in the manifest.
    if (comptime findFunction("scaleKernel")) |i| try testScaleKernel(device, pipelines[i].?);
    if (comptime findFunction("reverseKernel")) |i| try testReverseKernel(device, pipelines[i].?);
    if (comptime findFunction("sumKernel")) |i| try testSumKernel(device, pipelines[i].?);
    if (comptime findFunction("countKernel")) |i| try testCountKernel(device, pipelines[i].?);
    if (comptime findFunction("atomicOpsKernel")) |i| try testAtomicOpsKernel(device, pipelines[i].?);
    if (comptime findFunction("simdOpsKernel")) |i| try testSimdOpsKernel(device, pipelines[i].?);
    if (comptime findFunction("tgBufKernel")) |i| try testTgBufKernel(device, pipelines[i].?);
    if (comptime findFunction("reduceKernel")) |i| try testReduceKernel(device, pipelines[i].?);
    if (comptime findFunction("twoStageKernel")) |i| try testTwoStageKernel(device, pipelines[i].?);
    if (comptime findFunction("copyTextureKernel")) |i| try testCopyTextureKernel(device, pipelines[i].?);

    if (failed) std.process.exit(1);
}

/// The running macOS version, from `kern.osproductversion` ("26.3"); null
/// when it cannot be read.
fn hostMacos() ?air_target.Version {
    var buf: [32]u8 = undefined;
    var len: usize = buf.len;
    if (std.c.sysctlbyname("kern.osproductversion", &buf, &len, null, 0) != 0) return null;
    return air_target.Version.parse(std.mem.sliceTo(buf[0..len], 0));
}

fn stageInType(comptime f: shader.air.Function) ?type {
    for (f.args) |arg| switch (arg) {
        .stage_in => |T| return T,
        else => {},
    };
    return null;
}

fn findFunction(comptime name: []const u8) ?usize {
    for (shader.functions, 0..) |f, i| if (std.mem.eql(u8, f.name, name)) return i;
    return null;
}

fn checkRenderPipeline(device: objc.Object, vname: []const u8, vfn: objc.Object, fname: []const u8, ffn: objc.Object, comptime FragRet: type) !void {
    _ = makeRenderPipeline(device, vfn, ffn, FragRet, MTLPixelFormatBGRA8Unorm) catch |err| {
        std.debug.print("FAIL newRenderPipelineState {s} + {s}: {s}\n", .{ vname, fname, errorText(last_error) });
        if (err == error.NoPipeline) std.process.exit(1);
        return err;
    };
    std.debug.print("ok  render pipeline state {s} + {s}\n", .{ vname, fname });
}

var last_error: ?objc.Object = null;

/// A render pipeline for `FragRet`, the fragment's manifest return type
/// (DESIGN.md D7): a vector return is `[[color(0)]]`; a struct's `colorN`
/// fields set colour attachment N and a `depth` field sets
/// `depthAttachmentPixelFormat` (Depth32Float; without it Metal refuses
/// the pipeline with "depthAttachmentPixelFormat is not valid and shader
/// writes to depth", research/_results/r01.txt finding 5). Formats follow
/// the element type: half4 -> RGBA16Float, uint4 -> RGBA32Uint (a
/// BGRA8Unorm attachment rejects a uint4 output), int4 -> RGBA32Sint,
/// float4 -> `float_format` (BGRA8Unorm, the app's drawable format, for the
/// pipeline checks; RGBA32Float for the readback test).
fn makeRenderPipeline(device: objc.Object, vfn: objc.Object, ffn: objc.Object, comptime FragRet: type, float_format: u64) !objc.Object {
    const desc = objc.msgSend(?objc.Object, objc.class("MTLRenderPipelineDescriptor"), "new", .{}) orelse return error.NoDescriptor;
    _ = objc.msgSend(void, desc, "setVertexFunction:", .{vfn});
    _ = objc.msgSend(void, desc, "setFragmentFunction:", .{ffn});
    const attachments = objc.msgSend(?objc.Object, desc, "colorAttachments", .{}) orelse return error.NoAttachments;
    if (@typeInfo(FragRet) == .@"struct") {
        const st = @typeInfo(FragRet).@"struct";
        inline for (st.field_names, st.field_types) |name, ft| {
            if (comptime std.mem.eql(u8, name, "depth")) {
                _ = objc.msgSend(void, desc, "setDepthAttachmentPixelFormat:", .{MTLPixelFormatDepth32Float});
            } else {
                const idx = comptime renderTargetIndex(name);
                const attach = objc.msgSend(?objc.Object, attachments, "objectAtIndexedSubscript:", .{idx}) orelse return error.NoAttachment;
                _ = objc.msgSend(void, attach, "setPixelFormat:", .{pixelFormatFor(ft, float_format)});
            }
        }
    } else {
        const attach0 = objc.msgSend(?objc.Object, attachments, "objectAtIndexedSubscript:", .{@as(u64, 0)}) orelse return error.NoAttachment;
        _ = objc.msgSend(void, attach0, "setPixelFormat:", .{pixelFormatFor(FragRet, float_format)});
    }
    last_error = null;
    return objc.msgSend(?objc.Object, device, "newRenderPipelineStateWithDescriptor:error:", .{ desc, &last_error }) orelse error.NoPipeline;
}

/// `color0` -> 0, `color3` -> 3 (the manifest's fragment output rule, D2).
fn renderTargetIndex(comptime name: []const u8) u64 {
    if (!std.mem.startsWith(u8, name, "color")) @compileError("fragment output field '" ++ name ++ "' must be color<N> or depth");
    return std.fmt.parseInt(u64, name["color".len..], 10) catch @compileError("fragment output field '" ++ name ++ "' must be color<N> or depth");
}

fn pixelFormatFor(comptime T: type, float_format: u64) u64 {
    const v = @typeInfo(T).vector;
    return switch (@typeInfo(v.child)) {
        .float => |f| if (f.bits == 16) MTLPixelFormatRGBA16Float else float_format,
        .int => |n| if (n.signedness == .unsigned) MTLPixelFormatRGBA32Uint else MTLPixelFormatRGBA32Sint,
        else => @compileError("unsupported render target type " ++ @typeName(T)),
    };
}

// ── Dispatch plumbing ───────────────────────────────────────────────────────

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

/// One compute pass: encoder setup, `dispatch`, commit, wait.
const Dispatch = struct {
    queue: objc.Object,
    command_buffer: objc.Object,
    encoder: objc.Object,

    fn begin(device: objc.Object, pso: objc.Object) !Dispatch {
        const queue = objc.msgSend(?objc.Object, device, "newCommandQueue", .{}) orelse return error.NoCommandQueue;
        const command_buffer = objc.msgSend(?objc.Object, queue, "commandBuffer", .{}) orelse return error.NoCommandBuffer;
        const encoder = objc.msgSend(?objc.Object, command_buffer, "computeCommandEncoder", .{}) orelse return error.NoEncoder;
        _ = objc.msgSend(void, encoder, "setComputePipelineState:", .{pso});
        return .{ .queue = queue, .command_buffer = command_buffer, .encoder = encoder };
    }

    fn setBuffer(self: Dispatch, buffer: SharedBuffer, index: u64) void {
        _ = objc.msgSend(void, self.encoder, "setBuffer:offset:atIndex:", .{ buffer.object, @as(u64, 0), index });
    }

    fn setBytes(self: Dispatch, value: anytype, index: u64) void {
        _ = objc.msgSend(void, self.encoder, "setBytes:length:atIndex:", .{ @as(*const anyopaque, @ptrCast(value)), @as(u64, @sizeOf(@TypeOf(value.*))), index });
    }

    /// `setThreadgroupMemoryLength:atIndex:`: size of a `[[threadgroup(index)]]`
    /// buffer argument (threadgroup buffers have their own index space).
    fn setThreadgroupMemoryLength(self: Dispatch, len: u64, index: u64) void {
        _ = objc.msgSend(void, self.encoder, "setThreadgroupMemoryLength:atIndex:", .{ len, index });
    }

    /// `dispatchThreads:threadsPerThreadgroup:` (non-uniform threadgroups allowed).
    fn dispatchThreads(self: Dispatch, threads: u64, per_group: u64) void {
        _ = objc.msgSend(void, self.encoder, "dispatchThreads:threadsPerThreadgroup:", .{ size1d(threads), size1d(per_group) });
    }

    /// The same over a 2D grid, for a kernel taking `uint2` thread positions.
    fn dispatchThreads2D(self: Dispatch, w: u64, h: u64) void {
        const grid = MTLSize{ .width = w, .height = h, .depth = 1 };
        _ = objc.msgSend(void, self.encoder, "dispatchThreads:threadsPerThreadgroup:", .{ grid, grid });
    }

    /// `[[texture(index)]]` of a kernel.
    fn setTexture(self: Dispatch, texture: objc.Object, index: u64) void {
        _ = objc.msgSend(void, self.encoder, "setTexture:atIndex:", .{ texture, index });
    }

    /// `dispatchThreadgroups:threadsPerThreadgroup:`.
    fn dispatchThreadgroups(self: Dispatch, groups: u64, per_group: u64) void {
        _ = objc.msgSend(void, self.encoder, "dispatchThreadgroups:threadsPerThreadgroup:", .{ size1d(groups), size1d(per_group) });
    }

    fn finish(self: Dispatch) !void {
        _ = objc.msgSend(void, self.encoder, "endEncoding", .{});
        _ = objc.msgSend(void, self.command_buffer, "commit", .{});
        _ = objc.msgSend(void, self.command_buffer, "waitUntilCompleted", .{});
        if (objc.msgSend(u64, self.command_buffer, "status", .{}) == MTLCommandBufferStatusError) {
            const cb_err = objc.msgSend(?objc.Object, self.command_buffer, "error", .{});
            std.debug.print("FAIL command buffer: {s}\n", .{errorText(cb_err)});
            return error.CommandBufferFailed;
        }
    }
};

fn report(name: []const u8, ok: bool, comptime fmt: []const u8, args: anytype) void {
    std.debug.print("{s}  dispatch {s}: " ++ fmt ++ "\n", .{ if (ok) "ok" else "FAIL", name } ++ args);
    if (!ok) failed = true;
}

fn reportRender(name: []const u8, ok: bool, comptime fmt: []const u8, args: anytype) void {
    std.debug.print("{s}  render {s}: " ++ fmt ++ "\n", .{ if (ok) "ok" else "FAIL", name } ++ args);
    if (!ok) failed = true;
}

// ── Offscreen render plumbing ───────────────────────────────────────────────

/// A private render-target texture; read back through a blit into a
/// shared buffer (`copyToBuffer`), so no `getBytes:` and no MTLRegion.
fn renderTarget(device: objc.Object, format: u64, width: u64, height: u64) !objc.Object {
    const td = objc.msgSend(?objc.Object, objc.class("MTLTextureDescriptor"), "texture2DDescriptorWithPixelFormat:width:height:mipmapped:", .{ format, width, height, false }) orelse return error.NoTextureDescriptor;
    _ = objc.msgSend(void, td, "setUsage:", .{MTLTextureUsageRenderTarget});
    _ = objc.msgSend(void, td, "setStorageMode:", .{MTLStorageModePrivate});
    return objc.msgSend(?objc.Object, device, "newTextureWithDescriptor:", .{td}) orelse error.NoTexture;
}

/// `copyFromTexture:...toBuffer:...options:` of the whole level-0 image,
/// tightly packed at `bytes_per_pixel`.
fn copyToBuffer(blit: objc.Object, texture: objc.Object, buffer: SharedBuffer, width: u64, height: u64, bytes_per_pixel: u64, options: u64) void {
    _ = objc.msgSend(void, blit, "copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:sourceSize:toBuffer:destinationOffset:destinationBytesPerRow:destinationBytesPerImage:options:", .{
        texture,
        @as(u64, 0),
        @as(u64, 0),
        MTLOrigin{ .x = 0, .y = 0, .z = 0 },
        MTLSize{ .width = width, .height = height, .depth = 1 },
        buffer.object,
        @as(u64, 0),
        width * bytes_per_pixel,
        width * height * bytes_per_pixel,
        options,
    });
}

fn setColorAttachment(pass: objc.Object, index: u64, texture: objc.Object, clear: MTLClearColor) !void {
    const attachments = objc.msgSend(?objc.Object, pass, "colorAttachments", .{}) orelse return error.NoAttachments;
    const a = objc.msgSend(?objc.Object, attachments, "objectAtIndexedSubscript:", .{index}) orelse return error.NoAttachment;
    _ = objc.msgSend(void, a, "setTexture:", .{texture});
    _ = objc.msgSend(void, a, "setLoadAction:", .{MTLLoadActionClear});
    _ = objc.msgSend(void, a, "setStoreAction:", .{MTLStoreActionStore});
    _ = objc.msgSend(void, a, "setClearColor:", .{clear});
}

fn approx(got: f32, want: f32) bool {
    return @abs(got - want) <= 1e-5;
}

// ── Self-tests ──────────────────────────────────────────────────────────────

/// scaleKernel(in, out, Params{count, scale}, gid): out[gid] = in[gid] * scale for gid < count.
fn testScaleKernel(device: objc.Object, pso: objc.Object) !void {
    const n: u32 = 1024;
    const in = try SharedBuffer.create(device, n * 4);
    const out = try SharedBuffer.create(device, n * 4);
    for (in.items(f32), out.items(f32), 0..) |*i, *o, k| {
        i.* = @floatFromInt(k);
        o.* = -1.0;
    }
    const params = shader.Params{ .count = n - 10, .scale = 3.0 };

    const d = try Dispatch.begin(device, pso);
    d.setBuffer(in, 0);
    d.setBuffer(out, 1);
    d.setBytes(&params, 2);
    d.dispatchThreads(n, 64);
    try d.finish();

    var bad: usize = 0;
    for (out.items(f32), 0..) |o, k| {
        const want: f32 = if (k < params.count) @as(f32, @floatFromInt(k)) * params.scale else -1.0;
        if (o != want) {
            if (bad < 5) std.debug.print("    mismatch [{d}] got {d} want {d}\n", .{ k, o, want });
            bad += 1;
        }
    }
    report("scaleKernel", bad == 0, "{d} mismatches (out[5]={d} out[1020]={d})", .{ bad, out.items(f32)[5], out.items(f32)[1020] });
}

/// reverseKernel(in, out, tid_tg, tg_pos, tg_size, tidx): reverses each threadgroup of 64.
fn testReverseKernel(device: objc.Object, pso: objc.Object) !void {
    const n: u32 = 256;
    const tg: u32 = 64;
    const in = try SharedBuffer.create(device, n * 4);
    const out = try SharedBuffer.create(device, n * 4);
    for (in.items(f32), out.items(f32), 0..) |*i, *o, k| {
        i.* = @floatFromInt(k);
        o.* = -1.0;
    }

    const d = try Dispatch.begin(device, pso);
    d.setBuffer(in, 0);
    d.setBuffer(out, 1);
    d.dispatchThreadgroups(n / tg, tg);
    try d.finish();

    var bad: usize = 0;
    for (out.items(f32), 0..) |o, k| {
        const g = k / tg;
        const t = k % tg;
        const want: f32 = @floatFromInt(g * tg + (tg - 1 - t));
        if (o != want) {
            if (bad < 5) std.debug.print("    mismatch [{d}] got {d} want {d}\n", .{ k, o, want });
            bad += 1;
        }
    }
    const o = out.items(f32);
    report("reverseKernel", bad == 0, "{d} mismatches (out[0]={d} out[63]={d} out[64]={d})", .{ bad, o[0], o[63], o[64] });
}

/// sumKernel(in, out, gid, lane, sgid): lane 0 of each SIMD-group writes
/// simd_sum(in) + in[first lane + 1] to out[sgid]. One threadgroup of 64
/// threads (or one SIMD-group if the width does not divide 64).
fn testSumKernel(device: objc.Object, pso: objc.Object) !void {
    const width: u32 = @intCast(objc.msgSend(u64, pso, "threadExecutionWidth", .{}));
    const tg: u32 = if (64 % width == 0) 64 else width;
    const groups = tg / width;
    const in = try SharedBuffer.create(device, tg * 4);
    const out = try SharedBuffer.create(device, tg * 4);
    for (in.items(f32), out.items(f32), 0..) |*i, *o, k| {
        i.* = @floatFromInt(k + 1);
        o.* = -1.0;
    }

    const d = try Dispatch.begin(device, pso);
    d.setBuffer(in, 0);
    d.setBuffer(out, 1);
    d.dispatchThreadgroups(1, tg);
    try d.finish();

    var bad: usize = 0;
    for (0..groups) |g| {
        var sum: f32 = 0;
        for (in.items(f32)[g * width .. (g + 1) * width]) |v| sum += v;
        const want = sum + in.items(f32)[g * width + 1];
        const got = out.items(f32)[g];
        if (got != want) {
            std.debug.print("    mismatch simdgroup {d}: got {d} want {d}\n", .{ g, got, want });
            bad += 1;
        }
    }
    // Lanes other than 0 never write: the rest of `out` is untouched.
    for (out.items(f32)[groups..]) |o| if (o != -1.0) {
        bad += 1;
    };
    report("sumKernel", bad == 0, "{d} mismatches over {d} simdgroups of {d} (out[0]={d})", .{ bad, groups, width, out.items(f32)[0] });
}

/// countKernel(counter, in, gid): atomically counts the nonzero inputs.
fn testCountKernel(device: objc.Object, pso: objc.Object) !void {
    const n: u32 = 1000;
    const counter = try SharedBuffer.create(device, 4);
    const in = try SharedBuffer.create(device, n * 4);
    counter.items(u32)[0] = 0;
    var want: u32 = 0;
    for (in.items(u32), 0..) |*i, k| {
        i.* = if (k % 3 == 0) 1 else 0;
        want += i.*;
    }

    const d = try Dispatch.begin(device, pso);
    d.setBuffer(counter, 0);
    d.setBuffer(in, 1);
    d.dispatchThreads(n, 64);
    try d.finish();

    const got = counter.items(u32)[0];
    report("countKernel", got == want, "counter={d} want={d}", .{ got, want });
}

/// atomicOpsKernel(vals, in, gid, tidx): every atomic helper once per thread.
fn testAtomicOpsKernel(device: objc.Object, pso: objc.Object) !void {
    const n: u32 = 512;
    const vals = try SharedBuffer.create(device, 9 * 4);
    const in = try SharedBuffer.create(device, n * 4);
    const v = vals.items(u32);
    v[0] = 1000; // add
    v[1] = 0x0010_0000; // sub
    v[2] = 0; // max
    v[3] = 0xFFFF_FFFF; // min
    v[4] = 0; // or
    v[5] = 0xFFFF_FFFF; // and
    v[6] = 0x5A5A; // xor
    v[7] = 0; // exchange
    v[8] = 0; // threads counted via the threadgroup counter
    var want = [9]u32{ 1000, 0x0010_0000, 0, 0xFFFF_FFFF, 0, 0xFFFF_FFFF, 0x5A5A, 0, n };
    for (in.items(u32), 0..) |*i, k| {
        i.* = @intCast((k * 7919) % 1024 + 1);
        want[0] += i.*;
        want[1] -= i.*;
        want[2] = @max(want[2], i.*);
        want[3] = @min(want[3], i.*);
        want[4] |= i.*;
        want[5] &= i.*;
        want[6] ^= i.*;
    }

    const d = try Dispatch.begin(device, pso);
    d.setBuffer(vals, 0);
    d.setBuffer(in, 1);
    d.dispatchThreadgroups(n / 64, 64);
    try d.finish();

    var bad: usize = 0;
    const names = [_][]const u8{ "add", "sub", "max", "min", "or", "and", "xor", "exchange", "threadgroup add" };
    for (names, 0..) |name, k| {
        // exchange: the last writer wins, so any thread's gid + 1 is right.
        const ok = if (k == 7) v[k] >= 1 and v[k] <= n else v[k] == want[k];
        if (!ok) {
            std.debug.print("    mismatch {s}: got {d} want {d}\n", .{ name, v[k], want[k] });
            bad += 1;
        }
    }
    report("atomicOpsKernel", bad == 0, "{d} mismatches (add={d} sub={d} max={d} min={d} or={d} and={d} xor={d} xchg={d} tg={d})", .{ bad, v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8] });
}

/// simdOpsKernel(in, out, gid, lane, sgid, width): the SIMD-group helpers
/// on in[k] = 1 + (k % 2); see the kernel for the slot layout.
fn testSimdOpsKernel(device: objc.Object, pso: objc.Object) !void {
    const width: u32 = @intCast(objc.msgSend(u64, pso, "threadExecutionWidth", .{}));
    const tg: u32 = if (64 % width == 0) 64 else width;
    const groups = tg / width;
    const slots = shader.simd_ops_slots;
    const in = try SharedBuffer.create(device, tg * 4);
    const out = try SharedBuffer.create(device, groups * slots * 4);
    for (in.items(f32), 0..) |*i, k| i.* = @floatFromInt(1 + k % 2);
    for (out.items(f32)) |*o| o.* = -1.0;

    const d = try Dispatch.begin(device, pso);
    d.setBuffer(in, 0);
    d.setBuffer(out, 1);
    d.dispatchThreadgroups(1, tg);
    try d.finish();

    var bad: usize = 0;
    for (0..groups) |g| {
        const base = g * width;
        const fw: f32 = @floatFromInt(width);
        // gids of this simdgroup: base .. base + width - 1
        const gid_sum: f32 = @floatFromInt(width * base + (width * (width - 1)) / 2);
        const want = [slots]f32{
            fw * 1.5, // sum of alternating 1, 2
            2.0, // max
            1.0, // min
            @floatFromInt(@as(u64, 1) << @intCast(width / 2)), // product: 2^(width/2)
            fw * 1.5, // inclusive prefix sum at the last lane
            2.0, // broadcast of lane 3 (odd -> 2)
            2.0, // lane 0 shuffles lane 1
            2.0, // lane 0 shuffle_down 1 = lane 1
            1.0, // lane 1 shuffle_up 1 = lane 0
            2.0, // lane 0 shuffle_xor 1 = lane 1
            gid_sum, // simd_sum of uint gid
            2.0, // any(v > 1.5) = true, all = false
        };
        for (want, 0..) |w, j| {
            const got = out.items(f32)[g * slots + j];
            if (got != w) {
                std.debug.print("    mismatch simdgroup {d} slot {d}: got {d} want {d}\n", .{ g, j, got, w });
                bad += 1;
            }
        }
    }
    report("simdOpsKernel", bad == 0, "{d} mismatches over {d} simdgroups of {d}", .{ bad, groups, width });
}

/// tgBufKernel(in, out, tile[[threadgroup(0)]], gid uint3, tidx ushort, tg_size):
/// reverses each threadgroup of 64 through a host-sized threadgroup buffer.
fn testTgBufKernel(device: objc.Object, pso: objc.Object) !void {
    const n: u32 = 256;
    const tg: u32 = 64;
    const in = try SharedBuffer.create(device, n * 4);
    const out = try SharedBuffer.create(device, n * 4);
    for (in.items(f32), out.items(f32), 0..) |*i, *o, k| {
        i.* = @floatFromInt(k);
        o.* = -1.0;
    }

    const d = try Dispatch.begin(device, pso);
    d.setBuffer(in, 0);
    d.setBuffer(out, 1);
    d.setThreadgroupMemoryLength(tg * 4, 0);
    d.dispatchThreadgroups(n / tg, tg);
    try d.finish();

    var bad: usize = 0;
    for (out.items(f32), 0..) |o, k| {
        const g = k / tg;
        const t = k % tg;
        const want: f32 = @floatFromInt(g * tg + (tg - 1 - t));
        if (o != want) {
            if (bad < 5) std.debug.print("    mismatch [{d}] got {d} want {d}\n", .{ k, o, want });
            bad += 1;
        }
    }
    const o = out.items(f32);
    report("tgBufKernel", bad == 0, "{d} mismatches (out[0]={d} out[63]={d} out[64]={d})", .{ bad, o[0], o[63], o[64] });
}

/// reduceKernel(in, out, tidx, tg_size, tg_pos): out[g] = sum of the
/// threadgroup's 64 inputs, reduced through threadgroup memory with a
/// barrier inside a runtime-bounded loop. in[k] = k, so group g sums to
/// 64 * 64 * g + 2016.
fn testReduceKernel(device: objc.Object, pso: objc.Object) !void {
    const n: u32 = 256;
    const tg: u32 = 64;
    const groups = n / tg;
    const in = try SharedBuffer.create(device, n * 4);
    const out = try SharedBuffer.create(device, groups * 4);
    for (in.items(f32), 0..) |*i, k| i.* = @floatFromInt(k);
    for (out.items(f32)) |*o| o.* = -1.0;

    const d = try Dispatch.begin(device, pso);
    d.setBuffer(in, 0);
    d.setBuffer(out, 1);
    d.dispatchThreadgroups(groups, tg);
    try d.finish();

    var bad: usize = 0;
    for (out.items(f32), 0..) |o, g| {
        const want: f32 = @floatFromInt(tg * tg * g + (tg * (tg - 1)) / 2);
        if (o != want) {
            std.debug.print("    mismatch [{d}] got {d} want {d}\n", .{ g, o, want });
            bad += 1;
        }
    }
    const o = out.items(f32);
    report("reduceKernel", bad == 0, "{d} mismatches (out[0]={d} out[{d}]={d})", .{ bad, o[0], groups - 1, o[groups - 1] });
}

/// twoStageKernel(in, out, gid, lane, sgid, nsg, tg_pos): out[g] = sum of
/// the threadgroup's 256 inputs, a `simd_sum` per SIMD-group and a second
/// `simd_sum` of the partials by SIMD-group 0 (a SIMD-group collective
/// under `if (sgid == 0)`). in[k] = k, so group g sums to
/// 256 * 256 * g + 32640.
fn testTwoStageKernel(device: objc.Object, pso: objc.Object) !void {
    const tg: u32 = 256;
    const groups: u32 = 4;
    const n = tg * groups;
    const in = try SharedBuffer.create(device, n * 4);
    const out = try SharedBuffer.create(device, groups * 4);
    for (in.items(f32), 0..) |*i, k| i.* = @floatFromInt(k);
    for (out.items(f32)) |*o| o.* = -1.0;

    const d = try Dispatch.begin(device, pso);
    d.setBuffer(in, 0);
    d.setBuffer(out, 1);
    d.dispatchThreadgroups(groups, tg);
    try d.finish();

    var bad: usize = 0;
    for (out.items(f32), 0..) |o, g| {
        const want: f32 = @floatFromInt(tg * tg * g + (tg * (tg - 1)) / 2);
        if (o != want) {
            std.debug.print("    mismatch [{d}] got {d} want {d}\n", .{ g, o, want });
            bad += 1;
        }
    }
    const o = out.items(f32);
    report("twoStageKernel", bad == 0, "{d} mismatches over {d} groups of {d} (out[0]={d} out[{d}]={d})", .{ bad, groups, tg, o[0], groups - 1, o[groups - 1] });
}

/// vertexShaderInstanced + fragmentShaderMRT, offscreen: one point at the
/// centre of an 8x8 target with `baseInstance` 3, Uniforms {mvp identity,
/// tint (0.5, 0, 0, 0), time 4 (= point size), flags 1}, vertex position
/// (0, 0, 0.5, 1), texCoords (0.25, 0.75). A 4px point centred on pixel
/// (4, 4) covers pixels 2..5; point_coord at pixel (x, 4) is
/// ((x + 0.5 - 2) / 4, 0.625) (research/_results/r01.txt metadata_patterns
/// 11: 0.625 at the centre pixel of a 4px point). Expected:
///   pixel (4, 4): point_coord.x = 0.625 > 0.5 and flags bit 0 set -> discarded,
///                 every attachment keeps its clear value (9 / 9 / depth 1).
///   pixel (3, 4): point_coord = (0.375, 0.625), flat_id = instance_id 3 +
///                 base_vertex 0 + base_instance 3 + flags 1 = 7, front_facing
///                 = 1 for a point (r01 finding 9):
///                 color0 = (0.25, 0.75, 7, 1) + tint = (0.75, 0.75, 7, 1),
///                 color1 = half4(0.375, 0.625, 0.5, 1), depth = 0.5 * 0.5 = 0.25
///                 (r01 metadata_patterns 13: depth 0.25 read back for clip z 0.5).
// ── Texture sampling ────────────────────────────────────────────────────────
// Building a pipeline proves only that the AIR was accepted; these renders are
// what proves the sample intrinsic, the sampler word and the coordinates are
// right (DESIGN.md D7, research/_results/r03.txt open_questions 0).

const tex_side: u64 = 8;

const MTLRegion = extern struct { origin: MTLOrigin, size: MTLSize };

/// An 8x8 RGBA8 image whose every channel identifies its texel: red counts
/// columns, green counts rows, blue alternates. A sample from the wrong texel,
/// or with the axes swapped, changes the result.
fn testImage() [tex_side * tex_side * 4]u8 {
    var px: [tex_side * tex_side * 4]u8 = undefined;
    for (0..tex_side) |y| {
        for (0..tex_side) |x| {
            const i = (y * tex_side + x) * 4;
            px[i + 0] = @intCast(x * 32);
            px[i + 1] = @intCast(y * 32);
            px[i + 2] = if ((x + y) % 2 == 0) 255 else 0;
            px[i + 3] = 255;
        }
    }
    return px;
}

fn unorm(v: u8) f32 {
    return @as(f32, @floatFromInt(v)) / 255.0;
}

/// An 8x8 RGBA8Unorm texture holding `bytes`.
fn imageTexture(device: objc.Object, bytes: []const u8, usage: u64) !objc.Object {
    const td = objc.msgSend(?objc.Object, objc.class("MTLTextureDescriptor"), "texture2DDescriptorWithPixelFormat:width:height:mipmapped:", .{ MTLPixelFormatRGBA8Unorm, tex_side, tex_side, false }) orelse return error.NoTextureDescriptor;
    _ = objc.msgSend(void, td, "setUsage:", .{usage});
    const tex = objc.msgSend(?objc.Object, device, "newTextureWithDescriptor:", .{td}) orelse return error.NoTexture;
    const region = MTLRegion{
        .origin = .{ .x = 0, .y = 0, .z = 0 },
        .size = .{ .width = tex_side, .height = tex_side, .depth = 1 },
    };
    _ = objc.msgSend(void, tex, "replaceRegion:mipmapLevel:withBytes:bytesPerRow:", .{ region, @as(u64, 0), @as(*const anyopaque, @ptrCast(bytes.ptr)), tex_side * 4 });
    return tex;
}

/// Two triangles covering the target, `texCoords` running 0..1 with v growing
/// downwards. On an 8x8 target every pixel centre then lands exactly on the
/// centre of the texel with the same index, so bilinear filtering returns that
/// texel unchanged and the expected image is the source image.
fn fullscreenQuad() [6]shader.VertexIn {
    const n = @Vector(4, f32){ 0, 0, 1, 0 };
    const tl = shader.VertexIn{ .position = .{ -1, 1, 0, 1 }, .normal = n, .texCoords = .{ 0, 0 } };
    const bl = shader.VertexIn{ .position = .{ -1, -1, 0, 1 }, .normal = n, .texCoords = .{ 0, 1 } };
    const tr = shader.VertexIn{ .position = .{ 1, 1, 0, 1 }, .normal = n, .texCoords = .{ 1, 0 } };
    const br = shader.VertexIn{ .position = .{ 1, -1, 0, 1 }, .normal = n, .texCoords = .{ 1, 1 } };
    return .{ tl, bl, tr, tr, bl, br };
}

/// Samples the test image over a full-screen quad and compares every pixel
/// with the texel it must have read. `bound_sampler` binds an
/// `MTLSamplerState` for the `[[sampler(0)]]` variant instead of relying on
/// the shader's own constexpr sampler.
fn testTexturedRender(device: objc.Object, vfn: objc.Object, ffn: objc.Object, name: []const u8, bound_sampler: bool) !void {
    const pso = makeRenderPipeline(device, vfn, ffn, @Vector(4, f32), MTLPixelFormatRGBA32Float) catch |err| {
        reportRender(name, false, "newRenderPipelineState: {s}", .{errorText(last_error)});
        return err;
    };
    const image = testImage();
    const src = try imageTexture(device, &image, MTLTextureUsageShaderRead);
    const target = try renderTarget(device, MTLPixelFormatRGBA32Float, tex_side, tex_side);

    const pass = objc.msgSend(?objc.Object, objc.class("MTLRenderPassDescriptor"), "renderPassDescriptor", .{}) orelse return error.NoRenderPass;
    try setColorAttachment(pass, 0, target, .{ .red = 9, .green = 9, .blue = 9, .alpha = 9 });

    const quad = fullscreenQuad();
    const queue = objc.msgSend(?objc.Object, device, "newCommandQueue", .{}) orelse return error.NoCommandQueue;
    const command_buffer = objc.msgSend(?objc.Object, queue, "commandBuffer", .{}) orelse return error.NoCommandBuffer;
    const enc = objc.msgSend(?objc.Object, command_buffer, "renderCommandEncoderWithDescriptor:", .{pass}) orelse return error.NoEncoder;
    _ = objc.msgSend(void, enc, "setRenderPipelineState:", .{pso});
    _ = objc.msgSend(void, enc, "setVertexBytes:length:atIndex:", .{ @as(*const anyopaque, @ptrCast(&quad)), @as(u64, @sizeOf(@TypeOf(quad))), @as(u64, 0) });
    _ = objc.msgSend(void, enc, "setFragmentTexture:atIndex:", .{ src, @as(u64, 0) });
    if (bound_sampler) {
        const sd = objc.msgSend(?objc.Object, objc.class("MTLSamplerDescriptor"), "new", .{}) orelse return error.NoSamplerDescriptor;
        _ = objc.msgSend(void, sd, "setMinFilter:", .{MTLSamplerMinMagFilterNearest});
        _ = objc.msgSend(void, sd, "setMagFilter:", .{MTLSamplerMinMagFilterNearest});
        _ = objc.msgSend(void, sd, "setSAddressMode:", .{MTLSamplerAddressModeClampToEdge});
        _ = objc.msgSend(void, sd, "setTAddressMode:", .{MTLSamplerAddressModeClampToEdge});
        const ss = objc.msgSend(?objc.Object, device, "newSamplerStateWithDescriptor:", .{sd}) orelse return error.NoSamplerState;
        _ = objc.msgSend(void, enc, "setFragmentSamplerState:atIndex:", .{ ss, @as(u64, 0) });
    }
    _ = objc.msgSend(void, enc, "drawPrimitives:vertexStart:vertexCount:", .{ MTLPrimitiveTypeTriangle, @as(u64, 0), @as(u64, 6) });
    _ = objc.msgSend(void, enc, "endEncoding", .{});

    const buf = try SharedBuffer.create(device, tex_side * tex_side * 16);
    const blit = objc.msgSend(?objc.Object, command_buffer, "blitCommandEncoder", .{}) orelse return error.NoBlitEncoder;
    copyToBuffer(blit, target, buf, tex_side, tex_side, 16, MTLBlitOptionNone);
    _ = objc.msgSend(void, blit, "endEncoding", .{});
    _ = objc.msgSend(void, command_buffer, "commit", .{});
    _ = objc.msgSend(void, command_buffer, "waitUntilCompleted", .{});
    if (objc.msgSend(u64, command_buffer, "status", .{}) == MTLCommandBufferStatusError) {
        reportRender(name, false, "command buffer: {s}", .{errorText(objc.msgSend(?objc.Object, command_buffer, "error", .{}))});
        return error.CommandBufferFailed;
    }

    const px = buf.items(f32);
    var bad: usize = 0;
    for (0..tex_side) |y| {
        for (0..tex_side) |x| {
            const got = px[(y * tex_side + x) * 4 ..][0..4];
            const i = (y * tex_side + x) * 4;
            const want = [4]f32{ unorm(image[i + 0]), unorm(image[i + 1]), unorm(image[i + 2]), unorm(image[i + 3]) };
            for (got, want, 0..) |g, w, ch| if (!approx(g, w)) {
                if (bad < 8) std.debug.print("    pixel[{d},{d}].{d}: got {d} want {d}\n", .{ x, y, ch, g, w });
                bad += 1;
            };
        }
    }
    const mid = (5 * tex_side + 3) * 4;
    reportRender(name, bad == 0, "{d} mismatches over {d} pixels; pixel (3,5)=({d} {d} {d} {d})", .{
        bad, tex_side * tex_side, px[mid], px[mid + 1], px[mid + 2], px[mid + 3],
    });
}

/// `copyTextureKernel` reads every texel through the implicit read sampler and
/// writes it back with red and blue swapped.
fn testCopyTextureKernel(device: objc.Object, pso: objc.Object) !void {
    const name = "copyTextureKernel";
    const image = testImage();
    const src = try imageTexture(device, &image, MTLTextureUsageShaderRead);
    const blank: [tex_side * tex_side * 4]u8 = @splat(0);
    const dst = try imageTexture(device, &blank, MTLTextureUsageShaderWrite | MTLTextureUsageShaderRead);

    const d = try Dispatch.begin(device, pso);
    d.setTexture(src, 0);
    d.setTexture(dst, 1);
    d.dispatchThreads2D(tex_side, tex_side);
    try d.finish();

    const buf = try SharedBuffer.create(device, tex_side * tex_side * 4);
    const cb = objc.msgSend(?objc.Object, d.queue, "commandBuffer", .{}) orelse return error.NoCommandBuffer;
    const blit = objc.msgSend(?objc.Object, cb, "blitCommandEncoder", .{}) orelse return error.NoBlitEncoder;
    copyToBuffer(blit, dst, buf, tex_side, tex_side, 4, MTLBlitOptionNone);
    _ = objc.msgSend(void, blit, "endEncoding", .{});
    _ = objc.msgSend(void, cb, "commit", .{});
    _ = objc.msgSend(void, cb, "waitUntilCompleted", .{});

    var bad: usize = 0;
    for (0..tex_side * tex_side) |p| {
        const want = [4]u8{ image[p * 4 + 2], image[p * 4 + 1], image[p * 4 + 0], image[p * 4 + 3] };
        for (want, 0..) |w, ch| if (buf.bytes[p * 4 + ch] != w) {
            if (bad < 8) std.debug.print("    texel[{d}].{d}: got {d} want {d}\n", .{ p, ch, buf.bytes[p * 4 + ch], w });
            bad += 1;
        };
    }
    report(name, bad == 0, "{d} mismatches over {d} texels (texel 3 = {d} {d} {d} {d})", .{
        bad, tex_side * tex_side, buf.bytes[12], buf.bytes[13], buf.bytes[14], buf.bytes[15],
    });
}

fn testInstancedMRT(device: objc.Object, vfn: objc.Object, ffn: objc.Object) !void {
    const name = "vertexShaderInstanced + fragmentShaderMRT";
    const W: u64 = 8;
    const H: u64 = 8;
    const pso = makeRenderPipeline(device, vfn, ffn, shader.FragOut, MTLPixelFormatRGBA32Float) catch |err| {
        reportRender(name, false, "newRenderPipelineState (RGBA32Float + RGBA16Float + Depth32Float): {s}", .{errorText(last_error)});
        return err;
    };
    const c0 = try renderTarget(device, MTLPixelFormatRGBA32Float, W, H);
    const c1 = try renderTarget(device, MTLPixelFormatRGBA16Float, W, H);
    const dt = try renderTarget(device, MTLPixelFormatDepth32Float, W, H);

    const pass = objc.msgSend(?objc.Object, objc.class("MTLRenderPassDescriptor"), "renderPassDescriptor", .{}) orelse return error.NoRenderPass;
    try setColorAttachment(pass, 0, c0, .{ .red = 9, .green = 9, .blue = 9, .alpha = 9 });
    try setColorAttachment(pass, 1, c1, .{ .red = 9, .green = 9, .blue = 9, .alpha = 9 });
    const da = objc.msgSend(?objc.Object, pass, "depthAttachment", .{}) orelse return error.NoDepthAttachment;
    _ = objc.msgSend(void, da, "setTexture:", .{dt});
    _ = objc.msgSend(void, da, "setLoadAction:", .{MTLLoadActionClear});
    _ = objc.msgSend(void, da, "setStoreAction:", .{MTLStoreActionStore});
    _ = objc.msgSend(void, da, "setClearDepth:", .{@as(f64, 1.0)});

    // Depth writes are off without a depth-stencil state.
    const dsd = objc.msgSend(?objc.Object, objc.class("MTLDepthStencilDescriptor"), "new", .{}) orelse return error.NoDepthStencilDescriptor;
    _ = objc.msgSend(void, dsd, "setDepthCompareFunction:", .{MTLCompareFunctionAlways});
    _ = objc.msgSend(void, dsd, "setDepthWriteEnabled:", .{true});
    const dss = objc.msgSend(?objc.Object, device, "newDepthStencilStateWithDescriptor:", .{dsd}) orelse return error.NoDepthStencilState;

    const vertex = shader.VertexIn{
        .position = .{ 0, 0, 0.5, 1 },
        .normal = .{ 0, 0, 1, 0 },
        .texCoords = .{ 0.25, 0.75 },
    };
    const uniforms = shader.Uniforms{
        .mvp = .{ .{ 1, 0, 0, 0 }, .{ 0, 1, 0, 0 }, .{ 0, 0, 1, 0 }, .{ 0, 0, 0, 1 } },
        .tint = .{ 0.5, 0, 0, 0 },
        .time = 4.0,
        .flags = 1,
    };

    const queue = objc.msgSend(?objc.Object, device, "newCommandQueue", .{}) orelse return error.NoCommandQueue;
    const command_buffer = objc.msgSend(?objc.Object, queue, "commandBuffer", .{}) orelse return error.NoCommandBuffer;
    const enc = objc.msgSend(?objc.Object, command_buffer, "renderCommandEncoderWithDescriptor:", .{pass}) orelse return error.NoEncoder;
    _ = objc.msgSend(void, enc, "setRenderPipelineState:", .{pso});
    _ = objc.msgSend(void, enc, "setDepthStencilState:", .{dss});
    _ = objc.msgSend(void, enc, "setVertexBytes:length:atIndex:", .{ @as(*const anyopaque, @ptrCast(&vertex)), @as(u64, @sizeOf(shader.VertexIn)), @as(u64, 0) });
    _ = objc.msgSend(void, enc, "setVertexBytes:length:atIndex:", .{ @as(*const anyopaque, @ptrCast(&uniforms)), @as(u64, @sizeOf(shader.Uniforms)), @as(u64, 1) });
    _ = objc.msgSend(void, enc, "setFragmentBytes:length:atIndex:", .{ @as(*const anyopaque, @ptrCast(&uniforms)), @as(u64, @sizeOf(shader.Uniforms)), @as(u64, 0) });
    _ = objc.msgSend(void, enc, "drawPrimitives:vertexStart:vertexCount:instanceCount:baseInstance:", .{ MTLPrimitiveTypePoint, @as(u64, 0), @as(u64, 1), @as(u64, 1), @as(u64, 3) });
    _ = objc.msgSend(void, enc, "endEncoding", .{});

    const b0 = try SharedBuffer.create(device, W * H * 16);
    const b1 = try SharedBuffer.create(device, W * H * 8);
    const bd = try SharedBuffer.create(device, W * H * 4);
    const blit = objc.msgSend(?objc.Object, command_buffer, "blitCommandEncoder", .{}) orelse return error.NoBlitEncoder;
    copyToBuffer(blit, c0, b0, W, H, 16, MTLBlitOptionNone);
    copyToBuffer(blit, c1, b1, W, H, 8, MTLBlitOptionNone);
    copyToBuffer(blit, dt, bd, W, H, 4, MTLBlitOptionDepthFromDepthStencil);
    _ = objc.msgSend(void, blit, "endEncoding", .{});
    _ = objc.msgSend(void, command_buffer, "commit", .{});
    _ = objc.msgSend(void, command_buffer, "waitUntilCompleted", .{});
    if (objc.msgSend(u64, command_buffer, "status", .{}) == MTLCommandBufferStatusError) {
        reportRender(name, false, "command buffer: {s}", .{errorText(objc.msgSend(?objc.Object, command_buffer, "error", .{}))});
        return error.CommandBufferFailed;
    }

    const px0 = b0.items(f32);
    const px1 = b1.items(f16);
    const pxd = bd.items(f32);
    const kept = 4 * W + 3; // pixel (3, 4)
    const gone = 4 * W + 4; // pixel (4, 4), discarded
    const k0 = px0[kept * 4 ..][0..4];
    const k1 = px1[kept * 4 ..][0..4];
    const g0 = px0[gone * 4 ..][0..4];
    const g1 = px1[gone * 4 ..][0..4];
    const want0 = [4]f32{ 0.75, 0.75, 7, 1 };
    const want1 = [4]f16{ 0.375, 0.625, 0.5, 1 };
    var bad: usize = 0;
    for (k0, want0, 0..) |got, want, c| if (!approx(got, want)) {
        std.debug.print("    color0[3,4].{d}: got {d} want {d}\n", .{ c, got, want });
        bad += 1;
    };
    for (k1, want1, 0..) |got, want, c| if (got != want) {
        std.debug.print("    color1[3,4].{d}: got {d} want {d}\n", .{ c, got, want });
        bad += 1;
    };
    if (!approx(pxd[kept], 0.25)) {
        std.debug.print("    depth[3,4]: got {d} want 0.25\n", .{pxd[kept]});
        bad += 1;
    }
    for (g0, 0..) |got, c| if (got != 9) {
        std.debug.print("    color0[4,4].{d}: got {d} want clear 9 (discard)\n", .{ c, got });
        bad += 1;
    };
    for (g1, 0..) |got, c| if (got != 9) {
        std.debug.print("    color1[4,4].{d}: got {d} want clear 9 (discard)\n", .{ c, got });
        bad += 1;
    };
    if (pxd[gone] != 1.0) {
        std.debug.print("    depth[4,4]: got {d} want clear 1 (discard)\n", .{pxd[gone]});
        bad += 1;
    }
    reportRender(name, bad == 0, "{d} mismatches; pixel (3,4): color0=({d} {d} {d} {d}) color1=({d} {d} {d} {d}) depth={d}; pixel (4,4) discarded: color0.r={d} depth={d}", .{
        bad,   k0[0],     k0[1], k0[2], k0[3], k1[0], k1[1], k1[2], k1[3], pxd[kept],
        g0[0], pxd[gone],
    });
}

/// The whole NSError, `-description`: domain, code and every userInfo entry
/// (the localized summary is one of them). A bare "Compilation failed" says
/// nothing about where or why; the domain names the component that refused
/// (an `AGXMetal...` GPU compiler, for instance) and userInfo may hold more.
fn errorText(err: ?objc.Object) [*:0]const u8 {
    const e = err orelse return "(no NSError)";
    const desc = objc.msgSend(?objc.Object, e, "description", .{}) orelse return "(no description)";
    return objc.msgSend([*:0]const u8, desc, "UTF8String", .{});
}
