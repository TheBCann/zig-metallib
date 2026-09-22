//! Shaders written in Zig.
//!
//! Build pipeline (see build.zig):
//!   this file --zig (nvptx64 target)--> LLVM IR --air-splice--> AIR IR --xcrun metal--> default.metallib
//!
//! Rules for shader code here:
//!   * No `std` in shader bodies. The file is compiled for a GPU target with
//!     no runtime. The one exception is the `panic` declaration below.
//!   * Buffer pointers use `addrspace(.global)`, which lowers to LLVM
//!     addrspace(1) == Metal's `device` address space. Constant buffers are
//!     `*addrspace(.param) const T` (nvptx 4, renumbered to Metal's 2), and
//!     static threadgroup memory is `var x: [N]T addrspace(.shared)`.
//!   * Kernels are `callconv(.kernel)`, take their built-in ids as explicit
//!     parameters (described by `.builtin` manifest args) and call the AIR
//!     intrinsics through `gpu.zig`.
//!   * Shader structs are plain structs (not `extern`) so they may hold
//!     `@Vector` fields. The host-side vertex layout must match `@offsetOf`.
//!   * Every entry point is listed in `functions` below; that table is the
//!     single source of truth for the Metal binding metadata.
//!
//! Panic handler: this file is the root of the GPU compilation, so its
//! `panic` decides what a safety check compiles to. `std.debug.no_panic`
//! turns every panic into `llvm.trap` + `unreachable`, which Metal accepts;
//! the default handler would pull in formatting, `Io.Writer` vtables,
//! `llvm.returnaddress` and string tables that cannot become AIR. With it,
//! ReleaseSafe shader builds pass pipeline creation (research r11 finding
//! 20: v_safe_baseline / v_safe_i16 at ReleaseSafe). The build still uses
//! ReleaseFast (build.zig); this only keeps the other modes viable. On the
//! host the file is imported as a module, where `panic` is ignored.

const std = @import("std");
const builtin = @import("builtin");
pub const air = @import("air.zig");
pub const gpu = @import("gpu.zig");

pub const panic = std.debug.no_panic;

// --- Types shared with the host ---------------------------------------------

pub const VertexIn = struct {
    position: @Vector(4, f32),
    normal: @Vector(4, f32),
    texCoords: @Vector(2, f32),
};

pub const VertexOut = struct {
    position: @Vector(4, f32),
    normal: @Vector(3, f32),
    texCoords: @Vector(2, f32),
};

/// Constant buffer of `vertexShaderInstanced` (`[[buffer(1)]]`) and
/// `fragmentShaderMRT` (`[[buffer(0)]]`), bound with `setVertexBytes` /
/// `setFragmentBytes`. A plain struct because extern structs may not hold
/// vectors (research/_results/r04.json corrections); Zig's auto layout keeps
/// these fields in declaration order (alignments 16, 16, 4, 4), which the
/// comptime checks below pin down so the host can fill it field by field
/// and the `air.struct_type_info` offsets (0, 64, 80, 84; size 96) match
/// Apple's for the same MSL struct (research/_results/r01.txt
/// metadata_patterns 5).
pub const Uniforms = struct {
    /// `float4x4`: four `float4` columns, column-major.
    mvp: [4]@Vector(4, f32),
    tint: @Vector(4, f32),
    time: f32,
    flags: u32,
};

comptime {
    if (@offsetOf(Uniforms, "mvp") != 0 or @offsetOf(Uniforms, "tint") != 64 or
        @offsetOf(Uniforms, "time") != 80 or @offsetOf(Uniforms, "flags") != 84 or @sizeOf(Uniforms) != 96)
    {
        @compileError("Uniforms layout differs from the MSL struct {float4x4, float4, float, uint}");
    }
}

/// Output of `vertexShaderInstanced`: `point_size` is `[[point_size]]`
/// (consumed by the rasterizer, never a fragment parameter), `flat_id` an
/// integer varying (`[[flat]]`). Zig lays the fields out as position,
/// texCoords, point_size, flat_id; the splice reads them by `@offsetOf`.
pub const VertexOutInstanced = struct {
    position: @Vector(4, f32),
    point_size: f32,
    flat_id: u32,
    texCoords: @Vector(2, f32),
};

/// Output of `fragmentShaderMRT`: `[[color(0)]]` float4, `[[color(1)]]`
/// half4, `[[depth(any)]]`.
pub const FragOut = struct {
    color0: @Vector(4, f32),
    color1: @Vector(4, f16),
    depth: f32,
};

/// Constant buffer of `scaleKernel`, bound with `setBytes:length:atIndex:`.
/// `extern` so the host and the GPU agree on the layout.
pub const Params = extern struct { count: u32, scale: f32 };

/// Static threadgroup memory of `reverseKernel`: one slot per thread of a
/// 64-wide threadgroup (Metal reports it as staticThreadgroupMemoryLength=256).
var scratch: [64]f32 addrspace(.shared) = undefined;

/// Threadgroup counter of `atomicOpsKernel`.
var tg_counter: u32 addrspace(.shared) = undefined;

/// Reduction scratch of `reduceKernel` (one slot per thread of a 64-wide
/// threadgroup).
var red: [64]f32 addrspace(.shared) = undefined;

// --- Entry points -----------------------------------------------------------

pub fn vertexShader(
    vertexID: u32,
    vertices: [*]addrspace(.global) const VertexIn,
) VertexOut {
    const v = vertices[vertexID];
    return .{
        .position = v.position,
        .normal = @shuffle(f32, v.normal, undefined, [3]i32{ 0, 1, 2 }),
        .texCoords = v.texCoords,
    };
}

/// Samples `colorTexture` (`[[texture(0)]]`) at the interpolated texCoords
/// with a constexpr bilinear, clamp-to-edge sampler; the window shows the
/// host's checkerboard on the triangle. The sampler is a `[2 x i64]`
/// constant global the assembler relocates to constant memory and lists in
/// `!air.sampler_states` (gpu.constexprSampler). `zig build check` renders
/// this pair offscreen and compares pixels with a host bilinear filter.
pub fn fragmentShader(
    position: @Vector(4, f32),
    normal: @Vector(3, f32),
    texCoords: @Vector(2, f32),
    colorTexture: *addrspace(.global) const gpu.Texture2D(.sample),
) @Vector(4, f32) {
    _ = position;
    _ = normal;
    const s = gpu.constexprSampler(.{ .mag_filter = .linear, .min_filter = .linear });
    return gpu.sample2D(colorTexture, s, texCoords);
}

/// Same as `fragmentShader` with the sampler bound by the host as
/// `[[sampler(0)]]` (`setFragmentSamplerState:atIndex:`; the check tool
/// binds a nearest-filter MTLSamplerState and expects exact texel colours).
pub fn fragmentShaderBoundSampler(
    position: @Vector(4, f32),
    normal: @Vector(3, f32),
    texCoords: @Vector(2, f32),
    colorTexture: *addrspace(.global) const gpu.Texture2D(.sample),
    s: *const gpu.Sampler,
) @Vector(4, f32) {
    _ = position;
    _ = normal;
    return gpu.sample2D(colorTexture, s, texCoords);
}

/// Every vertex-stage builtin plus a constant `Uniforms` buffer with a
/// `float4x4` field. position = mvp * v.position; point_size = time;
/// flat_id = instance_id + base_vertex + base_instance + flags (Metal's
/// instance_id already includes base_instance). The check tool draws one
/// point with baseInstance 3 and reads flat_id 7 back through the flat
/// varying (research/_results/r01.txt ir_patterns 1: the same shape
/// rendered flatId=7 and a 4px point).
pub fn vertexShaderInstanced(
    vertexID: u32,
    instanceID: u32,
    baseVertex: u32,
    baseInstance: u32,
    vertices: [*]addrspace(.global) const VertexIn,
    uniforms: *addrspace(.param) const Uniforms,
) VertexOutInstanced {
    const v = vertices[vertexID];
    const m = uniforms.mvp;
    const p = v.position;
    const position = m[0] * @as(@Vector(4, f32), @splat(p[0])) +
        m[1] * @as(@Vector(4, f32), @splat(p[1])) +
        m[2] * @as(@Vector(4, f32), @splat(p[2])) +
        m[3] * @as(@Vector(4, f32), @splat(p[3]));
    return .{
        .position = position,
        .point_size = uniforms.time,
        .flat_id = instanceID + baseVertex + baseInstance + uniforms.flags,
        .texCoords = v.texCoords,
    };
}

/// Multiple render targets + depth from a struct return, `[[front_facing]]`
/// and `[[point_coord]]` inputs, the flat `flat_id` varying, a constant
/// buffer in the fragment stage and a branched `discard_fragment`. The
/// parameters follow the `VertexOutInstanced` fields minus `point_size`
/// (which Metal never passes to the fragment stage), then the builtins,
/// then the buffer, exactly like the manifest entry.
///   color0 = (texCoords, flat_id, front_facing) + tint
///   color1 = half4(point_coord, tint.x, 1)
///   depth  = position.z * 0.5
/// and the fragment is discarded when flags bit 0 is set and point_coord.x > 0.5.
pub fn fragmentShaderMRT(
    position: @Vector(4, f32),
    flat_id: u32,
    texCoords: @Vector(2, f32),
    front_facing: bool,
    point_coord: @Vector(2, f32),
    uniforms: *addrspace(.param) const Uniforms,
) FragOut {
    if (uniforms.flags & 1 != 0 and point_coord[0] > 0.5) gpu.discardFragment();
    const id: f32 = @floatFromInt(flat_id);
    const ff: f32 = if (front_facing) 1.0 else 0.0;
    const base = @Vector(4, f32){ texCoords[0], texCoords[1], id, ff };
    const c1 = @Vector(4, f32){ point_coord[0], point_coord[1], uniforms.tint[0], 1.0 };
    return .{
        .color0 = base + uniforms.tint,
        .color1 = @floatCast(c1),
        .depth = position[2] * 0.5,
    };
}

// --- Compute kernels ----------------------------------------------------------

/// out[gid] = in[gid] * params.scale for gid < params.count.
pub fn scaleKernel(
    in: [*]addrspace(.global) const f32,
    out: [*]addrspace(.global) f32,
    params: *addrspace(.param) const Params,
    gid: u32,
) callconv(.kernel) void {
    if (gid >= params.count) return;
    out[gid] = in[gid] * params.scale;
}

/// Reverses every 64-thread group of `in` into `out` through threadgroup
/// memory: each thread stores its element, the group synchronises, each
/// thread reads the mirrored slot.
pub fn reverseKernel(
    in: [*]addrspace(.global) const f32,
    out: [*]addrspace(.global) f32,
    tid_tg: u32,
    tg_pos: u32,
    tg_size: u32,
    tidx: u32,
) callconv(.kernel) void {
    scratch[tidx] = in[tg_pos * tg_size + tid_tg];
    gpu.threadgroupBarrier(.threadgroup_only);
    out[tg_pos * tg_size + tid_tg] = scratch[tg_size - 1 - tidx];
}

/// Per SIMD-group: out[sgid] = simd_sum(in) + in[first lane + 1], written by
/// lane 0 of each SIMD-group.
pub fn sumKernel(
    in: [*]addrspace(.global) const f32,
    out: [*]addrspace(.global) f32,
    gid: u32,
    lane: u32,
    sgid: u32,
) callconv(.kernel) void {
    const v = in[gid];
    const s = gpu.simdSum(v);
    const d = gpu.simdShuffleDown(v, 1);
    if (lane == 0) out[sgid] = s + d;
}

/// Counts the nonzero inputs with a device atomic.
pub fn countKernel(
    counter: *addrspace(.global) u32,
    in: [*]addrspace(.global) const u32,
    gid: u32,
) callconv(.kernel) void {
    if (in[gid] != 0) _ = gpu.atomicAdd(counter, 1);
}

/// Exercises every device atomic helper plus the threadgroup atomic, fence
/// and simdgroup barrier, so their AIR intrinsic names go through Metal's
/// runtime compiler. `vals` slots: 0 add, 1 sub, 2 max, 3 min, 4 or, 5 and,
/// 6 xor, 7 exchange, 8 threads counted through the threadgroup counter.
pub fn atomicOpsKernel(
    vals: [*]addrspace(.global) u32,
    in: [*]addrspace(.global) const u32,
    gid: u32,
    tidx: u32,
) callconv(.kernel) void {
    const v = in[gid];
    _ = gpu.atomicAdd(&vals[0], v);
    _ = gpu.atomicSub(&vals[1], v);
    _ = gpu.atomicMax(&vals[2], v);
    _ = gpu.atomicMin(&vals[3], v);
    _ = gpu.atomicOr(&vals[4], v);
    _ = gpu.atomicAnd(&vals[5], v);
    _ = gpu.atomicXor(&vals[6], v);
    _ = gpu.atomicExchange(&vals[7], gid + 1);
    gpu.atomicFence(.device_only);

    // Every thread zeroes the counter (same value, no branch): a branch on
    // `tidx` here would let LLVM jump-thread it through the barriers below
    // and duplicate them into divergent blocks, see gpu.threadgroupBarrier;
    // the assembler's uniformity analysis refuses that shape.
    tg_counter = 0;
    gpu.threadgroupBarrier(.threadgroup_only);
    _ = gpu.atomicAddThreadgroup(&tg_counter, 1);
    gpu.threadgroupBarrier(.threadgroup_only);
    gpu.simdgroupBarrier(.threadgroup_only);
    if (tidx == 0) _ = gpu.atomicAdd(&vals[8], tg_counter);
}

/// Number of `out` slots `simdOpsKernel` writes per SIMD-group.
pub const simd_ops_slots = 12;

/// Exercises the remaining SIMD-group helpers; each SIMD-group writes
/// `simd_ops_slots` results to `out[sgid * simd_ops_slots ..]`:
/// 0 sum, 1 max, 2 min, 3 product, 4 prefix inclusive sum (last lane),
/// 5 broadcast(lane 3), 6 shuffle(lane 1) from lane 0, 7 shuffle_down(1)
/// from lane 0, 8 shuffle_up(1) from lane 1, 9 shuffle_xor(1) from lane 0,
/// 10 simd_sum of the uint gid, 11 any(v > 1.5) * 2 + all(v > 1.5).
pub fn simdOpsKernel(
    in: [*]addrspace(.global) const f32,
    out: [*]addrspace(.global) f32,
    gid: u32,
    lane: u32,
    sgid: u32,
    width: u32,
) callconv(.kernel) void {
    const v = in[gid];
    const base = sgid * simd_ops_slots;
    const sum = gpu.simdSum(v);
    const max = gpu.simdMax(v);
    const min = gpu.simdMin(v);
    const product = gpu.simdProduct(v);
    const prefix = gpu.simdPrefixInclusiveSum(v);
    const bcast = gpu.simdBroadcast(v, 3);
    const shuf = gpu.simdShuffle(v, @intCast(lane ^ 1));
    const down = gpu.simdShuffleDown(v, 1);
    const up = gpu.simdShuffleUp(v, 1);
    const xor = gpu.simdShuffleXor(v, 1);
    const usum = gpu.simdSumU32(gid);
    const any = gpu.simdAny(v > 1.5);
    const all = gpu.simdAll(v > 1.5);
    if (lane == 0) {
        out[base + 0] = sum;
        out[base + 1] = max;
        out[base + 2] = min;
        out[base + 3] = product;
        out[base + 5] = bcast;
        out[base + 6] = shuf;
        out[base + 7] = down;
        out[base + 9] = xor;
        out[base + 10] = @floatFromInt(usum);
        out[base + 11] = @as(f32, if (any) 2.0 else 0.0) + @as(f32, if (all) 1.0 else 0.0);
    }
    if (lane == 1) out[base + 8] = up;
    if (lane == width - 1) out[base + 4] = prefix;
}

/// Same reversal as `reverseKernel`, through a threadgroup BUFFER argument
/// (`[[threadgroup(0)]]`, sized by the host with
/// `setThreadgroupMemoryLength:atIndex:`) instead of static threadgroup
/// memory, with the remaining builtin shapes: a `uint3` grid position, a
/// `ushort` thread index, and a threadgroup-scope fence. Metal's runtime
/// compiler and a verified dispatch cover all four (research r06 left
/// threadgroup buffer arguments and `<3 x i32>` parameters open).
pub fn tgBufKernel(
    in: [*]addrspace(.global) const f32,
    out: [*]addrspace(.global) f32,
    tile: [*]addrspace(.shared) f32,
    gid: @Vector(3, u32),
    tidx: u16,
    tg_size: u32,
) callconv(.kernel) void {
    tile[tidx] = in[gid[0]];
    gpu.threadgroupBarrier(.threadgroup_only);
    gpu.atomicFence(.threadgroup_only);
    out[gid[0]] = tile[tg_size - 1 - tidx];
}

/// Tree reduction of each threadgroup into `out[tg_pos]`: the canonical
/// threadgroup-memory pattern with a barrier inside a loop whose bound is a
/// runtime (but uniform) value, `tg_size / 2`. LLVM guards and rotates the
/// loop; the assembler's uniformity analysis sees that every branch the
/// barrier depends on is over `tg_size`, and accepts it. (`if (tidx < s)`
/// is a per-thread branch, but both arms rejoin before the barrier.)
pub fn reduceKernel(
    in: [*]addrspace(.global) const f32,
    out: [*]addrspace(.global) f32,
    tidx: u32,
    tg_size: u32,
    tg_pos: u32,
) callconv(.kernel) void {
    red[tidx] = in[tg_pos * tg_size + tidx];
    gpu.threadgroupBarrier(.threadgroup_only);
    var s: u32 = tg_size / 2;
    while (s > 0) : (s >>= 1) {
        if (tidx < s) red[tidx] += red[tidx + s];
        gpu.threadgroupBarrier(.threadgroup_only);
    }
    if (tidx == 0) out[tg_pos] = red[0];
}

/// Partial sums of `twoStageKernel`, one slot per SIMD-group of a
/// threadgroup (32 covers 1024 threads at a SIMD width of 32).
var partial: [32]f32 addrspace(.shared) = undefined;

/// Two-stage reduction of each threadgroup into `out[tg_pos]`: every
/// SIMD-group sums its lanes with `simd_sum`, lane 0 parks the partial in
/// threadgroup memory, and after a barrier SIMD-group 0 alone sums the
/// partials with a second `simd_sum`. That second call sits under
/// `if (sgid == 0)`, a branch on `simdgroup_index_in_threadgroup`: the
/// block runs for whole SIMD-groups, which is all a SIMD-group collective
/// needs (MSL defines it over the active lanes), so the assembler's
/// uniformity analysis accepts it at its `.simdgroup` level; a
/// `threadgroupBarrier` in the same block would be refused
/// (review2/v1_nocheck/check.log: 4 groups of 256, 0 mismatches).
pub fn twoStageKernel(
    in: [*]addrspace(.global) const f32,
    out: [*]addrspace(.global) f32,
    gid: u32,
    lane: u32,
    sgid: u32,
    nsg: u32,
    tg_pos: u32,
) callconv(.kernel) void {
    const s = gpu.simdSum(in[gid]);
    if (lane == 0) partial[sgid] = s;
    gpu.threadgroupBarrier(.threadgroup_only);
    if (sgid == 0) {
        const p: f32 = if (lane < nsg) partial[lane] else 0.0;
        const t = gpu.simdSum(p);
        if (lane == 0) out[tg_pos] = t;
    }
}

/// dst[gid] = src[gid] with the red and blue channels swapped: a `read`
/// texture (`[[texture(0)]]`) and a `write` texture (`[[texture(1)]]`) in
/// a kernel, `air.read_texture_2d` through the implicit read sampler and
/// `air.write_texture_2d` (research/texture-sampling/k01_write.ll,
/// h05_kernel_write.ll only proved assembly; the check tool dispatches
/// this over an 8x8 checkerboard and reads the result back).
pub fn copyTextureKernel(
    src: *addrspace(.global) const gpu.Texture2D(.read),
    dst: *addrspace(.global) gpu.Texture2D(.write),
    gid: @Vector(2, u32),
) callconv(.kernel) void {
    const c = gpu.read2D(src, gid, 0);
    gpu.write2D(dst, gid, @shuffle(f32, c, undefined, [4]i32{ 2, 1, 0, 3 }));
}

// --- Metal binding manifest ---------------------------------------------------

pub const functions = [_]air.Function{
    .{
        .name = "vertexShader",
        .stage = .vertex,
        .ret = VertexOut,
        .args = &.{
            .{ .vertex_id = "vertexID" },
            .{ .buffer = .{ .index = 0, .T = VertexIn, .name = "vertices" } },
        },
    },
    .{
        .name = "fragmentShader",
        .stage = .fragment,
        .ret = @Vector(4, f32),
        .args = &.{
            .{ .stage_in = VertexOut },
            .{ .texture = .{ .index = 0, .name = "colorTexture", .T = gpu.Texture2D(.sample) } },
        },
    },
    .{
        .name = "fragmentShaderBoundSampler",
        .stage = .fragment,
        .ret = @Vector(4, f32),
        .args = &.{
            .{ .stage_in = VertexOut },
            .{ .texture = .{ .index = 0, .name = "colorTexture", .T = gpu.Texture2D(.sample) } },
            .{ .sampler = .{ .index = 0, .name = "s" } },
        },
    },
    .{
        .name = "vertexShaderInstanced",
        .stage = .vertex,
        .ret = VertexOutInstanced,
        .args = &.{
            .{ .vertex_id = "vertexID" },
            .{ .builtin = .{ .kind = .instance_id, .T = u32, .name = "instanceID" } },
            .{ .builtin = .{ .kind = .base_vertex, .T = u32, .name = "baseVertex" } },
            .{ .builtin = .{ .kind = .base_instance, .T = u32, .name = "baseInstance" } },
            .{ .buffer = .{ .index = 0, .T = VertexIn, .name = "vertices" } },
            .{ .buffer = .{ .index = 1, .T = Uniforms, .name = "uniforms", .space = .constant } },
        },
    },
    .{
        .name = "fragmentShaderMRT",
        .stage = .fragment,
        .ret = FragOut,
        .args = &.{
            .{ .stage_in = VertexOutInstanced },
            .{ .builtin = .{ .kind = .front_facing, .T = bool, .name = "frontFacing" } },
            .{ .builtin = .{ .kind = .point_coord, .T = @Vector(2, f32), .name = "pointCoord" } },
            .{ .buffer = .{ .index = 0, .T = Uniforms, .name = "uniforms", .space = .constant } },
        },
    },
    .{
        .name = "scaleKernel",
        .stage = .kernel,
        .ret = void,
        .args = &.{
            .{ .buffer = .{ .index = 0, .T = f32, .name = "in" } },
            .{ .buffer = .{ .index = 1, .T = f32, .name = "out", .access = .read_write } },
            .{ .buffer = .{ .index = 2, .T = Params, .name = "params", .space = .constant } },
            .{ .builtin = .{ .kind = .thread_position_in_grid, .T = u32, .name = "gid" } },
        },
    },
    .{
        .name = "reverseKernel",
        .stage = .kernel,
        .ret = void,
        .args = &.{
            .{ .buffer = .{ .index = 0, .T = f32, .name = "in" } },
            .{ .buffer = .{ .index = 1, .T = f32, .name = "out", .access = .read_write } },
            .{ .builtin = .{ .kind = .thread_position_in_threadgroup, .T = u32, .name = "tid_tg" } },
            .{ .builtin = .{ .kind = .threadgroup_position_in_grid, .T = u32, .name = "tg_pos" } },
            .{ .builtin = .{ .kind = .threads_per_threadgroup, .T = u32, .name = "tg_size" } },
            .{ .builtin = .{ .kind = .thread_index_in_threadgroup, .T = u32, .name = "tidx" } },
        },
    },
    .{
        .name = "sumKernel",
        .stage = .kernel,
        .ret = void,
        .args = &.{
            .{ .buffer = .{ .index = 0, .T = f32, .name = "in" } },
            .{ .buffer = .{ .index = 1, .T = f32, .name = "out", .access = .read_write } },
            .{ .builtin = .{ .kind = .thread_position_in_grid, .T = u32, .name = "gid" } },
            .{ .builtin = .{ .kind = .thread_index_in_simdgroup, .T = u32, .name = "lane" } },
            .{ .builtin = .{ .kind = .simdgroup_index_in_threadgroup, .T = u32, .name = "sgid" } },
        },
    },
    .{
        .name = "countKernel",
        .stage = .kernel,
        .ret = void,
        .max_total_threads_per_threadgroup = 128,
        .args = &.{
            .{ .buffer = .{ .index = 0, .T = u32, .name = "counter", .access = .read_write } },
            .{ .buffer = .{ .index = 1, .T = u32, .name = "in" } },
            .{ .builtin = .{ .kind = .thread_position_in_grid, .T = u32, .name = "gid" } },
        },
    },
    .{
        .name = "atomicOpsKernel",
        .stage = .kernel,
        .ret = void,
        .args = &.{
            .{ .buffer = .{ .index = 0, .T = u32, .name = "vals", .access = .read_write } },
            .{ .buffer = .{ .index = 1, .T = u32, .name = "in" } },
            .{ .builtin = .{ .kind = .thread_position_in_grid, .T = u32, .name = "gid" } },
            .{ .builtin = .{ .kind = .thread_index_in_threadgroup, .T = u32, .name = "tidx" } },
        },
    },
    .{
        .name = "simdOpsKernel",
        .stage = .kernel,
        .ret = void,
        .args = &.{
            .{ .buffer = .{ .index = 0, .T = f32, .name = "in" } },
            .{ .buffer = .{ .index = 1, .T = f32, .name = "out", .access = .read_write } },
            .{ .builtin = .{ .kind = .thread_position_in_grid, .T = u32, .name = "gid" } },
            .{ .builtin = .{ .kind = .thread_index_in_simdgroup, .T = u32, .name = "lane" } },
            .{ .builtin = .{ .kind = .simdgroup_index_in_threadgroup, .T = u32, .name = "sgid" } },
            .{ .builtin = .{ .kind = .threads_per_simdgroup, .T = u32, .name = "width" } },
        },
    },
    .{
        .name = "tgBufKernel",
        .stage = .kernel,
        .ret = void,
        .args = &.{
            .{ .buffer = .{ .index = 0, .T = f32, .name = "in" } },
            .{ .buffer = .{ .index = 1, .T = f32, .name = "out", .access = .read_write } },
            .{ .buffer = .{ .index = 0, .T = f32, .name = "tile", .space = .threadgroup, .access = .read_write } },
            .{ .builtin = .{ .kind = .thread_position_in_grid, .T = @Vector(3, u32), .name = "gid" } },
            .{ .builtin = .{ .kind = .thread_index_in_threadgroup, .T = u16, .name = "tidx" } },
            .{ .builtin = .{ .kind = .threads_per_threadgroup, .T = u32, .name = "tg_size" } },
        },
    },
    .{
        .name = "reduceKernel",
        .stage = .kernel,
        .ret = void,
        .args = &.{
            .{ .buffer = .{ .index = 0, .T = f32, .name = "in" } },
            .{ .buffer = .{ .index = 1, .T = f32, .name = "out", .access = .read_write } },
            .{ .builtin = .{ .kind = .thread_index_in_threadgroup, .T = u32, .name = "tidx" } },
            .{ .builtin = .{ .kind = .threads_per_threadgroup, .T = u32, .name = "tg_size" } },
            .{ .builtin = .{ .kind = .threadgroup_position_in_grid, .T = u32, .name = "tg_pos" } },
        },
    },
    .{
        .name = "twoStageKernel",
        .stage = .kernel,
        .ret = void,
        .args = &.{
            .{ .buffer = .{ .index = 0, .T = f32, .name = "in" } },
            .{ .buffer = .{ .index = 1, .T = f32, .name = "out", .access = .read_write } },
            .{ .builtin = .{ .kind = .thread_position_in_grid, .T = u32, .name = "gid" } },
            .{ .builtin = .{ .kind = .thread_index_in_simdgroup, .T = u32, .name = "lane" } },
            .{ .builtin = .{ .kind = .simdgroup_index_in_threadgroup, .T = u32, .name = "sgid" } },
            .{ .builtin = .{ .kind = .simdgroups_per_threadgroup, .T = u32, .name = "nsg" } },
            .{ .builtin = .{ .kind = .threadgroup_position_in_grid, .T = u32, .name = "tg_pos" } },
        },
    },
    .{
        .name = "copyTextureKernel",
        .stage = .kernel,
        .ret = void,
        .args = &.{
            .{ .texture = .{ .index = 0, .name = "src", .T = gpu.Texture2D(.read) } },
            .{ .texture = .{ .index = 1, .name = "dst", .T = gpu.Texture2D(.write) } },
            .{ .builtin = .{ .kind = .thread_position_in_grid, .T = @Vector(2, u32), .name = "gid" } },
        },
    },
};

// --- Keep-alive --------------------------------------------------------------
// Nothing in this module calls the entry points, so without a reference the
// optimizer would drop them from the IR. These exports only exist on the GPU
// build; air-splice deletes the `__keep_*` helpers again. Kernels are
// exported directly under their own names (`define ptx_kernel void @name`):
// `export fn` would make the host build analyse their GPU pointer types, the
// GPU-gated `@export` does not.

comptime {
    if (builtin.target.cpu.arch == .nvptx64) {
        @export(&keepVertex, .{ .name = "__keep_vertexShader" });
        @export(&keepFragment, .{ .name = "__keep_fragmentShader" });
        @export(&keepFragmentBoundSampler, .{ .name = "__keep_fragmentShaderBoundSampler" });
        @export(&keepVertexInstanced, .{ .name = "__keep_vertexShaderInstanced" });
        @export(&keepFragmentMRT, .{ .name = "__keep_fragmentShaderMRT" });
        @export(&scaleKernel, .{ .name = "scaleKernel" });
        @export(&reverseKernel, .{ .name = "reverseKernel" });
        @export(&sumKernel, .{ .name = "sumKernel" });
        @export(&countKernel, .{ .name = "countKernel" });
        @export(&atomicOpsKernel, .{ .name = "atomicOpsKernel" });
        @export(&simdOpsKernel, .{ .name = "simdOpsKernel" });
        @export(&tgBufKernel, .{ .name = "tgBufKernel" });
        @export(&reduceKernel, .{ .name = "reduceKernel" });
        @export(&twoStageKernel, .{ .name = "twoStageKernel" });
        @export(&copyTextureKernel, .{ .name = "copyTextureKernel" });
    }
}

fn keepVertex() callconv(.nvptx_device) *const anyopaque {
    return @ptrCast(&vertexShader);
}

fn keepFragment() callconv(.nvptx_device) *const anyopaque {
    return @ptrCast(&fragmentShader);
}

fn keepFragmentBoundSampler() callconv(.nvptx_device) *const anyopaque {
    return @ptrCast(&fragmentShaderBoundSampler);
}

fn keepVertexInstanced() callconv(.nvptx_device) *const anyopaque {
    return @ptrCast(&vertexShaderInstanced);
}

fn keepFragmentMRT() callconv(.nvptx_device) *const anyopaque {
    return @ptrCast(&fragmentShaderMRT);
}
