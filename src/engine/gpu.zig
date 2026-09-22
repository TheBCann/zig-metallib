//! GPU helper module for shader sources: thin wrappers around AIR's
//! intrinsics, declared as `extern fn` with their exact AIR names so the
//! LLVM IR Zig emits already has Apple's declare/call shape
//! (research/_results/r06.txt ir_patterns 2-4 and 11; verified end to end in
//! research/compute-kernels/proto).
//!
//! Only shader code (compiled for nvptx64) calls anything in here. The host
//! imports the shader file too, but only reads its manifest; nothing here is
//! analysed there unless referenced, so the GPU address-space pointer types
//! in the signatures are harmless. Keep it that way: no `comptime {}` blocks,
//! no top-level `var`s, and `test`s only for the host-safe comptime sampler
//! encoder at the end of the file (tools/air/metadata.zig pulls them into
//! `zig build test`; a test that touched a GPU pointer type would break the
//! host build).
//!
//! Address spaces (DESIGN.md D1): `addrspace(.global)` = device (1),
//! `addrspace(.shared)` = threadgroup (3), `addrspace(.param)` = constant (2
//! after the assembler renumbers nvptx's 4).
//!
//! Texture helpers (sample2D, read2D, write2D, ...) and constexpr samplers
//! sit below the compute section.

const std = @import("std");
pub const air = @import("air.zig");

// ── Memory flags ────────────────────────────────────────────────────────────

/// `mem_flags` bitmask of `threadgroup_barrier` / `simdgroup_barrier` /
/// `atomic_thread_fence`: device 1, threadgroup 2, texture 4,
/// threadgroup_imageblock 8, object_data 16 (research/compute-kernels/k2.air.ll:
/// `air.wg.barrier(i32 2, i32 1)` for mem_threadgroup, `i32 3` for
/// threadgroup|device).
pub const MemFlags = packed struct(u32) {
    device: bool = false,
    threadgroup: bool = false,
    texture: bool = false,
    imageblock: bool = false,
    object_data: bool = false,
    _pad: u27 = 0,

    pub const none: MemFlags = .{};
    pub const device_only: MemFlags = .{ .device = true };
    pub const threadgroup_only: MemFlags = .{ .threadgroup = true };
    pub const all: MemFlags = .{ .device = true, .threadgroup = true, .texture = true };

    inline fn bits(self: MemFlags) i32 {
        return @bitCast(@as(u32, @bitCast(self)));
    }
};

// ── Barriers ────────────────────────────────────────────────────────────────

/// `threadgroup_barrier(flags)`: `air.wg.barrier(flags, 1)`; the second
/// argument is always 1 (threadgroup scope).
extern fn @"air.wg.barrier"(flags: i32, scope: i32) void;
/// `simdgroup_barrier(flags)`: `air.simdgroup.barrier(flags, 4)`.
extern fn @"air.simdgroup.barrier"(flags: i32, scope: i32) void;

/// Barriers must be reached by every thread of the threadgroup from the
/// same call site. Zig cannot mark the extern fn `convergent` (Apple's
/// compiler does), so LLVM may jump-thread a branch that precedes a barrier
/// through it and duplicate the call into divergent blocks; on the GPU the
/// copies then pair up as different barriers and the synchronisation is
/// lost (observed in atomicOpsKernel: `if (tidx == 0) x = 0; barrier;`,
/// review-b/fable-hazard/hazard.check.log: pipeline creation `ok`, wrong
/// result). The same happens to a barrier hidden in a `noinline` helper
/// and to the SIMD-group functions below (a duplicated `simdSum` adds up
/// only the lanes of its arm, review-b/r4/helper/check.log).
///
/// The assembler catches this at build time with a uniformity analysis
/// (tools/air/divergence.zig, lattice `air.Uniformity`): a barrier, or a
/// helper that makes one, is refused ("convergent call in divergent
/// control flow") when its block does not run for the whole threadgroup,
/// i.e. it depends on a branch over a per-thread value
/// (thread_position_in_grid, thread_index_in_threadgroup, a value loaded
/// through such an index, an atomic's result, ...) or over a per-SIMD-group
/// value (simdgroup_index_in_threadgroup, a simdSum result), or sits in a
/// loop some threads leave early. The SIMD-group functions below only need
/// their SIMD-group: `if (sgid == 0) { t = simdSum(partial[lane]); }`
/// (twoStageKernel) is accepted, `if (lane == 0) simdSum(v)` is not. A
/// loop bounded by a uniform value is fine, including the canonical
/// reduction `var s = tg_size / 2; while (s > 0) : (s >>= 1) { if (tidx <
/// s) red[tidx] += red[tidx + s]; threadgroupBarrier(...); }` (reduceKernel):
/// `threads_per_threadgroup`, `threadgroup_position_in_grid`, constant
/// buffer fields and comptime values are uniform. A Zig safety check on a
/// per-thread index (`scratch[tidx]` at ReleaseSafe, an explicit
/// `if (i >= n) @panic(...)`) traps instead of returning, so it does not
/// count as a per-thread exit and the barrier after it stays accepted.
/// When the analysis refuses a barrier that would be uniform in practice,
/// make the branch it depends on uniform (e.g. compute an `if (tidx == 0)`
/// store as an unconditional store of the same value, or move the
/// conditional work after the barrier). As in MSL, a barrier the analysis
/// cannot see to be uniform would be undefined behaviour on the GPU.
pub inline fn threadgroupBarrier(flags: MemFlags) void {
    @"air.wg.barrier"(flags.bits(), 1);
}

pub inline fn simdgroupBarrier(flags: MemFlags) void {
    @"air.simdgroup.barrier"(flags.bits(), 4);
}

// ── SIMD-group operations ───────────────────────────────────────────────────
// Naming: air.simd_<op>.<f32|u.i32|s.i32|i64>; lane/delta arguments are i16.
// All of them are convergent like the barriers (see threadgroupBarrier):
// call them in code every thread of the SIMD-group reaches, and branch on
// the lane index only after the call (sumKernel, simdOpsKernel).

extern fn @"air.simd_sum.f32"(v: f32) f32;
extern fn @"air.simd_max.f32"(v: f32) f32;
extern fn @"air.simd_min.f32"(v: f32) f32;
extern fn @"air.simd_product.f32"(v: f32) f32;
extern fn @"air.simd_prefix_inclusive_sum.f32"(v: f32) f32;
extern fn @"air.simd_broadcast.f32"(v: f32, lane: i16) f32;
extern fn @"air.simd_shuffle.f32"(v: f32, lane: i16) f32;
extern fn @"air.simd_shuffle_down.f32"(v: f32, delta: i16) f32;
extern fn @"air.simd_shuffle_up.f32"(v: f32, delta: i16) f32;
extern fn @"air.simd_shuffle_xor.f32"(v: f32, mask: i16) f32;
extern fn @"air.simd_sum.u.i32"(v: u32) u32;
extern fn @"air.simd_any"(b: bool) bool;
extern fn @"air.simd_all"(b: bool) bool;

pub inline fn simdSum(v: f32) f32 {
    return @"air.simd_sum.f32"(v);
}
pub inline fn simdMax(v: f32) f32 {
    return @"air.simd_max.f32"(v);
}
pub inline fn simdMin(v: f32) f32 {
    return @"air.simd_min.f32"(v);
}
pub inline fn simdProduct(v: f32) f32 {
    return @"air.simd_product.f32"(v);
}
pub inline fn simdPrefixInclusiveSum(v: f32) f32 {
    return @"air.simd_prefix_inclusive_sum.f32"(v);
}
pub inline fn simdBroadcast(v: f32, lane: u16) f32 {
    return @"air.simd_broadcast.f32"(v, @bitCast(lane));
}
pub inline fn simdShuffle(v: f32, lane: u16) f32 {
    return @"air.simd_shuffle.f32"(v, @bitCast(lane));
}
pub inline fn simdShuffleDown(v: f32, delta: u16) f32 {
    return @"air.simd_shuffle_down.f32"(v, @bitCast(delta));
}
pub inline fn simdShuffleUp(v: f32, delta: u16) f32 {
    return @"air.simd_shuffle_up.f32"(v, @bitCast(delta));
}
pub inline fn simdShuffleXor(v: f32, mask: u16) f32 {
    return @"air.simd_shuffle_xor.f32"(v, @bitCast(mask));
}
pub inline fn simdSumU32(v: u32) u32 {
    return @"air.simd_sum.u.i32"(v);
}
pub inline fn simdAny(b: bool) bool {
    return @"air.simd_any"(b);
}
pub inline fn simdAll(b: bool) bool {
    return @"air.simd_all"(b);
}

// ── Fragment stage ──────────────────────────────────────────────────────────

/// `discard_fragment()`: `declare void @air.discard_fragment()`, called in a
/// conditional block; execution continues after the call and the fragment's
/// outputs are dropped (research/_results/r01.txt ir_patterns 5, finding 11:
/// Zig emits the identical declare/call; verified rendering with the
/// discarded pixels left at the clear colour, verify-render-bindings/
/// v5_offscreen.log and my_shader.fragmentShaderMRT via `zig build check`).
extern fn @"air.discard_fragment"() void;

pub inline fn discardFragment() void {
    @"air.discard_fragment"();
}

// ── Atomics ─────────────────────────────────────────────────────────────────
// `air.atomic.<global|local>.<op>.<u|s>.i32(ptr, value, order, scope, true)`:
// order 0 = memory_order_relaxed (the only one this toolchain accepts for
// fetch ops), scope 2 = thread_scope_device for device memory, 1 =
// thread_scope_threadgroup for threadgroup memory. Returns the old value.
// Names observed in Apple's output: add.u, sub.s, max.s, min.u, and.u, or.u,
// xor.u, xchg (research/compute-kernels/k2.air.ll); the unsigned sub.u/max.u
// spellings follow the same pattern and, like every helper here, pass
// Metal's runtime compiler and a verified dispatch (my_shader.atomicOpsKernel
// / simdOpsKernel via `zig build check`).

const order_relaxed: i32 = 0;
const order_seq_cst: i32 = 5;
const scope_threadgroup: i32 = 1;
const scope_device: i32 = 2;

pub const DeviceU32 = *addrspace(.global) u32;
pub const ThreadgroupU32 = *addrspace(.shared) u32;

extern fn @"air.atomic.global.add.u.i32"(p: DeviceU32, v: u32, order: i32, scope: i32, b: bool) u32;
extern fn @"air.atomic.global.sub.u.i32"(p: DeviceU32, v: u32, order: i32, scope: i32, b: bool) u32;
extern fn @"air.atomic.global.max.u.i32"(p: DeviceU32, v: u32, order: i32, scope: i32, b: bool) u32;
extern fn @"air.atomic.global.min.u.i32"(p: DeviceU32, v: u32, order: i32, scope: i32, b: bool) u32;
extern fn @"air.atomic.global.or.u.i32"(p: DeviceU32, v: u32, order: i32, scope: i32, b: bool) u32;
extern fn @"air.atomic.global.and.u.i32"(p: DeviceU32, v: u32, order: i32, scope: i32, b: bool) u32;
extern fn @"air.atomic.global.xor.u.i32"(p: DeviceU32, v: u32, order: i32, scope: i32, b: bool) u32;
extern fn @"air.atomic.global.xchg.i32"(p: DeviceU32, v: u32, order: i32, scope: i32, b: bool) u32;
extern fn @"air.atomic.local.add.u.i32"(p: ThreadgroupU32, v: u32, order: i32, scope: i32, b: bool) u32;
/// `atomic_thread_fence(flags, order, scope)`. Native LLVM `fence` crashes
/// Metal's runtime compiler; this intrinsic is the only working form.
extern fn @"air.atomic.fence"(flags: i32, order: i32, scope: i32) void;

pub inline fn atomicAdd(p: DeviceU32, v: u32) u32 {
    return @"air.atomic.global.add.u.i32"(p, v, order_relaxed, scope_device, true);
}
pub inline fn atomicSub(p: DeviceU32, v: u32) u32 {
    return @"air.atomic.global.sub.u.i32"(p, v, order_relaxed, scope_device, true);
}
pub inline fn atomicMax(p: DeviceU32, v: u32) u32 {
    return @"air.atomic.global.max.u.i32"(p, v, order_relaxed, scope_device, true);
}
pub inline fn atomicMin(p: DeviceU32, v: u32) u32 {
    return @"air.atomic.global.min.u.i32"(p, v, order_relaxed, scope_device, true);
}
pub inline fn atomicOr(p: DeviceU32, v: u32) u32 {
    return @"air.atomic.global.or.u.i32"(p, v, order_relaxed, scope_device, true);
}
pub inline fn atomicAnd(p: DeviceU32, v: u32) u32 {
    return @"air.atomic.global.and.u.i32"(p, v, order_relaxed, scope_device, true);
}
pub inline fn atomicXor(p: DeviceU32, v: u32) u32 {
    return @"air.atomic.global.xor.u.i32"(p, v, order_relaxed, scope_device, true);
}
pub inline fn atomicExchange(p: DeviceU32, v: u32) u32 {
    return @"air.atomic.global.xchg.i32"(p, v, order_relaxed, scope_device, true);
}
pub inline fn atomicAddThreadgroup(p: ThreadgroupU32, v: u32) u32 {
    return @"air.atomic.local.add.u.i32"(p, v, order_relaxed, scope_threadgroup, true);
}
/// `atomic_thread_fence(flags, memory_order_seq_cst, scope)`; the scope is
/// device when `flags.device` is set, threadgroup otherwise
/// (research/compute-kernels/k3.air.ll: `(i32 1, i32 5, i32 2)` / `(i32 2, i32 5, i32 1)`).
pub inline fn atomicFence(flags: MemFlags) void {
    @"air.atomic.fence"(flags.bits(), order_seq_cst, if (flags.device) scope_device else scope_threadgroup);
}

// ── Textures and samplers ───────────────────────────────────────────────────
// Every intrinsic below is spelled exactly as Apple's compiler declares it
// (research/_results/r03.json ir_patterns 0-3, 11-14; hand-written
// libraries h01-h10 in research/texture-sampling pass pipeline creation,
// the pixel readbacks of `zig build check` prove the results). Two things
// Zig cannot say are fixed by the assembler (DESIGN.md D5-12c): the real
// return type of the sample/read intrinsics is `{ <4 x float>, i8 }` (the
// i8 is the sparse-residency status; extern structs may not hold vectors,
// so the declarations here return the bare vector and the assembler wraps
// every call in `extractvalue 0`), and the sampler parameter lives in
// Metal's constant address space, which nvptx has no spelling for (`*const
// Sampler` here, `ptr addrspace(2)` in the module).

/// Metal's `texture2d<float, access::<access>>`: an opaque handle only ever
/// used through a pointer, `*addrspace(.global) const Texture2D(.sample)`
/// (or `.read`) for textures the shader only reads and
/// `*addrspace(.global) Texture2D(.write)` (or `.read_write`) for textures
/// it writes. The manifest names the same type (`air.Texture.T`) and
/// derives the `air.sample` / `air.read` / ... metadata from
/// `texture_access`; the helpers pass the same value as the intrinsics'
/// trailing `i32 access` argument (the two must agree).
pub fn Texture2D(comptime access: air.TextureAccess) type {
    return opaque {
        pub const texture_access = access;
        pub const elem = f32;
    };
}

/// Metal's `sampler`: a `[2 x i64]` state word pair in constant memory,
/// either a `[[sampler(n)]]` argument (`*const Sampler` parameter,
/// `air.Arg.sampler`) or a constexpr one (`constexprSampler`).
pub const Sampler = opaque {};

/// The two 64-bit words Apple's compiler materialises for a constexpr
/// sampler (`@__air_sampler_state = internal addrspace(2) constant [2 x i64]`).
pub const SamplerState = [2]u64;

/// `address::` modes. `clamp_to_border` packs as the same field value as
/// `clamp_to_zero` (0); the border colour is a separate field
/// (research/texture-sampling/s41_border_only.ll, s12_clamp_border.ll).
pub const Address = enum(u8) { clamp_to_zero, clamp_to_edge, repeat, mirrored_repeat, clamp_to_border };
/// `mag_filter::` / `min_filter::` (bicubic is 2, s36/s37 vs 0x1449).
pub const Filter = enum(u2) { nearest = 0, linear = 1, bicubic = 2 };
/// `mip_filter::`.
pub const MipFilter = enum(u2) { none = 0, nearest = 1, linear = 2 };
/// `coord::` (pixel coordinates only set bit 15; the intrinsic arguments
/// do not change).
pub const Coord = enum(u1) { normalized = 0, pixel = 1 };
/// `compare_func::`; Apple encodes the default (never) as 8, the others
/// in MSL enum order.
pub const CompareFunc = enum(u4) { never = 8, less = 1, less_equal = 2, greater = 3, greater_equal = 4, equal = 5, not_equal = 6, always = 7 };
/// `border_color::`.
pub const BorderColor = enum(u2) { transparent_black = 0, opaque_black = 1, opaque_white = 2 };
/// `reduction::`.
pub const Reduction = enum(u2) { weighted_average = 0, minimum = 1, maximum = 2 };

/// Everything `constexpr sampler s(...)` can say, with MSL's defaults.
pub const SamplerDesc = struct {
    s_address: Address = .clamp_to_edge,
    t_address: Address = .clamp_to_edge,
    r_address: Address = .clamp_to_edge,
    mag_filter: Filter = .nearest,
    min_filter: Filter = .nearest,
    mip_filter: MipFilter = .none,
    coord: Coord = .normalized,
    compare_func: CompareFunc = .never,
    /// 1..16.
    max_anisotropy: u8 = 1,
    lod_clamp_min: f32 = 0.0,
    /// MSL's default is the largest finite half (0x7bff).
    lod_clamp_max: f32 = 65504.0,
    border_color: BorderColor = .transparent_black,
    reduction: Reduction = .weighted_average,
    lod_bias: f32 = 0.0,
};

/// `address::` field value: 3 bits, clamp_to_border shares 0 with clamp_to_zero.
fn addressBits(a: Address) u64 {
    return switch (a) {
        .clamp_to_zero, .clamp_to_border => 0,
        .clamp_to_edge => 1,
        .repeat => 2,
        .mirrored_repeat => 3,
    };
}

/// IEEE half bit pattern of an f32, the encoding of the lod fields.
fn half(x: f32) u16 {
    return @bitCast(@as(f16, @floatCast(x)));
}

/// The sampler state words for `d`, laid out as Apple's compiler does it
/// (verified from 45 one-setting-at-a-time configurations,
/// research/texture-sampling/sampler_decoded.txt and decode_sampler.py;
/// re-confirmed by research/_results/r08.json claim 2):
///   w0 [2:0] s_address, [5:3] t_address, [8:6] r_address (0 clamp_to_zero /
///   clamp_to_border, 1 clamp_to_edge, 2 repeat, 3 mirrored_repeat);
///   [10:9] mag_filter, [12:11] min_filter (0 nearest, 1 linear, 2 bicubic);
///   [14:13] mip_filter (0 none, 1 nearest, 2 linear); [15] coord (1 = pixel);
///   [19:16] compare_func (8 never, 1 less .. 7 always); [23:20] max_anisotropy - 1;
///   [39:24] lod_clamp_min as IEEE half; [55:40] lod_clamp_max as IEEE half;
///   [57:56] border_color; [59:58] reduction; bit 63 (floor's "is_constant")
///   is never set by Apple and stays 0.
///   w1 [15:0] lod_bias as IEEE half.
/// The default sampler is {0x007bff0000080049, 0}; (mag linear, min linear)
/// is {0x007bff0000080a49, 0}.
pub fn samplerState(comptime d: SamplerDesc) SamplerState {
    if (d.max_anisotropy < 1 or d.max_anisotropy > 16) @compileError("SamplerDesc.max_anisotropy must be 1..16");
    var w0: u64 = 0;
    w0 |= addressBits(d.s_address);
    w0 |= addressBits(d.t_address) << 3;
    w0 |= addressBits(d.r_address) << 6;
    w0 |= @as(u64, @backingInt(d.mag_filter)) << 9;
    w0 |= @as(u64, @backingInt(d.min_filter)) << 11;
    w0 |= @as(u64, @backingInt(d.mip_filter)) << 13;
    w0 |= @as(u64, @backingInt(d.coord)) << 15;
    w0 |= @as(u64, @backingInt(d.compare_func)) << 16;
    w0 |= @as(u64, d.max_anisotropy - 1) << 20;
    w0 |= @as(u64, half(d.lod_clamp_min)) << 24;
    w0 |= @as(u64, half(d.lod_clamp_max)) << 40;
    w0 |= @as(u64, @backingInt(d.border_color)) << 56;
    w0 |= @as(u64, @backingInt(d.reduction)) << 58;
    return .{ w0, half(d.lod_bias) };
}

/// `constexpr sampler s(...)`: a pointer to a per-descriptor constant global
/// holding `samplerState(d)`. Zig emits it as an addrspace-0 constant; the
/// assembler relocates it to Metal's constant space (DESIGN.md D1) and lists
/// it under `!air.sampler_states`, without which Metal's backend crashes at
/// pipeline creation (r03 finding 6: h04_no_sampler_md vs h01_constexpr).
pub inline fn constexprSampler(comptime d: SamplerDesc) *const Sampler {
    const Holder = struct {
        const state: SamplerState = samplerState(d);
    };
    return @ptrCast(&Holder.state);
}

const TexturePtr = *addrspace(.global) const anyopaque;
const MutableTexturePtr = *addrspace(.global) anyopaque;

/// `sample(s, coord)` and its `level(l)` / `bias(b)` variants: the header
/// wrapper `__metal_sample_texture_2d_t(t, s, coord, has_offset = true,
/// offset, is_explicit_level, level_or_bias, min_lod_clamp, access)`.
extern fn @"air.sample_texture_2d.v4f32"(tex: TexturePtr, s: *const Sampler, coord: @Vector(2, f32), has_offset: bool, offset: @Vector(2, i32), is_explicit_level: bool, level_or_bias: f32, min_lod_clamp: f32, access: i32) @Vector(4, f32);
/// The implicit sampler `read()` goes through.
extern fn @"air.get_read_sampler"() *const Sampler;
/// `read(coord[, lod])`: `(tex, get_read_sampler(), coord, offset 0, lod, access)`.
extern fn @"air.read_texture_2d.v4f32"(tex: TexturePtr, s: *const Sampler, coord: @Vector(2, i32), offset: @Vector(2, i32), lod: i32, access: i32) @Vector(4, f32);
/// `write(color, coord[, lod])`.
extern fn @"air.write_texture_2d.v4f32"(tex: MutableTexturePtr, coord: @Vector(2, i32), color: @Vector(4, f32), lod: i32, access: i32) void;
/// `get_width(lod)` / `get_height(lod)`.
extern fn @"air.get_width_texture_2d"(tex: TexturePtr, lod: i32) i32;
extern fn @"air.get_height_texture_2d"(tex: TexturePtr, lod: i32) i32;

/// The `i32 access` argument for a `*Texture2D(access)` pointer type.
fn accessOf(comptime Ptr: type) i32 {
    const info = @typeInfo(Ptr);
    if (info != .pointer or !@hasDecl(info.pointer.child, "texture_access"))
        @compileError("expected a pointer to gpu.Texture2D(access), got " ++ @typeName(Ptr));
    return @intCast(@backingInt(@as(air.TextureAccess, info.pointer.child.texture_access)));
}

fn requireWritable(comptime Ptr: type) void {
    const a: air.TextureAccess = @typeInfo(Ptr).pointer.child.texture_access;
    if (a != .write and a != .read_write) @compileError("write2D needs a Texture2D(.write) or Texture2D(.read_write), got " ++ @typeName(Ptr));
}

/// `tex.sample(s, uv)`: implicit level of detail (fragment stage only; the
/// vertex stage must use `sample2DLevel`, r03 ir_patterns 14).
pub inline fn sample2D(tex: *addrspace(.global) const Texture2D(.sample), s: *const Sampler, uv: @Vector(2, f32)) @Vector(4, f32) {
    return @"air.sample_texture_2d.v4f32"(@ptrCast(tex), s, uv, true, .{ 0, 0 }, false, 0.0, 0.0, accessOf(@TypeOf(tex)));
}

/// `tex.sample(s, uv, level(l))`.
pub inline fn sample2DLevel(tex: *addrspace(.global) const Texture2D(.sample), s: *const Sampler, uv: @Vector(2, f32), level: f32) @Vector(4, f32) {
    return @"air.sample_texture_2d.v4f32"(@ptrCast(tex), s, uv, true, .{ 0, 0 }, true, level, 0.0, accessOf(@TypeOf(tex)));
}

/// `tex.sample(s, uv, bias(b))`.
pub inline fn sample2DBias(tex: *addrspace(.global) const Texture2D(.sample), s: *const Sampler, uv: @Vector(2, f32), bias: f32) @Vector(4, f32) {
    return @"air.sample_texture_2d.v4f32"(@ptrCast(tex), s, uv, true, .{ 0, 0 }, false, bias, 0.0, accessOf(@TypeOf(tex)));
}

/// `tex.read(xy, lod)` for any `*Texture2D(...)` pointer; the access
/// argument follows the texture type (sample 0, read 1, read_write 3).
pub inline fn read2D(tex: anytype, xy: @Vector(2, u32), lod: u32) @Vector(4, f32) {
    const coord: @Vector(2, i32) = @bitCast(xy);
    return @"air.read_texture_2d.v4f32"(@ptrCast(tex), @"air.get_read_sampler"(), coord, .{ 0, 0 }, @bitCast(lod), accessOf(@TypeOf(tex)));
}

/// `tex.write(color, xy)` (mip level 0) for a `*Texture2D(.write)` or
/// `*Texture2D(.read_write)` pointer.
pub inline fn write2D(tex: anytype, xy: @Vector(2, u32), color: @Vector(4, f32)) void {
    requireWritable(@TypeOf(tex));
    const coord: @Vector(2, i32) = @bitCast(xy);
    @"air.write_texture_2d.v4f32"(@ptrCast(tex), coord, color, 0, accessOf(@TypeOf(tex)));
}

/// `tex.get_width(lod)`.
pub inline fn width2D(tex: anytype, lod: u32) u32 {
    _ = accessOf(@TypeOf(tex));
    return @bitCast(@"air.get_width_texture_2d"(@ptrCast(tex), @bitCast(lod)));
}

/// `tex.get_height(lod)`.
pub inline fn height2D(tex: anytype, lod: u32) u32 {
    _ = accessOf(@TypeOf(tex));
    return @bitCast(@"air.get_height_texture_2d"(@ptrCast(tex), @bitCast(lod)));
}

// ── Tests (host only: the sampler encoder is plain comptime arithmetic) ─────
// Expected words are Apple's, from research/texture-sampling/sampler_decoded.txt
// (s00..s52) and research/verify-textures-samplers (b_* sweep, r08 claim 2).

test "sampler encoder: defaults and filters (s00, s01, s03, s04, s36, s38)" {
    try std.testing.expectEqual(SamplerState{ 0x007bff0000080049, 0 }, comptime samplerState(.{}));
    try std.testing.expectEqual(SamplerState{ 0x007bff0000080a49, 0 }, comptime samplerState(.{ .mag_filter = .linear, .min_filter = .linear }));
    try std.testing.expectEqual(SamplerState{ 0x007bff0000080249, 0 }, comptime samplerState(.{ .mag_filter = .linear }));
    try std.testing.expectEqual(SamplerState{ 0x007bff0000080849, 0 }, comptime samplerState(.{ .min_filter = .linear }));
    try std.testing.expectEqual(SamplerState{ 0x007bff0000081449, 0 }, comptime samplerState(.{ .mag_filter = .bicubic, .min_filter = .bicubic }));
    try std.testing.expectEqual(SamplerState{ 0x007bff0000084049, 0 }, comptime samplerState(.{ .mip_filter = .linear }));
    try std.testing.expectEqual(SamplerState{ 0x007bff0000082049, 0 }, comptime samplerState(.{ .mip_filter = .nearest }));
    try std.testing.expectEqual(SamplerState{ 0x007bff0000084a92, 0 }, comptime samplerState(.{ .mag_filter = .linear, .min_filter = .linear, .mip_filter = .linear, .s_address = .repeat, .t_address = .repeat, .r_address = .repeat }));
}

test "sampler encoder: address modes and border colour (s08, s10, s11, s12, s13-s15, s22, s43)" {
    try std.testing.expectEqual(SamplerState{ 0x007bff0000080092, 0 }, comptime samplerState(.{ .s_address = .repeat, .t_address = .repeat, .r_address = .repeat }));
    try std.testing.expectEqual(SamplerState{ 0x007bff0000080000, 0 }, comptime samplerState(.{ .s_address = .clamp_to_zero, .t_address = .clamp_to_zero, .r_address = .clamp_to_zero }));
    try std.testing.expectEqual(SamplerState{ 0x007bff00000800db, 0 }, comptime samplerState(.{ .s_address = .mirrored_repeat, .t_address = .mirrored_repeat, .r_address = .mirrored_repeat }));
    try std.testing.expectEqual(SamplerState{ 0x007bff000008004a, 0 }, comptime samplerState(.{ .s_address = .repeat }));
    try std.testing.expectEqual(SamplerState{ 0x007bff0000080051, 0 }, comptime samplerState(.{ .t_address = .repeat }));
    try std.testing.expectEqual(SamplerState{ 0x007bff0000080089, 0 }, comptime samplerState(.{ .r_address = .repeat }));
    try std.testing.expectEqual(SamplerState{ 0x007bff0000080083, 0 }, comptime samplerState(.{ .s_address = .mirrored_repeat, .t_address = .clamp_to_zero, .r_address = .repeat }));
    try std.testing.expectEqual(SamplerState{ 0x027bff0000080000, 0 }, comptime samplerState(.{ .s_address = .clamp_to_border, .t_address = .clamp_to_border, .r_address = .clamp_to_border, .border_color = .opaque_white }));
    try std.testing.expectEqual(SamplerState{ 0x017bff0000080000, 0 }, comptime samplerState(.{ .s_address = .clamp_to_border, .t_address = .clamp_to_border, .r_address = .clamp_to_border, .border_color = .opaque_black }));
}

test "sampler encoder: coord, compare, anisotropy, lod clamp, reduction, bias (s16-s19, s23-s33, s40, s45, s49, s50, s55)" {
    try std.testing.expectEqual(SamplerState{ 0x007bff0000088049, 0 }, comptime samplerState(.{ .coord = .pixel }));
    try std.testing.expectEqual(SamplerState{ 0x007bff0000010049, 0 }, comptime samplerState(.{ .compare_func = .less }));
    try std.testing.expectEqual(SamplerState{ 0x007bff0000040049, 0 }, comptime samplerState(.{ .compare_func = .greater_equal }));
    try std.testing.expectEqual(SamplerState{ 0x007bff0000070049, 0 }, comptime samplerState(.{ .compare_func = .always }));
    try std.testing.expectEqual(SamplerState{ 0x007bff0000080049, 0 }, comptime samplerState(.{ .compare_func = .never }));
    try std.testing.expectEqual(SamplerState{ 0x007bff0000180049, 0 }, comptime samplerState(.{ .max_anisotropy = 2 }));
    try std.testing.expectEqual(SamplerState{ 0x007bff0000380049, 0 }, comptime samplerState(.{ .max_anisotropy = 4 }));
    try std.testing.expectEqual(SamplerState{ 0x007bff0000f80049, 0 }, comptime samplerState(.{ .max_anisotropy = 16 }));
    try std.testing.expectEqual(SamplerState{ 0x0044000000080049, 0 }, comptime samplerState(.{ .lod_clamp_min = 0.0, .lod_clamp_max = 4.0 }));
    try std.testing.expectEqual(SamplerState{ 0x0040003c00080049, 0 }, comptime samplerState(.{ .lod_clamp_min = 1.0, .lod_clamp_max = 2.0 }));
    try std.testing.expectEqual(SamplerState{ 0x0048003800080049, 0 }, comptime samplerState(.{ .lod_clamp_min = 0.5, .lod_clamp_max = 8.0 }));
    try std.testing.expectEqual(SamplerState{ 0x0063d00000080049, 0 }, comptime samplerState(.{ .lod_clamp_max = 1000.0 }));
    try std.testing.expectEqual(SamplerState{ 0x047bff0000080049, 0 }, comptime samplerState(.{ .reduction = .minimum }));
    try std.testing.expectEqual(SamplerState{ 0x087bff0000080049, 0 }, comptime samplerState(.{ .reduction = .maximum }));
    try std.testing.expectEqual(SamplerState{ 0x007bff0000080049, 0x3c00 }, comptime samplerState(.{ .lod_bias = 1.0 }));
    try std.testing.expectEqual(SamplerState{ 0x007bff0000080049, 0xb800 }, comptime samplerState(.{ .lod_bias = -0.5 }));
    // bias(2.0) + lod_clamp(1, 3) (r03 ir_patterns 4 notes).
    try std.testing.expectEqual(SamplerState{ 0x0042003c00080049, 0x4000 }, comptime samplerState(.{ .lod_bias = 2.0, .lod_clamp_min = 1.0, .lod_clamp_max = 3.0 }));
}
