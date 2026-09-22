//! Host-safe description of how shader entry points bind to Metal.
//!
//! A shader source file exports a `functions` table built from these types.
//! The same file is compiled twice: once for the GPU target (to get LLVM IR
//! for the shader bodies) and once on the host, where only this table is
//! referenced, so the GPU-only pointer types in the shader bodies are never
//! analysed. `tools/air_splice.zig` reads the table at comptime and derives
//! the `!air.*` metadata Apple's driver needs from the Zig types themselves.
//!
//! Everything here is a plain host type: no GPU address-space pointers, so
//! the file compiles on any target.

pub const Stage = enum { vertex, fragment, kernel };

/// AIR address space of a buffer argument: device = 1, constant = 2,
/// threadgroup = 3. In the shader body the matching Zig parameter is
/// `[*]addrspace(.global) T` (device), `*addrspace(.param) const T`
/// (constant; nvptx addrspace 4, renumbered by the assembler) or
/// `[*]addrspace(.shared) T` (threadgroup).
pub const Space = enum(u8) {
    device = 1,
    constant = 2,
    threadgroup = 3,
};

/// `device const T*` (read) vs `device T*` (read_write) in the reflection
/// data. Constant buffers are always read.
pub const Access = enum { read, read_write };

pub const Buffer = struct {
    /// `[[buffer(index)]]` slot the host binds with setVertexBuffer / setBytes.
    /// Threadgroup buffers use their own `[[threadgroup(index)]]` index space.
    index: u32,
    /// Element type of the buffer. Layout comes from `@offsetOf` / `@sizeOf`.
    T: type,
    /// Argument name that shows up in Metal reflection.
    name: []const u8,
    space: Space = .device,
    access: Access = .read,
};

/// How a shader may touch a texture: the MSL `access::` qualifier, and the
/// `i32` access argument every `air.*_texture_2d` intrinsic carries
/// (research/_results/r03.json ir_patterns 0: sample 0, read 1, write 2,
/// read_write 3; the metadata's `air.sample`/`air.read`/`air.write`/
/// `air.read_write` string must agree with it).
pub const TextureAccess = enum(u8) {
    sample = 0,
    read = 1,
    write = 2,
    read_write = 3,
};

pub const Texture = struct {
    /// `[[texture(index)]]` slot, bound with `setFragmentTexture:atIndex:` /
    /// `setTexture:atIndex:`.
    index: u32,
    name: []const u8,
    /// The texture's Zig type: `gpu.Texture2D(.sample)`, `.read`, `.write`
    /// or `.read_write`. The access is read from `T.texture_access`; the
    /// shader parameter is `*addrspace(.global) const T` (sample / read) or
    /// `*addrspace(.global) T` (write / read_write).
    T: type,

    pub fn access(comptime t: Texture) TextureAccess {
        return t.T.texture_access;
    }
};

/// `[[sampler(index)]]`: a sampler state the host binds with
/// `setFragmentSamplerState:atIndex:` / `setSamplerState:atIndex:`. The
/// shader parameter is `*const gpu.Sampler` (Zig cannot spell Metal's
/// constant address space on nvptx; the splice rebuilds the header as
/// `ptr addrspace(2)`).
pub const Sampler = struct {
    index: u32,
    name: []const u8,
};

/// Built-in inputs. Each one is an ordinary parameter of the entry point
/// whose only description is `!"air.<kind>"` in the argument metadata; the
/// enum tag spells the Metal attribute name verbatim.
pub const BuiltinKind = enum {
    // vertex stage
    vertex_id,
    instance_id,
    base_vertex,
    base_instance,
    // fragment stage
    front_facing,
    point_coord,
    sample_id,
    primitive_id,
    // kernel stage
    thread_position_in_grid,
    threads_per_grid,
    thread_position_in_threadgroup,
    threadgroup_position_in_grid,
    threads_per_threadgroup,
    threadgroups_per_grid,
    thread_index_in_threadgroup,
    thread_index_in_simdgroup,
    simdgroup_index_in_threadgroup,
    threads_per_simdgroup,
    simdgroups_per_threadgroup,
    dispatch_threads_per_threadgroup,

    /// How the builtin's value may differ between the threads of a
    /// threadgroup. The assembler's uniformity analysis
    /// (tools/air/divergence.zig) seeds its lattice from this: a branch on
    /// a `.thread` builtin makes the blocks under it per-thread (no barrier
    /// and no SIMD-group call may sit there), a branch on the `.simdgroup`
    /// builtin `simdgroup_index_in_threadgroup` makes them per-SIMD-group
    /// (SIMD-group calls are still fine, threadgroup barriers are not), and
    /// the sizes and the threadgroup's own position are `.threadgroup`
    /// uniform, so a loop bounded by `threads_per_threadgroup` may contain a
    /// barrier.
    pub fn uniformity(kind: BuiltinKind) Uniformity {
        return switch (kind) {
            .threads_per_grid,
            .threadgroup_position_in_grid,
            .threads_per_threadgroup,
            .threadgroups_per_grid,
            .threads_per_simdgroup,
            .simdgroups_per_threadgroup,
            .dispatch_threads_per_threadgroup,
            => .threadgroup,
            .simdgroup_index_in_threadgroup => .simdgroup,
            else => .thread,
        };
    }

    /// Whether every thread of a threadgroup sees the same value.
    pub fn isUniform(kind: BuiltinKind) bool {
        return kind.uniformity() == .threadgroup;
    }
};

/// The uniformity lattice of the assembler's convergent-call check
/// (tools/air/divergence.zig): the widest set of threads that agree on a
/// value, or that reach a block together. Ordered: `.threadgroup` <
/// `.simdgroup` < `.thread` (more divergent).
pub const Uniformity = enum(u2) {
    /// Every thread of the threadgroup sees the same value / reaches the
    /// block together. A threadgroup barrier needs this.
    threadgroup = 0,
    /// Every thread of one SIMD-group sees the same value, different
    /// SIMD-groups may differ (`simdgroup_index_in_threadgroup`, the result
    /// of a `simd_sum`, and a block under `if (sgid == 0)`). SIMD-group
    /// collectives and `simdgroup_barrier` are fine here, as MSL defines
    /// them over the active lanes of the SIMD-group; a threadgroup barrier
    /// is not.
    simdgroup = 1,
    /// May differ between any two threads (`thread_position_in_grid`,
    /// `thread_index_in_simdgroup`, an atomic's result, a block under
    /// `if (lane == 0)`). No convergent call may sit in such a block.
    thread = 2,

    /// The least upper bound (the more divergent of the two).
    pub fn join(a: Uniformity, b: Uniformity) Uniformity {
        return @fromBackingInt(@intCast(@max(@backingInt(a), @backingInt(b))));
    }

    /// Strictly more divergent than `b`.
    pub fn above(a: Uniformity, b: Uniformity) bool {
        return @backingInt(a) > @backingInt(b);
    }
};

pub const Builtin = struct {
    kind: BuiltinKind,
    /// The parameter's Zig type: `u32`/`u16` or `@Vector(2|3, u32|u16)` for
    /// the kernel ids and sizes, `bool` for `front_facing`, `@Vector(2, f32)`
    /// for `point_coord`.
    T: type,
    name: []const u8,
};

pub const Arg = union(enum) {
    /// `[[vertex_id]]`; the payload is the argument name. Shorthand for
    /// `.builtin = .{ .kind = .vertex_id, .T = u32, .name = ... }`.
    vertex_id: []const u8,
    /// A buffer in the address space `Buffer.space` of `T` elements.
    buffer: Buffer,
    /// Fragment stage: the vertex stage's output struct. Expands to one
    /// argument per field, interpolated by the rasterizer (integer fields
    /// flat), EXCEPT a `point_size` field: the rasterizer consumes it and
    /// Metal passes no parameter for it (research/_results/r01.txt
    /// ir_patterns 2: s1's fragment takes position, flatId, normal,
    /// texCoords and not pointSize). The Zig fragment function's
    /// parameters must follow the same rule.
    stage_in: type,
    /// `texture2d<float, <access>>`, access from `Texture.T.texture_access`.
    texture: Texture,
    /// `sampler` argument (`[[sampler(index)]]`).
    sampler: Sampler,
    /// A built-in input passed as an explicit parameter.
    builtin: Builtin,
};

pub const Function = struct {
    /// Symbol name in the metallib. Must match the Zig function's name.
    /// Sentinel-terminated so it can be handed to Objective-C directly.
    name: [:0]const u8,
    stage: Stage,
    args: []const Arg,
    /// vertex: the output struct type. Its fields map to `[[...]]`
    /// attributes by name and type: `position` is `[[position]]`, a `f32`
    /// field named `point_size` is `[[point_size]]`, integer fields
    /// (`u32`, `i32`, integer vectors) are `[[flat]]` varyings, everything
    /// else is a perspective-interpolated varying.
    /// fragment: the render-target vector type, e.g. `@Vector(4, f32)`
    /// (`[[color(0)]]`), OR an output struct whose fields are named
    /// `color0`..`colorN` (`[[color(N)]]`, `@Vector(4, f32|f16|u32|i32)`)
    /// and optionally `depth` (`f32`, `[[depth(any)]]`).
    /// kernel: `void`.
    ret: type,
    /// kernel only: `[[max_total_threads_per_threadgroup(N)]]`, emitted as
    /// `!{!"air.max_work_group_size", i32 N}` on the kernel node.
    max_total_threads_per_threadgroup: ?u32 = null,
};
