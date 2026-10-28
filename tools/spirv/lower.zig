//! SPIR-V module -> LLVM IR text in the assembler's dialect plus the runtime
//! manifest of every compute entry point (docs/spirv-plan.md stages 1-4,
//! docs/spirv.md for the tables). The text goes through the same
//! tools/air/assembler.zig and packer as the Zig path; `xcrun metal -Xclang
//! -opaque-pointers` reads it too (the attribution step), so it is written in
//! the LLVM-15-era spelling Apple's front end knows: hex float literals
//! (`0x3FF0000000000000`, `0xH3C00`), explicit vector constants instead of
//! `splat`, `undef` lanes in shuffle masks, no attributes or metadata.
//!
//! Shape of the output:
//!   * one `define void @<entry name>(...)` per GLCompute / Kernel entry
//!     point (Apple packs one module per function, so each entry is
//!     assembled on its own) and one `define private` per function the
//!     entries reach through OpFunctionCall;
//!   * AIR kernels take their buffers and builtins as parameters where SPIR-V
//!     has module-scope variables. Entry parameters, in this order: buffers
//!     by location index (descriptor bindings and the entry function's own
//!     OpFunctionParameters, OpenCL style), the PushConstant block, then the
//!     builtins in OpEntryPoint interface order. A helper that touches such a
//!     variable gets it as an extra trailing parameter (variables in id
//!     order), so one helper body serves every entry point;
//!   * StorageBuffer / PhysicalStorageBuffer / CrossWorkgroup / Generic
//!     pointers are `ptr addrspace(1)` (device), Uniform / PushConstant /
//!     UniformConstant `ptr addrspace(2)` (constant), Workgroup `ptr
//!     addrspace(3)` with each variable an `internal addrspace(3) global ...
//!     undef` (threadgroup memory, as on the Zig path; the Workgroup Block
//!     variables of SPV_KHR_workgroup_memory_explicit_layout alias, so they
//!     share one `[N x i8]` global, `Lowerer.workgroupBlocks`), Function / Private
//!     `ptr` (Private variables become allocas in the entry, passed to
//!     helpers). An Input variable is never a pointer here: it is the builtin
//!     parameter (`<3 x i32>`), an access chain into it picks a lane and a
//!     load is an `extractelement` (zero-extended when the variable is 64-bit,
//!     as OpenCL's `<3 x u64>` ids are);
//!   * types: Bool -> i1 in registers and i8 in memory (Metal has no i1
//!     memory type), so loads compare with 0 and stores zero-extend, and a
//!     bool inside a struct or array is an i8 there; Int 8-64 -> iN; Float
//!     16/32 -> half/float (no double: Metal has none); Vector -> <N x T>;
//!     Matrix -> an array of column vectors; Array -> [N x T]; RuntimeArray
//!     -> [0 x T] (last struct member only); Struct -> a literal struct whose
//!     natural AIR layout must reproduce every Offset decoration, else a
//!     packed `<{ ... }>` with `[N x i8]` padding members and remapped member
//!     indices. AIR's datalayout gives a 3-element vector the allocation of
//!     four, so in a packed struct a vec3 member is a `[3 x T]` (std430
//!     `vec3` + `float` at 12), as is the element of an array whose
//!     ArrayStride is three components; the vector is still loaded and stored
//!     as `<3 x T>` (store size 12). A struct ending in a vec3 has a tight
//!     form too (`Lowerer.tightLayout`), used where scalar block layout puts
//!     the next member or array element inside its AIR padding; loads and
//!     stores through a pointer to it use that form (`Member.tight`).
//!     ArrayStride must otherwise equal the element's allocation size and
//!     MatrixStride the column's; PhysicalStorageBuffer pointers may sit in
//!     buffers and convert to and from 64-bit integers (`inttoptr` /
//!     `ptrtoint`, `Fn.pointerIntCast`; GL_EXT_buffer_reference);
//!   * a Uniform variable of a BufferBlock struct (SPIR-V 1.0 SSBO) is device
//!     memory: every pointer derived from it is `ptr addrspace(1)` although
//!     its type says Uniform (`Fn.spaces`); a store into constant memory is
//!     refused; an initialized UniformConstant (OpenCL `__constant` table) is
//!     a `private addrspace(2) constant` global; descriptor arrays refused;
//!   * access chains are one GEP from the variable (a chain on a chain
//!     continues the root's GEP, unless a bitcast changed the pointee type
//!     in between: `Fn.copyPtrInfo`), and an aggregate load or store through a
//!     runtime-indexed element of private memory goes member by member:
//!     MTLCompilerService crashes on the aggregate form for a padded struct
//!     (`[2 x { i32, <4 x i32> }]`, exec test composite_select);
//!   * OpSelect of a struct/array/matrix is selected member by member;
//!   * stage 4 (`Fn.controlBarrier`, `memoryBarrier`, `atomic`):
//!     OpControlBarrier -> `air.wg.barrier(flags, 1)` (Workgroup) /
//!     `air.simdgroup.barrier(flags, 4)` (Subgroup; with a Workgroup or
//!     wider memory scope an `air.atomic.fence` at that scope first, since
//!     the SIMD-group barrier orders memory only within the SIMD-group),
//!     OpMemoryBarrier -> `air.atomic.fence(flags, 5, scope)`, the flags
//!     from the semantics' storage-class bits; OpAtomic* on device and
//!     threadgroup pointers ->
//!     native `atomicrmw` / `cmpxchg` / `load atomic` / `store atomic` with
//!     the semantics' ordering. Metal's limits, found by the `atom_*` probes
//!     (tools/spirv/probes.zig) and refused by name: 64-bit atomics are load,
//!     store and device min/max with the result unused (the rest crash
//!     MTLCompilerService), no float min/max (`atomicrmw fmin` is unknown to
//!     Metal's LLVM), and no float add on threadgroup memory: Metal builds
//!     `atomicrmw fadd ptr addrspace(3)` and computes a wrong answer (the
//!     corpus kernel hand-workgroup-atomics:atomic_fadd_tg; the refusal is
//!     lifted by `Options.allow_metal_miscompiles`, which only the corpus
//!     driver sets, to notice a Metal fix). The assembler's uniformity analysis checks every barrier
//!     (tools/air/divergence.zig follows the allocas of Function variables);
//!   * OpSelectionMerge / OpLoopMerge / OpLine / OpNop and NonSemantic.*
//!     OpExtInsts are dropped; blocks keep SPIR-V's order, which already puts
//!     dominators first; OpPhi is a `phi` with one entry per CFG edge
//!     (rendered after the function so a back edge may name a value that an
//!     expansion defines later);
//!   * decorations on results: precision, fast-math, wrap, aliasing and
//!     alignment hints are ignored (`ignorableDecoration`); SaturatedConversion
//!     and FPRoundingMode are implemented on the conversions (or refused where
//!     Metal cannot honour them); Volatile (a memory operand, or a variable or
//!     member decoration a pointer derives from: `Lowerer.findVolatile`)
//!     makes the access `load volatile` / `store volatile`; anything else is
//!     refused;
//!   * GLSL.std.450 and OpenCL.std map onto the assembler's intrinsic
//!     spellings: `llvm.*` where tools/air/intrinsics.zig already decides
//!     precise against fast (`llvm.sin` -> `air.sin` / `air.fast_sin`,
//!     `llvm.round` -> the sign-keeping expansion, `llvm.ldexp` ->
//!     `air.ldexp`), Apple's own `air.<fn>` / `air.fast_<fn>` names (from
//!     `xcrun metal` output for the MSL functions: `air.asin.f32`,
//!     `air.asinh.f32`, `air.rsqrt.f32`, `air.atan2.v4f32`,
//!     `air.dot.v4f32`, ...) for the functions intrinsics.zig does not
//!     rename, and expansions for the rest. Every mapping was built into a
//!     compute pipeline on Metal (docs/spirv.md, tools/spirv/probes.zig).
//!     OpenCL.std powr, normalize, length and distance keep OpenCL C's
//!     defined edge results (powr's NaNs, normalize of a zero or infinite
//!     vector, no overflow in the sum of squares; exec test
//!     opencl_edge_math), which GLSL leaves undefined;
//!   * alignment: a load or store claims only what its address is known to
//!     be aligned to (`Fn.knownAlign`): an access chain into a packed
//!     member narrows it, and OpSelect, phis, helper parameters and
//!     pointers kept in memory carry it on. Phis and parameters may be
//!     lowered before what narrows them, so `lower` repeats the lowering
//!     with what a round learned (`Learned`) until nothing narrows; a fact
//!     is used in the round that learns it, and a function's pointer phis
//!     are solved together (`Fn.settleAlign`), so call chains, forward phi
//!     chains and nested loops settle in two rounds (exec tests
//!     packed_pointer_flow, packed_pointer_call_chain);
//!   * producer workarounds (`Options.zig_workarounds`, on by default), keyed
//!     on the header's Generator word as Mesa's spirv_to_nir keys its
//!     glslang workarounds: a module from Zig's self-hosted backend
//!     (generator tool 41, `zig_generator_id` in Zig's spirv/spec.zig) that
//!     widens an UNSIGNED-typed integer to a SIGNED-typed one with OpSConvert
//!     is zero-extended. Zig picks OpSConvert/OpUConvert by the result's
//!     signedness (`buildConvert`, src/codegen/spirv/CodeGen.zig:3474 at
//!     f2f0d97a9b), so `@as(i32, @as(u16, x))` comes out as
//!     `OpSConvert %i32 %u16`, a sign extension by the SPIR-V specification;
//!     Zig's own semantics, and the nvptx build of the same source,
//!     zero-extend. buildConvert is the only place Zig emits OpSConvert. In
//!     the Vulkan flavour (`intType` keeps the signedness) a signed result
//!     with an unsigned operand type means the Zig source type was unsigned
//!     (an integer widening, the packed-struct field insert, a shift amount
//!     cast), or an `@intCast` pun whose defined value is non-negative, where
//!     zero and sign extension agree: the workaround restores exactly Zig's
//!     meaning. In the OpenCL/amdhsa flavour `intType` makes every integer
//!     unsigned, so the result type is unsigned too and `OpSConvert %u32
//!     %u16` may be a real `i16 -> i32` sign extension: the signed-result
//!     condition keeps those, and the bug, with the specification (tested both
//!     ways below; exec tests zig_convert_vulkan and zig_convert_opencl in
//!     tools/spirv/exec.zig run both shapes on the GPU). Found by the
//!     crosscheck bench (seed 2,
//!     tools/crosscheck/known/spirv_zig_unsigned_widen.zig; docs/spirv.md).
//!     The same rule has a mirror case: an unsigned result gets OpUConvert, a
//!     zero extension by the specification, even for a SIGNED source. Of the
//!     buildConvert callers that widen a signed source to an unsigned type,
//!     `airIntCast`'s composite-integer path (CodeGen.zig:6078; `@as(i128,
//!     @as(i16, x))`) means a sign extension: it converts the source into
//!     the u64 (or u32) limb 0 and then fills the upper limbs from limb 0's
//!     top bit with `OpBitcast %i64` + `OpShiftRightArithmetic ... 63`
//!     (CodeGen.zig:6111-6131). The packed-pointer store (`airStore`,
//!     CodeGen.zig:7521: a signed field into the unsigned host integer, then
//!     shifted and ORed in; unreachable at f2f0d97a9b, where a pointer to a
//!     packed-struct field is a TODO, CodeGen.zig:6909) means a zero
//!     extension, and an `@intCast` from
//!     signed to unsigned (CodeGen.zig:6150) has only non-negative defined
//!     values. So a widening OpUConvert from a signed-typed operand to an
//!     unsigned-typed result is a `sext` exactly when its result is
//!     sign-filled that way (`Lowerer.findSignFills`), and a `zext`
//!     otherwise. In the OpenCL flavour the operand is typed unsigned and
//!     the same sequence is also what `@as(i64, @bitCast(@as(u64, u16x))) >>
//!     63` emits, so it stays a `zext` there (the bug survives). Exec test
//!     zig_convert_wide; tools/crosscheck/known/spirv_zig_signed_widen.zig.
//!     Every other module gets the specification's semantics;
//!   * anything else is `error.Unsupported` with the disassembled
//!     instruction in `Diagnostics.message`, malformed input included (operand
//!     types are checked before any IR is printed: `Fn.checkTypes`; block
//!     structure, phis and dominance: `Fn.validateBody`; constants against
//!     their types: `Lowerer.constText`; the fuzz test below mutates real
//!     modules). The call graph is walked without recursion
//!     (`Lowerer.funcInfo`), so helper chains of any depth lower. Stage 5 (graphics interfaces,
//!     images) and the stage-4 constructs Metal lacks are refused by name.

const std = @import("std");
const Allocator = std.mem.Allocator;
const reader = @import("reader.zig");
const dis = @import("dis.zig");
const manifest = @import("manifest.zig");
const metadata = @import("../air/metadata.zig");
const air_target = @import("../air/target.zig");
const spec = reader.spec;
const Module = reader.Module;
const Instruction = reader.Instruction;
const Id = reader.Id;
const air = metadata.air;

pub const Options = struct {
    /// Target triple of the text (the assembler sets its own per profile).
    profile: air_target.Profile = air_target.default,
    /// `[[buffer(n)]]` of the PushConstant block (Metal has no push
    /// constants; the host passes the block with setBytes).
    push_constant_index: u32 = 30,
    /// The library's fast-math declaration: picks `air.fast_<fn>` over
    /// `air.<fn>` for the functions this file names directly, as
    /// intrinsics.zig does for the `llvm.*` ones.
    fast_math: bool = false,
    /// Lower only the entry point of this name (the others, and whatever
    /// only they reach, are left out): attributes a refusal to one kernel of
    /// a many-kernel module (tools/spirv_probe.zig).
    only_entry: ?[]const u8 = null,
    /// Undo Zig's SPIR-V backend bugs in modules it generated (see the
    /// module doc, "producer workarounds"). Off, such a module gets the
    /// specification's semantics, which reproduces the bug on Metal.
    zig_workarounds: bool = true,
    /// Lower the constructs that Metal's compiler builds but computes
    /// wrongly (docs/spirv.md, "Metal findings made while mapping") with
    /// the specification's mapping instead of refusing them. Only the
    /// corpus driver sets it: it executes such a kernel to notice when a
    /// macOS release fixes the miscompile. Today: OpAtomicFAddEXT on
    /// threadgroup memory.
    allow_metal_miscompiles: bool = false,
};

/// The Generator tool id of Zig's self-hosted SPIR-V backend (the high half
/// of the header's generator word; `spec.zig_generator_id` in Zig's source).
pub const zig_generator_tool: u16 = 41;

pub const Diagnostics = struct {
    message: []const u8 = "",
    /// Lowering rounds `lower` needed (pointer alignment; see `lower`).
    rounds: u64 = 0,
};

pub const Error = error{ Unsupported, OutOfMemory };

pub const Lowered = struct {
    /// The whole module: target lines, threadgroup globals, every entry and
    /// helper `define`, the declarations.
    ir: []const u8,
    /// One per GLCompute / Kernel entry point, in OpEntryPoint order.
    kernels: []const manifest.Kernel,
};

/// Lower every compute entry point of `m`. Everything is allocated in `a`,
/// which should be an arena (nothing is freed individually).
pub fn lower(a: Allocator, m: *const Module, opts: Options, diag: ?*Diagnostics) Error!Lowered {
    // Pointer alignment flows backwards through phis (a back edge),
    // helper parameters (a caller lowered after the callee) and memory: a
    // round that learns a narrower alignment than it assumed somewhere
    // (`Learned`) is lowered again with that knowledge. Within a round
    // what is learned is used at once (`Lowerer.learnedAlign`, `floorOf`):
    // helpers are lowered after the caller that first reaches them and a
    // phi is rendered after the phis before it, so a fact crosses a whole
    // chain of helper calls or forward phis in one round (it used to cross
    // one link per round, and a 64-deep helper chain ran out of rounds).
    // Only a back edge or a later caller needs another round, and a
    // function's pointer phis are solved together at its end
    // (`Fn.settleAlign`), so loops nested to any depth need just the one
    // (each phi used to take its incoming values as the round had lowered
    // them, crossing one loop level per round). A non-final round narrows
    // at least one fact (a phi, a parameter or a floor; a power of two, so
    // at most 34 times each), so this ends; in practice after two rounds.
    var learned: Learned = .{};
    var round: u64 = 0;
    const max_rounds: u64 = 64 + 34 * @as(u64, m.instructions.len);
    while (true) : (round += 1) {
        var l: Lowerer = .{ .a = a, .m = m, .opts = opts, .diag = diag, .learned = learned };
        const out = try l.run();
        if (!try l.mergeLearned(&learned)) {
            if (diag) |d| d.rounds = round + 1;
            return out;
        }
        if (round == max_rounds) return l.fail("pointer alignments did not settle after {d} lowering rounds", .{max_rounds});
    }
}

/// What a lowering round learned about pointer alignment that it could not
/// know where it needed it: the alignment of pointer values that are not
/// access-chain results (phis, helper parameters, pointers read back from
/// memory or returned by a call), when less than the pointee's own.
/// Without it a load through such a pointer would claim the pointee's
/// natural alignment (16 for a uvec4) on an address that a packed member
/// leaves 4-aligned: LLVM undefined behaviour.
const Learned = struct {
    /// Phi results and helper parameters: the least over their incoming
    /// values and call-site arguments.
    ids: std.AutoHashMapUnmanaged(Id, u64) = .empty,
    /// By pointee memory type: the least alignment of any narrowed pointer
    /// to it in the module, for pointers whose origin is not tracked
    /// (loaded from memory, extracted from a composite, returned by a call).
    floors: std.StringHashMapUnmanaged(u64) = .empty,
};

const Repr = enum { value, memory };

/// `alignment` is what the host provides and what an access with no better
/// knowledge may assume (for an Offset-packed struct: its largest member's,
/// since the block is laid out for it); `abi` is LLVM's ABI alignment under
/// AIR's datalayout, which decides where a natural parent struct places the
/// type: a packed `<{ ... }>` has ABI alignment 1, so a natural `{ i32,
/// <{ [3 x i32], i32, <4 x i32> }> }` puts the second member at byte 4,
/// not at the 16 its members would want (exec tests packed_in_natural_*).
const Layout = struct { size: u64, alignment: u64, abi: u64 };

const StructLayout = struct {
    text: []const u8,
    /// LLVM member index of each SPIR-V member (padding shifts them).
    index: []const u32,
    /// Byte offset of each SPIR-V member.
    offsets: []const u64,
    /// AIR size of each SPIR-V member as stored (a `[3 x T]` member: 3 lanes).
    sizes: []const u64,
    /// LLVM text of each SPIR-V member as stored.
    member_tys: []const []const u8,
    /// Members stored in their tight form (`Member.tight`).
    tight: []const bool,
    size: u64,
    /// See `Layout`.
    alignment: u64,
    abi: u64,
    has_runtime_array: bool,
};

/// A struct member or array element type with its storage form.
const Member = struct {
    id: Id,
    /// Stored in its tight form, without the bytes AIR's layout would add
    /// at its end: a 3-element vector as `[3 x T]` (AIR allocates `<3 x T>`
    /// like four elements), a struct as `Lowerer.tightLayout`'s packed form
    /// (no tail padding, its last member tight in turn). Scalar block layout
    /// (and std430 for a vec3) puts the next member or array element right
    /// after the data: `{ vec3 v; float f @12 }`, `{ Inner i; float f @12 }`
    /// with `Inner { vec3 v; }`, `Inner[]` at ArrayStride 12. Loads and
    /// stores of a tight value convert at the boundary (`Fn.memberToValue`,
    /// `Fn.valueToMember`): a tight vector is still accessed as `<3 x T>`,
    /// whose store size is the three components, a tight struct in its
    /// packed form, so the neighbour is never written.
    tight: bool = false,
};

const FuncInfo = struct {
    /// Module-scope variables that become parameters (buffers, push
    /// constants, builtins, Private), reached directly or through calls, by id.
    iface: []const Id = &.{},
    /// Variables that are LLVM globals (Workgroup; UniformConstant with an
    /// initializer) reached.
    workgroup: []const Id = &.{},
    done: bool = false,
};

/// Decorations on an id that change nothing this lowering computes, so it
/// may ignore them: precision and fast-math permissions (RelaxedPrecision
/// allows less precision, NoContraction and FPFastMathMode are decided by
/// the library-level `--fast-math`), wrap flags (they only make overflow
/// undefined), aliasing, alignment and uniformity hints, and the interface
/// decorations lowerEntry reads itself. Volatile is listed because it is
/// implemented on the accesses rather than on the result
/// (`Lowerer.findVolatile`, `Fn.setVolatile`). Anything else on an
/// instruction result is either implemented where it matters
/// (SaturatedConversion, FPRoundingMode RTE) or refused
/// (`Fn.checkDecorations`).
fn ignorableDecoration(d: spec.Decoration) bool {
    return switch (d) {
        .relaxed_precision, .no_contraction, .fp_fast_math_mode, .no_signed_wrap, .no_unsigned_wrap => true,
        .non_uniform, .uniform, .uniform_id, .restrict, .aliased, .restrict_pointer, .aliased_pointer => true,
        .@"volatile", .coherent, .non_writable, .non_readable => true,
        .alignment, .alignment_id, .max_byte_offset, .max_byte_offset_id => true,
        .user_semantic, .counter_buffer => true,
        .spec_id, .block, .buffer_block, .offset, .array_stride, .matrix_stride, .row_major, .col_major => true,
        .binding, .descriptor_set, .built_in, .func_param_attr, .linkage_attributes => true,
        else => false,
    };
}

const Lowerer = struct {
    a: Allocator,
    m: *const Module,
    opts: Options,
    diag: ?*Diagnostics,
    context: []const u8 = "module",
    /// Whether the last refusal quoted an instruction (`failInst`).
    quoted: bool = false,
    globals: std.AutoHashMapUnmanaged(Id, reader.Global) = .empty,
    name_counts: std.StringHashMapUnmanaged(u32) = .empty,
    entry_names: std.StringHashMapUnmanaged(void) = .empty,
    layouts: std.AutoHashMapUnmanaged(Id, StructLayout) = .empty,
    /// Tight forms of struct types (`tightLayout`).
    tight_layouts: std.AutoHashMapUnmanaged(Id, StructLayout) = .empty,
    /// Struct types whose layout is being computed (a struct containing
    /// itself is malformed and would recurse forever).
    layout_busy: std.AutoHashMapUnmanaged(Id, void) = .empty,
    /// Nesting depth of `ty` / `layout` (a malformed type can contain itself).
    type_depth: u32 = 0,
    decls: std.StringArrayHashMapUnmanaged([]const u8) = .empty,
    /// Module-level LLVM globals: threadgroup variables and initialized
    /// UniformConstant tables, by variable id.
    global_defs: std.AutoArrayHashMapUnmanaged(Id, []const u8) = .empty,
    infos: std.AutoHashMapUnmanaged(Id, FuncInfo) = .empty,
    /// The shared allocation of the Workgroup Block variables, once made
    /// (`workgroupBlocks`).
    wg_blocks: ?WorkgroupBlocks = null,
    /// Values defined inside function bodies (`valueOwners`).
    owners: std.AutoHashMapUnmanaged(Id, void) = .empty,
    owners_done: bool = false,
    /// Results of module-scope instructions that are not values a function
    /// may use (`moduleNonValues`), with their opcode.
    non_values: std.AutoHashMapUnmanaged(Id, spec.Opcode) = .empty,
    non_values_done: bool = false,
    /// Every function by id (`function`).
    funcs: std.AutoHashMapUnmanaged(Id, *const reader.Function) = .empty,
    helpers: std.AutoArrayHashMapUnmanaged(Id, void) = .empty,
    /// Ids carrying a whole-id decoration that is not in
    /// `ignorableDecoration` (checked when the id is an instruction result).
    decorated: std.AutoHashMapUnmanaged(Id, void) = .empty,
    /// In a Zig module with the workarounds on: ids whose sign bit Zig's
    /// composite-integer widening broadcasts into the upper limbs
    /// (`findSignFills`); a widening OpUConvert of a signed operand that
    /// defines one is a sign extension (module doc, "producer workarounds").
    zig_sign_fills: std.AutoHashMapUnmanaged(Id, void) = .empty,
    /// Pointer values into Volatile memory (`findVolatile`): every load and
    /// store through one is `volatile`.
    volatile_ptrs: std.AutoHashMapUnmanaged(Id, void) = .empty,
    /// Struct members decorated Volatile (glslang's `volatile buffer`).
    volatile_members: std.AutoHashMapUnmanaged(reader.MemberKey, void) = .empty,
    /// Memo of `volatileType`.
    volatile_types: std.AutoHashMapUnmanaged(Id, bool) = .empty,
    /// Pointer alignment from the previous round (`lower`), read-only here.
    learned: Learned = .{},
    /// What this round learned (`learnId`, `noteFloor`).
    next: Learned = .{},
    /// Pointee types whose `learned.floors` entry this round consulted.
    floors_used: std.StringHashMapUnmanaged(void) = .empty,

    fn fail(l: *Lowerer, comptime fmt: []const u8, args: anytype) Error {
        if (l.diag) |d| {
            const msg = l.a.print(fmt, args) catch "out of memory";
            d.message = l.a.print("{s}: {s}", .{ l.context, msg }) catch msg;
        }
        l.quoted = false;
        return error.Unsupported;
    }

    /// A refusal raised by `fail` while `inst` was being lowered (a type or
    /// constant it uses cannot be printed) again with `inst` quoted, as
    /// every refusal inside a function body is.
    fn quote(l: *Lowerer, inst: Instruction) Error {
        if (l.quoted) return error.Unsupported;
        const d = l.diag orelse return error.Unsupported;
        const prefix_len = l.context.len + 2;
        const why = if (std.mem.startsWith(u8, d.message, l.context) and d.message.len >= prefix_len) d.message[prefix_len..] else d.message;
        return l.failInst(inst, "{s}", .{why});
    }

    /// `failInst` on the instruction that defines constant `id`.
    fn failConst(l: *Lowerer, id: Id, comptime fmt: []const u8, args: anytype) Error {
        for (l.m.instructions) |inst| if (inst.result == id) return l.failInst(inst, fmt, args);
        return l.fail("constant %{d}: " ++ fmt, .{id} ++ args);
    }

    fn failInst(l: *Lowerer, inst: Instruction, comptime fmt: []const u8, args: anytype) Error {
        if (l.diag) |d| {
            const text = dis.instructionText(l.a, l.m, inst) catch "?";
            const msg = l.a.print(fmt, args) catch "out of memory";
            d.message = l.a.print("{s}: `{s}`: {s}", .{ l.context, text, msg }) catch msg;
        }
        l.quoted = true;
        return error.Unsupported;
    }

    /// The pointee's natural alignment for pointer type `pt` (what an access
    /// assumes without better knowledge).
    fn naturalAlign(l: *Lowerer, pt: Id) Error!u64 {
        return (try l.layout((try l.pointerOf(pt)).pointee)).alignment;
    }

    /// The natural alignment (`naturalAlign`) of pointer value `id`'s
    /// pointee, or null, without a refusal, when it has none: a question
    /// the alignment fixpoint (`Fn.settleAlign`) asks of every pointer,
    /// whose own uses refuse what cannot be lowered.
    fn quietNatural(l: *Lowerer, id: Id) ?u64 {
        const pt = l.m.resultTypeOf(id) orelse (l.globals.get(id) orelse return null).type;
        const saved = if (l.diag) |d| d.message else "";
        const quoted = l.quoted;
        return l.naturalAlign(pt) catch {
            if (l.diag) |d| d.message = saved;
            l.quoted = quoted;
            return null;
        };
    }

    /// Pointer `id` (of pointer type `pt`) is aligned to at least `al`
    /// wherever it comes from: kept for the next round when narrower than
    /// natural.
    fn learnId(l: *Lowerer, id: Id, pt: Id, al: u64) Error!void {
        if (al >= try l.naturalAlign(pt)) return;
        const gop = try l.next.ids.getOrPut(l.a, id);
        if (!gop.found_existing or al < gop.value_ptr.*) gop.value_ptr.* = al;
        try l.noteFloor(pt, al);
    }

    /// Some pointer of type `pt` is only `al`-aligned: lowers the floor of
    /// every pointer to that pointee whose origin is not tracked.
    fn noteFloor(l: *Lowerer, pt: Id, al: u64) Error!void {
        if (al >= try l.naturalAlign(pt)) return;
        const key = try l.ty((try l.pointerOf(pt)).pointee, .memory);
        const gop = try l.next.floors.getOrPut(l.a, key);
        if (!gop.found_existing or al < gop.value_ptr.*) gop.value_ptr.* = al;
    }

    /// The floor for pointers of type `pt` (`Learned.floors`): the previous
    /// round's, narrowed by what this round has learned so far.
    fn floorOf(l: *Lowerer, pt: Id) Error!?u64 {
        const key = try l.ty((try l.pointerOf(pt)).pointee, .memory);
        try l.floors_used.put(l.a, key, {});
        return minOpt(l.learned.floors.get(key), l.next.floors.get(key));
    }

    /// What phi or parameter `id` is known to be aligned to: the previous
    /// round's knowledge narrowed by this round's so far (`lower`).
    fn learnedAlign(l: *Lowerer, id: Id) ?u64 {
        return minOpt(l.learned.ids.get(id), l.next.ids.get(id));
    }

    /// Merge `next` into `learned`; whether this round assumed more than it
    /// learned (a phi, parameter or consulted floor got narrower), so that
    /// another round must lower with the merged knowledge.
    fn mergeLearned(l: *Lowerer, learned: *Learned) Error!bool {
        var changed = false;
        var merged: Learned = .{ .ids = try learned.ids.clone(l.a), .floors = try learned.floors.clone(l.a) };
        var it = l.next.ids.iterator();
        while (it.next()) |e| {
            const gop = try merged.ids.getOrPut(l.a, e.key_ptr.*);
            if (!gop.found_existing or e.value_ptr.* < gop.value_ptr.*) {
                gop.value_ptr.* = e.value_ptr.*;
                changed = true;
            }
        }
        var fit = l.next.floors.iterator();
        while (fit.next()) |e| {
            const gop = try merged.floors.getOrPut(l.a, e.key_ptr.*);
            if (!gop.found_existing or e.value_ptr.* < gop.value_ptr.*) {
                gop.value_ptr.* = e.value_ptr.*;
                if (l.floors_used.contains(e.key_ptr.*)) changed = true;
            }
        }
        learned.* = merged;
        return changed;
    }

    fn run(l: *Lowerer) Error!Lowered {
        const m = l.m;
        for (m.globals) |g| try l.globals.put(l.a, g.id, g);
        for (m.decorations) |d| {
            if (d.member != null) continue;
            if (d.as()) |k| if (ignorableDecoration(k)) continue;
            try l.decorated.put(l.a, d.target, {});
        }
        if (l.zigWorkarounds()) try l.findSignFills();
        try l.findVolatile();
        var it = m.names.valueIterator();
        while (it.next()) |n| {
            const gop = try l.name_counts.getOrPut(l.a, n.*);
            gop.value_ptr.* = if (gop.found_existing) gop.value_ptr.* + 1 else 1;
        }
        for (m.entry_points) |ep| {
            const model = ep.model();
            const wanted = if (l.opts.only_entry) |name| std.mem.eql(u8, name, ep.name) else true;
            if (wanted and model != .gl_compute and model != .kernel) {
                l.context = try l.a.print("entry point '{s}'", .{ep.name});
                return l.fail("execution model {s} is not a compute model (stage 5: vertex and fragment entry points)", .{if (model) |md| @tagName(md) else "unknown"});
            }
            // The entry name is the library's function name, printed as
            // given; `llvm.` and `air.` names are the intrinsics' (the
            // assembler would take the kernel for a declaration of one).
            if (std.mem.startsWith(u8, ep.name, "llvm.") or std.mem.startsWith(u8, ep.name, "air.")) {
                l.context = try l.a.print("entry point '{s}'", .{ep.name});
                return l.fail("an entry point name starting with 'llvm.' or 'air.' is reserved for LLVM and Metal intrinsics", .{});
            }
            const gop = try l.entry_names.getOrPut(l.a, ep.name);
            if (gop.found_existing) return l.fail("two entry points are named '{s}'", .{ep.name});
        }
        if (m.entry_points.len == 0) return l.fail("the module has no GLCompute or Kernel entry point", .{});
        if (l.opts.only_entry) |name| {
            if (!l.entry_names.contains(name)) return l.fail("no entry point is named '{s}'", .{name});
        }

        var defines: std.ArrayList(u8) = .empty;
        var kernels: std.ArrayList(manifest.Kernel) = .empty;
        for (m.entry_points) |ep| {
            if (l.opts.only_entry) |name| if (!std.mem.eql(u8, name, ep.name)) continue;
            try kernels.append(l.a, try l.lowerEntry(ep, &defines));
        }
        var i: usize = 0;
        while (i < l.helpers.count()) : (i += 1) try l.lowerHelper(l.helpers.keys()[i], &defines);

        var out: std.ArrayList(u8) = .empty;
        try out.print(l.a, "target datalayout = \"{s}\"\ntarget triple = \"{s}\"\n\n", .{ air_target.datalayout, l.opts.profile.triple });
        for (l.global_defs.values()) |d| try out.print(l.a, "{s}\n", .{d});
        if (l.global_defs.count() > 0) try out.append(l.a, '\n');
        try out.appendSlice(l.a, defines.items);
        for (l.decls.values()) |d| try out.print(l.a, "{s}\n", .{d});
        return .{ .ir = out.items, .kernels = kernels.items };
    }

    /// Whether the producer workarounds apply: they are on and the header's
    /// Generator tool is Zig's self-hosted backend.
    fn zigWorkarounds(l: *Lowerer) bool {
        return l.opts.zig_workarounds and l.m.generator >> 16 == zig_generator_tool;
    }

    /// Records in `zig_sign_fills` every id `%x` that Zig's composite-integer
    /// widening sign-fills from: `%s = OpBitcast %iW %x` followed by
    /// `OpShiftRightArithmetic %iW %s %c` with the constant `c` = W - 1, the
    /// sequence `airIntCast` emits (src/codegen/spirv/CodeGen.zig:6111-6131
    /// at f2f0d97a9b) to fill the upper limbs of an i128 (or wider) from
    /// limb 0 when the source is signed, after converting the source into
    /// limb 0 with buildConvert's OpUConvert (CodeGen.zig:6078).
    fn findSignFills(l: *Lowerer) Error!void {
        var casts: std.AutoHashMapUnmanaged(Id, Id) = .empty;
        for (l.m.instructions) |inst| {
            if (inst.op != .OpBitcast) continue;
            const r = inst.result orelse continue;
            try casts.put(l.a, r, inst.idAt(0) orelse continue);
        }
        for (l.m.instructions) |inst| {
            if (inst.op != .OpShiftRightArithmetic) continue;
            const int = switch (l.m.typeOf(inst.result_type orelse continue) orelse continue) {
                .int => |i| i,
                else => continue,
            };
            if (!int.signed or int.width == 0) continue;
            const amount = l.m.constantBits(inst.idAt(1) orelse continue) orelse continue;
            if (amount != int.width - 1) continue;
            const limb = casts.get(inst.idAt(0) orelse continue) orelse continue;
            try l.zig_sign_fills.put(l.a, limb, {});
        }
    }

    /// Records in `volatile_ptrs` every pointer value that points into
    /// memory SPIR-V declares Volatile: a variable, parameter or result
    /// decorated Volatile, an access chain through a struct member
    /// decorated Volatile (glslang's `volatile buffer` / `coherent volatile
    /// buffer` decorates every member), and whatever is derived from those
    /// (further access chains, OpCopyObject, pointer casts, OpSelect, OpPhi
    /// and helper parameters the pointer is passed to), by a worklist over
    /// the module, so the derivation order does not matter. SPIR-V: a
    /// Volatile access "cannot be eliminated, duplicated, or combined with
    /// other accesses". A plain `load` lets Metal hoist a polling read out
    /// of its loop (a spin-wait on a flag another SIMD-group sets never
    /// saw the store: exec tests volatile_spin_wait,
    /// volatile_member_spin_wait); `load volatile` re-reads memory.
    fn findVolatile(l: *Lowerer) Error!void {
        const m = l.m;
        var work: std.ArrayList(Id) = .empty;
        for (m.decorations) |d| {
            if (d.decoration != @backingInt(spec.Decoration.@"volatile")) continue;
            if (d.member) |k| {
                try l.volatile_members.put(l.a, .{ .id = d.target, .member = k }, {});
            } else if ((try l.volatile_ptrs.getOrPut(l.a, d.target)).found_existing == false) try work.append(l.a, d.target);
        }
        // Derivation edges: source pointer -> derived pointer.
        var users: std.AutoHashMapUnmanaged(Id, std.ArrayList(Id)) = .empty;
        const addUse = struct {
            fn add(a: Allocator, map: *std.AutoHashMapUnmanaged(Id, std.ArrayList(Id)), from: Id, to: Id) Error!void {
                const gop = try map.getOrPut(a, from);
                if (!gop.found_existing) gop.value_ptr.* = .empty;
                try gop.value_ptr.append(a, to);
            }
        }.add;
        for (m.instructions) |inst| {
            switch (inst.op) {
                .OpAccessChain, .OpInBoundsAccessChain, .OpPtrAccessChain, .OpInBoundsPtrAccessChain => {
                    const r = inst.result orelse continue;
                    const base = inst.idAt(0) orelse continue;
                    try addUse(l.a, &users, base, r);
                    if (l.chainHitsVolatileMember(inst, base)) {
                        if (!(try l.volatile_ptrs.getOrPut(l.a, r)).found_existing) try work.append(l.a, r);
                    }
                },
                .OpCopyObject, .OpBitcast, .OpPtrCastToGeneric, .OpGenericCastToPtr, .OpGenericCastToPtrExplicit => {
                    const r = inst.result orelse continue;
                    try addUse(l.a, &users, inst.idAt(0) orelse continue, r);
                },
                .OpSelect => {
                    const r = inst.result orelse continue;
                    for (1..3) |i| try addUse(l.a, &users, inst.idAt(i) orelse continue, r);
                },
                .OpPhi => {
                    const r = inst.result orelse continue;
                    var i: usize = 0;
                    while (i < inst.operands.len) : (i += 2) try addUse(l.a, &users, inst.idAt(i) orelse continue, r);
                },
                .OpFunctionCall => {
                    const callee = l.function(inst.idAt(0) orelse continue) orelse continue;
                    for (inst.operands[1..], 0..) |o, k| {
                        if (k >= callee.params.len) break;
                        try addUse(l.a, &users, o.asId() orelse continue, callee.params[k].id);
                    }
                },
                else => {},
            }
        }
        while (work.pop()) |id| {
            const us = users.get(id) orelse continue;
            for (us.items) |u| {
                if (!(try l.volatile_ptrs.getOrPut(l.a, u)).found_existing) try work.append(l.a, u);
            }
        }
    }

    /// Whether access chain `inst` (on `base`) steps through a struct member
    /// decorated Volatile.
    fn chainHitsVolatileMember(l: *Lowerer, inst: Instruction, base: Id) bool {
        if (l.volatile_members.count() == 0) return false;
        const pt = l.m.resultTypeOf(base) orelse return false;
        var cur = switch (l.m.typeOf(pt) orelse return false) {
            .pointer => |p| p.pointee,
            else => return false,
        };
        const skip: usize = switch (inst.op) {
            .OpPtrAccessChain, .OpInBoundsPtrAccessChain => 2,
            else => 1,
        };
        if (inst.operands.len < skip) return false;
        for (inst.operands[skip..], 0..) |o, n| {
            if (n > 64) return false;
            const iid = o.asId() orelse return false;
            switch (l.m.typeOf(cur) orelse return false) {
                .@"struct" => |st| {
                    const k = l.m.constantBits(iid) orelse return false;
                    if (k >= st.members.len) return false;
                    if (l.volatile_members.contains(.{ .id = cur, .member = @intCast(k) })) return true;
                    cur = st.members[@intCast(k)];
                },
                .array => |arr| cur = arr.elem,
                .runtime_array => |arr| cur = arr.elem,
                .matrix => |mt| cur = mt.column,
                else => return false,
            }
        }
        return false;
    }

    /// Whether a value of type `t` holds a struct member decorated Volatile
    /// (a whole-struct load or store of a `volatile buffer` block reads or
    /// writes Volatile memory).
    fn volatileType(l: *Lowerer, t: Id) Error!bool {
        if (l.volatile_members.count() == 0) return false;
        if (l.volatile_types.get(t)) |v| return v;
        try l.volatile_types.put(l.a, t, false);
        const v = switch (l.m.typeOf(t) orelse return false) {
            .@"struct" => |st| blk: {
                for (st.members, 0..) |mt, k| {
                    if (l.volatile_members.contains(.{ .id = t, .member = @intCast(k) })) break :blk true;
                    if (try l.volatileType(mt)) break :blk true;
                }
                break :blk false;
            },
            .array => |arr| try l.volatileType(arr.elem),
            .runtime_array => |arr| try l.volatileType(arr.elem),
            else => false,
        };
        try l.volatile_types.put(l.a, t, v);
        return v;
    }

    // ── Names ───────────────────────────────────────────────────────────────

    /// The OpName of `id` when it can serve as an LLVM identifier: unique in
    /// the module, a plain identifier, and not one of the spellings this file
    /// generates (`v12` values, `t3` temporaries, `b7` blocks, `a0` and `i9`
    /// parameters, `f4`/`g5` globals) or reserves
    /// (`__N` is the assembler's, `llvm.`/`air.` are intrinsics).
    fn usableName(l: *Lowerer, id: Id) ?[]const u8 {
        const n = l.m.name(id) orelse return null;
        if (!metadata.isPlainIdentifier(n)) return null;
        if (std.mem.startsWith(u8, n, "__") or std.mem.startsWith(u8, n, "llvm.") or std.mem.startsWith(u8, n, "air.")) return null;
        if (n.len >= 2 and std.mem.findScalar(u8, "vtbaifg", n[0]) != null) {
            var digits = true;
            for (n[1..]) |ch| if (!std.ascii.isDigit(ch)) {
                digits = false;
            };
            if (digits) return null;
        }
        if ((l.name_counts.get(n) orelse 0) != 1) return null;
        return n;
    }

    fn localName(l: *Lowerer, id: Id) Error![]const u8 {
        if (l.usableName(id)) |n| return n;
        return l.a.print("v{d}", .{id});
    }

    /// `@name` of a helper or a threadgroup global (`prefix` + id when the
    /// OpName is unusable or names an entry point). Entry points keep their
    /// OpEntryPoint name verbatim, so a generated name an entry point
    /// already has (a kernel called "g25" beside Workgroup variable %25)
    /// gets `.N` appended, the smallest N that is neither an entry name nor
    /// any OpName (`usableName` keeps OpNames off the bare `g<N>` / `f<N>`
    /// spellings): a symbol defined twice made the assembler panic or
    /// refuse the call.
    fn globalSym(l: *Lowerer, id: Id, prefix: []const u8) Error![]const u8 {
        if (l.usableName(id)) |n| {
            if (!l.entry_names.contains(n)) return n;
        }
        const base = try l.a.print("{s}{d}", .{ prefix, id });
        if (!l.entry_names.contains(base)) return base;
        var k: u64 = 0;
        while (true) : (k += 1) {
            const s = try l.a.print("{s}.{d}", .{ base, k });
            if (!l.entry_names.contains(s) and !l.name_counts.contains(s)) return s;
        }
    }

    fn entrySym(l: *Lowerer, name: []const u8) Error![]const u8 {
        var out: std.ArrayList(u8) = .empty;
        try metadata.appendGlobalName(l.a, &out, name);
        return out.items;
    }

    // ── Types and layout ────────────────────────────────────────────────────

    /// Every type the lowering prints, sizes or names passes through here,
    /// so a float with an FPEncoding (bfloat16, float8; reader.zig keeps
    /// the operand) is refused before anything could treat its bits as an
    /// IEEE half: no rule for it has been verified on Metal's compiler.
    fn typeOf(l: *Lowerer, id: Id) Error!reader.Type {
        const t = l.m.typeOf(id) orelse return l.fail("%{d} is not a type", .{id});
        switch (t) {
            .float => |f| if (f.encoding) |e| {
                const name = if (reader.enumerant(.fp_encoding, e)) |en| en.name else "an unknown encoding";
                return l.fail("float type %{d} has FPEncoding {s} ({d}-bit, not IEEE 754); only IEEE half and float are supported", .{ id, name, f.width });
            },
            else => {},
        }
        return t;
    }

    /// AIR address space of a storage class; null for Input (no pointer).
    fn addrSpace(l: *Lowerer, sc: u32) Error!?u32 {
        const s = std.enums.fromInt(spec.StorageClass, sc) orelse return l.fail("unknown storage class {d}", .{sc});
        return switch (s) {
            .function, .private => 0,
            .storage_buffer, .physical_storage_buffer, .cross_workgroup, .generic => 1,
            .uniform, .push_constant, .uniform_constant => 2,
            .workgroup => 3,
            .input => null,
            else => l.fail("storage class {t} has no Metal address space here (stage 5: graphics interfaces, images)", .{s}),
        };
    }

    fn ptrText(l: *Lowerer, sc: u32) Error![]const u8 {
        const as = try l.addrSpace(sc) orelse return l.fail("an Input pointer is usable only through OpLoad / OpAccessChain of its builtin variable", .{});
        return l.spaceText(as);
    }

    fn spaceText(l: *Lowerer, as: u32) Error![]const u8 {
        return if (as == 0) "ptr" else try l.a.print("ptr addrspace({d})", .{as});
    }

    const PointerType = @FieldType(reader.Type, "pointer");

    fn pointerOf(l: *Lowerer, t: Id) Error!PointerType {
        return switch (try l.typeOf(t)) {
            .pointer => |p| p,
            else => l.fail("type %{d} is not a pointer type", .{t}),
        };
    }

    fn isClass(sc: u32, want: spec.StorageClass) bool {
        return sc == @backingInt(want);
    }

    /// Whether `pointee` is a BufferBlock struct: SPIR-V 1.0-1.2's storage
    /// buffer, a Uniform variable of a BufferBlock-decorated struct (glslang
    /// writes every SSBO so for Vulkan 1.0/1.1). It is device memory,
    /// writable unless NonWritable, although its pointers say Uniform.
    fn isBufferBlock(l: *Lowerer, pointee: Id) bool {
        return l.m.hasDecoration(pointee, .buffer_block);
    }

    /// AIR address space of a module-scope variable (null for Input).
    fn globalSpace(l: *Lowerer, g: Id) Error!?u32 {
        const glob = l.globals.get(g).?;
        const pt = try l.pointerOf(glob.type);
        if (isClass(pt.storage_class, .uniform) and l.isBufferBlock(pt.pointee)) return 1;
        return l.addrSpace(pt.storage_class);
    }

    /// `ptr addrspace(N)` of a module-scope variable.
    fn globalPtrText(l: *Lowerer, g: Id) Error![]const u8 {
        const as = try l.globalSpace(g) orelse return l.fail("an Input pointer is usable only through OpLoad / OpAccessChain of its builtin variable", .{});
        return l.spaceText(as);
    }

    /// A variable that is an LLVM global rather than a parameter:
    /// Workgroup (threadgroup memory) and an initialized UniformConstant
    /// (OpenCL's program-scope `__constant` table).
    fn isModuleGlobal(l: *Lowerer, g: Id) bool {
        const glob = l.globals.get(g) orelse return false;
        if (isClass(glob.storage_class, .workgroup)) return true;
        return isClass(glob.storage_class, .uniform_constant) and glob.initializer != null;
    }

    fn ty(l: *Lowerer, t: Id, repr: Repr) Error![]const u8 {
        if (l.type_depth > 64) return l.fail("type %{d} nests deeper than 64 levels (does it contain itself?)", .{t});
        l.type_depth += 1;
        defer l.type_depth -= 1;
        switch (try l.typeOf(t)) {
            .void => return "void",
            .bool => return if (repr == .value) "i1" else "i8",
            .int => |i| return switch (i.width) {
                8, 16, 32, 64 => l.a.print("i{d}", .{i.width}),
                else => l.fail("{d}-bit integers are not supported", .{i.width}),
            },
            .float => |f| return switch (f.width) {
                16 => "half",
                32 => "float",
                64 => l.fail("64-bit floats: Metal has no double", .{}),
                else => l.fail("{d}-bit floats are not supported", .{f.width}),
            },
            .vector => |v| {
                switch (try l.typeOf(v.elem)) {
                    .bool, .int, .float => {},
                    else => return l.fail("vector type %{d} has a non-scalar component type", .{t}),
                }
                // SPIR-V vectors have 2, 3 or 4 components, 8 or 16 with
                // the Vector16 capability.
                if (v.len < 2 or (v.len > 4 and v.len != 8 and v.len != 16)) return l.fail("vector type %{d} has {d} components", .{ t, v.len });
                return l.a.print("<{d} x {s}>", .{ v.len, try l.ty(v.elem, repr) });
            },
            .matrix => |mt| {
                if (l.vecLen(mt.column) == null) return l.fail("matrix type %{d}: its column type is not a vector", .{t});
                return l.a.print("[{d} x {s}]", .{ mt.cols, try l.ty(mt.column, .memory) });
            },
            .array => |arr| {
                _ = try l.layout(t);
                return l.a.print("[{d} x {s}]", .{ try l.arrayLen(arr.len), try l.memberText(try l.elemOf(t)) });
            },
            .runtime_array => {
                _ = try l.layout(t);
                return l.a.print("[0 x {s}]", .{try l.memberText(try l.elemOf(t))});
            },
            .@"struct" => return (try l.structLayout(t)).text,
            .pointer => |p| return l.ptrText(p.storage_class),
            .function => return l.fail("a function type is not a value type", .{}),
            .forward_pointer => return l.fail("OpTypeForwardPointer is not supported", .{}),
            .other => |op| return l.fail("{t} is not supported (stage 5: images, samplers)", .{op}),
        }
    }

    fn arrayLen(l: *Lowerer, len_id: Id) Error!u64 {
        return l.m.constantBits(len_id) orelse l.fail("array length %{d} is not an integer constant", .{len_id});
    }

    fn isBoolish(l: *Lowerer, t: Id) bool {
        const tt = l.m.typeOf(t) orelse return false;
        return switch (tt) {
            .bool => true,
            .vector => |v| (l.m.typeOf(v.elem) orelse return false) == .bool,
            else => false,
        };
    }

    /// The element of an array or runtime array as it is stored: tight
    /// (`Member.tight`) when the ArrayStride is the element's tight size
    /// rather than its AIR size: three components for a 3-element vector
    /// (scalar block layout `vec3[]`), the tight size of a struct ending in
    /// one (`Inner[]` at stride 12 for `Inner { vec3 v; }`).
    fn elemOf(l: *Lowerer, t: Id) Error!Member {
        const elem = switch (try l.typeOf(t)) {
            .array => |arr| arr.elem,
            .runtime_array => |ra| ra.elem,
            else => return l.fail("type %{d} is not an array", .{t}),
        };
        const d = l.m.decoration(t, .array_stride) orelse return .{ .id = elem };
        const stride = d.firstInt() orelse return .{ .id = elem };
        switch (try l.typeOf(elem)) {
            .vector => |v| {
                if (v.len != 3) return .{ .id = elem };
                const lane: u64 = switch (try l.typeOf(v.elem)) {
                    .bool => 1,
                    .int => |i| i.width / 8,
                    .float => |fl| fl.width / 8,
                    else => return .{ .id = elem },
                };
                return .{ .id = elem, .tight = lane != 0 and stride == 3 * lane };
            },
            .@"struct" => {
                if (stride == (try l.layout(elem)).size) return .{ .id = elem };
                const tl = try l.tightLayout(elem) orelse return .{ .id = elem };
                return .{ .id = elem, .tight = tl.size == stride };
            },
            else => return .{ .id = elem },
        }
    }

    /// Whether `t` has a tight form smaller than its AIR size
    /// (`Member.tight`): a 3-element vector, or a struct with a tight
    /// layout.
    fn hasTight(l: *Lowerer, t: Id) Error!bool {
        return switch (try l.typeOf(t)) {
            .vector => |v| v.len == 3,
            .@"struct" => (try l.tightLayout(t)) != null,
            else => false,
        };
    }

    /// The tight form of struct `t` (`Member.tight`): the members at their
    /// offsets in a packed `<{ ... }>` with explicit padding and no tail
    /// padding, the last member in its own tight form; null when that is
    /// not smaller than the struct's layout (or the struct is CPacked,
    /// whose members keep their AIR allocation). Its `alignment` is the
    /// struct's, its ABI alignment 1.
    fn tightLayout(l: *Lowerer, t: Id) Error!?StructLayout {
        if (l.tight_layouts.get(t)) |tl| return tl;
        const sl = try l.structLayout(t);
        const members = (try l.typeOf(t)).@"struct".members;
        const n = members.len;
        if (n == 0 or l.m.hasDecoration(t, .c_packed)) return null;
        const last = n - 1;
        const last_tight = sl.tight[last] or try l.hasTight(members[last]);
        if (!last_tight) return null;
        const lly = try l.memberLayout(.{ .id = members[last], .tight = true });
        const size = sl.offsets[last] + lly.size;
        if (size >= sl.size) return null;
        const index = try l.a.alloc(u32, n);
        const sizes = try l.a.dupe(u64, sl.sizes);
        const tys = try l.a.dupe([]const u8, sl.member_tys);
        const tight = try l.a.dupe(bool, sl.tight);
        tight[last] = true;
        tys[last] = try l.memberText(.{ .id = members[last], .tight = true });
        sizes[last] = lly.size;
        var text: std.ArrayList(u8) = .empty;
        try text.appendSlice(l.a, "<{");
        var pos: u64 = 0;
        var k: u32 = 0;
        for (0..n) |i| {
            const off = sl.offsets[i];
            if (off > pos) {
                try text.print(l.a, "{s} [{d} x i8]", .{ if (k == 0) "" else ",", off - pos });
                k += 1;
            }
            try text.print(l.a, "{s} {s}", .{ if (k == 0) "" else ",", tys[i] });
            index[i] = k;
            k += 1;
            pos = off + sizes[i];
        }
        try text.appendSlice(l.a, " }>");
        const tl: StructLayout = .{ .text = text.items, .index = index, .offsets = sl.offsets, .sizes = sizes, .member_tys = tys, .tight = tight, .size = size, .alignment = sl.alignment, .abi = 1, .has_runtime_array = false };
        try l.tight_layouts.put(l.a, t, tl);
        return tl;
    }

    /// The layout of a struct member in its storage form.
    fn memberStructLayout(l: *Lowerer, mb: Member) Error!StructLayout {
        if (mb.tight) return (try l.tightLayout(mb.id)) orelse l.fail("struct %{d} has no tight form", .{mb.id});
        return l.structLayout(mb.id);
    }

    /// The LLVM text of a struct member or array element as it is stored.
    fn memberText(l: *Lowerer, mb: Member) Error![]const u8 {
        if (!mb.tight) return l.ty(mb.id, .memory);
        if (l.vecLen(mb.id) != null) return l.a.print("[3 x {s}]", .{try l.ty(l.scalarOf(mb.id), .memory)});
        return (try l.memberStructLayout(mb)).text;
    }

    fn memberLayout(l: *Lowerer, mb: Member) Error!Layout {
        if (!mb.tight) return l.layout(mb.id);
        if (l.vecLen(mb.id) != null) {
            const e = try l.layout(l.scalarOf(mb.id));
            return .{ .size = 3 * e.size, .alignment = e.alignment, .abi = e.abi };
        }
        const tl = try l.memberStructLayout(mb);
        return .{ .size = tl.size, .alignment = tl.alignment, .abi = tl.abi };
    }

    fn mulSize(l: *Lowerer, t: Id, x: u64, y: u64) Error!u64 {
        const r = std.math.mul(u64, x, y) catch return l.fail("type %{d} is larger than 2^64 bytes", .{t});
        if (r > std.math.maxInt(i32)) return l.fail("type %{d} occupies {d} bytes; AIR's metadata describes at most 2 GiB", .{ t, r });
        return r;
    }

    /// Allocation size and ABI alignment under AIR's datalayout (vectors:
    /// alignment = size rounded up to a power of two, `v96:128:128`).
    fn layout(l: *Lowerer, t: Id) Error!Layout {
        if (l.type_depth > 64) return l.fail("type %{d} nests deeper than 64 levels (does it contain itself?)", .{t});
        l.type_depth += 1;
        defer l.type_depth -= 1;
        switch (try l.typeOf(t)) {
            .bool => return .{ .size = 1, .alignment = 1, .abi = 1 },
            .int, .float => {
                _ = try l.ty(t, .memory);
                const w: u64 = l.m.scalarWidth(t).? / 8;
                return .{ .size = w, .alignment = w, .abi = w };
            },
            .vector => |v| {
                _ = try l.ty(t, .memory);
                const e = try l.layout(v.elem);
                const bytes = e.size * v.len;
                if (bytes > 1 << 20) return l.fail("vector type %{d} has {d} components", .{ t, v.len });
                const al = std.math.ceilPowerOfTwoAssert(u64, bytes);
                return .{ .size = std.mem.alignForward(u64, bytes, al), .alignment = al, .abi = al };
            },
            .matrix => |mt| {
                _ = try l.ty(t, .memory);
                const c = try l.layout(mt.column);
                return .{ .size = try l.mulSize(t, c.size, mt.cols), .alignment = c.alignment, .abi = c.abi };
            },
            .array => |arr| {
                try l.checkTypeDecorations(t);
                const e = try l.memberLayout(try l.elemOf(t));
                try l.checkStride(t, e.size);
                return .{ .size = try l.mulSize(t, e.size, try l.arrayLen(arr.len)), .alignment = e.alignment, .abi = e.abi };
            },
            .runtime_array => {
                try l.checkTypeDecorations(t);
                const e = try l.memberLayout(try l.elemOf(t));
                try l.checkStride(t, e.size);
                return .{ .size = 0, .alignment = e.alignment, .abi = e.abi };
            },
            .@"struct" => {
                const s = try l.structLayout(t);
                return .{ .size = s.size, .alignment = s.alignment, .abi = s.abi };
            },
            .pointer => return .{ .size = 8, .alignment = 8, .abi = 8 },
            else => return l.fail("type %{d} has no memory layout", .{t}),
        }
    }

    fn checkStride(l: *Lowerer, t: Id, elem_size: u64) Error!void {
        if (l.m.decoration(t, .array_stride)) |d| {
            const stride = d.firstInt() orelse 0;
            if (stride != elem_size) return l.fail("array type %{d} has ArrayStride {d} but its element occupies {d} bytes in AIR; strided arrays are not supported", .{ t, stride, elem_size });
        }
    }

    /// Refuse a decoration on a struct or array type, or on a struct
    /// member, that changes the type's meaning and is not implemented here:
    /// the layout ones this file reads (Offset, ArrayStride, MatrixStride,
    /// ColMajor, CPacked; RowMajor is refused where it is read), Block /
    /// BufferBlock, and the hints `ignorableDecoration` lists pass; anything
    /// else (GLSLShared / GLSLPacked's implementation-defined layouts, ...)
    /// is refused by name rather than laid out naturally without notice.
    fn checkTypeDecorations(l: *Lowerer, t: Id) Error!void {
        for (l.m.decorations) |d| {
            if (d.target != t) continue;
            const k = d.as() orelse return l.fail("type %{d} carries decoration {d}, which is not supported", .{ t, d.decoration });
            if (ignorableDecoration(k)) continue;
            if (k == .c_packed and d.member == null and (l.m.typeOf(t) orelse continue) == .@"struct") continue;
            if (d.member) |mi| return l.fail("struct %{d} member {d} carries decoration {t}, which is not supported", .{ t, mi, k });
            return l.fail("type %{d} carries decoration {t}, which is not supported", .{ t, k });
        }
    }

    fn structLayout(l: *Lowerer, t: Id) Error!StructLayout {
        if (l.layouts.get(t)) |s| return s;
        const members = switch (try l.typeOf(t)) {
            .@"struct" => |st| st.members,
            else => return l.fail("%{d} is used as a struct type but is not one", .{t}),
        };
        if (l.layout_busy.contains(t)) return l.fail("struct %{d} contains itself", .{t});
        try l.layout_busy.put(l.a, t, {});
        defer _ = l.layout_busy.remove(t);
        try l.checkTypeDecorations(t);
        const n = members.len;
        const tys = try l.a.alloc([]const u8, n);
        const lays = try l.a.alloc(Layout, n);
        const tight = try l.a.alloc(bool, n);
        @memset(tight, false);
        var runtime = false;
        for (members, 0..) |mt, i| {
            const mtt = try l.typeOf(mt);
            if (mtt == .runtime_array) {
                if (i != n - 1) return l.fail("struct %{d}: a runtime array must be the last member", .{t});
                runtime = true;
            }
            if (l.m.memberDecoration(t, @intCast(i), .row_major) != null) return l.fail("struct %{d} member {d}: RowMajor matrices are not supported", .{ t, i });
            if (l.m.memberDecoration(t, @intCast(i), .matrix_stride)) |d| {
                var inner = mt;
                var depth: u32 = 0;
                while (depth < 64) : (depth += 1) switch (try l.typeOf(inner)) {
                    .array => |arr| inner = arr.elem,
                    .runtime_array => |ra| inner = ra.elem,
                    else => break,
                };
                const col = switch (try l.typeOf(inner)) {
                    .matrix => |mx| mx.column,
                    else => return l.fail("struct %{d} member {d}: MatrixStride on a non-matrix", .{ t, i }),
                };
                const cl = try l.layout(col);
                if ((d.firstInt() orelse 0) != cl.size) return l.fail("struct %{d} member {d}: MatrixStride {d} differs from the column's AIR size {d}", .{ t, i, d.firstInt() orelse 0, cl.size });
            }
            tys[i] = try l.ty(mt, .memory);
            lays[i] = try l.layout(mt);
        }
        // AIR's natural layout: every member at the next multiple of its
        // LLVM ABI alignment (`Layout.abi`: 1 for a packed member).
        const offsets = try l.a.alloc(u64, n);
        var cur: u64 = 0;
        var abi: u64 = 1;
        var al: u64 = 1;
        for (lays, 0..) |ly, i| {
            const off = std.mem.alignForward(u64, cur, ly.abi);
            offsets[i] = off;
            cur = off + ly.size;
            abi = @max(abi, ly.abi);
            al = @max(al, ly.alignment);
        }
        var decorated: usize = 0;
        var matches = true;
        for (0..n) |i| {
            if (l.m.memberDecoration(t, @intCast(i), .offset)) |d| {
                decorated += 1;
                if ((d.firstInt() orelse 0) != offsets[i]) matches = false;
            }
        }
        if (decorated != 0 and decorated != n) return l.fail("struct %{d}: only some members carry an Offset decoration", .{t});
        const index = try l.a.alloc(u32, n);
        const sizes = try l.a.alloc(u64, n);
        var text: std.ArrayList(u8) = .empty;
        var result: StructLayout = undefined;
        if (l.m.hasDecoration(t, .c_packed)) {
            // OpenCL `__attribute__((packed))` (Kernel modules carry no
            // Offsets): every member right after the previous one, no
            // padding, alignment 1 for the struct and every access through
            // it (exec test cpacked_struct). Members keep their AIR
            // allocation (a 3-element vector occupies four lanes, as
            // OpenCL's `float3` does).
            try text.appendSlice(l.a, "<{");
            var pos: u64 = 0;
            for (tys, lays, 0..) |tt, ly, i| {
                if (l.m.memberDecoration(t, @intCast(i), .offset)) |d| {
                    if ((d.firstInt() orelse 0) != pos) return l.fail("struct %{d} is CPacked but member {d} has Offset {d}, not {d}", .{ t, i, d.firstInt() orelse 0, pos });
                }
                try text.print(l.a, "{s} {s}", .{ if (i == 0) "" else ",", tt });
                index[i] = @intCast(i);
                offsets[i] = pos;
                sizes[i] = ly.size;
                pos += ly.size;
            }
            try text.appendSlice(l.a, if (n == 0) "}>" else " }>");
            result = .{ .text = text.items, .index = index, .offsets = offsets, .sizes = sizes, .member_tys = tys, .tight = tight, .size = pos, .alignment = 1, .abi = 1, .has_runtime_array = runtime };
        } else if (matches) {
            try text.append(l.a, '{');
            for (tys, lays, 0..) |tt, ly, i| {
                index[i] = @intCast(i);
                sizes[i] = ly.size;
                try text.print(l.a, "{s} {s}", .{ if (i == 0) "" else ",", tt });
            }
            try text.appendSlice(l.a, if (n == 0) "}" else " }");
            result = .{ .text = text.items, .index = index, .offsets = offsets, .sizes = sizes, .member_tys = tys, .tight = tight, .size = std.mem.alignForward(u64, cur, abi), .alignment = al, .abi = abi, .has_runtime_array = runtime };
        } else {
            // Packed: every member exactly at its Offset, padding in between.
            // A 3-element vector is a `[3 x T]` here: AIR gives `<3 x T>`
            // the allocation of four, and std430/scalar layouts put the next
            // member right after the third component (vec3 + float at 12).
            // Loads and stores of the vector itself still use `<3 x T>`,
            // whose store size is the three components.
            try text.appendSlice(l.a, "<{");
            var pos: u64 = 0;
            var k: u32 = 0;
            var max_al: u64 = 1;
            // A struct member that ends where AIR would pad it (scalar
            // layout `{ Inner i; float f @12 }` with `Inner { vec3 v; }`)
            // is tight when the next member starts inside that padding.
            for (0..n) |i| {
                if (l.vecLen(members[i]) == 3) {
                    tight[i] = true;
                } else if (i + 1 < n and (try l.typeOf(members[i])) == .@"struct") {
                    const off = l.m.memberDecoration(t, @intCast(i), .offset).?.firstInt() orelse 0;
                    const next = l.m.memberDecoration(t, @intCast(i + 1), .offset).?.firstInt() orelse 0;
                    if (next >= off and next - off < lays[i].size) {
                        if (try l.tightLayout(members[i])) |tl| tight[i] = next - off >= tl.size;
                    }
                }
                if (tight[i]) {
                    tys[i] = try l.memberText(.{ .id = members[i], .tight = true });
                    lays[i] = try l.memberLayout(.{ .id = members[i], .tight = true });
                }
            }
            for (tys, lays, 0..) |tt, ly, i| {
                const off = l.m.memberDecoration(t, @intCast(i), .offset).?.firstInt() orelse 0;
                if (off < pos) return l.fail("struct %{d} member {d} at Offset {d} overlaps the previous member, which occupies bytes up to {d}", .{ t, i, off, pos });
                if (off > pos) {
                    try text.print(l.a, "{s} [{d} x i8]", .{ if (k == 0) "" else ",", off - pos });
                    k += 1;
                }
                try text.print(l.a, "{s} {s}", .{ if (k == 0) "" else ",", tt });
                index[i] = k;
                k += 1;
                offsets[i] = off;
                sizes[i] = ly.size;
                pos = off + ly.size;
                max_al = @max(max_al, ly.alignment);
            }
            try text.appendSlice(l.a, " }>");
            // The alignment the host provides for the block (its largest
            // member's); a member's own accesses are capped by its Offset
            // (Fn.aligns). LLVM's ABI alignment of a packed struct is 1,
            // which is what a parent's natural layout must use.
            result = .{ .text = text.items, .index = index, .offsets = offsets, .sizes = sizes, .member_tys = tys, .tight = tight, .size = pos, .alignment = max_al, .abi = 1, .has_runtime_array = runtime };
        }
        try l.layouts.put(l.a, t, result);
        return result;
    }

    /// Member `i` of struct `t` (index checked by the caller).
    fn structMember(l: *Lowerer, t: Id, i: usize) Error!Member {
        const st = (try l.typeOf(t)).@"struct";
        const sl = try l.structLayout(t);
        return .{ .id = st.members[i], .tight = sl.tight[i] };
    }

    /// Element type of a vector, or the type itself.
    fn scalarOf(l: *Lowerer, t: Id) Id {
        return switch (l.m.typeOf(t) orelse return t) {
            .vector => |v| v.elem,
            else => t,
        };
    }

    fn vecLen(l: *Lowerer, t: Id) ?u32 {
        return switch (l.m.typeOf(t) orelse return null) {
            .vector => |v| v.len,
            else => null,
        };
    }

    /// Apple's suffix: `f32`, `v4f32`, `i32`, `v2i16`.
    fn sfx(l: *Lowerer, t: Id) Error![]const u8 {
        const s = l.scalarOf(t);
        const base: []const u8 = switch (try l.typeOf(s)) {
            .float => |f| if (f.width == 16) "f16" else if (f.width == 32) "f32" else return l.fail("no AIR suffix for a {d}-bit float", .{f.width}),
            .int => |i| try l.a.print("i{d}", .{i.width}),
            .bool => "i1",
            else => return l.fail("no intrinsic suffix for type %{d}", .{t}),
        };
        if (l.vecLen(t)) |n| return l.a.print("v{d}{s}", .{ n, base });
        return base;
    }

    /// `i1` or `<N x i1>` shaped like `t`.
    fn boolTy(l: *Lowerer, t: Id) Error![]const u8 {
        if (l.vecLen(t)) |n| return l.a.print("<{d} x i1>", .{n});
        return "i1";
    }

    /// `t` with its element widened to `bits` (`i64`, `<4 x i64>`).
    fn intTyBits(l: *Lowerer, t: Id, bits: u32) Error![]const u8 {
        if (l.vecLen(t)) |n| return l.a.print("<{d} x i{d}>", .{ n, bits });
        return l.a.print("i{d}", .{bits});
    }

    /// MSL spelling for reflection (`int`, `float4`, `float4x4`, struct names).
    fn mslName(l: *Lowerer, t: Id) Error![]const u8 {
        if (l.type_depth > 64) return l.fail("type %{d} nests deeper than 64 levels (does it contain itself?)", .{t});
        l.type_depth += 1;
        defer l.type_depth -= 1;
        switch (try l.typeOf(t)) {
            .bool => return "bool",
            .int => |i| return switch (i.width) {
                8 => if (i.signed) "char" else "uchar",
                16 => if (i.signed) "short" else "ushort",
                32 => if (i.signed) "int" else "uint",
                else => if (i.signed) "long" else "ulong",
            },
            .float => |f| return if (f.width == 16) "half" else "float",
            .vector => |v| return l.a.print("{s}{d}", .{ try l.mslName(v.elem), v.len }),
            .matrix => |mt| {
                const col = switch (try l.typeOf(mt.column)) {
                    .vector => |v| v,
                    else => return l.fail("matrix type %{d}: its column type is not a vector", .{t}),
                };
                return l.a.print("{s}{d}x{d}", .{ try l.mslName(col.elem), mt.cols, col.len });
            },
            .array => |arr| return l.mslName(arr.elem),
            .runtime_array => |ra| return l.mslName(ra.elem),
            .@"struct" => return l.structName(t),
            // A pointer stored in a buffer (GL_EXT_buffer_reference: a
            // PhysicalStorageBuffer pointer member of a push-constant or
            // storage block) is described as the 8-byte address the host
            // writes there: an MTLBuffer's gpuAddress.
            .pointer => return "ulong",
            else => return l.fail("type %{d} has no MSL name", .{t}),
        }
    }

    fn structName(l: *Lowerer, t: Id) Error![]const u8 {
        if (l.m.name(t)) |full| {
            const short = if (std.mem.lastIndexOfScalar(u8, full, '.')) |dot| full[dot + 1 ..] else full;
            if (try l.sanitize(short)) |s| return s;
        }
        return l.a.print("struct{d}", .{t});
    }

    /// `[A-Za-z0-9_]` only, other bytes to `_`; null when empty.
    fn sanitize(l: *Lowerer, s: []const u8) Error!?[]const u8 {
        if (s.len == 0) return null;
        const out = try l.a.dupe(u8, s);
        for (out) |*ch| {
            if (!std.ascii.isAlphanumeric(ch.*) and ch.* != '_') ch.* = '_';
        }
        return out;
    }

    // ── Constants ───────────────────────────────────────────────────────────

    fn constText(l: *Lowerer, id: Id, repr: Repr) Error![]const u8 {
        if (l.type_depth > 64) return l.fail("constant %{d} nests deeper than 64 levels (does it contain itself?)", .{id});
        l.type_depth += 1;
        defer l.type_depth -= 1;
        const c = l.m.constantOf(id) orelse return l.fail("%{d} is not a constant", .{id});
        // The constant's own type must be printable and fit its value (an
        // OpConstantTrue of an integer type would print `add i32 true, ...`,
        // which Metal builds and Apple's front end rejects).
        _ = l.ty(c.type, repr) catch |err| {
            if (err != error.Unsupported) return err;
            const why = if (l.diag) |d| (if (std.mem.startsWith(u8, d.message, l.context) and d.message.len >= l.context.len + 2) d.message[l.context.len + 2 ..] else d.message) else "";
            return l.failConst(id, "{s}", .{why});
        };
        switch (c.value) {
            .bool => if ((try l.typeOf(c.type)) != .bool) return l.failConst(id, "a bool constant of the non-bool type %{d}", .{c.type}),
            .composite => |parts| {
                const tt = try l.typeOf(c.type);
                for (parts, 0..) |p, i| {
                    const want: Id = switch (tt) {
                        .vector => |v| v.elem,
                        .array => |arr| arr.elem,
                        .matrix => |mt| mt.column,
                        .@"struct" => |st| if (i < st.members.len) st.members[i] else continue,
                        else => return l.failConst(id, "a composite constant of the non-composite type %{d}", .{c.type}),
                    };
                    const pc = l.m.constantOf(p) orelse return l.failConst(id, "constituent %{d} is not a constant", .{p});
                    if (pc.type != want and !std.mem.eql(u8, try l.ty(pc.type, .memory), try l.ty(want, .memory))) return l.failConst(id, "constituent {d} (%{d}) does not have the type of the element it fills", .{ i, p });
                }
            },
            else => {},
        }
        return switch (c.value) {
            .bool => |b| if (repr == .value) (if (b) "true" else "false") else (if (b) "1" else "0"),
            .scalar => l.scalarText(c.type, l.m.constantBits(id) orelse return l.fail("constant %{d} has a literal but its type %{d} is not an integer or float type", .{ id, c.type })),
            .composite => |ids| l.compositeText(c.type, ids, repr),
            .null => l.nullText(c.type, repr),
            .undef => "undef",
            .spec_op => l.fail("OpSpecConstantOp %{d} is not supported", .{id}),
            .other => |op| l.fail("{t} %{d} is not supported", .{ op, id }),
        };
    }

    fn scalarText(l: *Lowerer, t: Id, bits: u64) Error![]const u8 {
        switch (try l.typeOf(t)) {
            .int => |i| {
                _ = try l.ty(t, .memory);
                return intText(l.a, bits, i.width);
            },
            .float => |f| return switch (f.width) {
                16 => l.a.print("0xH{X:0>4}", .{@as(u16, @truncate(bits))}),
                32 => l.a.print("0x{X:0>16}", .{widenF32(@truncate(bits))}),
                else => l.fail("64-bit floats: Metal has no double", .{}),
            },
            else => return l.fail("scalar constant of non-scalar type %{d}", .{t}),
        }
    }

    fn compositeText(l: *Lowerer, t: Id, ids: []const Id, repr: Repr) Error![]const u8 {
        var out: std.ArrayList(u8) = .empty;
        switch (try l.typeOf(t)) {
            .vector => |v| {
                if (ids.len != v.len) return l.fail("composite constant of %{d} has {d} constituents for {d} components", .{ t, ids.len, v.len });
                const et = try l.ty(v.elem, repr);
                try out.append(l.a, '<');
                for (ids, 0..) |e, i| try out.print(l.a, "{s}{s} {s}", .{ if (i == 0) "" else ", ", et, try l.constText(e, repr) });
                try out.append(l.a, '>');
            },
            .array, .matrix => {
                const elem: Member, const len: u64 = switch (try l.typeOf(t)) {
                    .array => |arr| .{ try l.elemOf(t), try l.arrayLen(arr.len) },
                    .matrix => |mt| .{ .{ .id = mt.column }, mt.cols },
                    else => unreachable,
                };
                if (ids.len != len) return l.fail("composite constant of %{d} has {d} constituents for {d} elements", .{ t, ids.len, len });
                const et = try l.memberText(elem);
                try out.append(l.a, '[');
                for (ids, 0..) |e, i| try out.print(l.a, "{s}{s} {s}", .{ if (i == 0) "" else ", ", et, try l.memberConst(e, elem) });
                try out.append(l.a, ']');
            },
            .@"struct" => return l.structConstText(t, ids, try l.structLayout(t)),
            else => return l.fail("composite constant of type %{d}", .{t}),
        }
        return out.items;
    }

    /// A struct constant in layout `sl` (the struct's own, or its tight
    /// form), padding members `undef`.
    fn structConstText(l: *Lowerer, t: Id, ids: []const Id, sl: StructLayout) Error![]const u8 {
        var out: std.ArrayList(u8) = .empty;
        const st = (try l.typeOf(t)).@"struct";
        if (ids.len != st.members.len) return l.fail("composite constant of %{d} has {d} constituents for {d} members", .{ t, ids.len, st.members.len });
        const is_packed = std.mem.startsWith(u8, sl.text, "<{");
        try out.appendSlice(l.a, if (is_packed) "<{ " else "{ ");
        var k: u32 = 0;
        for (ids, st.members, 0..) |e, mt, i| {
            while (k < sl.index[i]) : (k += 1) {
                // A padding member: its size is the gap before member i.
                const prev_end = if (i == 0) 0 else sl.offsets[i - 1] + sl.sizes[i - 1];
                try out.print(l.a, "{s}[{d} x i8] undef", .{ if (k == 0) "" else ", ", sl.offsets[i] - prev_end });
            }
            const mb: Member = .{ .id = mt, .tight = sl.tight[i] };
            try out.print(l.a, "{s}{s} {s}", .{ if (k == 0) "" else ", ", sl.member_tys[i], try l.memberConst(e, mb) });
            k += 1;
        }
        try out.appendSlice(l.a, if (is_packed) " }>" else " }");
        return out.items;
    }

    /// Constant `id` in the storage form of `mb` (a `[3 x T]` member is an
    /// array of its vector constant's components).
    fn memberConst(l: *Lowerer, id: Id, mb: Member) Error![]const u8 {
        if (!mb.tight) return l.constText(id, .memory);
        const c = l.m.constantOf(id) orelse return l.fail("%{d} is not a constant", .{id});
        if (l.vecLen(mb.id) == null) {
            // A tight struct.
            _ = try l.constText(id, .memory);
            return switch (c.value) {
                .composite => |parts| l.structConstText(mb.id, parts, try l.memberStructLayout(mb)),
                .null => "zeroinitializer",
                .undef => "undef",
                else => l.fail("constant %{d} of a struct type is not a composite", .{id}),
            };
        }
        switch (c.value) {
            .composite => |parts| {
                if (parts.len != 3) return l.fail("composite constant %{d} has {d} constituents for 3 components", .{ id, parts.len });
                const et = try l.ty(l.scalarOf(mb.id), .memory);
                var out: std.ArrayList(u8) = .empty;
                try out.append(l.a, '[');
                for (parts, 0..) |e, i| try out.print(l.a, "{s}{s} {s}", .{ if (i == 0) "" else ", ", et, try l.constText(e, .memory) });
                try out.append(l.a, ']');
                return out.items;
            },
            .null => return "zeroinitializer",
            .undef => return "undef",
            else => return l.fail("constant %{d} of a vector type is not a composite", .{id}),
        }
    }

    fn nullText(l: *Lowerer, t: Id, repr: Repr) Error![]const u8 {
        return switch (try l.typeOf(t)) {
            .bool => if (repr == .value) "false" else "0",
            .int => "0",
            .float => l.scalarText(t, 0),
            .pointer => "null",
            else => "zeroinitializer",
        };
    }

    /// `v` as a constant of `t`'s element type, splatted for a vector.
    fn splatConst(l: *Lowerer, t: Id, elem_text: []const u8) Error![]const u8 {
        const n = l.vecLen(t) orelse return elem_text;
        const et = try l.ty(l.scalarOf(t), .value);
        var out: std.ArrayList(u8) = .empty;
        try out.append(l.a, '<');
        for (0..n) |i| try out.print(l.a, "{s}{s} {s}", .{ if (i == 0) "" else ", ", et, elem_text });
        try out.append(l.a, '>');
        return out.items;
    }

    fn intConst(l: *Lowerer, t: Id, v: i64) Error![]const u8 {
        return l.splatConst(t, try l.a.print("{d}", .{v}));
    }

    /// The low bits of `bits` as a constant of `t`'s element width (signed
    /// decimal), splatted for a vector.
    fn intConstBits(l: *Lowerer, t: Id, bits: u64) Error![]const u8 {
        const w = l.m.scalarWidth(t) orelse return l.fail("integer constant of non-integer type %{d}", .{t});
        if (w < 1 or w > 64) return l.fail("{d}-bit integers are not supported", .{w});
        return l.splatConst(t, try intText(l.a, bits, w));
    }

    fn floatConst(l: *Lowerer, t: Id, v: f64) Error![]const u8 {
        const s = l.scalarOf(t);
        const width = l.m.scalarWidth(s) orelse return l.fail("float constant of non-float type %{d}", .{t});
        const bits: u64 = switch (width) {
            16 => @as(u16, @bitCast(@as(f16, @floatCast(v)))),
            32 => @as(u32, @bitCast(@as(f32, @floatCast(v)))),
            else => return l.fail("64-bit floats: Metal has no double", .{}),
        };
        return l.splatConst(t, try l.scalarText(s, bits));
    }

    fn zeroOf(l: *Lowerer, t: Id) Error![]const u8 {
        if (l.vecLen(t) != null) return "zeroinitializer";
        return switch (try l.typeOf(t)) {
            .float => l.scalarText(t, 0),
            .bool => "false",
            else => "0",
        };
    }

    // ── Declarations and threadgroup globals ────────────────────────────────

    fn declare(l: *Lowerer, name: []const u8, ret: []const u8, params: []const []const u8) Error!void {
        if (l.decls.contains(name)) return;
        var line: std.ArrayList(u8) = .empty;
        try line.print(l.a, "declare {s} @{s}(", .{ ret, name });
        for (params, 0..) |p, i| try line.print(l.a, "{s}{s}", .{ if (i == 0) "" else ", ", p });
        try line.append(l.a, ')');
        try l.decls.put(l.a, name, line.items);
    }

    /// Refuse a module-scope variable's initializer whose type is not the
    /// variable's pointee (the store would print the constant as the wrong
    /// type), quoting the OpVariable.
    fn checkInitializer(l: *Lowerer, g: Id, init: Id, pointee: Id) Error!void {
        const it = l.m.resultTypeOf(init) orelse return l.fail("variable %{d}: its initializer %{d} has no type", .{ g, init });
        if (it == pointee or std.mem.eql(u8, try l.ty(it, .memory), try l.ty(pointee, .memory))) return;
        const glob = l.globals.get(g).?;
        return l.failInst(l.m.instructions[glob.inst], "the initializer %{d} does not have the variable's pointee type", .{init});
    }

    /// Refuse a variable whose pointee has no fixed size: an
    /// OpTypeRuntimeArray, or a struct ending in one. SPIR-V allows a
    /// runtime array only in a StorageBuffer, PhysicalStorageBuffer or
    /// Uniform (BufferBlock) block, whose size the host's buffer decides;
    /// a Workgroup, Private or Function variable of one has no memory
    /// behind it (`[0 x T]`: threadgroup_bytes 0 or an `alloca [0 x T]`,
    /// so every element access was out of bounds and the kernel computed
    /// garbage).
    fn checkSized(l: *Lowerer, g: Id, pointee: Id, class: []const u8) Error!void {
        const runtime = switch (try l.typeOf(pointee)) {
            .runtime_array => true,
            .@"struct" => (try l.structLayout(pointee)).has_runtime_array,
            else => false,
        };
        if (runtime) return l.fail("{s} variable %{d} has a runtime-sized type (an OpTypeRuntimeArray, or a struct ending in one): only a StorageBuffer, PhysicalStorageBuffer or Uniform buffer block may end in a runtime array", .{ class, g });
    }

    /// Refuse an OpFunction whose Function Type operand is not an
    /// OpTypeFunction matching its result and parameter types.
    fn checkFunctionType(l: *Lowerer, func: *const reader.Function) Error!void {
        const inst = l.m.instructions[func.first];
        const ft = switch (l.m.typeOf(func.function_type) orelse return l.failInst(inst, "%{d} is not a type", .{func.function_type})) {
            .function => |ft| ft,
            else => return l.failInst(inst, "the function type %{d} is not an OpTypeFunction", .{func.function_type}),
        };
        if (ft.ret != func.result_type and !std.mem.eql(u8, try l.ty(ft.ret, .value), try l.ty(func.result_type, .value))) return l.failInst(inst, "the result type is not the function type's return type", .{});
        if (ft.params.len != func.params.len) return l.failInst(inst, "the function type has {d} parameters, the function {d}", .{ ft.params.len, func.params.len });
        for (ft.params, func.params, 0..) |want, p, i| {
            if (want != p.type and !std.mem.eql(u8, try l.ty(want, .value), try l.ty(p.type, .value))) return l.failInst(inst, "parameter {d} does not have the function type's parameter type", .{i});
        }
    }

    /// `@sym` of a module-level global (`isModuleGlobal`), defining it on
    /// first use: a Workgroup variable is `internal addrspace(3) global T
    /// undef` (threadgroup memory, as on the Zig path); an initialized
    /// UniformConstant is `private addrspace(2) constant T <init>` (Metal
    /// links program-scope data only from the constant space,
    /// docs/air-format.md 2.1).
    fn globalRef(l: *Lowerer, id: Id) Error![]const u8 {
        const g = l.globals.get(id).?;
        if (try l.isWorkgroupBlock(id)) {
            const al = try l.workgroupBlocks();
            return l.a.print("@{s}", .{al.sym});
        }
        const sym = try l.globalSym(id, "g");
        if (!l.global_defs.contains(id)) {
            const pointee = (try l.pointerOf(g.type)).pointee;
            const ly = try l.layout(pointee);
            const mt = try l.ty(pointee, .memory);
            if (isClass(g.storage_class, .workgroup)) {
                try l.checkSized(id, pointee, "Workgroup");
                if (g.initializer) |init| {
                    const c = l.m.constantOf(init);
                    const is_undef = c != null and c.?.value == .undef;
                    if (!is_undef) return l.fail("Workgroup variable %{d} has an initializer; Metal threadgroup memory cannot be initialized statically", .{id});
                }
                try l.global_defs.put(l.a, id, try l.a.print("@{s} = internal addrspace(3) global {s} undef, align {d}", .{ sym, mt, ly.alignment }));
            } else {
                const init = g.initializer.?;
                if (l.m.constantOf(init) == null) return l.fail("UniformConstant variable %{d}: its initializer %{d} is not a constant", .{ id, init });
                try l.checkInitializer(id, init, pointee);
                try l.global_defs.put(l.a, id, try l.a.print("@{s} = private addrspace(2) constant {s} {s}, align {d}", .{ sym, mt, try l.constText(init, .memory), ly.alignment }));
            }
        }
        return l.a.print("@{s}", .{sym});
    }

    /// Whether `g` is a Workgroup variable of a Block struct
    /// (SPV_KHR_workgroup_memory_explicit_layout, GL_EXT_shared_memory_block).
    fn isWorkgroupBlock(l: *Lowerer, g: Id) Error!bool {
        const glob = l.globals.get(g) orelse return false;
        if (!isClass(glob.storage_class, .workgroup)) return false;
        return l.m.hasDecoration((try l.pointerOf(glob.type)).pointee, .block);
    }

    const WorkgroupBlocks = struct { sym: []const u8, size: u64, alignment: u64 };

    /// The one threadgroup allocation every Workgroup Block variable of the
    /// module lives in. Under SPV_KHR_workgroup_memory_explicit_layout the
    /// Workgroup variables decorated Block alias each other (they share the
    /// workgroup's memory, which is how a shader puns shared memory between
    /// types), so they are one `internal addrspace(3) global [N x i8] undef`
    /// as large and as aligned as the largest of them, named after the
    /// first; each variable's accesses GEP from it with its own type (exec
    /// test workgroup_block_alias). Threadgroup memory starts undefined, as
    /// for any Workgroup variable here; an initializer other than OpUndef is
    /// refused as for the others.
    fn workgroupBlocks(l: *Lowerer) Error!WorkgroupBlocks {
        if (l.wg_blocks) |b| return b;
        var first: ?Id = null;
        var size: u64 = 0;
        var al: u64 = 1;
        for (l.m.globals) |glob| {
            if (!try l.isWorkgroupBlock(glob.id)) continue;
            if (glob.initializer) |init| {
                const c = l.m.constantOf(init);
                if (c == null or c.?.value != .undef) return l.fail("Workgroup variable %{d} has an initializer; Metal threadgroup memory cannot be initialized statically", .{glob.id});
            }
            const pointee = (try l.pointerOf(glob.type)).pointee;
            try l.checkSized(glob.id, pointee, "Workgroup");
            const ly = try l.layout(pointee);
            if (first == null) first = glob.id;
            size = @max(size, ly.size);
            al = @max(al, ly.alignment);
        }
        const id = first orelse return l.fail("no Workgroup Block variable", .{});
        const sym = try l.globalSym(id, "g");
        try l.global_defs.put(l.a, id, try l.a.print("@{s} = internal addrspace(3) global [{d} x i8] undef, align {d}", .{ sym, @max(size, 1), al }));
        l.wg_blocks = .{ .sym = sym, .size = size, .alignment = al };
        return l.wg_blocks.?;
    }

    // ── Call graph and interface ────────────────────────────────────────────

    fn storageClass(l: *Lowerer, g: Id) ?spec.StorageClass {
        const glob = l.globals.get(g) orelse return null;
        return std.enums.fromInt(spec.StorageClass, glob.storage_class);
    }

    /// An OpExtInst (or OpExtInstWithForwardRefsKHR) of a `NonSemantic.*`
    /// set (NonSemantic.Shader.DebugInfo.100 from glslang -gVS or DXC,
    /// NonSemantic.DebugPrintf): SPV_KHR_non_semantic_info, core in SPIR-V
    /// 1.6, lets a consumer remove these without changing semantics.
    fn isNonSemantic(l: *Lowerer, inst: Instruction) bool {
        if (inst.op != .OpExtInst and inst.op != .OpExtInstWithForwardRefsKHR) return false;
        const set = l.m.extSetOf(inst.idAt(0) orelse return false) orelse return false;
        return std.mem.startsWith(u8, set.name, "NonSemantic.");
    }

    /// The function `id` names (`Module.functionById` is a linear scan;
    /// this is a map built once per round, so a module of F functions costs
    /// O(F), not O(F^2)).
    fn function(l: *Lowerer, id: Id) ?*const reader.Function {
        if (l.funcs.count() == 0 and l.m.functions.len > 0) {
            l.funcs.ensureTotalCapacity(l.a, @intCast(l.m.functions.len)) catch return l.m.functionById(id);
            for (l.m.functions) |*fu| l.funcs.putAssumeCapacity(fu.id, fu);
        }
        return l.funcs.get(id);
    }

    /// Every value a function body defines (parameters and instruction
    /// results, labels excluded), in any function: a use of one outside its
    /// own function is refused (`Fn.validateBody`).
    fn valueOwners(l: *Lowerer) Error!*const std.AutoHashMapUnmanaged(Id, void) {
        if (!l.owners_done) {
            for (l.m.functions) |fu| {
                for (fu.params) |p| try l.owners.put(l.a, p.id, {});
                for (fu.blocks) |blk| {
                    for (l.m.instructions[blk.first + 1 .. blk.end]) |inst| if (inst.result) |r| try l.owners.put(l.a, r, {});
                }
            }
            l.owners_done = true;
        }
        return &l.owners;
    }

    /// The results of module-scope instructions other than types,
    /// constants, variables, OpUndef, OpExtInstImport, OpString and
    /// OpDecorationGroup, and other than NonSemantic.* OpExtInsts (debug
    /// info, whose ids other NonSemantic instructions name). The reader
    /// accepts such an instruction at module scope by its opcode class (an
    /// OpExtInst is "extension"), but SPIR-V allows no computation there
    /// (section 2.4) and the lowering prints none, so a function using its
    /// result is refused (`Fn.validateBody`); it used to treat the id as a
    /// module-scope value and print a use of an undefined IR value.
    fn moduleNonValues(l: *Lowerer) Error!*const std.AutoHashMapUnmanaged(Id, spec.Opcode) {
        if (!l.non_values_done) {
            var fi: usize = 0;
            var i: usize = 0;
            const insts = l.m.instructions;
            while (i < insts.len) : (i += 1) {
                while (fi < l.m.functions.len and l.m.functions[fi].end <= i) fi += 1;
                if (fi < l.m.functions.len and l.m.functions[fi].first <= i) {
                    i = l.m.functions[fi].end - 1;
                    continue;
                }
                const inst = insts[i];
                const r = inst.result orelse continue;
                switch (inst.op.class()) {
                    .type_declaration, .constant_creation => continue,
                    else => {},
                }
                switch (inst.op) {
                    .OpVariable, .OpUndef, .OpExtInstImport, .OpString, .OpDecorationGroup => continue,
                    else => {},
                }
                if (l.isNonSemantic(inst)) continue;
                try l.non_values.put(l.a, r, inst.op);
            }
            l.non_values_done = true;
        }
        return &l.non_values;
    }

    /// The module-scope variables function `fid` and everything it calls
    /// reach. The call graph is walked with an explicit stack, not by
    /// recursion: a valid module may chain thousands of helpers (each call
    /// level used to be a native stack frame; 10000 overflowed it), and a
    /// call to a function still on the stack is recursion, which SPIR-V
    /// forbids.
    fn funcInfo(l: *Lowerer, fid: Id) Error!FuncInfo {
        if (l.infos.get(fid)) |fi| {
            if (!fi.done) return l.fail("function %{d} is recursive; SPIR-V forbids recursion and Metal cannot run it", .{fid});
            return fi;
        }
        const Frame = struct { func: *const reader.Function, next: u32 };
        var stack: std.ArrayList(Frame) = .empty;
        try l.infos.put(l.a, fid, .{});
        try stack.append(l.a, .{ .func = l.function(fid) orelse return l.fail("OpFunctionCall of %{d}, which is not a function", .{fid}), .next = 0 });
        outer: while (stack.items.len > 0) {
            const top = &stack.items[stack.items.len - 1];
            const func = top.func;
            // Descend into the first callee not described yet.
            while (func.first + top.next < func.end) : (top.next += 1) {
                const inst = l.m.instructions[func.first + top.next];
                if (inst.op != .OpFunctionCall or l.isNonSemantic(inst)) continue;
                const callee = inst.idAt(0) orelse return l.fail("OpFunctionCall without a callee", .{});
                if (l.infos.get(callee)) |fi| {
                    if (!fi.done) return l.fail("function %{d} is recursive; SPIR-V forbids recursion and Metal cannot run it", .{callee});
                    continue;
                }
                const cf = l.function(callee) orelse return l.fail("OpFunctionCall of %{d}, which is not a function", .{callee});
                try l.infos.put(l.a, callee, .{});
                top.next += 1;
                try stack.append(l.a, .{ .func = cf, .next = 0 });
                continue :outer;
            }
            // Every callee is described: this function's variables.
            var iface: std.AutoArrayHashMapUnmanaged(Id, void) = .empty;
            var wg: std.AutoArrayHashMapUnmanaged(Id, void) = .empty;
            for (l.m.instructions[func.first..func.end]) |inst| {
                // A NonSemantic.* instruction is dropped (Fn.lowerInst), so the
                // variables its debug info names do not become parameters.
                if (l.isNonSemantic(inst)) continue;
                if (inst.op == .OpFunctionCall) {
                    const sub = l.infos.get(inst.idAt(0).?).?;
                    for (sub.iface) |g| try iface.put(l.a, g, {});
                    for (sub.workgroup) |g| try wg.put(l.a, g, {});
                }
                for (inst.operands) |o| {
                    const id = o.asId() orelse continue;
                    if (!l.globals.contains(id)) continue;
                    if (l.isModuleGlobal(id)) try wg.put(l.a, id, {}) else try iface.put(l.a, id, {});
                }
            }
            const ids = try l.a.dupe(Id, iface.keys());
            std.mem.sort(Id, ids, {}, std.sort.asc(Id));
            const wids = try l.a.dupe(Id, wg.keys());
            std.mem.sort(Id, wids, {}, std.sort.asc(Id));
            try l.infos.put(l.a, func.id, .{ .iface = ids, .workgroup = wids, .done = true });
            _ = stack.pop();
        }
        return l.infos.get(fid).?;
    }

    /// The builtin parameter type for an Input variable's pointee: `i32` or
    /// `<N x i32>` (a 64-bit OpenCL id is zero-extended at the load).
    fn builtinParamTy(l: *Lowerer, pointee: Id) Error![]const u8 {
        const s = l.scalarOf(pointee);
        const w = switch (try l.typeOf(s)) {
            .int => |i| i.width,
            else => return l.fail("builtin of non-integer type %{d}", .{pointee}),
        };
        if (w != 32 and w != 64) return l.fail("{d}-bit builtin variables are not supported (32 or 64)", .{w});
        if (l.vecLen(pointee)) |n| return l.a.print("<{d} x i32>", .{n});
        return "i32";
    }

    fn builtinKind(l: *Lowerer, g: Id) Error!air.BuiltinKind {
        const d = l.m.decoration(g, .built_in) orelse return l.fail("Input variable %{d} has no BuiltIn decoration (location inputs are stage 5)", .{g});
        const raw = d.firstInt() orelse 0;
        const v = std.math.cast(u32, raw) orelse return l.fail("unknown BuiltIn {d}", .{raw});
        const b = std.enums.fromInt(spec.BuiltIn, v) orelse return l.fail("unknown BuiltIn {d}", .{v});
        return switch (b) {
            .global_invocation_id => .thread_position_in_grid,
            .local_invocation_id => .thread_position_in_threadgroup,
            .workgroup_id => .threadgroup_position_in_grid,
            .workgroup_size => .threads_per_threadgroup,
            .num_workgroups => .threadgroups_per_grid,
            .local_invocation_index => .thread_index_in_threadgroup,
            .subgroup_local_invocation_id => .thread_index_in_simdgroup,
            .subgroup_id => .simdgroup_index_in_threadgroup,
            .subgroup_size, .subgroup_max_size => .threads_per_simdgroup,
            .num_subgroups => .simdgroups_per_threadgroup,
            .global_size => .threads_per_grid,
            else => l.fail("BuiltIn {t} has no AIR kernel attribute here", .{b}),
        };
    }

    fn u32Of(l: *Lowerer, v: u64) Error!u32 {
        if (v > std.math.maxInt(i32)) return l.fail("{d} does not fit AIR's 32-bit signed metadata integers", .{v});
        return @intCast(v);
    }

    fn argName(l: *Lowerer, id: Id, fallback: []const u8, n: u32) Error![]const u8 {
        if (l.m.name(id)) |name| {
            if (try l.sanitize(name)) |s| return s;
        }
        return l.a.print("{s}{d}", .{ fallback, n });
    }

    fn bufferDesc(l: *Lowerer, name_id: Id, pointee: Id, index: u32, source: manifest.Source, space: manifest.Space, access: manifest.Access) Error!manifest.Buffer {
        const ly = try l.layout(pointee);
        var b: manifest.Buffer = .{
            .index = index,
            .source = source,
            .name = try l.argName(name_id, "buffer", index),
            .space = space,
            .access = access,
            .size = try l.u32Of(ly.size),
            .alignment = try l.u32Of(ly.alignment),
            .type_name = try l.mslName(pointee),
        };
        switch (try l.typeOf(pointee)) {
            .@"struct" => |st| {
                const sl = try l.structLayout(pointee);
                const fields = try l.a.alloc(manifest.Field, st.members.len);
                for (st.members, fields, 0..) |mt, *f, i| {
                    const fname = if (l.m.memberName(pointee, @intCast(i))) |mn| try l.sanitize(mn) else null;
                    f.* = .{ .offset = try l.u32Of(sl.offsets[i]), .size = 0, .count = 0, .type_name = try l.mslName(mt), .name = fname orelse try l.a.print("m{d}", .{i}) };
                    switch (try l.typeOf(mt)) {
                        .array => |arr| {
                            f.size = try l.u32Of((try l.memberLayout(try l.elemOf(mt))).size);
                            f.count = try l.u32Of(try l.arrayLen(arr.len));
                        },
                        .runtime_array => {
                            f.size = try l.u32Of((try l.memberLayout(try l.elemOf(mt))).size);
                            f.count = 1;
                        },
                        else => f.size = try l.u32Of(sl.sizes[i]),
                    }
                }
                b.fields = fields;
                if (!sl.has_runtime_array) b.buffer_size = b.size;
            },
            else => {},
        }
        return b;
    }

    /// The workgroup size the host must dispatch with: a constant decorated
    /// `BuiltIn WorkgroupSize` when the module has one (it takes precedence
    /// over LocalSize / LocalSizeId, SPIR-V 1.6 section 3.21; glslang emits
    /// `LocalSize 1 1 1` plus such a constant for `local_size_x_id`), at its
    /// default value like every specialization constant here; else
    /// LocalSize / LocalSizeId. Every component must be a non-zero constant
    /// that fits 32 bits.
    fn localSize(l: *Lowerer, ep: reader.EntryPoint) Error![3]u32 {
        var size: [3]u32 = .{ 1, 1, 1 };
        for (ep.execution_modes) |mi| {
            const em = l.m.execution_modes[mi];
            switch (em.as() orelse continue) {
                .local_size => for (&size, 0..) |*s, k| {
                    const v = if (k < em.operands.len) em.operands[k].asInt() orelse 1 else return l.fail("LocalSize with {d} operands", .{em.operands.len});
                    s.* = std.math.cast(u32, v) orelse return l.fail("LocalSize {d} does not fit 32 bits", .{v});
                },
                .local_size_id => for (&size, 0..) |*s, k| {
                    const id = (if (k < em.operands.len) em.operands[k].asId() else null) orelse return l.fail("LocalSizeId without ids", .{});
                    const v = l.m.constantBits(id) orelse return l.fail("LocalSizeId %{d} is not a constant", .{id});
                    s.* = std.math.cast(u32, v) orelse return l.fail("LocalSizeId %{d} = {d} does not fit 32 bits", .{ id, v });
                },
                else => {},
            }
        }
        var found: ?Id = null;
        for (l.m.decorations) |d| {
            if (d.member != null or d.decoration != @backingInt(spec.Decoration.built_in)) continue;
            if ((d.firstInt() orelse continue) != @backingInt(spec.BuiltIn.workgroup_size)) continue;
            // An Input variable with this BuiltIn is the runtime builtin
            // parameter (threads_per_threadgroup), not a size.
            if (l.globals.contains(d.target)) continue;
            if (found) |prev| if (prev != d.target) return l.fail("two constants (%{d}, %{d}) are decorated BuiltIn WorkgroupSize", .{ prev, d.target });
            found = d.target;
            const c = l.m.constantOf(d.target) orelse return l.fail("%{d} is decorated BuiltIn WorkgroupSize but is neither a constant nor an Input variable", .{d.target});
            const comps = switch (c.value) {
                .composite => |ids| ids,
                else => return l.fail("the WorkgroupSize constant %{d} is not a composite of three integer constants", .{d.target}),
            };
            if (comps.len != 3) return l.fail("the WorkgroupSize constant %{d} has {d} components, not 3", .{ d.target, comps.len });
            for (&size, comps) |*s, cid| {
                const v = l.m.constantBits(cid) orelse return l.fail("component %{d} of the WorkgroupSize constant %{d} is not an integer constant (an OpSpecConstantOp is not evaluated)", .{ cid, d.target });
                s.* = std.math.cast(u32, v) orelse return l.fail("component %{d} of the WorkgroupSize constant %{d} is {d}, which does not fit 32 bits", .{ cid, d.target, v });
            }
        }
        for (size) |s| if (s == 0) return l.fail("the workgroup size {d} {d} {d} has a zero component", .{ size[0], size[1], size[2] });
        return size;
    }

    /// Refuse an execution mode that changes what the kernel computes and
    /// is not implemented: only the sizes (read by `localSize`) and modes
    /// that change nothing here pass. LocalSizeHint / LocalSizeHintId /
    /// VecTypeHint are hints; ContractionOff and FPFastMathDefault are
    /// fast-math permissions and restrictions, decided by the library-level
    /// `--fast-math` like the NoContraction / FPFastMathMode decorations;
    /// RoundingModeRTE for 16- and 32-bit floats is what Metal does (the
    /// probes check arithmetic and conversions against round-to-nearest-
    /// even references). RoundingModeRTZ, DenormPreserve, DenormFlushToZero
    /// and SignedZeroInfNanPreserve (SPV_KHR_float_controls, core in 1.4)
    /// ask for semantics Metal does not offer per kernel or that are not
    /// verified here, and are refused by name, as is anything else.
    fn checkExecutionModes(l: *Lowerer, ep: reader.EntryPoint) Error!void {
        for (ep.execution_modes) |mi| {
            const em = l.m.execution_modes[mi];
            const mode = em.as() orelse return l.fail("execution mode {d} is not supported", .{em.mode});
            switch (mode) {
                .local_size, .local_size_id, .local_size_hint, .local_size_hint_id, .vec_type_hint, .contraction_off, .fp_fast_math_default => {},
                .rounding_mode_rte => {
                    const w = (if (em.operands.len > 0) em.operands[0].asInt() else null) orelse return l.fail("RoundingModeRTE without a width", .{});
                    if (w != 16 and w != 32) return l.fail("RoundingModeRTE {d}: only 16- and 32-bit floats exist on Metal", .{w});
                },
                .rounding_mode_rtz => return l.fail("execution mode RoundingModeRTZ: Metal rounds float arithmetic to nearest even and has no round-toward-zero mode", .{}),
                .denorm_preserve, .denorm_flush_to_zero => return l.fail("execution mode {t}: Metal gives a kernel no control over denormals, so the mode cannot be honoured", .{mode}),
                .signed_zero_inf_nan_preserve => return l.fail("execution mode SignedZeroInfNanPreserve: not verified on Metal (and a --fast-math library may drop signed zeros, infinities and NaNs)", .{}),
                else => return l.fail("execution mode {t} is not supported", .{mode}),
            }
        }
    }

    // ── Entry points and helpers ────────────────────────────────────────────

    const PendingBuffer = struct { buffer: manifest.Buffer, target: Id, by_value: ?Id = null, ptr_ty: []const u8 };
    const PendingBuiltin = struct { builtin: manifest.Builtin, target: Id, param_ty: []const u8 };

    fn lowerEntry(l: *Lowerer, ep: reader.EntryPoint, out: *std.ArrayList(u8)) Error!manifest.Kernel {
        l.context = try l.a.print("entry point '{s}'", .{ep.name});
        const func = l.function(ep.function) orelse return l.fail("OpEntryPoint names %{d}, which is not a function", .{ep.function});
        if ((try l.typeOf(func.result_type)) != .void) return l.fail("an entry point function must return void", .{});
        try l.checkFunctionType(func);
        const info = try l.funcInfo(func.id);
        try l.checkExecutionModes(ep);

        var bufs: std.ArrayList(PendingBuffer) = .empty;
        var unbound: std.ArrayList(PendingBuffer) = .empty;
        var push: ?PendingBuffer = null;
        var builtins: std.ArrayList(PendingBuiltin) = .empty;
        var privates: std.ArrayList(Id) = .empty;
        for (info.iface) |g| {
            const glob = l.globals.get(g).?;
            const pointee = (try l.pointerOf(glob.type)).pointee;
            const sc = l.storageClass(g) orelse return l.fail("variable %{d} has an unknown storage class {d}", .{ g, glob.storage_class });
            switch (sc) {
                .storage_buffer, .uniform, .uniform_constant, .push_constant => {
                    switch (try l.typeOf(pointee)) {
                        .other => |op| return l.fail("variable %{d} is an {t}: images and samplers are stage 5", .{ g, op }),
                        .array, .runtime_array => {
                            // GLSL `buffer B { ... } bufs[2];`: one descriptor
                            // per element, but an AIR buffer parameter is one
                            // pointer, and a runtime-array Block element has
                            // size 0, so bufs[1] would alias bufs[0].
                            const elem = (try l.elemOf(pointee)).id;
                            if (l.m.hasDecoration(elem, .block) or l.m.hasDecoration(elem, .buffer_block))
                                return l.fail("variable %{d} is a descriptor array (an array of Block structs); Metal binds one buffer per [[buffer(n)]], so descriptor arrays are not supported: declare one variable per binding", .{g});
                        },
                        else => {},
                    }
                    // Vulkan forbids initializers on these; an OpenCL
                    // program-scope `__constant` table (UniformConstant) is a
                    // module global (isModuleGlobal), never a parameter.
                    if (glob.initializer != null) return l.fail("{t} variable %{d} has an initializer; only Private, Function and UniformConstant variables can be initialized here", .{ sc, g });
                },
                else => {},
            }
            switch (sc) {
                .storage_buffer, .uniform, .uniform_constant => {
                    const set = if (l.m.decoration(g, .descriptor_set)) |d| d.firstInt() orelse 0 else 0;
                    if (set != 0) return l.fail("variable %{d} is in DescriptorSet {d}; only set 0 maps onto Metal buffer indices", .{ g, set });
                    const device = (try l.globalSpace(g)) == 1;
                    const space: manifest.Space = if (device) .device else .constant;
                    const access: manifest.Access = if (device and !l.nonWritable(g, pointee)) .read_write else .read;
                    if (l.m.decoration(g, .binding)) |d| {
                        const raw = d.firstInt() orelse 0;
                        const binding = std.math.cast(u32, raw) orelse return l.fail("variable %{d} has Binding {d}", .{ g, raw });
                        try bufs.append(l.a, .{ .buffer = try l.bufferDesc(g, pointee, binding, .{ .binding = .{ .set = 0, .binding = binding } }, space, access), .target = g, .ptr_ty = try l.globalPtrText(g) });
                    } else {
                        try unbound.append(l.a, .{ .buffer = try l.bufferDesc(g, pointee, 0, .unbound, space, access), .target = g, .ptr_ty = try l.globalPtrText(g) });
                    }
                },
                .push_constant => {
                    if (push != null) return l.fail("more than one PushConstant variable", .{});
                    push = .{ .buffer = try l.bufferDesc(g, pointee, l.opts.push_constant_index, .push_constant, .constant, .read), .target = g, .ptr_ty = "ptr addrspace(2)" };
                },
                .input => {
                    const kind = try l.builtinKind(g);
                    const pty = try l.builtinParamTy(pointee);
                    const tn = if (l.vecLen(pointee)) |n| try l.a.print("uint{d}", .{n}) else "uint";
                    try builtins.append(l.a, .{ .builtin = .{ .kind = kind, .type_name = tn, .name = try l.argName(g, "builtin", g) }, .target = g, .param_ty = pty });
                },
                .private => try privates.append(l.a, g),
                .output => return l.fail("Output variable %{d}: outputs are graphics interfaces (stage 5)", .{g}),
                .cross_workgroup => return l.fail("module-scope CrossWorkgroup variable %{d}: Metal has no program-scope device memory", .{g}),
                else => return l.fail("module-scope {t} variable %{d} is not supported", .{ sc, g }),
            }
        }
        for (func.params, 0..) |p, i| {
            const n: u32 = @intCast(i);
            switch (try l.typeOf(p.type)) {
                .pointer => |pt| {
                    const sc = std.enums.fromInt(spec.StorageClass, pt.storage_class) orelse return l.fail("unknown storage class", .{});
                    const space: manifest.Space = switch (sc) {
                        .storage_buffer, .physical_storage_buffer, .cross_workgroup, .generic => .device,
                        .uniform, .uniform_constant, .push_constant => .constant,
                        .workgroup => return l.fail("entry parameter {d} is a Workgroup pointer (an OpenCL `local` argument): Metal sizes such memory per dispatch through [[threadgroup(n)]] arguments, which this lowering does not implement; declare a Workgroup variable instead", .{i}),
                        else => return l.fail("entry parameter {d} points into {t} memory, which a kernel cannot be given", .{ i, sc }),
                    };
                    const access: manifest.Access = if (space == .device and !l.m.hasDecoration(p.id, .non_writable)) .read_write else .read;
                    try bufs.append(l.a, .{ .buffer = try l.bufferDesc(p.id, pt.pointee, n, .{ .parameter = n }, space, access), .target = p.id, .ptr_ty = try l.ptrText(pt.storage_class) });
                },
                else => {
                    var b = try l.bufferDesc(p.id, p.type, n, .{ .parameter = n }, .constant, .read);
                    b.by_value = true;
                    try bufs.append(l.a, .{ .buffer = b, .target = p.id, .by_value = p.type, .ptr_ty = "ptr addrspace(2)" });
                },
            }
        }
        // Unbound buffers (Zig's plain `addrspace(.storage_buffer)` externs):
        // the lowest free indices in id order, recorded in the manifest.
        for (unbound.items) |*u| {
            var idx: u32 = 0;
            while (true) : (idx += 1) {
                var taken = push != null and push.?.buffer.index == idx;
                for (bufs.items) |b| taken = taken or b.buffer.index == idx;
                if (!taken) break;
            }
            u.buffer.index = idx;
            u.buffer.name = try l.argName(u.target, "buffer", idx);
            try bufs.append(l.a, u.*);
        }
        std.mem.sort(PendingBuffer, bufs.items, {}, struct {
            fn lt(_: void, x: PendingBuffer, y: PendingBuffer) bool {
                return x.buffer.index < y.buffer.index;
            }
        }.lt);
        for (bufs.items, 0..) |b, i| {
            if (i > 0 and bufs.items[i - 1].buffer.index == b.buffer.index) return l.fail("two buffers use [[buffer({d})]]", .{b.buffer.index});
            if (push) |pc| if (pc.buffer.index == b.buffer.index) return l.fail("the push constant index {d} is also a binding; pick another with --push-constant-index", .{b.buffer.index});
            if (b.buffer.index > 30) return l.fail("buffer index {d} exceeds Metal's 31 buffer slots", .{b.buffer.index});
        }
        // Builtins in interface order, then the rest by id.
        var ordered: std.ArrayList(PendingBuiltin) = .empty;
        for (ep.interface) |iid| for (builtins.items) |b| if (b.target == iid) try ordered.append(l.a, b);
        for (builtins.items) |b| {
            var seen = false;
            for (ordered.items) |o| seen = seen or o.target == b.target;
            if (!seen) try ordered.append(l.a, b);
        }

        var f: Fn = .{ .l = l };
        var params: std.ArrayList(manifest.Param) = .empty;
        var header: std.ArrayList(u8) = .empty;
        var prologue: std.ArrayList(u8) = .empty;
        try header.print(l.a, "define void @{s}(", .{try l.entrySym(ep.name)});
        var k: u32 = 0;
        const all_bufs = try l.a.alloc(PendingBuffer, bufs.items.len + @intFromBool(push != null));
        @memcpy(all_bufs[0..bufs.items.len], bufs.items);
        if (push) |pc| all_bufs[bufs.items.len] = pc;
        for (all_bufs) |b| {
            try header.print(l.a, "{s}{s} %a{d}", .{ if (k == 0) "" else ", ", b.ptr_ty, k });
            const pname = try l.a.print("%a{d}", .{k});
            if (b.by_value) |vt| {
                const ly = try l.layout(vt);
                const tmp = f.fresh();
                try prologue.print(l.a, "  {s} = load {s}, ptr addrspace(2) {s}, align {d}\n", .{ tmp, try l.ty(vt, .memory), pname, ly.alignment });
                if (l.isBoolish(vt)) {
                    const t2 = f.fresh();
                    try prologue.print(l.a, "  {s} = icmp ne {s} {s}, {s}\n", .{ t2, try l.ty(vt, .memory), tmp, if (l.vecLen(vt) != null) "zeroinitializer" else "0" });
                    try f.bind(b.target, t2);
                } else try f.bind(b.target, tmp);
            } else try f.bind(b.target, pname);
            try params.append(l.a, .{ .buffer = b.buffer });
            k += 1;
        }
        for (ordered.items) |b| {
            try header.print(l.a, "{s}{s} %a{d}", .{ if (k == 0) "" else ", ", b.param_ty, k });
            try f.vals.put(l.a, b.target, .{ .input = .{ .value = try l.a.print("%a{d}", .{k}), .param_ty = b.param_ty } });
            try params.append(l.a, .{ .builtin = b.builtin });
            k += 1;
        }
        try header.appendSlice(l.a, ") {\n");
        for (privates.items) |g| {
            const glob = l.globals.get(g).?;
            const pointee = (try l.pointerOf(glob.type)).pointee;
            try l.checkSized(g, pointee, "Private");
            const ly = try l.layout(pointee);
            const name = try l.a.print("%i{d}", .{g});
            try prologue.print(l.a, "  {s} = alloca {s}, align {d}\n", .{ name, try l.ty(pointee, .memory), ly.alignment });
            if (glob.initializer) |init| {
                try l.checkInitializer(g, init, pointee);
                try prologue.print(l.a, "  store {s} {s}, ptr {s}, align {d}\n", .{ try l.ty(pointee, .memory), try l.constText(init, .memory), name, ly.alignment });
            }
            try f.bind(g, name);
        }
        // Threadgroup memory: every Workgroup variable reached, the aliased
        // Block variables once (their shared allocation's size).
        var tg_bytes: u64 = 0;
        var blocks = false;
        for (info.workgroup) |g| {
            const glob = l.globals.get(g).?;
            if (!isClass(glob.storage_class, .workgroup)) continue;
            if (try l.isWorkgroupBlock(g)) {
                blocks = true;
                continue;
            }
            const pointee = (try l.pointerOf(glob.type)).pointee;
            try l.checkSized(g, pointee, "Workgroup");
            tg_bytes += (try l.layout(pointee)).size;
        }
        if (blocks) tg_bytes += (try l.workgroupBlocks()).size;

        const body = try f.lowerBody(func, prologue.items);
        try out.print(l.a, "{s}{s}}}\n\n", .{ header.items, body });
        return .{
            .name = try l.a.dupeSentinel(u8, ep.name, 0),
            .local_size = try l.localSize(ep),
            .params = params.items,
            .threadgroup_bytes = try l.u32Of(tg_bytes),
        };
    }

    /// Whether a buffer variable is read-only: NonWritable on the variable,
    /// or on every member of its Block struct.
    fn nonWritable(l: *Lowerer, g: Id, pointee: Id) bool {
        if (l.m.hasDecoration(g, .non_writable)) return true;
        const st = switch (l.m.typeOf(pointee) orelse return false) {
            .@"struct" => |s| s,
            else => return false,
        };
        if (st.members.len == 0) return false;
        for (0..st.members.len) |i| if (l.m.memberDecoration(pointee, @intCast(i), .non_writable) == null) return false;
        return true;
    }

    /// Whether any instruction of a function body uses value `id`.
    fn isUsed(l: *Lowerer, id: Id) bool {
        for (l.m.functions) |func| {
            for (l.m.instructions[func.first..func.end]) |inst| {
                if (inst.result == id) continue;
                for (inst.operands) |o| if (o.asId() == id) return true;
            }
        }
        return false;
    }

    fn helperSym(l: *Lowerer, fid: Id) Error![]const u8 {
        try l.helpers.put(l.a, fid, {});
        return l.globalSym(fid, "f");
    }

    fn lowerHelper(l: *Lowerer, fid: Id, out: *std.ArrayList(u8)) Error!void {
        const func = l.function(fid).?;
        const sym = try l.globalSym(fid, "f");
        l.context = try l.a.print("function @{s}", .{sym});
        try l.checkFunctionType(func);
        const info = try l.funcInfo(fid);
        var f: Fn = .{ .l = l };
        var header: std.ArrayList(u8) = .empty;
        try header.print(l.a, "define private {s} @{s}(", .{ try l.ty(func.result_type, .value), sym });
        var k: usize = 0;
        for (func.params) |p| {
            const name = try l.a.print("%{s}", .{try l.localName(p.id)});
            try header.print(l.a, "{s}{s} {s}", .{ if (k == 0) "" else ", ", try l.ty(p.type, .value), name });
            k += 1;
        }
        for (info.iface) |g| {
            const glob = l.globals.get(g).?;
            const name = try l.a.print("%i{d}", .{g});
            if (isClass(glob.storage_class, .input)) {
                const pty = try l.builtinParamTy((try l.pointerOf(glob.type)).pointee);
                try header.print(l.a, "{s}{s} {s}", .{ if (k == 0) "" else ", ", pty, name });
                try f.vals.put(l.a, g, .{ .input = .{ .value = name, .param_ty = pty } });
            } else {
                try header.print(l.a, "{s}{s} {s}", .{ if (k == 0) "" else ", ", try l.globalPtrText(g), name });
                try f.bind(g, name);
            }
            k += 1;
        }
        try header.appendSlice(l.a, ") {\n");
        const body = try f.lowerBody(func, "");
        try out.print(l.a, "{s}{s}}}\n\n", .{ header.items, body });
    }
};

/// Signed decimal of the low `width` bits of `bits`.
fn intText(a: Allocator, bits: u64, width: u32) Error![]const u8 {
    const shift: u6 = @intCast(64 - width);
    const v: i64 = @as(i64, @bitCast(bits << shift)) >> shift;
    return a.print("{d}", .{v});
}

fn minOpt(x: ?u64, y: ?u64) ?u64 {
    if (x) |xv| return if (y) |yv| @min(xv, yv) else xv;
    return y;
}

/// The largest power of two dividing `off` (any alignment for 0).
fn alignOfOffset(off: u64) u64 {
    return if (off == 0) std.math.maxInt(u64) else @as(u64, 1) << @intCast(@ctz(off));
}

/// The f64 bit pattern holding exactly the f32 `bits` (LLVM's hex float
/// literal for a float constant), NaN payloads kept.
fn widenF32(bits: u32) u64 {
    const sign: u64 = @as(u64, bits >> 31) << 63;
    const exp = (bits >> 23) & 0xFF;
    const mant: u64 = bits & 0x7FFFFF;
    if (exp == 0xFF) return sign | (@as(u64, 0x7FF) << 52) | (mant << 29);
    const x: f32 = @bitCast(bits);
    const d: f64 = x;
    return @bitCast(d);
}

const Val = union(enum) {
    /// An LLVM operand (`%v12`, `%a0`, `@tile`, a constant).
    op: []const u8,
    /// A pointer to (a lane of) a builtin Input variable.
    input: Input,
};

/// How an access-chain result was computed: one GEP from `root`, so a chain
/// built on it continues from the root (Apple's front end emits one GEP per
/// accessed member; a GEP from a runtime-indexed element pointer crashes
/// MTLCompilerService, Fn.splitAggregate).
const Chain = struct {
    root: []const u8,
    src: []const u8,
    pty: []const u8,
    idx: []const u8,
    inb: bool,
    /// Some index is not a constant.
    dynamic: bool,
    /// The type the indices end at is stored in its tight form there
    /// (`Member.tight`): a chain continued from this one walks the tight
    /// form's member indices, which padding members shift.
    tight: bool = false,
};

const Input = struct {
    value: []const u8,
    param_ty: []const u8,
    /// Typed lane index (`i32 0`) after an access chain.
    lane: ?[]const u8 = null,
};

const Math = enum {
    round,
    rint,
    trunc,
    fabs,
    sabs,
    fsign,
    fsign_nan0,
    ssign,
    floor,
    ceil,
    fract,
    radians,
    degrees,
    sin,
    cos,
    tan,
    asin,
    acos,
    atan,
    sinh,
    cosh,
    tanh,
    asinh,
    acosh,
    atanh,
    atan2,
    pow,
    powr,
    exp,
    log,
    exp2,
    log2,
    exp10,
    log10,
    sqrt,
    rsqrt,
    fmin,
    fmax,
    umin,
    umax,
    smin,
    smax,
    fclamp,
    uclamp,
    sclamp,
    mix_glsl,
    mix_ocl,
    step,
    smoothstep,
    fma,
    mad,
    ldexp,
    copysign,
    fmod,
    length,
    distance,
    cross,
    normalize,
    length_ocl,
    distance_ocl,
    normalize_ocl,
    find_ilsb,
    find_smsb,
    find_umsb,
    popcount,
    clz,
    ctz,
    s_mul_hi,
    u_mul_hi,
};

fn glslMath(op: spec.GlslOpcode) ?Math {
    return switch (op) {
        .Round => .round,
        .RoundEven => .rint,
        .Trunc => .trunc,
        .FAbs => .fabs,
        .SAbs => .sabs,
        .FSign => .fsign,
        .SSign => .ssign,
        .Floor => .floor,
        .Ceil => .ceil,
        .Fract => .fract,
        .Radians => .radians,
        .Degrees => .degrees,
        .Sin => .sin,
        .Cos => .cos,
        .Tan => .tan,
        .Asin => .asin,
        .Acos => .acos,
        .Atan => .atan,
        .Sinh => .sinh,
        .Cosh => .cosh,
        .Tanh => .tanh,
        .Asinh => .asinh,
        .Acosh => .acosh,
        .Atanh => .atanh,
        .Atan2 => .atan2,
        .Pow => .pow,
        .Exp => .exp,
        .Log => .log,
        .Exp2 => .exp2,
        .Log2 => .log2,
        .Sqrt => .sqrt,
        .InverseSqrt => .rsqrt,
        .FMin, .NMin => .fmin,
        .FMax, .NMax => .fmax,
        .UMin => .umin,
        .UMax => .umax,
        .SMin => .smin,
        .SMax => .smax,
        .FClamp, .NClamp => .fclamp,
        .UClamp => .uclamp,
        .SClamp => .sclamp,
        .FMix => .mix_glsl,
        .Step => .step,
        .SmoothStep => .smoothstep,
        .Fma => .fma,
        .Ldexp => .ldexp,
        .Length => .length,
        .Distance => .distance,
        .Cross => .cross,
        .Normalize => .normalize,
        .FindILsb => .find_ilsb,
        .FindSMsb => .find_smsb,
        .FindUMsb => .find_umsb,
        else => null,
    };
}

fn oclMath(op: spec.OpenClOpcode) ?Math {
    return switch (op) {
        .fabs => .fabs,
        .sqrt => .sqrt,
        .rsqrt => .rsqrt,
        .floor => .floor,
        .ceil => .ceil,
        .trunc => .trunc,
        .rint => .rint,
        .round => .round,
        .sin => .sin,
        .cos => .cos,
        .tan => .tan,
        .asin => .asin,
        .acos => .acos,
        .atan => .atan,
        .sinh => .sinh,
        .cosh => .cosh,
        .tanh => .tanh,
        .asinh => .asinh,
        .acosh => .acosh,
        .atanh => .atanh,
        .exp => .exp,
        .exp2 => .exp2,
        .exp10 => .exp10,
        .log => .log,
        .log2 => .log2,
        .log10 => .log10,
        .sign => .fsign_nan0,
        .atan2 => .atan2,
        .pow => .pow,
        .powr => .powr,
        .fmin => .fmin,
        .fmax => .fmax,
        .step => .step,
        .copysign => .copysign,
        .fmod => .fmod,
        .fma => .fma,
        .mad => .mad,
        .fclamp => .fclamp,
        .mix => .mix_ocl,
        .smoothstep => .smoothstep,
        .ldexp => .ldexp,
        .radians => .radians,
        .degrees => .degrees,
        .s_abs => .sabs,
        .popcount => .popcount,
        .clz => .clz,
        .ctz => .ctz,
        .s_min => .smin,
        .s_max => .smax,
        .u_min => .umin,
        .u_max => .umax,
        .s_mul_hi => .s_mul_hi,
        .u_mul_hi => .u_mul_hi,
        .s_clamp => .sclamp,
        .u_clamp => .uclamp,
        .normalize => .normalize_ocl,
        .length => .length_ocl,
        .distance => .distance_ocl,
        .cross => .cross,
        else => null,
    };
}

const Arg = struct { ty: []const u8, v: []const u8 };

/// Per-function lowering state.
const Fn = struct {
    l: *Lowerer,
    out: std.ArrayList(u8) = .empty,
    vals: std.AutoHashMapUnmanaged(Id, Val) = .empty,
    /// Address space of pointer values whose type does not decide it: a
    /// Uniform-class pointer derived from a BufferBlock variable is device
    /// memory (Lowerer.isBufferBlock).
    spaces: std.AutoHashMapUnmanaged(Id, u32) = .empty,
    /// Known alignment of pointers derived by access chains: a member of a
    /// packed struct is aligned only as far as its Offset allows.
    aligns: std.AutoHashMapUnmanaged(Id, u64) = .empty,
    /// Access-chain results (see `Chain`).
    chains: std.AutoHashMapUnmanaged(Id, Chain) = .empty,
    /// Pointers to a struct stored in its tight form (`Member.tight`): a
    /// member of a scalar-layout struct the next member overlaps the AIR
    /// padding of, or an element of an array strided at the tight size.
    /// Loads and stores through them use the tight form
    /// (`loadMember`, `storeMember`); anything else that would retype them
    /// is refused (`checkTight`).
    tight_ptrs: std.AutoHashMapUnmanaged(Id, void) = .empty,
    tmp: u32 = 0,
    inst: ?Instruction = null,
    /// The function's return type.
    ret_ty: Id = 0,
    /// Block label -> position in the function, and each block's successor
    /// labels in terminator operand order: computed once by `validateBody`
    /// (recomputing a switch's successors per DFS step was quadratic in
    /// its case count).
    block_index: std.AutoHashMapUnmanaged(Id, u32) = .empty,
    succs: []const []const Id = &.{},
    /// How each pointer value's alignment derives from others' (access
    /// chains, copies, selects, phis), for `settleAlign`.
    flow: std.AutoArrayHashMapUnmanaged(Id, Flow) = .empty,
    /// The load or store being lowered is Volatile (`setVolatile`): every
    /// `load` / `store` it prints says `volatile`.
    vol: bool = false,

    fn refuse(f: *Fn, comptime fmt: []const u8, args: anytype) Error {
        if (f.inst) |inst| return f.l.failInst(inst, fmt, args);
        return f.l.fail(fmt, args);
    }

    fn fresh(f: *Fn) []const u8 {
        f.tmp += 1;
        return f.l.a.print("%t{d}", .{f.tmp}) catch "%t";
    }

    fn emit(f: *Fn, comptime fmt: []const u8, args: anytype) Error!void {
        try f.out.print(f.l.a, "  " ++ fmt ++ "\n", args);
    }

    fn bind(f: *Fn, id: Id, text: []const u8) Error!void {
        try f.vals.put(f.l.a, id, .{ .op = text });
    }

    /// `%name` of an instruction's result.
    fn def(f: *Fn, inst: Instruction) Error![]const u8 {
        return f.l.a.print("%{s}", .{try f.l.localName(inst.result.?)});
    }

    fn val(f: *Fn, id: Id) Error!Val {
        if (f.vals.get(id)) |v| return v;
        const l = f.l;
        if (l.m.constantOf(id) != null) return .{ .op = try l.constText(id, .value) };
        if (l.globals.get(id)) |_| {
            if (l.isModuleGlobal(id)) return .{ .op = try l.globalRef(id) };
            return f.refuse("module-scope variable %{d} is not available in this function", .{id});
        }
        if (l.function(id) != null) return f.refuse("function %{d} used as a value", .{id});
        return .{ .op = try l.a.print("%{s}", .{try l.localName(id)}) };
    }

    fn op(f: *Fn, id: Id) Error![]const u8 {
        return switch (try f.val(id)) {
            .op => |s| s,
            .input => f.refuse("a pointer to a builtin Input variable can only be loaded from or indexed", .{}),
        };
    }

    fn typeId(f: *Fn, id: Id) Error!Id {
        return f.l.m.resultTypeOf(id) orelse f.refuse("%{d} is used but never defined with a type", .{id});
    }

    fn tyOf(f: *Fn, id: Id) Error![]const u8 {
        const t = try f.typeId(id);
        if ((f.l.m.typeOf(t) orelse return f.l.ty(t, .value)) == .pointer) {
            const as = try f.ptrSpace(id) orelse return f.refuse("a pointer to a builtin Input variable can only be loaded from or indexed", .{});
            return f.l.spaceText(as);
        }
        return f.l.ty(t, .value);
    }

    /// AIR address space of pointer value `id`: recorded (`spaces`), the
    /// variable's own (`Lowerer.globalSpace`), or its type's storage class.
    fn ptrSpace(f: *Fn, id: Id) Error!?u32 {
        if (f.spaces.get(id)) |as| return as;
        if (f.l.globals.contains(id)) return f.l.globalSpace(id);
        const pt = try f.l.pointerOf(try f.typeId(id));
        return f.l.addrSpace(pt.storage_class);
    }

    /// The pointee type of pointer value `id`.
    fn pointeeOf(f: *Fn, id: Id) Error!Id {
        return switch (try f.l.typeOf(try f.typeId(id))) {
            .pointer => |p| p.pointee,
            else => f.refuse("%{d} is not a pointer", .{id}),
        };
    }

    /// `id`'s scalar width when it is an integer or integer vector.
    fn intWidth(f: *Fn, t: Id) Error!u32 {
        const s = f.l.scalarOf(t);
        return switch (f.l.m.typeOf(s) orelse return f.refuse("%{d} is not a type", .{t})) {
            .int => |i| i.width,
            else => f.refuse("type %{d} is not an integer type", .{t}),
        };
    }

    /// Whether integer (or integer vector) type `t` has Signedness 1.
    fn intSigned(f: *Fn, t: Id) Error!bool {
        const s = f.l.scalarOf(t);
        return switch (f.l.m.typeOf(s) orelse return f.refuse("%{d} is not a type", .{t})) {
            .int => |i| i.signed,
            else => f.refuse("type %{d} is not an integer type", .{t}),
        };
    }

    /// `t`'s scalar width when it is a float or float vector.
    fn floatWidth(f: *Fn, t: Id) Error!u32 {
        const s = f.l.scalarOf(t);
        return switch (f.l.m.typeOf(s) orelse return f.refuse("%{d} is not a type", .{t})) {
            .float => |x| x.width,
            else => f.refuse("type %{d} is not a float type", .{t}),
        };
    }

    /// Whether `x` and `y` have the same shape (both scalars, or vectors of
    /// one length).
    fn sameShape(f: *Fn, x: Id, y: Id) bool {
        return f.l.vecLen(x) == f.l.vecLen(y);
    }

    fn typed(f: *Fn, id: Id) Error![]const u8 {
        return f.l.a.print("{s} {s}", .{ try f.tyOf(id), try f.op(id) });
    }

    fn opId(f: *Fn, i: usize) Error!Id {
        const inst = f.inst.?;
        return inst.idAt(i) orelse f.refuse("operand {d} is not an id", .{i});
    }

    /// `%t = <text>` and returns `%t`.
    fn tmpInst(f: *Fn, comptime fmt: []const u8, args: anytype) Error![]const u8 {
        const t = f.fresh();
        try f.out.print(f.l.a, "  {s} = " ++ fmt ++ "\n", .{t} ++ args);
        return t;
    }

    fn call(f: *Fn, ret: []const u8, name: []const u8, args: []const Arg) Error![]const u8 {
        const tys = try f.l.a.alloc([]const u8, args.len);
        var list: std.ArrayList(u8) = .empty;
        for (args, tys, 0..) |arg, *t, i| {
            t.* = arg.ty;
            try list.print(f.l.a, "{s}{s} {s}", .{ if (i == 0) "" else ", ", arg.ty, arg.v });
        }
        try f.l.declare(name, ret, tys);
        if (std.mem.eql(u8, ret, "void")) {
            try f.emit("call void @{s}({s})", .{ name, list.items });
            return "";
        }
        return f.tmpInst("call {s} @{s}({s})", .{ ret, name, list.items });
    }

    fn fromMem(f: *Fn, text: []const u8, t: Id) Error![]const u8 {
        if (!f.l.isBoolish(t)) return text;
        return f.tmpInst("icmp ne {s} {s}, {s}", .{ try f.l.ty(t, .memory), text, if (f.l.vecLen(t) != null) "zeroinitializer" else "0" });
    }

    fn toMem(f: *Fn, text: []const u8, t: Id) Error![]const u8 {
        if (!f.l.isBoolish(t)) return text;
        return f.tmpInst("zext {s} {s} to {s}", .{ try f.l.ty(t, .value), text, try f.l.ty(t, .memory) });
    }

    const Flow = struct {
        /// The pointers this one is computed from: an access chain's base,
        /// a copy's or cast's source, a select's or phi's operands.
        srcs: []const Id,
        /// What the instruction itself allows: an access chain's offsets and
        /// strides (none for the others).
        cap: u64 = std.math.maxInt(u64),
    };

    /// Every pointer value of the function in `flow`, aligned as its
    /// derivation allows given everything else's: the greatest fixpoint of
    /// `flow` from the function's leaves (parameters, variables, loaded
    /// pointers: `knownAlign`), by a worklist (an alignment only shrinks
    /// and is a power of two, so each value changes at most 64 times).
    /// Values are capped at the pointee's natural alignment, which is what
    /// no knowledge means (an access never claims more than its type's
    /// alignment, and a member's type is no more aligned than its parent,
    /// so the cap changes no access).
    ///
    /// A phi's alignment is used where the phi is used, which comes before
    /// its back edges' values are lowered, so `lower` lowers the module
    /// again with what this learned. Solving the whole function here makes
    /// that one more round however the phis nest: with each phi taking its
    /// incoming values' alignments as the round had lowered them, a fact
    /// crossed one loop level per round, and K nested loops whose innermost
    /// back edge carried a packed member's pointer took K + 1 rounds, each
    /// lowering the whole module into the same arena (quadratic time and
    /// memory; 1.66 GB for K = 800).
    fn settleAlign(f: *Fn) Error!std.AutoHashMapUnmanaged(Id, u64) {
        const l = f.l;
        const a = l.a;
        const top = std.math.maxInt(u64);
        var nat: std.AutoHashMapUnmanaged(Id, u64) = .empty;
        var got: std.AutoHashMapUnmanaged(Id, u64) = .empty;
        var users: std.AutoHashMapUnmanaged(Id, std.ArrayList(Id)) = .empty;
        for (f.flow.keys(), f.flow.values()) |id, fl| {
            try nat.put(a, id, l.quietNatural(id) orelse top);
            try got.put(a, id, top);
            for (fl.srcs) |src| {
                if (!f.flow.contains(src)) continue;
                const gop = try users.getOrPut(a, src);
                if (!gop.found_existing) gop.value_ptr.* = .empty;
                try gop.value_ptr.append(a, id);
            }
        }
        var work: std.ArrayList(Id) = .empty;
        var queued: std.AutoHashMapUnmanaged(Id, void) = .empty;
        var i = f.flow.count();
        while (i > 0) {
            i -= 1;
            const id = f.flow.keys()[i];
            try work.append(a, id);
            try queued.put(a, id, {});
        }
        while (work.pop()) |id| {
            _ = queued.remove(id);
            const fl = f.flow.get(id).?;
            var new: u64 = @min(fl.cap, nat.get(id).?);
            for (fl.srcs) |src| {
                const s = got.get(src) orelse leaf: {
                    const n = l.quietNatural(src) orelse top;
                    break :leaf @min(f.knownAlign(src) orelse n, n);
                };
                new = @min(new, s);
            }
            if (got.get(id).? <= new) continue;
            try got.put(a, id, new);
            if (users.get(id)) |us| for (us.items) |u| {
                if ((try queued.getOrPut(a, u)).found_existing) continue;
                try work.append(a, u);
            };
        }
        var it = got.iterator();
        while (it.next()) |e| {
            if (e.value_ptr.* >= nat.get(e.key_ptr.*).?) continue;
            try l.noteFloor(l.m.resultTypeOf(e.key_ptr.*).?, e.value_ptr.*);
        }
        return got;
    }

    /// The alignment of an Aligned memory operand starting at operand `i`,
    /// else the pointer's known alignment capped at `natural` (a member of a
    /// packed struct, `aligns`), else `natural`.
    fn accessAlign(f: *Fn, i: usize, ptr: Id, natural: u64) Error!u64 {
        const inst = f.inst.?;
        if (inst.intAt(i)) |bits| if (bits & 2 != 0) {
            const v = inst.intAt(i + 1) orelse return f.refuse("Aligned without a literal", .{});
            if (v == 0 or !std.math.isPowerOfTwo(v) or v > 1 << 32) return f.refuse("Aligned {d} is not a power of two", .{v});
            return v;
        };
        if (f.knownAlign(ptr)) |known| return @min(natural, known);
        return natural;
    }

    /// What pointer `id` is known to be aligned to beyond its pointee's
    /// natural alignment: an access chain's (`aligns`), or what an earlier
    /// round learned for a phi, helper parameter or untracked pointer
    /// (`Lowerer.learned`); null means natural.
    fn knownAlign(f: *Fn, id: Id) ?u64 {
        return f.aligns.get(id) orelse f.l.learnedAlign(id);
    }

    /// Record pointer `id`'s alignment (and lower its type's floor).
    fn setAlign(f: *Fn, id: Id, al: u64) Error!void {
        try f.aligns.put(f.l.a, id, al);
        if (f.l.m.resultTypeOf(id)) |pt| try f.l.noteFloor(pt, al);
    }

    /// A pointer whose origin is not tracked (loaded from memory, extracted
    /// from a composite, returned by a call) gets the floor of its type.
    fn untrackedPtr(f: *Fn, id: Id) Error!void {
        const pt = f.l.m.resultTypeOf(id) orelse return;
        if ((f.l.m.typeOf(pt) orelse return) != .pointer) return;
        if (try f.l.floorOf(pt)) |fl| try f.setAlign(id, fl);
    }

    /// `%t` holding the scalar `text` (of `scalar_ty`) in every lane of the
    /// vector type `vt`; `text` itself when `vt` is not a vector.
    fn splat(f: *Fn, vt: Id, text: []const u8) Error![]const u8 {
        const n = f.l.vecLen(vt) orelse return text;
        const vty = try f.l.ty(vt, .value);
        const ety = try f.l.ty(f.l.scalarOf(vt), .value);
        const t1 = try f.tmpInst("insertelement {s} undef, {s} {s}, i32 0", .{ vty, ety, text });
        return f.tmpInst("shufflevector {s} {s}, {s} undef, <{d} x i32> zeroinitializer", .{ vty, t1, vty, n });
    }

    /// Operand `id` as a value of type `rt`: a scalar is splatted when `rt` is
    /// a vector (OpenCL's `fmin(float4, float)` and friends).
    fn argAs(f: *Fn, arg_id: Id, rt: Id) Error![]const u8 {
        const at = try f.typeId(arg_id);
        const text = try f.op(arg_id);
        if (f.l.vecLen(rt) != null and f.l.vecLen(at) == null) return f.splat(rt, text);
        return text;
    }

    /// An integer operand converted to the width of `t`'s element (zext or
    /// trunc) and splatted to `t`'s shape.
    fn intAs(f: *Fn, arg_id: Id, t: Id, signed: bool) Error![]const u8 {
        const at = try f.typeId(arg_id);
        var text = try f.op(arg_id);
        const aw = try f.intWidth(at);
        const tw = try f.intWidth(t);
        const at_ty = try f.l.ty(at, .value);
        if (aw != tw) {
            const target = try f.l.intTyBits(at, tw);
            text = try f.tmpInst("{s} {s} {s} to {s}", .{ if (aw > tw) "trunc" else if (signed) "sext" else "zext", at_ty, text, target });
        }
        if (f.l.vecLen(t) != null and f.l.vecLen(at) == null) return f.splat(t, text);
        return text;
    }

    const Block = struct { label: Id, phis: []const Instruction, body: []const u8 };

    const Class = enum { int, float, bool };

    /// Refuse when `id`'s type is not `t` (as LLVM sees them).
    fn need(f: *Fn, id: Id, t: Id) Error!void {
        if (!std.mem.eql(u8, try f.tyOf(id), try f.l.ty(t, .value))) return f.refuse("%{d} does not have the type this instruction needs", .{id});
    }

    /// Refuse unless value `id` has SPIR-V type `t` (as stored in memory).
    fn needSame(f: *Fn, id: Id, t: Id, what: []const u8) Error!void {
        const it = try f.typeId(id);
        if (it == t or std.mem.eql(u8, try f.l.ty(it, .memory), try f.l.ty(t, .memory))) return;
        return f.refuse("{s} %{d} does not have the type %{d}", .{ what, id, t });
    }

    /// Refuse unless type `t` is a pointer in storage class `sc` to
    /// `pointee` (the same id, or a type with the same AIR memory form). An
    /// access chain's or OpCopyObject's result aliases or indexes its base,
    /// so a result type that says otherwise (a value type, another storage
    /// class, another pointee) would be printed for the base's IR value and
    /// used as that type: an access chain with no indexes and result type
    /// `u32` used to alias the base pointer, which an `add` then took as an
    /// integer and the assembler refused by IR line.
    fn needPointerTo(f: *Fn, t: Id, sc: u32, pointee: Id) Error!void {
        const l = f.l;
        const p = switch (try l.typeOf(t)) {
            .pointer => |p| p,
            else => return f.refuse("the result type %{d} is not a pointer type", .{t}),
        };
        if (p.storage_class != sc) {
            const got = if (reader.enumerant(.storage_class, p.storage_class)) |e| e.name else "an unknown storage class";
            const want = if (reader.enumerant(.storage_class, sc)) |e| e.name else "an unknown storage class";
            return f.refuse("the result type %{d} is a pointer into {s} memory, the base's is {s}", .{ t, got, want });
        }
        if (p.pointee == pointee or std.mem.eql(u8, try l.ty(p.pointee, .memory), try l.ty(pointee, .memory))) return;
        return f.refuse("the result type %{d} points to %{d}, but the instruction reaches %{d}", .{ t, p.pointee, pointee });
    }

    /// Refuse unless `t` is a scalar or vector of `class`.
    fn needClass(f: *Fn, t: Id, class: Class) Error!void {
        const ok = switch (f.l.m.typeOf(f.l.scalarOf(t)) orelse return f.refuse("%{d} is not a type", .{t})) {
            .int => class == .int,
            .float => class == .float,
            .bool => class == .bool,
            else => false,
        };
        if (!ok) return f.refuse("type %{d} is not {s}", .{ t, switch (class) {
            .int => "an integer type",
            .float => "a float type",
            .bool => "a bool type",
        } });
        _ = try f.l.ty(t, .value);
    }

    /// A bool (vector) shaped like `t`.
    fn needBoolLike(f: *Fn, b: Id, t: Id) Error!void {
        try f.needClass(b, .bool);
        if (f.l.vecLen(b) != f.l.vecLen(t)) return f.refuse("the bool operand or result is not shaped like the values", .{});
    }

    /// The operand-type rules of SPIR-V's validator that the lowering relies
    /// on: malformed input is refused here rather than printed as IR the
    /// assembler's Builder asserts on.
    fn checkTypes(f: *Fn, inst: Instruction) Error!void {
        const l = f.l;
        const rt = inst.result_type orelse 0;
        switch (inst.op) {
            .OpIAdd, .OpISub, .OpIMul, .OpSDiv, .OpUDiv, .OpSRem, .OpUMod, .OpSMod, .OpBitwiseAnd, .OpBitwiseOr, .OpBitwiseXor => {
                try f.needClass(rt, .int);
                try f.need(try f.opId(0), rt);
                try f.need(try f.opId(1), rt);
            },
            .OpFAdd, .OpFSub, .OpFMul, .OpFDiv, .OpFRem, .OpFMod => {
                try f.needClass(rt, .float);
                try f.need(try f.opId(0), rt);
                try f.need(try f.opId(1), rt);
            },
            .OpLogicalAnd, .OpLogicalOr => {
                try f.needClass(rt, .bool);
                try f.need(try f.opId(0), rt);
                try f.need(try f.opId(1), rt);
            },
            .OpSNegate, .OpNot, .OpBitReverse => {
                try f.needClass(rt, .int);
                try f.need(try f.opId(0), rt);
            },
            .OpFNegate => {
                try f.needClass(rt, .float);
                try f.need(try f.opId(0), rt);
            },
            .OpLogicalNot => {
                try f.needClass(rt, .bool);
                try f.need(try f.opId(0), rt);
            },
            .OpShiftLeftLogical, .OpShiftRightLogical, .OpShiftRightArithmetic => {
                try f.needClass(rt, .int);
                try f.need(try f.opId(0), rt);
                try f.needClass(try f.typeId(try f.opId(1)), .int);
                if (!f.sameShape(try f.typeId(try f.opId(1)), rt)) return f.refuse("the shift amount is not shaped like the base", .{});
            },
            .OpIEqual, .OpINotEqual, .OpUGreaterThan, .OpSGreaterThan, .OpUGreaterThanEqual, .OpSGreaterThanEqual, .OpULessThan, .OpSLessThan, .OpULessThanEqual, .OpSLessThanEqual, .OpFOrdEqual, .OpFUnordEqual, .OpFOrdNotEqual, .OpFUnordNotEqual, .OpFOrdLessThan, .OpFUnordLessThan, .OpFOrdGreaterThan, .OpFUnordGreaterThan, .OpFOrdLessThanEqual, .OpFUnordLessThanEqual, .OpFOrdGreaterThanEqual, .OpFUnordGreaterThanEqual, .OpOrdered, .OpUnordered, .OpLessOrGreater, .OpLogicalEqual, .OpLogicalNotEqual => {
                const x = try f.opId(0);
                const xt = try f.typeId(x);
                const class: Class = switch (inst.op) {
                    .OpIEqual, .OpINotEqual, .OpUGreaterThan, .OpSGreaterThan, .OpUGreaterThanEqual, .OpSGreaterThanEqual, .OpULessThan, .OpSLessThan, .OpULessThanEqual, .OpSLessThanEqual => .int,
                    .OpLogicalEqual, .OpLogicalNotEqual => .bool,
                    else => .float,
                };
                try f.needClass(xt, class);
                try f.need(try f.opId(1), xt);
                try f.needBoolLike(rt, xt);
            },
            .OpIsNan, .OpIsInf, .OpIsFinite => {
                const xt = try f.typeId(try f.opId(0));
                try f.needClass(xt, .float);
                try f.needBoolLike(rt, xt);
            },
            .OpSelect => {
                switch (try l.typeOf(rt)) {
                    .@"struct", .array, .matrix, .pointer => {},
                    else => {
                        const ct = try f.typeId(try f.opId(0));
                        try f.needClass(ct, .bool);
                        if (l.vecLen(ct) != null and l.vecLen(ct) != l.vecLen(rt)) return f.refuse("the condition is not shaped like the values", .{});
                    },
                }
                if ((try l.typeOf(rt)) != .pointer) {
                    try f.need(try f.opId(1), rt);
                    try f.need(try f.opId(2), rt);
                }
            },
            .OpBitcast => {
                const xt = try f.typeId(try f.opId(0));
                const from = try l.typeOf(xt);
                const to = try l.typeOf(rt);
                if (from != .pointer and to != .pointer) {
                    for ([_]Id{ xt, rt }) |t| switch (try l.typeOf(l.scalarOf(t))) {
                        .int, .float => {},
                        else => return f.refuse("OpBitcast of a non-numeric type", .{}),
                    };
                    _ = try l.ty(xt, .value);
                    _ = try l.ty(rt, .value);
                    const fb = @as(u64, l.m.scalarWidth(xt) orelse 0) * (l.vecLen(xt) orelse 1);
                    const tb = @as(u64, l.m.scalarWidth(rt) orelse 0) * (l.vecLen(rt) orelse 1);
                    if (fb != tb) return f.refuse("OpBitcast between {d} and {d} bits", .{ fb, tb });
                }
            },
            .OpVectorExtractDynamic => {
                const vt = try f.typeId(try f.opId(0));
                if (l.vecLen(vt) == null) return f.refuse("not a vector", .{});
                if (!std.mem.eql(u8, try l.ty(l.scalarOf(vt), .value), try l.ty(rt, .value))) return f.refuse("the result is not the vector's component type", .{});
                try f.needClass(try f.typeId(try f.opId(1)), .int);
                if (l.vecLen(try f.typeId(try f.opId(1))) != null) return f.refuse("the index is not a scalar", .{});
            },
            .OpVectorInsertDynamic => {
                if (l.vecLen(rt) == null) return f.refuse("not a vector", .{});
                try f.need(try f.opId(0), rt);
                try f.need(try f.opId(1), l.scalarOf(rt));
                try f.needClass(try f.typeId(try f.opId(2)), .int);
                if (l.vecLen(try f.typeId(try f.opId(2))) != null) return f.refuse("the index is not a scalar", .{});
            },
            .OpVectorTimesScalar => {
                if (l.vecLen(rt) == null) return f.refuse("not a vector", .{});
                try f.needClass(rt, .float);
                try f.need(try f.opId(0), rt);
                try f.need(try f.opId(1), l.scalarOf(rt));
            },
            .OpDot => {
                const vt = try f.typeId(try f.opId(0));
                if (l.vecLen(vt) == null) return f.refuse("OpDot of a non-vector", .{});
                try f.needClass(vt, .float);
                try f.need(try f.opId(1), vt);
                if (!std.mem.eql(u8, try l.ty(l.scalarOf(vt), .value), try l.ty(rt, .value))) return f.refuse("the result is not the vectors' component type", .{});
            },
            .OpAny, .OpAll => {
                try f.needClass(try f.typeId(try f.opId(0)), .bool);
                try f.needClass(rt, .bool);
                if (l.vecLen(rt) != null) return f.refuse("the result is not a scalar bool", .{});
            },
            .OpIAddCarry, .OpISubBorrow, .OpUMulExtended, .OpSMulExtended => {
                const xt = try f.typeId(try f.opId(0));
                try f.needClass(xt, .int);
                try f.need(try f.opId(1), xt);
            },
            .OpBitFieldUExtract, .OpBitFieldSExtract, .OpBitFieldInsert => {
                try f.needClass(rt, .int);
                try f.need(try f.opId(0), rt);
                if (inst.op == .OpBitFieldInsert) try f.need(try f.opId(1), rt);
                const first: usize = if (inst.op == .OpBitFieldInsert) 2 else 1;
                for (first..first + 2) |i| {
                    const t = try f.typeId(try f.opId(i));
                    try f.needClass(t, .int);
                    if (l.vecLen(t) != null) return f.refuse("Offset and Count must be scalars", .{});
                }
            },
            .OpCompositeInsert => try f.needSame(try f.opId(1), rt, "the composite"),
            .OpCopyObject => {
                const src = try f.opId(0);
                const st = try f.typeId(src);
                switch (try l.typeOf(st)) {
                    .pointer => |p| try f.needPointerTo(rt, p.storage_class, p.pointee),
                    else => try f.needSame(src, rt, "the operand"),
                }
            },
            .OpLoad => {
                const p = try f.opId(0);
                switch (try f.val(p)) {
                    .input => {},
                    .op => if (!std.mem.eql(u8, try l.ty(try f.pointeeOf(p), .memory), try l.ty(rt, .memory))) return f.refuse("the result type is not the pointer's pointee", .{}),
                }
            },
            .OpStore => {
                const p = try f.opId(0);
                switch (try f.val(p)) {
                    .input => {},
                    .op => if (!std.mem.eql(u8, try l.ty(try f.pointeeOf(p), .memory), try l.ty(try f.typeId(try f.opId(1)), .memory))) return f.refuse("the object's type is not the pointer's pointee", .{}),
                }
            },
            .OpBranchConditional => {
                const ct = try f.typeId(try f.opId(0));
                try f.needClass(ct, .bool);
                if (l.vecLen(ct) != null) return f.refuse("the condition is not a scalar bool", .{});
            },
            .OpSwitch => {
                const st = try f.typeId(try f.opId(0));
                try f.needClass(st, .int);
                if (l.vecLen(st) != null) return f.refuse("the selector is not a scalar integer", .{});
            },
            .OpReturn => if ((try l.typeOf(f.ret_ty)) != .void) return f.refuse("OpReturn in a function that returns a value", .{}),
            .OpReturnValue => {
                if ((try l.typeOf(f.ret_ty)) == .void) return f.refuse("OpReturnValue in a void function", .{});
                try f.need(try f.opId(0), f.ret_ty);
            },
            .OpFunctionCall => {
                const callee = l.function(try f.opId(0)) orelse return f.refuse("the callee is not a function", .{});
                if (!std.mem.eql(u8, try l.ty(callee.result_type, .value), try l.ty(rt, .value))) return f.refuse("the result type is not the callee's return type", .{});
            },
            .OpPtrEqual, .OpPtrNotEqual => {
                const xt = try f.tyOf(try f.opId(0));
                if (!std.mem.eql(u8, xt, try f.tyOf(try f.opId(1)))) return f.refuse("the pointers differ in type", .{});
                if ((try l.typeOf(try f.typeId(try f.opId(0)))) != .pointer) return f.refuse("not a pointer", .{});
                try f.needClass(rt, .bool);
            },
            else => {},
        }
    }

    /// How many CFG edges lead from block `pred` to block `to`: LLVM wants
    /// one phi entry per edge (a switch with two cases on one target, a
    /// conditional branch with equal targets), SPIR-V one per parent block.
    /// The number of CFG edges from block `pred` to block `to` (two switch
    /// cases on one target are two edges), from the successor lists
    /// `validateBody` computed once per block.
    fn edgeCount(f: *Fn, pred: Id, to: Id) usize {
        const bi = f.block_index.get(pred) orelse return 0;
        var n: usize = 0;
        for (f.succs[bi]) |s| n += @intFromBool(s == to);
        return n;
    }

    /// The successor labels of block `blk` (its terminator's targets, a
    /// switch's in operand order).
    fn successors(f: *Fn, blk: reader.Block, out: *std.ArrayList(Id)) Error!void {
        out.clearRetainingCapacity();
        const term = f.l.m.instructions[blk.end - 1];
        switch (term.op) {
            .OpBranch => try out.append(f.l.a, term.idAt(0) orelse return f.refuse("OpBranch without a target", .{})),
            .OpBranchConditional => for (1..3) |i| try out.append(f.l.a, term.idAt(i) orelse return f.refuse("OpBranchConditional without a target", .{})),
            .OpSwitch => {
                try out.append(f.l.a, term.idAt(1) orelse return f.refuse("OpSwitch without a default", .{}));
                var i: usize = 3;
                while (i < term.operands.len) : (i += 2) try out.append(f.l.a, term.idAt(i) orelse return f.refuse("OpSwitch case without a target", .{}));
            },
            else => {},
        }
    }

    /// The SPIR-V validity rules the IR's well-formedness rests on, checked
    /// before any of the function is printed, so a malformed module is
    /// refused with the offending instruction quoted instead of reaching the
    /// assembler as IR it rejects by line ("branch to undefined block", "use
    /// of undefined value") or, worse, accepts (a switch with a repeated
    /// case value, which Apple's front end rejects and Metal builds):
    /// every block ends in a terminator, every branch and switch target is a
    /// block of this function other than the first (the entry block has no
    /// predecessors), the control flow is reducible (Metal's compiler
    /// crashes on a loop with two entries), a switch's case literals are distinct, a phi
    /// has exactly one (value, parent) pair per predecessor block, and every
    /// value is defined in this function (or is a module-scope constant or
    /// variable, never the result of an instruction the reader let through
    /// at module scope: `Lowerer.moduleNonValues`) by an
    /// instruction that dominates its use (for a phi: dominates the end of
    /// the parent block) and comes before it in SPIR-V's block order, which
    /// the specification requires to put dominators first and the IR keeps.
    /// Uses in unreachable blocks are not dominance-checked (LLVM's rule),
    /// only ordered.
    fn validateBody(f: *Fn, func: *const reader.Function) Error!void {
        const l = f.l;
        const a = l.a;
        const insts = l.m.instructions;
        const nb = func.blocks.len;
        f.block_index = .empty;
        const index = &f.block_index;
        for (func.blocks, 0..) |blk, bi| {
            f.inst = insts[blk.first];
            if (blk.end <= blk.first + 1 or !reader.isTerminator(insts[blk.end - 1].op)) return f.refuse("block %{d} does not end in a terminator", .{blk.label});
            const gop = try index.getOrPut(a, blk.label);
            if (gop.found_existing) return f.refuse("block %{d} is declared twice", .{blk.label});
            gop.value_ptr.* = @intCast(bi);
        }
        // Where each value of this function is defined: block and position
        // (parameters before everything).
        const Def = struct { block: u32, pos: u32 };
        var defs: std.AutoHashMapUnmanaged(Id, Def) = .empty;
        for (func.params) |p| try defs.put(a, p.id, .{ .block = 0, .pos = 0 });
        for (func.blocks, 0..) |blk, bi| {
            for (blk.first + 1..blk.end) |ii| {
                const r = insts[ii].result orelse continue;
                f.inst = insts[ii];
                const gop = try defs.getOrPut(a, r);
                if (gop.found_existing) return f.refuse("%{d} is defined twice", .{r});
                gop.value_ptr.* = .{ .block = @intCast(bi), .pos = @intCast(ii - blk.first) };
            }
        }
        // Edges, the switch literals, predecessors.
        const preds = try a.alloc(std.ArrayList(u32), nb);
        @memset(preds, .empty);
        const succs = try a.alloc([]const Id, nb);
        // Successor positions, for the walks below.
        const succ_pos = try a.alloc([]const u32, nb);
        var succ: std.ArrayList(Id) = .empty;
        for (func.blocks, 0..) |blk, bi| {
            const term = insts[blk.end - 1];
            f.inst = term;
            try f.successors(blk, &succ);
            succs[bi] = try a.dupe(Id, succ.items);
            const pos = try a.alloc(u32, succ.items.len);
            for (succ.items, pos) |s, *sp| {
                const si = index.get(s) orelse return f.refuse("%{d} is not a block of this function", .{s});
                // SPIR-V 2.16.1: "The first block in a function definition
                // is the entry point of that function and must not be the
                // target of any branch"; LLVM's entry block has no
                // predecessors either (its IR here also holds the
                // prologue). Packed anyway, it crashed MTLCompilerService.
                if (si == 0) return f.refuse("a branch to the function's first block %{d}: SPIR-V forbids any branch to the entry block (section 2.16.1)", .{s});
                try preds[si].append(a, @intCast(bi));
                sp.* = si;
            }
            succ_pos[bi] = pos;
            if (term.op == .OpSwitch) {
                var seen: std.AutoHashMapUnmanaged(u64, void) = .empty;
                var i: usize = 2;
                while (i < term.operands.len) : (i += 2) {
                    const lit = term.intAt(i) orelse return f.refuse("OpSwitch case {d} is not a literal", .{(i - 2) / 2});
                    if ((try seen.getOrPut(a, lit)).found_existing) return f.refuse("OpSwitch has the case literal {d} twice", .{lit});
                }
            }
        }
        // Reverse postorder from the entry block, then immediate dominators
        // (Cooper, Harvey and Kennedy's iteration).
        const unvisited = std.math.maxInt(u32);
        const rpo_num = try a.alloc(u32, nb);
        @memset(rpo_num, unvisited);
        var order: std.ArrayList(u32) = .empty;
        // Retreating edges of the depth-first walk (to a block still on the
        // walk's stack): each must be a back edge, whose target dominates
        // its source, or the graph is irreducible (checked below).
        const Edge = struct { from: u32, to: u32 };
        var retreating: std.ArrayList(Edge) = .empty;
        {
            const Frame = struct { b: u32, next: u32 };
            var stack: std.ArrayList(Frame) = .empty;
            const seen = try a.alloc(bool, nb);
            @memset(seen, false);
            const on_stack = try a.alloc(bool, nb);
            @memset(on_stack, false);
            seen[0] = true;
            on_stack[0] = true;
            try stack.append(a, .{ .b = 0, .next = 0 });
            while (stack.items.len > 0) {
                const top = &stack.items[stack.items.len - 1];
                const ss = succ_pos[top.b];
                if (top.next < ss.len) {
                    const s = ss[top.next];
                    top.next += 1;
                    if (!seen[s]) {
                        seen[s] = true;
                        on_stack[s] = true;
                        try stack.append(a, .{ .b = s, .next = 0 });
                    } else if (on_stack[s]) try retreating.append(a, .{ .from = top.b, .to = s });
                    continue;
                }
                try order.append(a, top.b);
                on_stack[top.b] = false;
                _ = stack.pop();
            }
            std.mem.reverse(u32, order.items);
            for (order.items, 0..) |b, k| rpo_num[b] = @intCast(k);
        }
        f.succs = succs;
        const idom = try a.alloc(u32, nb);
        @memset(idom, unvisited);
        idom[0] = 0;
        var changed = true;
        while (changed) {
            changed = false;
            for (order.items[1..]) |b| {
                var new: u32 = unvisited;
                for (preds[b].items) |p| {
                    if (idom[p] == unvisited) continue;
                    if (new == unvisited) {
                        new = p;
                        continue;
                    }
                    var x = p;
                    var y = new;
                    while (x != y) {
                        while (rpo_num[x] > rpo_num[y]) x = idom[x];
                        while (rpo_num[y] > rpo_num[x]) y = idom[y];
                    }
                    new = x;
                }
                if (new != idom[b]) {
                    idom[b] = new;
                    changed = true;
                }
            }
        }
        // Dominator-tree intervals: `d` dominates `b` iff pre[d] <= pre[b]
        // and post[b] <= post[d].
        const pre = try a.alloc(u32, nb);
        const post = try a.alloc(u32, nb);
        {
            const kids = try a.alloc(std.ArrayList(u32), nb);
            @memset(kids, .empty);
            for (order.items[1..]) |b| try kids[idom[b]].append(a, b);
            const Frame = struct { b: u32, next: u32 };
            var stack: std.ArrayList(Frame) = .empty;
            var clock: u32 = 0;
            try stack.append(a, .{ .b = 0, .next = 0 });
            pre[0] = 0;
            while (stack.items.len > 0) {
                const top = &stack.items[stack.items.len - 1];
                if (top.next < kids[top.b].items.len) {
                    const c = kids[top.b].items[top.next];
                    top.next += 1;
                    clock += 1;
                    pre[c] = clock;
                    try stack.append(a, .{ .b = c, .next = 0 });
                    continue;
                }
                clock += 1;
                post[top.b] = clock;
                _ = stack.pop();
            }
        }
        const reachable = struct {
            fn is(r: []const u32, b: u32) bool {
                return r[b] != std.math.maxInt(u32);
            }
        }.is;
        const dominates = struct {
            fn is(pr: []const u32, po: []const u32, d: u32, b: u32) bool {
                return pr[d] <= pr[b] and po[b] <= po[d];
            }
        }.is;
        // Irreducible control flow: a cycle entered at two blocks (OpenCL C
        // `goto` into a loop; Kernel modules need not be structured), found
        // as a retreating edge whose target does not dominate its source
        // (a graph is reducible iff every retreating edge of a depth-first
        // walk is a back edge). Metal's compiler cannot take it:
        // MTLCompilerService crashes (XPC_ERROR_CONNECTION_INTERRUPTED) at
        // pipeline creation, also on Apple's own assembly of the same IR.
        for (retreating.items) |e| {
            if (dominates(pre, post, e.to, e.from)) continue;
            f.inst = insts[func.blocks[e.from].end - 1];
            return f.refuse("irreducible control flow: this branch to %{d} closes a cycle that can also be entered elsewhere than at %{d} (a loop with two entry blocks, as OpenCL C `goto` makes); MTLCompilerService crashes on it", .{ func.blocks[e.to].label, func.blocks[e.to].label });
        }
        const owners = try l.valueOwners();
        const non_values = try l.moduleNonValues();
        for (func.blocks, 0..) |blk, bi_usize| {
            const bi: u32 = @intCast(bi_usize);
            for (blk.first + 1..blk.end) |ii| {
                const inst = insts[ii];
                f.inst = inst;
                const pos: u32 = @intCast(ii - blk.first);
                if (inst.op == .OpPhi) {
                    if (inst.operands.len == 0 or inst.operands.len % 2 != 0) return f.refuse("OpPhi needs (value, parent) pairs", .{});
                    var listed: std.AutoHashMapUnmanaged(u32, void) = .empty;
                    var i: usize = 0;
                    while (i + 1 < inst.operands.len) : (i += 2) {
                        const parent = inst.idAt(i + 1) orelse return f.refuse("OpPhi operand {d} is not an id", .{i + 1});
                        const pi = index.get(parent) orelse return f.refuse("%{d} is not a block of this function", .{parent});
                        if (std.mem.findScalar(u32, preds[bi].items, pi) == null) return f.refuse("%{d} is not a parent block of %{d}", .{ parent, blk.label });
                        if ((try listed.getOrPut(a, pi)).found_existing) return f.refuse("OpPhi lists the parent %{d} twice", .{parent});
                        const v = inst.idAt(i) orelse return f.refuse("OpPhi operand {d} is not an id", .{i});
                        if (defs.get(v)) |d| {
                            if (reachable(rpo_num, pi) and !(reachable(rpo_num, d.block) and dominates(pre, post, d.block, pi))) return f.refuse("%{d} does not dominate the end of the parent block %{d}", .{ v, parent });
                        } else {
                            if (owners.contains(v)) return f.refuse("%{d} is defined in another function", .{v});
                            if (non_values.get(v)) |nop| return f.refuse("%{d} is the result of an {t} at module scope, where SPIR-V allows no computation", .{ v, nop });
                        }
                    }
                    for (preds[bi].items) |p| if (!listed.contains(p)) return f.refuse("OpPhi has no value for the parent block %{d}", .{func.blocks[p].label});
                    continue;
                }
                for (inst.operands) |o| {
                    const v = o.asId() orelse continue;
                    const d = defs.get(v) orelse {
                        if (owners.contains(v) and !index.contains(v)) return f.refuse("%{d} is defined in another function", .{v});
                        if (non_values.get(v)) |nop| if (!l.isNonSemantic(inst)) return f.refuse("%{d} is the result of an {t} at module scope, where SPIR-V allows no computation", .{ v, nop });
                        continue;
                    };
                    const before = d.block < bi or (d.block == bi and d.pos < pos);
                    if (!before) return f.refuse("%{d} is used before its definition in SPIR-V's block order", .{v});
                    if (reachable(rpo_num, bi) and !(reachable(rpo_num, d.block) and dominates(pre, post, d.block, bi))) return f.refuse("the definition of %{d} does not dominate this use", .{v});
                }
            }
        }
        f.inst = null;
    }

    fn lowerBody(f: *Fn, func: *const reader.Function, prologue: []const u8) Error![]const u8 {
        const l = f.l;
        var blocks: std.ArrayList(Block) = .empty;
        if (func.blocks.len == 0) return f.refuse("function %{d} has no blocks", .{func.id});
        try f.validateBody(func);
        f.ret_ty = func.result_type;
        for (func.blocks, 0..) |blk, bi| {
            f.out = .empty;
            if (bi == 0) try f.out.appendSlice(l.a, prologue);
            var phis: std.ArrayList(Instruction) = .empty;
            for (l.m.instructions[blk.first + 1 .. blk.end]) |inst| {
                f.inst = inst;
                if (inst.op == .OpPhi) {
                    try phis.append(l.a, inst);
                    continue;
                }
                f.lowerInst(inst) catch |err| return if (err == error.Unsupported) l.quote(inst) else err;
            }
            try blocks.append(l.a, .{ .label = blk.label, .phis = phis.items, .body = f.out.items });
        }
        var text: std.ArrayList(u8) = .empty;
        var ptr_phis: std.ArrayList(struct { id: Id, ty: Id }) = .empty;
        for (blocks.items) |blk| {
            try text.print(l.a, "b{d}:\n", .{blk.label});
            for (blk.phis) |inst| {
                f.inst = inst;
                const rt = inst.result_type.?;
                if (inst.operands.len == 0 or inst.operands.len % 2 != 0) return f.refuse("OpPhi needs (value, parent) pairs", .{});
                switch (try l.typeOf(rt)) {
                    .void, .function, .forward_pointer, .other => return f.refuse("OpPhi of a type without values", .{}),
                    else => {},
                }
                const is_ptr = (try l.typeOf(rt)) == .pointer;
                const pty = if (is_ptr) try f.tyOf(inst.result.?) else try l.ty(rt, .value);
                try text.print(l.a, "  {s} = phi {s} ", .{ try f.def(inst), pty });
                var i: usize = 0;
                var first = true;
                // A pointer phi is aligned as its least aligned incoming
                // value (`settleAlign`, below).
                var incoming: std.ArrayList(Id) = .empty;
                while (i + 1 < inst.operands.len) : (i += 2) {
                    const v = inst.idAt(i) orelse return f.refuse("OpPhi operand {d} is not an id", .{i});
                    const pred = inst.idAt(i + 1) orelse return f.refuse("OpPhi operand {d} is not an id", .{i + 1});
                    if (is_ptr and !std.mem.eql(u8, try f.tyOf(v), pty)) return f.refuse("a phi of pointers into different address spaces (a BufferBlock and a Uniform block) needs variable pointers", .{});
                    if (!is_ptr) {
                        if ((try f.typeId(v)) != rt and !std.mem.eql(u8, try f.tyOf(v), pty)) return f.refuse("the incoming value %{d} does not have the phi's type", .{v});
                    } else if ((try l.typeOf(try f.typeId(v))) != .pointer) return f.refuse("the incoming value %{d} is not a pointer", .{v});
                    if (f.tight_ptrs.contains(v)) return f.refuse("%{d} points to a struct stored without its tail padding (scalar block layout); a phi would access it at its full AIR size", .{v});
                    if (is_ptr) try incoming.append(l.a, v);
                    const vtext = try f.op(v);
                    const edges = f.edgeCount(pred, blk.label);
                    if (edges == 0) return f.refuse("%{d} is not a parent block of %{d}", .{ pred, blk.label });
                    for (0..edges) |_| {
                        try text.print(l.a, "{s}[ {s}, %b{d} ]", .{ if (first) "" else ", ", vtext, pred });
                        first = false;
                    }
                }
                if (is_ptr) {
                    try f.flow.put(l.a, inst.result.?, .{ .srcs = incoming.items });
                    try ptr_phis.append(l.a, .{ .id = inst.result.?, .ty = rt });
                }
                try text.append(l.a, '\n');
            }
            try text.appendSlice(l.a, blk.body);
        }
        f.inst = null;
        // The pointer phis' alignments, learned for the next round
        // (`lower`): their uses were lowered before the back edges' values.
        const settled = try f.settleAlign();
        for (ptr_phis.items) |ph| if (settled.get(ph.id)) |al| try l.learnId(ph.id, ph.ty, al);
        return text.items;
    }

    fn binName(o: spec.Opcode) ?[]const u8 {
        return switch (o) {
            .OpIAdd => "add",
            .OpISub => "sub",
            .OpIMul => "mul",
            .OpSDiv => "sdiv",
            .OpUDiv => "udiv",
            .OpSRem => "srem",
            .OpUMod => "urem",
            .OpFAdd => "fadd",
            .OpFSub => "fsub",
            .OpFMul => "fmul",
            .OpFDiv => "fdiv",
            .OpFRem => "frem",
            .OpBitwiseAnd, .OpLogicalAnd => "and",
            .OpBitwiseOr, .OpLogicalOr => "or",
            .OpBitwiseXor => "xor",
            else => null,
        };
    }

    fn cmpName(o: spec.Opcode) ?[]const u8 {
        return switch (o) {
            .OpIEqual, .OpLogicalEqual => "icmp eq",
            .OpINotEqual, .OpLogicalNotEqual => "icmp ne",
            .OpUGreaterThan => "icmp ugt",
            .OpSGreaterThan => "icmp sgt",
            .OpUGreaterThanEqual => "icmp uge",
            .OpSGreaterThanEqual => "icmp sge",
            .OpULessThan => "icmp ult",
            .OpSLessThan => "icmp slt",
            .OpULessThanEqual => "icmp ule",
            .OpSLessThanEqual => "icmp sle",
            .OpFOrdEqual => "fcmp oeq",
            .OpFUnordEqual => "fcmp ueq",
            .OpFOrdNotEqual => "fcmp one",
            .OpFUnordNotEqual => "fcmp une",
            .OpFOrdLessThan => "fcmp olt",
            .OpFUnordLessThan => "fcmp ult",
            .OpFOrdGreaterThan => "fcmp ogt",
            .OpFUnordGreaterThan => "fcmp ugt",
            .OpFOrdLessThanEqual => "fcmp ole",
            .OpFUnordLessThanEqual => "fcmp ule",
            .OpFOrdGreaterThanEqual => "fcmp oge",
            .OpFUnordGreaterThanEqual => "fcmp uge",
            .OpOrdered => "fcmp ord",
            .OpUnordered => "fcmp uno",
            .OpLessOrGreater => "fcmp one",
            else => null,
        };
    }

    fn lowerInst(f: *Fn, inst: Instruction) Error!void {
        const l = f.l;
        const a = l.a;
        // NonSemantic.* instructions (debug info, debug printf) may be
        // removed without changing semantics (SPV_KHR_non_semantic_info).
        if (l.isNonSemantic(inst)) return;
        if (inst.result) |rid| if (l.decorated.contains(rid)) try f.checkDecorations(inst);
        try f.checkTypes(inst);
        try f.checkRelocated(inst);
        try f.checkTight(inst);
        if (binName(inst.op)) |name| {
            return f.emit("{s} = {s} {s} {s}, {s}", .{ try f.def(inst), name, try l.ty(inst.result_type.?, .value), try f.op(try f.opId(0)), try f.op(try f.opId(1)) });
        }
        if (cmpName(inst.op)) |name| {
            return f.emit("{s} = {s} {s}, {s}", .{ try f.def(inst), name, try f.typed(try f.opId(0)), try f.op(try f.opId(1)) });
        }
        switch (inst.op) {
            .OpNop, .OpLine, .OpNoLine, .OpSelectionMerge, .OpLoopMerge, .OpLifetimeStart, .OpLifetimeStop, .OpModuleProcessed => {},
            .OpUndef => try f.bind(inst.result.?, "undef"),
            .OpCopyObject => {
                const src = try f.opId(0);
                try f.vals.put(a, inst.result.?, try f.val(src));
                try f.copyPtrInfo(src, inst.result.?);
            },

            .OpShiftLeftLogical, .OpShiftRightLogical, .OpShiftRightArithmetic => {
                const rt = inst.result_type.?;
                const amount = try f.intAs(try f.opId(1), rt, false);
                const name = switch (inst.op) {
                    .OpShiftLeftLogical => "shl",
                    .OpShiftRightLogical => "lshr",
                    else => "ashr",
                };
                try f.emit("{s} = {s} {s} {s}, {s}", .{ try f.def(inst), name, try l.ty(rt, .value), try f.op(try f.opId(0)), amount });
            },
            .OpSNegate => {
                const rt = inst.result_type.?;
                try f.emit("{s} = sub {s} {s}, {s}", .{ try f.def(inst), try l.ty(rt, .value), try l.zeroOf(rt), try f.op(try f.opId(0)) });
            },
            .OpFNegate => try f.emit("{s} = fneg {s}", .{ try f.def(inst), try f.typed(try f.opId(0)) }),
            .OpNot => {
                const rt = inst.result_type.?;
                try f.emit("{s} = xor {s}, {s}", .{ try f.def(inst), try f.typed(try f.opId(0)), try l.intConst(rt, -1) });
            },
            .OpLogicalNot => {
                const rt = inst.result_type.?;
                try f.emit("{s} = xor {s}, {s}", .{ try f.def(inst), try f.typed(try f.opId(0)), try l.splatConst(rt, "true") });
            },
            .OpSMod => {
                const rt = inst.result_type.?;
                const t = try l.ty(rt, .value);
                const x = try f.op(try f.opId(0));
                const y = try f.op(try f.opId(1));
                const z = try l.zeroOf(rt);
                const r = try f.tmpInst("srem {s} {s}, {s}", .{ t, x, y });
                const xr = try f.tmpInst("xor {s} {s}, {s}", .{ t, r, y });
                const neg = try f.tmpInst("icmp slt {s} {s}, {s}", .{ t, xr, z });
                const nz = try f.tmpInst("icmp ne {s} {s}, {s}", .{ t, r, z });
                const c = try f.tmpInst("and {s} {s}, {s}", .{ try l.boolTy(rt), neg, nz });
                const rb = try f.tmpInst("add {s} {s}, {s}", .{ t, r, y });
                try f.emit("{s} = select {s} {s}, {s} {s}, {s} {s}", .{ try f.def(inst), try l.boolTy(rt), c, t, rb, t, r });
            },
            .OpFMod => {
                const rt = inst.result_type.?;
                const t = try l.ty(rt, .value);
                const bt = try l.boolTy(rt);
                const x = try f.op(try f.opId(0));
                const y = try f.op(try f.opId(1));
                const z = try l.zeroOf(rt);
                const r = try f.tmpInst("frem {s} {s}, {s}", .{ t, x, y });
                const nz = try f.tmpInst("fcmp une {s} {s}, {s}", .{ t, r, z });
                const rn = try f.tmpInst("fcmp olt {s} {s}, {s}", .{ t, r, z });
                const yn = try f.tmpInst("fcmp olt {s} {s}, {s}", .{ t, y, z });
                const differ = try f.tmpInst("xor {s} {s}, {s}", .{ bt, rn, yn });
                const c = try f.tmpInst("and {s} {s}, {s}", .{ bt, nz, differ });
                const rb = try f.tmpInst("fadd {s} {s}, {s}", .{ t, r, y });
                try f.emit("{s} = select {s} {s}, {s} {s}, {s} {s}", .{ try f.def(inst), bt, c, t, rb, t, r });
            },
            .OpSelect => try f.select(inst),
            .OpIsNan => try f.emit("{s} = fcmp uno {s}, {s}", .{ try f.def(inst), try f.typed(try f.opId(0)), try f.op(try f.opId(0)) }),
            .OpIsInf, .OpIsFinite => {
                const x = try f.opId(0);
                const xt = try f.typeId(x);
                const abs = try f.call(try l.ty(xt, .value), try a.print("llvm.fabs.{s}", .{try l.sfx(xt)}), &.{.{ .ty = try l.ty(xt, .value), .v = try f.op(x) }});
                try f.emit("{s} = fcmp {s} {s} {s}, {s}", .{ try f.def(inst), if (inst.op == .OpIsInf) "oeq" else "one", try l.ty(xt, .value), abs, try l.floatConst(xt, std.math.inf(f64)) });
            },

            .OpUConvert, .OpSConvert, .OpFConvert => try f.convertWidth(inst),
            .OpConvertFToS, .OpConvertFToU, .OpConvertSToF, .OpConvertUToF => try f.convertKind(inst),
            .OpBitcast, .OpPtrCastToGeneric, .OpGenericCastToPtr => {
                const x = try f.opId(0);
                const xt = try f.typeId(x);
                const rt = inst.result_type.?;
                const from = try l.typeOf(xt);
                const to = try l.typeOf(rt);
                if (from == .pointer or to == .pointer) {
                    if (inst.op == .OpBitcast and (from != .pointer or to != .pointer)) return f.pointerIntCast(inst);
                    if (from != .pointer or to != .pointer) return f.refuse("conversions between pointers and integers are not supported", .{});
                    const fa = try l.addrSpace(from.pointer.storage_class);
                    const ta = try l.addrSpace(to.pointer.storage_class);
                    if (fa == null or ta == null or fa.? != ta.?) return f.refuse("a pointer cast between address spaces needs addrspacecast, which is unverified on Metal", .{});
                    try f.vals.put(a, inst.result.?, try f.val(x));
                    return f.copyPtrInfo(x, inst.result.?);
                }
                if (inst.op != .OpBitcast) return f.refuse("cast of a non-pointer", .{});
                const fty = try l.ty(xt, .value);
                const tty = try l.ty(rt, .value);
                if (std.mem.eql(u8, fty, tty)) return f.vals.put(a, inst.result.?, try f.val(x));
                try f.emit("{s} = bitcast {s} {s} to {s}", .{ try f.def(inst), fty, try f.op(x), tty });
            },

            .OpConvertUToPtr, .OpConvertPtrToU => try f.pointerIntCast(inst),
            .OpCompositeExtract => try f.compositeExtract(inst),
            .OpCompositeInsert => try f.compositeInsert(inst),
            .OpCompositeConstruct => try f.compositeConstruct(inst),
            .OpVectorShuffle => try f.vectorShuffle(inst),
            .OpVectorExtractDynamic => try f.emit("{s} = extractelement {s}, {s}", .{ try f.def(inst), try f.typed(try f.opId(0)), try f.typed(try f.opId(1)) }),
            .OpVectorInsertDynamic => try f.emit("{s} = insertelement {s}, {s}, {s}", .{ try f.def(inst), try f.typed(try f.opId(0)), try f.typed(try f.opId(1)), try f.typed(try f.opId(2)) }),
            .OpVectorTimesScalar => {
                const rt = inst.result_type.?;
                const s = try f.splat(rt, try f.op(try f.opId(1)));
                try f.emit("{s} = fmul {s} {s}, {s}", .{ try f.def(inst), try l.ty(rt, .value), try f.op(try f.opId(0)), s });
            },
            .OpDot => {
                const x = try f.opId(0);
                const xt = try f.typeId(x);
                if (l.vecLen(xt) == null) return f.refuse("OpDot of a non-vector", .{});
                const r = try f.dot(xt, try f.op(x), try f.op(try f.opId(1)));
                try f.bind(inst.result.?, r);
            },
            .OpAny, .OpAll => {
                const x = try f.opId(0);
                const xt = try f.typeId(x);
                const n = l.vecLen(xt) orelse return f.vals.put(a, inst.result.?, try f.val(x));
                const name = try a.print("air.{s}.v{d}i1", .{ if (inst.op == .OpAny) "any" else "all", n });
                try f.bind(inst.result.?, try f.call("i1", name, &.{.{ .ty = try l.ty(xt, .value), .v = try f.op(x) }}));
            },
            .OpIAddCarry, .OpISubBorrow => {
                // `{ a + b, a + b < a }` and `{ a - b, a < b }` in plain
                // instructions: `llvm.uadd/usub.with.overflow` on a vector
                // crashes MTLCompilerService (probe op_IAddCarry_v4u32).
                const x = try f.opId(0);
                const xt = try f.typeId(x);
                _ = try f.intWidth(xt);
                const t = try l.ty(xt, .value);
                const bt = try l.boolTy(xt);
                const xv = try f.op(x);
                const yv = try f.op(try f.opId(1));
                const add = inst.op == .OpIAddCarry;
                const r = try f.tmpInst("{s} {s} {s}, {s}", .{ if (add) "add" else "sub", t, xv, yv });
                const c = if (add) try f.tmpInst("icmp ult {s} {s}, {s}", .{ t, r, xv }) else try f.tmpInst("icmp ult {s} {s}, {s}", .{ t, xv, yv });
                const cw = try f.tmpInst("zext {s} {s} to {s}", .{ bt, c, t });
                try f.bind(inst.result.?, try f.pair(inst.result_type.?, t, r, cw));
            },
            .OpUMulExtended, .OpSMulExtended => {
                const x = try f.opId(0);
                const xt = try f.typeId(x);
                const t = try l.ty(xt, .value);
                const signed = inst.op == .OpSMulExtended;
                const lo_hi = try f.mulWide(xt, try f.op(x), try f.op(try f.opId(1)), signed);
                try f.bind(inst.result.?, try f.pair(inst.result_type.?, t, lo_hi[0], lo_hi[1]));
            },
            .OpBitCount => {
                const x = try f.opId(0);
                const xt = try f.typeId(x);
                _ = try f.intWidth(xt);
                _ = try f.intWidth(inst.result_type.?);
                if (!f.sameShape(xt, inst.result_type.?)) return f.refuse("OpBitCount result and operand differ in shape", .{});
                const t = try l.ty(xt, .value);
                const r = try f.call(t, try a.print("llvm.ctpop.{s}", .{try l.sfx(xt)}), &.{.{ .ty = t, .v = try f.op(x) }});
                try f.bind(inst.result.?, try f.resize(r, xt, inst.result_type.?));
            },
            .OpBitReverse => {
                const x = try f.opId(0);
                const xt = try f.typeId(x);
                _ = try f.intWidth(xt);
                const t = try l.ty(xt, .value);
                try f.bind(inst.result.?, try f.call(t, try a.print("llvm.bitreverse.{s}", .{try l.sfx(xt)}), &.{.{ .ty = t, .v = try f.op(x) }}));
            },
            .OpBitFieldUExtract, .OpBitFieldSExtract, .OpBitFieldInsert => try f.bitField(inst),

            .OpVariable => {
                const pt = try l.pointerOf(inst.result_type.?);
                if (!Lowerer.isClass(pt.storage_class, .function)) return f.refuse("a variable inside a function must be in the Function storage class", .{});
                const pointee = pt.pointee;
                try l.checkSized(inst.result.?, pointee, "Function");
                const ly = try l.layout(pointee);
                const res = try f.def(inst);
                try f.emit("{s} = alloca {s}, align {d}", .{ res, try l.ty(pointee, .memory), ly.alignment });
                if (inst.idAt(1)) |init| {
                    try f.needSame(init, pointee, "the initializer");
                    const v = if (l.m.constantOf(init) != null) try l.constText(init, .memory) else try f.toMem(try f.op(init), pointee);
                    try f.emit("store {s} {s}, ptr {s}, align {d}", .{ try l.ty(pointee, .memory), v, res, ly.alignment });
                }
            },
            .OpLoad => try f.load(inst),
            .OpStore => {
                const p = try f.opId(0);
                const pv = switch (try f.val(p)) {
                    .op => |s| s,
                    .input => return f.refuse("store to a builtin Input variable", .{}),
                };
                const obj = try f.opId(1);
                const ot = try f.typeId(obj);
                try f.checkWritable(p);
                const al = try f.accessAlign(2, p, (try l.layout(ot)).alignment);
                try f.setVolatile(inst, 2, &.{p}, ot);
                defer f.vol = false;
                if (f.tight_ptrs.contains(p)) return f.storeMember(p, ot, try f.op(obj), al);
                const v = try f.toMem(try f.op(obj), ot);
                if (try f.splitAggregate(p, ot)) |ch| return f.storeSplit(ch, .{ .id = ot }, v, al);
                try f.emit("store {s}{s} {s}, {s} {s}, align {d}", .{ f.vk(), try l.ty(ot, .memory), v, try f.tyOf(p), pv, al });
            },
            .OpCopyMemory => {
                const dst = try f.opId(0);
                const src = try f.opId(1);
                const pointee = try f.pointeeOf(src);
                if (try f.pointeeOf(dst) != pointee) return f.refuse("OpCopyMemory between different pointee types", .{});
                try f.checkWritable(dst);
                const mt = try l.ty(pointee, .memory);
                const natural = (try l.layout(pointee)).alignment;
                const sal = @min(natural, f.knownAlign(src) orelse natural);
                const dal = @min(natural, f.knownAlign(dst) orelse natural);
                try f.setVolatile(inst, 2, &.{ dst, src }, pointee);
                defer f.vol = false;
                if (f.tight_ptrs.contains(src) or f.tight_ptrs.contains(dst)) {
                    const v = if (f.tight_ptrs.contains(src)) try f.loadMember(src, pointee, sal) else try f.tmpInst("load {s}{s}, {s}, align {d}", .{ f.vk(), mt, try f.typed(src), sal });
                    if (f.tight_ptrs.contains(dst)) return f.storeMember(dst, pointee, v, dal);
                    return f.emit("store {s}{s} {s}, {s}, align {d}", .{ f.vk(), mt, v, try f.typed(dst), dal });
                }
                const v = if (try f.splitAggregate(src, pointee)) |ch| try f.loadSplit(ch, .{ .id = pointee }, sal) else try f.tmpInst("load {s}{s}, {s}, align {d}", .{ f.vk(), mt, try f.typed(src), sal });
                if (try f.splitAggregate(dst, pointee)) |ch| return f.storeSplit(ch, .{ .id = pointee }, v, dal);
                try f.emit("store {s}{s} {s}, {s}, align {d}", .{ f.vk(), mt, v, try f.typed(dst), dal });
            },
            .OpAccessChain, .OpInBoundsAccessChain => try f.accessChain(inst, false),
            .OpPtrAccessChain, .OpInBoundsPtrAccessChain => try f.accessChain(inst, true),
            .OpPtrEqual, .OpPtrNotEqual => try f.emit("{s} = icmp {s} {s}, {s}", .{ try f.def(inst), if (inst.op == .OpPtrEqual) "eq" else "ne", try f.typed(try f.opId(0)), try f.op(try f.opId(1)) }),
            .OpArrayLength => return f.refuse("Metal passes a kernel no buffer lengths, so OpArrayLength has no AIR equivalent; pass the length in a uniform or push-constant block", .{}),

            .OpFunctionCall => {
                const callee = try f.opId(0);
                const info = try l.funcInfo(callee);
                const cf = l.function(callee).?;
                if (inst.operands.len - 1 != cf.params.len) return f.refuse("the callee takes {d} arguments", .{cf.params.len});
                const sym = try l.helperSym(callee);
                var args: std.ArrayList(u8) = .empty;
                var n: usize = 0;
                for (inst.operands[1..], cf.params) |o, prm| {
                    const arg = o.asId() orelse return f.refuse("argument {d} is not an id", .{n});
                    const at = try f.tyOf(arg);
                    if (!std.mem.eql(u8, at, try l.ty(prm.type, .value))) return f.refuse("argument {d} is a {s} but the parameter is a {s} (a BufferBlock pointer passed as a Uniform one needs the callee specialized)", .{ n, at, try l.ty(prm.type, .value) });
                    try args.print(a, "{s}{s} {s}", .{ if (n == 0) "" else ", ", at, try f.op(arg) });
                    // The parameter is aligned as its least aligned argument.
                    if ((try l.typeOf(prm.type)) == .pointer) if (f.knownAlign(arg)) |al| try l.learnId(prm.id, prm.type, al);
                    n += 1;
                }
                for (info.iface) |g| {
                    const text = switch (try f.val(g)) {
                        .op => |s| try a.print("{s} {s}", .{ try l.globalPtrText(g), s }),
                        .input => |in| blk: {
                            if (in.lane != null) return f.refuse("builtin %{d} reached a call as a lane pointer", .{g});
                            break :blk try a.print("{s} {s}", .{ in.param_ty, in.value });
                        },
                    };
                    try args.print(a, "{s}{s}", .{ if (n == 0) "" else ", ", text });
                    n += 1;
                }
                const rt = try l.ty(inst.result_type.?, .value);
                if (std.mem.eql(u8, rt, "void")) {
                    try f.emit("call void @{s}({s})", .{ sym, args.items });
                } else try f.emit("{s} = call {s} @{s}({s})", .{ try f.def(inst), rt, sym, args.items });
                try f.untrackedPtr(inst.result.?);
            },
            .OpExtInst => try f.extInst(inst),
            .OpExtInstWithForwardRefsKHR => return f.refuse("OpExtInstWithForwardRefsKHR of a semantic instruction set", .{}),

            .OpBranch => try f.emit("br label %b{d}", .{try f.opId(0)}),
            .OpBranchConditional => try f.emit("br {s}, label %b{d}, label %b{d}", .{ try f.typed(try f.opId(0)), try f.opId(1), try f.opId(2) }),
            .OpSwitch => {
                const sel = try f.opId(0);
                const st = try f.typeId(sel);
                const w = l.m.scalarWidth(st) orelse 32;
                const t = try l.ty(st, .value);
                var cases: std.ArrayList(u8) = .empty;
                var i: usize = 2;
                while (i + 1 < inst.operands.len) : (i += 2) {
                    try cases.print(a, "    {s} {s}, label %b{d}\n", .{ t, try intText(a, inst.intAt(i).?, w), inst.idAt(i + 1).? });
                }
                // LLVM's own layout: the assembler joins the case lines up
                // to the lone `]`.
                try f.emit("switch {s} {s}, label %b{d} [\n{s}  ]", .{ t, try f.op(sel), try f.opId(1), cases.items });
            },
            .OpReturn => try f.emit("ret void", .{}),
            .OpReturnValue => try f.emit("ret {s}", .{try f.typed(try f.opId(0))}),
            .OpUnreachable => try f.emit("unreachable", .{}),
            .OpKill, .OpTerminateInvocation, .OpDemoteToHelperInvocation => return f.refuse("fragment-only instruction (stage 5)", .{}),
            .OpControlBarrier => try f.controlBarrier(),
            .OpMemoryBarrier => try f.memoryBarrier(),
            .OpAtomicLoad, .OpAtomicStore, .OpAtomicExchange, .OpAtomicCompareExchange, .OpAtomicCompareExchangeWeak, .OpAtomicIIncrement, .OpAtomicIDecrement, .OpAtomicIAdd, .OpAtomicISub, .OpAtomicSMin, .OpAtomicUMin, .OpAtomicSMax, .OpAtomicUMax, .OpAtomicAnd, .OpAtomicOr, .OpAtomicXor, .OpAtomicFAddEXT, .OpAtomicFMinEXT, .OpAtomicFMaxEXT => try f.atomic(inst),
            else => return f.refuse("{t} is not supported", .{inst.op}),
        }
    }

    // ── Stage 4: barriers and atomics ───────────────────────────────────────

    /// SPIR-V Scope values (section 3.27).
    const scope_cross_device = 0;
    const scope_device = 1;
    const scope_workgroup = 2;
    const scope_subgroup = 3;
    const scope_invocation = 4;
    const scope_queue_family = 5;

    /// MemorySemantics bits (section 3.25).
    const sem_acquire = 0x2;
    const sem_release = 0x4;
    const sem_acq_rel = 0x8;
    const sem_seq_cst = 0x10;
    const sem_ordering = sem_acquire | sem_release | sem_acq_rel | sem_seq_cst;
    const sem_uniform_memory = 0x40;
    const sem_subgroup_memory = 0x80;
    const sem_workgroup_memory = 0x100;
    const sem_cross_workgroup_memory = 0x200;
    const sem_atomic_counter_memory = 0x400;
    const sem_image_memory = 0x800;
    const sem_output_memory = 0x1000;
    /// MakeAvailable, MakeVisible (Vulkan memory model) and Volatile: the
    /// lowering always uses the strongest ordering the bits ask for, which
    /// makes the writes available and visible at that scope.
    const sem_ignored = 0x2000 | 0x4000 | 0x8000;

    /// AIR memory flags (`mem_flags`, src/engine/gpu.zig MemFlags).
    const mem_device = 1;
    const mem_threadgroup = 2;
    const mem_texture = 4;

    /// AIR `thread_scope` values (src/engine/gpu.zig: threadgroup 1, device
    /// 2; `air.simdgroup.barrier(flags, 4)`).
    const air_scope_threadgroup = 1;
    const air_scope_device = 2;
    const air_scope_simdgroup = 4;

    /// The value of a Scope or MemorySemantics operand: an integer constant
    /// (a specialization constant at its default value, as everywhere here).
    fn constOperand(f: *Fn, i: usize, what: []const u8) Error!u32 {
        const id = try f.opId(i);
        const bits = f.l.m.constantBits(id) orelse return f.refuse("the {s} operand %{d} is not an integer constant (an OpSpecConstantOp is not evaluated)", .{ what, id });
        return std.math.cast(u32, bits) orelse f.refuse("the {s} operand %{d} = {d} is not a 32-bit value", .{ what, id, bits });
    }

    /// AIR `mem_flags` for the storage-class bits of a MemorySemantics
    /// value: WorkgroupMemory -> threadgroup, UniformMemory (Vulkan buffers)
    /// and CrossWorkgroupMemory (OpenCL global) -> device, ImageMemory ->
    /// texture. SubgroupMemory names no storage Metal has (SPIR-V reserves
    /// it). AtomicCounterMemory orders atomic-counter memory, which a
    /// compute kernel here cannot have (no AtomicCounter variable lowers),
    /// so the bit orders nothing and is ignored: glslang's
    /// `memoryBarrier()` / `groupMemoryBarrier()` are `OpMemoryBarrier
    /// Device|Workgroup 0xD48`, AcquireRelease plus its
    /// MemorySemanticsAllMemory (Uniform | Workgroup | AtomicCounter |
    /// Image, glslang's spvIR.h). OutputMemory is a graphics interface.
    fn memFlags(f: *Fn, sem: u32) Error!u32 {
        if (sem & sem_output_memory != 0) return f.refuse("memory semantics 0x{x}: OutputMemory is a graphics interface (stage 5)", .{sem});
        const known = sem_ordering | sem_uniform_memory | sem_subgroup_memory | sem_workgroup_memory | sem_cross_workgroup_memory | sem_atomic_counter_memory | sem_image_memory | sem_ignored;
        if (sem & ~@as(u32, known) != 0) return f.refuse("memory semantics 0x{x} has bits this lowering does not know", .{sem});
        var flags: u32 = 0;
        if (sem & sem_workgroup_memory != 0) flags |= mem_threadgroup;
        if (sem & (sem_uniform_memory | sem_cross_workgroup_memory) != 0) flags |= mem_device;
        if (sem & sem_image_memory != 0) flags |= mem_texture;
        return flags;
    }

    /// The AIR scope of a memory scope: Workgroup (and Invocation, which a
    /// wider fence covers) -> threadgroup, Subgroup -> simdgroup, Device /
    /// QueueFamily / CrossDevice -> device (Metal's widest).
    fn airScope(f: *Fn, scope: u32) Error!u32 {
        return switch (scope) {
            scope_workgroup, scope_invocation => air_scope_threadgroup,
            scope_subgroup => air_scope_simdgroup,
            scope_device, scope_queue_family, scope_cross_device => air_scope_device,
            else => f.refuse("unknown memory scope {d}", .{scope}),
        };
    }

    /// OpControlBarrier Execution Memory Semantics. Execution scope
    /// Workgroup is `air.wg.barrier(flags, 1)` (MSL `threadgroup_barrier`),
    /// Subgroup `air.simdgroup.barrier(flags, 4)` (`simdgroup_barrier`);
    /// the flags come from the semantics' storage-class bits (GLSL
    /// `barrier()` = AcquireRelease|WorkgroupMemory -> mem_threadgroup,
    /// Apple's `i32 2, i32 1`). Metal has no barrier across threadgroups,
    /// so a Device or wider execution scope is refused. A threadgroup
    /// barrier orders device memory only for the threads of the
    /// threadgroup, so when the memory scope is Device (HLSL
    /// `DeviceMemoryBarrierWithGroupSync`) and the semantics name device
    /// memory with an ordering, a device-scope `air.atomic.fence` comes
    /// first. The SIMD-group barrier orders memory only among the threads
    /// of the SIMD-group (its second operand is the thread scope, fixed at
    /// simdgroup: Apple's front end compiles MSL `simdgroup_barrier(
    /// mem_threadgroup)` to `air.simdgroup.barrier(i32 2, i32 4)` and
    /// `atomic_thread_fence(mem_threadgroup, seq_cst, thread_scope_
    /// threadgroup)` to `air.atomic.fence(i32 2, i32 5, i32 1)`), while
    /// SPIR-V orders the accesses before the barrier at the Memory scope
    /// operand. So a Subgroup barrier whose memory scope is Workgroup or
    /// wider, with an ordering and a storage-class bit, gets
    /// `air.atomic.fence(flags, 5, scope)` first (the same rule as the
    /// device fence above; Subgroup and Invocation memory scopes need none).
    /// The assembler's uniformity analysis (tools/air/divergence.zig)
    /// refuses a barrier that not every thread of its scope reaches.
    fn controlBarrier(f: *Fn) Error!void {
        const exec = try f.constOperand(0, "execution scope");
        const mem = try f.constOperand(1, "memory scope");
        const sem = try f.constOperand(2, "memory semantics");
        const flags = try f.memFlags(sem);
        const mem_scope = try f.airScope(mem);
        if (@popCount(sem & sem_ordering) > 1) return f.refuse("memory semantics 0x{x} has more than one ordering bit", .{sem});
        const scope_arg: u32, const name: []const u8 = switch (exec) {
            scope_workgroup => .{ air_scope_threadgroup, "air.wg.barrier" },
            scope_subgroup => .{ air_scope_simdgroup, "air.simdgroup.barrier" },
            else => return f.refuse("execution scope {d}: Metal synchronises only a threadgroup (Workgroup) or a SIMD-group (Subgroup)", .{exec}),
        };
        const ordered = sem & sem_ordering != 0;
        if (exec == scope_subgroup) {
            if (ordered and flags != 0 and mem != scope_subgroup and mem != scope_invocation) {
                _ = try f.call("void", "air.atomic.fence", &.{
                    .{ .ty = "i32", .v = try f.l.a.print("{d}", .{flags}) },
                    .{ .ty = "i32", .v = "5" },
                    .{ .ty = "i32", .v = try f.l.a.print("{d}", .{mem_scope}) },
                });
            }
        } else if (ordered and flags & mem_device != 0 and (mem == scope_device or mem == scope_queue_family or mem == scope_cross_device)) {
            _ = try f.call("void", "air.atomic.fence", &.{
                .{ .ty = "i32", .v = "1" },
                .{ .ty = "i32", .v = "5" },
                .{ .ty = "i32", .v = "2" },
            });
        }
        _ = try f.call("void", name, &.{
            .{ .ty = "i32", .v = try f.l.a.print("{d}", .{flags}) },
            .{ .ty = "i32", .v = try f.l.a.print("{d}", .{scope_arg}) },
        });
    }

    /// OpMemoryBarrier Memory Semantics -> `air.atomic.fence(flags, 5,
    /// scope)` (MSL `atomic_thread_fence(flags, memory_order_seq_cst,
    /// scope)`; a native LLVM `fence` crashes Metal's compiler,
    /// docs/air-format.md). Sequential consistency is at least as strong
    /// as any ordering the semantics ask for. A barrier with no ordering
    /// bit or no storage-class bit orders nothing (SPIR-V 3.25: the
    /// storage-class bits say which memory the ordering applies to) and
    /// emits nothing.
    fn memoryBarrier(f: *Fn) Error!void {
        const mem = try f.constOperand(0, "memory scope");
        const sem = try f.constOperand(1, "memory semantics");
        const flags = try f.memFlags(sem);
        const scope = try f.airScope(mem);
        if (@popCount(sem & sem_ordering) > 1) return f.refuse("memory semantics 0x{x} has more than one ordering bit", .{sem});
        if (flags == 0 or sem & sem_ordering == 0) return;
        _ = try f.call("void", "air.atomic.fence", &.{
            .{ .ty = "i32", .v = try f.l.a.print("{d}", .{flags}) },
            .{ .ty = "i32", .v = "5" },
            .{ .ty = "i32", .v = try f.l.a.print("{d}", .{scope}) },
        });
    }

    /// The LLVM ordering of a MemorySemantics value: no ordering bit ->
    /// `monotonic` (Relaxed), Acquire / Release / AcquireRelease /
    /// SequentiallyConsistent -> the same words. A load cannot release and
    /// a store cannot acquire (SPIR-V forbids both; LLVM rejects them), so
    /// AcquireRelease is narrowed to the half that applies and the other
    /// half is refused.
    fn ordering(f: *Fn, sem: u32, kind: enum { rmw, load, store }) Error![]const u8 {
        _ = try f.memFlags(sem);
        const bits = sem & sem_ordering;
        if (@popCount(bits) > 1) return f.refuse("memory semantics 0x{x} has more than one ordering bit", .{sem});
        return switch (bits) {
            0 => "monotonic",
            sem_acquire => if (kind == .store) f.refuse("an atomic store cannot have Acquire semantics", .{}) else "acquire",
            sem_release => if (kind == .load) f.refuse("an atomic load cannot have Release semantics", .{}) else "release",
            sem_acq_rel => switch (kind) {
                .rmw => "acq_rel",
                .load => "acquire",
                .store => "release",
            },
            else => "seq_cst",
        };
    }

    /// OpAtomic* on a device (StorageBuffer, PhysicalStorageBuffer,
    /// CrossWorkgroup, a BufferBlock) or threadgroup (Workgroup) pointer ->
    /// native `atomicrmw` / `cmpxchg` / `load atomic` / `store atomic`,
    /// which Metal's compiler accepts (docs/air-format.md) and which carry
    /// the ordering of the semantics operand (`ordering`). The Scope operand
    /// is not encoded: the assembler records the system scope, which is at
    /// least as wide as any SPIR-V scope. Integer operations take i32 or
    /// i64 (which ones Metal builds for 64 bits is recorded in
    /// tools/spirv/corpus/EXPECTED.txt); OpAtomicFAddEXT is `atomicrmw
    /// fadd` on device memory only (Metal miscompiles it on threadgroup
    /// memory) and FMinEXT / FMaxEXT are refused. A CompareExchange returns the
    /// value it found (`extractvalue` 0 of `cmpxchg`); its Weak form has the
    /// same semantics (SPIR-V 3.37.18) and is emitted strong. The result
    /// of an atomic is per-thread for the uniformity analysis.
    fn atomic(f: *Fn, inst: Instruction) Error!void {
        const l = f.l;
        const p = try f.opId(0);
        const pv = switch (try f.val(p)) {
            .op => |s| s,
            .input => return f.refuse("an atomic on a builtin Input variable", .{}),
        };
        const space = try f.ptrSpace(p) orelse return f.refuse("an atomic on a pointer with no address space", .{});
        if (space != 1 and space != 3) return f.refuse("an atomic on {s} memory: Metal has atomics on device and threadgroup memory only", .{switch (space) {
            0 => "Function or Private",
            2 => "constant",
            else => "this",
        }});
        const t = try f.pointeeOf(p);
        const is_float = switch (inst.op) {
            .OpAtomicFAddEXT, .OpAtomicFMinEXT, .OpAtomicFMaxEXT => true,
            else => false,
        };
        switch (l.m.typeOf(t) orelse return f.refuse("%{d} is not a type", .{t})) {
            .int => |i| {
                if (is_float) return f.refuse("{t} on an integer", .{inst.op});
                if (i.width != 32 and i.width != 64) return f.refuse("{d}-bit atomics: Metal has 32- and 64-bit atomic integers", .{i.width});
                // 64 bits: MSL has atomic_ulong min/max on device memory,
                // returning nothing, and atomic load/store. Every other
                // 64-bit atomicrmw / cmpxchg, a min/max whose result is
                // used and a min/max on threadgroup memory crash
                // MTLCompilerService (XPC_ERROR_CONNECTION_INTERRUPTED; the
                // same IR assembled by Apple's `xcrun metal` crashes it too;
                // probes atom_*_u64_*).
                if (i.width == 64) switch (inst.op) {
                    .OpAtomicLoad, .OpAtomicStore => {},
                    .OpAtomicSMin, .OpAtomicUMin, .OpAtomicSMax, .OpAtomicUMax => {
                        if (space != 1) return f.refuse("64-bit atomic min/max on threadgroup memory: Metal has 64-bit atomic min/max on device memory only (MTLCompilerService crashes)", .{});
                        if (l.isUsed(inst.result.?)) return f.refuse("the result of a 64-bit atomic min/max is used: Metal's 64-bit atomic min/max return nothing (MSL's atomic_max_explicit on atomic_ulong is void) and using it crashes MTLCompilerService", .{});
                    },
                    else => return f.refuse("64-bit {t}: Metal has 64-bit atomic load, store, min and max only (MTLCompilerService crashes on the others)", .{inst.op}),
                };
            },
            .float => |x| switch (inst.op) {
                // LLVM's `atomicrmw fmin` / `fmax` are newer than Metal's
                // bitcode reader: pipeline creation fails with "Failed to
                // materializeAll" (probes atom_FMinEXT_f32_*).
                .OpAtomicFMinEXT, .OpAtomicFMaxEXT => return f.refuse("{t}: Metal's bitcode reader does not know `atomicrmw fmin/fmax` (\"Failed to materializeAll\") and MSL has no atomic float min/max", .{inst.op}),
                .OpAtomicFAddEXT, .OpAtomicLoad, .OpAtomicStore, .OpAtomicExchange => {
                    if (x.width != 32) return f.refuse("{d}-bit float atomics: Metal has 32-bit atomic floats", .{x.width});
                    // Metal builds `atomicrmw fadd ptr addrspace(3)` and
                    // computes the wrong answer on an M3 (macOS 26.3): 64
                    // threads adding 1.0 to a threadgroup float leave 0.0,
                    // and the old values repeat per SIMD-group, whichever
                    // ordering; Apple's `xcrun metal` on the same IR gives
                    // the same. MSL has no threadgroup atomic_float. Float
                    // load, store and exchange on threadgroup memory compute
                    // right (corpus kernels atomic_fadd_tg, atomic_fxchg_tg).
                    if (inst.op == .OpAtomicFAddEXT and space == 3 and !l.opts.allow_metal_miscompiles)
                        return f.refuse("OpAtomicFAddEXT on threadgroup memory: Metal builds `atomicrmw fadd ptr addrspace(3)` but computes a wrong sum (64 threads adding 1.0 leave 0.0; corpus kernel atomic_fadd_tg) and MSL has no threadgroup atomic_float", .{});
                },
                else => return f.refuse("{t} on a float", .{inst.op}),
            },
            else => return f.refuse("an atomic on a pointer to a non-scalar", .{}),
        }
        const tt = try l.ty(t, .value);
        const al = (try l.layout(t)).alignment;
        if (f.knownAlign(p)) |known| if (known < al) return f.refuse("an atomic on an address only {d}-byte aligned (a packed member): Metal's atomics need {d}", .{ known, al });
        const pty = try f.tyOf(p);
        if (inst.result_type) |rt| if (!std.mem.eql(u8, try l.ty(rt, .value), tt)) return f.refuse("the result type is not the pointer's pointee", .{});
        switch (inst.op) {
            .OpAtomicLoad => {
                const ord = try f.ordering(try f.constOperand(2, "memory semantics"), .load);
                try f.emit("{s} = load atomic {s}, {s} {s} {s}, align {d}", .{ try f.def(inst), tt, pty, pv, ord, al });
            },
            .OpAtomicStore => {
                const ord = try f.ordering(try f.constOperand(2, "memory semantics"), .store);
                const v = try f.opId(3);
                try f.need(v, t);
                try f.emit("store atomic {s} {s}, {s} {s} {s}, align {d}", .{ tt, try f.op(v), pty, pv, ord, al });
            },
            .OpAtomicCompareExchange, .OpAtomicCompareExchangeWeak => {
                const eq_sem = try f.constOperand(2, "equal semantics");
                const ne_sem = try f.constOperand(3, "unequal semantics");
                const success = try f.ordering(eq_sem, .rmw);
                // The failure ordering is a load's: no release half.
                const failure = try f.ordering(ne_sem, .load);
                const v = try f.opId(4);
                const cmp = try f.opId(5);
                try f.need(v, t);
                try f.need(cmp, t);
                if (l.m.typeOf(t).? == .float) return f.refuse("{t} on a float", .{inst.op});
                const pair_v = try f.tmpInst("cmpxchg {s} {s}, {s} {s}, {s} {s} {s} {s}, align {d}", .{ pty, pv, tt, try f.op(cmp), tt, try f.op(v), success, failure, al });
                try f.emit("{s} = extractvalue {{ {s}, i1 }} {s}, 0", .{ try f.def(inst), tt, pair_v });
            },
            else => {
                const ord = try f.ordering(try f.constOperand(2, "memory semantics"), .rmw);
                const name: []const u8, const v: []const u8 = switch (inst.op) {
                    .OpAtomicIIncrement => .{ "add", try l.intConst(t, 1) },
                    .OpAtomicIDecrement => .{ "sub", try l.intConst(t, 1) },
                    else => blk: {
                        const vid = try f.opId(3);
                        try f.need(vid, t);
                        break :blk .{ switch (inst.op) {
                            .OpAtomicExchange => "xchg",
                            .OpAtomicIAdd => "add",
                            .OpAtomicISub => "sub",
                            .OpAtomicSMin => "min",
                            .OpAtomicSMax => "max",
                            .OpAtomicUMin => "umin",
                            .OpAtomicUMax => "umax",
                            .OpAtomicAnd => "and",
                            .OpAtomicOr => "or",
                            .OpAtomicXor => "xor",
                            .OpAtomicFAddEXT => "fadd",
                            .OpAtomicFMinEXT => "fmin",
                            .OpAtomicFMaxEXT => "fmax",
                            else => unreachable,
                        }, try f.op(vid) };
                    },
                };
                try f.emit("{s} = atomicrmw {s} {s} {s}, {s} {s} {s}, align {d}", .{ try f.def(inst), name, pty, pv, tt, v, ord, al });
            },
        }
    }

    /// Refuse a decoration on `inst`'s result that changes its meaning and
    /// that this lowering does not implement. SaturatedConversion and
    /// FPRoundingMode are implemented by the conversions themselves
    /// (`convertWidth`, `convertKind`), which refuse the cases they cannot
    /// do; anything else not in `ignorableDecoration` is refused here.
    fn checkDecorations(f: *Fn, inst: Instruction) Error!void {
        const rid = inst.result.?;
        for (f.l.m.decorations) |d| {
            if (d.target != rid or d.member != null) continue;
            const k = d.as() orelse return f.refuse("decoration {d} on the result is not supported", .{d.decoration});
            if (ignorableDecoration(k)) continue;
            switch (k) {
                .saturated_conversion => switch (inst.op) {
                    .OpUConvert, .OpSConvert, .OpConvertFToS, .OpConvertFToU => {},
                    else => return f.refuse("SaturatedConversion on {t} is not supported", .{inst.op}),
                },
                .fp_rounding_mode => switch (inst.op) {
                    .OpFConvert, .OpConvertFToS, .OpConvertFToU, .OpConvertSToF, .OpConvertUToF => {},
                    else => return f.refuse("FPRoundingMode on {t} is not supported", .{inst.op}),
                },
                else => return f.refuse("decoration {t} on the result is not supported", .{k}),
            }
        }
    }

    /// The FPRoundingMode decoration of `inst`'s result.
    fn roundingMode(f: *Fn, inst: Instruction) Error!?spec.FPRoundingMode {
        const d = f.l.m.decoration(inst.result.?, .fp_rounding_mode) orelse return null;
        const raw = d.firstInt() orelse return f.refuse("FPRoundingMode without a mode", .{});
        return std.enums.fromInt(spec.FPRoundingMode, std.math.cast(u32, raw) orelse return f.refuse("FPRoundingMode {d}", .{raw})) orelse f.refuse("FPRoundingMode {d}", .{raw});
    }

    fn saturated(f: *Fn, inst: Instruction) bool {
        return f.l.m.hasDecoration(inst.result.?, .saturated_conversion);
    }

    /// OpUConvert / OpSConvert / OpFConvert. With SaturatedConversion a
    /// narrowing integer conversion clamps to the result's range first
    /// (OpenCL `convert_uchar_sat`); FPRoundingMode on a narrowing
    /// OpFConvert must be RTE, which is what `fptrunc` does on Metal
    /// (probe conv_FConvert_f32_f16 runs against a round-to-nearest-even
    /// reference); RTZ/RTP/RTN are refused.
    fn convertWidth(f: *Fn, inst: Instruction) Error!void {
        const l = f.l;
        const x = try f.opId(0);
        const xt = try f.typeId(x);
        const rt = inst.result_type.?;
        if (!f.sameShape(xt, rt)) return f.refuse("the operand and the result differ in shape", .{});
        const float = inst.op == .OpFConvert;
        const from = if (float) try f.floatWidth(xt) else try f.intWidth(xt);
        const to = if (float) try f.floatWidth(rt) else try f.intWidth(rt);
        const sat = f.saturated(inst);
        if (try f.roundingMode(inst)) |rm| {
            if (!float) return f.refuse("FPRoundingMode on an integer conversion", .{});
            if (to < from and rm != .rte) return f.refuse("FPRoundingMode {t}: a narrowing float conversion rounds to nearest even on Metal; other rounding modes are not implemented", .{rm});
        }
        if (sat and float) return f.refuse("SaturatedConversion on a float conversion", .{});
        _ = try l.ty(rt, .value);
        if (from == to) {
            try f.vals.put(l.a, inst.result.?, try f.val(x));
            return;
        }
        var src = try f.op(x);
        if (sat and to < from) {
            const t = try l.ty(xt, .value);
            const bt = try l.boolTy(xt);
            const signed = inst.op == .OpSConvert;
            const max: u64 = (@as(u64, 1) << @intCast(to - @intFromBool(signed))) - 1;
            const hi = try f.l.intConstBits(xt, max);
            const over = try f.tmpInst("icmp {s} {s} {s}, {s}", .{ if (signed) "sgt" else "ugt", t, src, hi });
            src = try f.tmpInst("select {s} {s}, {s} {s}, {s} {s}", .{ bt, over, t, hi, t, src });
            if (signed) {
                const lo = try f.l.intConstBits(xt, ~max);
                const under = try f.tmpInst("icmp slt {s} {s}, {s}", .{ t, src, lo });
                src = try f.tmpInst("select {s} {s}, {s} {s}, {s} {s}", .{ bt, under, t, lo, t, src });
            }
        }
        // Zig's backend picks the opcode by the result's signedness (module
        // doc, "producer workarounds"). It widens an unsigned source with
        // OpSConvert when the result is signed; only a signed-typed RESULT
        // identifies that case: Zig's OpenCL flavour types every integer
        // unsigned, so there an OpSConvert of a `u16` operand to `u32` may be
        // a real `i16 -> i32` sign extension. The mirror case, a signed
        // source widened with OpUConvert into the unsigned limb of an i128
        // whose upper limbs are then sign-filled from it, is a sign
        // extension; other signed-to-unsigned widenings keep `zext`.
        const zig = to > from and l.zigWorkarounds();
        const zig_unsigned_widen = zig and inst.op == .OpSConvert and !try f.intSigned(xt) and try f.intSigned(rt);
        const zig_signed_widen = zig and inst.op == .OpUConvert and try f.intSigned(xt) and !try f.intSigned(rt) and
            l.zig_sign_fills.contains(inst.result.?);
        const name = switch (inst.op) {
            .OpUConvert => if (zig_signed_widen) "sext" else if (to > from) "zext" else "trunc",
            .OpSConvert => if (zig_unsigned_widen) "zext" else if (to > from) "sext" else "trunc",
            else => if (to > from) "fpext" else "fptrunc",
        };
        try f.emit("{s} = {s} {s} {s} to {s}", .{ try f.def(inst), name, try l.ty(xt, .value), src, try l.ty(rt, .value) });
    }

    /// OpConvertFToS/FToU/SToF/UToF. Float to integer truncates (RTZ);
    /// FPRoundingMode RTE/RTP/RTN round first with `llvm.rint` /
    /// `llvm.ceil` / `llvm.floor` (OpenCL `convert_int_rte`), and
    /// SaturatedConversion clamps: NaN to 0, out-of-range values to the
    /// nearest representable one (OpenCL `convert_int_sat`). Integer to
    /// float: RTE is Metal's rounding; another mode is accepted only when
    /// the conversion is exact (every integer of the width fits the
    /// significand), else refused.
    fn convertKind(f: *Fn, inst: Instruction) Error!void {
        const l = f.l;
        const x = try f.opId(0);
        const xt = try f.typeId(x);
        const rt = inst.result_type.?;
        if (!f.sameShape(xt, rt)) return f.refuse("the operand and the result differ in shape", .{});
        const to_int = inst.op == .OpConvertFToS or inst.op == .OpConvertFToU;
        const rm = try f.roundingMode(inst);
        const sat = f.saturated(inst);
        if (!to_int) {
            const iw = try f.intWidth(xt);
            const fw = try f.floatWidth(rt);
            if (sat) return f.refuse("SaturatedConversion on a conversion to float", .{});
            if (rm) |mode| {
                const significand: u32 = if (fw == 16) 11 else 24;
                if (mode != .rte and iw > significand) return f.refuse("FPRoundingMode {t}: a {d}-bit integer to a {d}-bit float rounds to nearest even on Metal; other rounding modes are not implemented", .{ mode, iw, fw });
            }
            return f.emit("{s} = {s} {s} to {s}", .{ try f.def(inst), if (inst.op == .OpConvertSToF) "sitofp" else "uitofp", try f.typed(x), try l.ty(rt, .value) });
        }
        _ = try f.floatWidth(xt);
        const iw = try f.intWidth(rt);
        const signed = inst.op == .OpConvertFToS;
        var src = try f.op(x);
        if (rm) |mode| {
            const name: ?[]const u8 = switch (mode) {
                .rte => "llvm.rint",
                .rtz => null,
                .rtp => "llvm.ceil",
                .rtn => "llvm.floor",
            };
            if (name) |n| src = try f.unary(n, xt, src);
        }
        const conv = if (signed) "fptosi" else "fptoui";
        const t = try l.ty(xt, .value);
        const ity = try l.ty(rt, .value);
        if (!sat) return f.emit("{s} = {s} {s} {s} to {s}", .{ try f.def(inst), conv, t, src, ity });
        // Bounds as floats: 2^(w-1) / -2^(w-1) (signed), 2^w / 0 (unsigned).
        // A bound beyond the float's range is infinite, so `oge +inf` /
        // `ole -inf` still pick exactly the values to clamp.
        const bt = try l.boolTy(xt);
        const fw = try f.floatWidth(xt);
        const two_w = std.math.pow(f64, 2, @floatFromInt(if (signed) iw - 1 else iw));
        const too_big = fw == 16 and two_w > 65504;
        const hi = try l.floatConst(xt, if (too_big) std.math.inf(f64) else two_w);
        const lo_inf = signed and too_big;
        const lo = try l.floatConst(xt, if (!signed) 0 else if (lo_inf) -std.math.inf(f64) else -two_w);
        const nan = try f.tmpInst("fcmp uno {s} {s}, {s}", .{ t, src, src });
        const big = try f.tmpInst("fcmp oge {s} {s}, {s}", .{ t, src, hi });
        const small = try f.tmpInst("fcmp {s} {s} {s}, {s}", .{ if (lo_inf) "ole" else "olt", t, src, lo });
        const b1 = try f.tmpInst("or {s} {s}, {s}", .{ bt, nan, big });
        const bad = try f.tmpInst("or {s} {s}, {s}", .{ bt, b1, small });
        const safe = try f.tmpInst("select {s} {s}, {s} {s}, {s} {s}", .{ bt, bad, t, try l.zeroOf(xt), t, src });
        const r = try f.tmpInst("{s} {s} {s} to {s}", .{ conv, t, safe, ity });
        const max_bits: u64 = if (signed) (@as(u64, 1) << @intCast(iw - 1)) - 1 else if (iw == 64) std.math.maxInt(u64) else (@as(u64, 1) << @intCast(iw)) - 1;
        const max = try l.intConstBits(rt, max_bits);
        const min = if (signed) try l.intConstBits(rt, ~max_bits) else try l.zeroOf(rt);
        const r1 = try f.tmpInst("select {s} {s}, {s} {s}, {s} {s}", .{ bt, big, ity, max, ity, r });
        try f.bind(inst.result.?, try f.tmpInst("select {s} {s}, {s} {s}, {s} {s}", .{ bt, small, ity, min, ity, r1 }));
    }

    /// OpConvertUToPtr / OpConvertPtrToU, and OpBitcast between a pointer
    /// and a 64-bit integer or a 2 x 32-bit vector: `inttoptr` / `ptrtoint`
    /// (an integer of another width zero-extended or truncated; a vector
    /// bitcast to `i64` first). This is how GL_EXT_buffer_reference
    /// addresses reach a shader (`Data(pc.addr)` from a uint64, the uvec2
    /// form of GL_EXT_buffer_reference_uvec2, DXC's vk::RawBufferLoad) and
    /// OpenCL's integer-to-pointer casts. Only device pointers
    /// (PhysicalStorageBuffer, StorageBuffer, CrossWorkgroup, Generic:
    /// address space 1) convert: the address is an MTLBuffer's gpuAddress
    /// (the host makes the buffer resident with useResource); threadgroup,
    /// constant and private addresses have no such integer form on Metal.
    fn pointerIntCast(f: *Fn, inst: Instruction) Error!void {
        const l = f.l;
        const x = try f.opId(0);
        const xt = try f.typeId(x);
        const rt = inst.result_type.?;
        const to_ptr = (try l.typeOf(rt)) == .pointer;
        const pt = if (to_ptr) rt else xt;
        const it = if (to_ptr) xt else rt;
        const ptr_ty = switch (try l.typeOf(pt)) {
            .pointer => |pp| pp,
            else => return f.refuse("{t} needs a pointer {s}", .{ inst.op, if (to_ptr) "result" else "operand" }),
        };
        const as = try l.addrSpace(ptr_ty.storage_class) orelse return f.refuse("an Input pointer has no address", .{});
        if (as != 1) return f.refuse("{t} of a {s} pointer: only device addresses (an MTLBuffer's gpuAddress) convert to and from integers on Metal", .{ inst.op, switch (as) {
            0 => "Function or Private",
            2 => "constant",
            3 => "Workgroup",
            else => "non-device",
        } });
        const pty = try l.spaceText(as);
        // The integer side: a scalar integer, or (OpBitcast only) a vector
        // of two 32-bit integers.
        const iw = try f.intWidth(it);
        const lanes = l.vecLen(it);
        if (lanes) |n| if (inst.op != .OpBitcast or n != 2 or iw != 32) return f.refuse("the integer side of {t} must be a scalar or (OpBitcast) a 2 x 32-bit vector", .{inst.op});
        if (inst.op == .OpBitcast and lanes == null and iw != 64) return f.refuse("OpBitcast between a pointer and a {d}-bit integer: pointers are 64 bits", .{iw});
        const ity = try l.ty(it, .value);
        if (to_ptr) {
            var v = try f.op(x);
            if (lanes != null) {
                v = try f.tmpInst("bitcast {s} {s} to i64", .{ ity, v });
            } else if (iw != 64) {
                v = try f.tmpInst("zext {s} {s} to i64", .{ ity, v });
            }
            try f.emit("{s} = inttoptr i64 {s} to {s}", .{ try f.def(inst), v, pty });
            try f.untrackedPtr(inst.result.?);
            return;
        }
        const p = try f.tmpInst("ptrtoint {s} {s} to i64", .{ try f.tyOf(x), try f.op(x) });
        if (lanes != null) return f.emit("{s} = bitcast i64 {s} to {s}", .{ try f.def(inst), p, ity });
        if (iw == 64) return f.bind(inst.result.?, p);
        try f.emit("{s} = {s} i64 {s} to {s}", .{ try f.def(inst), if (iw < 64) "trunc" else "zext", p, ity });
    }

    /// Record that pointer `dst` is `src` (same address space and known
    /// alignment) for OpCopyObject and pointer bitcasts. The access chain
    /// that computed `src` (`chains`) carries over only when `dst` points to
    /// the same memory type: a chain built on `dst` continues the root's GEP
    /// with indices walked against `dst`'s pointee, which must then be the
    /// type the root's indices end at. A bitcast to another pointee (OpenCL
    /// pointer casts, Zig's `@ptrCast(&buf.data)[i]`) starts a fresh GEP
    /// from the address instead.
    fn copyPtrInfo(f: *Fn, src: Id, dst: Id) Error!void {
        const l = f.l;
        const t = l.m.resultTypeOf(dst) orelse return;
        const dpt = switch (l.m.typeOf(t) orelse return) {
            .pointer => |p| p,
            else => return,
        };
        if (l.globals.contains(src) or f.spaces.contains(src)) {
            if (try f.ptrSpace(src)) |as| try f.spaces.put(l.a, dst, as);
        }
        if (f.knownAlign(src)) |al| try f.setAlign(dst, al);
        try f.flow.put(l.a, dst, .{ .srcs = try l.a.dupe(Id, &.{src}) });
        const spt = try l.pointerOf(try f.typeId(src));
        if (spt.pointee == dpt.pointee or std.mem.eql(u8, try l.ty(spt.pointee, .memory), try l.ty(dpt.pointee, .memory))) {
            if (f.chains.get(src)) |ch| try f.chains.put(l.a, dst, ch);
            if (f.tight_ptrs.contains(src)) try f.tight_ptrs.put(l.a, dst, {});
        }
    }

    /// Whether pointer value `id` lives in another address space than its
    /// SPIR-V type says: a Uniform-class pointer into a BufferBlock variable
    /// is device memory (`spaces`, `Lowerer.globalSpace`).
    fn relocated(f: *Fn, id: Id) Error!bool {
        const l = f.l;
        const t = l.m.resultTypeOf(id) orelse return false;
        const pt = switch (l.m.typeOf(t) orelse return false) {
            .pointer => |p| p,
            else => return false,
        };
        if (!f.spaces.contains(id) and !l.globals.contains(id)) return false;
        // A storage class with no address space is refused where the value
        // is used; this check only compares the two spaces.
        const real = (f.ptrSpace(id) catch return false) orelse return false;
        const typed_as = (l.addrSpace(pt.storage_class) catch return false) orelse return false;
        return real != typed_as;
    }

    /// Refuse a relocated pointer (`relocated`) wherever its SPIR-V type
    /// would be printed for it: stored into memory (VariablePointers: the
    /// variable's type says `ptr addrspace(2)`, so the store and every later
    /// load would retype a device pointer as a constant one), a Function
    /// variable's initializer, a return value, a composite member, a call
    /// argument. It may be the address of a load, store, access chain or
    /// copy, and a copy, a same-class pointer bitcast or an OpSelect keeps
    /// its space; OpPtrEqual / OpPtrNotEqual / OpPtrDiff need both operands
    /// in one space. (A phi is checked in lowerBody, a call's arguments
    /// against the callee's parameter types as well.)
    fn checkRelocated(f: *Fn, inst: Instruction) Error!void {
        for (inst.operands, 0..) |o, i| {
            const id = o.asId() orelse continue;
            if (!try f.relocated(id)) continue;
            const ok = switch (inst.op) {
                .OpLoad, .OpStore, .OpAccessChain, .OpInBoundsAccessChain, .OpPtrAccessChain, .OpInBoundsPtrAccessChain, .OpCopyObject, .OpBitcast => i == 0,
                // The pointer of an atomic (its other operands are values).
                .OpAtomicLoad, .OpAtomicStore, .OpAtomicExchange, .OpAtomicCompareExchange, .OpAtomicCompareExchangeWeak, .OpAtomicIIncrement, .OpAtomicIDecrement, .OpAtomicIAdd, .OpAtomicISub, .OpAtomicSMin, .OpAtomicUMin, .OpAtomicSMax, .OpAtomicUMax, .OpAtomicAnd, .OpAtomicOr, .OpAtomicXor, .OpAtomicFAddEXT, .OpAtomicFMinEXT, .OpAtomicFMaxEXT => i == 0,
                .OpCopyMemory, .OpCopyMemorySized => i <= 1,
                .OpSelect => i == 1 or i == 2,
                .OpPtrEqual, .OpPtrNotEqual, .OpPtrDiff => blk: {
                    const x = inst.idAt(0) orelse break :blk false;
                    const y = inst.idAt(1) orelse break :blk false;
                    break :blk (try f.ptrSpace(x)) == (try f.ptrSpace(y));
                },
                else => false,
            };
            if (!ok) return f.refuse("%{d} points into a BufferBlock (device memory, although its type says Uniform); {t} would give it its type's constant address space: a pointer into a BufferBlock can only be loaded from, stored to, indexed, copied or selected", .{ id, inst.op });
        }
    }

    /// Refuse a pointer to a tight struct (`tight_ptrs`) wherever it would
    /// be accessed as its type's full AIR size by code that cannot know:
    /// passed to a helper, returned, selected, stored, put in a composite,
    /// indexed as an array of such structs (OpPtrAccessChain). Loads,
    /// stores, copies and access chains into its members handle it.
    fn checkTight(f: *Fn, inst: Instruction) Error!void {
        if (f.tight_ptrs.count() == 0) return;
        for (inst.operands, 0..) |o, i| {
            const id = o.asId() orelse continue;
            if (!f.tight_ptrs.contains(id)) continue;
            const ok = switch (inst.op) {
                .OpLoad, .OpStore, .OpAccessChain, .OpInBoundsAccessChain, .OpCopyObject => i == 0,
                .OpCopyMemory => i <= 1,
                else => false,
            };
            if (!ok) return f.refuse("%{d} points to a struct stored without its tail padding (scalar block layout: the next member or element starts inside it); {t} would access it at its full AIR size: such a pointer can only be loaded from, stored to, copied or indexed into", .{ id, inst.op });
        }
    }

    /// Refuse a store through a pointer into constant memory: Uniform,
    /// PushConstant and UniformConstant blocks are `constant` (address space
    /// 2) on Metal, where a store assembles and builds but does nothing.
    fn checkWritable(f: *Fn, p: Id) Error!void {
        if ((try f.ptrSpace(p)) == 2) return f.refuse("store into constant memory: a Uniform, PushConstant or UniformConstant block is read-only on Metal (an SSBO is a StorageBuffer variable, or a Uniform one of a BufferBlock struct)", .{});
    }

    /// OpSelect. Scalars and vectors are one `select`; a pointer keeps its
    /// operands' address space; a struct, array or matrix (SPIR-V 1.4) is
    /// selected member by member, because an aggregate `select` crashes
    /// MTLCompilerService for some shapes (`{ i32, <4 x i32> }`: probe
    /// op_Select_struct_u32_v4u32) and Apple's own assembler emits the
    /// same crashing construct.
    fn select(f: *Fn, inst: Instruction) Error!void {
        const l = f.l;
        const rt = inst.result_type.?;
        const c = try f.opId(0);
        const x = try f.opId(1);
        const y = try f.opId(2);
        switch (try l.typeOf(rt)) {
            .@"struct", .array, .matrix => {
                if ((try l.typeOf(try f.typeId(c))) != .bool) return f.refuse("a composite OpSelect needs a scalar bool condition", .{});
                var budget: usize = 4096;
                const r = try f.selectMember(try f.op(c), .{ .id = rt }, try f.op(x), try f.op(y), &budget);
                try f.bind(inst.result.?, r);
            },
            .pointer => {
                const xt = try f.tyOf(x);
                if (!std.mem.eql(u8, xt, try f.tyOf(y))) return f.refuse("a select of pointers into different address spaces needs variable pointers", .{});
                try f.emit("{s} = select {s}, {s} {s}, {s} {s}", .{ try f.def(inst), try f.typed(c), xt, try f.op(x), xt, try f.op(y) });
                if (try f.ptrSpace(x)) |as| try f.spaces.put(l.a, inst.result.?, as);
                // Aligned as the less aligned of the two.
                try f.flow.put(l.a, inst.result.?, .{ .srcs = try l.a.dupe(Id, &.{ x, y }) });
                if (f.knownAlign(x) != null or f.knownAlign(y) != null) {
                    const natural = try l.naturalAlign(rt);
                    try f.setAlign(inst.result.?, @min(f.knownAlign(x) orelse natural, f.knownAlign(y) orelse natural));
                }
            },
            else => try f.emit("{s} = select {s}, {s}, {s}", .{ try f.def(inst), try f.typed(c), try f.typed(x), try f.typed(y) }),
        }
    }

    fn selectMember(f: *Fn, c: []const u8, mb: Member, x: []const u8, y: []const u8, budget: *usize) Error![]const u8 {
        const l = f.l;
        if (budget.* == 0) return f.refuse("OpSelect of a composite with more than 4096 members in all", .{});
        budget.* -= 1;
        const agg = try l.memberText(mb);
        var elems: []const Member = &.{};
        var idx: []const u32 = &.{};
        var etys: []const []const u8 = &.{};
        if (mb.tight and l.vecLen(mb.id) != null) {
            const e: Member = .{ .id = l.scalarOf(mb.id) };
            const et = try l.ty(e.id, .memory);
            const ms = try l.a.alloc(Member, 3);
            const ts = try l.a.alloc([]const u8, 3);
            @memset(ms, e);
            @memset(ts, et);
            elems = ms;
            idx = &.{ 0, 1, 2 };
            etys = ts;
        } else switch (try l.typeOf(mb.id)) {
            .@"struct" => |st| {
                const sl = try l.memberStructLayout(mb);
                const ms = try l.a.alloc(Member, st.members.len);
                for (ms, st.members, sl.tight) |*m, id, a3| m.* = .{ .id = id, .tight = a3 };
                elems = ms;
                idx = sl.index;
                etys = sl.member_tys;
            },
            .array, .matrix => {
                const e: Member, const n: u64 = switch (try l.typeOf(mb.id)) {
                    .array => |arr| .{ try l.elemOf(mb.id), try l.arrayLen(arr.len) },
                    .matrix => |mt| .{ .{ .id = mt.column }, mt.cols },
                    else => unreachable,
                };
                if (n > budget.*) return f.refuse("OpSelect of a composite with more than 4096 members in all", .{});
                const ms = try l.a.alloc(Member, @intCast(n));
                const is = try l.a.alloc(u32, @intCast(n));
                const ts = try l.a.alloc([]const u8, @intCast(n));
                const et = try l.memberText(e);
                for (ms, is, ts, 0..) |*m, *i, *t, k| {
                    m.* = e;
                    i.* = @intCast(k);
                    t.* = et;
                }
                elems = ms;
                idx = is;
                etys = ts;
            },
            else => return f.tmpInst("select i1 {s}, {s} {s}, {s} {s}", .{ c, agg, x, agg, y }),
        }
        var cur: []const u8 = "undef";
        for (elems, idx, etys) |e, k, et| {
            const xe = try f.tmpInst("extractvalue {s} {s}, {d}", .{ agg, x, k });
            const ye = try f.tmpInst("extractvalue {s} {s}, {d}", .{ agg, y, k });
            const s = try f.selectMember(c, e, xe, ye, budget);
            cur = try f.tmpInst("insertvalue {s} {s}, {s} {s}, {d}", .{ agg, cur, et, s, k });
        }
        return cur;
    }

    /// A member read out of an aggregate (`text`, its stored form) as a
    /// value of its SPIR-V type: a `[3 x T]` becomes `<3 x T>`, a tight
    /// struct its own layout (member by member), a bool compares with 0.
    fn memberToValue(f: *Fn, text: []const u8, mb: Member) Error![]const u8 {
        if (!mb.tight) return f.fromMem(text, mb.id);
        const l = f.l;
        if (l.vecLen(mb.id) == null) return f.convertStruct(text, try l.memberStructLayout(mb), try l.structLayout(mb.id), mb.id);
        const at = try l.memberText(mb);
        const et = try l.ty(l.scalarOf(mb.id), .memory);
        const vt = try l.ty(mb.id, .memory);
        var cur: []const u8 = "undef";
        for (0..3) |k| {
            const e = try f.tmpInst("extractvalue {s} {s}, {d}", .{ at, text, k });
            cur = try f.tmpInst("insertelement {s} {s}, {s} {s}, i32 {d}", .{ vt, cur, et, e, k });
        }
        return f.fromMem(cur, mb.id);
    }

    /// The inverse of `memberToValue`.
    fn valueToMember(f: *Fn, text: []const u8, mb: Member) Error![]const u8 {
        const m = try f.toMem(text, mb.id);
        if (!mb.tight) return m;
        const l = f.l;
        if (l.vecLen(mb.id) == null) return f.convertStruct(m, try l.structLayout(mb.id), try l.memberStructLayout(mb), mb.id);
        const at = try l.memberText(mb);
        const et = try l.ty(l.scalarOf(mb.id), .memory);
        const vt = try l.ty(mb.id, .memory);
        var cur: []const u8 = "undef";
        for (0..3) |k| {
            const e = try f.tmpInst("extractelement {s} {s}, i32 {d}", .{ vt, m, k });
            cur = try f.tmpInst("insertvalue {s} {s}, {s} {s}, {d}", .{ at, cur, et, e, k });
        }
        return cur;
    }

    /// Struct `t`'s value `text` in layout `from` rebuilt in layout `to`
    /// (its own layout and its tight form, either way): member by member,
    /// a member whose storage form differs converted through its value.
    fn convertStruct(f: *Fn, text: []const u8, from: StructLayout, to: StructLayout, t: Id) Error![]const u8 {
        const l = f.l;
        const members = (try l.typeOf(t)).@"struct".members;
        var cur: []const u8 = "undef";
        for (members, 0..) |mt, i| {
            var e = try f.tmpInst("extractvalue {s} {s}, {d}", .{ from.text, text, from.index[i] });
            if (from.tight[i] != to.tight[i]) {
                const v = try f.memberToValue(e, .{ .id = mt, .tight = from.tight[i] });
                e = try f.valueToMember(v, .{ .id = mt, .tight = to.tight[i] });
            }
            cur = try f.tmpInst("insertvalue {s} {s}, {s} {s}, {d}", .{ to.text, cur, to.member_tys[i], e, to.index[i] });
        }
        return cur;
    }

    /// One level of an aggregate path: the element at literal index `k` of
    /// a struct/array/matrix member `mb`, or null for a vector (a lane).
    const Step = struct { next: Member, index: u32 };

    fn aggStep(f: *Fn, mb: Member, k: u64) Error!?Step {
        const l = f.l;
        if (mb.tight and l.vecLen(mb.id) != null) {
            if (k >= 3) return f.refuse("index {d} past a 3-component vector", .{k});
            return .{ .next = .{ .id = l.scalarOf(mb.id) }, .index = @intCast(k) };
        }
        switch (try l.typeOf(mb.id)) {
            .@"struct" => |st| {
                if (k >= st.members.len) return f.refuse("member index {d} of a struct with {d} members", .{ k, st.members.len });
                const sl = try l.memberStructLayout(mb);
                return .{ .next = .{ .id = st.members[k], .tight = sl.tight[k] }, .index = sl.index[k] };
            },
            .array => |arr| {
                if (k >= try l.arrayLen(arr.len)) return f.refuse("index {d} past an array of {d}", .{ k, try l.arrayLen(arr.len) });
                return .{ .next = try l.elemOf(mb.id), .index = @intCast(k) };
            },
            .matrix => |mt| {
                if (k >= mt.cols) return f.refuse("column {d} of a matrix with {d}", .{ k, mt.cols });
                return .{ .next = .{ .id = mt.column }, .index = @intCast(k) };
            },
            .vector => |v| {
                if (k >= v.len) return f.refuse("component {d} of a {d}-component vector", .{ k, v.len });
                return null;
            },
            else => return f.refuse("index into a non-composite", .{}),
        }
    }

    /// `{ T, T }` struct of `rt` holding `x`, `y` (member indices through the
    /// layout, which may be packed).
    fn pair(f: *Fn, rt: Id, t: []const u8, x: []const u8, y: []const u8) Error![]const u8 {
        const sl = try f.l.structLayout(rt);
        if (sl.index.len != 2) return f.refuse("the result type must be a struct of two members", .{});
        const st = (try f.l.typeOf(rt)).@"struct";
        if (!std.mem.eql(u8, try f.l.ty(st.members[0], .value), t) or !std.mem.eql(u8, try f.l.ty(st.members[1], .value), t)) return f.refuse("the result struct's members must have the operands' type", .{});
        const xm = try f.valueToMember(x, .{ .id = st.members[0], .tight = sl.tight[0] });
        const ym = try f.valueToMember(y, .{ .id = st.members[1], .tight = sl.tight[1] });
        const s0 = try f.tmpInst("insertvalue {s} undef, {s} {s}, {d}", .{ sl.text, sl.member_tys[0], xm, sl.index[0] });
        return f.tmpInst("insertvalue {s} {s}, {s} {s}, {d}", .{ sl.text, s0, sl.member_tys[1], ym, sl.index[1] });
    }

    /// Low and high halves of the double-width product. Up to 32 bits the
    /// operands are widened; a 64-bit product is assembled from 32-bit
    /// halves, because `mul i128` crashes MTLCompilerService (probe
    /// op_UMulExtended_u64).
    fn mulWide(f: *Fn, xt: Id, x: []const u8, y: []const u8, signed: bool) Error![2][]const u8 {
        const l = f.l;
        const t = try l.ty(xt, .value);
        const w = try f.intWidth(xt);
        if (w == 64) return f.mulWide64(xt, x, y, signed);
        const wt = try l.intTyBits(xt, w * 2);
        const ext = if (signed) "sext" else "zext";
        const x2 = try f.tmpInst("{s} {s} {s} to {s}", .{ ext, t, x, wt });
        const y2 = try f.tmpInst("{s} {s} {s} to {s}", .{ ext, t, y, wt });
        const p = try f.tmpInst("mul {s} {s}, {s}", .{ wt, x2, y2 });
        const lo = try f.tmpInst("trunc {s} {s} to {s}", .{ wt, p, t });
        const shift = if (l.vecLen(xt)) |n| blk: {
            var s: std.ArrayList(u8) = .empty;
            try s.append(l.a, '<');
            for (0..n) |i| try s.print(l.a, "{s}i{d} {d}", .{ if (i == 0) "" else ", ", w * 2, w });
            try s.append(l.a, '>');
            break :blk s.items;
        } else try l.a.print("{d}", .{w});
        const h = try f.tmpInst("lshr {s} {s}, {s}", .{ wt, p, shift });
        const hi = try f.tmpInst("trunc {s} {s} to {s}", .{ wt, h, t });
        return .{ lo, hi };
    }

    fn mulWide64(f: *Fn, xt: Id, x: []const u8, y: []const u8, signed: bool) Error![2][]const u8 {
        const l = f.l;
        const t = try l.ty(xt, .value);
        const m32 = try l.intConst(xt, 0xFFFFFFFF);
        const s32 = try l.intConst(xt, 32);
        const lo = try f.tmpInst("mul {s} {s}, {s}", .{ t, x, y });
        const xl = try f.tmpInst("and {s} {s}, {s}", .{ t, x, m32 });
        const xh = try f.tmpInst("lshr {s} {s}, {s}", .{ t, x, s32 });
        const yl = try f.tmpInst("and {s} {s}, {s}", .{ t, y, m32 });
        const yh = try f.tmpInst("lshr {s} {s}, {s}", .{ t, y, s32 });
        const ll = try f.tmpInst("mul {s} {s}, {s}", .{ t, xl, yl });
        const lh = try f.tmpInst("mul {s} {s}, {s}", .{ t, xl, yh });
        const hl = try f.tmpInst("mul {s} {s}, {s}", .{ t, xh, yl });
        const hh = try f.tmpInst("mul {s} {s}, {s}", .{ t, xh, yh });
        const llh = try f.tmpInst("lshr {s} {s}, {s}", .{ t, ll, s32 });
        const lhl = try f.tmpInst("and {s} {s}, {s}", .{ t, lh, m32 });
        const hll = try f.tmpInst("and {s} {s}, {s}", .{ t, hl, m32 });
        const mid1 = try f.tmpInst("add {s} {s}, {s}", .{ t, llh, lhl });
        const mid2 = try f.tmpInst("add {s} {s}, {s}", .{ t, mid1, hll });
        const carry = try f.tmpInst("lshr {s} {s}, {s}", .{ t, mid2, s32 });
        const lhh = try f.tmpInst("lshr {s} {s}, {s}", .{ t, lh, s32 });
        const hlh = try f.tmpInst("lshr {s} {s}, {s}", .{ t, hl, s32 });
        const h1 = try f.tmpInst("add {s} {s}, {s}", .{ t, hh, lhh });
        const h2 = try f.tmpInst("add {s} {s}, {s}", .{ t, h1, hlh });
        var hi = try f.tmpInst("add {s} {s}, {s}", .{ t, h2, carry });
        if (signed) {
            // The signed high half: subtract y where x < 0 and x where y < 0.
            const z = try l.zeroOf(xt);
            const bt = try l.boolTy(xt);
            const xn = try f.tmpInst("icmp slt {s} {s}, {s}", .{ t, x, z });
            const yn = try f.tmpInst("icmp slt {s} {s}, {s}", .{ t, y, z });
            const cx = try f.tmpInst("select {s} {s}, {s} {s}, {s} {s}", .{ bt, xn, t, y, t, z });
            const cy = try f.tmpInst("select {s} {s}, {s} {s}, {s} {s}", .{ bt, yn, t, x, t, z });
            const s1 = try f.tmpInst("sub {s} {s}, {s}", .{ t, hi, cx });
            hi = try f.tmpInst("sub {s} {s}, {s}", .{ t, s1, cy });
        }
        return .{ lo, hi };
    }

    /// `text` of integer type `from` converted to the width of `to`.
    fn resize(f: *Fn, text: []const u8, from: Id, to: Id) Error![]const u8 {
        const fw = try f.intWidth(from);
        const tw = try f.intWidth(to);
        if (fw == tw) return text;
        return f.tmpInst("{s} {s} {s} to {s}", .{ if (tw > fw) "zext" else "trunc", try f.l.ty(from, .value), text, try f.l.ty(to, .value) });
    }

    fn bitField(f: *Fn, inst: Instruction) Error!void {
        const l = f.l;
        const rt = inst.result_type.?;
        const t = try l.ty(rt, .value);
        const bt = try l.boolTy(rt);
        const w: i64 = try f.intWidth(rt);
        const base = try f.op(try f.opId(0));
        const wc = try l.intConst(rt, w);
        const ones = try l.intConst(rt, -1);
        const z = try l.zeroOf(rt);
        switch (inst.op) {
            .OpBitFieldUExtract => {
                // Count 0 selects the final value: with Offset = width
                // (defined by SPIR-V: Offset + Count <= width) the `lshr`
                // is poison, `and poison, 0` stays poison, and LLVM folds a
                // constant one away (Metal then drops the store; exec test
                // bitfield_full_offset). For Count > 0 the shifts are in
                // range.
                const off = try f.intAs(try f.opId(1), rt, false);
                const cnt = try f.intAs(try f.opId(2), rt, false);
                const sh = try f.tmpInst("lshr {s} {s}, {s}", .{ t, base, off });
                const wmc = try f.tmpInst("sub {s} {s}, {s}", .{ t, wc, cnt });
                const m0 = try f.tmpInst("lshr {s} {s}, {s}", .{ t, ones, wmc });
                const field = try f.tmpInst("and {s} {s}, {s}", .{ t, sh, m0 });
                const isz = try f.tmpInst("icmp eq {s} {s}, {s}", .{ t, cnt, z });
                try f.bind(inst.result.?, try f.tmpInst("select {s} {s}, {s} {s}, {s} {s}", .{ bt, isz, t, z, t, field }));
            },
            .OpBitFieldSExtract => {
                const off = try f.intAs(try f.opId(1), rt, false);
                const cnt = try f.intAs(try f.opId(2), rt, false);
                const t1 = try f.tmpInst("sub {s} {s}, {s}", .{ t, wc, off });
                const t2 = try f.tmpInst("sub {s} {s}, {s}", .{ t, t1, cnt });
                const s1 = try f.tmpInst("shl {s} {s}, {s}", .{ t, base, t2 });
                const t3 = try f.tmpInst("sub {s} {s}, {s}", .{ t, wc, cnt });
                const s2 = try f.tmpInst("ashr {s} {s}, {s}", .{ t, s1, t3 });
                const isz = try f.tmpInst("icmp eq {s} {s}, {s}", .{ t, cnt, z });
                try f.bind(inst.result.?, try f.tmpInst("select {s} {s}, {s} {s}, {s} {s}", .{ bt, isz, t, z, t, s2 }));
            },
            else => {
                const ins = try f.op(try f.opId(1));
                const off = try f.intAs(try f.opId(2), rt, false);
                const cnt = try f.intAs(try f.opId(3), rt, false);
                const wmc = try f.tmpInst("sub {s} {s}, {s}", .{ t, wc, cnt });
                const m0 = try f.tmpInst("lshr {s} {s}, {s}", .{ t, ones, wmc });
                const mask = try f.tmpInst("shl {s} {s}, {s}", .{ t, m0, off });
                const nm = try f.tmpInst("xor {s} {s}, {s}", .{ t, mask, ones });
                const keep = try f.tmpInst("and {s} {s}, {s}", .{ t, base, nm });
                const sh = try f.tmpInst("shl {s} {s}, {s}", .{ t, ins, off });
                const part = try f.tmpInst("and {s} {s}, {s}", .{ t, sh, mask });
                const comb = try f.tmpInst("or {s} {s}, {s}", .{ t, keep, part });
                const isz = try f.tmpInst("icmp eq {s} {s}, {s}", .{ t, cnt, z });
                try f.bind(inst.result.?, try f.tmpInst("select {s} {s}, {s} {s}, {s} {s}", .{ bt, isz, t, base, t, comb }));
            },
        }
    }

    /// Set `vol` for the access `inst` makes through `ptrs` to a value of
    /// type `t`: Volatile when its memory operands (starting at operand
    /// `i`; OpCopyMemory's second set is the source's) say so, a pointer
    /// points into Volatile memory (`Lowerer.findVolatile`), or `t` holds a
    /// Volatile member.
    fn setVolatile(f: *Fn, inst: Instruction, i: usize, ptrs: []const Id, t: Id) Error!void {
        const l = f.l;
        f.vol = false;
        if (inst.intAt(i)) |mask| {
            if (mask & 1 != 0) f.vol = true;
            if (inst.op == .OpCopyMemory) {
                // Aligned's literal, MakePointerAvailable's and
                // MakePointerVisible's scopes follow the first mask.
                const params: usize = @popCount(mask & (0x2 | 0x8 | 0x10));
                if (inst.intAt(i + 1 + params)) |m2| if (m2 & 1 != 0) {
                    f.vol = true;
                };
            }
        }
        for (ptrs) |p| if (l.volatile_ptrs.contains(p)) {
            f.vol = true;
        };
        if (try l.volatileType(t)) f.vol = true;
    }

    /// `volatile ` while lowering a Volatile access (`vol`).
    fn vk(f: *Fn) []const u8 {
        return if (f.vol) "volatile " else "";
    }

    fn load(f: *Fn, inst: Instruction) Error!void {
        const l = f.l;
        const p = try f.opId(0);
        const rt = inst.result_type.?;
        const vty = try l.ty(rt, .value);
        switch (try f.val(p)) {
            .input => |in| {
                var v = in.value;
                var vt = in.param_ty;
                if (in.lane) |lane| {
                    v = try f.tmpInst("extractelement {s} {s}, {s}", .{ in.param_ty, in.value, lane });
                    vt = "i32";
                }
                if (std.mem.eql(u8, vt, vty)) return f.bind(inst.result.?, v);
                if (l.m.scalarWidth(rt) != 64) return f.refuse("load of a {s} builtin as {s}", .{ vt, vty });
                return f.bind(inst.result.?, try f.tmpInst("zext {s} {s} to {s}", .{ vt, v, vty }));
            },
            .op => |pv| {
                const al = try f.accessAlign(1, p, (try l.layout(rt)).alignment);
                try f.setVolatile(inst, 1, &.{p}, rt);
                defer f.vol = false;
                if (f.tight_ptrs.contains(p)) return f.bind(inst.result.?, try f.loadMember(p, rt, al));
                const mt = try l.ty(rt, .memory);
                if (try f.splitAggregate(p, rt)) |ch| return f.bind(inst.result.?, try f.loadSplit(ch, .{ .id = rt }, al));
                if (l.isBoolish(rt)) {
                    const t = try f.tmpInst("load {s}{s}, {s} {s}, align {d}", .{ f.vk(), mt, try f.tyOf(p), pv, al });
                    return f.bind(inst.result.?, try f.fromMem(t, rt));
                }
                try f.emit("{s} = load {s}{s}, {s} {s}, align {d}", .{ try f.def(inst), f.vk(), mt, try f.tyOf(p), pv, al });
                try f.untrackedPtr(inst.result.?);
            },
        }
    }

    /// A load through a pointer to a struct member stored in its tight form
    /// (`Member.tight`, `tight_ptrs`): the tight form is read, so no byte
    /// past the member is touched, and rebuilt as a value of `t`.
    fn loadMember(f: *Fn, p: Id, t: Id, al: u64) Error![]const u8 {
        const l = f.l;
        const mb: Member = .{ .id = t, .tight = true };
        const v = if (try f.splitAggregate(p, t)) |ch|
            try f.loadSplit(ch, mb, al)
        else
            try f.tmpInst("load {s}{s}, {s} {s}, align {d}", .{ f.vk(), try l.memberText(mb), try f.tyOf(p), try f.op(p), al });
        return f.memberToValue(v, mb);
    }

    /// The store counterpart of `loadMember`: only the member's bytes are
    /// written (the next member may start inside its AIR padding).
    fn storeMember(f: *Fn, p: Id, t: Id, value: []const u8, al: u64) Error!void {
        const l = f.l;
        const mb: Member = .{ .id = t, .tight = true };
        const v = try f.valueToMember(value, mb);
        if (try f.splitAggregate(p, t)) |ch| return f.storeSplit(ch, mb, v, al);
        try f.emit("store {s}{s} {s}, {s} {s}, align {d}", .{ f.vk(), try l.memberText(mb), v, try f.tyOf(p), try f.op(p), al });
    }

    fn accessChain(f: *Fn, inst: Instruction, ptr_form: bool) Error!void {
        const l = f.l;
        const a = l.a;
        const base = try f.opId(0);
        const rest_ops = inst.operands[1..];
        switch (try f.val(base)) {
            .input => |in| {
                if (ptr_form) return f.refuse("OpPtrAccessChain on a builtin Input variable", .{});
                if (rest_ops.len == 0) return f.vals.put(a, inst.result.?, .{ .input = in });
                if (rest_ops.len != 1 or in.lane != null) return f.refuse("a builtin vector has one level of components", .{});
                if (std.mem.findScalar(u8, in.param_ty, '<') == null) return f.refuse("index into a scalar builtin", .{});
                var lane = in;
                const lid = rest_ops[0].asId() orelse return f.refuse("the index is not an id", .{});
                try f.needClass(try f.typeId(lid), .int);
                if (l.vecLen(try f.typeId(lid)) != null) return f.refuse("the index is not a scalar", .{});
                lane.lane = try f.typed(lid);
                return f.vals.put(a, inst.result.?, .{ .input = lane });
            },
            .op => {},
        }
        const bt = try f.typeId(base);
        const pt = try l.pointerOf(bt);
        const as = try f.ptrSpace(base) orelse return f.refuse("access chain on an Input pointer", .{});
        const pty = try l.spaceText(as);
        var cur: Member = .{ .id = pt.pointee };
        var ch: Chain = .{ .root = try f.op(base), .src = try l.ty(cur.id, .memory), .pty = pty, .idx = "", .inb = inst.op == .OpInBoundsAccessChain or inst.op == .OpInBoundsPtrAccessChain, .dynamic = false };
        var idx: std.ArrayList(u8) = .empty;
        var ids = rest_ops;
        // What the result's address is known to be aligned to: the base's,
        // narrowed by every constant offset and stride on the way (`al`
        // collects the latter; `flow` keeps it for `settleAlign`).
        const base_natural = (try l.layout(pt.pointee)).alignment;
        var al: u64 = std.math.maxInt(u64);
        if (ptr_form) {
            if (ids.len == 0) return f.refuse("OpPtrAccessChain without an element index", .{});
            const size = (try l.layout(cur.id)).size;
            if (l.m.decoration(bt, .array_stride)) |d| {
                if ((d.firstInt() orelse 0) != size) return f.refuse("ArrayStride {d} on the pointer type differs from the pointee's AIR size {d}", .{ d.firstInt() orelse 0, size });
            }
            const eid = ids[0].asId() orelse return f.refuse("the element index is not an id", .{});
            _ = try f.intWidth(try f.typeId(eid));
            try idx.appendSlice(a, try f.typed(eid));
            ch.dynamic = l.m.constantBits(eid) == null;
            al = @min(al, f.stepAlign(eid, size));
            ids = ids[1..];
        } else if (f.chains.get(base)) |bc| {
            // Continue the base's GEP from its root, in the form the root's
            // type holds the base's pointee in: a tight struct's member
            // indices count its padding members (it used to walk the
            // natural layout's indices into the tight form, landing a
            // member store on the padding or the wrong member).
            ch.root = bc.root;
            ch.src = bc.src;
            ch.inb = ch.inb and bc.inb;
            ch.dynamic = bc.dynamic;
            cur.tight = bc.tight;
            try idx.appendSlice(a, bc.idx);
        } else {
            if (f.tight_ptrs.contains(base)) {
                cur.tight = true;
                ch.src = try l.memberText(cur);
            }
            try idx.appendSlice(a, "i64 0");
        }
        if (ids.len == 0 and !ptr_form) {
            try f.needPointerTo(inst.result_type.?, pt.storage_class, cur.id);
            try f.vals.put(a, inst.result.?, try f.val(base));
            return f.copyPtrInfo(base, inst.result.?);
        }
        for (ids) |o| {
            const iid = o.asId() orelse return f.refuse("an index is not an id", .{});
            _ = try f.intWidth(try f.typeId(iid));
            if (f.l.vecLen(try f.typeId(iid)) != null) return f.refuse("an access chain index must be a scalar", .{});
            const vec = if (cur.tight) l.vecLen(cur.id) else switch (try l.typeOf(cur.id)) {
                .vector => |v| v.len,
                else => null,
            };
            if (vec != null) {
                // GEP into a vector: close the aggregate GEP, then step by
                // whole elements (the lanes are packed; `[3 x T]` alike).
                const e = l.scalarOf(cur.id);
                ch.root = try f.tmpInst("getelementptr {s}{s}, {s} {s}, {s}", .{ if (ch.inb) "inbounds " else "", ch.src, pty, ch.root, idx.items });
                ch.src = try l.ty(e, .memory);
                idx = .empty;
                try idx.appendSlice(a, try f.typed(iid));
                ch.dynamic = ch.dynamic or l.m.constantBits(iid) == null;
                al = @min(al, f.stepAlign(iid, (try l.layout(e)).size));
                cur = .{ .id = e };
                continue;
            }
            switch (try l.typeOf(cur.id)) {
                .@"struct" => |st| {
                    const k = l.m.constantBits(iid) orelse return f.refuse("struct member index %{d} is not a constant", .{iid});
                    if (k >= st.members.len) return f.refuse("struct member index {d} out of range", .{k});
                    const sl = try l.memberStructLayout(cur);
                    try idx.print(a, ", i32 {d}", .{sl.index[k]});
                    al = @min(al, alignOfOffset(sl.offsets[k]));
                    cur = .{ .id = st.members[k], .tight = sl.tight[k] };
                },
                .array, .runtime_array => {
                    const e = try l.elemOf(cur.id);
                    try idx.print(a, ", {s}", .{try f.typed(iid)});
                    ch.dynamic = ch.dynamic or l.m.constantBits(iid) == null;
                    al = @min(al, f.stepAlign(iid, (try l.memberLayout(e)).size));
                    cur = e;
                },
                .matrix => |mt| {
                    try idx.print(a, ", {s}", .{try f.typed(iid)});
                    ch.dynamic = ch.dynamic or l.m.constantBits(iid) == null;
                    al = @min(al, f.stepAlign(iid, (try l.layout(mt.column)).size));
                    cur = .{ .id = mt.column };
                },
                else => return f.refuse("access chain into a non-composite type", .{}),
            }
        }
        try f.needPointerTo(inst.result_type.?, pt.storage_class, cur.id);
        ch.idx = idx.items;
        ch.tight = cur.tight;
        try f.emit("{s} = getelementptr {s}{s}, {s} {s}, {s}", .{ try f.def(inst), if (ch.inb) "inbounds " else "", ch.src, pty, ch.root, ch.idx });
        try f.spaces.put(a, inst.result.?, as);
        try f.setAlign(inst.result.?, @min(al, f.knownAlign(base) orelse base_natural));
        try f.flow.put(a, inst.result.?, .{ .srcs = try a.dupe(Id, &.{base}), .cap = al });
        try f.chains.put(a, inst.result.?, ch);
        if (cur.tight and l.vecLen(cur.id) == null) try f.tight_ptrs.put(a, inst.result.?, {});
    }

    /// Whether an aggregate load or store through `p` must go member by
    /// member: `p` points at a struct/array element of private (Function /
    /// Private) memory picked by a runtime index. MTLCompilerService crashes
    /// (XPC_ERROR_CONNECTION_INTERRUPTED) on an aggregate load from such a
    /// pointer, and on a GEP that starts from it, when the struct has
    /// interior padding (`[2 x { i32, <4 x i32> }]`; exec test
    /// composite_select); one GEP from the root per member, which is what
    /// Apple's front end emits for the MSL twin, builds and runs.
    fn splitAggregate(f: *Fn, p: Id, t: Id) Error!?Chain {
        const ch = f.chains.get(p) orelse return null;
        if (!ch.dynamic) return null;
        if ((try f.ptrSpace(p)) != 0) return null;
        return switch (try f.l.typeOf(t)) {
            .@"struct", .array, .matrix => ch,
            else => null,
        };
    }

    const Leaf = struct { id: Id, ty: []const u8, path: []const u8, gep: []const u8, offset: u64 };

    /// The scalar and vector members of `mb`, with their extractvalue path,
    /// GEP indices and byte offset.
    fn leaves(f: *Fn, mb: Member, path: []const u8, gep: []const u8, off: u64, out: *std.ArrayList(Leaf)) Error!void {
        const l = f.l;
        if (out.items.len >= 256) return f.refuse("an aggregate of more than 256 members accessed through a runtime index into private memory", .{});
        if (mb.tight and l.vecLen(mb.id) != null) {
            const e = l.scalarOf(mb.id);
            const es = (try l.layout(e)).size;
            for (0..3) |k| try out.append(l.a, .{ .id = e, .ty = try l.ty(e, .memory), .path = try l.a.print("{s}, {d}", .{ path, k }), .gep = try l.a.print("{s}, i32 {d}", .{ gep, k }), .offset = off + k * es });
            return;
        }
        switch (try l.typeOf(mb.id)) {
            .@"struct" => |st| {
                const sl = try l.memberStructLayout(mb);
                for (st.members, 0..) |m, i| {
                    try f.leaves(.{ .id = m, .tight = sl.tight[i] }, try l.a.print("{s}, {d}", .{ path, sl.index[i] }), try l.a.print("{s}, i32 {d}", .{ gep, sl.index[i] }), off + sl.offsets[i], out);
                }
            },
            .array, .matrix => {
                const e: Member, const n: u64 = switch (try l.typeOf(mb.id)) {
                    .array => |arr| .{ try l.elemOf(mb.id), try l.arrayLen(arr.len) },
                    .matrix => |mt| .{ .{ .id = mt.column }, mt.cols },
                    else => unreachable,
                };
                if (n > 256) return f.refuse("an aggregate of more than 256 members accessed through a runtime index into private memory", .{});
                const stride = (try l.memberLayout(e)).size;
                for (0..@intCast(n)) |k| try f.leaves(e, try l.a.print("{s}, {d}", .{ path, k }), try l.a.print("{s}, i32 {d}", .{ gep, k }), off + k * stride, out);
            },
            else => try out.append(l.a, .{ .id = mb.id, .ty = try l.ty(mb.id, .memory), .path = path, .gep = gep, .offset = off }),
        }
    }

    fn loadSplit(f: *Fn, ch: Chain, mb: Member, al: u64) Error![]const u8 {
        const l = f.l;
        var ls: std.ArrayList(Leaf) = .empty;
        try f.leaves(mb, "", "", 0, &ls);
        const agg = try l.memberText(mb);
        var cur: []const u8 = "undef";
        for (ls.items) |lf| {
            const p = try f.tmpInst("getelementptr {s}{s}, {s} {s}, {s}{s}", .{ if (ch.inb) "inbounds " else "", ch.src, ch.pty, ch.root, ch.idx, lf.gep });
            const la = @min(al, alignOfOffset(lf.offset), (try l.layout(lf.id)).alignment);
            const v = try f.tmpInst("load {s}{s}, {s} {s}, align {d}", .{ f.vk(), lf.ty, ch.pty, p, la });
            cur = try f.tmpInst("insertvalue {s} {s}, {s} {s}{s}", .{ agg, cur, lf.ty, v, lf.path });
        }
        return cur;
    }

    fn storeSplit(f: *Fn, ch: Chain, mb: Member, value: []const u8, al: u64) Error!void {
        const l = f.l;
        var ls: std.ArrayList(Leaf) = .empty;
        try f.leaves(mb, "", "", 0, &ls);
        const agg = try l.memberText(mb);
        for (ls.items) |lf| {
            const v = try f.tmpInst("extractvalue {s} {s}{s}", .{ agg, value, lf.path });
            const p = try f.tmpInst("getelementptr {s}{s}, {s} {s}, {s}{s}", .{ if (ch.inb) "inbounds " else "", ch.src, ch.pty, ch.root, ch.idx, lf.gep });
            const la = @min(al, alignOfOffset(lf.offset), (try l.layout(lf.id)).alignment);
            try f.emit("store {s}{s} {s}, {s} {s}, align {d}", .{ f.vk(), lf.ty, v, ch.pty, p, la });
        }
    }

    /// The alignment an index step of `stride` bytes keeps: a constant
    /// index gives its exact offset's, a runtime one the stride's.
    fn stepAlign(f: *Fn, index: Id, stride: u64) u64 {
        if (f.l.m.constantBits(index)) |k| return alignOfOffset(k *% stride);
        return alignOfOffset(stride);
    }

    fn compositeExtract(f: *Fn, inst: Instruction) Error!void {
        const l = f.l;
        const comp = try f.opId(0);
        const ct = try f.typeId(comp);
        const indices = inst.operands[1..];
        if (indices.len == 0) return f.refuse("OpCompositeExtract without an index", .{});
        var path: std.ArrayList(u8) = .empty;
        var cur: Member = .{ .id = ct };
        var lane: ?u64 = null;
        for (indices) |o| {
            const k = o.asInt() orelse return f.refuse("an index is not a literal", .{});
            if (lane != null) return f.refuse("index past a vector component", .{});
            if (try f.aggStep(cur, k)) |st| {
                try path.print(l.a, ", {d}", .{st.index});
                cur = st.next;
            } else lane = k;
        }
        const want = if (lane != null) l.scalarOf(cur.id) else cur.id;
        if (!std.mem.eql(u8, try l.ty(want, .value), try l.ty(inst.result_type.?, .value))) return f.refuse("the result type is not the type of the extracted part", .{});
        if (path.items.len == 0) {
            return f.emit("{s} = extractelement {s}, i32 {d}", .{ try f.def(inst), try f.typed(comp), lane.? });
        }
        const t = try f.tmpInst("extractvalue {s} {s}{s}", .{ try l.ty(ct, .memory), try f.op(comp), path.items });
        const v = try f.memberToValue(t, cur);
        if (lane) |k| {
            return f.emit("{s} = extractelement {s} {s}, i32 {d}", .{ try f.def(inst), try l.ty(cur.id, .value), v, k });
        }
        try f.bind(inst.result.?, v);
        try f.untrackedPtr(inst.result.?);
    }

    fn compositeInsert(f: *Fn, inst: Instruction) Error!void {
        const l = f.l;
        const obj = try f.opId(0);
        const comp = try f.opId(1);
        const ct = try f.typeId(comp);
        const indices = inst.operands[2..];
        if (indices.len == 0) return f.refuse("OpCompositeInsert without an index", .{});
        var path: std.ArrayList(u8) = .empty;
        var cur: Member = .{ .id = ct };
        var lane: ?u64 = null;
        for (indices) |o| {
            const k = o.asInt() orelse return f.refuse("an index is not a literal", .{});
            if (lane != null) return f.refuse("index past a vector component", .{});
            if (try f.aggStep(cur, k)) |st| {
                try path.print(l.a, ", {d}", .{st.index});
                cur = st.next;
            } else lane = k;
        }
        const want = if (lane != null) l.scalarOf(cur.id) else cur.id;
        if (!std.mem.eql(u8, try l.ty(want, .value), try f.tyOf(obj))) return f.refuse("the object's type is not the type of the replaced part", .{});
        if (path.items.len == 0) {
            return f.emit("{s} = insertelement {s}, {s}, i32 {d}", .{ try f.def(inst), try f.typed(comp), try f.typed(obj), lane.? });
        }
        const agg_ty = try l.ty(ct, .memory);
        const agg = try f.op(comp);
        var member: []const u8 = undefined;
        if (lane) |k| {
            const old = try f.tmpInst("extractvalue {s} {s}{s}", .{ agg_ty, agg, path.items });
            const old_v = try f.memberToValue(old, cur);
            const new_v = try f.tmpInst("insertelement {s} {s}, {s}, i32 {d}", .{ try l.ty(cur.id, .value), old_v, try f.typed(obj), k });
            member = try f.valueToMember(new_v, cur);
        } else member = try f.valueToMember(try f.op(obj), cur);
        try f.emit("{s} = insertvalue {s} {s}, {s} {s}{s}", .{ try f.def(inst), agg_ty, agg, try l.memberText(cur), member, path.items });
    }

    fn compositeConstruct(f: *Fn, inst: Instruction) Error!void {
        const l = f.l;
        const rt = inst.result_type.?;
        const parts = inst.operands;
        switch (try l.typeOf(rt)) {
            .vector => |v| {
                const vty = try l.ty(rt, .value);
                const ety = try l.ty(v.elem, .value);
                var cur: []const u8 = "undef";
                var lane: u32 = 0;
                for (parts) |o| {
                    const pid = o.asId() orelse return f.refuse("a constituent is not an id", .{});
                    const ptid = try f.typeId(pid);
                    if (!std.mem.eql(u8, try l.ty(l.scalarOf(ptid), .value), ety)) return f.refuse("a constituent's component type is not the vector's", .{});
                    const n = l.vecLen(ptid) orelse 1;
                    if (lane + n > v.len) return f.refuse("the constituents have more than {d} components", .{v.len});
                    if (l.vecLen(ptid) != null) {
                        const pv = try f.op(pid);
                        const pty = try l.ty(ptid, .value);
                        for (0..n) |i| {
                            const e = try f.tmpInst("extractelement {s} {s}, i32 {d}", .{ pty, pv, i });
                            cur = try f.tmpInst("insertelement {s} {s}, {s} {s}, i32 {d}", .{ vty, cur, ety, e, lane });
                            lane += 1;
                        }
                    } else {
                        cur = try f.tmpInst("insertelement {s} {s}, {s} {s}, i32 {d}", .{ vty, cur, ety, try f.op(pid), lane });
                        lane += 1;
                    }
                }
                if (lane != v.len) return f.refuse("the constituents have {d} components for {d}", .{ lane, v.len });
                try f.bind(inst.result.?, cur);
            },
            .@"struct", .array, .matrix => {
                const agg_ty = try l.ty(rt, .memory);
                const count: u64 = switch (try l.typeOf(rt)) {
                    .@"struct" => |st| st.members.len,
                    .array => |arr| try l.arrayLen(arr.len),
                    .matrix => |mt| mt.cols,
                    else => unreachable,
                };
                if (parts.len != count) return f.refuse("{d} constituents for {d} members", .{ parts.len, count });
                var cur: []const u8 = "undef";
                for (parts, 0..) |o, i| {
                    const pid = o.asId() orelse return f.refuse("a constituent is not an id", .{});
                    const st = (try f.aggStep(.{ .id = rt }, i)).?;
                    if (!std.mem.eql(u8, try l.ty(st.next.id, .value), try f.tyOf(pid))) return f.refuse("constituent {d} does not have the member's type", .{i});
                    const m = try f.valueToMember(try f.op(pid), st.next);
                    cur = try f.tmpInst("insertvalue {s} {s}, {s} {s}, {d}", .{ agg_ty, cur, try l.memberText(st.next), m, st.index });
                }
                try f.bind(inst.result.?, cur);
            },
            else => return f.refuse("construct of a non-composite type", .{}),
        }
    }

    fn vectorShuffle(f: *Fn, inst: Instruction) Error!void {
        const l = f.l;
        const v1 = try f.opId(0);
        const v2 = try f.opId(1);
        const t1 = try f.typeId(v1);
        const t2 = try f.typeId(v2);
        const rt = inst.result_type.?;
        const comps = inst.operands[2..];
        const n1 = l.vecLen(t1) orelse return f.refuse("OpVectorShuffle of a non-vector", .{});
        const n2 = l.vecLen(t2) orelse return f.refuse("OpVectorShuffle of a non-vector", .{});
        const rn = l.vecLen(rt) orelse return f.refuse("OpVectorShuffle to a non-vector", .{});
        if (comps.len != rn) return f.refuse("{d} components for a {d}-component result", .{ comps.len, rn });
        for (comps) |o| {
            const c = o.asInt() orelse return f.refuse("a component is not a literal", .{});
            if (c != 0xFFFFFFFF and c >= n1 + n2) return f.refuse("component {d} past the {d} lanes of the operands", .{ c, n1 + n2 });
        }
        if (t1 == t2 or std.mem.eql(u8, try l.ty(t1, .value), try l.ty(t2, .value))) {
            var mask: std.ArrayList(u8) = .empty;
            try mask.print(l.a, "<{d} x i32> <", .{comps.len});
            for (comps, 0..) |o, i| {
                const c = o.asInt().?;
                if (c == 0xFFFFFFFF) try mask.print(l.a, "{s}i32 undef", .{if (i == 0) "" else ", "}) else try mask.print(l.a, "{s}i32 {d}", .{ if (i == 0) "" else ", ", c });
            }
            try mask.append(l.a, '>');
            return f.emit("{s} = shufflevector {s}, {s}, {s}", .{ try f.def(inst), try f.typed(v1), try f.typed(v2), mask.items });
        }
        // Operands of different lengths: lane by lane.
        const vty = try l.ty(rt, .value);
        const ety = try l.ty(l.scalarOf(rt), .value);
        var cur: []const u8 = "undef";
        for (comps, 0..) |o, i| {
            const c = o.asInt().?;
            if (c == 0xFFFFFFFF) continue;
            const e = if (c < n1)
                try f.tmpInst("extractelement {s}, i32 {d}", .{ try f.typed(v1), c })
            else
                try f.tmpInst("extractelement {s}, i32 {d}", .{ try f.typed(v2), c - n1 });
            cur = try f.tmpInst("insertelement {s} {s}, {s} {s}, i32 {d}", .{ vty, cur, ety, e, i });
        }
        try f.bind(inst.result.?, cur);
    }

    /// `air.dot.v<N>f32(x, y)` (the MSL `dot`).
    fn dot(f: *Fn, vt: Id, x: []const u8, y: []const u8) Error![]const u8 {
        const l = f.l;
        const vty = try l.ty(vt, .value);
        return f.call(try l.ty(l.scalarOf(vt), .value), try l.a.print("air.dot.{s}", .{try l.sfx(vt)}), &.{ .{ .ty = vty, .v = x }, .{ .ty = vty, .v = y } });
    }

    fn unary(f: *Fn, name: []const u8, rt: Id, x: []const u8) Error![]const u8 {
        const t = try f.l.ty(rt, .value);
        return f.call(t, try f.l.a.print("{s}.{s}", .{ name, try f.l.sfx(rt) }), &.{.{ .ty = t, .v = x }});
    }

    fn binary(f: *Fn, name: []const u8, rt: Id, x: []const u8, y: []const u8) Error![]const u8 {
        const t = try f.l.ty(rt, .value);
        return f.call(t, try f.l.a.print("{s}.{s}", .{ name, try f.l.sfx(rt) }), &.{ .{ .ty = t, .v = x }, .{ .ty = t, .v = y } });
    }

    /// `air.<fn>` or `air.fast_<fn>` by the library's fast-math choice.
    fn airName(f: *Fn, base: []const u8) Error![]const u8 {
        return f.l.a.print("air.{s}{s}", .{ if (f.l.opts.fast_math) "fast_" else "", base });
    }

    fn length(f: *Fn, vt: Id, x: []const u8) Error![]const u8 {
        if (f.l.vecLen(vt) == null) return f.unary("llvm.fabs", vt, x);
        const st = f.l.scalarOf(vt);
        return f.unary("llvm.sqrt", st, try f.dot(vt, x, x));
    }

    /// OpenCL C's length: like `length`, but without the overflow and
    /// underflow of the sum of squares (OpenCL bounds its error in ulps, so
    /// length(1e30, 0) is 1e30, not inf): `x` is scaled by a power of two
    /// first (`oclScale`, exact) and the result scaled back. Half vectors
    /// are measured in float (`oclLength`).
    fn lengthOcl(f: *Fn, vt: Id, x: []const u8) Error![]const u8 {
        const l = f.l;
        const n = l.vecLen(vt) orelse return f.unary("llvm.fabs", vt, x);
        if (try f.floatWidth(vt) == 16) {
            const xf = try f.tmpInst("fpext <{d} x half> {s} to <{d} x float>", .{ n, x, n });
            const len = try f.oclLength(n, xf);
            return f.tmpInst("fptrunc float {s} to half", .{len});
        }
        return f.oclLength(n, x);
    }

    /// The length of the `<n x float>` `x`, scaled (`oclScale`) and scaled
    /// back.
    fn oclLength(f: *Fn, n: u32, x: []const u8) Error![]const u8 {
        const sc = try f.oclScale(n, x);
        const len = try f.call("float", "llvm.sqrt.f32", &.{.{ .ty = "float", .v = try f.dotF32(n, sc.v, sc.v) }});
        return f.tmpInst("fmul float {s}, {s}", .{ len, sc.back });
    }

    /// `x` divided by its length, for the `<n x float>` `x` (`oclScale`
    /// first, so neither the sum of squares nor the length over- or
    /// underflows).
    fn oclNormalize(f: *Fn, n: u32, x: []const u8) Error![]const u8 {
        const sc = try f.oclScale(n, x);
        const len = try f.call("float", "llvm.sqrt.f32", &.{.{ .ty = "float", .v = try f.dotF32(n, sc.v, sc.v) }});
        return f.tmpInst("fdiv <{d} x float> {s}, {s}", .{ n, sc.v, try f.splatF32(n, len) });
    }

    fn dotF32(f: *Fn, n: u32, x: []const u8, y: []const u8) Error![]const u8 {
        const vty = try f.l.a.print("<{d} x float>", .{n});
        return f.call("float", try f.l.a.print("air.dot.v{d}f32", .{n}), &.{ .{ .ty = vty, .v = x }, .{ .ty = vty, .v = y } });
    }

    fn splatF32(f: *Fn, n: u32, text: []const u8) Error![]const u8 {
        const t1 = try f.tmpInst("insertelement <{d} x float> undef, float {s}, i32 0", .{ n, text });
        return f.tmpInst("shufflevector <{d} x float> {s}, <{d} x float> undef, <{d} x i32> zeroinitializer", .{ n, t1, n, n });
    }

    /// `r` = sin, tan or asin of `x`, with x returned where x is ±0: C99
    /// Annex F (required by OpenCL C 3.0 section 7.5, for half too) gives
    /// sin(±0) = tan(±0) = asin(±0) = ±0, and Metal's precise
    /// `air.sin.f16`, `air.tan.f16` and `air.asin.f16` return +0 for -0
    /// (Apple's assembler on the same IR too; every other negative half,
    /// subnormals included, keeps its sign, and so do f32 and the other
    /// odd functions; exec test half_signed_zero). Strict half only, as
    /// for atan: fast math may drop signed zeros.
    fn signedZero(f: *Fn, rt: Id, x: []const u8, r: []const u8) Error![]const u8 {
        const l = f.l;
        if (l.opts.fast_math or try f.floatWidth(rt) != 16) return r;
        const t = try l.ty(rt, .value);
        const z = try f.tmpInst("fcmp oeq {s} {s}, {s}", .{ t, x, try l.zeroOf(rt) });
        return f.tmpInst("select {s} {s}, {s} {s}, {s} {s}", .{ try l.boolTy(rt), z, t, x, t, r });
    }

    const Scaled = struct { v: []const u8, back: []const u8 };

    /// The `<n x float>` `x` times a power of two that keeps its sum of
    /// squares finite and normal (libclc's normalize does the same): 2^-70
    /// when `dot(x, x)` overflows, 2^86 when it is below the smallest
    /// normal (zero included), else 1; `back` is the inverse factor for a
    /// length. Infinite and NaN elements stay so. A half vector is widened
    /// to float first (`lengthOcl`, `normalize_ocl`): half's range is too
    /// narrow for one factor to do both (an element of 2^-24 needs 2^17 to
    /// square to a normal half, which takes a 2^-7 neighbour to 2^20, past
    /// half's 65504; with the 2^10 this used, normalize(half2(2^-24, 0))
    /// was (inf, NaN) and length(half2(2^-24, 0)) 0; exec test
    /// half_geometric_subnormal), while every half and every sum of their
    /// squares is a normal float.
    fn oclScale(f: *Fn, n: u32, x: []const u8) Error!Scaled {
        const a = f.l.a;
        const flt = struct {
            fn text(al: Allocator, v: f32) Error![]const u8 {
                return al.print("0x{X:0>16}", .{widenF32(@bitCast(v))});
            }
        }.text;
        const l2 = try f.dotF32(n, x, x);
        const big = try f.tmpInst("fcmp oeq float {s}, {s}", .{ l2, try flt(a, std.math.inf(f32)) });
        const small = try f.tmpInst("fcmp olt float {s}, {s}", .{ l2, try flt(a, 0x1p-126) });
        const one = try flt(a, 1);
        const s1 = try f.tmpInst("select i1 {s}, float {s}, float {s}", .{ small, try flt(a, 0x1p86), one });
        const sc = try f.tmpInst("select i1 {s}, float {s}, float {s}", .{ big, try flt(a, 0x1p-70), s1 });
        const b1 = try f.tmpInst("select i1 {s}, float {s}, float {s}", .{ small, try flt(a, 0x1p-86), one });
        const back = try f.tmpInst("select i1 {s}, float {s}, float {s}", .{ big, try flt(a, 0x1p70), b1 });
        const v = try f.tmpInst("fmul <{d} x float> {s}, {s}", .{ n, x, try f.splatF32(n, sc) });
        return .{ .v = v, .back = back };
    }

    fn extInst(f: *Fn, inst: Instruction) Error!void {
        const set = inst.ext_set orelse return f.refuse("extended instruction set is neither GLSL.std.450 nor OpenCL.std", .{});
        const raw = inst.intAt(1) orelse return f.refuse("OpExtInst without an instruction number", .{});
        const num = std.math.cast(u16, raw) orelse return f.refuse("unknown extended instruction {d}", .{raw});
        const which: Math = switch (set) {
            .@"GLSL.std.450" => blk: {
                const o = std.enums.fromInt(spec.GlslOpcode, num) orelse return f.refuse("unknown GLSL.std.450 instruction {d}", .{num});
                break :blk glslMath(o) orelse return f.refuse("GLSL.std.450 {t} is not supported", .{o});
            },
            .@"OpenCL.std" => blk: {
                const o = std.enums.fromInt(spec.OpenClOpcode, num) orelse return f.refuse("unknown OpenCL.std instruction {d}", .{num});
                break :blk oclMath(o) orelse return f.refuse("OpenCL.std {t} is not supported", .{o});
            },
            .core => return f.refuse("OpExtInst of the core set", .{}),
        };
        const rt = inst.result_type.?;
        const args = inst.operands[2..];
        var ids: [4]Id = undefined;
        for (args, 0..) |o, i| {
            if (i >= ids.len) return f.refuse("too many operands", .{});
            ids[i] = o.asId() orelse return f.refuse("operand {d} is not an id", .{i});
        }
        const r = try f.math(which, rt, ids[0..args.len]);
        try f.bind(inst.result.?, r);
    }

    fn math(f: *Fn, which: Math, rt: Id, ids: []const Id) Error![]const u8 {
        const l = f.l;
        const a = l.a;
        const t = try l.ty(rt, .value);
        const bt = try l.boolTy(rt);
        const arity: usize = switch (which) {
            .atan2, .pow, .powr, .copysign, .fmin, .fmax, .umin, .umax, .smin, .smax, .step, .ldexp, .distance, .distance_ocl, .cross, .fmod, .s_mul_hi, .u_mul_hi => 2,
            .fclamp, .uclamp, .sclamp, .fma, .mad, .mix_glsl, .mix_ocl, .smoothstep => 3,
            else => 1,
        };
        if (ids.len != arity) return f.refuse("{t} takes {d} operands", .{ which, arity });
        // Operand classes and shapes: every operand has the result's
        // component type (ldexp's exponent: an integer), as a vector of the
        // result's length or a scalar (OpenCL's `fmin(float4, float)`).
        const int_fn = switch (which) {
            .sabs, .ssign, .umin, .umax, .smin, .smax, .uclamp, .sclamp, .find_ilsb, .find_smsb, .find_umsb, .popcount, .clz, .ctz, .s_mul_hi, .u_mul_hi => true,
            else => false,
        };
        try f.needClass(rt, if (int_fn) .int else .float);
        const reduces = which == .length or which == .distance or which == .length_ocl or which == .distance_ocl;
        if (reduces and l.vecLen(rt) != null) return f.refuse("{t} returns a scalar", .{which});
        if (which == .cross and l.vecLen(rt) == null) return f.refuse("cross of scalars", .{});
        for (ids, 0..) |arg, i| {
            const at = try f.typeId(arg);
            if (which == .ldexp and i == 1) {
                try f.needClass(at, .int);
            } else if (!std.mem.eql(u8, try l.ty(l.scalarOf(at), .value), try l.ty(l.scalarOf(rt), .value))) {
                return f.refuse("operand {d} does not have the result's component type", .{i});
            }
            if (!reduces and l.vecLen(at) != null and l.vecLen(at) != l.vecLen(rt)) return f.refuse("operand {d} is not shaped like the result", .{i});
            if (reduces and i > 0 and l.vecLen(at) != l.vecLen(try f.typeId(ids[0]))) return f.refuse("the operands differ in shape", .{});
        }
        const x = try f.argAs(ids[0], rt);
        switch (which) {
            .round, .rint, .trunc, .fabs, .floor, .ceil, .sin, .cos, .tan, .exp, .log, .exp2, .log2, .exp10, .log10, .sqrt => {
                const name = try a.print("llvm.{t}", .{which});
                const r = try f.unary(name, rt, x);
                if (which == .sin or which == .tan) return f.signedZero(rt, x, r);
                return r;
            },
            .asin => return f.signedZero(rt, x, try f.unary(try f.airName("asin"), rt, x)),
            .acos, .sinh, .cosh, .tanh, .asinh, .acosh, .atanh, .rsqrt => return f.unary(try f.airName(@tagName(which)), rt, x),
            // atan and atan2 have the sign of x (atan) or y (atan2) in
            // every case (C99 Annex F, which OpenCL C 3.0 section 7.5
            // requires): atan(±0) = ±0, atan2(±0, x > 0) = ±0, and a result
            // that underflows keeps y's sign. Metal's precise
            // `air.atan.f32` returns +0 for -0, and `air.atan2.f32(-1e-30,
            // 1e30)` +0 (Apple's `precise::atan(-0.0f)` gives the same
            // 0x00000000), so the strict f32 forms take the sign back with
            // `llvm.copysign` (exec test atan_signed_zero). Half precision
            // and the fast forms already return -0.
            .atan, .atan2 => {
                const r = if (which == .atan)
                    try f.unary(try f.airName("atan"), rt, x)
                else
                    try f.binary(try f.airName("atan2"), rt, x, try f.argAs(ids[1], rt));
                if (l.opts.fast_math or try f.floatWidth(rt) != 32) return r;
                return f.binary("llvm.copysign", rt, r, x);
            },
            .pow => return f.binary("llvm.pow", rt, x, try f.argAs(ids[1], rt)),
            .powr => {
                // OpenCL C's powr is pow for x >= 0 with its own edge
                // cases (OpenCL C 3.0, "Additional Requirements Beyond
                // C99"): NaN for x < 0, for (±0, ±0), (+inf, ±0), (+1,
                // ±inf) and any NaN operand, where pow gives 1 or a finite
                // value; and +0 / +inf for x = -0 where pow keeps the sign
                // for odd integer y. So: pow(|x|, y), NaN where defined so.
                const y = try f.argAs(ids[1], rt);
                const z = try l.zeroOf(rt);
                const inf = try l.floatConst(rt, std.math.inf(f64));
                const p = try f.binary("llvm.pow", rt, try f.unary("llvm.fabs", rt, x), y);
                const neg = try f.tmpInst("fcmp ult {s} {s}, {s}", .{ t, x, z });
                const ynan = try f.tmpInst("fcmp uno {s} {s}, {s}", .{ t, y, y });
                const yz = try f.tmpInst("fcmp oeq {s} {s}, {s}", .{ t, y, z });
                const x0 = try f.tmpInst("fcmp oeq {s} {s}, {s}", .{ t, x, z });
                const xinf = try f.tmpInst("fcmp oeq {s} {s}, {s}", .{ t, x, inf });
                const x0i = try f.tmpInst("or {s} {s}, {s}", .{ bt, x0, xinf });
                const zero_pow = try f.tmpInst("and {s} {s}, {s}", .{ bt, yz, x0i });
                const x1 = try f.tmpInst("fcmp oeq {s} {s}, {s}", .{ t, x, try l.floatConst(rt, 1) });
                const yinf = try f.tmpInst("fcmp oeq {s} {s}, {s}", .{ t, try f.unary("llvm.fabs", rt, y), inf });
                const one_inf = try f.tmpInst("and {s} {s}, {s}", .{ bt, x1, yinf });
                const n1 = try f.tmpInst("or {s} {s}, {s}", .{ bt, neg, ynan });
                const n2 = try f.tmpInst("or {s} {s}, {s}", .{ bt, n1, zero_pow });
                const nan = try f.tmpInst("or {s} {s}, {s}", .{ bt, n2, one_inf });
                return f.tmpInst("select {s} {s}, {s} {s}, {s} {s}", .{ bt, nan, t, try l.floatConst(rt, std.math.nan(f64)), t, p });
            },
            .copysign => return f.binary("llvm.copysign", rt, x, try f.argAs(ids[1], rt)),
            .fmin => return f.binary("llvm.minnum", rt, x, try f.argAs(ids[1], rt)),
            .fmax => return f.binary("llvm.maxnum", rt, x, try f.argAs(ids[1], rt)),
            .umin, .umax, .smin, .smax => return f.binary(try a.print("llvm.{t}", .{which}), rt, x, try f.argAs(ids[1], rt)),
            .fclamp => return f.binary("llvm.minnum", rt, try f.binary("llvm.maxnum", rt, x, try f.argAs(ids[1], rt)), try f.argAs(ids[2], rt)),
            .uclamp => return f.binary("llvm.umin", rt, try f.binary("llvm.umax", rt, x, try f.argAs(ids[1], rt)), try f.argAs(ids[2], rt)),
            .sclamp => return f.binary("llvm.smin", rt, try f.binary("llvm.smax", rt, x, try f.argAs(ids[1], rt)), try f.argAs(ids[2], rt)),
            .fma => return f.call(t, try a.print("llvm.fma.{s}", .{try l.sfx(rt)}), &.{ .{ .ty = t, .v = x }, .{ .ty = t, .v = try f.argAs(ids[1], rt) }, .{ .ty = t, .v = try f.argAs(ids[2], rt) } }),
            .mad => {
                const p = try f.tmpInst("fmul {s} {s}, {s}", .{ t, x, try f.argAs(ids[1], rt) });
                return f.tmpInst("fadd {s} {s}, {s}", .{ t, p, try f.argAs(ids[2], rt) });
            },
            .fmod => return f.tmpInst("frem {s} {s}, {s}", .{ t, x, try f.argAs(ids[1], rt) }),
            .sabs => return f.call(t, try a.print("llvm.abs.{s}", .{try l.sfx(rt)}), &.{ .{ .ty = t, .v = x }, .{ .ty = "i1", .v = "false" } }),
            .fsign, .fsign_nan0 => {
                const z = try l.zeroOf(rt);
                const gt = try f.tmpInst("fcmp ogt {s} {s}, {s}", .{ t, x, z });
                const lt = try f.tmpInst("fcmp olt {s} {s}, {s}", .{ t, x, z });
                var rest = x;
                if (which == .fsign_nan0) {
                    const nan = try f.tmpInst("fcmp uno {s} {s}, {s}", .{ t, x, x });
                    rest = try f.tmpInst("select {s} {s}, {s} {s}, {s} {s}", .{ bt, nan, t, z, t, x });
                }
                const s1 = try f.tmpInst("select {s} {s}, {s} {s}, {s} {s}", .{ bt, lt, t, try l.floatConst(rt, -1), t, rest });
                return f.tmpInst("select {s} {s}, {s} {s}, {s} {s}", .{ bt, gt, t, try l.floatConst(rt, 1), t, s1 });
            },
            .ssign => {
                const z = try l.zeroOf(rt);
                const gt = try f.tmpInst("icmp sgt {s} {s}, {s}", .{ t, x, z });
                const lt = try f.tmpInst("icmp slt {s} {s}, {s}", .{ t, x, z });
                const s1 = try f.tmpInst("select {s} {s}, {s} {s}, {s} {s}", .{ bt, lt, t, try l.intConst(rt, -1), t, z });
                return f.tmpInst("select {s} {s}, {s} {s}, {s} {s}", .{ bt, gt, t, try l.intConst(rt, 1), t, s1 });
            },
            .fract => {
                const fl = try f.unary("llvm.floor", rt, x);
                return f.tmpInst("fsub {s} {s}, {s}", .{ t, x, fl });
            },
            .radians => return f.tmpInst("fmul {s} {s}, {s}", .{ t, x, try l.floatConst(rt, std.math.pi / 180.0) }),
            .degrees => return f.tmpInst("fmul {s} {s}, {s}", .{ t, x, try l.floatConst(rt, 180.0 / std.math.pi) }),
            .mix_glsl => {
                const y = try f.argAs(ids[1], rt);
                const s = try f.argAs(ids[2], rt);
                const one_minus = try f.tmpInst("fsub {s} {s}, {s}", .{ t, try l.floatConst(rt, 1), s });
                const p1 = try f.tmpInst("fmul {s} {s}, {s}", .{ t, x, one_minus });
                const p2 = try f.tmpInst("fmul {s} {s}, {s}", .{ t, y, s });
                return f.tmpInst("fadd {s} {s}, {s}", .{ t, p1, p2 });
            },
            .mix_ocl => {
                const y = try f.argAs(ids[1], rt);
                const s = try f.argAs(ids[2], rt);
                const d = try f.tmpInst("fsub {s} {s}, {s}", .{ t, y, x });
                const p = try f.tmpInst("fmul {s} {s}, {s}", .{ t, d, s });
                return f.tmpInst("fadd {s} {s}, {s}", .{ t, x, p });
            },
            .step => {
                // step(edge, x): 0 where x < edge, else 1.
                const v = try f.argAs(ids[1], rt);
                const c = try f.tmpInst("fcmp olt {s} {s}, {s}", .{ t, v, x });
                return f.tmpInst("select {s} {s}, {s} {s}, {s} {s}", .{ bt, c, t, try l.zeroOf(rt), t, try l.floatConst(rt, 1) });
            },
            .smoothstep => {
                const e1 = try f.argAs(ids[1], rt);
                const v = try f.argAs(ids[2], rt);
                const num = try f.tmpInst("fsub {s} {s}, {s}", .{ t, v, x });
                const den = try f.tmpInst("fsub {s} {s}, {s}", .{ t, e1, x });
                const q = try f.tmpInst("fdiv {s} {s}, {s}", .{ t, num, den });
                const lo = try f.binary("llvm.maxnum", rt, q, try l.zeroOf(rt));
                const c = try f.binary("llvm.minnum", rt, lo, try l.floatConst(rt, 1));
                const two_c = try f.tmpInst("fmul {s} {s}, {s}", .{ t, c, try l.floatConst(rt, 2) });
                const three = try f.tmpInst("fsub {s} {s}, {s}", .{ t, try l.floatConst(rt, 3), two_c });
                const sq = try f.tmpInst("fmul {s} {s}, {s}", .{ t, c, c });
                return f.tmpInst("fmul {s} {s}, {s}", .{ t, sq, three });
            },
            .ldexp => {
                const et = try f.typeId(ids[1]);
                var e = try f.op(ids[1]);
                const ew = try f.intWidth(et);
                if (ew != 32) e = try f.tmpInst("{s} {s} {s} to {s}", .{ if (ew > 32) "trunc" else "sext", try l.ty(et, .value), e, try l.intTyBits(et, 32) });
                if (l.vecLen(rt) != null and l.vecLen(et) == null) {
                    const n = l.vecLen(rt).?;
                    const vt = try a.print("<{d} x i32>", .{n});
                    const t1 = try f.tmpInst("insertelement {s} undef, i32 {s}, i32 0", .{ vt, e });
                    e = try f.tmpInst("shufflevector {s} {s}, {s} undef, <{d} x i32> zeroinitializer", .{ vt, t1, vt, n });
                }
                const it = if (l.vecLen(rt)) |n| try a.print("<{d} x i32>", .{n}) else "i32";
                const isfx = if (l.vecLen(rt)) |n| try a.print("v{d}i32", .{n}) else "i32";
                return f.call(t, try a.print("llvm.ldexp.{s}.{s}", .{ try l.sfx(rt), isfx }), &.{ .{ .ty = t, .v = x }, .{ .ty = it, .v = e } });
            },
            .length => return f.length(try f.typeId(ids[0]), try f.op(ids[0])),
            .distance => {
                const vt = try f.typeId(ids[0]);
                const d = try f.tmpInst("fsub {s} {s}, {s}", .{ try l.ty(vt, .value), try f.op(ids[0]), try f.op(ids[1]) });
                return f.length(vt, d);
            },
            .normalize => {
                if (l.vecLen(rt) == null) {
                    const abs = try f.unary("llvm.fabs", rt, x);
                    return f.tmpInst("fdiv {s} {s}, {s}", .{ t, x, abs });
                }
                const len = try f.length(rt, x);
                return f.tmpInst("fdiv {s} {s}, {s}", .{ t, x, try f.splat(rt, len) });
            },
            .length_ocl => return f.lengthOcl(try f.typeId(ids[0]), try f.op(ids[0])),
            .distance_ocl => {
                const vt = try f.typeId(ids[0]);
                const d = try f.tmpInst("fsub {s} {s}, {s}", .{ try l.ty(vt, .value), try f.op(ids[0]), try f.op(ids[1]) });
                return f.lengthOcl(vt, d);
            },
            .normalize_ocl => {
                // OpenCL C's normalize (3.0, geometric functions): v when
                // every element is zero (GLSL leaves that undefined; v/|v|
                // is NaN), NaN when an element is NaN, and with infinite
                // elements as if each were copysign(1, e) and every finite
                // one 0 * e. A scalar is so ±1, or itself when 0 or NaN.
                const z = try l.zeroOf(rt);
                const one = try l.floatConst(rt, 1);
                const sgn = try f.binary("llvm.copysign", rt, one, x);
                const isz = try f.tmpInst("fcmp oeq {s} {s}, {s}", .{ t, x, z });
                const n = l.vecLen(rt) orelse {
                    const isn = try f.tmpInst("fcmp uno {s} {s}, {s}", .{ t, x, x });
                    const keep = try f.tmpInst("or i1 {s}, {s}", .{ isz, isn });
                    return f.tmpInst("select i1 {s}, {s} {s}, {s} {s}", .{ keep, t, x, t, sgn });
                };
                const isinf = try f.tmpInst("fcmp oeq {s} {s}, {s}", .{ t, try f.unary("llvm.fabs", rt, x), try l.floatConst(rt, std.math.inf(f64)) });
                const anyinf = try f.call("i1", try a.print("air.any.v{d}i1", .{n}), &.{.{ .ty = bt, .v = isinf }});
                const zx = try f.tmpInst("fmul {s} {s}, {s}", .{ t, x, z });
                const repl = try f.tmpInst("select {s} {s}, {s} {s}, {s} {s}", .{ bt, isinf, t, sgn, t, zx });
                const v = try f.tmpInst("select i1 {s}, {s} {s}, {s} {s}", .{ anyinf, t, repl, t, x });
                const q = if (try f.floatWidth(rt) == 16) blk: {
                    const vf = try f.tmpInst("fpext <{d} x half> {s} to <{d} x float>", .{ n, v, n });
                    break :blk try f.tmpInst("fptrunc <{d} x float> {s} to <{d} x half>", .{ n, try f.oclNormalize(n, vf), n });
                } else try f.oclNormalize(n, v);
                const allz = try f.call("i1", try a.print("air.all.v{d}i1", .{n}), &.{.{ .ty = bt, .v = isz }});
                return f.tmpInst("select i1 {s}, {s} {s}, {s} {s}", .{ allz, t, x, t, q });
            },
            .cross => {
                const y = try f.op(ids[1]);
                if (l.vecLen(rt) != 3 and l.vecLen(rt) != 4) return f.refuse("cross of a vector that is not 3 or 4 wide", .{});
                const n = l.vecLen(rt).?;
                const m1 = if (n == 3) "<3 x i32> <i32 1, i32 2, i32 0>" else "<4 x i32> <i32 1, i32 2, i32 0, i32 3>";
                const m2 = if (n == 3) "<3 x i32> <i32 2, i32 0, i32 1>" else "<4 x i32> <i32 2, i32 0, i32 1, i32 3>";
                const x_yzx = try f.tmpInst("shufflevector {s} {s}, {s} undef, {s}", .{ t, x, t, m1 });
                const y_zxy = try f.tmpInst("shufflevector {s} {s}, {s} undef, {s}", .{ t, y, t, m2 });
                const x_zxy = try f.tmpInst("shufflevector {s} {s}, {s} undef, {s}", .{ t, x, t, m2 });
                const y_yzx = try f.tmpInst("shufflevector {s} {s}, {s} undef, {s}", .{ t, y, t, m1 });
                const p1 = try f.tmpInst("fmul {s} {s}, {s}", .{ t, x_yzx, y_zxy });
                const p2 = try f.tmpInst("fmul {s} {s}, {s}", .{ t, x_zxy, y_yzx });
                const r = try f.tmpInst("fsub {s} {s}, {s}", .{ t, p1, p2 });
                if (n == 4) return f.tmpInst("insertelement {s} {s}, {s} {s}, i32 3", .{ t, r, try l.ty(l.scalarOf(rt), .value), try l.floatConst(l.scalarOf(rt), 0) });
                return r;
            },
            .find_ilsb, .ctz => {
                const xt = try f.typeId(ids[0]);
                const xty = try l.ty(xt, .value);
                const tz = try f.call(xty, try a.print("llvm.cttz.{s}", .{try l.sfx(xt)}), &.{ .{ .ty = xty, .v = x }, .{ .ty = "i1", .v = "false" } });
                if (which == .ctz) return tz;
                const isz = try f.tmpInst("icmp eq {s} {s}, {s}", .{ xty, x, try l.zeroOf(xt) });
                return f.tmpInst("select {s} {s}, {s} {s}, {s} {s}", .{ try l.boolTy(xt), isz, xty, try l.intConst(xt, -1), xty, tz });
            },
            .find_umsb, .find_smsb, .clz => {
                const xt = try f.typeId(ids[0]);
                const xty = try l.ty(xt, .value);
                const w: i64 = try f.intWidth(xt);
                var y = x;
                if (which == .find_smsb) {
                    const s = try f.tmpInst("ashr {s} {s}, {s}", .{ xty, x, try l.intConst(xt, w - 1) });
                    y = try f.tmpInst("xor {s} {s}, {s}", .{ xty, x, s });
                }
                const lz = try f.call(xty, try a.print("llvm.ctlz.{s}", .{try l.sfx(xt)}), &.{ .{ .ty = xty, .v = y }, .{ .ty = "i1", .v = "false" } });
                if (which == .clz) return lz;
                return f.tmpInst("sub {s} {s}, {s}", .{ xty, try l.intConst(xt, w - 1), lz });
            },
            .popcount => return f.unary("llvm.ctpop", rt, x),
            .s_mul_hi, .u_mul_hi => {
                const lo_hi = try f.mulWide(rt, x, try f.argAs(ids[1], rt), which == .s_mul_hi);
                return lo_hi[1];
            },
        }
    }
};

// ── Tests ───────────────────────────────────────────────────────────────────

const asm_ = @import("asm.zig");
const assembler = @import("../air/assembler.zig");

fn has(hay: []const u8, needle: []const u8) bool {
    return std.mem.find(u8, hay, needle) != null;
}

pub const TestResult = struct { ir: []const u8, kernels: []const manifest.Kernel, message: []const u8 };

/// Assemble `text`, read it and lower it; on success also assemble every
/// kernel through tools/air/assembler.zig, so a lowering the assembler
/// cannot read fails the test.
pub fn lowerTextForTest(a: Allocator, text: []const u8, opts: Options) !TestResult {
    var adiag: asm_.Diagnostics = .{};
    const bytes = asm_.assemble(a, try withNames(a, text), &adiag) catch |err| {
        std.debug.print("asm line {d}: {s}\n", .{ adiag.line, adiag.message });
        return err;
    };
    var rdiag: reader.Diagnostics = .{};
    var m = reader.read(a, bytes, &rdiag) catch |err| {
        std.debug.print("read word {d}: {s}\n", .{ rdiag.word_offset, rdiag.message });
        return err;
    };
    var diag: Diagnostics = .{};
    const out = lower(a, &m, opts, &diag) catch |err| switch (err) {
        error.Unsupported => return .{ .ir = "", .kernels = &.{}, .message = diag.message },
        else => return err,
    };
    for (out.kernels) |k| {
        const fm = try manifest.functionMetadata(a, k);
        _ = assembler.assemble(a, out.ir, .{ .entry = k.name, .fm = fm, .check_operand_types = true }, air_target.default, .{ .fast_math = opts.fast_math }) catch |err| {
            std.debug.print("assembling {s} failed: {t}\n{s}\n", .{ k.name, err, out.ir });
            return err;
        };
    }
    return .{ .ir = out.ir, .kernels = out.kernels, .message = "" };
}

/// `text` with an `OpName %x "x"` for every symbolic result id that has
/// none, so the lowered IR in the tests reads `%x = add ...` rather than
/// `%v31 = add ...` (the assembler drops symbolic names).
fn withNames(a: Allocator, text: []const u8) ![]const u8 {
    var named: std.StringHashMapUnmanaged(void) = .empty;
    var names: std.ArrayList(u8) = .empty;
    var it = std.mem.splitScalar(u8, text, '\n');
    while (it.next()) |raw| {
        const line = std.mem.trim(u8, raw, " \t\r");
        if (std.mem.startsWith(u8, line, "OpName %")) {
            const end = std.mem.findScalar(u8, line[8..], ' ') orelse continue;
            try named.put(a, line[8..][0..end], {});
        }
    }
    it = std.mem.splitScalar(u8, text, '\n');
    while (it.next()) |raw| {
        const line = std.mem.trim(u8, raw, " \t\r");
        if (line.len < 2 or line[0] != '%' or std.ascii.isDigit(line[1])) continue;
        const end = std.mem.find(u8, line, " = Op") orelse continue;
        const sym = line[1..end];
        if (std.mem.startsWith(u8, line[end + 3 ..], "OpType") or std.mem.startsWith(u8, line[end + 3 ..], "OpLabel")) continue;
        if (named.contains(sym)) continue;
        try named.put(a, sym, {});
        try names.print(a, "OpName %{s} \"{s}\"\n", .{ sym, sym });
    }
    var out: std.ArrayList(u8) = .empty;
    var inserted = false;
    it = std.mem.splitScalar(u8, text, '\n');
    while (it.next()) |raw| {
        const line = std.mem.trim(u8, raw, " \t\r");
        if (!inserted and (std.mem.startsWith(u8, line, "OpDecorate") or std.mem.startsWith(u8, line, "OpMemberDecorate") or (line.len > 0 and line[0] == '%'))) {
            try out.appendSlice(a, names.items);
            inserted = true;
        }
        try out.appendSlice(a, raw);
        try out.append(a, '\n');
    }
    return out.items;
}

/// A Vulkan compute module: `in` (binding 0, read-only) and `out` (binding 1)
/// storage buffers of 64 elements of %T, the global invocation id, and one
/// kernel `k` whose body is `body` (it may use %gid_x, %pin_x / %pout_x
/// element pointers, %x the loaded input). `types` adds type/constant lines.
fn kernelModule(a: Allocator, elem: []const u8, types: []const u8, body: []const u8) ![]const u8 {
    return kernelModuleWith(a, elem, "", types, body);
}

/// `kernelModule` with extra decoration lines.
fn kernelModuleWith(a: Allocator, elem: []const u8, decorations: []const u8, types: []const u8, body: []const u8) ![]const u8 {
    return a.print(
        \\OpCapability Shader
        \\OpCapability Int64
        \\OpCapability Int16
        \\OpCapability Int8
        \\OpCapability Float16
        \\%glsl = OpExtInstImport "GLSL.std.450"
        \\OpMemoryModel Logical GLSL450
        \\OpEntryPoint GLCompute %k "k" %gid
        \\OpExecutionMode %k LocalSize 64 1 1
        \\OpName %in "in"
        \\OpName %out "out"
        \\OpName %Buf "Buf"
        \\OpMemberName %Buf 0 "data"
        \\OpDecorate %gid BuiltIn GlobalInvocationId
        \\OpDecorate %arr ArrayStride {[stride]d}
        \\OpDecorate %Buf Block
        \\OpMemberDecorate %Buf 0 Offset 0
        \\OpDecorate %in DescriptorSet 0
        \\OpDecorate %in Binding 0
        \\OpDecorate %in NonWritable
        \\OpDecorate %out DescriptorSet 0
        \\OpDecorate %out Binding 1
        \\{[decorations]s}
        \\%void = OpTypeVoid
        \\%fn_void = OpTypeFunction %void
        \\%bool = OpTypeBool
        \\%u32 = OpTypeInt 32 0
        \\%i32 = OpTypeInt 32 1
        \\%u64 = OpTypeInt 64 0
        \\%f32 = OpTypeFloat 32
        \\%v3u32 = OpTypeVector %u32 3
        \\%c0 = OpConstant %u32 0
        \\%c1 = OpConstant %u32 1
        \\%c64 = OpConstant %u32 64
        \\{[types]s}
        \\%arr = OpTypeArray {[elem]s} %c64
        \\%Buf = OpTypeStruct %arr
        \\%p_Buf = OpTypePointer StorageBuffer %Buf
        \\%p_elem = OpTypePointer StorageBuffer {[elem]s}
        \\%p_gid = OpTypePointer Input %v3u32
        \\%gid = OpVariable %p_gid Input
        \\%in = OpVariable %p_Buf StorageBuffer
        \\%out = OpVariable %p_Buf StorageBuffer
        \\%k = OpFunction %void None %fn_void
        \\%entry = OpLabel
        \\%gid_v = OpLoad %v3u32 %gid
        \\%gid_x = OpCompositeExtract %u32 %gid_v 0
        \\%pin_x = OpAccessChain %p_elem %in %c0 %gid_x
        \\%pout_x = OpAccessChain %p_elem %out %c0 %gid_x
        \\%x = OpLoad {[elem]s} %pin_x
        \\{[body]s}
        \\OpReturn
        \\OpFunctionEnd
        \\
    , .{ .elem = elem, .decorations = decorations, .types = types, .body = body, .stride = strideOf(elem) });
}

fn strideOf(elem: []const u8) u32 {
    if (std.mem.eql(u8, elem, "%v4f32") or std.mem.eql(u8, elem, "%v3f32") or std.mem.eql(u8, elem, "%v4i32")) return 16;
    if (std.mem.eql(u8, elem, "%u64") or std.mem.eql(u8, elem, "%v2f32")) return 8;
    if (std.mem.eql(u8, elem, "%f16")) return 2;
    return 4;
}

fn expectLowered(a: Allocator, elem: []const u8, types: []const u8, body: []const u8, needles: []const []const u8) !void {
    const text = try kernelModule(a, elem, types, body);
    const r = try lowerTextForTest(a, text, .{});
    if (r.message.len > 0) {
        std.debug.print("refused: {s}\n", .{r.message});
        return error.TestUnexpectedResult;
    }
    for (needles) |n| {
        if (!has(r.ir, n)) {
            std.debug.print("missing `{s}` in:\n{s}\n", .{ n, r.ir });
            return error.TestUnexpectedResult;
        }
    }
}

fn expectRefused(a: Allocator, elem: []const u8, types: []const u8, body: []const u8, needle: []const u8) !void {
    const text = try kernelModule(a, elem, types, body);
    const r = try lowerTextForTest(a, text, .{});
    if (r.message.len == 0 or !has(r.message, needle)) {
        std.debug.print("expected a refusal with `{s}`, got `{s}`\n{s}\n", .{ needle, r.message, r.ir });
        return error.TestUnexpectedResult;
    }
}

test "interface: Block buffers by binding, builtin lane loads, access chains, manifest" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const text = try kernelModule(a, "%i32", "", "%y = OpIAdd %i32 %x %x\nOpStore %pout_x %y");
    const r = try lowerTextForTest(a, text, .{});
    try std.testing.expectEqualStrings("", r.message);
    try std.testing.expect(has(r.ir, "target triple = \"air64_v28-apple-macosx26.0.0\""));
    try std.testing.expect(has(r.ir, "define void @k(ptr addrspace(1) %a0, ptr addrspace(1) %a1, <3 x i32> %a2) {"));
    try std.testing.expect(has(r.ir, "= extractelement <3 x i32> %a2, i32 0"));
    try std.testing.expect(has(r.ir, "= getelementptr { [64 x i32] }, ptr addrspace(1) %a0, i64 0, i32 0, i32 %gid_x"));
    try std.testing.expect(has(r.ir, "%x = load i32, ptr addrspace(1) %pin_x, align 4"));
    try std.testing.expect(has(r.ir, "%y = add i32 %x, %x"));
    try std.testing.expect(has(r.ir, "store i32 %y, ptr addrspace(1) %pout_x, align 4"));
    try std.testing.expectEqual(1, r.kernels.len);
    const k = r.kernels[0];
    try std.testing.expectEqual([3]u32{ 64, 1, 1 }, k.local_size);
    try std.testing.expectEqual(3, k.params.len);
    try std.testing.expectEqualStrings("in", k.params[0].buffer.name);
    try std.testing.expectEqual(manifest.Access.read, k.params[0].buffer.access);
    try std.testing.expectEqual(manifest.Access.read_write, k.params[1].buffer.access);
    try std.testing.expectEqual(@as(?u32, 256), k.params[0].buffer.buffer_size);
    try std.testing.expectEqualStrings("Buf", k.params[0].buffer.type_name);
    const field = k.params[0].buffer.fields.?[0];
    try std.testing.expectEqual(0, field.offset);
    try std.testing.expectEqual(4, field.size);
    try std.testing.expectEqual(64, field.count);
    try std.testing.expectEqualStrings("int", field.type_name);
    try std.testing.expectEqualStrings("data", field.name);
    try std.testing.expectEqual(air.BuiltinKind.thread_position_in_grid, k.params[2].builtin.kind);
    try std.testing.expectEqualStrings("uint3", k.params[2].builtin.type_name);
}

test "integer arithmetic, shifts with mixed widths, SMod and FMod expansions, comparisons, select" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try expectLowered(a, "%i32", "%c3 = OpConstant %i32 3\n%c2_64 = OpConstant %u64 2",
        \\%r1 = OpISub %i32 %x %c3
        \\%r2 = OpIMul %i32 %r1 %x
        \\%r3 = OpSDiv %i32 %r2 %c3
        \\%r4 = OpSRem %i32 %r3 %c3
        \\%r5 = OpSMod %i32 %r4 %x
        \\%r6 = OpShiftLeftLogical %i32 %r5 %c2_64
        \\%r7 = OpShiftRightArithmetic %i32 %r6 %c1
        \\%r8 = OpNot %i32 %r7
        \\%r9 = OpSNegate %i32 %r8
        \\%c = OpSLessThan %bool %r9 %c3
        \\%r10 = OpSelect %i32 %c %r9 %x
        \\OpStore %pout_x %r10
    , &.{
        "%r1 = sub i32 %x, 3",
        "%r4 = srem i32 %r3, 3",
        "= srem i32 %r4, %x",
        "= icmp slt i32",
        "= trunc i64 2 to i32",
        "%r7 = ashr i32 %r6, 1",
        "%r8 = xor i32 %r7, -1",
        "%r9 = sub i32 0, %r8",
        "%c = icmp slt i32 %r9, 3",
        "%r10 = select i1 %c, i32 %r9, i32 %x",
    });
    try expectLowered(a, "%f32", "%f2 = OpConstant %f32 2.5",
        \\%r1 = OpFMod %f32 %x %f2
        \\%r2 = OpFRem %f32 %r1 %f2
        \\%r3 = OpFNegate %f32 %r2
        \\%c1b = OpFUnordLessThan %bool %r3 %x
        \\%c2b = OpIsNan %bool %r3
        \\%c3b = OpLogicalOr %bool %c1b %c2b
        \\%c4b = OpIsInf %bool %x
        \\%c5b = OpLogicalAnd %bool %c3b %c4b
        \\%r4 = OpSelect %f32 %c5b %r3 %x
        \\OpStore %pout_x %r4
    , &.{
        "= frem float %x, 0x4004000000000000",
        "= fcmp une float",
        "%r2 = frem float %r1, 0x4004000000000000",
        "%r3 = fneg float %r2",
        "%c1b = fcmp ult float %r3, %x",
        "%c2b = fcmp uno float %r3, %r3",
        "%c3b = or i1 %c1b, %c2b",
        "call float @llvm.fabs.f32(float %x)",
        "%c4b = fcmp oeq float %t",
        "0x7FF0000000000000",
        "declare float @llvm.fabs.f32(float)",
    });
}

test "producer workaround: a Zig module's OpSConvert of an unsigned source zero-extends" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    // What Zig's backend emits for `@as(i32, @as(u16, @truncate(x)))` (the
    // crosscheck's t16u helper), scalar and vector, and a signed source.
    const types = "%u16 = OpTypeInt 16 0\n%i8 = OpTypeInt 8 1\n%v2u16 = OpTypeVector %u16 2\n%v2i32 = OpTypeVector %i32 2";
    const body =
        \\%n = OpUConvert %u16 %x
        \\%w = OpSConvert %i32 %n
        \\%nv = OpCompositeConstruct %v2u16 %n %n
        \\%wv = OpSConvert %v2i32 %nv
        \\%e = OpCompositeExtract %i32 %wv 1
        \\%s8 = OpUConvert %i8 %x
        \\%ws = OpSConvert %i32 %s8
        \\%sum = OpIAdd %i32 %w %e
        \\%sum2 = OpIAdd %i32 %sum %ws
        \\%r = OpBitcast %u32 %sum2
        \\OpStore %pout_x %r
    ;
    const module = try kernelModule(a, "%u32", types, body);
    const from_zig = try std.mem.concat(a, u8, &.{ "; Generator: 0x00290900\n", module });
    const zig = try lowerTextForTest(a, from_zig, .{});
    try std.testing.expectEqualStrings("", zig.message);
    try std.testing.expect(has(zig.ir, "%w = zext i16 %n to i32"));
    try std.testing.expect(has(zig.ir, "%wv = zext <2 x i16> %t2 to <2 x i32>"));
    try std.testing.expect(has(zig.ir, "%ws = sext i8 %s8 to i32")); // a signed source keeps sext
    // Switched off, or from another producer: the specification's sign extension.
    const off = try lowerTextForTest(a, from_zig, .{ .zig_workarounds = false });
    try std.testing.expect(has(off.ir, "%w = sext i16 %n to i32"));
    try std.testing.expect(has(off.ir, "%wv = sext <2 x i16> %t2 to <2 x i32>"));
    const other = try lowerTextForTest(a, try std.mem.concat(a, u8, &.{ "; Generator: 0x00080001\n", module }), .{});
    try std.testing.expect(has(other.ir, "%w = sext i16 %n to i32"));
    try std.testing.expect(has(other.ir, "%wv = sext <2 x i16> %t2 to <2 x i32>"));
}

test "producer workaround: only a signed result marks Zig's unsigned widening" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    // Zig's OpenCL flavour types every integer unsigned, so `OpSConvert %u32
    // %u16` there is `@as(i32, @as(i16, x))`, a real sign extension: the
    // workaround must leave it (and every narrowing OpSConvert) alone.
    const types = "%u16 = OpTypeInt 16 0\n%v2u16 = OpTypeVector %u16 2\n%v2u32 = OpTypeVector %u32 2";
    const body =
        \\%n = OpUConvert %u16 %x
        \\%w = OpSConvert %u32 %n
        \\%nv = OpCompositeConstruct %v2u16 %n %n
        \\%wv = OpSConvert %v2u32 %nv
        \\%e = OpCompositeExtract %u32 %wv 0
        \\%back = OpSConvert %u16 %e
        \\%wb = OpUConvert %u32 %back
        \\%sum = OpIAdd %u32 %w %wb
        \\OpStore %pout_x %sum
    ;
    const module = try kernelModule(a, "%u32", types, body);
    const zig = try lowerTextForTest(a, try std.mem.concat(a, u8, &.{ "; Generator: 0x00290900\n", module }), .{});
    try std.testing.expectEqualStrings("", zig.message);
    try std.testing.expect(has(zig.ir, "%w = sext i16 %n to i32"));
    try std.testing.expect(has(zig.ir, "%wv = sext <2 x i16> %t2 to <2 x i32>"));
    try std.testing.expect(has(zig.ir, "%back = trunc i32 %e to i16"));
    try std.testing.expect(!has(zig.ir, "zext i16 %n"));
}

test "producer workaround: a signed source widened into a sign-filled limb sign-extends" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    // `%l` is Zig's composite-integer widening of an i16 into the u64 limb 0
    // of an i128, whose upper limbs are filled from limb 0's top bit: a sign
    // extension. `%p` is the packed-pointer store's shape (shifted and ORed:
    // a zero extension), `%q` is shifted by less than 63 (not a sign fill),
    // `%m` has an unsigned operand (the OpenCL typing, or user code doing
    // `@as(i64, @bitCast(@as(u64, u16x))) >> 63`): all three stay `zext`.
    const types =
        \\%u16 = OpTypeInt 16 0
        \\%i16 = OpTypeInt 16 1
        \\%i64 = OpTypeInt 64 1
        \\%c63 = OpConstant %i64 63
        \\%c62 = OpConstant %i64 62
        \\%c8w = OpConstant %u64 8
    ;
    const body =
        \\%h = OpUConvert %u16 %x
        \\%s = OpBitcast %i16 %h
        \\%l = OpUConvert %u64 %s
        \\%li = OpBitcast %i64 %l
        \\%fill = OpShiftRightArithmetic %i64 %li %c63
        \\%p = OpUConvert %u64 %s
        \\%ps = OpShiftLeftLogical %u64 %p %c8w
        \\%q = OpUConvert %u64 %s
        \\%qi = OpBitcast %i64 %q
        \\%qs = OpShiftRightArithmetic %i64 %qi %c62
        \\%m = OpUConvert %u64 %h
        \\%mi = OpBitcast %i64 %m
        \\%ms = OpShiftRightArithmetic %i64 %mi %c63
        \\%fu = OpBitcast %u64 %fill
        \\%o1 = OpBitwiseOr %u64 %fu %ps
        \\%qu = OpBitcast %u64 %qs
        \\%o2 = OpBitwiseOr %u64 %o1 %qu
        \\%mu = OpBitcast %u64 %ms
        \\%o3 = OpBitwiseOr %u64 %o2 %mu
        \\%r = OpUConvert %u32 %o3
        \\OpStore %pout_x %r
    ;
    const module = try kernelModule(a, "%u32", types, body);
    const from_zig = try std.mem.concat(a, u8, &.{ "; Generator: 0x00290900\n", module });
    const zig = try lowerTextForTest(a, from_zig, .{});
    try std.testing.expectEqualStrings("", zig.message);
    try std.testing.expect(has(zig.ir, "%l = sext i16 %h to i64"));
    try std.testing.expect(has(zig.ir, "%p = zext i16 %h to i64"));
    try std.testing.expect(has(zig.ir, "%q = zext i16 %h to i64"));
    try std.testing.expect(has(zig.ir, "%m = zext i16 %h to i64"));
    try std.testing.expect(has(zig.ir, "%r = trunc i64 %o3 to i32"));
    // Switched off, or from another producer: the specification's zero extension.
    const off = try lowerTextForTest(a, from_zig, .{ .zig_workarounds = false });
    try std.testing.expectEqualStrings("", off.message);
    try std.testing.expect(has(off.ir, "%l = zext i16 %h to i64"));
    const other = try lowerTextForTest(a, try std.mem.concat(a, u8, &.{ "; Generator: 0x00080001\n", module }), .{});
    try std.testing.expectEqualStrings("", other.message);
    try std.testing.expect(has(other.ir, "%l = zext i16 %h to i64"));
}

test "producer workaround on Zig's i128 widenings (testdata/convert_wide.zig)" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    // i16 and i32 sources widened into an i128's sign-filled limb 0
    // (`OpUConvert %u64 %i16` / `%i32`): `sext` with the workaround. The u16
    // source and the `@intCast` of an i16 to u64 stay `zext`.
    const wide = @embedFile("testdata/convert-wide.spv");
    const Case = struct { workarounds: bool, sext16: usize, zext16: usize, sext32: usize, zext32: usize };
    const cases = [_]Case{
        .{ .workarounds = true, .sext16 = 1, .zext16 = 2, .sext32 = 1, .zext32 = 0 },
        .{ .workarounds = false, .sext16 = 0, .zext16 = 3, .sext32 = 0, .zext32 = 1 },
    };
    for (cases) |c| {
        var m = try reader.read(a, wide, null);
        try std.testing.expectEqual(zig_generator_tool, m.generator >> 16);
        var diag: Diagnostics = .{};
        const out = try lower(a, &m, .{ .zig_workarounds = c.workarounds }, &diag);
        try std.testing.expectEqual(c.sext16, std.mem.count(u8, out.ir, "= sext i16 "));
        try std.testing.expectEqual(c.zext16, std.mem.count(u8, out.ir, "= zext i16 "));
        try std.testing.expectEqual(c.sext32, std.mem.count(u8, out.ir, "= sext i32 "));
        try std.testing.expectEqual(c.zext32, std.mem.count(u8, out.ir, "= zext i32 "));
    }
}

test "producer workaround on Zig's own output, both flavours (testdata/convert_*.zig)" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const Case = struct { bytes: []const u8, workarounds: bool, zext16: usize, sext16: usize, zext8: usize, sext8: usize };
    // Vulkan: `@as(i32, u16)` and `@as(i32, u8)` are `OpSConvert %i32 %u16`
    // / `%u8` (Zig's bug, zero-extended by the workaround), `@as(i32, i16)`
    // is `OpSConvert %i32 %i16` (sign-extended either way). OpenCL: both
    // sources are signed (`OpSConvert %u32 %u16` / `%u8`, every type
    // unsigned) and stay sign extensions with the workaround on.
    const vulkan = @embedFile("testdata/convert-vulkan.spv");
    const opencl = @embedFile("testdata/convert-opencl.spv");
    const cases = [_]Case{
        .{ .bytes = vulkan, .workarounds = true, .zext16 = 1, .sext16 = 1, .zext8 = 1, .sext8 = 0 },
        .{ .bytes = vulkan, .workarounds = false, .zext16 = 0, .sext16 = 2, .zext8 = 0, .sext8 = 1 },
        .{ .bytes = opencl, .workarounds = true, .zext16 = 0, .sext16 = 1, .zext8 = 0, .sext8 = 1 },
        .{ .bytes = opencl, .workarounds = false, .zext16 = 0, .sext16 = 1, .zext8 = 0, .sext8 = 1 },
    };
    for (cases) |c| {
        var m = try reader.read(a, c.bytes, null);
        try std.testing.expectEqual(zig_generator_tool, m.generator >> 16);
        var diag: Diagnostics = .{};
        const out = try lower(a, &m, .{ .zig_workarounds = c.workarounds }, &diag);
        try std.testing.expectEqual(c.zext16, std.mem.count(u8, out.ir, "= zext i16 "));
        try std.testing.expectEqual(c.sext16, std.mem.count(u8, out.ir, "= sext i16 "));
        try std.testing.expectEqual(c.zext8, std.mem.count(u8, out.ir, "= zext i8 "));
        try std.testing.expectEqual(c.sext8, std.mem.count(u8, out.ir, "= sext i8 "));
    }
}

test "conversions and bitcasts, i8/i16/i64 widths" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try expectLowered(a, "%u32", "%u8 = OpTypeInt 8 0\n%i16 = OpTypeInt 16 1\n%h = OpTypeFloat 16\n%m1 = OpConstant %i16 -1",
        \\%b = OpUConvert %u8 %x
        \\%w = OpSConvert %i16 %b
        \\%w2 = OpIAdd %i16 %w %m1
        \\%l = OpSConvert %u64 %w2
        \\%f = OpConvertUToF %f32 %l
        \\%hf = OpFConvert %h %f
        \\%back = OpFConvert %f32 %hf
        \\%s = OpConvertFToS %i32 %back
        \\%u = OpBitcast %u32 %s
        \\%fb = OpBitcast %f32 %u
        \\%u2 = OpConvertFToU %u32 %fb
        \\OpStore %pout_x %u2
    , &.{
        "%b = trunc i32 %x to i8",
        "%w = sext i8 %b to i16",
        "%w2 = add i16 %w, -1",
        "%l = sext i16 %w2 to i64",
        "%f = uitofp i64 %l to float",
        "%hf = fptrunc float %f to half",
        "%back = fpext half %hf to float",
        "%s = fptosi float %back to i32",
        "%fb = bitcast i32 %s to float",
        "%u2 = fptoui float %fb to i32",
    });
}

test "bools: i8 in memory, i1 in registers; a struct with a bool member" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try expectLowered(a, "%u32", "%p_fb = OpTypePointer Function %bool\n%S = OpTypeStruct %u32 %bool\n%p_fs = OpTypePointer Function %S\n%true = OpConstantTrue %bool",
        \\%c = OpUGreaterThan %bool %x %c1
        \\%sv = OpCompositeConstruct %S %x %c
        \\%cb = OpCompositeExtract %bool %sv 1
        \\%sv2 = OpCompositeInsert %S %true %sv 1
        \\%n = OpLogicalNot %bool %cb
        \\%e = OpLogicalEqual %bool %n %c
        \\%r = OpSelect %u32 %e %x %c0
        \\OpStore %pout_x %r
    , &.{
        "= zext i1 %c to i8",
        "= insertvalue { i32, i8 } %t",
        "= extractvalue { i32, i8 } %t",
        "= icmp ne i8 %t",
        "%sv2 = insertvalue { i32, i8 } %t",
        ", i8 %t",
        "%n = xor i1 %t",
        ", true",
    });
}

test "function-scope bool variables are i8 allocas with zext stores and compare loads" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const text =
        \\OpCapability Shader
        \\OpMemoryModel Logical GLSL450
        \\OpEntryPoint GLCompute %k "k"
        \\%void = OpTypeVoid
        \\%fn = OpTypeFunction %void
        \\%bool = OpTypeBool
        \\%pb = OpTypePointer Function %bool
        \\%t = OpConstantTrue %bool
        \\%k = OpFunction %void None %fn
        \\%e = OpLabel
        \\%v = OpVariable %pb Function %t
        \\%l = OpLoad %bool %v
        \\%n = OpLogicalNot %bool %l
        \\OpStore %v %n
        \\OpReturn
        \\OpFunctionEnd
    ;
    const r = try lowerTextForTest(a, text, .{});
    try std.testing.expectEqualStrings("", r.message);
    try std.testing.expect(has(r.ir, "%v = alloca i8, align 1"));
    try std.testing.expect(has(r.ir, "store i8 1, ptr %v, align 1"));
    try std.testing.expect(has(r.ir, "= load i8, ptr %v, align 1"));
    try std.testing.expect(has(r.ir, "= icmp ne i8 %t1, 0"));
    try std.testing.expect(has(r.ir, "= zext i1 %n to i8"));
    try std.testing.expect(has(r.ir, "define void @k() {"));
}

test "vectors: construct, extract, insert, shuffle with an undefined lane, dynamic lanes, times-scalar, dot, any/all" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try expectLowered(a, "%v4f32", "%v4f32 = OpTypeVector %f32 4\n%v2f32 = OpTypeVector %f32 2\n%v4b = OpTypeVector %bool 4\n%half_ = OpConstant %f32 0.5",
        \\%e0 = OpCompositeExtract %f32 %x 0
        \\%v2 = OpCompositeConstruct %v2f32 %e0 %e0
        \\%v4 = OpCompositeConstruct %v4f32 %v2 %e0 %e0
        \\%s1 = OpVectorShuffle %v4f32 %v4 %x 7 0 4294967295 2
        \\%s2 = OpVectorShuffle %v4f32 %v2 %x 0 3 5 1
        \\%ins = OpCompositeInsert %v4f32 %e0 %s1 2
        \\%dyn = OpVectorExtractDynamic %f32 %ins %gid_x
        \\%dyn2 = OpVectorInsertDynamic %v4f32 %s2 %dyn %c1
        \\%sc = OpVectorTimesScalar %v4f32 %dyn2 %half_
        \\%d = OpDot %f32 %sc %x
        \\%cmp = OpFOrdLessThan %v4b %sc %x
        \\%any = OpAny %bool %cmp
        \\%all = OpAll %bool %cmp
        \\%both = OpLogicalAnd %bool %any %all
        \\%sel = OpSelect %v4f32 %cmp %sc %x
        \\%sel2 = OpSelect %v4f32 %both %sel %x
        \\%fin = OpCompositeInsert %v4f32 %d %sel2 0
        \\OpStore %pout_x %fin
    , &.{
        "%e0 = extractelement <4 x float> %x, i32 0",
        "insertelement <2 x float> undef, float %e0, i32 0",
        "%s1 = shufflevector <4 x float> %t",
        "<4 x i32> <i32 7, i32 0, i32 undef, i32 2>",
        "%ins = insertelement <4 x float> %s1, float %e0, i32 2",
        "%dyn = extractelement <4 x float> %ins, i32 %gid_x",
        "%dyn2 = insertelement <4 x float> %t",
        "shufflevector <4 x float> %t",
        "<4 x i32> zeroinitializer",
        "%sc = fmul <4 x float> %dyn2, %t",
        "call float @air.dot.v4f32(<4 x float> %sc, <4 x float> %x)",
        "%cmp = fcmp olt <4 x float> %sc, %x",
        "call i1 @air.any.v4i1(<4 x i1> %cmp)",
        "call i1 @air.all.v4i1(<4 x i1> %cmp)",
        "%sel = select <4 x i1> %cmp, <4 x float> %sc, <4 x float> %x",
    });
}

test "control flow: phi loop with a back-edge value defined by an expansion, switch, helper call, unreachable" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const text =
        \\OpCapability Shader
        \\OpCapability Int64
        \\%glsl = OpExtInstImport "GLSL.std.450"
        \\OpMemoryModel Logical GLSL450
        \\OpEntryPoint GLCompute %k "k" %gid
        \\OpExecutionMode %k LocalSize 32 2 1
        \\OpName %helper "helper"
        \\OpDecorate %gid BuiltIn LocalInvocationIndex
        \\OpDecorate %Buf Block
        \\OpMemberDecorate %Buf 0 Offset 0
        \\OpDecorate %rt ArrayStride 4
        \\OpDecorate %out DescriptorSet 0
        \\OpDecorate %out Binding 2
        \\%void = OpTypeVoid
        \\%fn = OpTypeFunction %void
        \\%bool = OpTypeBool
        \\%u32 = OpTypeInt 32 0
        \\%i32 = OpTypeInt 32 1
        \\%u64 = OpTypeInt 64 0
        \\%fn_h = OpTypeFunction %u32 %u32
        \\%c0 = OpConstant %u32 0
        \\%c1 = OpConstant %u32 1
        \\%c10 = OpConstant %u32 10
        \\%big = OpConstant %u64 4294967296
        \\%rt = OpTypeRuntimeArray %u32
        \\%Buf = OpTypeStruct %rt
        \\%pBuf = OpTypePointer StorageBuffer %Buf
        \\%pu = OpTypePointer StorageBuffer %u32
        \\%pin = OpTypePointer Input %u32
        \\%gid = OpVariable %pin Input
        \\%out = OpVariable %pBuf StorageBuffer
        \\%k = OpFunction %void None %fn
        \\%entry = OpLabel
        \\%g = OpLoad %u32 %gid
        \\OpBranch %head
        \\%head = OpLabel
        \\%i = OpPhi %u32 %c0 %entry %next %body
        \\%acc = OpPhi %u32 %c0 %entry %acc2 %body
        \\%cond = OpULessThan %bool %i %c10
        \\OpLoopMerge %exit %body None
        \\OpBranchConditional %cond %body %exit
        \\%body = OpLabel
        \\%acc2 = OpExtInst %u32 %glsl UMax %acc %i
        \\%next = OpIAdd %u32 %i %c1
        \\OpBranch %head
        \\%exit = OpLabel
        \\%wide = OpUConvert %u64 %acc
        \\OpSelectionMerge %join None
        \\OpSwitch %wide %join 1 %case1 4294967296 %case2
        \\%case1 = OpLabel
        \\OpBranch %join
        \\%case2 = OpLabel
        \\OpBranch %join
        \\%join = OpLabel
        \\%v = OpPhi %u32 %acc %exit %c1 %case1 %c10 %case2
        \\%h = OpFunctionCall %u32 %helper %v
        \\%p = OpAccessChain %pu %out %c0 %g
        \\OpStore %p %h
        \\OpReturn
        \\%dead = OpLabel
        \\OpUnreachable
        \\OpFunctionEnd
        \\%helper = OpFunction %u32 None %fn_h
        \\%arg = OpFunctionParameter %u32
        \\%hb = OpLabel
        \\%idx = OpLoad %u32 %gid
        \\%sum = OpIAdd %u32 %arg %idx
        \\OpReturnValue %sum
        \\OpFunctionEnd
    ;
    const r = try lowerTextForTest(a, text, .{});
    try std.testing.expectEqualStrings("", r.message);
    try std.testing.expect(has(r.ir, "define void @k(ptr addrspace(1) %a0, i32 %a1) {"));
    try std.testing.expect(has(r.ir, "%i = phi i32 [ 0, %bentry ], [ %next, %bbody ]") or has(r.ir, "phi i32 [ 0, %b"));
    // The back edge names the UMax call's temporary, defined after the phi.
    try std.testing.expect(has(r.ir, "%acc = phi i32 [ 0, %b"));
    try std.testing.expect(has(r.ir, "call i32 @llvm.umax.i32(i32 %acc, i32 %i)"));
    try std.testing.expect(has(r.ir, "switch i64 %wide, label %b"));
    try std.testing.expect(has(r.ir, " [\n    i64 1, label %b"));
    try std.testing.expect(has(r.ir, "\n    i64 4294967296, label %b"));
    try std.testing.expect(has(r.ir, "%h = call i32 @helper(i32 %v, i32 %a1)"));
    try std.testing.expect(has(r.ir, "define private i32 @helper(i32 %arg, i32 %i"));
    try std.testing.expect(has(r.ir, "unreachable"));
    // A runtime-array Block: [0 x i32], no buffer_size, count 1.
    try std.testing.expect(has(r.ir, "getelementptr { [0 x i32] }, ptr addrspace(1) %a0, i64 0, i32 0, i32 %a1"));
    const b = r.kernels[0].params[0].buffer;
    try std.testing.expectEqual(@as(?u32, null), b.buffer_size);
    try std.testing.expectEqual(1, b.fields.?[0].count);
    try std.testing.expectEqual(2, b.index);
    try std.testing.expectEqual(air.BuiltinKind.thread_index_in_threadgroup, r.kernels[0].params[1].builtin.kind);
    try std.testing.expectEqual([3]u32{ 32, 2, 1 }, r.kernels[0].local_size);
}

test "layout: Offsets that differ from AIR's natural layout give a packed struct, a vec3 member there is [3 x T]; overlap is refused" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const module =
        \\OpCapability Shader
        \\OpMemoryModel Logical GLSL450
        \\OpEntryPoint GLCompute %k "k"
        \\OpDecorate %P Block
        \\OpMemberDecorate %P 0 Offset 0
        \\OpMemberDecorate %P 1 Offset {d}
        \\OpDecorate %pc_var Binding 0
        \\OpDecorate %pc_var DescriptorSet 0
        \\OpDecorate %buf_var Binding 1
        \\OpDecorate %buf_var DescriptorSet 0
        \\OpDecorate %B Block
        \\OpMemberDecorate %B 0 Offset 0
        \\%void = OpTypeVoid
        \\%fn = OpTypeFunction %void
        \\%f32 = OpTypeFloat 32
        \\%v3 = OpTypeVector %f32 3
        \\%P = OpTypeStruct %v3 %f32
        \\%pP = OpTypePointer Uniform %P
        \\%pf = OpTypePointer Uniform %f32
        \\%B = OpTypeStruct %f32
        \\%pB = OpTypePointer StorageBuffer %B
        \\%psf = OpTypePointer StorageBuffer %f32
        \\%u32 = OpTypeInt 32 0
        \\%c0 = OpConstant %u32 0
        \\%c1 = OpConstant %u32 1
        \\%pc_var = OpVariable %pP Uniform
        \\%buf_var = OpVariable %pB StorageBuffer
        \\%k = OpFunction %void None %fn
        \\%e = OpLabel
        \\%p = OpAccessChain %pf %pc_var %c1
        \\%v = OpLoad %f32 %p
        \\%q = OpAccessChain %psf %buf_var %c0
        \\OpStore %q %v
        \\OpReturn
        \\OpFunctionEnd
    ;
    // Offset 20: natural would be 16, so the struct is packed with padding;
    // the vec3 is stored as [3 x float] there (12 bytes, then 8 of padding).
    const r = try lowerTextForTest(a, try a.print(module, .{20}), .{});
    try std.testing.expectEqualStrings("", r.message);
    try std.testing.expect(has(r.ir, "getelementptr <{ [3 x float], [8 x i8], float }>, ptr addrspace(2) %a0, i64 0, i32 2"));
    const pb = r.kernels[0].params[0].buffer;
    try std.testing.expectEqual(manifest.Space.constant, pb.space);
    try std.testing.expectEqual(24, pb.size);
    try std.testing.expectEqual(20, pb.fields.?[1].offset);
    try std.testing.expectEqualStrings("float3", pb.fields.?[0].type_name);
    try std.testing.expectEqual(12, pb.fields.?[0].size);
    // Offset 16 is natural: a literal struct.
    const r16 = try lowerTextForTest(a, try a.print(module, .{16}), .{});
    try std.testing.expect(has(r16.ir, "getelementptr { <3 x float>, float }, ptr addrspace(2) %a0, i64 0, i32 1"));
    // Offset 12 (std430 `vec3` + `float`): the float follows the third
    // component directly, and its load is aligned 4 (its Offset), not 16.
    const r12 = try lowerTextForTest(a, try a.print(module, .{12}), .{});
    try std.testing.expectEqualStrings("", r12.message);
    try std.testing.expect(has(r12.ir, "getelementptr <{ [3 x float], float }>, ptr addrspace(2) %a0, i64 0, i32 1"));
    try std.testing.expect(has(r12.ir, "%v = load float, ptr addrspace(2) %p, align 4"));
    try std.testing.expectEqual(16, r12.kernels[0].params[0].buffer.size);
    // Offset 8 is inside the vector's three components.
    const r8 = try lowerTextForTest(a, try a.print(module, .{8}), .{});
    try std.testing.expect(has(r8.message, "overlaps the previous member"));
}

test "GLSL.std.450: llvm spellings for intrinsics.zig, air names, expansions" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try expectLowered(a, "%f32", "%v3f32 = OpTypeVector %f32 3",
        \\%a1 = OpExtInst %f32 %glsl Sin %x
        \\%a2 = OpExtInst %f32 %glsl Asin %a1
        \\%a3 = OpExtInst %f32 %glsl InverseSqrt %a2
        \\%a4 = OpExtInst %f32 %glsl Atan2 %a3 %x
        \\%a5 = OpExtInst %f32 %glsl Round %a4
        \\%a6 = OpExtInst %f32 %glsl RoundEven %a5
        \\%a7 = OpExtInst %f32 %glsl Fract %a6
        \\%a8 = OpExtInst %f32 %glsl FSign %a7
        \\%a9 = OpExtInst %f32 %glsl FClamp %a8 %x %a1
        \\%a10 = OpExtInst %f32 %glsl FMix %a9 %x %a1
        \\%a11 = OpExtInst %f32 %glsl SmoothStep %a10 %x %a1
        \\%a12 = OpExtInst %f32 %glsl Step %a11 %x
        \\%a13 = OpExtInst %f32 %glsl Fma %a12 %x %a1
        \\%a14 = OpExtInst %f32 %glsl Ldexp %a13 %gid_x
        \\%a15 = OpExtInst %f32 %glsl Pow %a14 %x
        \\%v = OpCompositeConstruct %v3f32 %a15 %x %a1
        \\%n = OpExtInst %v3f32 %glsl Normalize %v
        \\%c = OpExtInst %v3f32 %glsl Cross %n %v
        \\%len = OpExtInst %f32 %glsl Length %c
        \\%dist = OpExtInst %f32 %glsl Distance %c %v
        \\%sum = OpFAdd %f32 %len %dist
        \\OpStore %pout_x %sum
    , &.{
        "call float @llvm.sin.f32(float %x)",
        "call float @air.asin.f32(",
        "call float @air.rsqrt.f32(",
        "call float @air.atan2.f32(",
        "call float @llvm.round.f32(",
        "call float @llvm.rint.f32(",
        "call float @llvm.floor.f32(",
        "fcmp ogt float",
        "0x3FF0000000000000",
        "0xBFF0000000000000",
        "call float @llvm.maxnum.f32(",
        "call float @llvm.minnum.f32(",
        "call float @llvm.fma.f32(",
        "call float @llvm.ldexp.f32.i32(float %t",
        "call float @llvm.pow.f32(",
        "call float @air.dot.v3f32(<3 x float>",
        "call float @llvm.sqrt.f32(",
        "shufflevector <3 x float> %t",
        "<3 x i32> <i32 1, i32 2, i32 0>",
    });
    // Fast math picks Apple's fast functions for the names this file chooses.
    const text = try kernelModule(a, "%f32", "", "%y = OpExtInst %f32 %glsl Asin %x\nOpStore %pout_x %y");
    const r = try lowerTextForTest(a, text, .{ .fast_math = true });
    try std.testing.expect(has(r.ir, "@air.fast_asin.f32("));
}

test "GLSL.std.450 integer functions and bit instructions" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try expectLowered(a, "%i32", "%c4 = OpConstant %i32 4\n%c8 = OpConstant %u32 8\n%RS = OpTypeStruct %i32 %i32",
        \\%b1 = OpExtInst %i32 %glsl SAbs %x
        \\%b2 = OpExtInst %i32 %glsl SSign %b1
        \\%b3 = OpExtInst %i32 %glsl SClamp %b2 %x %c4
        \\%b4 = OpExtInst %i32 %glsl FindILsb %b3
        \\%b5 = OpExtInst %i32 %glsl FindSMsb %b4
        \\%b6 = OpExtInst %i32 %glsl FindUMsb %b5
        \\%b7 = OpBitCount %i32 %b6
        \\%b8 = OpBitReverse %i32 %b7
        \\%b9 = OpBitFieldUExtract %i32 %b8 %c1 %c8
        \\%b10 = OpBitFieldSExtract %i32 %b9 %c4 %c8
        \\%b11 = OpBitFieldInsert %i32 %b10 %x %c4 %c8
        \\%m = OpSMulExtended %RS %b11 %x
        \\%hi = OpCompositeExtract %i32 %m 1
        \\%cy = OpIAddCarry %RS %hi %x
        \\%c = OpCompositeExtract %i32 %cy 1
        \\OpStore %pout_x %c
    , &.{
        "call i32 @llvm.abs.i32(i32 %x, i1 false)",
        "call i32 @llvm.smax.i32(",
        "call i32 @llvm.cttz.i32(",
        "call i32 @llvm.ctlz.i32(",
        "call i32 @llvm.ctpop.i32(",
        "call i32 @llvm.bitreverse.i32(",
        "sext i32 %t",
        "mul i64",
        "lshr i64 %t",
        "icmp ult i32",
        "insertvalue { i32, i32 } undef",
    });
}

test "OpenCL flavour: pointer and value parameters, 64-bit ids, OpenCL.std" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const text =
        \\OpCapability Kernel
        \\OpCapability Addresses
        \\OpCapability Int64
        \\%ocl = OpExtInstImport "OpenCL.std"
        \\OpMemoryModel Physical64 OpenCL
        \\OpEntryPoint Kernel %k "kern" %gid
        \\OpDecorate %gid BuiltIn GlobalInvocationId
        \\OpDecorate %src NonWritable
        \\%void = OpTypeVoid
        \\%f32 = OpTypeFloat 32
        \\%u32 = OpTypeInt 32 0
        \\%u64 = OpTypeInt 64 0
        \\%v3u64 = OpTypeVector %u64 3
        \\%pcw = OpTypePointer CrossWorkgroup %f32
        \\%pin = OpTypePointer Input %v3u64
        \\%fn = OpTypeFunction %void %pcw %pcw %f32
        \\%gid = OpVariable %pin Input
        \\%k = OpFunction %void None %fn
        \\%dst = OpFunctionParameter %pcw
        \\%src = OpFunctionParameter %pcw
        \\%scale = OpFunctionParameter %f32
        \\%e = OpLabel
        \\%g = OpLoad %v3u64 %gid
        \\%i = OpCompositeExtract %u64 %g 0
        \\%ps = OpInBoundsPtrAccessChain %pcw %src %i
        \\%v = OpLoad %f32 %ps Aligned 4
        \\%m = OpExtInst %f32 %ocl fmax %v %scale
        \\%s = OpExtInst %f32 %ocl sign %m
        \\%pd = OpInBoundsPtrAccessChain %pcw %dst %i
        \\OpStore %pd %s Aligned 4
        \\OpReturn
        \\OpFunctionEnd
    ;
    const r = try lowerTextForTest(a, text, .{});
    try std.testing.expectEqualStrings("", r.message);
    try std.testing.expect(has(r.ir, "define void @kern(ptr addrspace(1) %a0, ptr addrspace(1) %a1, ptr addrspace(2) %a2, <3 x i32> %a3) {"));
    try std.testing.expect(has(r.ir, "= load float, ptr addrspace(2) %a2, align 4"));
    try std.testing.expect(has(r.ir, "zext <3 x i32> %a3 to <3 x i64>"));
    try std.testing.expect(has(r.ir, "%ps = getelementptr inbounds float, ptr addrspace(1) %a1, i64 %i"));
    try std.testing.expect(has(r.ir, "call float @llvm.maxnum.f32(float %v, float %t1)"));
    try std.testing.expect(has(r.ir, "fcmp uno float"));
    const k = r.kernels[0];
    try std.testing.expectEqual(manifest.Access.read_write, k.params[0].buffer.access);
    try std.testing.expectEqual(manifest.Access.read, k.params[1].buffer.access);
    try std.testing.expect(k.params[2].buffer.by_value);
    try std.testing.expectEqual(manifest.Space.constant, k.params[2].buffer.space);
    try std.testing.expectEqual(2, k.params[2].buffer.index);
    try std.testing.expectEqualStrings("float", k.params[2].buffer.type_name);
}

test "push constants, Private and Workgroup variables, unbound buffers" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const text =
        \\OpCapability Shader
        \\OpMemoryModel Logical GLSL450
        \\OpEntryPoint GLCompute %k "k"
        \\OpName %tile "tile"
        \\OpDecorate %PC Block
        \\OpMemberDecorate %PC 0 Offset 0
        \\OpDecorate %arr ArrayStride 4
        \\%void = OpTypeVoid
        \\%fn = OpTypeFunction %void
        \\%u32 = OpTypeInt 32 0
        \\%c0 = OpConstant %u32 0
        \\%c7 = OpConstant %u32 7
        \\%c16 = OpConstant %u32 16
        \\%PC = OpTypeStruct %u32
        \\%ppc = OpTypePointer PushConstant %PC
        \\%ppcu = OpTypePointer PushConstant %u32
        \\%arr = OpTypeArray %u32 %c16
        \\%pwg = OpTypePointer Workgroup %arr
        \\%pwgu = OpTypePointer Workgroup %u32
        \\%ppriv = OpTypePointer Private %u32
        \\%psb = OpTypePointer StorageBuffer %arr
        \\%psbu = OpTypePointer StorageBuffer %u32
        \\%pc = OpVariable %ppc PushConstant
        \\%tile = OpVariable %pwg Workgroup
        \\%priv = OpVariable %ppriv Private %c7
        \\%plain = OpVariable %psb StorageBuffer
        \\%k = OpFunction %void None %fn
        \\%e = OpLabel
        \\%pp = OpAccessChain %ppcu %pc %c0
        \\%n = OpLoad %u32 %pp
        \\%pt = OpAccessChain %pwgu %tile %n
        \\%pv = OpLoad %u32 %priv
        \\OpStore %pt %pv
        \\%t = OpLoad %u32 %pt
        \\%pd = OpAccessChain %psbu %plain %n
        \\OpStore %pd %t
        \\OpReturn
        \\OpFunctionEnd
    ;
    const r = try lowerTextForTest(a, text, .{ .push_constant_index = 12 });
    try std.testing.expectEqualStrings("", r.message);
    try std.testing.expect(has(r.ir, "@tile = internal addrspace(3) global [16 x i32] undef, align 4"));
    try std.testing.expect(has(r.ir, "define void @k(ptr addrspace(1) %a0, ptr addrspace(2) %a1) {"));
    try std.testing.expect(has(r.ir, "%i"));
    try std.testing.expect(has(r.ir, " = alloca i32, align 4"));
    try std.testing.expect(has(r.ir, "store i32 7, ptr %i"));
    try std.testing.expect(has(r.ir, "getelementptr [16 x i32], ptr addrspace(3) @tile, i64 0, i32 %n"));
    const k = r.kernels[0];
    try std.testing.expectEqual(64, k.threadgroup_bytes);
    try std.testing.expectEqual(manifest.Source.unbound, k.params[0].buffer.source);
    try std.testing.expectEqual(0, k.params[0].buffer.index);
    try std.testing.expectEqual(manifest.Source.push_constant, k.params[1].buffer.source);
    try std.testing.expectEqual(12, k.params[1].buffer.index);
}

test "refusals name the instruction: stage 5, ArrayLength, descriptor sets, double" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try expectRefused(a, "%u32", "", "%y = OpExtInst %u32 %glsl PackHalf2x16 %x", "GLSL.std.450 PackHalf2x16 is not supported");
    try expectRefused(a, "%u32", "%f64 = OpTypeFloat 64", "%d = OpConvertUToF %f64 %x", "Metal has no double");
    // A bfloat16 is 16 bits like a half; it must be refused, not lowered as one.
    try expectRefused(a, "%u32", "%bf = OpTypeFloat 16 BFloat16KHR\n%one = OpConstant %bf 0x3f80", "%s = OpFAdd %bf %one %one", "FPEncoding BFloat16KHR");
    try expectRefused(a, "%u32", "%bf = OpTypeFloat 16 BFloat16KHR\n%bfv = OpTypeVector %bf 2", "%v = OpConvertUToF %bf %x", "has FPEncoding BFloat16KHR (16-bit, not IEEE 754)");
    try expectRefused(a, "%u32", "%f8 = OpTypeFloat 8 Float8E4M3EXT", "%v = OpConvertUToF %f8 %x", "FPEncoding Float8E4M3EXT");
    try expectRefused(a, "%u32", "", "%n = OpArrayLength %u32 %out 0", "no buffer lengths");
    try expectRefused(a, "%u32", "%c3 = OpConstant %u32 3", "OpKill\n%after = OpLabel", "fragment-only");

    const vertex =
        \\OpCapability Shader
        \\OpMemoryModel Logical GLSL450
        \\OpEntryPoint Vertex %v "v"
        \\%void = OpTypeVoid
        \\%fn = OpTypeFunction %void
        \\%v = OpFunction %void None %fn
        \\%e = OpLabel
        \\OpReturn
        \\OpFunctionEnd
    ;
    const rv = try lowerTextForTest(a, vertex, .{});
    try std.testing.expect(has(rv.message, "entry point 'v': execution model vertex is not a compute model (stage 5"));

    const set1 = try kernelModule(a, "%u32", "", "OpStore %pout_x %x");
    const patched = try std.mem.replaceOwned(u8, a, set1, "OpDecorate %out DescriptorSet 0", "OpDecorate %out DescriptorSet 1");
    const rs = try lowerTextForTest(a, patched, .{});
    try std.testing.expect(has(rs.message, "DescriptorSet 1"));
}

test "the core instructions docs/spirv.md lists as not supported are refused by name" {
    // docs/spirv.md's "not supported" table and the Instructions section's
    // refusal list name these; the review found the docs claiming every
    // arithmetic, conversion and composite instruction while these fell to
    // lowerInst's `{t} is not supported`. Keep the two in step.
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const types =
        \\%v4f32 = OpTypeVector %f32 4
        \\%v4u32 = OpTypeVector %u32 4
        \\%v4bool = OpTypeVector %bool 4
        \\%m4 = OpTypeMatrix %v4f32 4
        \\%S1 = OpTypeStruct %f32
        \\%S2 = OpTypeStruct %f32
        \\%f1 = OpConstant %f32 1
        \\%c3 = OpConstant %u32 3
        \\%c16 = OpConstant %u32 16
        \\%true = OpConstantTrue %bool
    ;
    const m = "%m = OpCompositeConstruct %m4 %x %x %x %x\n";
    const xs = "%xs = OpCompositeExtract %f32 %x 0\n";
    const cases = [_]struct { body: []const u8, op: []const u8 }{
        .{ .body = m ++ "%r = OpMatrixTimesVector %v4f32 %m %x", .op = "OpMatrixTimesVector" },
        .{ .body = m ++ "%r = OpVectorTimesMatrix %v4f32 %x %m", .op = "OpVectorTimesMatrix" },
        .{ .body = m ++ "%r = OpMatrixTimesScalar %m4 %m %f1", .op = "OpMatrixTimesScalar" },
        .{ .body = m ++ "%r = OpMatrixTimesMatrix %m4 %m %m", .op = "OpMatrixTimesMatrix" },
        .{ .body = "%r = OpOuterProduct %m4 %x %x", .op = "OpOuterProduct" },
        .{ .body = m ++ "%r = OpTranspose %m4 %m", .op = "OpTranspose" },
        .{ .body = xs ++ "%s = OpCompositeConstruct %S1 %xs\n%r = OpCopyLogical %S2 %s", .op = "OpCopyLogical" },
        .{ .body = "%r = OpQuantizeToF16 %v4f32 %x", .op = "OpQuantizeToF16" },
        .{ .body = xs ++ "%i = OpBitcast %i32 %xs\n%r = OpSatConvertSToU %u32 %i", .op = "OpSatConvertSToU" },
        .{ .body = xs ++ "%u = OpBitcast %u32 %xs\n%r = OpSatConvertUToS %i32 %u", .op = "OpSatConvertUToS" },
        .{ .body = "%r = OpIsNormal %v4bool %x", .op = "OpIsNormal" },
        .{ .body = "%r = OpSignBitSet %v4bool %x", .op = "OpSignBitSet" },
        .{ .body = "%xu = OpBitcast %v4u32 %x\n%r = OpUDot %u32 %xu %xu", .op = "OpUDot" },
        .{ .body = "%xu = OpBitcast %v4u32 %x\n%r = OpSDot %u32 %xu %xu", .op = "OpSDot" },
        .{ .body = "%xu = OpBitcast %v4u32 %x\n%r = OpSUDot %u32 %xu %xu", .op = "OpSUDot" },
        .{ .body = "%xu = OpBitcast %v4u32 %x\n%r = OpUDotAccSat %u32 %xu %xu %c3", .op = "OpUDotAccSat" },
        .{ .body = "%r = OpSizeOf %u32 %pin_x", .op = "OpSizeOf" },
        .{ .body = "%r = OpGroupIAdd %u32 %c3 Reduce %c3", .op = "OpGroupIAdd" },
        .{ .body = "%r = OpSubgroupBallotKHR %v4u32 %true", .op = "OpSubgroupBallotKHR" },
        .{ .body = "%r = OpPtrDiff %u64 %pin_x %pin_x", .op = "OpPtrDiff" },
        .{ .body = "OpCopyMemorySized %pout_x %pin_x %c16", .op = "OpCopyMemorySized" },
        .{ .body = "%r = OpGroupNonUniformElect %bool %c3", .op = "OpGroupNonUniformElect" },
        .{ .body = "%r = OpGroupNonUniformFAdd %v4f32 %c3 Reduce %x", .op = "OpGroupNonUniformFAdd" },
    };
    for (cases) |c| {
        try expectRefused(a, "%v4f32", types, c.body, try a.print("{s} is not supported", .{c.op}));
    }
}

test "std140 strides AIR cannot express and the atomic flag instructions are refused as docs/spirv.md lists" {
    // docs/spirv.md's "What is supported" table claimed std140 and every
    // OpAtomic* on 32-bit integers; the docs review found std140's
    // stride-16 scalar and vec2 arrays and OpAtomicFlagTestAndSet / Clear
    // refused. The docs now list them as not supported; keep the two in step.
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const decorations =
        \\OpDecorate %U Block
        \\OpMemberDecorate %U 0 Offset 0
        \\OpMemberDecorate %U 0 ColMajor
        \\OpMemberDecorate %U 0 MatrixStride {[mstride]d}
        \\OpDecorate %ubo DescriptorSet 0
        \\OpDecorate %ubo Binding 2
        \\OpDecorate %ua ArrayStride {[astride]d}
        \\OpDecorate %U2 Block
        \\OpMemberDecorate %U2 0 Offset 0
        \\OpDecorate %ubo2 DescriptorSet 0
        \\OpDecorate %ubo2 Binding 3
    ;
    const types =
        \\%v2f32 = OpTypeVector %f32 2
        \\%v3f32 = OpTypeVector %f32 3
        \\%v4f32 = OpTypeVector %f32 4
        \\%m2 = OpTypeMatrix %v2f32 2
        \\%c4 = OpConstant %u32 4
        \\%U = OpTypeStruct %m2
        \\%p_U = OpTypePointer Uniform %U
        \\%ubo = OpVariable %p_U Uniform
        \\%ua = OpTypeArray {[elem]s} %c4
        \\%U2 = OpTypeStruct %ua
        \\%p_U2 = OpTypePointer Uniform %U2
        \\%p_ue = OpTypePointer Uniform {[elem]s}
        \\%p_col = OpTypePointer Uniform %v2f32
        \\%ubo2 = OpVariable %p_U2 Uniform
    ;
    const body =
        \\%pe = OpAccessChain %p_ue %ubo2 %c0 %c1
        \\%e = OpLoad {[elem]s} %pe
        \\%pm = OpAccessChain %p_col %ubo %c0 %c1
        \\%col = OpLoad %v2f32 %pm
        \\OpStore %pout_x %x
    ;
    const cases = [_]struct { elem: []const u8, astride: u32, mstride: u32, why: []const u8 }{
        // std140 `float w[4]`, `int w[4]`, `vec2 w[4]`: the stride rounds up to 16.
        .{ .elem = "%f32", .astride = 16, .mstride = 8, .why = "has ArrayStride 16 but its element occupies 4 bytes in AIR" },
        .{ .elem = "%i32", .astride = 16, .mstride = 8, .why = "has ArrayStride 16 but its element occupies 4 bytes in AIR" },
        .{ .elem = "%v2f32", .astride = 16, .mstride = 8, .why = "has ArrayStride 16 but its element occupies 8 bytes in AIR" },
        // std140 `mat2`: each vec2 column at a 16-byte MatrixStride.
        .{ .elem = "%v4f32", .astride = 16, .mstride = 16, .why = "MatrixStride 16 differs from the column's AIR size 8" },
        // What std140 and AIR agree on: vec3 and vec4 arrays at 16, std430 / scalar `float[]` at 4.
        .{ .elem = "%v4f32", .astride = 16, .mstride = 8, .why = "" },
        .{ .elem = "%v3f32", .astride = 16, .mstride = 8, .why = "" },
        .{ .elem = "%f32", .astride = 4, .mstride = 8, .why = "" },
    };
    for (cases) |c| {
        const strides = .{ .astride = c.astride, .mstride = c.mstride };
        const elem = .{ .elem = c.elem };
        const text = try kernelModuleWith(a, "%u32", try a.print(decorations, strides), try a.print(types, elem), try a.print(body, elem));
        const r = try lowerTextForTest(a, text, .{});
        const ok = if (c.why.len == 0) r.message.len == 0 else has(r.message, c.why);
        if (!ok) {
            std.debug.print("{s} stride {d} / {d}: expected `{s}`, got `{s}`\n", .{ c.elem, c.astride, c.mstride, c.why, r.message });
            return error.TestUnexpectedResult;
        }
    }
    try expectRefused(a, "%u32", "%c2 = OpConstant %u32 2", "%f = OpAtomicFlagTestAndSet %bool %pout_x %c2 %c0", "OpAtomicFlagTestAndSet is not supported");
    try expectRefused(a, "%u32", "%c2 = OpConstant %u32 2", "OpAtomicFlagClear %pout_x %c2 %c0", "OpAtomicFlagClear is not supported");
}

/// The constants the stage-4 tests use: scopes (Device 1, Workgroup 2,
/// Subgroup 3) and semantics (Acquire|Uniform 66, Release|Uniform 68,
/// AcquireRelease|Uniform 72, SeqCst|Uniform 80, AcquireRelease|Workgroup
/// 264, Release 4, Acquire 2).
const stage4_consts =
    \\%c2 = OpConstant %u32 2
    \\%c3 = OpConstant %u32 3
    \\%c4 = OpConstant %u32 4
    \\%c66 = OpConstant %u32 66
    \\%c68 = OpConstant %u32 68
    \\%c72 = OpConstant %u32 72
    \\%c80 = OpConstant %u32 80
    \\%c264 = OpConstant %u32 264
    \\%u16 = OpTypeInt 16 0
    \\%arr64 = OpTypeArray %u32 %c64
    \\%p_wg_arr = OpTypePointer Workgroup %arr64
    \\%p_wg_u32 = OpTypePointer Workgroup %u32
    \\%p_fn_u32 = OpTypePointer Function %u32
    \\%p_sb_u16 = OpTypePointer StorageBuffer %u16
    \\%tile = OpVariable %p_wg_arr Workgroup
;

test "stage 4: barriers map onto air.wg.barrier / air.simdgroup.barrier with the semantics' memory flags" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    // GLSL barrier(): Workgroup, Workgroup, AcquireRelease|WorkgroupMemory
    // -> mem_threadgroup at threadgroup scope (Apple's `i32 2, i32 1`).
    try expectLowered(a, "%u32", stage4_consts, "OpControlBarrier %c2 %c2 %c264", &.{ "call void @air.wg.barrier(i32 2, i32 1)", "declare void @air.wg.barrier(i32, i32)" });
    // Subgroup execution scope -> the SIMD-group barrier (scope 4); with a
    // Subgroup (or Invocation) memory scope nothing more.
    for ([_][]const u8{ "OpControlBarrier %c3 %c3 %c264", "OpControlBarrier %c3 %c4 %c264", "OpControlBarrier %c3 %c2 %c4" }) |body| {
        const r = try lowerTextForTest(a, try kernelModule(a, "%u32", stage4_consts, body), .{});
        try std.testing.expect(r.message.len == 0);
        try std.testing.expect(has(r.ir, if (std.mem.endsWith(u8, body, "%c4")) "call void @air.simdgroup.barrier(i32 0, i32 4)" else "call void @air.simdgroup.barrier(i32 2, i32 4)"));
        try std.testing.expect(!has(r.ir, "air.atomic.fence"));
    }
    // A Subgroup barrier with a Workgroup or Device memory scope orders the
    // memory at that scope too: the SIMD-group barrier alone orders it only
    // within the SIMD-group, so a fence at the memory scope comes first
    // (Apple: simdgroup_barrier(mem_threadgroup) is `(i32 2, i32 4)`,
    // atomic_thread_fence(mem_threadgroup, seq_cst, thread_scope_threadgroup)
    // is `air.atomic.fence(i32 2, i32 5, i32 1)`).
    try expectLowered(a, "%u32", stage4_consts, "OpControlBarrier %c3 %c2 %c264", &.{"call void @air.atomic.fence(i32 2, i32 5, i32 1)\n  call void @air.simdgroup.barrier(i32 2, i32 4)"});
    {
        // Device memory scope, UniformMemory: one device fence, not two.
        const r = try lowerTextForTest(a, try kernelModule(a, "%u32", stage4_consts, "OpControlBarrier %c3 %c1 %c72"), .{});
        try std.testing.expect(has(r.ir, "call void @air.atomic.fence(i32 1, i32 5, i32 2)\n  call void @air.simdgroup.barrier(i32 1, i32 4)"));
        try std.testing.expectEqual(1, std.mem.count(u8, r.ir, "call void @air.atomic.fence"));
    }
    // No storage-class bit: mem_none (MSL threadgroup_barrier(mem_none)).
    try expectLowered(a, "%u32", stage4_consts, "OpControlBarrier %c2 %c2 %c4", &.{"call void @air.wg.barrier(i32 0, i32 1)"});
    // HLSL DeviceMemoryBarrierWithGroupSync: memory scope Device with
    // UniformMemory: a device-scope fence, then the barrier with mem_device.
    try expectLowered(a, "%u32", stage4_consts, "OpControlBarrier %c2 %c1 %c72", &.{"call void @air.atomic.fence(i32 1, i32 5, i32 2)\n  call void @air.wg.barrier(i32 1, i32 1)"});
    // Workgroup memory scope with UniformMemory: mem_device, no fence.
    {
        const r = try lowerTextForTest(a, try kernelModule(a, "%u32", stage4_consts, "OpControlBarrier %c2 %c2 %c72"), .{});
        try std.testing.expect(has(r.ir, "call void @air.wg.barrier(i32 1, i32 1)"));
        try std.testing.expect(!has(r.ir, "air.atomic.fence"));
    }
    // OpMemoryBarrier: air.atomic.fence(flags, seq_cst, scope).
    try expectLowered(a, "%u32", stage4_consts, "OpMemoryBarrier %c1 %c72", &.{"call void @air.atomic.fence(i32 1, i32 5, i32 2)"});
    try expectLowered(a, "%u32", stage4_consts, "OpMemoryBarrier %c2 %c264", &.{"call void @air.atomic.fence(i32 2, i32 5, i32 1)"});
    try expectLowered(a, "%u32", stage4_consts, "OpMemoryBarrier %c3 %c264", &.{"call void @air.atomic.fence(i32 2, i32 5, i32 4)"});
    // An ordering without memory (or memory without an ordering) orders
    // nothing: no fence.
    {
        const r = try lowerTextForTest(a, try kernelModule(a, "%u32", stage4_consts, "OpMemoryBarrier %c2 %c4"), .{});
        try std.testing.expect(r.message.len == 0 and !has(r.ir, "air.atomic.fence"));
    }
    // glslang's memoryBarrier() / groupMemoryBarrier(): AcquireRelease |
    // Uniform | Workgroup | AtomicCounter | Image (0xD48). The atomic-counter
    // bit orders nothing a kernel has: device, threadgroup and texture flags.
    {
        const consts = stage4_consts ++ "\n%c3400 = OpConstant %u32 3400";
        try expectLowered(a, "%u32", consts, "OpMemoryBarrier %c1 %c3400", &.{"call void @air.atomic.fence(i32 7, i32 5, i32 2)"});
        try expectLowered(a, "%u32", consts, "OpMemoryBarrier %c2 %c3400", &.{"call void @air.atomic.fence(i32 7, i32 5, i32 1)"});
        // AtomicCounterMemory alone (with an ordering) orders nothing.
        const r = try lowerTextForTest(a, try kernelModule(a, "%u32", stage4_consts ++ "\n%c1032 = OpConstant %u32 1032", "OpMemoryBarrier %c1 %c1032"), .{});
        try std.testing.expect(r.message.len == 0 and !has(r.ir, "air.atomic.fence"));
    }
    // Refusals: no barrier across threadgroups, graphics memory, two
    // ordering bits.
    try expectRefused(a, "%u32", stage4_consts, "OpControlBarrier %c1 %c1 %c72", "`OpControlBarrier %c1 %c1 %c72`: execution scope 1: Metal synchronises only a threadgroup");
    try expectRefused(a, "%u32", stage4_consts ++ "\n%c4104 = OpConstant %u32 4104", "OpControlBarrier %c2 %c2 %c4104", "memory semantics 0x1008: OutputMemory is a graphics interface");
    try expectRefused(a, "%u32", stage4_consts ++ "\n%c6 = OpConstant %u32 6", "OpMemoryBarrier %c2 %c6", "more than one ordering bit");
    try expectRefused(a, "%u32", stage4_consts ++ "\n%cx = OpConstant %u32 1048576", "OpMemoryBarrier %c2 %cx", "bits this lowering does not know");
    // A non-constant scope.
    try expectRefused(a, "%u32", stage4_consts, "OpControlBarrier %x %c2 %c264", "is not an integer constant");
}

test "stage 4: atomics on device and threadgroup pointers are native atomicrmw / cmpxchg / load and store atomic with the semantics' ordering" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try expectLowered(a, "%u32", stage4_consts,
        \\%r0 = OpAtomicIAdd %u32 %pout_x %c1 %c0 %x
        \\%r1 = OpAtomicISub %u32 %pout_x %c1 %c66 %x
        \\%r2 = OpAtomicSMin %u32 %pout_x %c1 %c68 %x
        \\%r3 = OpAtomicSMax %u32 %pout_x %c1 %c72 %x
        \\%r4 = OpAtomicUMin %u32 %pout_x %c1 %c80 %x
        \\%r5 = OpAtomicUMax %u32 %pout_x %c1 %c0 %x
        \\%r6 = OpAtomicAnd %u32 %pout_x %c1 %c0 %x
        \\%r7 = OpAtomicOr %u32 %pout_x %c1 %c0 %x
        \\%r8 = OpAtomicXor %u32 %pout_x %c1 %c0 %x
        \\%r9 = OpAtomicExchange %u32 %pout_x %c1 %c0 %x
        \\%r10 = OpAtomicIIncrement %u32 %pout_x %c1 %c0
        \\%r11 = OpAtomicIDecrement %u32 %pout_x %c1 %c0
        \\%r12 = OpAtomicCompareExchange %u32 %pout_x %c1 %c72 %c66 %x %c64
        \\%r13 = OpAtomicCompareExchangeWeak %u32 %pout_x %c1 %c0 %c0 %x %c64
        \\%r14 = OpAtomicLoad %u32 %pout_x %c1 %c66
        \\%r15 = OpAtomicLoad %u32 %pout_x %c1 %c72
        \\OpAtomicStore %pout_x %c1 %c68 %r14
        \\OpAtomicStore %pout_x %c1 %c72 %r15
    , &.{
        "%r0 = atomicrmw add ptr addrspace(1) %pout_x, i32 %x monotonic, align 4",
        "%r1 = atomicrmw sub ptr addrspace(1) %pout_x, i32 %x acquire, align 4",
        "%r2 = atomicrmw min ptr addrspace(1) %pout_x, i32 %x release, align 4",
        "%r3 = atomicrmw max ptr addrspace(1) %pout_x, i32 %x acq_rel, align 4",
        "%r4 = atomicrmw umin ptr addrspace(1) %pout_x, i32 %x seq_cst, align 4",
        "%r5 = atomicrmw umax ptr addrspace(1) %pout_x, i32 %x monotonic, align 4",
        "%r6 = atomicrmw and ",
        "%r7 = atomicrmw or ",
        "%r8 = atomicrmw xor ",
        "%r9 = atomicrmw xchg ptr addrspace(1) %pout_x, i32 %x monotonic, align 4",
        "%r10 = atomicrmw add ptr addrspace(1) %pout_x, i32 1 monotonic, align 4",
        "%r11 = atomicrmw sub ptr addrspace(1) %pout_x, i32 1 monotonic, align 4",
        // Value then Comparator in SPIR-V; comparator then new value in LLVM.
        "= cmpxchg ptr addrspace(1) %pout_x, i32 64, i32 %x acq_rel acquire, align 4\n  %r12 = extractvalue { i32, i1 } %t",
        "= cmpxchg ptr addrspace(1) %pout_x, i32 64, i32 %x monotonic monotonic, align 4\n  %r13 = extractvalue",
        "%r14 = load atomic i32, ptr addrspace(1) %pout_x acquire, align 4",
        // AcquireRelease narrowed to the half that applies.
        "%r15 = load atomic i32, ptr addrspace(1) %pout_x acquire, align 4",
        "store atomic i32 %r14, ptr addrspace(1) %pout_x release, align 4",
        "store atomic i32 %r15, ptr addrspace(1) %pout_x release, align 4",
    });
    // Threadgroup memory: a Workgroup variable, element and whole.
    try expectLowered(a, "%u32", stage4_consts ++ "\n%counter = OpVariable %p_wg_u32 Workgroup",
        \\%pt = OpAccessChain %p_wg_u32 %tile %gid_x
        \\%w0 = OpAtomicIAdd %u32 %pt %c2 %c264 %x
        \\%w1 = OpAtomicUMax %u32 %counter %c2 %c0 %w0
        \\OpAtomicStore %counter %c2 %c0 %w1
    , &.{
        "@tile = internal addrspace(3) global [64 x i32] undef",
        "%w0 = atomicrmw add ptr addrspace(3) %pt, i32 %x acq_rel, align 4",
        "%w1 = atomicrmw umax ptr addrspace(3) @counter, i32 %w0 monotonic, align 4",
        "store atomic i32 %w1, ptr addrspace(3) @counter monotonic, align 4",
    });
    // 64 bits: load, store, and min/max on device memory with the result
    // unused (all Metal has); float add, load, store, exchange.
    try expectLowered(a, "%u64", stage4_consts,
        \\%r = OpAtomicUMax %u64 %pout_x %c1 %c0 %x
        \\%s = OpAtomicSMin %u64 %pout_x %c1 %c0 %x
        \\%l = OpAtomicLoad %u64 %pout_x %c1 %c0
        \\OpAtomicStore %pout_x %c1 %c0 %l
    , &.{ "%r = atomicrmw umax ptr addrspace(1) %pout_x, i64 %x monotonic, align 8", "%s = atomicrmw min ptr addrspace(1) %pout_x, i64 %x monotonic, align 8", "%l = load atomic i64, ptr addrspace(1) %pout_x monotonic, align 8", "store atomic i64 %l, ptr addrspace(1) %pout_x monotonic, align 8" });
    try expectRefused(a, "%u64", stage4_consts, "%r = OpAtomicUMax %u64 %pout_x %c1 %c0 %x\nOpStore %pout_x %r", "the result of a 64-bit atomic min/max is used");
    try expectRefused(a, "%u64", stage4_consts, "%r = OpAtomicIAdd %u64 %pout_x %c1 %c0 %x", "64-bit OpAtomicIAdd: Metal has 64-bit atomic load, store, min and max only");
    try expectRefused(a, "%u64", stage4_consts, "%r = OpAtomicCompareExchange %u64 %pout_x %c1 %c0 %c0 %x %x", "64-bit OpAtomicCompareExchange");
    try expectRefused(a, "%u64", stage4_consts ++ "\n%p_wg_u64 = OpTypePointer Workgroup %u64\n%w64 = OpVariable %p_wg_u64 Workgroup", "%r = OpAtomicUMax %u64 %w64 %c2 %c0 %x", "64-bit atomic min/max on threadgroup memory");
    try expectLowered(a, "%f32", stage4_consts,
        \\%r = OpAtomicFAddEXT %f32 %pout_x %c1 %c0 %x
        \\%e = OpAtomicExchange %f32 %pout_x %c1 %c0 %r
    , &.{ "%r = atomicrmw fadd ptr addrspace(1) %pout_x, float %x monotonic, align 4", "%e = atomicrmw xchg ptr addrspace(1) %pout_x, float %r" });
    try expectRefused(a, "%f32", stage4_consts, "%m = OpAtomicFMaxEXT %f32 %pout_x %c1 %c0 %x", "does not know `atomicrmw fmin/fmax`");
    // Threadgroup floats: load, store and exchange are lowered; Metal
    // miscompiles `atomicrmw fadd ptr addrspace(3)` (corpus kernel
    // atomic_fadd_tg), so the add is refused unless the caller asks for the
    // miscompile (the corpus driver's finding check).
    {
        const tg_f32 = stage4_consts ++ "\n%p_wg_f32 = OpTypePointer Workgroup %f32\n%ftg = OpVariable %p_wg_f32 Workgroup";
        try expectLowered(a, "%f32", tg_f32,
            \\OpAtomicStore %ftg %c2 %c0 %x
            \\%e = OpAtomicExchange %f32 %ftg %c2 %c0 %x
            \\%l = OpAtomicLoad %f32 %ftg %c2 %c0
            \\OpStore %pout_x %l
        , &.{ "store atomic float %x, ptr addrspace(3) @ftg monotonic, align 4", "%e = atomicrmw xchg ptr addrspace(3) @ftg, float %x monotonic, align 4", "%l = load atomic float, ptr addrspace(3) @ftg monotonic, align 4" });
        const fadd = "%r = OpAtomicFAddEXT %f32 %ftg %c2 %c0 %x\nOpStore %pout_x %r";
        try expectRefused(a, "%f32", tg_f32, fadd, "OpAtomicFAddEXT on threadgroup memory: Metal builds `atomicrmw fadd ptr addrspace(3)` but computes a wrong sum");
        const r = try lowerTextForTest(a, try kernelModule(a, "%f32", tg_f32, fadd), .{ .allow_metal_miscompiles = true });
        try std.testing.expect(r.message.len == 0);
        try std.testing.expect(has(r.ir, "%r = atomicrmw fadd ptr addrspace(3) @ftg, float %x monotonic, align 4"));
    }
    // Refusals.
    try expectRefused(a, "%u32", stage4_consts ++ "\n%fv_ty = OpTypePointer Function %u32",
        \\%fv = OpVariable %p_fn_u32 Function
        \\%r = OpAtomicIAdd %u32 %fv %c1 %c0 %x
    , "Metal has atomics on device and threadgroup memory only");
    try expectRefused(a, "%u32", stage4_consts, "%r = OpAtomicLoad %u32 %pout_x %c1 %c68", "an atomic load cannot have Release semantics");
    try expectRefused(a, "%u32", stage4_consts, "OpAtomicStore %pout_x %c1 %c66 %x", "an atomic store cannot have Acquire semantics");
    try expectRefused(a, "%u32", stage4_consts, "%r = OpAtomicIAdd %u32 %pout_x %c1 %x %x", "is not an integer constant");
    try expectRefused(a, "%f32", stage4_consts, "%r = OpAtomicIAdd %f32 %pout_x %c1 %c0 %x", "OpAtomicIAdd on a float");
    try expectRefused(a, "%u32", stage4_consts, "%r = OpAtomicFAddEXT %u32 %pout_x %c1 %c0 %x", "OpAtomicFAddEXT on an integer");
    try expectRefused(a, "%u32", stage4_consts ++ "\n%c5_16 = OpConstant %u16 5",
        \\%p16 = OpBitcast %p_sb_u16 %pout_x
        \\%r = OpAtomicIAdd %u16 %p16 %c1 %c0 %c5_16
    , "16-bit atomics");
}

test "stage 4: the uniformity analysis accepts a barrier where every thread arrives and refuses one behind a per-thread branch" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    // In the join block after a per-thread branch: accepted.
    const join =
        \\%lt = OpULessThan %bool %gid_x %c64
        \\OpSelectionMerge %m None
        \\OpBranchConditional %lt %then %m
        \\%then = OpLabel
        \\OpStore %pout_x %x
        \\OpBranch %m
        \\%m = OpLabel
        \\OpControlBarrier %c2 %c2 %c264
    ;
    try expectLowered(a, "%u32", stage4_consts, join, &.{"call void @air.wg.barrier(i32 2, i32 1)"});
    // Inside the per-thread branch: the assembler refuses it.
    const inside =
        \\%lt = OpULessThan %bool %gid_x %c64
        \\OpSelectionMerge %m None
        \\OpBranchConditional %lt %then %m
        \\%then = OpLabel
        \\OpControlBarrier %c2 %c2 %c264
        \\OpBranch %m
        \\%m = OpLabel
    ;
    try std.testing.expectError(error.Unsupported, lowerTextForTest(a, try kernelModule(a, "%u32", stage4_consts, inside), .{}));
}

test "stage 4: glslang's tree reduction (a barrier in a loop counted in a Function variable) passes the uniformity analysis" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const text = @embedFile("corpus/hand-glslang-stage4.spvasm");
    const r = try lowerTextForTest(a, text, .{});
    try std.testing.expectEqualStrings("", r.message);
    try expectHas(r.ir, "call void @air.wg.barrier(i32 2, i32 1)");
    // The atomics go through Uniform pointers into BufferBlock SSBOs:
    // device memory.
    try expectHas(r.ir, "atomicrmw add ptr addrspace(1) ");
    try expectHas(r.ir, "cmpxchg ptr addrspace(1) ");
    try std.testing.expectEqual(256, r.kernels[0].threadgroup_bytes);
    // The same loop bounded by the per-thread `lid` instead of
    // gl_WorkGroupSize.x / 2: the threads leave it at different iterations
    // and the barrier inside is refused by the assembler.
    const divergent = try std.mem.replaceOwned(u8, a, text, "%37 = OpUDiv %uint %36 %uint_2", "%37 = OpUDiv %uint %15 %uint_2");
    try std.testing.expect(!std.mem.eql(u8, divergent, text));
    try std.testing.expectError(error.Unsupported, lowerTextForTest(a, divergent, .{ .only_entry = "reduce" }));
}

test "the Zig-shape testdata binaries lower and assemble" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    inline for (.{ "k-spirv64-vulkan.spv", "k-spirv64-opencl.spv", "desc.spv", "cf.spv", "narrow.spv", "convert-vulkan.spv", "convert-opencl.spv" }) |name| {
        const bytes = @embedFile("testdata/" ++ name);
        var m = try reader.read(a, bytes, null);
        var diag: Diagnostics = .{};
        const out = lower(a, &m, .{}, &diag) catch |err| {
            std.debug.print("{s}: {s}\n", .{ name, diag.message });
            return err;
        };
        for (out.kernels) |k| {
            const fm = try manifest.functionMetadata(a, k);
            _ = try assembler.assemble(a, out.ir, .{ .entry = k.name, .fm = fm, .check_operand_types = true }, air_target.default, .{ .fast_math = false });
        }
    }
}

/// Read, lower and assemble `bytes`, ignoring every error: a malformed or
/// unsupported module must come back as an error, never a crash.
fn fuzzOne(a: Allocator, bytes: []const u8) void {
    var m = reader.read(a, bytes, null) catch return;
    const out = lower(a, &m, .{}, null) catch return;
    for (out.kernels) |k| {
        const fm = manifest.functionMetadata(a, k) catch return;
        _ = assembler.assemble(a, out.ir, .{ .entry = k.name, .fm = fm, .check_operand_types = true }, air_target.default, .{ .fast_math = false }) catch continue;
    }
}

/// Mutations per input module in the fuzz test below (a local run with a
/// larger count found the crash sites this file now refuses).
const fuzz_iterations = 40;

test "truncated and bit-flipped modules are refused, never crash" {
    assembler.print_failures = false;
    defer assembler.print_failures = true;
    var prng = std.Random.DefaultPrng.init(0x5b1f);
    const rand = prng.random();
    const binaries = .{ "k-spirv64-vulkan.spv", "k-spirv64-opencl.spv", "desc.spv", "cf.spv", "narrow.spv" };
    const texts = .{ "hand-scalars", "hand-vectors", "hand-control-flow", "hand-glsl-std-450", "hand-opencl-std", "wgsl-boids", "wgsl-collatz", "wgsl-access", "wgsl-bits" };
    var inputs: [binaries.len + texts.len][]const u8 = undefined;
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    inline for (binaries, 0..) |name, i| inputs[i] = @embedFile("testdata/" ++ name);
    inline for (texts, 0..) |name, i| inputs[binaries.len + i] = try asm_.assemble(arena.allocator(), @embedFile("corpus/" ++ name ++ ".spvasm"), null);
    for (inputs) |bytes| {
        const words = bytes.len / 4;
        for (0..fuzz_iterations) |_| {
            var it = std.heap.ArenaAllocator.init(std.testing.allocator);
            defer it.deinit();
            const a = it.allocator();
            const buf = try a.dupe(u8, bytes);
            const len: usize = switch (rand.uintLessThan(u8, 4)) {
                // Truncate at a word boundary past the header.
                0 => 4 * (5 + rand.uintLessThan(usize, words - 5)),
                // Flip one to three bits of the instruction stream.
                1 => blk: {
                    for (0..1 + rand.uintLessThan(usize, 3)) |_| {
                        const bit = rand.uintLessThan(usize, (bytes.len - 20) * 8);
                        buf[20 + bit / 8] ^= @as(u8, 1) << @intCast(bit % 8);
                    }
                    break :blk buf.len;
                },
                // Replace a word with a small number (an id, an index, a count).
                2 => blk: {
                    const w = 5 + rand.uintLessThan(usize, words - 5);
                    std.mem.writeInt(u32, buf[4 * w ..][0..4], rand.uintLessThan(u32, 80), .little);
                    break :blk buf.len;
                },
                // Replace a word with random bits.
                else => blk: {
                    const w = 5 + rand.uintLessThan(usize, words - 5);
                    std.mem.writeInt(u32, buf[4 * w ..][0..4], rand.int(u32), .little);
                    break :blk buf.len;
                },
            };
            fuzzOne(a, buf[0..len]);
        }
    }
}

/// `text` must be refused (by the reader as malformed, or by the lowering
/// with a message containing `needle`), never crash.
fn expectRejected(a: Allocator, text: []const u8, needle: []const u8) !void {
    const bytes = try asm_.assemble(a, try withNames(a, text), null);
    var m = reader.read(a, bytes, null) catch return;
    var diag: Diagnostics = .{};
    if (lower(a, &m, .{}, &diag)) |out| {
        std.debug.print("expected a refusal with `{s}`, got:\n{s}\n", .{ needle, out.ir });
        return error.TestUnexpectedResult;
    } else |err| switch (err) {
        error.Unsupported => if (!has(diag.message, needle)) {
            std.debug.print("expected `{s}` in `{s}`\n", .{ needle, diag.message });
            return error.TestUnexpectedResult;
        },
        else => return err,
    }
}

test "SPIR-V 1.0 SSBOs: a Uniform BufferBlock is device memory, writable; the UBO beside it stays constant and read-only" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const exec = @import("exec.zig");
    const t = (try exec.all(a))[0];
    try std.testing.expectEqualStrings("bufferblock_ssbo", t.name);
    const r = try lowerTextForTest(a, t.text, .{});
    try std.testing.expectEqualStrings("", r.message);
    try std.testing.expect(has(r.ir, "define void @bufferblock_ssbo(ptr addrspace(1) %a0, ptr addrspace(1) %a1, ptr addrspace(2) %a2, <3 x i32> %a3) {"));
    try std.testing.expect(has(r.ir, "store i32 %y, ptr addrspace(1) %po, align 4"));
    try std.testing.expect(has(r.ir, "%kk = load i32, ptr addrspace(2) %pk, align 4"));
    const k = r.kernels[0];
    try std.testing.expectEqual(manifest.Space.device, k.params[1].buffer.space);
    try std.testing.expectEqual(manifest.Access.read_write, k.params[1].buffer.access);
    try std.testing.expectEqual(manifest.Space.constant, k.params[2].buffer.space);
    try std.testing.expectEqual(manifest.Access.read, k.params[2].buffer.access);
    // A store into the UBO is refused, not dropped by Metal.
    const bad = try std.mem.replaceOwned(u8, a, t.text, "%po = OpAccessChain %pu %out %c0 %g", "%po = OpAccessChain %pu %ubo %c0");
    const rb = try lowerTextForTest(a, bad, .{});
    try std.testing.expect(has(rb.message, "store into constant memory"));
    // NonWritable on the BufferBlock's member: read-only device memory.
    const ro = try std.mem.replaceOwned(u8, a, t.text, "OpDecorate %U Block", "OpMemberDecorate %Buf 0 NonWritable\nOpDecorate %U Block");
    const rr = try lowerTextForTest(a, ro, .{});
    try std.testing.expectEqual(manifest.Access.read, rr.kernels[0].params[0].buffer.access);
    try std.testing.expectEqual(manifest.Space.device, rr.kernels[0].params[0].buffer.space);
}

test "descriptor arrays and initialized buffer variables are refused by name" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const module =
        \\OpCapability Shader
        \\OpMemoryModel Logical GLSL450
        \\OpEntryPoint GLCompute %k "k"
        \\OpDecorate %rt ArrayStride 4
        \\OpDecorate %Buf Block
        \\OpMemberDecorate %Buf 0 Offset 0
        \\OpDecorate %bufs DescriptorSet 0
        \\OpDecorate %bufs Binding 0
        \\%void = OpTypeVoid
        \\%fn = OpTypeFunction %void
        \\%u32 = OpTypeInt 32 0
        \\%c0 = OpConstant %u32 0
        \\%c1 = OpConstant %u32 1
        \\%c2 = OpConstant %u32 2
        \\%rt = OpTypeRuntimeArray %u32
        \\%Buf = OpTypeStruct %rt
        \\%BufArr = OpTypeArray %Buf %c2
        \\%pArr = OpTypePointer StorageBuffer %BufArr
        \\%pu = OpTypePointer StorageBuffer %u32
        \\%bufs = OpVariable %pArr StorageBuffer
        \\%k = OpFunction %void None %fn
        \\%e = OpLabel
        \\%p = OpAccessChain %pu %bufs %c1 %c0 %c0
        \\OpStore %p %c2
        \\OpReturn
        \\OpFunctionEnd
    ;
    const r = try lowerTextForTest(a, module, .{});
    try std.testing.expect(has(r.message, "is a descriptor array"));
    const init_sb = try std.mem.replaceOwned(u8, a, module, "%bufs = OpVariable %pArr StorageBuffer", "%z = OpConstantNull %BufArr\n%bufs = OpVariable %pArr StorageBuffer %z");
    const r2 = try lowerTextForTest(a, init_sb, .{});
    try std.testing.expect(r2.message.len > 0);
}

test "an initialized UniformConstant is a constant global in address space 2, not a buffer" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const exec = @import("exec.zig");
    const t = (try exec.all(a))[1];
    try std.testing.expectEqualStrings("constant_table", t.name);
    const r = try lowerTextForTest(a, t.text, .{});
    try std.testing.expectEqualStrings("", r.message);
    try std.testing.expect(has(r.ir, "@tbl = private addrspace(2) constant [4 x i32] [i32 11, i32 22, i32 33, i32 44], align 4"));
    try std.testing.expect(has(r.ir, "define void @constant_table(ptr addrspace(1) %a0, ptr addrspace(1) %a1, <3 x i32> %a2) {"));
    try std.testing.expect(has(r.ir, "getelementptr inbounds [4 x i32], ptr addrspace(2) @tbl, i64 0, i32 %i"));
    try std.testing.expectEqual(3, r.kernels[0].params.len);
    try std.testing.expectEqualStrings("dst", r.kernels[0].params[0].buffer.name);
    try std.testing.expectEqualStrings("src", r.kernels[0].params[1].buffer.name);
}

test "SaturatedConversion clamps, FPRoundingMode rounds float-to-int and is refused where Metal cannot honour it" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try expectLowered(a, "%u32", "%u8 = OpTypeInt 8 0", "%s = OpUConvert %u8 %x\n%w = OpUConvert %u32 %s\nOpStore %pout_x %w", &.{});
    const sat = try kernelModuleWith(a, "%u32", "OpDecorate %s SaturatedConversion", "%u8 = OpTypeInt 8 0", "%s = OpUConvert %u8 %x\n%w = OpUConvert %u32 %s\nOpStore %pout_x %w");
    const r = try lowerTextForTest(a, sat, .{});
    try std.testing.expectEqualStrings("", r.message);
    try std.testing.expect(has(r.ir, "icmp ugt i32 %x, 255"));
    try std.testing.expect(has(r.ir, "select i1 %t1, i32 255, i32 %x"));
    try std.testing.expect(has(r.ir, "%s = trunc i32 %t2 to i8"));
    const fsat = try kernelModuleWith(a, "%f32", "OpDecorate %s SaturatedConversion\nOpDecorate %s FPRoundingMode RTN", "%i8 = OpTypeInt 8 1", "%s = OpConvertFToS %i8 %x\n%w = OpSConvert %i32 %s\n%wf = OpConvertSToF %f32 %w\nOpStore %pout_x %wf");
    const rf = try lowerTextForTest(a, fsat, .{});
    try std.testing.expectEqualStrings("", rf.message);
    try std.testing.expect(has(rf.ir, "call float @llvm.floor.f32(float %x)"));
    try std.testing.expect(has(rf.ir, "fcmp uno float"));
    try std.testing.expect(has(rf.ir, "fptosi float %t"));
    try std.testing.expect(has(rf.ir, "i8 127"));
    try std.testing.expect(has(rf.ir, "i8 -128"));
    // RTZ on a narrowing float conversion, and on an inexact int-to-float.
    const rtz = try kernelModuleWith(a, "%f32", "OpDecorate %h FPRoundingMode RTZ", "%f16 = OpTypeFloat 16", "%h = OpFConvert %f16 %x\n%b = OpFConvert %f32 %h\nOpStore %pout_x %b");
    try std.testing.expect(has((try lowerTextForTest(a, rtz, .{})).message, "FPRoundingMode rtz"));
    const itof = try kernelModuleWith(a, "%u32", "OpDecorate %f FPRoundingMode RTP", "", "%f = OpConvertUToF %f32 %x\n%b = OpConvertFToU %u32 %f\nOpStore %pout_x %b");
    try std.testing.expect(has((try lowerTextForTest(a, itof, .{})).message, "FPRoundingMode rtp"));
    // Exact conversions take any mode; RTE is Metal's own.
    const exact = try kernelModuleWith(a, "%u32", "OpDecorate %f FPRoundingMode RTZ\nOpDecorate %g FPRoundingMode RTE", "%u16 = OpTypeInt 16 0\n%f16 = OpTypeFloat 16", "%n = OpUConvert %u16 %x\n%f = OpConvertUToF %f32 %n\n%g = OpFConvert %f16 %f\n%h = OpFConvert %f32 %g\n%b = OpConvertFToU %u32 %h\nOpStore %pout_x %b");
    try std.testing.expectEqualStrings("", (try lowerTextForTest(a, exact, .{})).message);
    // Other semantic decorations on a result are refused by name.
    const inv = try kernelModuleWith(a, "%u32", "OpDecorate %y Invariant", "", "%y = OpIAdd %u32 %x %x\nOpStore %pout_x %y");
    try std.testing.expect(has((try lowerTextForTest(a, inv, .{})).message, "decoration invariant on the result is not supported"));
    // Precision and fast-math hints are ignored.
    const rp = try kernelModuleWith(a, "%u32", "OpDecorate %y RelaxedPrecision\nOpDecorate %y NoSignedWrap", "", "%y = OpIAdd %u32 %x %x\nOpStore %pout_x %y");
    try std.testing.expectEqualStrings("", (try lowerTextForTest(a, rp, .{})).message);
}

test "OpSelect of a struct and an array goes member by member" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try expectLowered(a, "%u32", "%v4u32 = OpTypeVector %u32 4\n%Sv = OpTypeStruct %u32 %v4u32\n%nS = OpConstantNull %Sv\n%c2 = OpConstant %u32 2\n%A2 = OpTypeArray %Sv %c2",
        \\%vv = OpCompositeConstruct %v4u32 %x %x %x %x
        \\%sv = OpCompositeConstruct %Sv %x %vv
        \\%lt = OpULessThan %bool %x %c1
        \\%sel = OpSelect %Sv %lt %sv %nS
        \\%ar = OpCompositeConstruct %A2 %sel %sv
        \\%ar2 = OpSelect %A2 %lt %ar %ar
        \\%e = OpCompositeExtract %u32 %ar2 1 1 2
        \\OpStore %pout_x %e
    , &.{
        "select i1 %lt, i32 %t",
        "select i1 %lt, <4 x i32> %t",
        "extractvalue { i32, <4 x i32> } zeroinitializer, 1",
        "insertvalue { i32, <4 x i32> } undef, i32 %t",
        "extractvalue [2 x { i32, <4 x i32> }] %t",
    });
}

test "a runtime-indexed struct of a private array is loaded and stored member by member, through one GEP each" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const exec = @import("exec.zig");
    const t = (try exec.all(a))[3];
    try std.testing.expectEqualStrings("composite_select", t.name);
    const r = try lowerTextForTest(a, t.text, .{});
    try std.testing.expectEqualStrings("", r.message);
    try std.testing.expect(has(r.ir, "getelementptr [2 x { i32, <4 x i32> }], ptr %var, i64 0, i32 %sidx, i32 0"));
    try std.testing.expect(has(r.ir, "getelementptr [2 x { i32, <4 x i32> }], ptr %var, i64 0, i32 %sidx, i32 1"));
    try std.testing.expect(has(r.ir, "load <4 x i32>, ptr %t"));
    try std.testing.expect(!has(r.ir, "load { i32, <4 x i32> }"));
}

test "a phi gets one entry per CFG edge: two switch cases on one target, equal branch targets" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const exec = @import("exec.zig");
    const t = (try exec.all(a))[4];
    try std.testing.expectEqualStrings("phi_edges", t.name);
    const r = try lowerTextForTest(a, t.text, .{});
    try std.testing.expectEqualStrings("", r.message);
    try std.testing.expect(has(r.ir, "%r = phi i32 [ 100, %be ], [ 100, %be ], [ 200, %bother ]") or has(r.ir, "phi i32 [ 100, %b"));
    try std.testing.expect(has(r.ir, ", [ %r, %b"));
}

test "NonSemantic instructions are dropped, and the variables they name stay out of the interface" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const exec = @import("exec.zig");
    const t = (try exec.all(a))[6];
    try std.testing.expectEqualStrings("nonsemantic_debuginfo", t.name);
    const r = try lowerTextForTest(a, t.text, .{});
    try std.testing.expectEqualStrings("", r.message);
    try std.testing.expectEqual(2, r.kernels[0].params.len);
    try std.testing.expect(!has(r.ir, "%scope"));
}

test "vec3 layouts: a scalar-layout vec3 array is [N x [3 x T]] and its elements load and store as <3 x T>" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const exec = @import("exec.zig");
    const t = (try exec.all(a))[5];
    try std.testing.expectEqualStrings("vec3_layouts", t.name);
    const r = try lowerTextForTest(a, t.text, .{});
    try std.testing.expectEqualStrings("", r.message);
    try std.testing.expect(has(r.ir, "getelementptr { [0 x [3 x float]] }, ptr addrspace(1) %a3, i64 0, i32 0, i32 %g"));
    try std.testing.expect(has(r.ir, "%w = load <3 x float>, ptr addrspace(1) %pe, align 4"));
    try std.testing.expect(has(r.ir, "store <3 x float> %w2, ptr addrspace(1) %pe2, align 4"));
    try std.testing.expect(has(r.ir, "load <{ [3 x float], float }>, ptr addrspace(1) %pr, align 4"));
    try std.testing.expect(has(r.ir, "store float %s1, ptr addrspace(1) %ps, align 4"));
    try std.testing.expectEqual(12, r.kernels[0].params[3].buffer.fields.?[0].size);
}

test "malformed instructions are refused, not crashes (the review's reproductions)" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const types = "%S = OpTypeStruct %u32 %u32\n%sc = OpConstantComposite %S %c0 %c1\n%pfu = OpTypePointer Function %u32";
    const cases = [_][2][]const u8{
        .{ "%y = OpCompositeExtract %u32 %sc 5\nOpStore %pout_x %y", "member index 5 of a struct with 2" },
        .{ "%y = OpCompositeExtract %v3u32 %gid_v\nOpStore %pout_x %x", "without an index" },
        .{ "%y = OpCompositeInsert %S %c0 %sc 7\nOpStore %pout_x %x", "member index 7" },
        .{ "%y = OpCompositeConstruct %S %x %x %x\nOpStore %pout_x %x", "3 constituents for 2 members" },
        .{ "%y = OpVectorShuffle %v3u32 %x %x 0 1 2\nOpStore %pout_x %x", "non-vector" },
        .{ "%y = OpUConvert %u64 %sc\nOpStore %pout_x %x", "is not an integer type" },
        .{ "%y = OpVariable %u32 Function\nOpStore %pout_x %x", "not a pointer type" },
        .{ "OpCopyMemory %c0 %c1\nOpStore %pout_x %x", "is not a pointer" },
        .{ "%y = OpAccessChain %pfu %x %c0\nOpStore %pout_x %x", "not a pointer type" },
        .{ "%y = OpIAddCarry %u32 %x %x\nOpStore %pout_x %y", "used as a struct type" },
        .{ "%y = OpExtInst %u32 %glsl 70000 %x\nOpStore %pout_x %y", "unknown extended instruction 70000" },
        .{ "%y = OpLoad %u32 %pin_x Aligned 3\nOpStore %pout_x %y", "Aligned 3 is not a power of two" },
        .{ "%y = OpIAdd %u32 %x %sc\nOpStore %pout_x %y", "does not have the type" },
        .{ "%y = OpExtInst %u32 %glsl FAbs %x\nOpStore %pout_x %y", "is not a float type" },
        .{ "%y = OpBitcast %u64 %x\nOpStore %pout_x %x", "between 32 and 64 bits" },
    };
    for (cases) |c| {
        try expectRejected(a, try kernelModule(a, "%u32", types, c[0]), c[1]);
    }
    // Sizes: a 2^31-element storage array, a 2^62-element Function array.
    try expectRejected(a, try kernelModule(a, "%u32", "%big = OpConstant %u32 2147483648\n%B2 = OpTypeArray %u32 %big\n%pB2 = OpTypePointer StorageBuffer %B2\n%huge = OpVariable %pB2 StorageBuffer\n%pbu = OpTypePointer StorageBuffer %u32", "%y = OpAccessChain %pbu %huge %c0\nOpStore %y %x"), "AIR's metadata describes at most 2 GiB");
    try expectRejected(a, try kernelModule(a, "%u32", "%big = OpConstant %u64 4611686018427387904\n%B2 = OpTypeArray %u32 %big\n%pB2 = OpTypePointer Function %B2", "%y = OpVariable %pB2 Function\nOpStore %pout_x %x"), "larger than 2^64 bytes");
    // LocalSizeId of a 2^33 constant.
    const lsid = try std.mem.replaceOwned(u8, a, try kernelModule(a, "%u32", "%c33 = OpConstant %u64 8589934592", "OpStore %pout_x %x"), "OpExecutionMode %k LocalSize 64 1 1", "OpExecutionModeId %k LocalSizeId %c33 %c1 %c1");
    try expectRejected(a, lsid, "does not fit 32 bits");
    // A constant whose type is a struct but which carries a literal.
    try expectRejected(a, try kernelModule(a, "%u32", "%S1 = OpTypeStruct %u32\n%odd = OpConstant %S1 5", "%y = OpCompositeExtract %u32 %odd 0\nOpStore %pout_x %y"), "");
}

/// The text of tools/spirv/exec.zig's test `name`.
fn execText(a: Allocator, name: []const u8) ![]const u8 {
    const exec = @import("exec.zig");
    for (try exec.all(a)) |t| if (std.mem.eql(u8, t.name, name)) return t.text;
    return error.TestUnexpectedResult;
}

fn expectHas(ir: []const u8, needle: []const u8) !void {
    if (has(ir, needle)) return;
    std.debug.print("missing `{s}` in:\n{s}\n", .{ needle, ir });
    return error.TestUnexpectedResult;
}

test "a packed struct inside one with natural Offsets packs the parent: LLVM aligns a packed struct to 1" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    // std430 `{ uint count; Particle ps[]; }` and std140 `{ uint count;
    // Light l; }`: the array / struct at 16 needs explicit padding.
    const r = try lowerTextForTest(a, try execText(a, "packed_in_natural"), .{});
    try std.testing.expectEqualStrings("", r.message);
    try expectHas(r.ir, "getelementptr <{ i32, [12 x i8], [0 x <{ [3 x i32], i32, <4 x i32> }>] }>, ptr addrspace(1) %a0, i64 0, i32 2, i32 %g, i32 1");
    try expectHas(r.ir, "getelementptr <{ i32, [12 x i8], <{ [3 x i32], i32, <4 x i32> }> }>, ptr addrspace(2) %a2, i64 0, i32 2, i32 1");
    try expectHas(r.ir, "%cy = load i32, ptr addrspace(1) %pcy, align 4");
    const b = r.kernels[0].params[0].buffer;
    try std.testing.expectEqual(16, b.fields.?[1].offset);
    try std.testing.expectEqual(16, r.kernels[0].params[2].buffer.fields.?[1].offset);
    try std.testing.expectEqual(48, r.kernels[0].params[2].buffer.size);
    // Scalar layout: a packed member at 16 packs the parent; at 4 the
    // natural layout already puts it there (ABI alignment 1), and the
    // uvec4 inside is loaded with the alignment its offset allows.
    const s = try lowerTextForTest(a, try execText(a, "scalar_packed"), .{});
    try std.testing.expectEqualStrings("", s.message);
    try expectHas(s.ir, "getelementptr <{ i32, [12 x i8], [0 x <{ i32, <4 x i32> }>] }>, ptr addrspace(1) %a0, i64 0, i32 2, i32 %g, i32 1");
    try expectHas(s.ir, "load <4 x i32>, ptr addrspace(1) %pv, align 4");
    try expectHas(s.ir, "getelementptr { i32, <{ i32, <4 x i32> }> }, ptr addrspace(1) %a2, i64 0, i32 1, i32 1");
    try expectHas(s.ir, "load <4 x i32>, ptr addrspace(1) %pv3, align 4");
    try std.testing.expectEqual(24, s.kernels[0].params[2].buffer.size);
}

test "CPacked structs have no padding and are accessed with alignment 1; other layout decorations are refused" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const text = try execText(a, "cpacked_struct");
    const r = try lowerTextForTest(a, text, .{});
    try std.testing.expectEqualStrings("", r.message);
    try expectHas(r.ir, "getelementptr inbounds <{ i8, i32 }>, ptr addrspace(1) %a1, i64 %g, i32 1");
    try expectHas(r.ir, "%x = load i32, ptr addrspace(1) %pm, align 1");
    const b = r.kernels[0].params[1].buffer;
    try std.testing.expectEqual(5, b.size);
    try std.testing.expectEqual(1, b.alignment);
    try std.testing.expectEqual(1, b.fields.?[1].offset);
    // An Offset that contradicts CPacked, and a layout decoration this
    // lowering does not implement.
    try expectRejected(a, try std.mem.replaceOwned(u8, a, text, "OpDecorate %S CPacked", "OpDecorate %S CPacked\nOpMemberDecorate %S 0 Offset 0\nOpMemberDecorate %S 1 Offset 4"), "is CPacked but member 1 has Offset 4");
    try expectRejected(a, try std.mem.replaceOwned(u8, a, text, "OpDecorate %S CPacked", "OpDecorate %S GLSLPacked"), "carries decoration glsl_packed");
    try expectRejected(a, try std.mem.replaceOwned(u8, a, text, "OpDecorate %S CPacked", "OpMemberDecorate %S 1 Location 3"), "member 1 carries decoration location");
}

test "execution modes: sizes, hints and RTE pass; float controls Metal cannot honour are refused by name" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const base = try kernelModule(a, "%f32", "", "%y = OpFAdd %f32 %x %x\nOpStore %pout_x %y");
    const ls = "OpExecutionMode %k LocalSize 64 1 1";
    const refused = [_][2][]const u8{
        .{ "RoundingModeRTZ 32", "RoundingModeRTZ" },
        .{ "DenormPreserve 32", "no control over denormals" },
        .{ "DenormFlushToZero 16", "no control over denormals" },
        .{ "SignedZeroInfNanPreserve 32", "SignedZeroInfNanPreserve: not verified" },
        .{ "RoundingModeRTE 64", "only 16- and 32-bit floats" },
        .{ "SubgroupSize 32", "execution mode subgroup_size is not supported" },
    };
    for (refused) |c| {
        const text = try std.mem.replaceOwned(u8, a, base, ls, try a.print("{s}\nOpExecutionMode %k {s}", .{ ls, c[0] }));
        try expectRejected(a, text, c[1]);
    }
    for ([_][]const u8{ "RoundingModeRTE 32", "RoundingModeRTE 16", "ContractionOff", "LocalSizeHint 8 1 1" }) |mode| {
        const text = try std.mem.replaceOwned(u8, a, base, ls, try a.print("{s}\nOpExecutionMode %k {s}", .{ ls, mode }));
        const r = try lowerTextForTest(a, text, .{});
        try std.testing.expectEqualStrings("", r.message);
    }
}

test "a constant decorated BuiltIn WorkgroupSize overrides LocalSize in the manifest" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const text = try execText(a, "workgroup_size_constant");
    const r = try lowerTextForTest(a, text, .{});
    try std.testing.expectEqualStrings("", r.message);
    try std.testing.expectEqual([3]u32{ 64, 1, 1 }, r.kernels[0].local_size);
    // It wins over LocalSizeId too.
    const by_id = try std.mem.replaceOwned(u8, a, text, "OpExecutionMode %k LocalSize 1 1 1", "OpExecutionModeId %k LocalSizeId %c1 %c1 %c1");
    try std.testing.expectEqual([3]u32{ 64, 1, 1 }, (try lowerTextForTest(a, by_id, .{})).kernels[0].local_size);
    // A component computed by OpSpecConstantOp is not evaluated: refused
    // (here the body does not read the constant, so the manifest is what
    // refuses it).
    const unread = try std.mem.replaceOwned(u8, a, text, "%sz = OpCompositeExtract %u32 %wsz 0", "%sz = OpCopyObject %u32 %sx");
    const op = try std.mem.replaceOwned(u8, a, unread, "%wsz = OpSpecConstantComposite %v3u32 %sx %c1 %c1", "%sy = OpSpecConstantOp %u32 IAdd %sx %c1\n%wsz = OpSpecConstantComposite %v3u32 %sx %sy %c1");
    try expectRejected(a, op, "is not an integer constant");
    // A zero component.
    const zero = try std.mem.replaceOwned(u8, a, text, "%wsz = OpSpecConstantComposite %v3u32 %sx %c1 %c1", "%wsz = OpSpecConstantComposite %v3u32 %sx %c0 %c1");
    try expectRejected(a, zero, "has a zero component");
}

test "a pointer into a BufferBlock is refused where its Uniform type would be printed (VariablePointers)" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const module =
        \\OpCapability Shader
        \\OpCapability VariablePointers
        \\OpMemoryModel Logical GLSL450
        \\OpEntryPoint GLCompute %k "k" %gid
        \\OpExecutionMode %k LocalSize 1 1 1
        \\OpDecorate %gid BuiltIn GlobalInvocationId
        \\OpDecorate %rt ArrayStride 4
        \\OpDecorate %B BufferBlock
        \\OpMemberDecorate %B 0 Offset 0
        \\OpDecorate %in DescriptorSet 0
        \\OpDecorate %in Binding 0
        \\OpDecorate %out DescriptorSet 0
        \\OpDecorate %out Binding 1
        \\%void = OpTypeVoid
        \\%fn = OpTypeFunction %void
        \\%u32 = OpTypeInt 32 0
        \\%v3u32 = OpTypeVector %u32 3
        \\%c0 = OpConstant %u32 0
        \\%rt = OpTypeRuntimeArray %u32
        \\%B = OpTypeStruct %rt
        \\%pB = OpTypePointer Uniform %B
        \\%pu = OpTypePointer Uniform %u32
        \\%pfp = OpTypePointer Function %pu
        \\%pfB = OpTypePointer Function %pB
        \\%pin = OpTypePointer Input %v3u32
        \\%gid = OpVariable %pin Input
        \\%in = OpVariable %pB Uniform
        \\%out = OpVariable %pB Uniform
        \\%k = OpFunction %void None %fn
        \\%e = OpLabel
        \\%tmp = OpVariable %pfp Function
        \\%gv = OpLoad %v3u32 %gid
        \\%g = OpCompositeExtract %u32 %gv 0
        \\%pi = OpAccessChain %pu %in %c0 %g
        \\OpStore %tmp %pi
        \\%pi2 = OpLoad %pu %tmp
        \\%x = OpLoad %u32 %pi2
        \\%po = OpAccessChain %pu %out %c0 %g
        \\OpStore %po %x
        \\OpReturn
        \\OpFunctionEnd
    ;
    // The review's t6: stored into a Function variable of type `ptr Uniform`.
    try expectRejected(a, module, "points into a BufferBlock (device memory, although its type says Uniform); OpStore");
    // As a Function variable's initializer.
    const init = try std.mem.replaceOwned(u8, a, module, "%tmp = OpVariable %pfp Function", "%tmp = OpVariable %pfp Function\n%tb = OpVariable %pfB Function %in");
    try expectRejected(a, init, "OpVariable would give it");
    // Used directly, the same pointers lower to device-memory accesses.
    const direct = try std.mem.replaceOwned(u8, a, module, "OpStore %tmp %pi\n%pi2 = OpLoad %pu %tmp\n%x = OpLoad %u32 %pi2", "%x = OpLoad %u32 %pi");
    const r = try lowerTextForTest(a, direct, .{});
    try std.testing.expectEqualStrings("", r.message);
    try expectHas(r.ir, "%x = load i32, ptr addrspace(1) %pi, align 4");
}

test "the assembler's operand type check catches a lowering type bug that would otherwise assemble" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const r = try lowerTextForTest(a, try kernelModule(a, "%u32", "", "OpStore %pout_x %x"), .{});
    try std.testing.expectEqualStrings("", r.message);
    const bad = try std.mem.replaceOwned(u8, a, r.ir, "store i32 %x, ptr addrspace(1) %pout_x", "store i32 %x, ptr addrspace(2) %pout_x");
    try std.testing.expect(!std.mem.eql(u8, bad, r.ir));
    const fm = try manifest.functionMetadata(a, r.kernels[0]);
    // Off (air-splice's text, relocated by design): assembles.
    _ = try assembler.assemble(a, bad, .{ .entry = "k", .fm = fm }, air_target.default, .{ .fast_math = false });
    // On (this file's text): a syntax error.
    try std.testing.expectError(error.Syntax, assembler.assemble(a, bad, .{ .entry = "k", .fm = fm, .check_operand_types = true }, air_target.default, .{ .fast_math = false }));
}

test "a packed member's alignment survives OpSelect, phis, helper parameters and pointers kept in memory" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    // &ps[g].v of a scalar-layout `{ uint @0; uvec4 @4 }` (stride 20) is
    // 4-aligned; every load of a uvec4 through it must say so.
    const text = try execText(a, "packed_pointer_flow");
    const r = try lowerTextForTest(a, text, .{});
    try std.testing.expectEqualStrings("", r.message);
    try expectHas(r.ir, "%vs = load <4 x i32>, ptr addrspace(1) %psel, align 4");
    try expectHas(r.ir, "%hv = load <4 x i32>, ptr addrspace(1) %hp, align 4");
    try expectHas(r.ir, "%lv = load <4 x i32>, ptr addrspace(1) %pl, align 4");
    try expectHas(r.ir, "%wv = load <4 x i32>, ptr addrspace(1) %pv, align 4");
    try std.testing.expect(!has(r.ir, "align 16"));
    // With std430 Offsets (member at 16, stride 32) nothing is narrowed.
    const natural = try std.mem.replaceOwned(u8, a, try std.mem.replaceOwned(u8, a, text, "OpMemberDecorate %P 1 Offset 4", "OpMemberDecorate %P 1 Offset 16"), "ArrayStride 20", "ArrayStride 32");
    const rn = try lowerTextForTest(a, natural, .{});
    try std.testing.expectEqualStrings("", rn.message);
    for ([_][]const u8{ "%vs", "%hv", "%lv", "%wv" }) |v| try expectHas(rn.ir, try a.print("{s} = load <4 x i32>, ptr addrspace(1) %{s}, align 16", .{ v, switch (v[1]) {
        'v' => "psel",
        'h' => "hp",
        'l' => "pl",
        else => "pv",
    } }));
}

test "an atomic on an address a packed member leaves under-aligned is refused" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const text =
        \\OpCapability Shader
        \\OpCapability Int16
        \\OpMemoryModel Logical GLSL450
        \\OpEntryPoint GLCompute %k "k"
        \\OpExecutionMode %k LocalSize 1 1 1
        \\OpDecorate %S Block
        \\OpMemberDecorate %S 0 Offset 0
        \\OpMemberDecorate %S 1 Offset 2
        \\OpDecorate %buf DescriptorSet 0
        \\OpDecorate %buf Binding 0
        \\%void = OpTypeVoid
        \\%fn = OpTypeFunction %void
        \\%u16 = OpTypeInt 16 0
        \\%u32 = OpTypeInt 32 0
        \\%c0 = OpConstant %u32 0
        \\%c1 = OpConstant %u32 1
        \\%S = OpTypeStruct %u16 %u32
        \\%pS = OpTypePointer StorageBuffer %S
        \\%pu = OpTypePointer StorageBuffer %u32
        \\%buf = OpVariable %pS StorageBuffer
        \\%k = OpFunction %void None %fn
        \\%e = OpLabel
        \\%p = OpAccessChain %pu %buf %c1
        \\%old = OpAtomicIAdd %u32 %p %c1 %c0 %c1
        \\OpReturn
        \\OpFunctionEnd
    ;
    try expectRejected(a, text, "an atomic on an address only 2-byte aligned");
    // At Offset 4 the same atomic lowers.
    const r = try lowerTextForTest(a, try std.mem.replaceOwned(u8, a, text, "OpMemberDecorate %S 1 Offset 2", "OpMemberDecorate %S 1 Offset 4"), .{});
    try std.testing.expectEqualStrings("", r.message);
    try expectHas(r.ir, "atomicrmw add ptr addrspace(1) %p, i32 1 monotonic, align 4");
}

test "OpBitFieldUExtract at Offset = width, Count = 0 selects 0 rather than masking a poison shift" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const r = try lowerTextForTest(a, try execText(a, "bitfield_full_offset"), .{});
    try std.testing.expectEqualStrings("", r.message);
    // The stored value is the select on Count == 0, whose other arm is the
    // masked field (whose `lshr` by 32 is poison and folds away).
    try expectHas(r.ir, "%t1 = lshr i32 -559038737, 32");
    try expectHas(r.ir, "%t4 = and i32 %t1, %t3");
    try expectHas(r.ir, "%t6 = select i1 %t5, i32 0, i32 %t4");
    try expectHas(r.ir, "store i32 %t6, ptr addrspace(1) %po0, align 4");
}

test "OpenCL powr, normalize, length and distance keep OpenCL C's edge cases; GLSL's do not need them" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const r = try lowerTextForTest(a, try execText(a, "opencl_edge_math"), .{});
    try std.testing.expectEqualStrings("", r.message);
    // powr: pow(|x|, y), NaN selected at the defined edges; pow stays pow.
    try expectHas(r.ir, "%t2 = call float @llvm.pow.f32(float %t1, float %x1)");
    try expectHas(r.ir, "%t17 = select i1 %t16, float 0x7FF8000000000000, float %t2");
    try expectHas(r.ir, "store float %t17, ptr addrspace(1) %po0, align 4");
    try expectHas(r.ir, "call float @llvm.pow.f32(float %x0, float %x1)");
    // normalize: the zero test, the infinity replacement, the scaling.
    try expectHas(r.ir, "call i1 @air.all.v2i1(");
    try expectHas(r.ir, "call i1 @air.any.v2i1(");
    try expectHas(r.ir, "call <2 x float> @llvm.copysign.v2f32(");
    try expectHas(r.ir, "float 0x3B90000000000000");
    // length / distance scale back by the inverse power of two.
    try expectHas(r.ir, "%t273 = fmul float %t272, %t267");
    try expectHas(r.ir, "store float %t273, ptr addrspace(1) %po20, align 4");
    // GLSL Normalize of a vector is still v / length(v).
    try expectLowered(a, "%v4f32", "%f32_ = OpTypeFloat 32\n%v4f32 = OpTypeVector %f32 4", "%y = OpExtInst %v4f32 %glsl Normalize %x\nOpStore %pout_x %y", &.{"fdiv <4 x float> %x, %t"});
}

// ── Review round 1 (lowering): regressions ─────────────────────────────────

test "atan and atan2 take the sign of x / y back with llvm.copysign (strict f32 only)" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const body = "%r = OpExtInst %f32 %glsl Atan %x\n%r2 = OpExtInst %f32 %glsl Atan2 %x %r\nOpStore %pout_x %r2";
    const r = try lowerTextForTest(a, try kernelModule(a, "%f32", "", body), .{});
    try std.testing.expectEqualStrings("", r.message);
    try expectHas(r.ir, "= call float @air.atan.f32(float %x)");
    try expectHas(r.ir, "= call float @air.atan2.f32(float %x, float %t2)");
    try std.testing.expectEqual(2, std.mem.count(u8, r.ir, "= call float @llvm.copysign.f32("));
    try expectHas(r.ir, ", float %x)\n");
    // Apple's fast forms and half precision already keep the sign.
    const fast = try lowerTextForTest(a, try kernelModule(a, "%f32", "", body), .{ .fast_math = true });
    try expectHas(fast.ir, "@air.fast_atan.f32");
    try std.testing.expect(!has(fast.ir, "copysign"));
    const half = try lowerTextForTest(a, try kernelModule(a, "%f16", "%f16 = OpTypeFloat 16", "%r = OpExtInst %f16 %glsl Atan %x\nOpStore %pout_x %r"), .{});
    try std.testing.expectEqualStrings("", half.message);
    try std.testing.expect(!has(half.ir, "copysign"));
    // The OpenCL execution test (atan(-0), atan2(-1e-30, 1e30), ...).
    const ocl = try lowerTextForTest(a, try execText(a, "atan_signed_zero"), .{});
    try std.testing.expectEqual(2, std.mem.count(u8, ocl.ir, "= call float @llvm.copysign.f32(float"));
}

test "an access chain on a pointer bitcast to another pointee GEPs from the cast address" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const r = try lowerTextForTest(a, try execText(a, "bitcast_access_chain"), .{});
    try std.testing.expectEqualStrings("", r.message);
    try expectHas(r.ir, "%p0 = getelementptr inbounds { [16 x i32] }, ptr addrspace(1) %a0, i64 0, i32 0\n");
    try expectHas(r.ir, "%p2 = getelementptr inbounds [4 x <4 x i32>], ptr addrspace(1) %p0, i64 0, i64 1\n");
    try expectHas(r.ir, "%v = load <4 x i32>, ptr addrspace(1) %p2, align 16");
    // A copy of the same type still continues the root's GEP.
    const text = try std.mem.replaceOwned(u8, a, try execText(a, "bitcast_access_chain"), "%pb = OpBitcast %parr4v4 %p0\n%p2 = OpInBoundsAccessChain %pv4 %pb %c1", "%pc = OpCopyObject %parr16 %p0\n%pq = OpInBoundsAccessChain %pu %pc %c1\n%pb = OpBitcast %parr4v4 %p0\n%p2 = OpInBoundsAccessChain %pv4 %pb %c1");
    const text2 = try std.mem.replaceOwned(u8, a, text, "%pv4 = OpTypePointer StorageBuffer %v4", "%pv4 = OpTypePointer StorageBuffer %v4\n%pu = OpTypePointer StorageBuffer %u32");
    const r2 = try lowerTextForTest(a, text2, .{});
    try std.testing.expectEqualStrings("", r2.message);
    try expectHas(r2.ir, "%pq = getelementptr inbounds { [16 x i32] }, ptr addrspace(1) %a0, i64 0, i32 0, i64 1\n");
}

test "Workgroup Block variables alias: one threadgroup allocation, the largest" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const r = try lowerTextForTest(a, try execText(a, "workgroup_block_alias"), .{});
    try std.testing.expectEqualStrings("", r.message);
    try expectHas(r.ir, "@wa = internal addrspace(3) global [512 x i8] undef, align 4\n");
    try std.testing.expectEqual(1, std.mem.count(u8, r.ir, "addrspace(3) global"));
    try expectHas(r.ir, "getelementptr { [64 x i32] }, ptr addrspace(3) @wa, i64 0, i32 0, i32 %l");
    try expectHas(r.ir, "getelementptr { [64 x i32], [64 x i32] }, ptr addrspace(3) @wa, i64 0, i32 0, i32 %r");
    try std.testing.expectEqual(512, r.kernels[0].threadgroup_bytes);
}

test "scalar layout: a struct ending in a vec3 is stored without its AIR tail padding" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const base = try execText(a, "scalar_tight_structs");
    const r = try lowerTextForTest(a, base, .{});
    try std.testing.expectEqualStrings("", r.message);
    // P[] at ArrayStride 12, B { Inner @0; float @12 }.
    try expectHas(r.ir, "getelementptr { [0 x <{ [3 x float] }>] }, ptr addrspace(1) %a0, i64 0, i32 0, i32 %g");
    try expectHas(r.ir, "= load <{ [3 x float] }>, ptr addrspace(1) %ps, align 4");
    try expectHas(r.ir, "getelementptr { [0 x <{ <{ [3 x float] }>, float }>] }, ptr addrspace(1) %a2, i64 0, i32 0, i32 %g, i32 0");
    try expectHas(r.ir, "store <{ [3 x float] }> %t");
    try std.testing.expect(!has(r.ir, "store { <3 x float> }"));
    try std.testing.expect(!has(r.ir, "load { <3 x float> }"));
    // A constant B stored whole: the tight member's constant is packed too.
    const with_const = try std.mem.replaceOwned(u8, a, base, "OpStore %pi %ni\n", "OpStore %pi %ni\n%pbe = OpAccessChain %pB %bb %c0 %g\nOpStore %pbe %cb\n");
    const with_ty = try std.mem.replaceOwned(u8, a, with_const, "%pf = OpTypePointer StorageBuffer %f32", "%pf = OpTypePointer StorageBuffer %f32\n%pB = OpTypePointer StorageBuffer %B\n%cv = OpConstantComposite %v3f32 %f2 %f2 %f2\n%ci = OpConstantComposite %Inner %cv\n%cb = OpConstantComposite %B %ci %f2");
    const rc = try lowerTextForTest(a, with_ty, .{});
    try std.testing.expectEqualStrings("", rc.message);
    try expectHas(rc.ir, "store <{ <{ [3 x float] }>, float }> <{ <{ [3 x float] }> <{ [3 x float] [float 0x4000000000000000, float 0x4000000000000000, float 0x4000000000000000] }>, float 0x4000000000000000 }>");
    // A tight pointer may not leave the load/store/chain paths.
    const escaped = try std.mem.replaceOwned(u8, a, base, "%s = OpLoad %P %ps", "%ps2 = OpPtrAccessChain %pP %ps %c1\n%s = OpLoad %P %ps2");
    const re = try lowerTextForTest(a, escaped, .{});
    try std.testing.expect(has(re.message, "OpPtrAccessChain would access it at its full AIR size"));
}

test "helper chains: 10000 calls lower without recursion, 70 carry a packed alignment in two rounds" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const exec = @import("exec.zig");
    for ([_]usize{ 70, 10000 }) |depth| {
        const bytes = try asm_.assemble(a, try exec.callChainModule(a, depth), null);
        var m = try reader.read(a, bytes, null);
        var diag: Diagnostics = .{};
        const out = lower(a, &m, .{}, &diag) catch |err| {
            std.debug.print("depth {d}: {t} {s}\n", .{ depth, err, diag.message });
            return err;
        };
        // The leaf's uvec4 load claims the Offset's 4, not uvec4's 16.
        try std.testing.expectEqual(1, std.mem.count(u8, out.ir, "= load <4 x i32>, ptr addrspace(1) %"));
        try std.testing.expect(!has(out.ir, "align 16"));
        try std.testing.expect(diag.rounds <= 2);
    }
    // The 70-deep chain also assembles.
    const r = try lowerTextForTest(a, try exec.callChainModule(a, 70), .{});
    try std.testing.expectEqualStrings("", r.message);
}

test "malformed bodies and constants are refused with the instruction quoted" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const cases = [_]struct { types: []const u8, body: []const u8, quote: []const u8, why: []const u8 }{
        .{ .types = "", .body = "OpSelectionMerge %m None\nOpSwitch %x %m 1 %sa 1 %sb\n%sa = OpLabel\nOpBranch %m\n%sb = OpLabel\nOpBranch %m\n%m = OpLabel", .quote = "`OpSwitch %x %", .why = "OpSwitch has the case literal 1 twice" },
        .{ .types = "%ct = OpConstantTrue %u32", .body = "%y = OpIAdd %u32 %x %ct\nOpStore %pout_x %y", .quote = "`%ct = OpConstantTrue %", .why = "a bool constant of the non-bool type" },
        .{ .types = "%hv = OpTypeVector %u32 4294967295\n%n = OpConstantNull %hv", .body = "%y = OpCompositeExtract %u32 %n 0\nOpStore %pout_x %y", .quote = "`%y = OpCompositeExtract %", .why = "has 4294967295 components" },
        .{ .types = "", .body = "OpBranch %c1\n%m = OpLabel", .quote = "`OpBranch %c1`", .why = "is not a block of this function" },
        .{ .types = "", .body = "%y = OpIAdd %u32 %z %x\n%z = OpIAdd %u32 %x %x\nOpStore %pout_x %y", .quote = "`%y = OpIAdd %", .why = "is used before its definition in SPIR-V's block order" },
        .{ .types = "%cb = OpConstantTrue %bool", .body = "OpSelectionMerge %m None\nOpBranchConditional %cb %t %m\n%t = OpLabel\n%z = OpIAdd %u32 %x %x\nOpBranch %m\n%m = OpLabel\nOpStore %pout_x %z", .quote = "`OpStore %pout_x %z`", .why = "does not dominate this use" },
        .{ .types = "%cb = OpConstantTrue %bool", .body = "OpSelectionMerge %m None\nOpBranchConditional %cb %t %m\n%t = OpLabel\nOpBranch %m\n%m = OpLabel\n%p = OpPhi %u32 %x %t\nOpStore %pout_x %p", .quote = "`%p = OpPhi %", .why = "OpPhi has no value for the parent block" },
        .{ .types = "%cb = OpConstantTrue %bool\n%f1 = OpConstant %f32 1", .body = "OpSelectionMerge %m None\nOpBranchConditional %cb %t %m\n%t = OpLabel\nOpBranch %m\n%m = OpLabel\n%p = OpPhi %u32 %x %t %f1 %entry\nOpStore %pout_x %p", .quote = "`%p = OpPhi %", .why = "does not have the phi's type" },
        .{ .types = "%p_fn_f32 = OpTypePointer Function %f32", .body = "%vv = OpVariable %p_fn_f32 Function %c1\nOpStore %pout_x %x", .quote = "`%vv = OpVariable %", .why = "does not have the type" },
    };
    for (cases) |c| {
        const r = try lowerTextForTest(a, try kernelModule(a, "%u32", c.types, c.body), .{});
        if (!has(r.message, c.quote) or !has(r.message, c.why)) {
            std.debug.print("expected {s}: ...{s}, got `{s}`\n", .{ c.quote, c.why, r.message });
            return error.TestUnexpectedResult;
        }
    }
    // A value of another function, an OpFunction typed by a non-function.
    {
        const text = try a.print("{s}%h = OpFunction %void None %fn_void\n%he = OpLabel\n%hy = OpIAdd %u32 %x %x\nOpReturn\nOpFunctionEnd\n", .{try kernelModule(a, "%u32", "", "%cr = OpFunctionCall %void %h\nOpStore %pout_x %x")});
        const r = try lowerTextForTest(a, text, .{});
        try std.testing.expect(has(r.message, "`%hy = OpIAdd %") and has(r.message, "is defined in another function"));
    }
    {
        const text = try std.mem.replaceOwned(u8, a, try kernelModule(a, "%u32", "", "OpStore %pout_x %x"), "%k = OpFunction %void None %fn_void", "%k = OpFunction %void None %u32");
        const r = try lowerTextForTest(a, text, .{});
        try std.testing.expect(has(r.message, "`%k = OpFunction %") and has(r.message, "`: the function type %") and has(r.message, "is not an OpTypeFunction"));
    }
}

test "ill-typed results of CompositeInsert, access chains and CopyObject, and module-scope computations, are refused with the instruction quoted" {
    // Round-2 fuzzing (docs review): each of these reached the assembler as
    // ill-typed IR, refused by IR line ("operand's type differs ...", "use of
    // undefined value"), or was packed silently.
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const cases = [_]struct { types: []const u8, body: []const u8, quote: []const u8, why: []const u8 }{
        .{ .types = "%v4u32 = OpTypeVector %u32 4", .body = "%v = OpCompositeConstruct %v4u32 %x %x %x %x\n%w = OpCompositeInsert %v3u32 %x %v 0\n%e = OpCompositeExtract %u32 %w 1\nOpStore %pout_x %e", .quote = "`%w = OpCompositeInsert %", .why = "the composite %" },
        .{ .types = "", .body = "%q = OpAccessChain %u32 %pout_x\n%e = OpIAdd %u32 %q %x\nOpStore %pout_x %e", .quote = "`%q = OpAccessChain %", .why = "is not a pointer type" },
        .{ .types = "%p_sb_f32 = OpTypePointer StorageBuffer %f32\n%ff = OpConstant %f32 1.5", .body = "%q = OpAccessChain %p_sb_f32 %pout_x\nOpStore %q %ff", .quote = "`%q = OpAccessChain %", .why = "but the instruction reaches %" },
        .{ .types = "%p_fn_u32 = OpTypePointer Function %u32", .body = "%q = OpAccessChain %p_fn_u32 %pout_x\nOpStore %q %x", .quote = "`%q = OpAccessChain %", .why = "is a pointer into Function memory, the base's is StorageBuffer" },
        .{ .types = "%p_fn_u32 = OpTypePointer Function %u32", .body = "%q = OpAccessChain %p_fn_u32 %out %c0 %gid_x\nOpStore %q %x", .quote = "`%q = OpAccessChain %", .why = "is a pointer into Function memory, the base's is StorageBuffer" },
        .{ .types = "%p_sb_f32 = OpTypePointer StorageBuffer %f32\n%ff = OpConstant %f32 1.5", .body = "%q = OpAccessChain %p_sb_f32 %out %c0 %gid_x\nOpStore %q %ff", .quote = "`%q = OpAccessChain %", .why = "but the instruction reaches %" },
        .{ .types = "%p_fn_u32 = OpTypePointer Function %u32", .body = "%q = OpCopyObject %p_fn_u32 %pout_x\nOpStore %q %x", .quote = "`%q = OpCopyObject %", .why = "is a pointer into Function memory, the base's is StorageBuffer" },
        .{ .types = "", .body = "%q = OpCopyObject %f32 %x\n%e = OpBitcast %u32 %q\nOpStore %pout_x %e", .quote = "`%q = OpCopyObject %", .why = "the operand %" },
        .{ .types = "%cf = OpConstant %f32 -2.5\n%e = OpExtInst %f32 %glsl FAbs %cf", .body = "%u = OpConvertFToU %u32 %e\nOpStore %pout_x %u", .quote = "`%u = OpConvertFToU %", .why = "is the result of an OpExtInst at module scope" },
    };
    for (cases) |c| {
        const r = try lowerTextForTest(a, try kernelModule(a, "%u32", c.types, c.body), .{});
        if (!has(r.message, c.quote) or !has(r.message, c.why)) {
            std.debug.print("expected {s}: ...{s}, got `{s}`\n", .{ c.quote, c.why, r.message });
            return error.TestUnexpectedResult;
        }
    }
    // The well-typed forms still lower: an access chain without indexes and
    // a CopyObject of the same pointer type, a CompositeInsert into its own
    // type, and an access chain whose result type is a second, identical
    // pointer type (Zig's ArrayStride-decorated pointers).
    const ok = try lowerTextForTest(a, try kernelModule(a, "%u32", "%p_elem2 = OpTypePointer StorageBuffer %u32",
        \\%q = OpAccessChain %p_elem %pout_x
        \\%r = OpCopyObject %p_elem %q
        \\%s = OpAccessChain %p_elem2 %out %c0 %gid_x
        \\%g2 = OpCompositeInsert %v3u32 %x %gid_v 1
        \\%e = OpCompositeExtract %u32 %g2 1
        \\OpStore %r %e
        \\OpStore %s %e
    ), .{});
    try std.testing.expectEqualStrings("", ok.message);
}

test "GL_EXT_buffer_reference: pointers in a push-constant block, from a uint64 and a uvec2" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const r = try lowerTextForTest(a, try execText(a, "buffer_reference"), .{});
    try std.testing.expectEqualStrings("", r.message);
    try expectHas(r.ir, "%src = load ptr addrspace(1), ptr addrspace(2) %psrc, align 8");
    try expectHas(r.ir, "getelementptr { [0 x i32] }, ptr addrspace(1) %src, i64 0, i32 0, i32 %g");
    try expectHas(r.ir, "%aux = inttoptr i64 %auxu to ptr addrspace(1)");
    try expectHas(r.ir, "= bitcast <2 x i32> %aux2v to i64");
    try expectHas(r.ir, "= ptrtoint ptr addrspace(1) %src to i64");
    const pc = r.kernels[0].params[0].buffer;
    try std.testing.expectEqual(30, pc.index);
    try std.testing.expectEqualStrings("ulong", pc.fields.?[0].type_name);
    try std.testing.expectEqual(48, pc.size);
    // Threadgroup and private addresses have no integer form on Metal.
    try expectRefused(a, "%u32", "%p_wg_u32 = OpTypePointer Workgroup %u32\n%c16_64 = OpConstant %u64 16", "%wp = OpConvertUToPtr %p_wg_u32 %c16_64\n%y = OpLoad %u32 %wp\nOpStore %pout_x %y", "OpConvertUToPtr of a Workgroup pointer: only device addresses");
    try expectRefused(a, "%u32", "%p_sb_u32 = OpTypePointer StorageBuffer %u32", "%wp = OpBitcast %p_sb_u32 %x\nOpStore %wp %x", "OpBitcast between a pointer and a 32-bit integer");
}

// ── Review round 2 (lowering): regressions ─────────────────────────────────

test "Volatile memory operands, variables and members give volatile loads and stores" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    // The memory operand (and the Vulkan memory model's form of it).
    const r = try lowerTextForTest(a, try kernelModule(a, "%u32", "", "%y = OpLoad %u32 %pin_x Volatile\n%z = OpLoad %u32 %pin_x Volatile|Aligned 4\nOpStore %pout_x %y Volatile\nOpStore %pout_x %z"), .{});
    try std.testing.expectEqualStrings("", r.message);
    try std.testing.expectEqual(2, std.mem.count(u8, r.ir, "= load volatile i32, ptr addrspace(1) %pin_x, align 4"));
    try expectHas(r.ir, "store volatile i32 %y, ptr addrspace(1) %pout_x, align 4");
    try expectHas(r.ir, "store i32 %z, ptr addrspace(1) %pout_x, align 4");
    // A plain access stays plain (the kernel's own `%x` load).
    try expectHas(r.ir, "%x = load i32, ptr addrspace(1) %pin_x, align 4");
    // OpCopyMemory: either memory operand set (the source's is the second).
    const cm = try lowerTextForTest(a, try kernelModule(a, "%u32", "", "OpCopyMemory %pout_x %pin_x None Volatile"), .{});
    try std.testing.expectEqualStrings("", cm.message);
    try expectHas(cm.ir, "= load volatile i32, ptr addrspace(1) %pin_x, align 4");
    try expectHas(cm.ir, "store volatile i32 %t1, ptr addrspace(1) %pout_x, align 4");
    // A variable decorated Volatile, through an access chain and a copy.
    const dv = try lowerTextForTest(a, try kernelModuleWith(a, "%u32", "OpDecorate %out Volatile", "", "%cp = OpCopyObject %p_elem %pout_x\nOpStore %cp %x"), .{});
    try std.testing.expectEqualStrings("", dv.message);
    try expectHas(dv.ir, "store volatile i32 %x, ptr addrspace(1) %pout_x, align 4");
    try expectHas(dv.ir, "%x = load i32, ptr addrspace(1) %pin_x, align 4");
    // glslang's `volatile buffer`: members decorated Volatile, plain
    // accesses; the GPU test polls such a flag.
    const g = try lowerTextForTest(a, try execText(a, "volatile_member_spin_wait"), .{});
    try std.testing.expectEqualStrings("", g.message);
    try std.testing.expectEqual(2, std.mem.count(u8, g.ir, "= load volatile i32"));
    try std.testing.expectEqual(3, std.mem.count(u8, g.ir, "store volatile i32"));
    try std.testing.expect(!has(g.ir, "= load i32"));
    const m = try lowerTextForTest(a, try execText(a, "volatile_spin_wait"), .{});
    try std.testing.expectEqualStrings("", m.message);
    try expectHas(m.ir, "store volatile i32 42, ptr addrspace(1) %a0, align 4");
    try expectHas(m.ir, "%v = load volatile i32, ptr addrspace(1) %a0, align 4");
    try expectHas(m.ir, "store i32 %v, ptr addrspace(1) %o1, align 4");
}

test "half sin, tan and asin return x where x is ±0 (strict only); f32 and fast math are unchanged" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const body = "%s = OpExtInst %f16 %glsl Sin %x\n%t = OpExtInst %f16 %glsl Tan %s\n%r = OpExtInst %f16 %glsl Asin %t\nOpStore %pout_x %r";
    const h = try lowerTextForTest(a, try kernelModule(a, "%f16", "%f16 = OpTypeFloat 16", body), .{});
    try std.testing.expectEqualStrings("", h.message);
    try std.testing.expectEqual(3, std.mem.count(u8, h.ir, "= fcmp oeq half "));
    try expectHas(h.ir, "= fcmp oeq half %x, 0xH0000");
    try expectHas(h.ir, "= select i1 %t2, half %x, half %t1");
    const fast = try lowerTextForTest(a, try kernelModule(a, "%f16", "%f16 = OpTypeFloat 16", body), .{ .fast_math = true });
    try std.testing.expect(!has(fast.ir, "fcmp oeq"));
    const f = try lowerTextForTest(a, try kernelModule(a, "%f32", "", "%s = OpExtInst %f32 %glsl Sin %x\n%r = OpExtInst %f32 %glsl Asin %s\nOpStore %pout_x %r"), .{});
    try std.testing.expect(!has(f.ir, "fcmp oeq"));
    const e = try lowerTextForTest(a, try execText(a, "half_signed_zero"), .{});
    try std.testing.expectEqualStrings("", e.message);
    try std.testing.expectEqual(3, std.mem.count(u8, e.ir, "= fcmp oeq half "));
}

test "OpenCL normalize and length of a half vector are computed in float" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const r = try lowerTextForTest(a, try execText(a, "half_geometric_subnormal"), .{});
    try std.testing.expectEqualStrings("", r.message);
    try std.testing.expectEqual(2, std.mem.count(u8, r.ir, "= fpext <2 x half> "));
    try expectHas(r.ir, "= fptrunc <2 x float> ");
    try expectHas(r.ir, "= fptrunc float ");
    try expectHas(r.ir, "call float @air.dot.v2f32(");
    try std.testing.expect(!has(r.ir, "@air.dot.v2f16"));
    // 2^86 for the small sums, 2^-70 for the overflowing ones.
    try expectHas(r.ir, "float 0x4550000000000000");
    try expectHas(r.ir, "float 0x3B90000000000000");
}

test "an access chain continued from a tight struct element indexes the tight form" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const r = try lowerTextForTest(a, try execText(a, "tight_chain_continued"), .{});
    try std.testing.expectEqualStrings("", r.message);
    // `&t[g].b` is member 2 of `<{ i16, [2 x i8], i32, i32, i32, [3 x float] }>`
    // (the padding is member 1), `&t[g].v` member 5.
    try expectHas(r.ir, "%q = getelementptr { [0 x <{ i16, [2 x i8], i32, i32, i32, [3 x float] }>] }, ptr addrspace(1) %a0, i64 0, i32 0, i32 %g, i32 2\n");
    try expectHas(r.ir, "ptr addrspace(1) %a0, i64 0, i32 0, i32 %g, i32 5\n");
    try expectHas(r.ir, "%qi = getelementptr { [0 x <{ float, [12 x i8], [3 x float] }>] }, ptr addrspace(1) %a1, i64 0, i32 0, i32 %g, i32 2\n");
    // The whole-I store through `&b[g].i` writes the tight form only.
    try expectHas(r.ir, "store <{ [3 x float] }> ");
}

test "entry points named like generated symbols or intrinsics" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const f20 = try lowerTextForTest(a, try execText(a, "f20"), .{});
    try std.testing.expectEqualStrings("", f20.message);
    try expectHas(f20.ir, "define void @f20(");
    try expectHas(f20.ir, "call void @f20.0(i32 %x, ptr addrspace(1) %a0)");
    try expectHas(f20.ir, "define private void @f20.0(");
    // A Workgroup variable's `@g<N>` beside a kernel of that name.
    const base = try kernelModule(a, "%u32", "%wa = OpTypeArray %u32 %c64\n%p_wa = OpTypePointer Workgroup %wa\n%pw_u32 = OpTypePointer Workgroup %u32\n%25 = OpVariable %p_wa Workgroup", "%pw = OpAccessChain %pw_u32 %25 %gid_x\nOpStore %pw %x\n%y = OpLoad %u32 %pw\nOpStore %pout_x %y");
    const g25 = try lowerTextForTest(a, try std.mem.replaceOwned(u8, a, base, "\"k\"", "\"g25\""), .{});
    try std.testing.expectEqualStrings("", g25.message);
    try expectHas(g25.ir, "@g25.0 = internal addrspace(3) global [64 x i32] undef");
    try expectHas(g25.ir, "define void @g25(");
    // An OpName that is the fallback's spelling is skipped too.
    const named = try lowerTextForTest(a, try std.mem.replaceOwned(u8, a, try std.mem.replaceOwned(u8, a, base, "\"k\"", "\"g25\""), "OpName %in \"in\"", "OpName %in \"in\"\nOpName %x \"g25.0\""), .{});
    try std.testing.expectEqualStrings("", named.message);
    try expectHas(named.ir, "@g25.1 = internal addrspace(3) global");
    for ([_][]const u8{ "air.wg.barrier", "llvm.sqrt.f32" }) |name| {
        const r = try lowerTextForTest(a, try std.mem.replaceOwned(u8, a, try kernelModule(a, "%u32", "", "OpStore %pout_x %x"), "\"k\"", try a.print("\"{s}\"", .{name})), .{});
        try std.testing.expect(has(r.message, "reserved for LLVM and Metal intrinsics"));
    }
}

test "a branch to the entry block and irreducible control flow are refused with the branch quoted" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const cb = "%cb = OpConstantTrue %bool";
    const cases = [_]struct { body: []const u8, quote: []const u8, why: []const u8 }{
        .{ .body = "OpBranchConditional %cb %entry %m\n%m = OpLabel\nOpStore %pout_x %x", .quote = "`OpBranchConditional %cb %", .why = "SPIR-V forbids any branch to the entry block" },
        .{ .body = "OpBranch %entry\n%m = OpLabel\nOpStore %pout_x %x", .quote = "`OpBranch %", .why = "SPIR-V forbids any branch to the entry block" },
        // Entered at A and at B (OpenCL C `goto` into a loop).
        .{ .body = "OpBranchConditional %cb %ba %bb\n%ba = OpLabel\nOpBranch %bb\n%bb = OpLabel\nOpBranchConditional %cb %ba %m\n%m = OpLabel\nOpStore %pout_x %x", .quote = "`OpBranchConditional %cb %", .why = "irreducible control flow" },
    };
    for (cases) |c| {
        const r = try lowerTextForTest(a, try kernelModule(a, "%u32", cb, c.body), .{});
        if (!has(r.message, c.quote) or !has(r.message, c.why)) {
            std.debug.print("expected {s}: ...{s}, got `{s}`\n", .{ c.quote, c.why, r.message });
            return error.TestUnexpectedResult;
        }
    }
    // The same loop entered at B only is a natural loop.
    const ok = try lowerTextForTest(a, try kernelModule(a, "%u32", cb, "OpBranch %bb\n%ba = OpLabel\nOpBranch %bb\n%bb = OpLabel\nOpBranchConditional %cb %ba %m\n%m = OpLabel\nOpStore %pout_x %x"), .{});
    try std.testing.expectEqualStrings("", ok.message);
}

test "Workgroup, Private and Function variables of runtime-sized types are refused" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const rta = "%rta = OpTypeRuntimeArray %u32\n%prta_w = OpTypePointer Workgroup %rta\n%pw = OpTypePointer Workgroup %u32\n%tile = OpVariable %prta_w Workgroup";
    const r1 = try lowerTextForTest(a, try kernelModule(a, "%u32", rta, "%t = OpAccessChain %pw %tile %gid_x\nOpStore %t %x"), .{});
    try std.testing.expect(has(r1.message, "Workgroup variable %") and has(r1.message, "has a runtime-sized type"));
    // A Workgroup Block struct ending in one (the aliased allocation).
    const blk = try kernelModuleWith(a, "%u32", "OpDecorate %WB Block\nOpMemberDecorate %WB 0 Offset 0\nOpMemberDecorate %WB 1 Offset 4\nOpDecorate %rta ArrayStride 4", "%rta = OpTypeRuntimeArray %u32\n%WB = OpTypeStruct %u32 %rta\n%pwb = OpTypePointer Workgroup %WB\n%pw = OpTypePointer Workgroup %u32\n%wb = OpVariable %pwb Workgroup", "%t = OpAccessChain %pw %wb %c0\nOpStore %t %x");
    const r2 = try lowerTextForTest(a, blk, .{});
    try std.testing.expect(has(r2.message, "has a runtime-sized type"));
    const fnv = try lowerTextForTest(a, try kernelModule(a, "%u32", "%rta = OpTypeRuntimeArray %u32\n%prta_f = OpTypePointer Function %rta", "OpStore %pout_x %x"), .{});
    try std.testing.expectEqualStrings("", fnv.message);
    const fr = try lowerTextForTest(a, try std.mem.replaceOwned(u8, a, try kernelModule(a, "%u32", "%rta = OpTypeRuntimeArray %u32\n%prta_f = OpTypePointer Function %rta", "OpStore %pout_x %x"), "%x = OpLoad", "%lv = OpVariable %prta_f Function\n%x = OpLoad"), .{});
    try std.testing.expect(has(fr.message, "`%lv = OpVariable %") and has(fr.message, "Function variable %") and has(fr.message, "runtime-sized"));
    // A fixed-size Workgroup array is fine.
    const ok = try lowerTextForTest(a, try kernelModule(a, "%u32", "%wa = OpTypeArray %u32 %c64\n%p_wa = OpTypePointer Workgroup %wa\n%pw = OpTypePointer Workgroup %u32\n%tile = OpVariable %p_wa Workgroup", "%t = OpAccessChain %pw %tile %gid_x\nOpStore %t %x"), .{});
    try std.testing.expectEqualStrings("", ok.message);
}

/// K nested loops whose innermost back edge carries a pointer into a member
/// at Offset 1 out through every loop's phi (find-robustness-1's
/// gen_rounds.py).
fn nestedPhiModule(a: Allocator, k: usize) ![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    try out.appendSlice(a,
        \\OpCapability Shader
        \\OpCapability Int8
        \\OpCapability StorageBuffer8BitAccess
        \\OpExtension "SPV_KHR_8bit_storage"
        \\OpMemoryModel Logical GLSL450
        \\OpEntryPoint GLCompute %k "k" %gid
        \\OpExecutionMode %k LocalSize 1 1 1
        \\OpDecorate %gid BuiltIn GlobalInvocationId
        \\OpDecorate %buf Binding 0
        \\OpDecorate %buf DescriptorSet 0
        \\OpDecorate %blk Block
        \\OpMemberDecorate %blk 0 Offset 0
        \\OpDecorate %arr ArrayStride 4
        \\OpDecorate %pk Binding 1
        \\OpDecorate %pk DescriptorSet 0
        \\OpDecorate %ps Block
        \\OpMemberDecorate %ps 0 Offset 0
        \\OpMemberDecorate %ps 1 Offset 1
        \\%void = OpTypeVoid
        \\%fn = OpTypeFunction %void
        \\%u32 = OpTypeInt 32 0
        \\%u8 = OpTypeInt 8 0
        \\%bool = OpTypeBool
        \\%v3 = OpTypeVector %u32 3
        \\%pv3 = OpTypePointer Input %v3
        \\%gid = OpVariable %pv3 Input
        \\%c0 = OpConstant %u32 0
        \\%c1 = OpConstant %u32 1
        \\%c2 = OpConstant %u32 2
        \\%arr = OpTypeRuntimeArray %u32
        \\%blk = OpTypeStruct %arr
        \\%pblk = OpTypePointer StorageBuffer %blk
        \\%pu32 = OpTypePointer StorageBuffer %u32
        \\%buf = OpVariable %pblk StorageBuffer
        \\%ps = OpTypeStruct %u8 %u32
        \\%pps = OpTypePointer StorageBuffer %ps
        \\%pk = OpVariable %pps StorageBuffer
        \\%k = OpFunction %void None %fn
        \\%entry = OpLabel
        \\%g = OpLoad %v3 %gid
        \\%x = OpCompositeExtract %u32 %g 0
        \\%pa = OpAccessChain %pu32 %buf %c0 %x
        \\%cond = OpULessThan %bool %x %c2
        \\OpBranch %h1
        \\
    );
    for (1..k + 1) |i| {
        try out.print(a, "%h{d} = OpLabel\n%ph{d} = OpPhi %pu32 ", .{ i, i });
        if (i == 1) try out.appendSlice(a, "%pa %entry ") else try out.print(a, "%ph{d} %h{d} ", .{ i - 1, i - 1 });
        if (i == k) try out.print(a, "%narrow %l{d}\n", .{k}) else try out.print(a, "%ph{d} %l{d}\n", .{ i + 1, i });
        const t = if (i == k) try a.print("%l{d}", .{k}) else try a.print("%h{d}", .{i + 1});
        const f = if (i > 1) try a.print("%l{d}", .{i - 1}) else "%exit";
        try out.print(a, "OpBranchConditional %cond {s} {s}\n", .{ t, f });
    }
    try out.print(a, "%l{d} = OpLabel\n%narrow = OpAccessChain %pu32 %pk %c1\nOpBranch %h{d}\n", .{ k, k });
    var i = k - 1;
    while (i > 0) : (i -= 1) try out.print(a, "%l{d} = OpLabel\nOpBranch %h{d}\n", .{ i, i });
    try out.appendSlice(a, "%exit = OpLabel\nOpStore %ph1 %x\nOpReturn\nOpFunctionEnd\n");
    return out.items;
}

test "pointer phis nested K loops deep settle in two lowering rounds" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_]usize{ 1, 3, 50 }) |k| {
        const bytes = try asm_.assemble(a, try nestedPhiModule(a, k), null);
        var m = try reader.read(a, bytes, null);
        var diag: Diagnostics = .{};
        const out = lower(a, &m, .{}, &diag) catch |err| {
            std.debug.print("K = {d}: {t} {s}\n", .{ k, err, diag.message });
            return err;
        };
        // The store through the outermost phi claims the Offset's 1.
        try expectHas(out.ir, "store i32 %v");
        try expectHas(out.ir, ", align 1\n");
        try std.testing.expect(!has(out.ir, "store i32 %v19, ptr addrspace(1) %v20, align 4"));
        try std.testing.expectEqual(2, diag.rounds);
    }
    const r = try lowerTextForTest(a, try nestedPhiModule(a, 3), .{});
    try std.testing.expectEqualStrings("", r.message);
}
