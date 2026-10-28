# SPIR-V compute shaders on Metal

`spirv-splice` turns a SPIR-V compute module into a Metal library with this
project's own assembler and packer, no Apple tooling involved. It is the
SPIR-V front end of [spirv-plan.md](spirv-plan.md); stages 1-4 (compute,
scalars and vectors, control flow, the interface, threadgroup memory,
barriers and atomics) are done, stage 5 (graphics: vertex and fragment entry
points, images, samplers) is not. The crosscheck bench runs through it
([below](#the-crosscheck-bench-through-spir-v)): Zig's own SPIR-V backend
compiles the random programs, spirv-splice builds the library, and the GPU
must agree with the CPU. `zig build spirv-corpus` builds every kernel of the
corpus (engine output from naga and Zig; hand-written modules in glslang's and DXC's
shapes, since neither tool is installed here) and runs the ones with known answers
([below](#the-corpus)).

```
module.spv ──tools/spirv/reader.zig──▶ Module
           ──tools/spirv/lower.zig───▶ LLVM IR text + runtime manifest (tools/spirv/manifest.zig)
           ──tools/air/assembler.zig──▶ bitcode ──tools/air/metallib.zig──▶ .metallib
```

Everything after the text is the Zig path's code ([pipeline.md](pipeline.md)), so the
intrinsic table, the uniformity analysis and the Metal workarounds apply to SPIR-V
shaders unchanged.

## What is supported

| area | supported | verified by |
| --- | --- | --- |
| execution models | GLCompute (Vulkan, `Logical` / `PhysicalStorageBuffer64` GLSL450) and Kernel (OpenCL, `Physical64`) | corpus, probes |
| interface | StorageBuffer / BufferBlock SSBOs, Uniform and PushConstant blocks, OpenCL pointer and value parameters, `__constant` tables, Workgroup variables (Block ones aliased, SPV_KHR_workgroup_memory_explicit_layout), Private variables, the compute builtins, PhysicalStorageBuffer pointers stored in buffers or converted from integers (GL_EXT_buffer_reference) | [The interface](#the-interface), exec tests |
| types | bool, 8-64-bit integers, half, float, vectors of 2, 3, 4, 8 and 16 lanes, matrices (as storage: loaded, stored, built and taken apart; no matrix arithmetic, below), arrays, runtime arrays, structs with any Offset layout AIR can express: std430 and scalar block layout, and std140 except where its 16-byte rounding leaves gaps AIR has no form for (next table) | [Types and layout](#types-and-layout), probes |
| instructions | the arithmetic, bit, comparison, conversion, composite, memory and control-flow instructions of the core grammar listed in [Instructions](#instructions), which are all of those sections except the ones in the next table; GLSL.std.450 and OpenCL.std math | [Instructions](#instructions), 703 probes run against a CPU reference |
| synchronisation | OpControlBarrier (Workgroup and Subgroup execution scope), OpMemoryBarrier, with the semantics' memory flags | [Barriers and atomics](#barriers-and-atomics), probes, known-answer kernels |
| atomics | every OpAtomic* except OpAtomicFlagTestAndSet / OpAtomicFlagClear (next table) on 32-bit integers in device and threadgroup memory with every memory order; 32-bit float load, store, exchange (device and threadgroup) and add (device); 64-bit load, store and device min/max | [Barriers and atomics](#barriers-and-atomics), probes, known-answer kernels |

| not supported | why |
| --- | --- |
| Vertex, Fragment and the other graphics execution models; Input / Output location variables; OpKill and the fragment instructions | stage 5 (graphics) |
| images, samplers, OpImageTexelPointer (image atomics) | stage 5 |
| OpArrayLength | Metal passes a kernel no buffer lengths; pass the length in a uniform |
| DescriptorSet > 0 | one flat `[[buffer(n)]]` space; flattening sets is a host-side policy not chosen yet |
| descriptor arrays (an array of Block structs) | an AIR buffer parameter is one pointer; one variable per binding is needed |
| double | Metal has no 64-bit float |
| std140 arrays whose ArrayStride exceeds the element's AIR size: scalars, `vec2`s and structs whose AIR size is not a multiple of 16 (no `vec3` / `vec4` member to align them) at std140's rounded-up stride (`uniform U { float w[4]; }`, `ivec2 v[8]`, `struct { float a, b; } s[4]` at 16), and std140 matrices with 2-component columns at MatrixStride 16 (`mat2`, `mat3x2`); std140 `vec3` / `vec4` arrays and `mat3` / `mat4` match AIR and are supported | an AIR array or matrix is dense, so the 12 or 8 bytes after each element have no form there; refused by name ("array type %N has ArrayStride 16 but its element occupies 4 bytes in AIR"); declare such members std430-style or as `vec4` (test "std140 strides AIR cannot express ..." in tools/spirv/lower.zig) |
| 64-bit atomic add, sub, and, or, xor, exchange, increment, decrement, compare-exchange; 64-bit min/max whose result is used or in threadgroup memory | MTLCompilerService crashes (MSL's only 64-bit atomics are `atomic_ulong` min/max returning void, load and store) |
| OpAtomicFMinEXT / FMaxEXT | Metal's bitcode reader does not know `atomicrmw fmin/fmax`; MSL has no float atomic min/max |
| atomics on Function / Private memory | Metal has atomics on device and threadgroup memory only |
| execution scopes wider than Workgroup | Metal synchronises a threadgroup or a SIMD-group, nothing wider |
| Workgroup variable initializers (zero-initialized workgroup memory) and Workgroup entry parameters (OpenCL `local` arguments) | threadgroup memory has no static initializer on Metal, and dynamically sized threadgroup arguments (`[[threadgroup(n)]]`) are not implemented |
| matrix arithmetic: OpMatrixTimesVector, OpVectorTimesMatrix, OpMatrixTimesScalar, OpMatrixTimesMatrix, OpOuterProduct, OpTranspose (GLSL/HLSL `m * v`, `transpose(m)`) | not mapped yet; a module that multiplies by a matrix is refused (a matrix in a buffer can still be loaded, stored and taken apart by column) |
| OpCopyLogical (SPIR-V 1.4+: glslang for Vulkan 1.2/1.3 emits it to copy a struct or array between a buffer's laid-out type and a local type, `Particle p = buf.particles[i];`) | not mapped yet |
| OpQuantizeToF16, OpSatConvertSToU, OpSatConvertUToS (saturating signed-unsigned conversions), OpIsNormal, OpSignBitSet, OpSDot / OpUDot / OpSUDot and their AccSat forms (integer dot products, core since SPIR-V 1.6), OpPtrDiff, OpCopyMemorySized, OpGenericCastToPtrExplicit, OpGenericPtrMemSemantics, OpSizeOf, OpAtomicFlagTestAndSet / OpAtomicFlagClear, the named barriers | not mapped (each would need a probe and a reference); saturating conversions through the SaturatedConversion decoration are mapped ([Instructions](#instructions)) |
| subgroup and group operations (OpGroupNonUniform*, OpGroup*, OpSubgroup*KHR) | not mapped yet (MSL's `simd_*` functions would be their counterparts); the Subgroup barrier and the subgroup builtins are supported |
| GLSL.std.450 Frexp, Modf, Pack/Unpack, Interpolate*, Determinant, MatrixInverse, FaceForward, Reflect, Refract; the OpenCL.std functions not mapped | not mapped (each would need a probe and a reference) |
| OpSpecConstantOp; specialization at load time | spec constants are lowered at their default values |
| RoundingModeRTZ, DenormPreserve, DenormFlushToZero, SignedZeroInfNanPreserve | Metal gives a kernel no control over rounding and denormals |

Every unsupported construct is refused by name with the offending instruction, never
lowered approximately.

The reader decodes every enumerant of the SPIR-V 1.6 rev 7 grammar, vendor ones
included: the vendored tables (tools/spirv/spec.zig, generated by Zig's
gen_spirv_spec.zig) lack about 380 of them, among them DXC's `UserTypeGOOGLE`, which
`-fspv-reflect` puts on every resource, and the Sampled1D / Image1D capabilities;
tools/spirv/spec_extra.zig supplies them, so such a module decodes and its reflection
decorations are ignored like `UserSemantic` (corpus hand-dxc-reflect, executed). An
enumerant in no grammar is accepted when no words follow it and otherwise refused by
name ("Decoration enumerant N is not in the SPIR-V grammar tables"). Opcodes Zig's
tables lack (vendor instructions, OpCooperativeMatrix*EXT) are refused by the reader as
unknown.

## Usage

```
zig build spirv-splice        # installs zig-out/bin/spirv-splice, spirv-probe, spirv-probe-run, metallib-check
zig-out/bin/spirv-splice [--target=<profile>] [--allow-unverified] [--fast-math=<bool>]
                         [--push-constant-index=<n>] [--manifest=<out.txt>] [--dis]
                         [--zig-workarounds=<bool>]
                         <in.spv|in.spvasm> <out.metallib> [out.ll]
zig-out/bin/metallib-check --kernels=<a,b,...> <out.metallib>
```

The options follow air-splice's rules: options before the files, a `-` argument
among the files is an error, unverified deployment targets are refused unless
`--allow-unverified`. `--fast-math` declares fast math and picks Apple's `air.fast_*`
functions, as for the Zig path. `--push-constant-index` is the `[[buffer(n)]]` of the
PushConstant block (default 30). `--manifest` writes the interface a host needs to
bind buffers; `--dis` prints the module first. `--zig-workarounds=false` lowers a
module from Zig's SPIR-V backend by the specification alone, without undoing that
backend's known conversion bugs ([below](#a-bug-in-zigs-spir-v-backend)). A text input (no SPIR-V magic) is
assembled with tools/spirv/asm.zig. `out.ll` is the IR plus metadata, which
`xcrun metal -Xclang -opaque-pointers out.ll -o apple.metallib` accepts: the
attribution path (our assembler and packer against Apple's, on the same text). It
spells the intrinsics as the assembler calls them (`air.sin.f32`, `air.fast_pow.f32`,
`air.ldexp.f32`, ...) and carries the assembler's `convergent` attributes, but keeps
`frem` and `llvm.round`, which the assembler rewrites around Metal findings
([below](#the-zig-shape-modules-and-apples-assembler)).

Every GLCompute or Kernel entry point becomes one function of the library, under its
OpEntryPoint name (one starting with `llvm.` or `air.`, the intrinsics' prefixes, is
refused; a helper or threadgroup global whose generated symbol an entry point already
has takes the next free `.N` suffix, `@f20.0` beside a kernel named `f20`, exec test
f20); the workgroup size is recorded in the manifest, not in the library
(the host dispatches with it). Refusals name the entry point and quote the instruction:

```
spirv-splice: k.spvasm: entry point 'k': `%83 = OpAtomicIAdd %3 %36 %57 %35 %41`: 64-bit OpAtomicIAdd: Metal has 64-bit atomic load, store, min and max only (MTLCompilerService crashes on the others)
```

Execution modes:

| mode | here |
| --- | --- |
| LocalSize, LocalSizeId | the manifest's workgroup size (each component a non-zero constant that fits 32 bits) |
| a constant decorated BuiltIn WorkgroupSize | takes precedence over both (SPIR-V 1.6 section 3.21; glslang's `local_size_x_id` emits `LocalSize 1 1 1` plus such a specialization constant), at its default value (exec test workgroup_size_constant dispatches with it) |
| LocalSizeHint, LocalSizeHintId, VecTypeHint | hints, ignored |
| ContractionOff, FPFastMathDefault | fast-math restrictions and permissions: the library-level `--fast-math` decides, as for the NoContraction / FPFastMathMode decorations |
| RoundingModeRTE 16 / 32 | what Metal does (the probes check arithmetic and conversions against round-to-nearest-even references) |
| RoundingModeRTZ, DenormPreserve, DenormFlushToZero, SignedZeroInfNanPreserve | refused by name: Metal has no per-kernel rounding or denormal control, and a fast-math library may drop signed zeros, infinities and NaNs |
| anything else | refused by name |

### The manifest

```
# spirv-splice manifest 1
kernel crosscheckKernel local_size 64 1 1 threadgroup_bytes 0
  buffer 0 in0 device read size 16384 align 4 type struct135 binding 0 set 0
    field 0 4 4096 int m0
  buffer 1 in1 device read size 16384 align 4 type struct135 binding 1 set 0
    field 0 4 4096 int m0
  buffer 2 out device read_write size 16384 align 4 type struct135 binding 2 set 0
    field 0 4 4096 int m0
  buffer 3 params constant read size 4 align 4 type struct131 binding 3 set 0
    field 0 4 0 uint m0
  builtin thread_position_in_grid global_invocation_id uint3
end
```

(zig-crosscheck.spvasm.) A `buffer` line is `[[buffer(index)]]`, its name, AIR space
(`device` 1, `constant` 2), access, the pointee's size and alignment, the reflection
type name, and where it comes from: `binding B set 0`, `push_constant`, `parameter N`
(an entry-function parameter; `by_value` when the shader reads the value from that
constant buffer) or `unbound` (a buffer variable without a Binding decoration, given
the lowest free index). `field` lines are the `air.struct_type_info` entries.

## The interface

| SPIR-V | AIR parameter | metadata |
| --- | --- | --- |
| StorageBuffer variable (Binding B, DescriptorSet 0) | `ptr addrspace(1)`, `[[buffer(B)]]` | `air.buffer`, `air.read` when the variable (or every member of its Block) is NonWritable, else `air.read_write` |
| Uniform variable of a **BufferBlock** struct (SPIR-V 1.0-1.2 SSBO, glslang for Vulkan 1.0) | `ptr addrspace(1)`, `[[buffer(B)]]`; every pointer derived from it is device memory although its type says Uniform | as StorageBuffer |
| Uniform / UniformConstant variable | `ptr addrspace(2)`, `[[buffer(B)]]` | `air.buffer`, `air.read`, address space 2 |
| UniformConstant variable with an initializer (OpenCL program-scope `__constant`) | `@x = private addrspace(2) constant T <init>`, no parameter | none |
| PushConstant variable | `ptr addrspace(2)`, `[[buffer(--push-constant-index)]]` | as Uniform |
| entry parameter: StorageBuffer/PhysicalStorageBuffer/CrossWorkgroup/Generic pointer | `ptr addrspace(1)`, `[[buffer(position)]]` | `air.read_write` unless NonWritable |
| entry parameter: a value | `ptr addrspace(2)` loaded in the prologue | `air.read`, the value's type |
| Input variable, BuiltIn | `i32` / `<N x i32>` | `air.<kind>`, `uint` / `uintN` |
| Workgroup variable | `@x = internal addrspace(3) global T undef` | none (static threadgroup memory) |
| Workgroup variables of Block structs (SPV_KHR_workgroup_memory_explicit_layout, GLSL `shared` blocks) | one `@x = internal addrspace(3) global [N x i8] undef` for all of them, N the largest (they alias: that is how a shader puns shared memory between types); `threadgroup_bytes` counts it once | none (exec test workgroup_block_alias) |
| a PhysicalStorageBuffer pointer stored in a block (GL_EXT_buffer_reference) | `ptr addrspace(1)` loaded from the block; the field is an 8-byte `ulong` in the manifest, which the host fills with an MTLBuffer's `gpuAddress` (and makes that buffer resident with `useResource`) | the block's field (exec test buffer_reference) |
| Private variable | an `alloca` in the entry, passed to helpers | none |

A fixed-size Block is described like MSL's `device B&` (`air.buffer_size`, then
`air.struct_type_info` with an array member's element size and length, Apple's
`!{i32 0, i32 4, i32 4096, !"int", !"data"}`); a Block ending in a runtime array like
`device R*` (no `air.buffer_size`, count 1). Those shapes come from Apple's own output
for the MSL twin.

Builtins: GlobalInvocationId → `thread_position_in_grid`, LocalInvocationId →
`thread_position_in_threadgroup`, WorkgroupId → `threadgroup_position_in_grid`,
WorkgroupSize → `threads_per_threadgroup`, NumWorkgroups → `threadgroups_per_grid`,
LocalInvocationIndex → `thread_index_in_threadgroup`, SubgroupLocalInvocationId →
`thread_index_in_simdgroup`, SubgroupId → `simdgroup_index_in_threadgroup`,
SubgroupSize/SubgroupMaxSize → `threads_per_simdgroup`, NumSubgroups →
`simdgroups_per_threadgroup`, GlobalSize → `threads_per_grid`. A 64-bit builtin
(OpenCL's `<3 x u64>` ids) is an `i32` parameter zero-extended at the load. Parameter
order: buffers by index, the push constant block, builtins in interface order.

Refused: DescriptorSet > 0, descriptor arrays (an array of Block structs: one AIR
buffer parameter cannot be N descriptors), an initializer on a StorageBuffer, Uniform
or PushConstant variable, a store into constant memory (Metal builds it and drops the
store), two buffers at one index, an index above 30, Output and location Input
variables, images and samplers (stage 5), module-scope CrossWorkgroup variables (Metal
has no program-scope device memory), Workgroup pointer parameters of the entry point
(OpenCL `local` arguments: Metal sizes such memory per dispatch through
`[[threadgroup(n)]]` arguments, not implemented) and Workgroup initializers (Metal
threadgroup memory has no static initializer), and Workgroup, Private and Function
variables of a runtime-sized type (an OpTypeRuntimeArray, or a struct ending in one:
SPIR-V allows one only in a buffer block, whose size the host's buffer gives; such a
Workgroup array used to become a 0-byte threadgroup global, every access out of
bounds). A helper function may take Workgroup pointers (`ptr addrspace(3)`).

A pointer into a BufferBlock variable is device memory although its SPIR-V type says
Uniform, so it is refused wherever that type would be printed for it: stored into
memory (VariablePointers: the variable's type, and every later load, would say
constant memory), as a Function variable's initializer, returned, put into a
composite, passed to a helper whose parameter type is the Uniform one, or in a phi
together with a real Uniform pointer. It may be loaded from, stored to, indexed,
copied, bitcast within its class and selected; OpPtrEqual / OpPtrNotEqual need both
operands in one address space (OpPtrDiff is refused whatever its operands).

## Types and layout

Bool is `i1` in registers and `i8` in memory (Metal has no `i1` memory type): a bool
load compares with 0, a store zero-extends, a bool inside a struct or array is an `i8`
there. Integers 8-64 bits, half and float; double is refused (Metal has none). Vectors
are `<N x T>`, matrices arrays of column vectors, arrays `[N x T]`, runtime arrays
`[0 x T]`. A struct is a literal struct when AIR's natural layout puts every member at
its Offset decoration, else a packed `<{ ... }>` with `[N x i8]` padding and remapped
member indices. AIR allocates a 3-element vector like a 4-element one, so in a packed
struct a `vec3` member is a `[3 x T]` (12 bytes: std430/scalar `vec3` followed by a
`float` at Offset 12), and an array whose ArrayStride is three components (scalar
layout `vec3[]`) is `[N x [3 x T]]`; the vector itself is still loaded and stored as
`<3 x T>`, whose store size is the three components, so a neighbour is never written
(exec test vec3_layouts: whole-struct and member stores, adjacent threads). A member
inside the components (Offset 8 after a vec3) is refused with the offsets. Otherwise
ArrayStride must equal the element's AIR size and MatrixStride the column's, so
std140's rounded-up stride for an array of scalars, `vec2`s or structs whose AIR size is
not a multiple of 16 (`float w[4]`, `struct { float a, b; } s[4]`) and its 16-byte MatrixStride for `vec2` columns (`mat2`) are
refused (std140 `vec3` and `vec4` arrays, `mat3` and `mat4` are AIR's own layout); RowMajor
is refused, and so is an Offset-packed struct whose ArrayStride runs past its last
member (std140 `{ vec3 @0; float @12; float @16 }` at stride 32: the element would
need tail padding). LLVM gives a packed `<{ ... }>` ABI alignment 1, so a struct with
natural-looking Offsets that contains a packed one is packed too, with explicit
padding (exec test packed_in_natural: std430 `{ uint count; Particle ps[]; }`, std140
`{ uint count; Light l; }`). Scalar block layout also packs a struct after a member
struct that ends in a vec3 (`struct Inner { vec3 v; }; buffer B { Inner i; float f; }`
puts f at Offset 12) and strides an array of such structs at 12 (`P p[]` for `P { vec3
v; }`), where AIR would give Inner 16 bytes. Such a member or element is stored in its
tight form: a packed struct without tail padding whose last member is tight in turn
(`<{ [3 x float] }>`, 12 bytes), chosen only where the next member or the ArrayStride
needs it (std430 keeps `{ <3 x float> }`). A load or store through a pointer to it reads
or writes the tight form and converts member by member, so the neighbour is never
written; such a pointer may be loaded from, stored to, copied (OpCopyObject,
OpCopyMemory) and indexed into, and is refused anywhere it would be accessed at the
full AIR size (passed to a helper, selected, a phi, OpPtrAccessChain) (exec test
scalar_tight_structs: whole-element loads and stores by adjacent threads, a whole-Inner
store next to f). An OpenCL `CPacked` struct has no padding at all and
alignment 1 (exec test cpacked_struct: 5-byte `{ uchar; uint }` elements, unaligned
loads); an Offset that contradicts CPacked is refused, and so is any other decoration
on a type or member that would change its layout or meaning (GLSLShared, GLSLPacked,
a member Location, ...), by name.

Alignment: a load or store says what its address is known to be aligned to, never
more. An access chain into a packed member knows it from the Offsets and strides on
the way (`&ps[g].v` of a scalar-layout `{ uint @0; uvec4 @4 }` at stride 20 is
4-aligned, not uvec4's 16; an `align 16` there would be LLVM undefined behaviour).
That knowledge follows the pointer through OpCopyObject, pointer bitcasts and OpSelect
(the less aligned operand), a phi (its least aligned incoming value), a helper's
pointer parameter (its least aligned argument), and a pointer stored to and loaded
back from memory or returned by a call (the least alignment of any narrowed pointer to
that pointee in the module). Phis and parameters can be lowered before the values
that narrow them (a loop's back edge, a caller lowered after its callee), so the
module is lowered again with what a round learned until nothing narrows further
(`lower`, `Learned`). What a round learns is used in the same round, and helpers are
lowered after the caller that first reaches them, so a fact crosses a whole chain of
helper calls or forward phis at once: a module without such pointers is lowered once,
and one with them twice, however deep the chain (a 70-call chain used to take one round
per call and was refused after 64; exec test packed_pointer_call_chain). Within a
function the pointer phis are solved together, from every access chain, copy, select
and phi of the function at once (`Fn.settleAlign`), so nested loops settle in the same
second round: K nested loops whose innermost back edge carried a packed member's
pointer out through every loop's phi took K + 1 rounds, each lowering the whole module
again into the same arena (K = 800: 31.8 s and 1.66 GB; now 0.2 s and 27 MB; test
"pointer phis nested K loops deep settle in two lowering rounds"). Exec test
packed_pointer_flow runs all four paths. An atomic through an address known to be under-aligned (a `uint`
at Offset 2) is refused.

## Instructions

| SPIR-V | LLVM IR |
| --- | --- |
| IAdd ISub IMul SDiv UDiv SRem UMod, bitwise, shifts | the same instruction; a shift amount of another width is zero-extended or truncated |
| SMod | `srem`, then `+ b` where the remainder is non-zero and its sign differs from b's |
| FAdd FSub FMul FDiv FRem FNegate | `fadd` ... `frem` (the assembler calls the exact `air.fmod`), `fneg` |
| FMod | `frem` + `b` where non-zero and the signs differ |
| SNegate / Not / LogicalNot | `sub 0, x` / `xor -1` / `xor true` |
| comparisons, Ordered/Unordered, LessOrGreater | `icmp` / `fcmp` (FOrd → `o*`, FUnord → `u*`) |
| IsNan / IsInf / IsFinite | `fcmp uno x, x` / `fabs` + `oeq inf` / `fabs` + `one inf` |
| Select | `select` (scalar or vector condition); a struct, array or matrix member by member (extractvalue, select, insertvalue) |
| U/S/FConvert, ConvertFToS/FToU/SToF/UToF | `zext/sext/trunc`, `fpext/fptrunc`, `fpto*i`, `*itofp` (same width: no instruction) |
| ... with SaturatedConversion | integer narrowing clamps first (`icmp` + `select`); float to integer maps NaN to 0 and clamps out-of-range values to the result's range |
| ConvertUToPtr / ConvertPtrToU, Bitcast between a pointer and an `i64` or `<2 x i32>` | `inttoptr` / `ptrtoint` (a vector bitcast to `i64` first) for device pointers only (PhysicalStorageBuffer, StorageBuffer, CrossWorkgroup, Generic): GL_EXT_buffer_reference's `Data(addr)` and uvec2 addresses, OpenCL integer casts (exec test buffer_reference) |
| ... with FPRoundingMode | float to integer: RTE/RTP/RTN round first (`llvm.rint/ceil/floor`), RTZ is `fpto*i`; a narrowing OpFConvert or an inexact integer to float: RTE only (Metal rounds to nearest even), other modes refused |
| Bitcast, PtrCastToGeneric, GenericCastToPtr | `bitcast`; pointer casts within one address space vanish, others are refused |
| CompositeExtract / Insert / Construct | `extractvalue`/`insertvalue` and `extractelement`/`insertelement`, bools converted at the aggregate boundary |
| VectorShuffle | `shufflevector` (0xFFFFFFFF → `undef` lane); operands of different lengths lane by lane |
| VectorExtractDynamic / InsertDynamic / VectorTimesScalar | `extractelement` / `insertelement` / `fmul` by a splat |
| Dot / Any / All | `air.dot.vNfT` / `air.any.vNi1` / `air.all.vNi1` |
| IAddCarry / ISubBorrow | `add`/`sub` + `icmp ult` (the `llvm.*.with.overflow` form crashes Metal on vectors) |
| UMulExtended / SMulExtended | widen, multiply, split; 64-bit operands from 32-bit halves (`mul i128` crashes Metal) |
| BitCount / BitReverse | `llvm.ctpop` (resized to the result width) / `llvm.bitreverse` |
| BitFieldUExtract / SExtract / Insert | shift and mask expansions, exact for every count including 0 and the full width; Count 0 selects the final value (0, or the base for Insert), because at Offset = width, which SPIR-V defines, the shift is poison and LLVM folds a masked poison away (exec test bitfield_full_offset: with constant operands Metal dropped the store) |
| Variable (Function) / Load / Store / CopyMemory | `alloca` / `load` / `store` with `Aligned`, else what the pointer is known to be aligned to ([Types and layout](#types-and-layout)), else the pointee's alignment; `load volatile` / `store volatile` for a Volatile access (below) |
| AccessChain, InBounds, Ptr, InBoundsPtr | `getelementptr [inbounds]`; a vector level steps by whole elements |
| PtrEqual / PtrNotEqual | `icmp eq/ne` |
| Phi, Branch, BranchConditional, Switch, Return, ReturnValue, Unreachable, FunctionCall | the same; a phi repeats its `[value, %parent]` pair once per CFG edge from that parent (two switch cases on one target, equal branch targets); SelectionMerge, LoopMerge, Line, Nop dropped |
| ExtInst of a `NonSemantic.*` set (debug info, debug printf) | dropped (SPV_KHR_non_semantic_info); the variables it names do not become parameters |

Access chains are one GEP from the variable: a chain built on another chain (or on an
OpCopyObject of one) continues its root's GEP. A pointer cast to another pointee type
(OpBitcast of OpenCL pointer casts, Zig's `@ptrCast(&buf.data)[i]`) starts a new GEP
from the cast address, typed by the new pointee (exec test bitcast_access_chain; it
used to continue the root's GEP with the new indices, reading word 1 of `[16 x i32]`
for element 1 of `[4 x <4 x i32>]`). A chain continued from a pointer to a struct held
in its tight form ([Types and layout](#types-and-layout)) indexes the tight form's
members, which its padding members renumber, and keeps the tight form for a whole-struct
access (exec test tight_chain_continued: `p = &t[g]; p->b = x` for scalar-layout `T {
uint16_t a; uint b; ...; vec3 v; }` stored x into `a`, the GEP's member 1 being the
padding after it, and `(&p->v)->y` landed on `v.x`). An aggregate load or store through a runtime-indexed struct or array
element of private memory (Function/Private) is split into one GEP and one load or
store per scalar or vector member, as Apple's front end emits for the MSL twin: the
aggregate form crashes MTLCompilerService (below).

Decorations on an instruction result: RelaxedPrecision, NoContraction, FPFastMathMode,
NoSignedWrap/NoUnsignedWrap, NonUniform, Coherent and the aliasing and alignment hints
change nothing the lowering computes and are ignored; SaturatedConversion and
FPRoundingMode are implemented on the conversions as above; any other decoration on a
result (Invariant, ...) is refused by name. Volatile is implemented: a load or store
with a Volatile memory operand (OpLoad, OpStore, either memory operand set of
OpCopyMemory; the Vulkan memory model's `Volatile|MakePointerVisible|NonPrivatePointer`
too), or through a pointer derived from a variable or a struct member decorated
Volatile (glslang's `volatile buffer` decorates every member; `Lowerer.findVolatile`
follows access chains, copies, casts, selects, phis and helper arguments), or of a
struct holding such a member, is a `load volatile` / `store volatile`. SPIR-V: a
Volatile access "cannot be eliminated, duplicated, or combined with other accesses";
with plain loads Metal hoisted the read out of a polling loop, so a spin-wait on a flag
another SIMD-group sets polled its cap and saw 0 (exec tests volatile_spin_wait,
volatile_member_spin_wait; Apple's assembler on the same plain text does the same, and
on the text with `volatile` added computes right). Malformed operand types (an IAdd of a
struct, a float index, a bitcast between sizes) are refused with the instruction quoted,
before the assembler sees any IR, and so is a result type that disagrees with what an
instruction aliases or indexes (`Fn.checkTypes`, `Fn.needPointerTo`): an
OpCompositeInsert whose result type is not its composite's, an access chain whose
result is not a pointer, or points into another storage class or to another type than
the one its indexes reach (with no indexes it used to alias the base pointer, which an
`add` then took as an integer), and an OpCopyObject of another type. So is a function body that breaks SPIR-V's validity rules the IR rests on
(`Fn.validateBody`): a branch or switch target that is not a block of the function or
is its first block (SPIR-V 2.16.1; packed, it crashed MTLCompilerService), irreducible
control flow (a cycle entered at two blocks, which a Kernel module may have from OpenCL
C `goto`: MTLCompilerService crashes on it, also on Apple's assembly of the same IR,
while the same loop entered once builds and computes right), a
repeated switch case literal, a phi without exactly one value per predecessor or with
a value of another type, a value of another function, the result of an instruction
that computes at module scope (an OpExtInst there: the reader admits it by its opcode
class, SPIR-V allows no computation outside a function, and the lowering used to print a
use of an undefined IR value; `Lowerer.moduleNonValues`), or a use its definition does not
dominate or that comes before it in block order; and a constant whose value does not
fit its type (OpConstantTrue of an integer, a composite constituent of another type, a
vector of other than 2, 3, 4, 8 or 16 lanes), a variable initializer of another type
and an OpFunction whose function type does not match. Every such refusal, and every
refusal raised while an instruction is lowered, quotes the instruction (the review's
fuzzing found modules that reached the assembler as IR it refused by line, and a
repeated switch case and an `add i32 true` that it packed and Metal built; round 2's
found the CompositeInsert, access chain and module-scope OpExtInst shapes above: test
"ill-typed results of CompositeInsert, access chains and CopyObject, and module-scope
computations, are refused with the instruction quoted"). These checks are the rules the
lowering relies on, not a SPIR-V validator: a module malformed in a way none of them
covers can still reach the assembler, which then refuses it by IR line (`spirv-splice:
line N: ...`, then `assembling <entry>: ... (the assembler's message is above)`).

GLSL.std.450 and OpenCL.std (same mapping by name):

| GLSL.std.450 | OpenCL.std | emitted |
| --- | --- | --- |
| Round, RoundEven, Trunc, FAbs, Floor, Ceil, Sqrt | round, rint, trunc, fabs, floor, ceil, sqrt | `llvm.round` (the assembler's sign-keeping expansion), `llvm.rint`, `llvm.trunc`, ... |
| Sin, Cos, Tan, Exp, Pow, Log, Exp2, Log2 | sin, cos, tan, exp, exp10, pow, log, log2, log10 | `llvm.*`; intrinsics.zig renames sin/cos/tan/exp/exp10/pow/log10 to `air.*` / `air.fast_*`; strict half sin and tan (and asin, next row) then select x where x is ±0: Metal's precise `air.sin.f16`, `air.tan.f16` and `air.asin.f16` return +0 for -0 (Apple's assembler too), where C99 Annex F, required by OpenCL C 3.0 section 7.5, gives ±0; every other negative half, f32 and the other odd functions keep the sign (exec test half_signed_zero) |
| | powr | `llvm.pow(fabs(x), y)`, NaN selected where OpenCL C defines it: x < 0, (±0, ±0), (+inf, ±0), (+1, ±inf), a NaN operand (pow gives 1 or a value there); `fabs` makes powr(-0, 3) +0 and powr(-0, -3) +inf |
| Asin Acos Atan Sinh Cosh Tanh Asinh Acosh Atanh, Atan2, InverseSqrt | the same names, atan2, rsqrt | `air.<fn>.<T>` (`air.fast_<fn>` under fast math), Apple's names from `xcrun metal`; strict f32 atan and atan2 then `llvm.copysign` from x / y: Metal's precise `air.atan.f32(-0.0)` is +0.0 and `air.atan2.f32(-1e-30, 1e30)` +0.0 (Apple's `precise::atan` too), where C99 Annex F, required by OpenCL C 3.0 section 7.5, gives -0.0 (exec test atan_signed_zero; half and the fast forms return -0.0) |
| FMin/NMin, FMax/NMax, FClamp/NClamp | fmin, fmax, fclamp | `llvm.minnum`, `llvm.maxnum`, max then min |
| UMin UMax SMin SMax, UClamp SClamp, SAbs | u_min ... s_clamp, s_abs | `llvm.umin/umax/smin/smax`, `llvm.abs` |
| Fma | fma; mad | `llvm.fma`; `fmul` + `fadd` |
| FMix | mix | `x*(1-a) + y*a`; OpenCL `x + (y-x)*a` |
| Step, SmoothStep | step, smoothstep | `select`, `t*t*(3-2t)` of the clamped ratio |
| FSign, SSign | sign | select chains (OpenCL: NaN → 0) |
| Fract, Radians, Degrees | radians, degrees | `x - floor(x)`, `fmul` by the constant |
| Ldexp | ldexp | `llvm.ldexp` → `air.ldexp` (half always precise: `air.fast_ldexp.f16` crashes Metal) |
| Length, Distance, Normalize, Cross | cross | `air.dot` + `llvm.sqrt` + arithmetic, shuffles for cross |
| | length, distance, normalize | OpenCL C's special cases on top: the vector is scaled by a power of two first when its sum of squares overflows (2^-70) or is below the smallest normal (2^86), and a length scaled back, a half vector computed in float (no one factor fits half's range: with 2^10, normalize(half2(2^-24, 0)) was (inf, NaN) and length(half2(2^-24, 0)) 0; exec test half_geometric_subnormal), so length(1e30, 0) is 1e30, not inf (libclc does the same); normalize returns an all-zero vector unchanged and replaces infinite elements by ±1 and the others by 0 before dividing (GLSL leaves both undefined; v / length(v) gave NaN); a scalar normalize is ±1, or itself for 0 and NaN |
| FindILsb, FindUMsb, FindSMsb | ctz, clz, popcount | `llvm.cttz/ctlz` with the -1 / sign rules, `llvm.ctpop` |
| | s_mul_hi, u_mul_hi, copysign, fmod | the high half as UMulExtended, `llvm.copysign`, `frem` |

Refused: Frexp, Modf, Pack/Unpack, Interpolate*, Determinant, MatrixInverse,
FaceForward, Reflect, Refract and the OpenCL functions not listed; the matrix
instructions (OpMatrixTimesVector, OpVectorTimesMatrix, OpMatrixTimesScalar,
OpMatrixTimesMatrix, OpOuterProduct, OpTranspose), OpCopyLogical, OpQuantizeToF16,
OpSatConvertSToU / UToS, OpIsNormal, OpSignBitSet, the integer dot products (OpSDot,
OpUDot, OpSUDot and their AccSat forms), OpPtrDiff, OpCopyMemorySized and the subgroup
and group operations, each with "<op> is not supported" (test "the core instructions
docs/spirv.md lists as not supported are refused by name" in tools/spirv/lower.zig);
OpArrayLength (Metal
gives a kernel no buffer length: pass it in a uniform); OpSpecConstantOp; OpKill and
the other fragment instructions; the barrier and atomic forms of the next section's
refusal list; anything else, by name.

## Barriers and atomics

Stage 4 of the plan. Workgroup variables are threadgroup memory (`internal
addrspace(3) global T undef`, reported by Metal as `staticThreadgroupMemoryLength` and
in the manifest as `threadgroup_bytes`); barriers become Apple's barrier intrinsics;
atomics become native LLVM atomics, which Metal's compiler accepts and lowers itself
([air-format.md](air-format.md)).

| SPIR-V | LLVM IR / AIR |
| --- | --- |
| OpControlBarrier Workgroup *M* *S* | `call void @air.wg.barrier(i32 flags, i32 1)` (MSL `threadgroup_barrier`); GLSL `barrier()` (2 2 264) is Apple's `(i32 2, i32 1)`; the declaration and every call are `convergent` (below) |
| OpControlBarrier Subgroup *M* *S* | `call void @air.simdgroup.barrier(i32 flags, i32 4)` (`simdgroup_barrier`) |
| OpControlBarrier Workgroup Device *S* with UniformMemory or CrossWorkgroupMemory and an ordering (HLSL `DeviceMemoryBarrierWithGroupSync`) | `air.atomic.fence(i32 1, i32 5, i32 2)` first: a threadgroup barrier orders device memory only within the threadgroup |
| OpControlBarrier Subgroup *M* *S*, *M* Workgroup or wider, *S* with an ordering and a storage-class bit | `air.atomic.fence(i32 flags, i32 5, i32 scope of M)` first, then `air.simdgroup.barrier(i32 flags, i32 4)`: the SIMD-group barrier orders memory only among the threads of the SIMD-group (its second operand is the thread scope; Apple compiles MSL `simdgroup_barrier(mem_threadgroup)` to `(i32 2, i32 4)` and a threadgroup-scope `atomic_thread_fence` to `air.atomic.fence(i32 2, i32 5, i32 1)`), and SPIR-V orders the earlier accesses at *M* (probes `sync_ControlBarrier_sg_wgscope`, `_sg_devscope`; exec `sg_reverse`) |
| OpMemoryBarrier *M* *S* | `call void @air.atomic.fence(i32 flags, i32 5, i32 scope)` (`atomic_thread_fence(flags, seq_cst, scope)`; a native `fence` crashes Metal); nothing when *S* has no ordering or no storage-class bit (it orders nothing) |
| OpAtomicLoad / OpAtomicStore | `load atomic` / `store atomic` with the ordering, `align` of the type |
| OpAtomicExchange, IAdd, ISub, SMin, SMax, UMin, UMax, And, Or, Xor, FAddEXT | `atomicrmw xchg / add / sub / min / max / umin / umax / and / or / xor / fadd` (FAddEXT on device memory only, see below) |
| OpAtomicIIncrement / IDecrement | `atomicrmw add / sub` of 1 |
| OpAtomicCompareExchange(Weak) *Value* *Comparator* | `cmpxchg ptr, T comparator, T value success failure` + `extractvalue 0` (Weak has the same semantics in SPIR-V and is emitted strong) |

Memory semantics: the storage-class bits give the AIR `mem_flags` (WorkgroupMemory →
threadgroup 2; UniformMemory, CrossWorkgroupMemory → device 1; ImageMemory → texture
4; none → `mem_none` 0; AtomicCounterMemory orders memory a kernel cannot have and is
ignored, so glslang's `memoryBarrier()` / `groupMemoryBarrier()`, `OpMemoryBarrier
Device|Workgroup 0xD48` (AcquireRelease with glslang's MemorySemanticsAllMemory), is
`air.atomic.fence(i32 7, i32 5, i32 2|1)`; OutputMemory is refused); the ordering bit gives the LLVM ordering (none → `monotonic`,
Acquire → `acquire`, Release → `release`, AcquireRelease → `acq_rel`,
SequentiallyConsistent → `seq_cst`). A load cannot release and a store cannot acquire,
so their AcquireRelease is narrowed to `acquire` / `release` and a Release load or an
Acquire store is refused; a compare-exchange's Unequal semantics are its failure
ordering (a load's). On an atomic, MakeAvailable and MakeVisible need nothing more
(every ordering is at least as strong), nor does the Volatile bit: a native atomic load
in a polling loop re-reads memory (the review's control for the volatile exec tests:
an OpAtomicLoad spin-wait saw the other SIMD-group's store on its first poll, where a
plain load never did). Memory scopes: Workgroup and Invocation → threadgroup
1, Subgroup → simdgroup 4, Device / QueueFamily / CrossDevice → device 2 (the fences'
third operand). An atomic's own Scope operand is not encoded: native atomics carry the
system scope, at least as wide as any SPIR-V scope.

| type, memory | Metal | here |
| --- | --- | --- |
| 32-bit integer, device or threadgroup | every operation, every ordering | all mapped (probes `atom_*_u32_*`) |
| 32-bit float, device | add, load, store, exchange | mapped (probes `atom_*_f32_dev`, exec `atomic_fadd`) |
| 32-bit float, threadgroup: load, store, exchange | build and compute right | mapped (probes `atom_{Load,Store,Exchange}_f32_tg`, exec `atomic_fxchg_tg`) |
| 32-bit float add, threadgroup | `atomicrmw fadd ptr addrspace(3)` builds but computes a wrong sum: 64 threads adding 1.0 to a threadgroup float leave 0.0 and the old values they get back are 0..31 twice (counted per SIMD-group), with every ordering, also when Apple's `xcrun metal` assembles the same IR; MSL has no threadgroup `atomic_float` (`atomic_fetch_add_explicit` on one does not compile, `-std=metal3.0` to `metal4.0`) | refused (`atom_FAddEXT_f32_tg`); `zig build spirv-corpus` builds hand-workgroup-atomics:atomic_fadd_tg anyway and checks that the wrong answer persists |
| 32-bit float min/max | `atomicrmw fmin/fmax` fails pipeline creation with "Failed to materializeAll"; Apple's front end does not parse it ("expected binary operation in atomicrmw"); MSL has no float atomic min/max | refused (probes `atom_FMinEXT_f32_*`) |
| 64-bit integer load / store, device or threadgroup | builds | mapped (`atom_Load_u64_*`, `atom_Store_u64_*`) |
| 64-bit min/max, device, result unused | builds, and computes the signed and unsigned extremes (exec `atomic_minmax64`) | mapped (`atom_SMax_u64_dev`, ...) |
| 64-bit min/max with the result used, or in threadgroup memory; every other 64-bit read-modify-write and compare-exchange | MTLCompilerService crashes (XPC_ERROR_CONNECTION_INTERRUPTED), also when Apple's `xcrun metal` assembles the same IR text | refused (`atom_*_u64_*`, the naga int64 modules) |
| Function / Private memory, constant memory | no atomics | refused |

The barrier is convergent, so the assembler's uniformity analysis
(tools/air/divergence.zig, [pipeline.md](pipeline.md)) decides whether every thread of
the threadgroup reaches it, from the manifest's builtins (`thread_position_in_grid`,
`thread_position_in_threadgroup` per-thread; `threadgroup_position_in_grid`,
`threads_per_threadgroup`, buffer contents uniform). A barrier in the join block after
a per-thread branch or in a loop bounded by a uniform value is accepted; one inside a
per-thread branch, or in a loop that threads leave at different iterations, is refused
("convergent call in divergent control flow", printed as `spirv-splice: line N: ...`
with N a line of the lowered IR that `spirv-splice in.spv out.metallib out.ll` writes).
Because the SPIR-V path keeps every OpVariable Function as an `alloca` where the Zig
path's LLVM has promoted locals to SSA values, the analysis follows private variables:
a variable holds what was stored in it, through the variable itself, a
`getelementptr` into it (OpAccessChain to a struct member, an array element or a
vector lane) and a helper it is passed to by pointer, which reads it and may store
into it. So a loop counter kept in a Function variable (glslang's and Zig's SPIR-V
shape) is as uniform as the values stored into it and the blocks the stores sit in,
and so is a loop bound that glslang -V routes through Function variables: `uint tiles
= ceilDiv(pc.n, 64u)` (every helper argument is copied into a `param` variable and
passed by pointer), `P p = P(pc.n); ... p.n`, `uvec2 r = uvec2(pc.n, 64u); ... r.x`.
Those three shapes, with a barrier in the tiled loop, were refused until review round
2; they are the exec tests uniform_bound_helper_by_pointer, uniform_bound_local_struct
and uniform_bound_local_vector (two workgroups of 64, `pc.n` = 200, every output
matching the CPU), and the same kernels with the bound read from the local id are
refused (exec.zig's test). A store at a per-thread index, a store only some threads
make, or a per-thread value stored by the helper makes the variable per-thread; a
variable whose address escapes (stored, compared, passed to `llvm.memcpy` or another
declaration, or to a helper that lets it escape) is per-thread; and `load atomic` is
per-thread like every atomic result. default.metallib is byte-identical with the change
(the analysis only accepts or refuses).

The analysis checks the IR as it is handed to Metal, and Metal optimizes it again. The
SPIR-V path's IR is unoptimized, so it keeps glslang's shape for a workgroup counter or
histogram: `if (lid == 0) cell = 0; barrier(); atomicAdd(cell, 1); barrier(); if (lid ==
0) out = cell;`. Every barrier is in uniform control flow, valid SPIR-V, and accepted
here. Metal's jump threading, however, duplicates the block between the two branches on
the same condition into both arms unless the barrier is declared `convergent`, and the
lanes of SIMD-group 0 then reach different copies of the barriers. With the
attribute-less declaration this project used to write, the exec test
barrier_workgroup_counter (that kernel, 16 workgroups of 64) gave 0x40000021 in every
workgroup, 33 of 64 increments, and barrier_shared_flag (`if (lid < 16) flag[lid] = 1;
barrier(); tile[lid] = x; barrier(); if (lid < 16) out = tile[lid + 16];`) read 0 in
every lid < 16 word; the reviews' reproducers (a glslang histogram, a Workgroup
array-of-struct counter, 22 of 30 OpAtomic read-modify-write kernels seeded by thread 0)
lost increments the same way, also through Apple's assembler on the same text, and
computed right with only `convergent` added to the declaration. The assembler now
writes Apple's attribute sets, on the Zig path too (`convergent mustprogress nounwind
willreturn` on the declaration of every barrier and SIMD-group collective, `convergent
nounwind willreturn` on every call, `convergent` on every helper reaching one;
[air-format.md](air-format.md#23-intrinsics)); all of those kernels compute right
(30 of 30 atomics, the histogram's 256 per bin), and so do the two exec tests.

## Runtime validation

Metal's runtime compiler is the only verdict that counts, so every mapping is built
into a compute pipeline, and the probe kernels are executed against a CPU reference.

- **Probes** (`tools/spirv/probes.zig`): one tiny module per mapped instruction and
  type, 703 in all — each GLSL.std.450 function at f32, v4f32 and f16 (the geometric
  ones at 1-4 lanes, the integer ones at i8-i64 and v4i32), each OpenCL.std function in
  the OpenCL flavour (Kernel, CrossWorkgroup pointer parameters, `<3 x u64>` ids), every
  arithmetic, bitwise, shift and comparison instruction at i8/i16/i32/i64/v4i32 and
  f16/f32/v4f32, every conversion between the integer widths and to and from the float
  types, bitcasts across lane counts, the carries and wide multiplies at 16-64 bits,
  the bit-field instructions, the vector and composite instructions, OpSelect of a
  struct, an array and a nested struct with a bool and a vec3, and the conversions with
  SaturatedConversion and FPRoundingMode RTE/RTP/RTN (saturating at every width,
  NaN and infinities included). Stage 4 adds 118 pipeline-only probes (`atom_*`,
  `sync_*`: every atomic instruction at u32, u64 and f32 on a device and a threadgroup
  pointer, every memory ordering on IAdd / Load / Store / CompareExchange, every
  barrier and fence flag combination, the Subgroup barrier with a Workgroup and a Device
  memory scope; 31 of them are expected refusals that pin the 64-bit, float min/max and
  threadgroup float add limits above), 821 probes in all; their results depend on
  the other threads, so the known-answer kernels of [the corpus](#the-corpus) execute
  them instead.
- **Execution tests** (`tools/spirv/exec.zig`): 26 whole modules with fixed inputs
  and every output word known: a BufferBlock SSBO beside a UBO, an OpenCL `__constant`
  table, OpenCL `convert_*_sat` / `_rte` at their edges, OpSelect of a struct loaded
  from a runtime-indexed Function array, phis under a two-case switch and an
  equal-target branch, std430 `vec3` + `float` and scalar-layout `vec3[]` read and
  written, NonSemantic.Shader.DebugInfo.100 in the body, packed structs inside
  natural ones and scalar-layout `{ uint; uvec4 }` arrays, an OpenCL CPacked struct
  (align 1 loads), a WorkgroupSize constant overriding LocalSize (dispatched with the
  manifest's size), Zig's integer widenings in
  both flavours and into an i128's limbs, an interface decorated only through
  decoration groups, OpBitField* at Offset = width with Count 0 (constant and runtime
  operands), OpenCL powr / normalize / length / distance at their defined edges and
  at 1e30 / 1e-30, and a packed member's pointer through OpSelect, a helper, a loop phi
  and a Function variable, and two barrier kernels in glslang's shape that Metal
  jump-threads unless the barrier is `convergent` (barrier_workgroup_counter,
  barrier_shared_flag; dispatched with the manifest's workgroup size), and the six
  for the review's lowering defects (round 1): OpenCL atan(-0) and atan2(-1e-30, 1e30)
  keep the sign (atan_signed_zero), an access chain on a pointer bitcast to another
  pointee (bitcast_access_chain), two Workgroup Block variables punning one allocation
  (workgroup_block_alias), scalar-layout structs ending in a vec3 loaded and stored
  whole without touching the neighbour (scalar_tight_structs), a packed member's
  pointer through 70 helper calls (packed_pointer_call_chain), and GL_EXT_buffer_reference
  pointers from a push-constant block, a uint64 and a uvec2 into four buffers the
  runner only makes resident and whose `gpuAddress` it writes (buffer_reference). Each
  of the six computed a wrong answer or was refused before its fix. The six of round 2
  (lowering): spin-waits on a flag another SIMD-group sets, through Volatile memory
  operands (volatile_spin_wait) and glslang's Volatile members (volatile_member_spin_wait),
  half sin, tan and asin of ±0 (half_signed_zero; strict only, as fast math may drop a
  signed zero), OpenCL normalize and length of half vectors with subnormal elements
  (half_geometric_subnormal), access chains continued from tight scalar-layout elements
  (tight_chain_continued) and an entry point named like a generated helper symbol (f20);
  each polled to its cap, kept the wrong sign, returned inf / NaN, wrote the wrong
  words or was refused before its fix.
- **Corpus** (`tools/spirv/corpus/`, 153 entry points): each lowered on its own;
  `zig build spirv-corpus` builds them in one process, executes the thirteen kernels
  with known answers and checks that the threadgroup float add finding still
  reproduces ([The corpus](#the-corpus)).

`tools/spirv/run_probes.sh <dir> [--fast-math=true]` lowers each kernel into its own
library (`spirv-probe`), builds its pipeline (`metallib-check --kernels`) and compares
the table with [tools/spirv/corpus/EXPECTED.txt](../tools/spirv/corpus/EXPECTED.txt).
`zig-out/bin/spirv-probe-run [--fast-math=true]` dispatches 64 threads of every probe
over edge values and random bits (kept inside the instruction's defined domain) and
compares each lane with `tools/spirv/reference.zig`: bit-exact for exact operations
(an f32 subnormal result may come back flushed, the library declares
`denorms_disable`), within 2e-4 (f32) / 2e-2 (f16) relative for the transcendental and
geometric functions, whose precision is Apple's (`zig build survey`).

Results on an Apple M3, macOS 26.3 (2026-10-21):

| check | strict | `--fast-math=true` |
| --- | --- | --- |
| pipelines, probes + corpus (974 kernels; S2: 815; 941 before docs review round 2 added 33 i8 / i16 probes) | 933 ok, 41 refused as expected, 0 Metal errors | the same |
| execution, 703 probes (rerun 2026-10-21 after docs review round 2) | all match (82882 lanes) | all match (79375 lanes; lanes with non-finite floats skipped, as fast math assumes; OpenCL powr and the geometric functions now include edge lanes, so fewer are finite) |
| execution, 35 whole-module tests (26, the three `uniform_bound_*` of review round 2 and the six of round 2's lowering fixes, rerun 2026-10-21) | all match | 34 match, half_signed_zero skipped (strict only) |
| `zig build spirv-corpus`: 156 kernels (the corpus and tools/spirv/zig_kernels.zig) built in process, 13 executed with known answers, 1 Metal finding executed | 146 ok, 10 refused as expected, 13 match, the finding reproduces | the same |
| `metal-objdump -d`, 796 + the first 7 execution libraries (S2); the stage-4 libraries (zig_kernels, hand-glslang-stage4, hand-workgroup-atomics, naga atomicOps, atomicOps-float32, atomicOps-int64-min-max, atomicCompareExchange, abstract-types-operators) | all parse | |
| Apple's assembler on the first 7 execution modules' `out.ll` (S2) and on the 8 stage-4 libraries' `out.ll` (hand-workgroup-atomics through `spirv-splice --allow-metal-miscompiles`: without it, its refused atomic_fadd_tg refuses the whole module and there is no `out.ll`) | all build, all pipelines ok (20 stage-4 kernels: zig_kernels 3, hand-glslang-stage4 2, hand-workgroup-atomics 9, atomicCompareExchange 2, the other four naga modules 1 each) | |
| `spirv-probe-run --apple=<dir>`: Apple's assembler on every probe's and execution test's `out.ll`, executed ([above](#the-zig-shape-modules-and-apples-assembler)) | 698 of 703 probes match (the 5 others are the `frem` finding the printed text keeps), 35 of 35 execution tests (rerun 2026-10-21 after docs review round 2) | 701 of 703, 34 of 34 (half_signed_zero is strict only) |

The 41 expected refusals: 31 stage-4 probes (26 64-bit atomics, 4 float min/max, the
threadgroup float add), hand-workgroup-atomics:atomic_fadd_tg (the same finding), the
naga int64 modules (atomicOps-int64: a 64-bit OpAtomicIAdd; atomicCompareExchange-int64:
two 64-bit compare-exchanges), stage 5 (4: naga access's vertex and fragment entries,
the two atomicTexture image variables), a double (hand-scalars `float_convert`) and
GLSL.std.450 PackSnorm4x8 (naga bits). The 13 stage-4 refusals of S2 (the corpus'
barrier and atomic kernels, naga's atomicOps, atomicCompareExchange,
abstract-types-operators) now build.

Metal findings made while mapping (each fixed in the lowering, the probe names it):

| construct | Metal | now |
| --- | --- | --- |
| `llvm.uadd/usub.with.overflow.v4i32` | MTLCompilerService crashes (XPC_ERROR_CONNECTION_INTERRUPTED); the scalar forms pass | `add`/`sub` + `icmp ult` (op_IAddCarry_v4u32) |
| `mul i128` (the 64-bit high product) | crashes | 32-bit halves (op_UMulExtended_u64, ocl_u_mul_hi_u64) |
| `air.fast_ldexp.f16` | crashes; Apple's own `-ffast-math` output calls `air.ldexp.f16` | intrinsics.zig keeps half ldexp precise (glsl_Ldexp_f16) |
| `air.fast_fmod.v4f32(-0, inf)` | NaN (the precise `air.fmod` returns -0) | fast-math semantics, not a mapping error |
| a runtime-indexed element of a private `[2 x { i32, <4 x i32> }]` (interior padding) loaded whole, or a GEP that starts from that element pointer | crashes (XPC_ERROR_CONNECTION_INTERRUPTED), with our assembler and with Apple's on the same text; `{ i32, i32 }`, `{ <4 x i32>, i32 }`, `[4 x i32]` elements and constant indices pass | one GEP from the variable per member (exec test composite_select); the review's "aggregate select" crash was this load, not the select |
| 64-bit `atomicrmw` (add, sub, and, or, xor; xchg and min/max whose result is used; min/max on threadgroup memory) and 64-bit `cmpxchg` | crashes, with our assembler and with Apple's on the same text (`atomicrmw add ptr addrspace(1) %p, i64 %v monotonic` in an `out.ll`); a min/max whose result is unused, `load atomic` and `store atomic` pass | refused by name (probes atom_*_u64_*) |
| `atomicrmw fmin` / `fmax` | "Failed to materializeAll" (Apple's front end: "expected binary operation in atomicrmw") | refused (atom_FMinEXT_f32_*) |
| `atomicrmw fadd ptr addrspace(3)` (threadgroup float add) | builds, then computes a wrong answer: the adds never reach threadgroup memory (64 threads adding 1.0 leave 0.0) and the returned old values are counted separately per SIMD-group (0..31 twice); the same with seq_cst, with plain loads and stores around it, with a 32-thread threadgroup, and from Apple's `xcrun metal -Xclang -opaque-pointers` on the same IR; device float add, threadgroup float load/store/exchange and threadgroup integer atomics compute right | refused by name (atom_FAddEXT_f32_tg); the corpus driver's finding check (hand-workgroup-atomics:atomic_fadd_tg) runs it and fails when Metal starts computing it right |

### The Zig-shape modules and Apple's assembler

The four Zig SPIR-V backend outputs of the corpus (and zig-narrow-ints), lowered,
packed, checked, then assembled by Apple from the printed `out.ll`:

| module | ours | `metal-objdump` | `xcrun metal -Xclang -opaque-pointers out.ll` | Apple's library |
| --- | --- | --- | --- | --- |
| zig-vulkan-params (pointer parameters, PhysicalStorageBuffer64) | ok | ok | ok | ok |
| zig-opencl-params (Kernel, CrossWorkgroup) | ok | ok | ok | ok |
| zig-crosscheck (Block buffers, Uniform, GlobalInvocationId) | ok | ok | ok | ok |
| zig-control-flow (loop, switch, phis, Workgroup array, GLSL.std.450) | ok, 256 bytes threadgroup memory | ok | ok | ok, 256 bytes |
| zig-narrow-ints (unbound buffers, negative i8/i16 constants) | ok | ok | ok | ok |

`out.ll` used to keep the `llvm.*` spellings that tools/air/intrinsics.zig renames
inside our assembler, and Apple's front end passes some of them through to Metal. On the
probes' `out.ll`, `llvm.pow` (glsl_Pow_f32, ocl_pow_f32), `llvm.sin` (glsl_Sin_f32) and
`llvm.log10` (ocl_log10_f32) crash MTLCompilerService at pipeline creation
(XPC_ERROR_CONNECTION_INTERRUPTED), while `llvm.exp10.f32` (ocl_exp10_f32) and
`llvm.ldexp.f32.i32` (glsl_Ldexp_f32) reach Metal unchanged and fail to link
("Undefined symbols: llvm.exp10.f32, referenced from: ocl_exp10_f32"); measured again
on 2026-10-21 by putting those spellings back into today's `out.ll` of the six probes,
where the `air.*` spellings build all six. Of the then 18 execution modules,
opencl_edge_math's did not build a pipeline. The printed
text now spells every rename exactly as the assembler applies it, with the same fast-math
choice (`intrinsics.printForApple`, shared with air-splice's `shader.air.ll` and
`crosscheck.air.ll`), and carries the same `convergent` attributes. It keeps the forms the
assembler rewrites around a recorded Metal finding (`frem` → `air.fmod`, `llvm.round` →
the sign-keeping expansion, `llvm.smul/umul.fix.sat` → `air.mad_sat`), so Apple's build of
the text still reproduces those findings.

`zig-out/bin/spirv-probe-run --apple=<dir>` runs every probe and execution test on
Apple's build of its `out.ll` instead of ours (2026-10-21, M3, macOS 26.3):

| | strict | `--fast-math=true` |
| --- | --- | --- |
| the six probes above and opencl_edge_math | all build, all match | all build, all match |
| 703 probes | 698 match; 5 differ: ocl_fmod_v4f32, op_FRem_v4f32, op_FMod_v4f32 (`frem` of -0 by inf is NaN, not -0) and op_FRem_f16, op_FMod_f16 (Apple's half `frem` is not exact: the first differing lane of a run was 2 to 16 f16 ulps off, e.g. 28.75 for fmod(308, 39.906) = 28.656 (0x4f30 for 0x4f2a, 6 ulps), -38.0 for fmod(-1316, 182.5) = -38.5 (16 ulps), 12.25 for fmod(343.5, 36.8125) = 12.1875 (8 ulps), 54.0 for 54.0625 (2 ulps), or +0 for the exact -0 of fmod(-1359, 453); which lane differs first depends on the inputs, which `--only` changes): the `frem` finding, which our library avoids through `air.fmod` | 701 match; op_FRem_f16 and op_FMod_f16 differ the same way |
| 35 execution tests | all match | 34 match (half_signed_zero skipped: strict only) |

The six probes' libraries from the two assemblers also gave bit-identical results on
the same random inputs.

Reproduce: `zig-out/bin/spirv-splice --manifest=m.txt tools/spirv/corpus/zig-crosscheck.spvasm
x.metallib x.ll`, `zig-out/bin/metallib-check --kernels=crosscheckKernel x.metallib`,
`xcrun metal -Xclang -opaque-pointers x.ll -o apple.metallib` and the same check on it.

## The corpus

```sh
zig build spirv-corpus [-Dmetal-fast-math=true]
```

runs tools/spirv/corpus_driver.zig, which assembles every
`tools/spirv/corpus/*.spvasm`, lowers and packs each entry point on its own, loads the
library with `newLibraryWithData`, builds a compute pipeline and compares the outcome
with the file's rows of [EXPECTED.txt](../tools/spirv/corpus/EXPECTED.txt) (the table
run_probes.sh also checks, there in one process per kernel). It then builds
tools/spirv/zig_kernels.zig, which build.zig compiles with Zig's self-hosted SPIR-V
backend, and dispatches the thirteen kernels with known answers, comparing every output
word with a CPU computation. A kernel the lowering refuses because Metal miscompiles it
(today hand-workgroup-atomics:atomic_fadd_tg, the threadgroup float add) is built once
more with `allow_metal_miscompiles` and executed against its known answer, which must
still differ (`finding reproduces`): when a macOS release fixes the miscompile, the
step fails and says to lift the refusal. One row per kernel: name, pipeline outcome,
execution outcome; exit 1 on any outcome that differs from EXPECTED.txt, a Zig kernel
that does not build, a known-answer kernel that does not match, or a finding that no
longer reproduces.

| file | source | entry points | ok | refused (why) | executed |
| --- | --- | --- | --- | --- | --- |
| wgsl-collatz | naga | 1 | 1 | | 1 |
| wgsl-boids, wgsl-bitcast, wgsl-6772-unpack-expr-accesses, wgsl-abstract-types-operators | naga | 4 | 4 | | |
| wgsl-abstract-types-const | naga | 0 (constants only; assembled and read) | | | |
| wgsl-access | naga | 3 | 1 | 2 (vertex, fragment: stage 5) | |
| wgsl-bits | naga | 1 | | 1 (PackSnorm4x8) | |
| wgsl-atomicOps, wgsl-atomicOps-float32, wgsl-atomicOps-int64-min-max | naga | 3 | 3 | | |
| wgsl-atomicCompareExchange | naga | 2 | 2 | | |
| wgsl-atomicOps-int64, wgsl-atomicCompareExchange-int64 | naga | 3 | | 3 (64-bit add, compare-exchange) | |
| wgsl-atomicTexture, wgsl-atomicTexture-int64 | naga | 2 | | 2 (images: stage 5) | |
| zig-vulkan-params, zig-opencl-params, zig-crosscheck, zig-control-flow, zig-narrow-ints | Zig 0.18.0-dev.253 | 5 | 5 | | |
| hand-glsl-std-450, hand-opencl-std | hand | 101 | 101 | | |
| hand-scalars | hand | 8 | 7 | 1 (double) | |
| hand-vectors, hand-control-flow | hand | 8 | 8 | | |
| hand-workgroup-atomics | hand | 9 | 8 | 1 (threadgroup float add: a Metal finding, executed by the finding check) | 6 |
| hand-glslang-stage4 | hand, glslang's shape | 2 | 2 | | 2 |
| hand-dxc-reflect | hand, DXC's `-fspv-reflect` shape | 1 | 1 | | 1 |
| tools/spirv/zig_kernels.zig | Zig's SPIR-V backend, built by `zig build spirv-corpus` | 3 | 3 | | 3 |

| known-answer kernel | what runs | checked |
| --- | --- | --- |
| wgsl-collatz:main | naga's WGSL Collatz kernel (loop, helper call, OpUMod / OpUDiv, a runtime-array SSBO), 4096 threads, LocalSize 1 | each of 1..4096 replaced by its Collatz step count, as `collatz` in the driver computes it in wrapping u32 |
| hand-workgroup-atomics:tg_reverse, zig_kernels:tg_reverse | `tile[lid] = in0[gid]; barrier; out[gid] = tile[63 - lid]`, hand-written and from Zig's `std.spirv.workgroupBarrier()` | all 4096 words |
| hand-workgroup-atomics:atomic_counter, zig_kernels:atomic_counter | OpAtomicIAdd of 1 on one device word by every thread (the Zig one through SPIR-V inline assembly: Zig's SPIR-V backend has no atomics) | the old values are a permutation of 0..n-1, the counter ends at n |
| hand-workgroup-atomics:atomic_minmax64 | 64-bit SMax / SMin / UMax / UMin of `(in0 << 32) \| in1` | the four extremes, signed and unsigned |
| hand-workgroup-atomics:atomic_fadd | OpAtomicFAddEXT of 0.5 by 4096 threads | the old values are 0, 0.5, ..., 2047.5 in some order, the sum is 2048 |
| hand-workgroup-atomics:atomic_fxchg_tg | a threadgroup float: OpAtomicStore 0.0, barrier, OpAtomicExchange of lid + 1, barrier, OpAtomicLoad | per workgroup, every thread loads the same final value (one of 1..64) and the 64 old values are the other values of 0..64 |
| hand-workgroup-atomics:sg_reverse | `tile[lid] = in0[gid]`, OpControlBarrier Subgroup with memory scope Workgroup (fence + SIMD-group barrier), `out[gid] = tile[lid - sglid + sgsize - 1 - sglid]` (SubgroupLocalInvocationId, SubgroupSize) | all 4096 words, reversed within each SIMD-group of the pipeline's `threadExecutionWidth` (32 on the M3). Within one SIMD-group the barrier is not observable on the M3, so this checks the mapping builds and computes, not that the fence is needed |
| hand-glslang-stage4:reduce | a tree reduction in glslang's shape (hand-written): Function variables for `lid` and the loop counter, a barrier in a loop bounded by `gl_WorkGroupSize.x / 2`, SPIR-V 1.0 BufferBlock SSBOs | the wrapping sum of each workgroup's 64 inputs |
| hand-glslang-stage4:atomics | glslang's atomicAdd and atomicCompSwap on an SSBO (Uniform pointers into BufferBlock memory) | the counter values are a permutation; per slot exactly one thread swapped 0 for gid + 1 and every other thread of the slot saw that value |
| hand-dxc-reflect:main | DXC's `-fspv-reflect` output shape: `OpDecorateString` UserTypeGOOGLE on both structured buffers and UserSemantic on the builtin (reflection hints, ignored), SPIR-V 1.0 BufferBlock SSBOs at bindings 0 and 1 | `output[i] = input[i] * 3 + 1` for all 4096 words |
| zig_kernels:control_flow | testdata/cf.zig's kernel with the barrier it lacked, computing tools/spirv/zig_kernels_body.zig (loop, switch, early break, labeled-break phis, Floor, Fma, min/max) | all 4096 words against the same `body` compiled for the host |

Results (Apple M3, macOS 26.3, 2026-10-21): 156 kernels, 146 ok, 10 refused as
expected, the 13 known-answer kernels match, and the threadgroup float add finding
reproduces (`workgroup 0: final value 0, want 64`), strict and
`-Dmetal-fast-math=true`.

The barrier test is not vacuous: with the OpControlBarrier deleted from
hand-workgroup-atomics:tg_reverse (a scratch copy of the file run through the same
driver binary), the kernel mismatched in each of three runs (`out[64] = 0x0`: the
second SIMD-group read the tile before the first had written it).
zig-control-flow.spvasm, the S1 corpus copy of testdata/cf.zig, has exactly that race
(it reads `tile[63 - lid]` without a barrier, written before Zig's SPIR-V backend
could express one here), so it is built but not executed; zig_kernels:control_flow is
its race-free twin. Its `body` differs from cf.zig's in its float operations: cf.zig
bit-casts small integers to float, which makes subnormals that Metal flushes, and takes
`@sqrt`, which Metal does not round correctly, so a bit-exact CPU twin needs the
shifted-integer conversions, products, Floor and Fma of zig_kernels_body.zig.

## The crosscheck bench through SPIR-V

The differential bench of [crosscheck.md](crosscheck.md) has a SPIR-V mode. The same
generated `body(a, b)` is compiled by Zig's self-hosted SPIR-V backend instead of
LLVM's nvptx backend, lowered by spirv-splice instead of air-splice, and run by the
same driver against the same CPU build:

```sh
zig build crosscheck -Dseed=7 -Dcrosscheck-spirv=true [-Dcrosscheck-floats=true]
zig build crosscheck -Dcrosscheck-spirv=true -Dcrosscheck-body=<file.zig>
zig build crosscheck-apple -Dseed=7 -Dcrosscheck-spirv=true   # Apple's assembler on spirv-splice's out.ll
zig build reduce -Dseed=7 -Dcrosscheck-spirv=true             # candidates judged through SPIR-V
CROSSCHECK_CACHE_EVERY=5 CROSSCHECK_ARGS="-Dcrosscheck-spirv=true" tools/crosscheck/sweep.sh 1 20
```

`src/engine/crosscheck_harness_spirv.zig` is the kernel in the standard Vulkan compute
shape, which is what an engine's SPIR-V looks like, rather than the old pointer-parameter
shape:

| harness | SPIR-V | spirv-splice | driver |
| --- | --- | --- | --- |
| `in0`, `in1`: `@extern(*addrspace(.storage_buffer) const Buf, … binding 0 / 1)` | StorageBuffer Block `{ [4096 x i32] }`, NonWritable | `[[buffer(0)]]`, `[[buffer(1)]]` device, read | `setBuffer` 0, 1 |
| `out`: binding 2 | StorageBuffer Block | `[[buffer(2)]]` device, read_write | `setBuffer` 2 |
| `params`: `@extern(*addrspace(.uniform) const Params, … binding 3)` | Uniform Block `{ u32 }` | `[[buffer(3)]]` constant, 4 bytes | `setBytes` 3 |
| `spirv.global_invocation_id[0]` | Input, BuiltIn GlobalInvocationId | `thread_position_in_grid` | `dispatchThreads` 4096 |
| `callconv(.{ .spirv_kernel = .{ .x = 64, … } })` | LocalSize 64 1 1 (manifest) | | threadgroup 64 |

build.zig compiles it for `spirv64-vulkan` with the `generic` CPU plus `float16`,
`int64`, `int8` and `int16` (without a feature the backend refuses the type) and
`use_llvm = false`, hands the emitted `.spv` to `spirv-splice --fast-math=<as the Zig
path>`, and passes the library to the driver. spirv-splice's `out.ll` is what
`crosscheck-apple` assembles with `xcrun metal`. A SPIR-V seed adds about 7 MB to
`.zig-cache` (an nvptx seed about 50 MB: its air-splice imports the body).

**What the generator leaves out.** Zig's SPIR-V backend stops with `TODO (SPIR-V):
implement AIR tag add_sat` (`sub_sat`, `mul_sat`, `popcount`, `byte_swap`,
`bit_reverse`) and `implement unary operation 'clz' for vulkan os` (`ctz`), on i32 and
on `@Vector(4, i32)` (one-operator kernels, `zig build-obj -target spirv64-vulkan
-fno-llvm -O ReleaseFast -mcpu=generic+int64+int8+int16+float16`, 2026-10-21). `-Dcrosscheck-spirv=true` passes `--spirv` to the generator, which
leaves out exactly those operators (`program.spirv_unsupported`); a draw that lands on
one is drawn again from the same menu, so a seed that never lands on one prints the
same file as without the flag (`zig build test` checks this over 800 integer and 300
float seeds, and that every other operator is still drawn). A body that was changed
says so in its header (`// Generated with --spirv: N operator draws ... were drawn
again`). Integer programs are long, so only 39 of seeds 1-5000 avoid every excluded
operator (214 in float mode); a SPIR-V seed is usually a different program from the
nvptx seed of the same number, and the comparison of the two front ends below runs the
`--spirv` bodies through nvptx with `-Dcrosscheck-body`.

### A bug in Zig's SPIR-V backend

The first SPIR-V seed that failed (integer seed 2, 1,797 of 4,096 mismatched; its body
matched through nvptx and failed the same through Apple's assembler) came down to the
`t16u`, `t8u` and `mul16` helpers: `@as(i32, x)` for `x: u16` comes back sign-extended.
Zig's backend picks the conversion opcode by the result's signedness (`buildConvert`,
`src/codegen/spirv/CodeGen.zig:3474` at f2f0d97a9b), so it emits `OpSConvert %i32 %u16`,
which the specification defines as a sign extension whatever the operand's type, and
which spirv-splice therefore lowered to `sext`.

The same rule has a mirror case, found by a review rather than by a sweep (the
generator never draws 128-bit integers): an unsigned result gets OpUConvert, a zero
extension, even for a signed source. `airIntCast` widens an i16 or i32 into an i128
(stored as u64 limbs) by converting the source into limb 0 with buildConvert
(`CodeGen.zig:6078`) and then filling the upper limbs from limb 0's top bit
(`OpBitcast %i64`, `OpShiftRightArithmetic ... 63`, `CodeGen.zig:6111-6131`), so
`@as(i128, @as(i16, -1))` comes out as `OpUConvert %u64 %i16`: limb 0 is
zero-extended, its top bit is 0, the fill is 0, and the value is 65535.

**The workaround.** spirv-splice restores Zig's meaning, the way Mesa's `spirv_to_nir`
keys its glslang workarounds on the generator id: an OpSConvert is a `zext` when all of
these hold, and follows the specification otherwise:

- the header's Generator tool is 41 (Zig's self-hosted backend, `spec.zig_generator_id`);
- it widens (a narrowing OpSConvert is a `trunc` either way);
- its operand's OpTypeInt is unsigned **and its result's is signed**.

and an OpUConvert is a `sext` when all of these hold:

- the Generator tool is 41 and it widens, as above;
- its operand's OpTypeInt is signed and its result's is unsigned;
- its result is sign-filled: it is the operand of an `OpBitcast` to a signed integer
  type of the same width whose result is shifted by `OpShiftRightArithmetic` by that
  width minus one (`Lowerer.findSignFills` in tools/spirv/lower.zig), which is the
  sequence `airIntCast` emits after the limb conversion.

buildConvert is the only place the backend emits OpSConvert. In the Vulkan flavour
(`intType`, `CodeGen.zig:518`, keeps each type's signedness) a signed result with an
unsigned operand type comes from a Zig source type that was unsigned (an integer
widening, a packed-struct field insert, a shift amount cast to the base type), all of
which mean a zero extension, or from an `@intCast` between equal-width types that
`airIntCast` puns without an instruction, whose defined value is non-negative, where the
two extensions agree. In the OpenCL and amdhsa flavours `intType` makes every integer
type unsigned, so `OpSConvert %u32 %u16` there may be a real `@as(i32, @as(i16, x))`:
the signed-result condition leaves those modules to the specification, which also means
the bug is not undone there (it cannot be told apart from a sign extension).

For the mirror case, the Vulkan-flavour callers of buildConvert that widen a signed
source to an unsigned type are three. The composite-integer path above means a sign
extension. The packed-pointer store (`airStore`, `CodeGen.zig:7521`: a signed field
widened into the unsigned host integer, then shifted and ORed in) means a zero
extension. It cannot be reached at f2f0d97a9b, because a pointer to a packed-struct
field is a TODO (`CodeGen.zig:6909`), but the sign-fill condition keeps it a zero
extension if that changes. An `@intCast` from signed to unsigned (`CodeGen.zig:6150`)
has only non-negative defined values, where the two extensions agree, so a type-only
rule would also have been right for it. The OpenCL flavour types the operand, the
bitcast and the arithmetic shift unsigned (`OpShiftRightArithmetic %u64`), and
`@as(i64, @bitCast(@as(u64, u16x))) >> 63` emits the same sequence there, so its
OpUConvert stays a zero extension (the bug survives in that flavour too).
`--zig-workarounds=false` (`-Dspirv-zig-workarounds=false` on the bench) turns the
workaround off:

| `tools/crosscheck/known/` | `spirv_zig_unsigned_widen.zig` | `spirv_zig_signed_widen.zig` |
| --- | --- | --- |
| `-Dcrosscheck-spirv=true` | 4096/4096 match | 4096/4096 match |
| `-Dcrosscheck-spirv=true -Dspirv-zig-workarounds=false` | 1,801/4,096 mismatched | 1,232/4,096 mismatched |
| nvptx | 4096/4096 match | 4096/4096 match |
| `crosscheck-apple -Dcrosscheck-spirv=true` | 4096/4096 match | 4096/4096 match |

The signed-widen body's mismatches are one off and of the wrong sign (body(65535,
536563569) = GPU 2542 against CPU -2543): the i128 lost its sign.

Tests, both ways: the lowering's unit tests cover the Vulkan scalar and vector forms
(`zext`), a signed source (`sext`), the switch, a module from another generator
(`sext`), and the OpenCL shape (an unsigned result type: `sext`, and a narrowing
OpSConvert stays `trunc`); a further test lowers Zig's own output for
`tools/spirv/testdata/convert_vulkan.zig` (u16 and u8 to i32: `zext`; i16 to i32:
`sext`) and `convert_opencl.zig` (i16 and i8 to i32, typed `OpSConvert %u32 %u16` /
`%u8`: `sext`) with the workaround on and off. The execution tests
`zig_convert_vulkan` and `zig_convert_opencl` (`spirv-probe-run --only=zig_convert`) run
both modules on the GPU over 1, 0x7fff, 0x8000 and 0xffff80f1 and match every word.
Before the signed-result condition, the OpenCL module's two sign extensions were
lowered to `zext` (0x8000 would have come back as 32768 instead of -32768). For the
mirror case, a unit test lowers the sign-filled limb conversion (`sext`) beside the
packed-store shape, a shift by less than the width minus one and an unsigned operand
(all `zext`), with the workaround off and from another generator (`zext`); another
lowers Zig's own output for `tools/spirv/testdata/convert_wide.zig` (i16 and i32 into
an i128: `sext`; u16 into an i128 and `@intCast` of an i16 to u64: `zext`), and the
execution test `zig_convert_wide` runs it on the GPU over the same four inputs and
matches all 32 words (0x8000 gives 0xffffffff in bits 64-127).

**Upstream report** (text for an issue on Zig's tracker):

> **SPIR-V backend: integer widening picks OpSConvert/OpUConvert by the destination's signedness**
>
> `buildConvert` in `src/codegen/spirv/CodeGen.zig` (line 3474 at f2f0d97a9b) chooses
> the integer conversion opcode by the destination's signedness:
>
> ```zig
> if (dst_scalar.isSignedInt(zcu)) break :blk .OpSConvert;
> break :blk .OpUConvert;
> ```
>
> The SPIR-V specification defines OpSConvert as a sign extension of its operand and
> OpUConvert as a zero extension, regardless of the operand's type, so a widening must
> pick the opcode by the **source's** signedness (`src_scalar.isSignedInt(zcu)`; for a
> narrowing both truncate). Two visible effects:
>
> 1. an unsigned value widened to a signed type is sign-extended (`OpSConvert %i32 %u16`);
> 2. a signed value widened into a composite integer is zero-extended: `airIntCast`
>    converts the source into the unsigned limb 0 with `buildConvert(limb_ty, src)`
>    (line 6078), which emits `OpUConvert %u64 %i16`, and then derives the upper limbs'
>    sign fill from limb 0's top bit (lines 6111-6131), which is now always 0.
>
> In the OpenCL flavour, where `intType` makes every integer type unsigned, the opcode
> is the only place the signedness survives, so a consumer cannot undo either effect.
>
> Reproducer 1 (`repro.zig`):
>
> ```zig
> const In = extern struct { x: u32 };
> const Out = extern struct { y: i32 };
> const in = @extern(*addrspace(.storage_buffer) const In, .{ .name = "in", .decoration = .{ .descriptor = .{ .set = 0, .binding = 0 } } });
> const out = @extern(*addrspace(.storage_buffer) Out, .{ .name = "out", .decoration = .{ .descriptor = .{ .set = 0, .binding = 1 } } });
>
> export fn main() callconv(.{ .spirv_kernel = .{ .x = 1, .y = 1, .z = 1 } }) void {
>     const lo: u16 = @truncate(in.x);
>     out.y = lo; // @as(i32, lo)
> }
> ```
>
> `zig build-obj repro.zig -target spirv64-vulkan -fno-llvm -O ReleaseFast -mcpu=generic+int16`
> (0.18.0-dev.253+16d31ceed) emits (Generator 0x00290900; an excerpt of the disassembly):
>
> ```
> %u32 = OpTypeInt 32 0
> %u16 = OpTypeInt 16 0
> %i32 = OpTypeInt 32 1
> %lo = OpUConvert %u16 %13
> %18 = OpSConvert %i32 %lo
> ```
>
> Expected: `%18 = OpUConvert %i32 %lo`; with `in.x = 0xffff` a conforming consumer must
> then store 65535. Actual: `OpSConvert`, so it stores -1 (observed on Metal through a
> SPIR-V-to-AIR translator; the LLVM backend's nvptx build of the same source stores
> 65535). `-target spirv64-opencl` emits `%25 = OpSConvert %u32 %lo` for the same source,
> where every integer type is unsigned and the opcode is the only signedness left, so a
> consumer cannot tell this zero extension from a real `@as(i32, @as(i16, x))`.
>
> Reproducer 2 (`repro2.zig`, same `In`, `in` and `out` declarations with
> `Out = extern struct { y: u32 }`):
>
> ```zig
> export fn main() callconv(.{ .spirv_kernel = .{ .x = 1, .y = 1, .z = 1 } }) void {
>     const s: i16 = @bitCast(@as(u16, @truncate(in.x)));
>     const w: i128 = s; // @as(i128, s)
>     out.y = @truncate(@as(u128, @bitCast(w)) >> 64);
> }
> ```
>
> The same command emits:
>
> ```
> %u16 = OpTypeInt 16 0
> %i16 = OpTypeInt 16 1
> %u64 = OpTypeInt 64 0
> %i64 = OpTypeInt 64 1
> %27 = OpConstant %i64 63
> %s = OpBitcast %i16 %15
> %22 = OpUConvert %u64 %s
> %25 = OpBitcast %i64 %22
> %28 = OpShiftRightArithmetic %i64 %25 %27
> ```
>
> Expected: `%22 = OpSConvert %u64 %s`; with `in.x = 0xffff` (s = -1) a conforming
> consumer must store 0xffffffff (bits 64-95 of -1). Actual: `OpUConvert`, so a
> conforming consumer computes limb 0 = 0x000000000000ffff and a fill `%28` of 0, and
> stores 0. (Observed on Metal through the same translator lowering by the
> specification: a kernel built around this widening disagrees with the CPU for every
> negative `s`, and the nvptx build of the same source agrees.) `-target
> spirv64-opencl` emits `OpUConvert %u64 %s` with `%s` typed `u16`.
>
> Suggested fix: in the integer branch, pick `.OpSConvert` when
> `src_scalar.isSignedInt(zcu)` and `.OpUConvert` otherwise (a narrowing or
> equal-width conversion truncates or reinterprets either way). Callers that widen a
> signed value and want its bits zero-extended (the packed-pointer store in `airStore`,
> line 7521, which shifts the widened field into the host integer without masking it)
> should then convert from the field's unsigned type instead.

### What Zig's SPIR-V backend cannot compile

Besides the operators above, Zig refuses module-scope constant tables that a pointer is
taken to: both `tools/crosscheck/regress/` bodies (`pointer_phi_join.zig`,
`pointer_phi_select.zig`) stop with `TODO (SPIR-V): pointer to constant
'pointer_phi_join.tab'`, so the two pointer-phi regressions run through nvptx only (both
match there). `tools/crosscheck/known/mul_sat.zig` stops at `implement AIR tag mul_sat`.
The generator's programs never take a pointer to a constant, so nothing else in the
sweeps is affected.

### Results

<!--RESULTS-START-->
Apple M3, macOS 26.3, Zig 0.18.0-dev.253, 2026-10-21. Every seed below is 4,096 GPU/CPU
comparisons.

**Sweeps.** `CROSSCHECK_CACHE_EVERY=5 CROSSCHECK_ARGS="-Dcrosscheck-spirv=true"
tools/crosscheck/sweep.sh 1 20`: 20 match. The same with `-Dcrosscheck-floats=true`: 20
match (seeds 10, 14 and 15 were first `ERROR` with `NoSpaceLeft` while the disk was
full; the whole float sweep was rerun on a disk with room, with the workaround's final
form below, and all 20 matched, as did the nvptx runs of float seeds 14, 15 and 17-20).
"Redrawn" is the number of operator draws `--spirv` replaced; the nvptx columns run the same `--spirv` body through the nvptx front end
(`zig build crosscheck -Dseed=N [-Dcrosscheck-floats=true] -Dcrosscheck-body=<the
--spirv body>`), so both front ends compile the same program on the same inputs:

| seed | integer: redrawn | SPIR-V | nvptx | float: redrawn | SPIR-V | nvptx |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 10 | match | match | 9 | match | match |
| 2 | 8 | match | match | 0 | match | match |
| 3 | 29 | match | match | 14 | match | match |
| 4 | 10 | match | match | 3 | match | match |
| 5 | 1 | match | match | 3 | match | match |
| 6 | 8 | match | match | 11 | match | match |
| 7 | 5 | match | match | 9 | match | match |
| 8 | 12 | match | match | 6 | match | match |
| 9 | 17 | match | match | 7 | match | match |
| 10 | 19 | match | match | 4 | match | match |
| 11 | 18 | match | match | 10 | match | match |
| 12 | 19 | match | match | 8 | match | match |
| 13 | 12 | match | match | 3 | match | match |
| 14 | 6 | match | match | 3 | match | match |
| 15 | 7 | match | match | 12 | match | match |
| 16 | 31 | match | match | 6 | match | match |
| 17 | 3 | match | match | 16 | match | match |
| 18 | 16 | match | match | 15 | match | match |
| 19 | 2 | match | match | 4 | match | match |
| 20 | 15 | match | match | 4 | match | match |

Before the workaround above, integer seed 2 was 1,797/4,096 mismatched through SPIR-V
(and through `crosscheck-apple`) and matched through nvptx; with it, it matches. The
reducer judges through the same path: `crosscheck-reduce --seed=2 --spirv
--no-zig-workarounds` (what `zig build reduce -Dseed=2 -Dcrosscheck-spirv=true
-Dspirv-zig-workarounds=false` runs) shrank the program from 17 statements / 143 nodes
to `return t16u(a);` in 18 candidates, every one built with `-Dcrosscheck-spirv=true
-Dspirv-zig-workarounds=false`. Float
seed 2 drew no excluded operator, so its program is also the nvptx bench's float seed 2.

**The recorded findings** (`zig build crosscheck [-Dcrosscheck-spirv=true]
-Dcrosscheck-body=tools/crosscheck/known/<file>`, default seed 1):

| reproducer | SPIR-V | nvptx | why |
| --- | --- | --- | --- |
| add_overflow_bit.zig | match | 1,990 mismatched | Zig's SPIR-V backend computes the overflow bit with sign comparisons (`icmp slt` of the operands and the sum), not `llvm.sadd.with.overflow`; the finding is in Metal's lowering of that intrinsic, so it does not arise |
| sub_overflow_bit.zig | match | 2,103 mismatched | the same for `llvm.ssub.with.overflow` |
| fma_constant_product_flush.zig | 223 mismatched | 223 mismatched | GLSL.std.450 Fma with two constant multiplicands becomes the same `llvm.fma`, and Metal folds it the same way |
| hdiv_constant.zig | 15 mismatched | 15 mismatched | OpFDiv of a half by a constant: the same `fdiv half` |
| round_negative_zero.zig | match | match | worked around in the assembler (`llvm.round` expansion), which GLSL.std.450 Round reaches too |
| rem_inexact.zig | match | match | worked around (`frem` -> `air.fmod`), reached by OpFRem |
| fmod_half_negative_zero.zig | match | match | worked around (`copysign` for half) |
| mul_sat.zig | does not compile: `TODO (SPIR-V): implement AIR tag mul_sat` | match (worked around) | |
| spirv_zig_unsigned_widen.zig | match; 1,801 mismatched with `-Dspirv-zig-workarounds=false` | match | Zig's SPIR-V backend, above |
| spirv_zig_signed_widen.zig | match; 1,232 mismatched with `-Dspirv-zig-workarounds=false` | match | the mirror case, above |

Through `crosscheck-apple -Dcrosscheck-spirv=true`, where Apple's assembler builds the
library from spirv-splice's `out.ll` (which still has `llvm.round` and `frem`, as the
Zig path's printed IR does): round_negative_zero 949 mismatched, rem_inexact 1,067,
fma_constant_product_flush 223, hdiv_constant 15, the nvptx counts exactly; integer seed
3 and float seed 4 match. So every Metal finding that the SPIR-V path's IR contains
reproduces through it with the same count: they are independent of the front end.
(The SPIR-V and `crosscheck-apple` rows were rerun after the workaround gained its
signed-result condition, with the counts above.)

**Regress bodies**: `pointer_phi_join.zig` and `pointer_phi_select.zig` do not compile on
Zig's SPIR-V backend (`TODO (SPIR-V): pointer to constant`, above); both match through
nvptx.
<!--RESULTS-END-->

**Rerun after the `convergent` attributes and the printed-IR renames** (2026-10-21):
`CROSSCHECK_CACHE_EVERY=5 tools/crosscheck/sweep.sh 1 5` in all four modes (nvptx and
`-Dcrosscheck-spirv=true`, integer and `-Dcrosscheck-floats=true`): 20 of 20 match. Every
`tools/crosscheck/known/` body through `crosscheck` and `crosscheck-apple`, nvptx and
SPIR-V, gives the counts above and in [crosscheck.md](crosscheck.md#findings): nvptx
add 1,990 / sub 2,103 / fma 223 / hdiv 15 both ways; round, rem, mul_sat and the half
fmod match through ours and fail through Apple's assembler (949, 1,067, the
MTLCompilerService crash, 672); SPIR-V the table above, with 1,232 / 1,801 mismatched for
the two widenings under `-Dspirv-zig-workarounds=false`.

**Rerun after stage 4** (2026-10-21, with the barrier/atomic lowering and the
uniformity analysis's private-variable rule in place): `CROSSCHECK_CACHE_EVERY=5
CROSSCHECK_ARGS="-Dcrosscheck-spirv=true" tools/crosscheck/sweep.sh 1 20`: 20 match;
the same with `-Dcrosscheck-floats=true`: 20 match. Through `-Dcrosscheck-spirv=true`:
spirv_zig_unsigned_widen and spirv_zig_signed_widen 4096/4096 match (1,801 and 1,232
mismatched with `-Dspirv-zig-workarounds=false`), fma_constant_product_flush 223 and
hdiv_constant 15 mismatched: the counts above.
## Reproducing every claim

From the repository root, on an Apple-silicon Mac (each command prints its own verdict):

| claim | command |
| --- | --- |
| the lowering produces the IR the tables above say, refuses what they say, and the uniformity analysis accepts and refuses the barriers above | `zig build test` |
| every probe and corpus kernel builds or is refused exactly as EXPECTED.txt records, with no Metal error | `tools/spirv/run_probes.sh <scratch dir> [--fast-math=true]` (delete the directory afterwards) |
| the probes compute what their instruction means | `zig build spirv-splice && zig-out/bin/spirv-probe-run [--fast-math=true]` |
| the corpus builds as recorded, the thirteen known-answer kernels match (Collatz, reverse, counters, 64-bit min/max, device float add, threadgroup float exchange, SIMD-group reverse, the reduction and atomics in glslang's shape, DXC's reflection shape, Zig's control flow) and the threadgroup float add still computes a wrong sum | `zig build spirv-corpus [-Dmetal-fast-math=true]` |
| threadgroup float add is Metal's miscompile, whoever assembles it | `zig-out/bin/spirv-splice --allow-metal-miscompiles tools/spirv/corpus/hand-workgroup-atomics.spvasm ours.metallib x.ll` (x.ll holds `atomicrmw fadd ptr addrspace(3) @ftg, float 0x3FF0000000000000 monotonic`), `xcrun metal -Xclang -opaque-pointers x.ll -o apple.metallib`, then dispatch `atomic_fadd_tg` from each library (4096 threads, threadgroups of 64, a zeroed 4096-word buffer at index 2) and decode `out[gid] = old * 128 + final`: both give final 0 in every workgroup and old values 0..31 twice; `atomic_fxchg_tg` from apple.metallib gives a final value and 64 distinct old values per workgroup. MSL: `threadgroup atomic_float f; atomic_fetch_add_explicit(&f, 1.0f, memory_order_relaxed);` fails with "no matching function" under `-std=metal3.0`, `metal3.2`, `metal4.0` |
| one module: its library, manifest, IR; Metal's verdict; Apple's tools on it | `zig-out/bin/spirv-splice --manifest=m.txt <in.spvasm> x.metallib x.ll`, `zig-out/bin/metallib-check --kernels=<a,b> x.metallib`, `xcrun metal-objdump -d x.metallib`, `xcrun metal -Xclang -opaque-pointers x.ll -o apple.metallib` and metallib-check on that. One refused entry point refuses the whole module (no library, no `x.ll`); for hand-workgroup-atomics, whose atomic_fadd_tg is refused as a Metal miscompile, add `--allow-metal-miscompiles` |
| a 64-bit `atomicrmw add` crashes MTLCompilerService whoever assembles it | `zig-out/bin/spirv-probe probes <dir>`, then `zig-out/bin/spirv-splice <dir>/atom_Load_u64_dev.spvasm x.metallib x.ll`, replace the `load atomic i64, ptr addrspace(1) %P monotonic, align 8` line of x.ll with `atomicrmw add ptr addrspace(1) %P, i64 %V monotonic, align 8` (%V the loaded input), `xcrun metal -Xclang -opaque-pointers x.ll -o add.metallib`, `zig-out/bin/metallib-check --kernels=atom_Load_u64_dev add.metallib`: `FAIL ... XPC_ERROR_CONNECTION_INTERRUPTED` |
| `atomicrmw fmin` is unknown to Metal's LLVM | the same with hand-workgroup-atomics' x.ll and `atomicrmw fadd` -> `fmin`: Apple's front end stops with "expected binary operation in atomicrmw" |
| the reverse test detects a missing barrier | copy hand-workgroup-atomics.spvasm without tg_reverse's OpControlBarrier into a scratch directory with that file's EXPECTED.txt rows and run `zig-out/bin/spirv-corpus <dir> <dir>/EXPECTED.txt zig-out/bin/zig_kernels.spv`: tg_reverse MISMATCH |
| the crosscheck bench through SPIR-V | the commands of [the bench section](#the-crosscheck-bench-through-spir-v) |
| the Zig path's library | `zig build metallib && shasum -a 256 zig-out/bin/default.metallib` (`83798793…`: `5a179469…` until the assembler wrote `convergent` attributes and my_shader.zig gained barrierJoinKernel; with the attributes left out and without that kernel the library is byte-identical to `5a179469…`), `zig build check -- zig-out/bin/default.metallib` (45 ok) |
| the barrier needs `convergent` | `zig-out/bin/spirv-probe-run --only=barrier_` and `zig build check -Dshader-optimize=debug -- zig-out/bin/default.metallib` (barrierJoinKernel) pass; with `attributesFor` in tools/air/assembler.zig returning `.none` both fail (16 of 16 / 32 of 32 totals 0x40000021) |
| Apple's assembler builds the printed IR into a library that computes the same | `zig-out/bin/spirv-probe-run [--fast-math=true] --apple=<scratch dir>` ([table](#the-zig-shape-modules-and-apples-assembler)); for the Zig path `xcrun metal -Xclang -opaque-pointers zig-out/bin/shader.air.ll -o apple.metallib`, `zig build check -- apple.metallib` (45 ok) |

## Tests

`zig build test` runs, in the `tools/spirv_splice.zig` test module: the lowering of
every instruction family against expected IR text (each lowered module is also
assembled to bitcode), the interface and manifest (binding order, access, builtins,
`air.struct_type_info` with Offset padding, push constants, unbound buffers, Private
and Workgroup variables, BufferBlock SSBOs, constant tables), refusals with the quoted
instruction, the review's malformed-instruction reproductions, the Zig testdata
binaries, every probe and execution test (lowered and assembled, no GPU), the
reference's hand-computed lanes, spirv-splice's option parsing, and a fuzz test that
truncates and bit-flips the testdata and corpus modules (560 mutants per run; local
runs of 42 000 mutants each, with four seeds, found the crash sites now refused and
end clean): every
mutant must come back as an error, never a crash. It also compiles spirv-probe and
spirv-probe-run. In reader.zig: 300 entry points of one function with 300 execution
modes decode inside a 4 MB allocator (linking the modes once after decoding; the
earlier per-mode copy needed E * M^2 / 2 words, 54 MB here and gigabytes from an
80 KB module), the supplemented enumerant tables (no entry shadows a vendored value,
bit enums in ascending single-bit order, UserTypeGOOGLE, Sampled1D, RoundingModeRTZ by
name), DXC's `OpDecorateString UserTypeGOOGLE` decoding with its string, an unknown
decoration with a parameter refused by name, and a vendor MemoryAccess bit's id
parameter decoded after Aligned's literal.

Stage 4 adds: the barrier and fence lowering for every scope and flag combination
(including the device-scope fence before a Workgroup barrier with Device memory scope,
the fence at the memory scope before a Subgroup barrier with a Workgroup or Device
memory scope, exactly one fence there, none for a Subgroup or Invocation memory scope
or a barrier that orders nothing, nothing for an OpMemoryBarrier that orders nothing, and the refusals: a Device
execution scope, graphics memory, two ordering bits, unknown bits, a non-constant
scope); every atomic instruction on device and threadgroup pointers with each ordering
(AcquireRelease narrowed for loads and stores, compare-exchange's operand order and
failure ordering), 64-bit load/store/min/max and float add/exchange, and the refusals
(Function memory, a Release load, an Acquire store, a float IAdd, an integer FAdd,
a threadgroup float FAdd unless `allow_metal_miscompiles`,
16-bit, 64-bit add and compare-exchange, a used 64-bit min/max result, 64-bit min/max
on threadgroup memory, float min/max); a barrier after a per-thread branch accepted
and one inside it refused by the assembler; glslang's tree reduction accepted and the
same loop bounded by a per-thread value refused; the stage-4 probes (each lowers, or is
refused with its recorded reason). In the assembler's test module
(tools/air/divergence.zig): a loop counter kept in an alloca is uniform when what is
stored is, per-thread when the step is per-thread or a store sits under a per-thread
branch, per-thread when the alloca escapes (passed to a call, indexed, stored), and a
`load atomic` is per-thread. The corpus driver's test module checks its references
(Collatz counts, the control-flow body by hand, the threadgroup float check against
the right answers and against what Metal computes for the add) and the EXPECTED.txt
parsing; spirv-splice's option tests cover `--allow-metal-miscompiles`; the
build also compiles tools/spirv/zig_kernels.zig with Zig's SPIR-V backend.

The `convergent` fix adds: in the assembler's test module, the attribute sets on a
barrier's and a SIMD-group collective's declaration and calls, `convergent` on a helper
that reaches a barrier and on the call to it, none on `air.atomic.fence` or on a helper
that does not; chains of 3000 dependent adds, pointer selects, getelementptrs and
insertvalues assembling with a type anchor every 256 levels (none at 10); and the
operand the Builder's `typeOfWip` follows for every instruction kind that recomputes
its type. In intrinsics.zig, `printForApple` on a module with every rename (powi's
converted exponent, a duplicate `llvm.rint` declaration dropped), the kept workaround
forms, convergent declarations, calls and helpers, fast-math spellings, attribute
group numbers after the text's own, and a module it must leave byte for byte. The
execution tests barrier_workgroup_counter and barrier_shared_flag run on the GPU
(spirv-probe-run), and metallib-check dispatches barrierJoinKernel.

spirv-splice, spirv-probe-run and every lowered module in these tests assemble with
the assembler's `check_operand_types` on: each `TYPE VALUE` operand must have the type the value was
defined with (an address space included), so a lowering bug that prints the wrong
pointer space is a syntax error rather than bitcode that assembles. It is off for
air-splice, whose text is relocated by design (address spaces 4/101 become 2, 5
becomes 0), and only the operands parsed as `TYPE VALUE` are checked (a binary
operation's right-hand side and phi incomings keep their own type handling); a test
shows a deliberately misprinted store caught with it on and assembling with it off.

The fixes of the review's lowering defects (round 1) add, in tools/spirv/lower.zig:
atan / atan2 followed by `llvm.copysign` at strict f32 and not under fast math or for
half; the bitcast access chain's fresh GEP and an OpCopyObject's continued one; one
`[512 x i8]` threadgroup global for two Workgroup Block variables and
`threadgroup_bytes` 512; the tight forms of scalar-layout `P[]` at stride 12 and `{
Inner; float @12 }` in loads, stores and a struct constant, and the refusal of an
OpPtrAccessChain on such a pointer; helper chains of 70 and 10000 calls lowering in at
most two rounds with the leaf's uvec4 load at `align 4` (the 10000 chain overflowed the
native stack before); a refusal quoting the instruction for a repeated switch literal,
an OpConstantTrue of an integer, a 4294967295-lane OpConstantNull, a branch to a
constant, a use before its definition, a use its definition does not dominate, a phi
missing a parent, a phi value of another type, a variable initializer of another type,
a value of another function and an OpFunction typed by an integer; GL_EXT_buffer_reference's
pointer loads, `inttoptr`, the uvec2 `bitcast`, `ptrtoint` and the manifest's `ulong`
fields, with a Workgroup `OpConvertUToPtr` and a 32-bit pointer bitcast refused; and
glslang's `memoryBarrier()` semantics 0xD48 as `air.atomic.fence(i32 7, ...)`. The six
execution tests above run them on the GPU; spirv-probe-run's runner binds a test buffer
only through `useResource` when it is `unbound` and writes buffer addresses into
another buffer (`exec.Buffer.addresses`).

The fixes of round 2's lowering defects add, in tools/spirv/lower.zig: `load volatile`
/ `store volatile` for a Volatile memory operand (OpLoad, OpStore, OpCopyMemory's
source set), a variable decorated Volatile reached through an OpCopyObject, and
glslang's Volatile members, with plain accesses left plain; the ±0 select after strict
half Sin, Tan and Asin and none for f32 or under fast math; OpenCL normalize and length
of a half vector widened to float (`air.dot.v2f32`, no `air.dot.v2f16`); the tight
form's member indices in chains continued from a scalar-layout element and its tight
whole-struct store; `@f20.0` beside a kernel named `f20`, `@g25.0` beside one named
`g25` (and `@g25.1` when an OpName already says `g25.0`), and entry names starting with
`air.` or `llvm.` refused; a branch to the entry block (conditional or not) and a loop
entered at two blocks refused with the branch quoted, the same loop entered once
lowered; Workgroup, Workgroup Block and Function variables of runtime-sized types
refused; and 1, 3 and 50 nested loops carrying a packed member's pointer lowering in
two rounds with an `align 1` store. `strict_only` in exec.zig marks an execution test
that checks what fast math may change (spirv-probe-run skips it under
`--fast-math=true`).

The review fixes add: the alignment of a packed member through OpSelect, a loop phi,
a helper parameter and a pointer kept in a Function variable (all `align 4`, and
`align 16` with std430 Offsets, where nothing is narrowed), an atomic on a `uint` at
Offset 2 refused, OpBitFieldUExtract's select at Offset = width with Count 0, and the
OpenCL powr / normalize / length / distance expansions next to GLSL's unchanged
Normalize. The reference models OpenCL's edge cases (powr at ±0, ±1, ±inf, NaN and
negative x; length, distance and normalize on unclamped inputs, zero vectors and
infinite elements included), and three execution tests run them on the GPU
(bitfield_full_offset, opencl_edge_math, packed_pointer_flow).
