# SPIR-V front end: plan

Goal: load SPIR-V compute shaders into Metal through this project's assembler and packer,
without Apple's toolchain, so that engines which already emit SPIR-V (and Vulkan games
running through translation layers) can build Metal libraries ahead of time, and
asynchronously, from the same source. Graphics stages come after compute.

## Pipeline

```
module.spv ──tools/spirv/reader.zig──▶ Module (types, constants, decorations, globals, functions)
          ──tools/spirv/lower.zig───▶ LLVM IR text (this assembler's dialect) + runtime manifest
          ──tools/air/assembler.zig ──▶ bitcode ──tools/air/metallib.zig──▶ .metallib
```

The reader decodes the binary (header, instruction stream, every operand through the
grammar tables), the lowerer prints the IR text the assembler already accepts: opaque
pointers with AIR address spaces (1 device, 2 constant, 3 threadgroup), `phi` nodes,
calls to `air.*` intrinsics where SPIR-V has a builtin, and a manifest for the kernel's
interface. Everything after the text, the assembler, metadata and container, is shared
with the Zig path, so the Metal findings and workarounds apply to SPIR-V shaders too.

## Status

**Done (S1): the decoder, the text tools and the corpus.**

- `tools/spirv/spec.zig`: the SPIR-V 1.6 grammar tables, vendored from the Zig compiler
  (`src/codegen/spirv/spec.zig`, commit f2f0d97a9b, MIT; the header says so). Opcodes,
  operand kinds with their categories and enumerants, the GLSL.std.450 and OpenCL.std
  tables, every enum. It compiles unchanged on the pinned nightly. Zig's generator keeps
  only part of the grammar's enumerants (about 380 missing, nearly all vendor-tagged,
  e.g. DXC's `UserTypeGOOGLE`); `tools/spirv/spec_extra.zig` adds them from the Khronos
  grammar and `reader.enumerants` merges the two for reader, dis and asm.
- `tools/spirv/reader.zig`: bytes to `Module`. Header in either endianness; operands
  decoded by the grammar (enum parameters such as `Aligned 4`, bit enums, strings,
  variadic tails, composite pairs); context-dependent numbers by the result type's width
  (64-bit constants and 64-bit `OpSwitch` cases take two words); `OpExtInst` operands by
  the set's table; tables of types, constants (scalars as raw bits, composites, null,
  undef, spec constants), decorations per id and member, names, entry points with their
  execution modes, extended-instruction imports, globals, functions with parameters and
  blocks. Decoration groups (`OpDecorationGroup` / `OpGroupDecorate` /
  `OpGroupMemberDecorate`, which SPIRV-LLVM-Translator emits) are expanded onto their
  targets, so `NonWritable` or `DescriptorSet 1` applied through a group is seen on the
  variable. `error.Malformed` names the word offset.
- `tools/spirv/dis.zig`: spirv-dis style text (`%N` or `%name` ids, enumerant names,
  GLSL/OpenCL instruction names, typed literals). `zig build spirv-dis -- file.spv`.
- `tools/spirv/asm.zig`: the text assembler for the corpus (numeric ids kept, symbolic
  ids numbered after them, forward references, typed literals, `; Bound:` honoured; an id
  that would push the bound past `maxInt(u32)` is a syntax error, not a crash).
- `tools/spirv/corpus/`: 16 naga snapshots (wgpu, MIT), the Zig binaries as text (four
  in S1; zig-narrow-ints since: five), hand-written files (six in S1: one kernel per
  GLSL.std.450 / OpenCL.std instruction, scalars, vectors, control flow and the
  interface, workgroup memory and atomics; hand-glslang-stage4 and hand-dxc-reflect
  since: eight); see its README.
- Tests (`zig build test`): the reader on hand-assembled word arrays covering every
  operand category, two functions with a phi loop and a 64-bit switch, and malformed
  inputs; the disassembler on the same; `asm -> read -> dis -> asm` byte-identical on
  every corpus file; `dis -> asm` reproducing the Zig binaries byte for byte (all five
  today, `tools/spirv/asm.zig`).

**What Zig's SPIR-V backend emits** (facts the lowering is written against; from
`tools/spirv/testdata/*.spv`, reproducible with the commands in the corpus headers):

- Vulkan: `OpCapability Shader | Int64 | PhysicalStorageBufferAddresses [| Float16]`,
  `OpExtension "SPV_KHR_physical_storage_buffer"`, `OpMemoryModel PhysicalStorageBuffer64
  GLSL450` (naga: `Logical GLSL450` with `SPV_KHR_storage_buffer_storage_class`),
  `OpEntryPoint GLCompute`, `OpExecutionMode LocalSize`. Debug: `OpString`, `OpSource Zig
  2304`, `OpSourceExtension`, `OpName`/`OpMemberName`, `OpLine`.
- Decorations: `Block`, member `Offset`, `ArrayStride` (on array types and on pointer
  types), `Binding`, `DescriptorSet`, `NonWritable`, `BuiltIn`.
- Interface: `OpVariable Input` for builtins (read with `OpLoad` or
  `OpInBoundsAccessChain` + `OpLoad`), `Uniform` for `*addrspace(.uniform) const T`,
  `StorageBuffer` Block structs via `@extern(..., .decoration = .{ .descriptor = ... })`,
  `Workgroup` for `extern var tile: [64]i32 addrspace(.shared)`, `UniformConstant` /
  `CrossWorkgroup` pointer parameters in the old shape.
- Bodies: every local is an `OpVariable Function` at the top of the entry block,
  `OpStore`/`OpLoad ... Aligned N`, `OpInBoundsAccessChain` with `OpUConvert`ed u64
  indices (`OpPtrAccessChain` in the old shape), `OpFunctionCall` to helpers, an extra
  `OpLabel; OpUnreachable` block at the end of every function, `OpSelectionMerge` /
  `OpLoopMerge`, `OpSwitch`, many `OpPhi` of u32 tags followed by `OpIEqual` dispatch
  (Zig's labeled-break lowering), `OpBitcast` int<->float, `OpExtInst GLSL.std.450`.
  Blocks are already in dominator order.
- OpenCL: `Kernel | Addresses | Int64`, `Physical64 OpenCL`, `OpEntryPoint Kernel`, all
  integers unsigned, `CrossWorkgroup` pointers, `OpInBoundsPtrAccessChain`, extra
  `OpBitcast` between identical pointer types.
- Does not compile on the backend (`TODO (SPIR-V): implement AIR tag ...`): `+| -| *|`,
  `@popCount`, `@clz`, `@ctz`, `@byteSwap`, `@bitReverse`; f16/f64 need
  `-mcpu=generic+float16` / `+float64`.

**Done (S2): the lowering, the runtime manifest, `spirv-splice`, runtime validation.**
See [spirv.md](spirv.md) for the mapping tables and the results.

- `tools/spirv/lower.zig`: stages 1-3. Types (bool as `i8` in memory, packed structs
  where the Offset decorations differ from AIR's natural layout), the interface
  (Block buffers, Uniform and PushConstant blocks, builtins, entry-point parameters
  OpenCL style, Workgroup globals, Private allocas; helpers take the variables they
  touch as extra parameters), every arithmetic, comparison, conversion, bit, vector,
  composite, memory and control-flow instruction of §4, GLSL.std.450 and OpenCL.std,
  refusals that quote the instruction (core instructions outside §4, such as matrix
  arithmetic and OpCopyLogical, are refused: spirv.md's "not supported" table). Phis are rendered after their function, so a
  back edge may name a value an expansion defines later.
- `tools/spirv/manifest.zig`: a kernel's interface at runtime and its conversion to
  `metadata.FunctionMetadata`; `assembler.assemble` and `metadata.lowerModule` /
  `printManifestWith` take it at runtime (default.metallib unchanged, byte for byte).
  The buffer metadata reproduces Apple's for the MSL twin (`air.buffer_size` for a
  fixed-size struct, `air.struct_type_info` with array element size and length).
- `tools/spirv_splice.zig` (`zig build spirv-splice`), `metallib-check --kernels=a,b`.
- Runtime validation: 670 one-instruction probe modules (`tools/spirv/probes.zig`) and
  the 145 corpus entry points, each its own library, all built into pipelines
  (`tools/spirv/run_probes.sh`, table in `tools/spirv/corpus/EXPECTED.txt`) and the
  probes executed against a CPU reference (`spirv-probe-run`, `tools/spirv/reference.zig`)
  in strict and fast-math mode, plus 18 whole-module execution tests with fixed answers
  (`tools/spirv/exec.zig`); Apple's assembler accepts the printed IR of the
  Zig-shape modules and its libraries pass the same check. Four Metal crashes found
  and avoided: vector `llvm.uadd/usub.with.overflow`, `mul i128`, `air.fast_ldexp.f16`,
  and a runtime-indexed padded struct of a private array loaded whole or indexed again.
- Review round 1 (fixed): SPIR-V 1.0 SSBOs (Uniform + BufferBlock) are device memory;
  descriptor arrays and stores into constant memory are refused; OpenCL program-scope
  `__constant` tables are `addrspace(2)` constant globals; SaturatedConversion and
  FPRoundingMode are implemented (or refused where Metal cannot honour them) and every
  other semantic decoration on a result is refused; OpSelect of composites goes member
  by member; phis get one entry per CFG edge; NonSemantic.* instructions are dropped;
  `vec3` + scalar at Offset 12 (and 12-byte vec3 arrays) are `[3 x T]` in memory;
  malformed input is refused instead of crashing (a fuzz test in `zig build test`).
- Review rounds 2-3 (fixed): OpBitFieldUExtract at Offset = width with Count 0 selects
  0 instead of masking a poison shift (Metal dropped the store when the operands were
  constants); OpenCL powr, normalize, length and distance follow OpenCL C's edge cases
  and do not overflow in the sum of squares (GLSL's keep the plain forms); the
  alignment a packed member gives a pointer follows it through OpSelect, phis, helper
  parameters and memory (the module is lowered again until no alignment narrows), and
  an atomic through an under-aligned address is refused. Exec tests
  bitfield_full_offset, opencl_edge_math and packed_pointer_flow.

- Final review, round 1, lowering (fixed): strict f32 atan / atan2 keep the sign of a
  zero result (`llvm.copysign`; Metal's precise functions return +0); an access chain
  on a pointer bitcast to another pointee starts its own GEP; Workgroup Block
  variables (SPV_KHR_workgroup_memory_explicit_layout) alias in one threadgroup
  allocation; scalar-layout structs ending in a vec3 have a tight form (a nested member
  followed at Offset 12, arrays at stride 12); the call graph is walked without
  recursion (a 10000-deep helper chain overflowed the stack) and alignment facts are
  used in the round that learns them (a 64-deep chain ran out of rounds); malformed
  bodies (branch targets, switch literals, phis, dominance) and constants are refused
  with the instruction quoted; glslang's `memoryBarrier()` (AtomicCounterMemory)
  lowers; GL_EXT_buffer_reference pointers in buffers and from integers lower to
  `ptr addrspace(1)`, `inttoptr`, `ptrtoint`. Six execution tests, docs/spirv.md.
- Final review, round 2, lowering (fixed): Volatile memory operands, variables and
  members give `load volatile` / `store volatile` (Metal hoisted a plain polling load
  out of its loop); strict half sin, tan and asin return ±0 for ±0 (Metal's precise
  half functions give +0 for -0); OpenCL normalize, length and distance of half
  vectors are computed in float (subnormal elements gave inf / NaN); an access chain
  continued from a tight scalar-layout element indexes the tight form; entry points
  named like generated symbols (`g25`, `f20`) no longer collide, `air.` / `llvm.` names
  are refused; a branch to the entry block, irreducible control flow (both crash
  MTLCompilerService) and runtime arrays outside buffer blocks are refused; a function's
  pointer phis settle together (K nested loops took K + 1 whole-module lowering rounds)
  and validateBody computes each block's successors once (a 16000-case switch was
  quadratic). Six execution tests, docs/spirv.md.

**Done (S3): the crosscheck bench through SPIR-V.** See
[spirv.md](spirv.md#the-crosscheck-bench-through-spir-v).

- `zig build crosscheck -Dcrosscheck-spirv=true`: the body is compiled by Zig's
  self-hosted SPIR-V backend inside `src/engine/crosscheck_harness_spirv.zig` (the
  standard Vulkan compute shape: Block buffers at bindings 0-2, a Uniform block at 3,
  GlobalInvocationId, LocalSize 64), spirv-splice builds the library, and the unchanged
  driver runs it. `crosscheck-apple`, `reduce`, `sweep.sh` and the reproduce hints carry
  the flag; the generator's `--spirv` leaves out exactly the operators the backend
  cannot compile (saturating arithmetic, `@popCount`, `@clz`, `@ctz`, `@byteSwap`,
  `@bitReverse`; scalar and vector).
- The first sweep found a bug in Zig's SPIR-V backend: an unsigned value widened to a
  signed type is emitted as OpSConvert, a sign extension (`buildConvert` picks the
  opcode by the result's signedness). spirv-splice undoes it for modules whose Generator
  is Zig, on a widening OpSConvert from an unsigned-typed operand to a signed-typed
  result only (`--zig-workarounds`, on by default, the way Mesa keys glslang
  workarounds on the generator id; Zig's OpenCL flavour types every integer unsigned,
  so its `OpSConvert %u32 %u16` stays a sign extension);
  `tools/crosscheck/known/spirv_zig_unsigned_widen.zig` measures it, and docs/spirv.md
  has the upstream report text. A review found the mirror case: a signed value widened
  into an i128 limb gets OpUConvert, a zero extension, and loses its sign; spirv-splice
  sign-extends a widening OpUConvert of a signed-typed operand in a Zig module only when
  its result is sign-filled the way Zig's composite-integer cast does it
  (`tools/crosscheck/known/spirv_zig_signed_widen.zig`, exec test zig_convert_wide).
- 20 integer and 20 float seeds match through SPIR-V, and the same bodies through
  nvptx; the Metal findings reproduce through SPIR-V with the nvptx counts where the
  IR has the same construct (fma constant product, f16 division by a constant; the
  three worked-around ones through `crosscheck-apple`). The overflow-bit findings do not
  arise: Zig's SPIR-V backend computes the bit with sign comparisons instead of
  `llvm.sadd/ssub.with.overflow`.

**Done (S4): barriers, atomics and the corpus execution driver.** See
[spirv.md](spirv.md#barriers-and-atomics) and [spirv.md](spirv.md#the-corpus).

- `tools/spirv/lower.zig`: OpControlBarrier (Workgroup / Subgroup execution scope) ->
  `air.wg.barrier(flags, 1)` / `air.simdgroup.barrier(flags, 4)` with the memory flags
  of the semantics' storage-class bits (and a device-scope `air.atomic.fence` first
  when the memory scope is Device and device memory is ordered; for a Subgroup barrier
  with a Workgroup or wider memory scope, a fence at that scope first, because the
  SIMD-group barrier orders memory only within the SIMD-group); OpMemoryBarrier ->
  `air.atomic.fence(flags, seq_cst, scope)`; every OpAtomic* on device and threadgroup
  pointers (BufferBlock Uniform pointers included) -> native `atomicrmw` / `cmpxchg` /
  `load atomic` / `store atomic` with the ordering of the semantics operand. Workgroup
  variables were already threadgroup globals (S2).
- Metal's limits, each found by a pipeline probe and refused by name: 64-bit atomics
  are load, store and device min/max with the result unused (every other 64-bit
  read-modify-write, a used min/max result and threadgroup min/max crash
  MTLCompilerService, also when Apple's assembler builds the same IR); `atomicrmw
  fmin/fmax` is unknown to Metal's LLVM. One found by execution: `atomicrmw fadd` on
  threadgroup memory builds but computes a wrong sum (64 threads adding 1.0 leave
  0.0; Apple's assembler too; MSL has no threadgroup `atomic_float`), so threadgroup
  OpAtomicFAddEXT is refused and `zig build spirv-corpus` re-runs it to notice a fix.
- tools/air/divergence.zig: the uniformity analysis follows private variables (an
  alloca whose address only feeds plain loads and stores holds what was stored in it)
  and makes `load atomic` per-thread, so glslang's and Zig's loop counters in Function
  variables keep a uniformly bounded loop's barrier legal; default.metallib unchanged.
  Final review round 2 (shared code, fixed): it also follows a variable through
  `getelementptr` (OpAccessChain into a local struct or vector) and into helpers that
  take it by pointer (glslang -V's calling convention), so a tiled loop bounded by
  `ceilDiv(pc.n, 64u)`, `p.n` or `r.x` keeps its barrier (exec tests uniform_bound_*);
  post-dominance and loops come from dominator trees instead of n x n tables, functions
  are re-analysed from a worklist, and the assembler and the analysis find functions in
  hash maps (a 1000-block chain behind one divergent branch: 17 s -> 0.04 s; 1600
  sequential ifs: 111 s -> 0.3 s; 20 000 helpers: 113 s -> 2.6 s); default.metallib
  unchanged. Refusals from the SPIR-V path now start `spirv-splice:` and no longer blame
  Zig.
- 118 stage-4 pipeline probes; 788 probes and 152 corpus entry points in all (940
  pipelines: 899 ok, 41 refused as expected, no Metal error, strict and fast math;
  since hand-dxc-reflect, 153 entry points and 941 pipelines, 900 ok in EXPECTED.txt; since
  the docs review's round 2, 821 probes, 703 of them executed (the integer comparisons,
  SNegate / Not / BitCount / BitReverse and the GLSL.std.450 integer functions at i8 and i16
  too, as docs/spirv.md claimed), 974 pipelines, 933 ok).
- `zig build spirv-corpus` (tools/spirv/corpus_driver.zig): the corpus in one process
  against EXPECTED.txt, plus thirteen kernels executed with known answers: naga's Collatz,
  the threadgroup reverse and the atomic counter written by hand and in Zig
  (tools/spirv/zig_kernels.zig through Zig's SPIR-V backend, atomics as SPIR-V inline
  assembly), 64-bit min/max, device float atomic add, threadgroup float
  store/exchange/load, a SIMD-group reverse through a Subgroup barrier, a tree reduction
  and atomicAdd/atomicCompSwap in glslang's shape, written by hand
  (tools/spirv/corpus/hand-glslang-stage4.spvasm), DXC's `-fspv-reflect` shape
  (hand-dxc-reflect, added after S4), and cf.zig's control flow made race-free against its CPU twin. All match; the
  threadgroup float add, built anyway, still computes the wrong sum.

**What remains** (each is refused by name today):

- **Graphics stages** (stage 5): Vertex and Fragment entry points, Input / Output
  variables by Location, `BuiltIn Position` / `FragCoord` / `FrontFacing` / ..., the
  render-pipeline metadata the Zig path already writes for its vertex and fragment
  shaders, OpKill / OpDemoteToHelperInvocation -> `air.discard_fragment`.
- **Images and samplers**: OpTypeImage / OpTypeSampler / OpTypeSampledImage variables
  as `[[texture(n)]]` / `[[sampler(n)]]` parameters, OpImageSample* / OpImageRead /
  OpImageWrite / OpImageQuerySize -> the `air.sample_texture_*` / `air.read_texture_*`
  family (docs/air-format.md 2.4), OpImageTexelPointer atomics.
- **OpArrayLength**: Metal passes a kernel no buffer length; a host convention (the
  length in a constant buffer at a fixed index, as SPIRV-Cross's "buffer size
  buffer") would have to be chosen and written into the manifest.
- **DescriptorSet > 0**: needs a flattening policy (sets to buffer-index ranges, or
  argument buffers) recorded in the manifest.
- **Descriptor arrays** (arrays of Block structs): one AIR buffer parameter is one
  pointer; argument buffers or one parameter per element.
- Smaller gaps: 64-bit atomic add / and / or / xor / exchange / compare-exchange (no
  Metal form; a lowering through 32-bit compare-exchange loops would need new blocks
  inside an instruction), float atomic min/max (the same, through a 32-bit
  compare-exchange loop), Workgroup initializers (zero-initialized workgroup memory: a
  prologue store by every thread plus a barrier), Workgroup entry parameters (OpenCL
  `local` arguments: `[[threadgroup(n)]]`), OpSpecConstantOp, and the GLSL.std.450 /
  OpenCL.std functions not mapped (Frexp, Modf, Pack/Unpack, ...).

## Scope, in stages

1. **Compute, scalar integers and floats.** Capabilities Shader/Kernel, Int64, Float16;
   the GLSL450 memory model with `PhysicalStorageBuffer64` (what Zig's own SPIR-V backend
   emits for `spirv64-vulkan`) and the OpenCL model (`spirv64-opencl`); OpTypeInt/Float/
   Bool/Vector/Pointer/Struct/Array/RuntimeArray/Function; OpConstant*, OpUndef; OpFunction,
   OpFunctionParameter, OpFunctionCall; OpVariable (Function and Workgroup storage),
   OpLoad/OpStore, Op(InBounds)(Ptr)AccessChain → `getelementptr`; integer and float
   arithmetic, comparisons, conversions, `OpSelect`, bit operations; `OpExtInst` for the
   GLSL.std.450 and OpenCL.std math sets → `llvm.*`/`air.*` as the intrinsic table maps them.
2. **Control flow.** SPIR-V blocks map one to one onto LLVM blocks; `OpPhi` is `phi`;
   `OpLoopMerge`/`OpSelectionMerge` carry no semantics for the assembler and are dropped;
   `OpSwitch` is `switch`. The block-order pass already handles any order.
3. **The interface.** Decorations (`DescriptorSet`, `Binding`, `BuiltIn GlobalInvocationId`
   / `LocalInvocationId` / `WorkgroupId`, `Block`, `Offset`) become the manifest the splice
   needs: each storage-buffer binding a `device` buffer argument, push constants a
   `constant` buffer, builtins the `air.*` attributes. The manifest is generated per
   module, so `air_splice` grows a runtime manifest next to the comptime one.
4. **Vectors and shared memory.** `OpVectorShuffle`, `OpCompositeExtract/Insert/Construct`
   lane by lane (the assembler's vector rules), `Workgroup` variables as threadgroup
   memory, `OpControlBarrier` → `air.wg.barrier`, atomics → native LLVM atomics (which
   Metal accepts; the `air.atomic.*` intrinsics are only needed for the fence). Done.
5. **Graphics.** Vertex and fragment entry points: inputs and outputs by `Location`,
   `BuiltIn Position`/`FragCoord`, textures and samplers (`OpImageSampleImplicitLod` →
   `air.sample_texture_2d.v4f32`).

## Testing, from day one

The crosscheck bench is the test harness. Zig compiles the same generated `body` to SPIR-V
(`-target spirv64-vulkan -fno-llvm`, 796 bytes for a one-line kernel; also
`spirv64-opencl`), the reader turns it into a library, and the driver compares the GPU
with the CPU as it does today: a `-Dcrosscheck-spirv=true` mode, with the reducer and
the Apple attribution step unchanged. Unit tests use hand-assembled modules (there is no
SPIR-V toolchain on the machine; the reader gets a small disassembler for its own
diagnostics). Zig's SPIR-V output for the kernel above uses OpPtrAccessChain, OpUConvert,
function-scope OpVariable with OpLoad/OpStore for every local, OpLine and OpString: stage
1 needs exactly these.

## Asynchronous compilation

Nothing in the pipeline needs the GPU: a host can convert and pack SPIR-V on worker
threads and hand Metal a finished library, overlapping Metal's own pipeline-state
compilation with rendering. The reader is written to be callable as a library function
(bytes in, library bytes out) for that use.

## Risks

- SPIR-V's structured control flow is stricter than LLVM's, which helps; the other
  direction (LLVM `phi` placement for `OpPhi` with many predecessors) is routine.
- Vulkan's 64-bit buffer device addresses and the OpenCL model both become AIR `device`
  pointers; descriptor-set layouts have no AIR equivalent and become buffer indices.
- `PhysicalStorageBuffer64` loads carry explicit alignment; the assembler's `load`/`store`
  take `align`, so this is a matter of printing it.
