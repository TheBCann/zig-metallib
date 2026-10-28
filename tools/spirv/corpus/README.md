# SPIR-V corpus

Text modules (`.spvasm`) that `tools/spirv/asm.zig` assembles, `tools/spirv/reader.zig`
decodes and `tools/spirv/dis.zig` prints back. `zig build test` assembles every file,
disassembles the result and assembles that again, and requires the bytes to match
(`asm.zig`, test "assemble -> read -> dis -> assemble is the identity on every corpus
file"); `zig build spirv-dis -- <file>` prints any of them. The list of files lives in
`asm.zig` (`corpus_files`); a new file goes there too.

`zig build spirv-corpus` (tools/spirv/corpus_driver.zig) builds every entry point of these
files into a compute pipeline in one process, compares the outcomes with EXPECTED.txt and
executes the 13 kernels with known answers (wgsl-collatz; the reverse, counter, 64-bit
min/max, device float add, threadgroup float exchange (atomic_fxchg_tg) and SIMD-group
reverse (sg_reverse) kernels of hand-workgroup-atomics; both kernels of
hand-glslang-stage4; hand-dxc-reflect; the three Zig kernels of
tools/spirv/zig_kernels.zig) against CPU computations.
`tools/spirv/run_probes.sh <dir>` lowers every entry point of these files on its own (and
the one-instruction probes of `tools/spirv/probes.zig`), builds a compute pipeline from each
library and compares the outcomes with [EXPECTED.txt](EXPECTED.txt): one row per kernel,
`<file>:<entry>` (or the probe name), a tab, then `ok`, `refused: <why>` or `Metal: <error>`.
An expected refusal is a pass. See [docs/spirv.md](../../../docs/spirv.md); the per-kernel
structure below exists so that a refusal by Metal's runtime compiler is attributable to one
instruction.

## naga snapshots (`wgsl-*.spvasm`)

Copied verbatim from the wgpu repository, `naga/tests/out/spv/<file>` (the SPIR-V naga
emits for the WGSL tests of the same name under `naga/tests/in/wgsl/`), MIT license, see
[LICENSE.MIT](LICENSE.MIT). Real engine output: `Logical GLSL450` with
`SPV_KHR_storage_buffer_storage_class`, `OpSource` with the WGSL text, `OpLine`,
`OpName`/`OpMemberName`, `OpConstantNull`, structured loops with `loop_bound` guards.

| file | what it exercises |
| --- | --- |
| wgsl-collatz.spvasm | compute: a `while` loop with a phi-free local, helper calls, `OpUMod`/`OpUDiv` guards, runtime array in a Block |
| wgsl-boids.spvasm | compute: two storage buffers, a uniform block, vec2 math, `OpExtInst` Distance/Length/Normalize/FClamp, one structured loop over the particles (one `OpLoopMerge`, `if`/`continue` selections inside it) |
| wgsl-bits.spvasm | `OpBitCount`, `OpBitReverse`, `OpBitFieldUExtract/SExtract/Insert`, the GLSL.std.450 Pack/Unpack instructions (which the lowering refuses) |
| wgsl-bitcast.spvasm | `OpBitcast` between every 32-bit scalar and vector type |
| wgsl-access.spvasm | access chains through structs, arrays, matrices and dynamic indices, `OpArrayLength` |
| wgsl-abstract-types-const.spvasm, wgsl-abstract-types-operators.spvasm | constant folding output at 32 bits only (i32, u32, f32): the first is constants alone (`OpConstant`, `OpConstantComposite` of vectors, a matrix, arrays and a struct; no function), the second a compute entry point with `OpConstantNull`-initialized Function variables, `OpIAdd`/`OpFAdd`/`OpShiftLeftLogical`, helper calls and an `OpControlBarrier` |
| wgsl-6772-unpack-expr-accesses.spvasm | a tiny module (`Int8`): a u32 constant `OpBitcast` to 4 x i8 and 4 x u8, widened with `OpSConvert` / `OpUConvert` to 4 x i32 / u32, then `OpCompositeExtract`; it imports GLSL.std.450 but has no `OpExtInst` |
| wgsl-atomicOps.spvasm, wgsl-atomicOps-int64.spvasm, wgsl-atomicOps-int64-min-max.spvasm, wgsl-atomicOps-float32.spvasm | every `OpAtomic*` on storage and workgroup pointers, 64-bit atomics, `OpAtomicFAddEXT` |
| wgsl-atomicCompareExchange.spvasm, wgsl-atomicCompareExchange-int64.spvasm | `OpAtomicCompareExchange` loops |
| wgsl-atomicTexture.spvasm, wgsl-atomicTexture-int64.spvasm | `OpImageTexelPointer` atomics (which the lowering refuses: stage 5) |

## Zig's SPIR-V backend (`zig-*.spvasm`)

Disassembled by `tools/spirv/dis.zig` from the binaries in `tools/spirv/testdata/`, each
of which `zig build-obj <src> -target spirv64-<os> -fno-llvm -O ReleaseFast [-mcpu=generic+float16 | +int8+int16]`
on Zig 0.18.0-dev.253+16d31ceed reproduces byte for byte (the header comment of each
file has the exact command). The binaries themselves are embedded by the tests, which
require `dis -> asm` to reproduce them exactly.

| file | source | shape |
| --- | --- | --- |
| zig-vulkan-params.spvasm | testdata/k.zig | `kernel(out, in_a, in_b, gid)` with pointer parameters: `PhysicalStorageBuffer64 GLSL450`, `OpPtrAccessChain`, `OpUConvert`, every local an `OpVariable Function` |
| zig-opencl-params.spvasm | testdata/k.zig | the same for `spirv64-opencl`: `Kernel`/`Addresses`, `Physical64 OpenCL`, `CrossWorkgroup`, `OpInBoundsPtrAccessChain`, unsigned integers only |
| zig-crosscheck.spvasm | testdata/desc.zig | the crosscheck kernel in standard Vulkan form: Block structs with `DescriptorSet`/`Binding`/`NonWritable`, a `Uniform` block, `BuiltIn GlobalInvocationId`, `LocalSize 64 1 1`, Zig's phi dispatch for `return` |
| zig-control-flow.spvasm | testdata/cf.zig | loop + switch + labeled-break phis, `OpExtInst` FAbs/Sqrt/Floor/Fma/SMin/SMax, `OpBitcast` int<->float, a `Workgroup` array, `LocalInvocationId`/`WorkgroupId` |
| zig-narrow-ints.spvasm | testdata/narrow.zig | `OpConstant` of negative `i8`/`i16` (sign-extended words 0xFFFFFFFF / 0xFFFFFFFE, SPIR-V 2.2.1) and a large `u8`; `-mcpu=generic+int8+int16`, plain `addrspace(.storage_buffer)` arrays without bindings |

## Hand-written (`hand-*.spvasm`)

Written for this project, one GLCompute (or Kernel) entry point per instruction group;
the header comment of each file lists its kernels and bindings.

| file | kernels |
| --- | --- |
| hand-glsl-std-450.spvasm | one kernel per GLSL.std.450 instruction the lowering maps (50: FAbs ... Ldexp, the integer min/max/clamp/find, Length/Distance/Normalize/Cross) |
| hand-opencl-std.spvasm | the OpenCL.std counterparts (51) in the OpenCL flavour with `CrossWorkgroup` pointer parameters |
| hand-scalars.spvasm | bool locals, integer arithmetic (`OpSMod`/`OpSRem` signs), every comparison, i8/i16/i64 and the conversions, `OpIAddCarry`/`OpISubBorrow`/`OpUMulExtended`/`OpSMulExtended`, bit fields, float arithmetic with every ordered and unordered comparison, `OpIsNan`/`OpIsInf`, f16/f64 conversions |
| hand-vectors.spvasm | 2/3/4/8-lane vectors: construct/extract/insert, shuffles (with an undefined lane), dynamic lanes, `OpVectorTimesScalar`, `OpDot`, `OpAny`/`OpAll`, lane-wise select and conversions |
| hand-control-flow.spvasm | `OpSwitch` with several targets (signed and 64-bit selectors), nested `OpPhi` loops with an early exit, a `PushConstant` block and every kernel builtin, helper chains, `OpUnreachable`, `OpUndef` in a phi |
| hand-workgroup-atomics.spvasm | the threadgroup reverse (`OpControlBarrier`), an atomic counter, every atomic on device and workgroup pointers with each memory semantics, `OpMemoryBarrier`, 64-bit atomic min/max with the results unused, float atomic add |
| hand-glslang-stage4.spvasm | written by hand in the shape glslang -V (Vulkan 1.0, unoptimized) emits, not glslang's output (glslang is not installed here; the header's Generator 0x0008000b is glslang's id, set to imitate it), for a shared-memory tree reduction (a barrier in a loop bounded by `gl_WorkGroupSize.x / 2`, the counter in a Function variable) and for atomicAdd / atomicCompSwap on a BufferBlock SSBO; the GLSL is in its header |
| hand-dxc-reflect.spvasm | the shape DXC emits with `-fspv-reflect` (hand-written, no DXC here): `OpDecorateString` `UserTypeGOOGLE` on every resource and `UserSemantic` on the builtin, SPIR-V 1.0 structured buffers (Uniform + BufferBlock), `OpModuleProcessed`; UserTypeGOOGLE is one of the enumerants tools/spirv/spec_extra.zig adds to the vendored tables; executed: out = in * 3 + 1 |
