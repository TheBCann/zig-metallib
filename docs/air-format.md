# The `.metallib` format, as used here

Apple documents neither the `.metallib` container nor the AIR bitcode dialect. What
follows was reverse-engineered from `xcrun metal` output (metalfe-32023.883, macOS 26.3,
`air64_v28`) and confirmed three ways: by repacking Apple's own bitcode and loading the
result, by loading our own bitcode, and by running every entry point on the GPU
(`zig build check`). Where a field's meaning is unknown, this says so.

All integers are **little-endian**. Every *tag* is `4-byte id, u16 payload length,
payload`.

---

## 1. Container

```
┌─────────────────────────────┐ 0
│ header (88 bytes)           │
├─────────────────────────────┤ 88
│ function list               │   u32 count, then one entry per function
├─────────────────────────────┤
│ header extension            │   HDYN, RLST, UUID, ENDT
├─────────────────────────────┤
│ public metadata             │   one stub per function
├─────────────────────────────┤
│ private metadata            │   one stub per function
├─────────────────────────────┤
│ bitcode                     │   wrapped + 16-byte-padded blob per function
├─────────────────────────────┤
│ dynamic header              │   NAME "<library name>\0", ENDT
├─────────────────────────────┤
│ reflection list             │   u32 count = 0
└─────────────────────────────┘ file size
```

### 1.1 Header (88 bytes)

| offset | type | value | meaning |
| --- | --- | --- | --- |
| 0 | `char[4]` | `"MTLB"` | magic |
| 4 | u16 | `0x8001` | undocumented; looks like platform + flags |
| 6 | u16 | `2` | format version major |
| 8 | u16 | `9` | format version minor |
| 10 | u16 | `0x8100` | undocumented |
| 12 | u32 | `26` | undocumented; matches the macOS SDK major |
| 16 | u64 | | total file size |
| 24 | u64 | `88` | function-list offset |
| 32 | u64 | | function-list size. Apple's value **excludes the final `ENDT`**, and the reader stops at `ENDT` anyway |
| 40 | u64 | | public-metadata offset |
| 48 | u64 | | public-metadata size |
| 56 | u64 | | private-metadata offset |
| 64 | u64 | | private-metadata size |
| 72 | u64 | | bitcode-section offset |
| 80 | u64 | | bitcode-section size |

### 1.2 Function list

`u32 count`, then per function: `u32 entry_size` (**including its own 4 bytes**),
followed by tags and a bare `ENDT` (no length field).

| tag | payload | meaning |
| --- | --- | --- |
| `NAME` | cstring | entry-point symbol, NUL-terminated |
| `TYPE` | u8 | stage: `0` vertex, `1` fragment, `2` kernel |
| `HASH` | 32 bytes | SHA-256 of the function's **padded, wrapped** bitcode blob |
| `OFFT` | 3 × u64 | offsets of this function's public metadata, private metadata and bitcode, each relative to its own section |
| `VERS` | 4 × u16 | AIR version major/minor, then Metal language major/minor — here `2, 8, 4, 0` |
| `MDSZ` | u64 | length of the padded, wrapped bitcode blob |
| `RFLT` | u64 | `0`; reflection-related, unused here |

### 1.3 Header extension

Three tags plus a terminator, 70 bytes total:

| tag | payload | meaning |
| --- | --- | --- |
| `HDYN` | 2 × u64 | offset and size of the dynamic header |
| `RLST` | 2 × u64 | offset and size of the reflection list |
| `UUID` | 16 bytes | library identity. Ours is the first 16 bytes of a SHA-256 over all bitcode blobs, so identical shaders produce byte-identical libraries — the runtime uses it as a cache key |
| `ENDT` | — | terminator |

### 1.4 Metadata sections

Both the public and the private metadata section hold one **stub per function**:
`u32 8` followed by `"ENDT"` (8 bytes, the count includes itself). Metal accepts these
empty stubs; all the reflection information it actually uses lives in the bitcode's
`!air.*` metadata.

### 1.5 Bitcode section

Per function, a **Darwin bitcode wrapper** followed by the LLVM bitcode:

| offset | type | value |
| --- | --- | --- |
| 0 | u32 | `0x0B17C0DE` |
| 4 | u32 | `0` (version) |
| 8 | u32 | `20` (offset of the bitcode) |
| 12 | u32 | unpadded bitcode length |
| 16 | u32 | `0xFFFFFFFF` (CPU type) |
| 20 | … | the bitcode (`BC\xC0\xDE` …) |

Each blob is then **zero-padded to a multiple of 16 bytes**. This matters:
`metal-objdump` rejects unpadded blobs with `unknown magic`, while the runtime does not
care. `MDSZ`, `HASH` and the bitcode-section size all describe the *padded* bytes; the
wrapper's own size field stays the *unpadded* length.

### 1.6 Dynamic header and reflection list

Dynamic header: a `NAME` tag holding the library name (`default.metallib\0`) plus
`ENDT`. Reflection list: a single `u32 0`. The runtime ignores the reflection list, but
`metal-objdump` refuses to open a library that has no count there.

---

## 2. Bitcode dialect

```llvm
target triple = "air64_v28-apple-macosx26.0.0"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
```

**One module per entry point.** Apple packages each function in its own module, and so
do we: the module contains the entry point, every function it reaches through calls, and
only the globals and declarations those name.

**Bitcode version.** Metal's runtime accepts bitcode written by a current LLVM with
**opaque pointers**; this was measured, and it contradicts the typed-pointer downgrade
other AIR projects perform. Two limits apply: Apple's LLVM-15-era *reader* rejects
attributes and instruction flags newer than it knows (`Failed to materializeAll`), so
LLVM ≥ 19 flags (`trunc nuw`, `zext nneg`, `or disjoint`, `icmp samesign`, gep
`nuw`/`nusw`) must not be encoded; and `xcrun metal` needs `-Xclang -opaque-pointers` to
parse the equivalent *text*.

### 2.1 Address spaces

| space | meaning |
| --- | --- |
| 0 | thread (locals) |
| 1 | device |
| 2 | constant |
| 3 | threadgroup |

Nothing else exists. NVPTX's 4 (param) and 5 (local) both crash the runtime compiler, so
the assembler renumbers 4 → 2 and 5 → 0.

**Program-scope data must live in address space 2.** A `constant` global left in space 0
fails pipeline creation with `Undefined symbols: ___anon_N`; a *mutable* global in
space 0 fails the same way and has nowhere to go (Metal's only writable program-scope
storage is device or threadgroup memory).

### 2.2 Entry-point signatures

Entry points are **plain functions** — no special calling convention (a `ptx_kernel`
convention on a kernel is dropped). Parameters appear in manifest order.

| stage | returns |
| --- | --- |
| vertex | the output struct **by value**, as a packed literal struct: `<{ <4 x float>, <3 x float>, <2 x float> }>` |
| fragment | a vector (`<4 x float>`) or, for MRT, a packed struct of the colour vectors and the depth float |
| kernel | `void` |

Zig returns such structs via `sret`; the rewrite converts that to the by-value form.

### 2.3 Intrinsics

Apple's *frontend* accepts every spelling below. Only pipeline creation tells them
apart, and a rejected one usually **aborts `MTLCompilerService`** rather than producing a
diagnostic.

| `llvm.*` | behaviour at pipeline creation | what we emit |
| --- | --- | --- |
| `sin cos tan log10 pow` | abort: *unable to legalize instruction* (all widths) | `air.fast_sin` … `air.fast_pow` |
| `exp10 ldexp` | `Undefined symbols` | `air.fast_exp10`, `air.fast_ldexp` |
| `powi.<T>.i32` | abort | `air.fast_pow.<T>` with the exponent `sitofp`-converted |
| `minimum maximum` | abort | `air.fast_fmin`, `air.fast_fmax` |
| `nearbyint roundeven` | abort | `llvm.rint` (which passes) |
| `vector.reduce.*` | abort (every op, every width) | an `extractelement` + scalar-op chain; `air.any.vNi1` / `air.all.vNi1` for bool vectors; an op with no expansion is refused |
| `bitcast <N x i1> ↔ iN` | abort (both directions) | `zext` to `<N x i32>` + shifts + `icmp` |
| `ctpop.i4` | abort | refused with a hint |
| `nvvm.*` | no AIR equivalent | refused with a hint |
| `fence` | abort | refused; use `air.atomic.fence` |
| `sqrt exp exp2 log log2 fabs floor ceil trunc round rint fma fmuladd minnum maxnum copysign ctlz cttz ctpop bswap bitreverse`, `umin/umax/smin/smax`, `*.sat`, `*.with.overflow`, `assume`, `trap`, `memcpy/memset/memmove` | accepted | unchanged |

Native `atomicrmw`, `cmpxchg`, `load atomic` and `store atomic` are accepted;
`llvm.memcpy/memmove/memset.pX.pY` carry the operand address spaces in their name, so
the suffix must be rewritten when constant relocation changes one.

`air.wg.barrier`, `air.simdgroup.barrier` and every `air.simd_*` collective are
**convergent** in Apple's own declarations. Nothing in the bitcode records that, so a
duplicated call (LLVM jump-threading a barrier into both arms of a branch) compiles
cleanly and then desynchronises the threadgroup at runtime — which is why the assembler
runs a uniformity analysis before emitting one. See
[pipeline.md](pipeline.md#4-the-assembler-toolsairassemblerzig).

### 2.4 Texture intrinsics

The sample/read family returns **`{ <4 x float>, i8 }`** — the colour plus a
sparse-residency status byte — and takes the sampler as **`ptr addrspace(2)`**:

```llvm
declare { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)
declare { <4 x float>, i8 } @air.sample_texture_2d_grad.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x float>, <2 x float>, <2 x float>, float, i1, <2 x i32>, i32)
declare { <4 x float>, i8 } @air.gather_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i32, i32)
declare { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1), ptr addrspace(2), <2 x i32>, <2 x i32>, i32, i32)
declare ptr addrspace(2) @air.get_read_sampler()
declare void @air.write_texture_2d.v4f32(ptr addrspace(1), <2 x i32>, <4 x float>, i32, i32)
declare i32 @air.get_width_texture_2d(ptr addrspace(1), i32)
declare i32 @air.get_height_texture_2d(ptr addrspace(1), i32)
```

The trailing `i32` is the access qualifier: `0` sample, `1` read, `2` write, `3`
read_write. It must agree with the access string in the texture's metadata node.

Zig can express neither the struct return (an `extern struct` may not hold a vector) nor
the constant-space sampler pointer, so `src/engine/gpu.zig` declares the bare-vector
form and both the rewrite and the assembler replace the declaration with the canonical
one, binding each call's result through `extractvalue 0`. Metal *tolerates* the
bare-vector declaration, but the canonical form is what Apple emits and what the pixel
checks run against.

### 2.5 The sampler word

A `constexpr sampler` is a `[2 x i64]` constant in address space 2, passed straight to
the intrinsic. The bit layout below was derived from 45 one-setting-at-a-time
configurations compiled with Apple's compiler; `gpu.samplerState` encodes it at
comptime and its tests pin the values.

**word 0**

| bits | field | values |
| --- | --- | --- |
| 2:0 | s address | `0` clamp_to_zero / clamp_to_border, `1` clamp_to_edge, `2` repeat, `3` mirrored_repeat |
| 5:3 | t address | as above |
| 8:6 | r address | as above |
| 10:9 | mag filter | `0` nearest, `1` linear, `2` bicubic |
| 12:11 | min filter | as above |
| 14:13 | mip filter | `0` none, `1` nearest, `2` linear |
| 15 | coord | `1` = pixel coordinates |
| 19:16 | compare func | `8` never, `1` less … `7` always |
| 23:20 | max anisotropy − 1 | 0…15 |
| 39:24 | lod clamp min | IEEE **half** bit pattern |
| 55:40 | lod clamp max | IEEE half; MSL's default is `0x7bff` (65504) |
| 57:56 | border colour | |
| 59:58 | reduction | |
| 63 | — | never set by Apple |

**word 1**: bits 15:0 hold the lod bias as an IEEE half.

The default sampler is `{0x007bff0000080049, 0}`; `mag_filter = .linear,
min_filter = .linear` gives `{0x007bff0000080a49, 0}`.

Every constexpr sampler global **must** also be listed in `!air.sampler_states`
(§3.5). A module that samples through one without that list makes Metal's backend
**crash** at pipeline creation.

---

## 3. Metadata

### 3.1 Module level

```llvm
!llvm.module.flags = !{...}          ; see below
!llvm.ident = !{!"zig air-splice"}
!air.version = !{!{i32 2, i32 8, i32 0}}
!air.language_version = !{!{!"Metal", i32 4, i32 0, i32 0}}
!air.compile_options = !{!{!"air.compile.denorms_disable"},
                         !{!"air.compile.fast_math_enable"},
                         !{!"air.compile.framebuffer_fetch_enable"}}
```

`llvm.module.flags` carries `wchar_size 4`, `frame-pointer 2`, and Metal's binding
limits, copied from Apple's output: `air.max_device_buffers 31`,
`air.max_constant_buffers 31`, `air.max_threadgroup_buffers 31`, `air.max_textures 128`,
`air.max_read_write_textures 8`, `air.max_samplers 16`.

### 3.2 Per-stage lists

```llvm
!air.vertex   = !{!22, !37}
!air.fragment = !{!45, !54, !67}
!air.kernel   = !{!75, ...}
```

Each referenced node describes one entry point:

- vertex / fragment: `!{ptr @fn, !outputs, !inputs}`
- kernel: `!{ptr @fn, !{}, !inputs}` — the empty tuple is the outputs slot — optionally
  followed by `!{!"air.max_work_group_size", i32 N}`

### 3.3 Outputs

| field | node |
| --- | --- |
| vertex `position` | `!{!"air.position", !"air.arg_type_name", !"float4", !"air.arg_name", !"position"}` |
| vertex `point_size` (must be `f32`) | `!{!"air.point_size", !"air.arg_type_name", !"float", !"air.arg_name", !"point_size"}` — no `generated()` name |
| any other vertex field | `!{!"air.vertex_output", !"generated(<mangled>)", !"air.arg_type_name", …, !"air.arg_name", …}` |
| fragment, vector return | `!{!"air.render_target", i32 0, i32 0, !"air.arg_type_name", !"float4"}` |
| fragment `colorN` | `!{!"air.render_target", i32 N, i32 0, !"air.arg_type_name", !"float4"\|"half4"\|"uint4", !"air.arg_name", !"colorN"}` |
| fragment `depth` | `!{!"air.depth", !"air.depth_qualifier", !"air.any", !"air.arg_type_name", !"float", !"air.arg_name", !"depth"}` |

`generated(...)` is Itanium-style: length-prefixed name, then the type — `Dv<N>_<elem>`
for a vector, the bare element code otherwise (`f` float, `Dh` half, `i` int, `j`
uint). `generated(6normalDv3_f)`, `generated(6flatIdi)`, `generated(1af)`.

### 3.4 Inputs

One node per IR parameter, in parameter order. The leading `i32` is the parameter index.

| argument | node |
| --- | --- |
| buffer | `!{i32 P, !"air.buffer", !"air.location_index", i32 <index>, i32 1, !"air.read"\|!"air.read_write", !"air.address_space", i32 1\|2\|3, [!"air.struct_type_info", !N,] !"air.arg_type_size", i32 <sizeOf>, !"air.arg_type_align_size", i32 <alignOf>, !"air.arg_type_name", !"<msl name>", !"air.arg_name", !"<name>"}` |
| texture | `!{i32 P, !"air.texture", !"air.location_index", i32 <index>, i32 1, !"air.sample"\|"air.read"\|"air.write"\|"air.read_write", !"air.arg_type_name", !"texture2d<float, <access>>", !"air.arg_name", !"<name>"}` |
| sampler | `!{i32 P, !"air.sampler", !"air.location_index", i32 <index>, i32 1, !"air.arg_type_name", !"sampler", !"air.arg_name", !"<name>"}` |
| builtin | `!{i32 P, !"air.<kind>", !"air.arg_type_name", !"uint"\|"uint2"\|"uint3"\|"ushort"\|"bool"\|"float2", !"air.arg_name", !"<name>"}` |
| `stage_in` `position` | `!{i32 P, !"air.position", !"air.center", !"air.no_perspective", !"air.arg_type_name", !"float4", !"air.arg_name", !"position"}` |
| `stage_in` integer field | `!{i32 P, !"air.fragment_input", !"generated(...)", !"air.flat", !"air.arg_type_name", …, !"air.arg_name", …}` |
| `stage_in` other field | `!{i32 P, !"air.fragment_input", !"generated(...)", !"air.center", !"air.perspective", !"air.arg_type_name", …, !"air.arg_name", …}` |

Only struct element types carry `air.struct_type_info`; scalars and vectors are named by
their MSL spelling (`float`, `uint2`, …). `air.struct_type_info` is a flat list of
`offset, size, 0, "type", "name"` per field, measured with `@offsetOf`/`@sizeOf`. An
array of float vectors is a matrix: `[4]@Vector(4, f32)` spells `float4x4` (64 bytes).

Integer varyings are always `air.flat` — Apple's compiler emits that even without the
MSL qualifier.

### 3.5 Sampler states

```llvm
!air.sampler_states = !{!22}
!22 = !{!"air.sampler_state", ptr addrspace(2) @gpu.constexprSampler.Holder.state}
```

Required, for every stage, in every module that samples through a constexpr sampler.
Omitting it crashes Metal's backend at pipeline creation rather than producing an error.

---

## 4. Checking your own output

| command | checks |
| --- | --- |
| `xcrun metal-objdump -d lib.metallib` | the container parses and the bitcode disassembles; catches padding and offset mistakes |
| `zig build check -- lib.metallib` | Metal loads it, builds every pipeline, and the GPU produces the right values |
| `xcrun metal -Xclang -opaque-pointers shader.air.ll -o apple.metallib` | Apple's assembler agrees with our text; then run `zig build check -- apple.metallib` to compare the two libraries |

When `MTLCompilerService` aborts, the report in
`~/Library/Logs/DiagnosticReports/MTLCompilerService-*.ips` names the instruction it
could not legalize — the fastest route from *“pipeline creation failed”* to the
offending intrinsic.
