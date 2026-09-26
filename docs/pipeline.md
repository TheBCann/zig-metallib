# The pipeline

How a Zig function becomes a Metal shader here, stage by stage — and what to do when
one of the stages refuses.

```
my_shader.zig ──▶ LLVM IR ──▶ rewrite ──▶ assemble ──▶ pack ──▶ default.metallib
   (1)              (2)         (3)         (4)         (5)
```

Everything except step 1 lives in `tools/`, is written in Zig, and runs as part of
`zig build`. Step 1 is Zig's own LLVM backend.

---

## 1. Zig source → LLVM IR

`src/engine/my_shader.zig` is compiled **twice** from the same file:

- for **`nvptx64-cuda`** (`sm_75`), with `-femit-llvm-ir`, to get the IR of the shader
  bodies. `build.zig` does this with `addObject(... .use_llvm = true)` and takes
  `getEmittedLlvmIr()`; nothing is linked and no object is kept.
- for the **host**, as a plain module, where only the `functions` manifest is read.
  Zig does not analyse a function that nothing references, so the GPU
  address-space pointers in the shader bodies never reach the host compile.

### Why NVPTX

Zig permits address-space qualified pointers only on GPU targets, and AIR is not one of
Zig's targets. NVPTX is the closest target that accepts the address spaces we need:

| Zig spelling | NVPTX space | AIR space | Use |
| --- | --- | --- | --- |
| `[*]addrspace(.global) T` | 1 | 1 device | `device T*` buffers, textures |
| `*addrspace(.param) const T` | 4 | 2 constant | `constant T*` uniform buffers |
| `[*]addrspace(.shared) T` | 3 | 3 threadgroup | `threadgroup T*` buffers |
| `var x: [N]T addrspace(.shared)` | 3 | 3 threadgroup | static threadgroup memory |
| (unqualified) | 0 | 0 thread | locals |

Zig rejects a spelled `addrspace(.constant)` on NVPTX, hence `.param` plus the
assembler's 4 → 2 renumbering. Address spaces 4 and 5 both crash Metal's runtime
compiler, so neither survives into the bitcode. Nothing else NVPTX-specific is kept:
`ptx_kernel` calling conventions are dropped, and `llvm.nvvm.*` intrinsics are refused
outright (thread ids are explicit manifest parameters, not intrinsics).

### Keeping entry points alive

Zig's LLVM backend emits nothing for a `pub fn` no one calls. Kernels declare
`callconv(.kernel)` and are `@export`ed under their own name. Vertex and fragment
functions cannot be exported directly (their signatures are not C-ABI), so the file has
a `comptime` block, active only on `nvptx64`, that exports a tiny
`callconv(.nvptx_device)` thunk returning `@ptrCast(&theShader)` under the name
`__keep_<shader>`. The thunk forces the shader into the IR; the rewrite drops every
`__keep_*` define afterwards.

### Safety checks

`pub const panic = std.debug.no_panic;` makes every Zig safety check compile to
`llvm.trap` + `unreachable`, which Metal accepts. `zig build -Dshader-optimize=safe` is
a useful verification build: the uniformity analysis treats trapping blocks as
no-return, so a barrier after a bounds check is still accepted.

---

## 2–3. The rewrite (`tools/air_splice.zig`)

The IR Zig emits is LLVM-20-era and NVPTX-shaped; Apple's reader is LLVM-15-era and
AIR-shaped. The rewrite is pure text and does:

- **target**: `datalayout` and `triple` → `air64_v28-apple-macosx26.0.0`.
- **entry headers**: rebuilt from the manifest — `define <ret> @name(<params>)` with
  builtins typed from `Builtin.T`, buffers as `ptr addrspace(1|2|3)`, `stage_in` fields
  by their Zig type. If the Zig parameter types disagree with the manifest the build
  **fails**: a manifest `u16` builtin over a Zig `u32` parameter would otherwise
  assemble and run with thread ids wrapping at 65536.
- **returns**: Zig returns a vertex (or MRT fragment) output struct via `sret`; AIR
  returns it by value as a packed struct. The rewrite allocates the struct, rewires the
  pointer, and turns each `ret void` into loads plus an `insertvalue` chain.
- **names**: `@module.name` → `@name`; quoted names (`@"mod.Vec(4).sum"`, Zig's
  spelling inside generic instantiations) are followed and re-emitted verbatim.
- **pruning**: every `define` no entry point reaches (`__keep_*`, panic helpers) is
  dropped; reached helpers keep `private`/`fastcc`. `llvm.lifetime.*` is removed — its
  one-argument form segfaults Apple's frontend.
- **cleanup**: attribute groups, Zig metadata, and flags newer than Apple's LLVM
  (`captures(none)`, `initializes(...)`, `dead_on_unwind`, gep `nuw`/`nusw`, `nneg`,
  `disjoint`, `samesign`).
- **address spaces**: 4 → 2 and `ptr addrspace(5)` → `ptr` in the text, so the debug
  `.ll` matches what the assembler lowers.
- **texture intrinsics**: the `declare` is replaced by Apple's canonical signature and
  each call becomes a call returning `{ <colour>, i8 }` plus `extractvalue 0`; sampler
  operands are retyped to `ptr addrspace(2)` and sampler words are printed in address
  space 2. See `docs/air-format.md`.

The optional third CLI argument writes this text plus the metadata to a `.ll` file
(`zig build metallib` puts it in `zig-out/bin/shader.air.ll`). It is a debugging aid
*and* a cross-check: `xcrun metal -Xclang -opaque-pointers shader.air.ll` must accept
it, and the library Apple builds from it must pass `zig build check` too.

---

## 4. The assembler (`tools/air/assembler.zig`)

Parses the rewritten text and emits bitcode with `std.zig.llvm.Builder` — one module
per entry point, because Apple packages one module per function. A module contains that
entry point, every `define` it reaches through calls, and only the globals and
declarations those functions name.

**Supported**: integer/float/vector/array/struct and named struct types; globals;
declarations; `ret br switch phi select icmp fcmp` arithmetic `fneg freeze` casts
`getelementptr load store` (plain and atomic) `atomicrmw cmpxchg alloca insertvalue
extractvalue insertelement extractelement shufflevector call unreachable`; literal,
aggregate, `splat`, `c"…"` and `getelementptr` constant expressions. Anything else is
`error.Unsupported` with the line.

Three rules are worth knowing because they shape how you write shaders:

**Constants move to address space 2.** Zig emits comptime tables, sampler words and
strings as address-space-0 constants. A constant left there fails pipeline creation with
`Undefined symbols: ___anon_N`, so the assembler relocates them. To keep that
consistent without text surgery, **named operands always take the type they were
defined with**; a textual type annotation only shapes literal constants, and a GEP
result inherits its base pointer's address space. An aggregate that *stores the address*
of a relocated constant (a `[]const T` slice of a comptime table, `[4 x ptr]`, …) is
refused: a `load ptr` from it would produce a thread-space pointer into constant memory.
Index comptime tables directly instead of taking their address.

A module-level **mutable** global has no Metal address space at all and is refused with
a hint: use a local, a device buffer, or threadgroup memory.

**Intrinsics go through a table** (`tools/air/intrinsics.zig`). `llvm.sin`, `cos`,
`tan`, `log10`, `exp10`, `pow`, `powi`, `minimum`, `maximum`, `nearbyint`, `roundeven`
and every `llvm.vector.reduce.*` are accepted by Apple's *frontend* and then abort
`MTLCompilerService`. They are renamed to their `air.fast_*` form, mapped to
`llvm.rint`, or expanded into scalar chains. `bitcast` between `<N x i1>` and `iN`
(what `@bitCast` of a bool vector and `@reduce(.Or)` fold into) crashes the backend both
ways and is expanded with shifts. `sqrt exp exp2 log log2 fabs floor ceil trunc round
rint fma fmuladd minnum maxnum copysign ctlz cttz ctpop bswap bitreverse` and the
saturating/overflow families pass unchanged.

**Convergent calls are checked** (`tools/air/divergence.zig`). Apple marks the barriers
and every SIMD-group collective `convergent`; Zig cannot put that attribute on an
`extern fn`, so LLVM happily duplicates such a call into both arms of a branch. Metal
compiles the result and the threadgroup silently desynchronises. A uniformity analysis
over three levels (`threadgroup` < `simdgroup` < `thread`), seeded from
`BuiltinKind.uniformity`, decides which threads reach a block together:

- accepted: barriers in straight-line code, in the join block after a per-thread
  branch, in a loop whose trip count is uniform (`threads_per_threadgroup / 2`, a
  constant-buffer field), after a uniform early return, after a trapping safety check;
  SIMD-group calls under `if (sgid == 0)`.
- refused: a barrier behind `if (gid >= n) return;`, a barrier in a loop some threads
  `break` out of on a per-thread condition, a threadgroup barrier under
  `if (sgid == 0)`, a `simd_sum` under `if (lane == 0)`.

The error message names the offending line and says how to restructure the code.

---

## 5. Metadata and the container

`tools/air/metadata.zig` turns the manifest and the Zig types into the `!air.*`
metadata Apple's driver reads, as a tree of `Node` values walked by two backends: text
(for the `.ll`) and `std.zig.llvm.Builder` (for the bitcode). Types are measured with
`@offsetOf`/`@sizeOf`/`@alignOf`, so reflection cannot drift from the struct.

`tools/air/metallib.zig` writes the MTLB container. Both are documented byte by byte in
**[air-format.md](air-format.md)**.

---

## The manifest

`src/engine/air.zig`. Host-safe by design: no GPU pointer types, so the file compiles
anywhere.

```zig
pub const functions = [_]air.Function{
    .{
        .name = "vertexShader",          // must match the Zig function's name
        .stage = .vertex,                // .vertex | .fragment | .kernel
        .ret = VertexOut,                // see the table below
        .args = &.{
            .{ .vertex_id = "vertexID" },
            .{ .buffer = .{ .index = 0, .T = VertexIn, .name = "vertices" } },
        },
    },
    // ...
};
```

`Function.ret` by stage:

| stage | `ret` | becomes |
| --- | --- | --- |
| `.vertex` | output struct | `position` → `[[position]]`; a `f32` field named `point_size` → `[[point_size]]`; integer fields → `[[flat]]` varyings; anything else → perspective-interpolated varying |
| `.fragment` | `@Vector(4, f32)` | `[[color(0)]]` |
| `.fragment` | output struct | fields `color0…colorN` → `[[color(N)]]` (`f32`/`f16`/`u32`/`i32` vectors), optional `depth: f32` → `[[depth(any)]]` |
| `.kernel` | `void` | — |

`Arg` variants:

| variant | fields | shader parameter |
| --- | --- | --- |
| `.buffer` | `index`, `T`, `name`, `space = .device`, `access = .read` | `[*]addrspace(.global) [const] T`, `*addrspace(.param) const T`, `[*]addrspace(.shared) T` |
| `.texture` | `index`, `name`, `T = gpu.Texture2D(access)` | `*addrspace(.global) [const] gpu.Texture2D(access)` |
| `.sampler` | `index`, `name` | `*const gpu.Sampler` |
| `.builtin` | `kind`, `T`, `name` | the plain Zig type (`u32`, `@Vector(2,u32)`, `bool`, `@Vector(2,f32)`) |
| `.vertex_id` | name only | shorthand for `.builtin{ .kind = .vertex_id, .T = u32 }` |
| `.stage_in` | the vertex output type | one fragment parameter per field **except `point_size`** |

`BuiltinKind` covers the vertex stage (`vertex_id`, `instance_id`, `base_vertex`,
`base_instance`), the fragment stage (`front_facing`, `point_coord`, `sample_id`,
`primitive_id`) and the kernel stage (`thread_position_in_grid`, `threads_per_grid`,
`thread_position_in_threadgroup`, `threadgroup_position_in_grid`,
`threads_per_threadgroup`, `threadgroups_per_grid`, `thread_index_in_threadgroup`,
`thread_index_in_simdgroup`, `simdgroup_index_in_threadgroup`, `threads_per_simdgroup`,
`simdgroups_per_threadgroup`, `dispatch_threads_per_threadgroup`). Each kind also
declares its `uniformity`, which is what the convergent-call analysis reasons with.

Kernels may set `max_total_threads_per_threadgroup` (emitted as
`air.max_work_group_size`).

---

## Shader-side helpers (`src/engine/gpu.zig`)

Thin `extern fn` wrappers whose names are the AIR intrinsic names, so the IR Zig emits
already has Apple's call shape.

| group | helpers |
| --- | --- |
| barriers | `threadgroupBarrier`, `simdgroupBarrier`, `atomicFence` |
| SIMD-group | `simdSum`, `simdMax`, `simdMin`, `simdProduct`, `simdPrefixInclusiveSum`, `simdBroadcast`, `simdShuffle`, `simdShuffleDown`, `simdShuffleUp`, `simdShuffleXor`, `simdSumU32`, `simdAny`, `simdAll` |
| atomics | `atomicAdd/Sub/Max/Min/Or/And/Xor/Exchange` (device), `atomicAddThreadgroup` |
| textures | `Texture2D(access)`, `sample2D`, `sample2DLevel`, `sample2DBias`, `read2D`, `write2D`, `width2D`, `height2D` |
| samplers | `SamplerDesc`, `samplerState` (the comptime word encoder), `constexprSampler` |
| fragment | `discardFragment` |

Constraints on this file: no `comptime {}` blocks, no top-level `var`, and tests only
for the host-safe sampler encoder — anything else would drag GPU pointer types into the
host build.

---

## Adding a shader

1. **Write it** in `src/engine/my_shader.zig` with GPU pointer types for its buffers
   and textures. Kernels get `callconv(.kernel) void`.
2. **Keep it alive**: add `@export(&yourKernel, .{ .name = "yourKernel" })` for a
   kernel, or a `keepYourShader` thunk plus
   `@export(&keepYourShader, .{ .name = "__keep_yourShader" })` for vertex/fragment,
   inside the `nvptx64` `comptime` block.
3. **Describe it** in `functions`. `zig build` refuses the build if the IR parameters
   and the manifest disagree, so this cannot silently drift.
4. **Prove it runs**: add a dispatch or render check in `tools/metallib_check.zig` that
   feeds known inputs and compares what comes back. A pipeline that merely *builds*
   proves nothing about what it computes.
5. Run the gates: `zig build test`, `zig build`,
   `zig build check -- zig-out/bin/default.metallib`.

---

## Build graph

| step | does |
| --- | --- |
| `zig build` | shader IR → `air-splice` → `default.metallib` → embedded in the app |
| `zig build run` | builds and launches the window |
| `zig build test` | the `air_splice` module's tests (assembler, metadata, divergence, packer, sampler encoder) |
| `zig build metallib` | installs `zig-out/bin/default.metallib` and `shader.air.ll` |
| `zig build check -- <lib>` | loads any `.metallib` into Metal, builds every pipeline, runs the dispatch and render self-tests |
| `zig build check -- --kernel=<name> <lib>` | builds one compute pipeline from *any* library, no manifest: the control CI uses on an Apple-compiled kernel |
| `-Dshader-optimize=<mode>` | optimize mode of the GPU module (`fast` default; `safe` keeps Zig's safety checks) |
| `-Dmetal-target=<profile>` | the macOS the library targets: `macos26` (default), `macos15`, `macos14`, `macos13` ([air-format.md §1.7](air-format.md#17-deployment-targets)) |
| `-Dallow-unverified-target=true` | build a profile other than `macos26` anyway (they are refused by default; see below) |

`air-splice` can also be run by hand:
`air-splice [--target=<profile>] [--allow-unverified] <in.ll> <out.metallib> [out.ll]`.

`zig build` prints a warning when the running Zig differs from the version pinned in
`build.zig.zon`, because `std.zig.llvm.Builder` changes between nightlies.

### Deployment targets

The library is stamped for macOS 26 by default. `tools/air/target.zig` also carries the
macOS 15, 14 and 13 profiles, whose stamps match Apple's output byte for byte. They are
refused unless you pass `-Dallow-unverified-target=true`. macOS 26 rejects this
project's opaque-pointer bitcode under an older AIR version, and `std.zig.llvm.Builder`
cannot write the typed pointers Apple uses there. The opt-in exists for one experiment:
build `-Dmetal-target=macos15 -Dallow-unverified-target=true` **on macOS 15** and run
`zig build check`. If macOS 15 reads opaque pointers natively (it skips the upgrader
that fails on 26), that profile becomes verifiable.

### Continuous integration

`.github/workflows/ci.yml` runs on GitHub's `macos-15` and `macos-latest` runners. It
reads the Zig version from `build.zig.zon` and runs `zig build test`, which also
compiles `metallib-check`. It then runs `zig build install metallib`, which compiles the
app and installs the library. A control step follows: Apple's own compiler builds a
trivial kernel from MSL written inside the workflow (the repo keeps no `.metal` files),
`metallib-check --kernel=control` builds its pipeline, and on macOS 26 runners Apple
also re-assembles this project's printed IR and runs the full check on it. The job
summary tabulates the GPU's name and each result. If the pipeline from Apple's own
kernel fails too, the runner's GPU is the cause; if it passes and ours fails, the cause
is this project's output, and the Apple-assembled IR tells our assembler and packer
apart from the IR itself. A missing Metal toolchain, a failed Apple compile and a runner
without a Metal device each get their own "not run" row, so none of them reads as a GPU
failure. The runtime checks run where the runner is macOS 26 or newer, and on macOS 15
the experiment above runs without failing the build. GitHub's arm64 runners are virtual
machines and may expose an "Apple Paravirtual device", so the checks can run on a
virtual GPU; confirm a failure there on a real Mac before blaming the library. With no
Metal device at all, `metallib-check` prints `SKIP no Metal device` and exits 77. The
step still passes (green), but it adds a `::warning::` annotation and a job-summary line
saying nothing was checked. ziglang.org deletes old nightlies, so the setup step fails
when no mirror still serves the pinned version.

---

## When something fails

| symptom | where to look |
| --- | --- |
| `air-splice: line N: …` | the assembler or rewriter refused an IR construct; the message names the line and usually the fix |
| `air-splice: <fn>: IR has N parameters, manifest lists M` | the manifest and the Zig signature disagree |
| `air-splice: <fn>: parameter K ('x') is iN in the IR but the manifest says iM` | `Builtin.T` does not match the Zig parameter type |
| `convergent call in divergent control flow` | a barrier or SIMD call sits where only some threads arrive; restructure as the message suggests |
| `FAIL newLibraryWithData: …` | the container is malformed — packer territory |
| `FAIL newRenderPipelineState / newComputePipelineState: …` | the bitcode or metadata is wrong; Metal's message is usually specific (`Undefined symbols: …`) |
| `Compilation failed due to an interrupted connection: XPC_ERROR_CONNECTION_INTERRUPTED` | `MTLCompilerService` **crashed**. Something in the IR is unsupported in a way Apple does not diagnose; check `~/Library/Logs/DiagnosticReports/MTLCompilerService*.ips` for the failing instruction, and compare against the intrinsic table |
| `FAIL dispatch …: N mismatches` | it compiled and ran, and computed the wrong thing — the most interesting failure; suspect the convergent-call rules or an intrinsic mapping |
| `metal-objdump: unknown magic` | a bitcode blob is not padded to 16 bytes |
| `air-splice: refusing --target=…` | that deployment target is not verified; the message says why (see *Deployment targets*) |
| `note library targets macOS N.n but this is macOS M.m` (then `Unsupported target triple`) | the library targets a newer macOS *major* than the one running, which Metal refuses. A newer minor of the same major loads fine. Rebuild for this macOS with `-Dmetal-target` |
| `Failed to upgrade function bitcode` | an older-AIR library with opaque pointers on macOS 26 — the reason the older profiles are unverified |
| `SKIP no Metal device available` | no usable GPU (typically a CI runner); nothing was checked, exit code 77 |

A useful bisection tool: `zig build metallib`, then
`xcrun metal -Xclang -opaque-pointers zig-out/bin/shader.air.ll -o /tmp/apple.metallib`
and `zig build check -- /tmp/apple.metallib`. If Apple's build of the same text passes
and ours does not, the bug is in the assembler or the packer; if both fail, the IR
itself is wrong.
