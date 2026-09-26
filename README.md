# zig-metallib — a Metal graphics pipeline written entirely in Zig

Shaders are written in Zig. They are compiled to Apple AIR bitcode, packed into a
`.metallib`, and loaded by a Metal app — **without Apple's Metal Shading Language
toolchain anywhere in the build**. No `.metal` files, no `xcrun metal`, no checked-in
`.ll`, no C or Objective-C sources: every AppKit and Metal call goes through
`objc_msgSend` from Zig.

```
src/engine/my_shader.zig        Zig shader source + the binding manifest
        │
        │  zig build-obj -target nvptx64-cuda -mcpu=sm_75 -femit-llvm-ir   (Zig's LLVM)
        ▼
    LLVM IR text
        │
        │  tools/air_splice.zig          rewrite: AIR target, entry headers from the
        │                                manifest, sret → by-value, attribute cleanup
        │  tools/air/assembler.zig       IR text → bitcode via std.zig.llvm.Builder
        │  tools/air/metadata.zig        manifest + Zig types → !air.* metadata
        │  tools/air/metallib.zig        MTLB container packer
        ▼
   default.metallib  ──@embedFile──▶  src/main.zig  ──newLibraryWithData:──▶  Metal
```

The only Apple binaries involved are the ones that must be: the Metal *runtime*
(`MTLCompilerService`, which turns AIR into GPU code when a pipeline is created) and,
optionally, `xcrun metal` / `metal-objdump` used **outside** the build to cross-check
what we produce.

## Quick start

```sh
zig build                                              # shaders + library + app
zig build run                                          # opens a window, textured triangle
zig build test                                         # 90 tests
zig build metallib                                     # zig-out/bin/{default.metallib,shader.air.ll}
zig build check -- zig-out/bin/default.metallib         # load into Metal, run everything
xcrun metal-objdump -d zig-out/bin/default.metallib     # read our container back
```

Requires a nightly Zig (pinned to **0.17.0-dev.2307+392b17125** in `build.zig.zon`) and
macOS 26 or newer to run the result. `std.zig.llvm.Builder` is an internal
standard-library API and moves between nightlies, so `zig build` warns when a different
Zig runs it. ziglang.org deletes old nightlies, so an exact pin may need a community
mirror to fetch.

## What is verified, and how

`zig build check` is the gate that matters. Apple's frontend accepting IR proves
nothing — many constructs assemble happily and then crash `MTLCompilerService` at
pipeline creation — so every feature here is checked by *executing* it: a compute
dispatch whose results are read back and compared, or an offscreen render whose pixels
are read back and compared.

| gate | what it proves | result |
| --- | --- | --- |
| `zig build test` | rewriter, assembler, metadata, packer, uniformity analysis, target profiles | 90 tests pass |
| `zig build` | Zig → IR → container, end to end | builds |
| `zig build check -- …` | Metal loads it, compiles every pipeline, and the GPU computes the right answers | **42 ok, 0 fail** |
| `xcrun metal-objdump -d` | our container is readable by Apple's own tooling | passes |
| `xcrun metal` on `shader.air.ll` | Apple's assembler accepts our printed IR; the library *it* builds also passes `zig build check` | **42 ok, 0 fail** |

The last row is the strongest cross-check available without Apple's source: from the
same IR text, Apple's assembler and ours produce libraries that both run correctly.

The 42 checks cover 15 entry points — 2 vertex, 3 fragment, 10 kernels — including:

- pixel-exact textured renders (all 64 pixels of an 8×8 target), once through a
  `constexpr` sampler baked into the shader and once through a host-bound
  `[[sampler(0)]]`;
- an MRT render (two colour attachments of different formats + depth) with
  `discard_fragment`, instancing, `base_instance`, `front_facing` and `point_coord`;
- 10 compute dispatches: threadgroup memory, barriers, SIMD-group reductions,
  shuffles and prefix sums, device and threadgroup atomics, a two-stage tree
  reduction, and a texture read/write kernel.

## Scope and honest limits

- **One real GPU.** The GPU checks above were measured on macOS 26.3 (25D125), Apple
  M3 (10-core GPU), Metal 4, `air64_v28-apple-macosx26.0.0`. CI builds the project and
  passes the unit tests on GitHub's macOS 15.7.9 and 26.6.2 runners. Those runners have
  only a virtual GPU ("Apple Paravirtual device"), though. On the 26.6.2 runner it
  could not build the first compute pipeline (`scaleKernel`) from this project's IR,
  even when Apple assembled it, while it built one from Apple's MSL-compiled control
  kernel; the checker stops at the first failure, so nothing after it was tried. The
  cause is opaque pointers: in a later run the same Apple control kernel built a
  pipeline when compiled with typed pointers and failed when compiled with
  `-Xclang -opaque-pointers`, same kernel, same compiler (CI run 36266828343), while
  both pass on the M3. This project's assembler writes only opaque pointers. On
  15.7.9 only the unverified macOS 15 library could run, and it failed the same way,
  which settles nothing. That one known failure is reported as informational on the
  virtual GPU; any other failure there, and any failure on a real GPU, fails the run
  ([docs/pipeline.md](docs/pipeline.md#continuous-integration)). The container header
  carries values copied from one toolchain's output; other Metal versions may differ.
- **macOS 26 and newer only.** Libraries are stamped for macOS 26 (`-Dmetal-target`).
  Profiles for macOS 13–15 exist, reproducing Apple's container layout and stamps,
  but they are refused by default. macOS 26 rejects this project's opaque-pointer
  bitcode under an older AIR version (`Failed to upgrade function bitcode`). Apple uses
  typed pointers for those targets, and `std.zig.llvm.Builder` can only write opaque
  ones. See [docs/pipeline.md](docs/pipeline.md#deployment-targets).
- **The AIR format is undocumented.** The container layout, the `!air.*` metadata and
  the sampler bit layout here were reverse-engineered from Apple's output and
  confirmed by repacking, loading and running. A few header fields are reproduced
  without knowing their meaning (see `docs/air-format.md`).
- **The shader target is a detour.** Zig only permits GPU address spaces on GPU
  targets, so shaders are compiled for `nvptx64` and the AIR-specific work happens
  afterwards. Nothing NVPTX-specific survives the rewrite.
- **Unsupported IR fails loudly.** Anything the assembler does not handle is
  `error.Unsupported` with the offending line, never silently dropped.
- **Not a general Metal binding.** `src/objc.zig` is the minimum `msgSend` surface the
  example needs.

## Prior art

Writing `.metallib` files without Apple's compiler has been done before — floor/libfloor,
LLAIR, Metal.jl and metal-ir-pipeline all emit AIR — but they go through a **patched or
pinned LLVM** (typically a typed-pointer downgrade) and target compute, from C++ or
Julia. What is different here:

- shaders are written in **Zig**, a general-purpose language, with `struct`s and
  `comptime` describing the binding layout;
- the bitcode is produced by **`std.zig.llvm.Builder`** — a stock standard-library
  bitcode writer, no LLVM fork, no C++ dependency;
- the `!air.*` reflection metadata is **derived at comptime from the Zig types**
  themselves, so the manifest cannot drift from the shader signatures;
- it does **graphics** (vertex/fragment, MRT, depth, textures, samplers), not only
  compute;
- Metal's runtime compiler on Apple silicon (measured on an M3, macOS 26.3) accepts
  **opaque-pointer, LLVM-20-era bitcode for macOS 26 (AIR 2.8)**, even though Apple's
  own compiler still emits typed pointers. GitHub's virtual GPU rejects opaque
  pointers even at AIR 2.8, Apple's own included (see *One real GPU* above). For older
  targets the typed-pointer requirement other projects work around is real: macOS 26
  refuses opaque pointers under an AIR 2.7-or-older stamp, and Apple's assembler
  reproduces the refusal. Both findings were measured here, not assumed.

## Layout

```
build.zig                   the whole pipeline as a build graph
src/main.zig                NSApplication + CAMetalLayer + render loop, via msgSend
src/objc.zig                objc_msgSend, autorelease pools, dispatch_data_create
src/engine/my_shader.zig    the shaders and the `functions` manifest
src/engine/air.zig          manifest types (host-safe: no GPU pointers)
src/engine/gpu.zig          shader-side helpers: barriers, SIMD, atomics, textures, samplers
tools/air_splice.zig        IR text rewrite + the CLI that drives the whole conversion
tools/air/assembler.zig     LLVM IR text → bitcode (std.zig.llvm.Builder)
tools/air/divergence.zig    uniformity analysis behind the convergent-call check
tools/air/intrinsics.zig    which llvm.* Metal accepts, renames, or cannot take at all
tools/air/metadata.zig      manifest → !air.* metadata (text and bitcode)
tools/air/metallib.zig      MTLB container writer
tools/air/target.zig        deployment-target profiles: triple, versions, header stamps
tools/metallib_check.zig    loads a library into Metal and runs every entry point
docs/pipeline.md            how the pipeline works, and how to add a shader
docs/air-format.md          the container, bitcode dialect and metadata reference
.github/workflows/ci.yml    build, test and runtime checks on GitHub's macOS runners
```

## Documentation

- **[docs/pipeline.md](docs/pipeline.md)** — each stage, the manifest reference, the
  GPU helper surface, and what to do when something fails.
- **[docs/air-format.md](docs/air-format.md)** — the `.metallib` container byte
  layout, the AIR bitcode dialect, the metadata nodes and the sampler word.
