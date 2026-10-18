# Crosscheck: differential testing of the shader pipeline

The crosscheck bench generates random shader logic, runs it on the Metal device and on
the CPU, and compares the results. The two sides are **one Zig function** compiled twice,
so they must agree. A mismatch is a codegen bug: in this project's rewriter or assembler,
in Apple's GPU compiler, or in our understanding of the IR.

This complements `zig build check`, which runs 42 hand-written checks against one
shader. Random programs reach IR shapes nobody wrote by hand.

## Running it

```sh
zig build crosscheck -Dseed=7              # one generated kernel, GPU vs CPU
tools/crosscheck/sweep.sh 1 200            # a range of seeds, each outcome sorted
```

`sweep.sh` runs under macOS's stock `/bin/bash` (3.2). Every seed lands in one bucket:

| Bucket | Meaning |
| --- | --- |
| `match` | GPU and CPU agree on every element |
| `MISMATCH` | they disagree: a codegen bug, the most valuable finding |
| `REFUSED` | this project's assembler refused a construct: a feature gap |
| `METAL` | Metal's compiler rejected or failed on the library |
| `SKIP` | no Metal device (a CI runner without a GPU) |
| `ERROR` | anything else, e.g. the generator emitted invalid Zig |

Anything but a match is saved under `zig-out/crosscheck/<seed>/` as the generated
`body.zig` plus the full log, so it can be reproduced with `zig build crosscheck
-Dseed=<seed>` and reduced. The script exits 1 if any seed was a MISMATCH, REFUSED, METAL
or ERROR.

## How it works

1. `tools/crosscheck/gen.zig` turns a seed into `crosscheck_body.zig`, whose
   `body(a: i32, b: i32) i32` is a random expression tree. The seed fully determines it.
2. `build.zig` compiles that one file twice:
   - for nvptx64 inside `src/engine/crosscheck_harness.zig`, whose kernel computes
     `out[gid] = body(in0[gid], in1[gid])`; a second `air-splice` (using the harness as
     its manifest) assembles it into `crosscheck.metallib`;
   - for the host, inside `tools/crosscheck/driver.zig`.
3. The driver fills two 4,096-element input buffers from the seed (about 40% edge
   values such as 0, ±1, the i32 limits and single-bit masks; the rest random), runs the
   kernel on the GPU, runs `body` on the CPU, and compares every element.

The kernel's shape is fixed (two inputs, one output, a count, the thread id), so only
`body` changes between seeds. That keeps the manifest and the GPU runner constant.

## Why only some operators (milestone 1)

The generator uses only operators that are total in Zig: `+%`, `-%`, `*%`, `&`, `|`, `^`
and `~`, plus wrapping negation (`0 -% x`). Every generated expression is defined for all
2^64 input pairs, so a GPU/CPU difference can never be undefined behaviour diverging
between targets. Ordinary `+` would overflow into undefined behaviour, which LLVM may
legally compile differently for nvptx and aarch64.

Shifts, division, comparisons and selects, narrower and wider integer types, vectors,
control flow and floats come next, each added only after its exact semantics, and how to
keep generated code free of undefined behaviour, are pinned down. Floats will need
explicit tolerance rules (contraction, denormals, transcendental precision).

## What has been verified

On an Apple M3 (macOS 26.3, Zig 0.18.0-dev.253), 2026-10-18:

- 66 seeds (1–41, 100–108, 200–215), 270,336 GPU/CPU comparisons, all match. Finding no
  bug with these operators is expected: they are the most heavily used integer
  operations. Milestone 1 establishes the pipeline the riskier constructs will run on.
- The buckets were checked by injecting one failure of each kind: a wrong CPU answer
  sorted as `MISMATCH`, invalid generated Zig as `ERROR`, and a threadgroup barrier under
  a per-thread `if` (which the assembler's divergence check refuses) as `REFUSED`. The
  `METAL` and `SKIP` buckets match the driver's exact output lines but have not yet been
  triggered end to end.
- The default build is untouched: `default.metallib` is byte-identical with and without
  the bench.

## Next

- Widen the generator, starting with shifts and guarded division, then types, control
  flow and floats. This is what will start producing REFUSED and MISMATCH results.
- A reducer that shrinks a failing seed to a minimal program before it is reported.
- Later, the same generated shaders on Asahi Linux's Mesa stack once its M3 GPU driver
  works, so the two compilers can be compared on the same hardware.
