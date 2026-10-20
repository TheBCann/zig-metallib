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
zig build crosscheck -Dcrosscheck-body=path/to/body.zig   # a hand-written body
zig build crosscheck-apple -Dcrosscheck-body=path/to/body.zig   # same kernel, library built by Apple's assembler
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

`-Dcrosscheck-body=<file>` runs a hand-written or reduced body through the same pipeline
instead of a generated one. The file must define `pub const seed: u64` and
`pub fn body(a: i32, b: i32) i32`.

`zig build crosscheck-apple` answers whose bug a finding is. It runs the same kernel, but
the library is built from our printed IR by Apple's own assembler (`xcrun metal -Xclang
-opaque-pointers`, as CI does for `shader.air.ll`) instead of `tools/air/assembler.zig`. A
finding that persists is in Metal's GPU compiler; one that disappears is in our assembler
or container. It needs the Metal toolchain (`xcrun metal --version`).

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

## What the generator emits

All on i32: wrapping `+% -% *%`, saturating `+| -|`, bitwise `& | ^ ~`, wrapping
negation, `@min`/`@max` (signed, and unsigned through a bit-cast), shifts (`<<`,
arithmetic and logical `>>`), signed and unsigned division and remainder (`@divTrunc`,
`@divFloor`, `@rem`, `@mod`, `/`, `%`), `@popCount`, `@clz`, `@ctz`, `@byteSwap`,
`@bitReverse`, `@abs`, the overflow bit of `@mulWithOverflow`, and `if`/`else` selects on
one or two signed or unsigned comparisons.

Other integer widths, through helpers that return i32: truncation to `i8`/`u8`/`i16`/`u16`
and sign- or zero-extension back, `i8` wrapping add, `u16` wrapping multiply, the high 32
bits of the exact 64-bit signed and unsigned products, a 64-bit average, and the high half
of a 64-bit shift. The 64-bit operations cannot overflow: the product of two 32-bit values
always fits in 64 bits.

Control flow. A body is a small program rather than one expression: up to four local
variables, assignments, `if`/`else`, `switch` on a masked value, bounded `for` loops over
a range of at most 8 iterations, `break` and `continue` under a condition, and a final
expression. The generator builds a syntax tree first and prints it afterwards, so its
post-passes can check that every local is assigned and read, and so a reducer can shrink
a failing program structurally.

Every generated program is defined, and terminates, for all 2^64 input pairs, so a
GPU/CPU difference can never be undefined behaviour diverging between targets:

- arithmetic either wraps or saturates; ordinary `+` would overflow into undefined
  behaviour, which LLVM may legally compile differently for nvptx and aarch64;
- shift amounts are masked to 5 bits, and Zig's `<<` wraps (LLVM `shl` without
  `nsw`/`nuw`);
- every loop is a `for` over a range of at most 8 iterations, so it ends; `break` and
  `continue` conditions are runtime values, never comptime-known, or the statements after
  them would be unreachable code, which Zig rejects;
- division and remainder go through helpers that return the dividend for a zero divisor
  and for `minInt / -1`. `@rem` and `@mod` need the `minInt` guard too: Zig's Debug build
  defines `@rem(minInt, -1)` as 0, but its release lowering is a plain LLVM `srem`, which
  is undefined for that input.

The host driver is a Debug build, so its safety checks double as an undefined-behaviour
detector for the generator: a slip would panic on the CPU (bucket `ERROR`) instead of
silently diverging.

Operations with a known finding (below) are left out of the generator so new sweeps
surface new problems. `tools/crosscheck/gen.zig` says where to add them back.

## Findings

Found on an Apple M3, macOS 26.3, with Zig 0.18.0-dev.253. Each has a one-line reproducer
in `tools/crosscheck/known/`:

| Operation | Zig lowers it to | Result | Reproducer |
| --- | --- | --- | --- |
| `a *\| b` (saturating multiply) | `llvm.smul.fix.sat.i32` | building the compute pipeline fails: Metal's compiler service connection is interrupted | `zig build crosscheck -Dcrosscheck-body=tools/crosscheck/known/mul_sat.zig` |
| `@addWithOverflow(a, b)[1]` | overflow bit of `llvm.sadd.with.overflow.i32` | GPU and CPU disagree on 1,990 of 4,096 inputs | `zig build crosscheck -Dcrosscheck-body=tools/crosscheck/known/add_overflow_bit.zig` |
| `@subWithOverflow(a, b)[1]` | overflow bit of `llvm.ssub.with.overflow.i32` | GPU and CPU disagree on 2,103 of 4,096 inputs | `zig build crosscheck -Dcrosscheck-body=tools/crosscheck/known/sub_overflow_bit.zig` |

**Attribution: all three are faults in Metal's GPU compiler.** Each reproducer was also
run through `zig build crosscheck-apple`, where Apple's own assembler builds the library
from our printed IR, and failed identically: the same crash, and the same mismatch counts
(1,990 and 2,103). That rules out this project's bitcode encoder and container. Our text
rewrite is ruled out too: the intrinsic declarations and calls in the rewritten IR are the
ones Zig emitted, apart from the attribute group our cleanup drops from every `declare`.
This project's assembler passes all three intrinsics to Metal unchanged
(`tools/air/intrinsics.zig`). The overflow bit of `@mulWithOverflow` matches.

**In this project.** The first sweep with control flow produced the bench's first
`REFUSED` results: `air-splice: use of undefined value` on seeds 5 and 21. LLVM had laid a
loop's latch block out before its header, so the latch's `add` read a value whose defining
`phi` only appeared later in the text. That is legal IR, because the header still
dominates the latch, but the assembler resolved operands in text order. The splice now
emits blocks in a reverse postorder from the entry, which puts every definition before
its non-phi uses (`reorderBlocks` in `tools/air_splice.zig`, with a regression test of the
exact shape). Eight of the fifteen functions in the sample shader had blocks out of that
order, so `default.metallib` changed bytes with this fix; the printed IR is the same
multiset of unchanged blocks, and all 42 checks and Apple's assembler cross-check still
pass.

## What has been verified

On an Apple M3 (macOS 26.3, Zig 0.18.0-dev.253):

- **Milestone 1** (wrapping arithmetic and bitwise operators only): 66 seeds, 270,336
  GPU/CPU comparisons, all match. The buckets were checked by injecting one failure of
  each kind: a wrong CPU answer sorted as `MISMATCH`, invalid generated Zig as `ERROR`,
  and a threadgroup barrier under a per-thread `if` (which the assembler's divergence
  check refuses) as `REFUSED`. `SKIP` matches the driver's exact output line but has not
  been triggered end to end.
- **Expanded generator**, before any GPU run: 200 generated bodies compile and run
  without a safety-check panic in a Debug build over an edge-value grid plus 2,000 random
  input pairs each.
- **First GPU sweep** of the expanded generator, seeds 1–100: 82 match, 11 `METAL`, 7
  `MISMATCH`, no `REFUSED` or `ERROR`. Every `METAL` seed used `*|`. One-operator kernels
  then tested each operation on its own: all match except the three findings above,
  which is the first end-to-end trigger of the `METAL` bucket.
- **Sweep with the findings excluded**, seeds 1–100: all 100 match. The menus changed, so
  these are 100 fresh programs; no failure beyond the three findings showed up.
- **Integer widths added**: 200 generated bodies clean under Debug safety, a 100-seed
  sweep with all 100 matching, and each of the 10 new operations matching in a
  one-operator kernel. No new findings.
- **Attribution of the findings**: a passing seed (3) also passes through
  `crosscheck-apple`, so that path tests something; each of the three findings fails the
  same way through it. See "Findings".
- **Control flow added**: 200 generated programs valid, free of undefined behaviour and
  terminating under Debug safety; the first GPU sweep refused seeds 5 and 21 (the
  assembler bug described under "Findings"); after the fix those two match, four
  one-construct kernels (a plain bounded `for`, `break`, `continue`, nested loops with a
  `switch`) match, and a fresh 100-seed sweep is 100 match.
- The default build is untouched: `default.metallib` is byte-identical with and without
  the bench.

## Next

- Vectors and floats. Floats need explicit tolerance rules (contraction, denormals,
  transcendental precision).
- A reducer that shrinks a failing seed to a minimal program before it is reported.
- Later, the same generated shaders on Asahi Linux's Mesa stack once its M3 GPU driver
  works, so the two compilers can be compared on the same hardware.
