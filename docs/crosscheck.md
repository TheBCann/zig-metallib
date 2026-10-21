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
zig build crosscheck -Dseed=7 -Dcrosscheck-floats=true          # float mode (docs/crosscheck-float-policy.md)
CROSSCHECK_ARGS="-Dcrosscheck-floats=true" tools/crosscheck/sweep.sh 1 200
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

A failing run prints the exact command that reproduces it: the seed is always part of it,
because the inputs come from `-Dseed` even when the body is hand-written, and a failure
seen through the Apple step names that step.

`zig build crosscheck-apple` answers whose bug a finding is. It runs the same kernel, but
the library is built from our printed IR by Apple's own assembler (`xcrun metal -Xclang
-opaque-pointers`, as CI does for `shader.air.ll`) instead of `tools/air/assembler.zig`. A
finding that persists is in Metal's GPU compiler; one that disappears is in our assembler
or container. It needs the Metal toolchain (`xcrun metal --version`), and it re-assembles
on every run rather than from the build cache, so the verdict always reflects the
installed toolchain.

## Reducing a failure

```sh
zig build reduce -Dseed=8                        # shrink seed 8's program
zig build reduce -Dseed=8 -Dcrosscheck-known=true   # the same with the known findings enabled
```

`crosscheck-reduce` (`tools/crosscheck/reduce.zig`) shrinks a failing seed's program to a
minimal one that fails the same way. The generator builds a syntax tree before printing
(`tools/crosscheck/program.zig`, shared by both tools), so the reduction is structural:
delete statements and whole blocks, hoist the body of an `if`, `for` or `switch` into its
parent, shorten loops, drop `else` branches and second comparisons, replace expressions
by their children or by a leaf, and delete locals nothing references. Each candidate is
judged by the real pipeline, `zig build crosscheck -Dcrosscheck-body=<candidate>`,
classified as `sweep.sh` does, and kept only when the result is the same bucket with the
same message (numbers blanked, so a line number does not make two instances of one failure
look different; for a mismatch, any mismatch counts). Candidates stay valid Zig by
construction: the printer declares a local `const` when it is never assigned and
discards it when it is never read, loop control hoisted out of its loop is dropped, and a
candidate that is invalid anyway lands in another bucket and is not kept. Every pass
strictly shrinks the program, so the reduction terminates. Measured on seed 8 with the
known findings enabled: 23 statements and 181 expression nodes down to 4 and 17, in 127
pipeline runs and about seven minutes on the M3, leaving a program whose mismatch is
visibly the `@subWithOverflow` bit.

The result goes to `zig-out/crosscheck/reduce/<seed>/`: `original.zig`, `reduced.zig`
(its header records the failure and the sizes before and after) and `candidate.zig`, the
last candidate tried. The candidates are built in a cache directory of their own under
that folder, deleted when the reduction ends: every candidate is a new body, and hundreds
of kernel, library and driver builds would otherwise fill the project's cache and the disk.
The reduced file runs like any other body, under the seed and mode it was judged with, which
its header repeats: `zig build crosscheck -Dseed=<seed> [-Dcrosscheck-floats=true]
-Dcrosscheck-body=zig-out/crosscheck/reduce/<seed>/reduced.zig`.

`-Dcrosscheck-known=true` makes the generator emit the operations recorded under
"Findings" again. It exists to re-test them on a newer macOS, and to give the reducer
something to shrink: with it, seed 8 mismatches.

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
- signed `@min`/`@max` go through i32 helpers. A bare `@max(x, @as(i32, 0))` has the
  narrowed type `u31` in Zig, which `@byteSwap` rejects (about 0.2% of seeds failed to
  compile before this) and which would run every operator above it at that width;
- division and remainder go through helpers that return the dividend for a zero divisor
  and for `minInt / -1`. `@rem` and `@mod` need the `minInt` guard too: Zig's Debug build
  defines `@rem(minInt, -1)` as 0, but its release lowering is a plain LLVM `srem`, which
  is undefined for that input.

The host driver, and the body's host compilation, are pinned to Debug in `build.zig`
whatever `-Doptimize` says, so their safety checks double as an undefined-behaviour
detector for the generator: a slip would panic on the CPU (bucket `ERROR`) instead of
silently diverging. The driver's `info` line shows `cpu-build debug` to prove it.

Operations with a known finding (below) are left out of the generator so new sweeps
surface new problems. `tools/crosscheck/gen.zig` says where to add them back.

**Float mode** (`-Dcrosscheck-floats=true`) adds f32 subtrees to the programs, following
[docs/crosscheck-float-policy.md](crosscheck-float-policy.md): inputs and locals carry
float values as bit patterns; a float subtree starts from a flushed bit-cast or an integer
conversion and ends in canonical bits or a guarded conversion; every operation goes
through a flushing helper, which is also the contraction barrier. The driver's edge inputs
gain float bit patterns, and the reducer handles float programs. Float mode is opt-in so
that every integer seed keeps its program.

## Findings

Found on an Apple M3, macOS 26.3, with Zig 0.18.0-dev.253. Each has a one-line reproducer
in `tools/crosscheck/known/`:

| Operation | Zig lowers it to | Result | Reproducer |
| --- | --- | --- | --- |
| `a *\| b` (saturating multiply) | `llvm.smul.fix.sat.i32` | building the compute pipeline fails: Metal's compiler service connection is interrupted | `zig build crosscheck -Dcrosscheck-body=tools/crosscheck/known/mul_sat.zig` |
| `@addWithOverflow(a, b)[1]` | overflow bit of `llvm.sadd.with.overflow.i32` | GPU and CPU disagree on 1,990 of 4,096 inputs | `zig build crosscheck -Dcrosscheck-body=tools/crosscheck/known/add_overflow_bit.zig` |
| `@subWithOverflow(a, b)[1]` | overflow bit of `llvm.ssub.with.overflow.i32` | GPU and CPU disagree on 2,103 of 4,096 inputs | `zig build crosscheck -Dcrosscheck-body=tools/crosscheck/known/sub_overflow_bit.zig` |
| `@round(x)` on f32 | `llvm.round.f32` | returns `+0.0` where the result is `-0.0` (949 of 4,096 inputs; `floor`, `ceil`, `trunc` are exact) | `zig build crosscheck -Dcrosscheck-body=tools/crosscheck/known/round_negative_zero.zig` |
| `@rem(x, y)` on f32 | `frem` | not the exact remainder: 0 for 1.02e-10, 8 for 6.91, `-inf` for finite operands (1,067 of 4,096) | `zig build crosscheck -Dcrosscheck-body=tools/crosscheck/known/rem_inexact.zig` |
| `@mulAdd(f32, C1, C2, x)` with a denormal constant product | `llvm.fma.f32` with two constant multiplicands | the compiler folds the product and flushes it, so the fma degenerates to `x`; the hardware fma keeps the exact intermediate (223 of 4,096; 266 with the float-mode inputs) | `zig build crosscheck -Dcrosscheck-body=tools/crosscheck/known/fma_constant_product_flush.zig` |

**Attribution: all six are faults in Metal's GPU compiler.** Each reproducer was also
run through `zig build crosscheck-apple`, where Apple's own assembler builds the library
from our printed IR, and failed identically: the same crash, and the same mismatch counts
(1,990, 2,103, 949, 1,067 and 223). That rules out this project's bitcode encoder and container. Our text
rewrite is ruled out too: the intrinsic declarations and calls in the rewritten IR are the
ones Zig emitted, apart from the attribute group our cleanup drops from every `declare`.
This project's assembler passes all six intrinsics to Metal unchanged
(`tools/air/intrinsics.zig`). The overflow bit of `@mulWithOverflow` matches. The two float
findings came out of the probes behind [docs/crosscheck-float-policy.md](crosscheck-float-policy.md),
which also records the GPU behaviours that are normalised rather than reported.

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

**Review, 2026-10-20.** An independent review of the generator, the block-order pass and
the verdict path, with every finding checked by three skeptics (reproduce, trace, impact),
confirmed six issues. Five are fixed: the `@min`/`@max` narrowing above; the splice's
identifier scanner lacked `-`, so a label such as `.unr-lcssa` (LLVM's name after loop
unrolling) would not have been recognised as a block and its branch target would have been
truncated, defeating the block-order pass for that shape; `-Doptimize` could switch the CPU
side out of Debug; the reproduce hints omitted the seed and named the wrong step for the
Apple run; and the Apple step's result was cached across toolchain updates. The sixth was
a limitation in the assembler, since fixed: it typed a `phi` of pointers from the incoming
values it had already seen in text order (`phiPointerType`), so a pointer `phi` whose only
non-null incoming is defined in a later block kept address space 0 instead of the relocated
constant space 2 and was refused. The assembler now lowers a function again when the phi's
real address space shows up later: in its incomings once the body is lowered, in a `select`
or `icmp` against a relocated pointer, through values derived from the phi by
`getelementptr` or `select` (the pointer-bump idiom), or through another phi that depends
on it (a conditional update kept as a branch, or an lcssa phi of an inner loop). Each
retry settles one more phi, and the final attempt still checks every type, so a wrong
guess is refused rather than emitted (`lowerDefine` and `retryIfUnsettledPhi` in
`tools/air/assembler.zig`). A `select` between a pointer and `null` takes the pointer's
address space in either order. This fix was itself reviewed the same way: five findings
confirmed and fixed (the retry bound, phis vetoing each other, derived values), two refuted,
one of them by showing Metal accepts the unused declaration an abandoned attempt can leave
behind. Residual gap: a settled phi whose mismatching incoming is a non-phi value derived
from two unsettled phis is still refused; the general remedy is an order-independent
pointer-type pre-pass, which the retry approximates one phi per attempt. Two real shader
bodies with these shapes live in `tools/crosscheck/regress/` and must keep matching:

```sh
zig build crosscheck -Dcrosscheck-body=tools/crosscheck/regress/pointer_phi_join.zig
zig build crosscheck -Dcrosscheck-body=tools/crosscheck/regress/pointer_phi_select.zig
```

The review harness is saved as `.claude/workflows/crosscheck-review.js`; from Claude Code,
`/crosscheck-review` runs it again after changes to the bench or the splice.

**LLVM 22 spellings (fixed 2026-10-20).** LLVM 22 prints a float constant that does not
round-trip through decimal as `f0x` plus the bit pattern of its own type (`float
f0xCF000000`), prints NaN and infinity constants as special values (`+qnan`,
`-nan(0x3FFFFE)`, `snan`, `inf`), and emits `llvm.is.fpclass` with an inline `/* ... */`
comment for `fabs(x) < FLT_MIN`. Apple's LLVM-15-era reader knows none of these, and
neither did this assembler, so shaders with such constants were refused; the sample shader
only escaped because its constants print in decimal. The assembler now parses all of them
and lowers `is.fpclass` to integer bit tests, and the splice rewrites the constants to the
legacy spellings (`0xC1E0000000000000`, `0xH3C00`, `0xFFFFFFFFC0000000`) and drops the
comments in the text it hands to Apple's assembler, which then accepts it (verified
through `crosscheck-apple`, including a float-mode program).

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
- **After the review fixes**: 206 generated programs (seeds 1–200 plus the six that used
  to fail to compile) are valid, free of undefined behaviour and terminating under Debug
  safety; the six run and match on the GPU; a fresh 100-seed sweep is 100 match; the
  default library is byte-identical; 96 unit tests.
- **Float mode**: 200 float-mode programs valid, free of undefined behaviour and
  terminating under Debug safety; integer seeds byte-identical over 300 seeds. The first
  float sweep gave 90 match, 8 `MISMATCH` (all the signalling-NaN difference in
  `fmin`/`fmax`, now normalised) and 2 `REFUSED` (LLVM 22 spellings, now parsed); after
  those changes a fresh 100-seed float sweep is 100 match, a float program passes through
  `crosscheck-apple`, the default library is byte-identical and 114 unit tests pass.
- **Float mode after its review**: eight confirmed findings fixed (the fma constant-product
  flush recorded as the sixth Metal finding; exact-width float constants so a signalling
  NaN stays signalling; strings left alone by the text rewrites; the reducer judging under
  the failing seed's inputs and mode, cleaning its cache on every exit and reverting the
  trial it was in when its budget ran out). Another 100-seed float sweep is 100 match; 115
  unit tests.
- The default build is untouched: `default.metallib` is byte-identical with and without
  the bench.

## Next

- Vectors and floats. Floats need explicit tolerance rules (contraction, denormals,
  transcendental precision).
- A reducer that shrinks a failing seed to a minimal program before it is reported.
- Later, the same generated shaders on Asahi Linux's Mesa stack once its M3 GPU driver
  works, so the two compilers can be compared on the same hardware.
