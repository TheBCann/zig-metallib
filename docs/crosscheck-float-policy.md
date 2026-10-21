# Floats in the crosscheck bench: policy

*Confirmed 2026-10-20; the open questions at the end were settled as recommended.* The
crosscheck bench compares what the GPU and the CPU compute for the
same Zig program. For integers "the same" means bit-equal. For floats it needs a policy,
because some differences are not bugs: the GPU flushes single-precision denormals,
canonicalises NaNs, and fuses a multiply and an add that the source wrote separately. This
document fixes what the bench normalises, what it requires to match exactly, what it only
measures, and what it already found while being written. Float programs are generated with
`-Dcrosscheck-floats=true`, so the integer seeds referenced elsewhere keep their meaning.

## What the probes established

Every probe below is a hand-written body run through the real pipeline on the Apple M3
(macOS 26.3, Zig 0.18.0-dev.253), with floats carried as i32 bit patterns and compared
bit for bit over the driver's 4,096 input pairs. "Raw" is the operation alone; "normalised"
applies the rules of the next section inside the program.

| Operation | Raw | Normalised | What the difference was |
| --- | --- | --- | --- |
| `+`, `-`, `*`, `/` on f32 | 1,100–1,500 differ | **all match** | NaN payload/sign, denormals flushed on input and output |
| `@mulAdd` (explicit fma) | 1,424 differ | **all match** | same |
| `x * y + x` written separately | 1,429 differ | 6 differ | the GPU flushes the denormal intermediate product that the CPU keeps; a flush between the two operations models it, then all match. Not contraction: `x * (1 + 2^-23) - x`, which a fused multiply-add computes differently for nearly every x, matches on all 4,096 inputs |
| `@sqrt` | 1,024 differ | **all match** | denormal inputs flushed (`sqrt(-denormal)` is NaN on the CPU, `-0.0` on the GPU) |
| f16 `*`, `+`, `/`, `@sqrt` | 986 differ | **all match** | NaN payloads only: **f16 denormals are kept** |
| f16 `@rem` | | 9 differ through Apple's front end | **finding:** `air.fmod.f16` (MSL `fmod` on half) returns `+0.0` for an exact `-0.0` remainder; the assembler's `frem` lowering adds a `copysign` for half, then all match |
| `@floor`, `@ceil`, `@trunc` | | **all match** | |
| `@round` | | 949 differ | **finding:** the GPU returns `+0.0` where the result is `-0.0`; worked around in the assembler since 2026-10-20, now all match |
| `@min`, `@max` | 369 differ | 58 differ, then **all match** | which zero is returned is unspecified by IEEE; canonicalising the zero sign after them fixes it |
| comparisons and `if`/`else` | | **all match** | |
| `i32 → f32`, `i16 → f32` | | **all match** | |
| `f32 → i32` (clamped, NaN guarded) | | **all match** | |
| `@rem` on f32 | | 1,067 differ | **finding:** not the exact remainder (0 for 1.02e-10, 8 for 6.91, `-inf` for finite operands); worked around in the assembler since 2026-10-20, now all match |
| `@exp` | 718 differ | | besides NaN/denormals, a 5-ulp difference in one sample |
| `@sin` | 1,754 differ | | the GPU returns 0 for huge arguments (`sin(1.7e22)`) |
| 4-lane vectors, lane-wise | | **all match** | with the flush between dependent operations |
| `@setFloatMode(.optimized)` | CPU panics | | Zig's Debug build treats float division by zero as undefined in that mode |

Three behaviours of the GPU are systematic and documented by Metal, so they are
normalised rather than reported: single-precision denormals are flushed to zero on input,
on output and between dependent operations (half precision is not), NaNs are canonicalised,
and `fmin`/`fmax` treat a signalling NaN as quiet. The GPU does **not** contract a separate
multiply and add when the IR carries no `contract` flag: the six differences in the
`x * y + x` row were first read as contraction, and a probe that a fused multiply-add
computes differently for nearly every input, `x * (1 + 2^-23) - x`, matches on all 4,096
inputs (2026-10-20); the six were denormal intermediate products, flushed on the GPU and
kept on the CPU. The signalling-NaN behaviour was found by the first float-mode sweep: eight
of a hundred seeds mismatched, every one through a signalling NaN reaching `fmin`/`fmax`
(the reducer brought seed 15 down to `fmax(fin(a), FLT_MIN)` in two minutes).

## The policy

1. **Representation.** Float programs use the existing i32 harness: the generator derives
   float values from the two integer inputs by bit-cast and returns the result's bits.
   Nothing changes in the driver or the kernel wrapper. The driver's edge-value list gains
   float edge patterns: ±0, ±1, ±0.5, ±2, ±FLT_MIN, ±FLT_MAX, ±inf, quiet NaN with
   payloads, the largest denormal, 2^24 and its neighbours, and the integer-conversion
   limits.
2. **Normalisation, written into the generated program** so both sides compute the same
   thing and the comparison stays bit-exact:
   - every f32 input and the result of every f32 operation pass through a flush helper
     (denormal → signed zero), a no-op on the GPU and an emulation of it on the CPU; the
     helper tests the exponent bits rather than comparing, so LLVM does not turn it into
     `llvm.is.fpclass`;
   - a NaN entering from an integer bit pattern is quieted to the canonical NaN: Metal's
     `fmin`/`fmax` treat a signalling NaN like a quiet one and return the other operand,
     while the CPU returns NaN (IEEE minNum with a signalling input), a documented
     difference that would otherwise be rediscovered on most seeds;
   - a NaN becomes the canonical NaN (`0x7fc00000`, f16 `0x7e00`) where a float turns back
     into bits;
   - the result of `@min`/`@max` is passed through `+ 0.0`, which turns `-0.0` into `+0.0`
     and changes nothing else, because IEEE leaves the choice of zero unspecified;
   - everywhere else the sign of zero is significant.
3. **Intermediate results.** The flush helper between dependent operations makes the CPU
   flush what the GPU flushes, a denormal product before the add that consumes it. It
   would also stop a contraction, but Metal does not contract separate operations (above),
   so the fused multiply-add is covered only where the program asks for it, by `@mulAdd`.
4. **Exact mode** (a difference is a finding): `+`, `-`, `*`, `/`, negation, `@abs`,
   `@sqrt`, `@mulAdd`, `@floor`, `@ceil`, `@trunc`, `@round` and `@rem` (through the
   assembler's workarounds), `@min`, `@max`, comparisons feeding
   `if`/`else`, `i32 → f32`, guarded `f32 → i32`, `f32 ↔ f16` and f16 arithmetic, 2- and
   4-lane vectors lane-wise, and reductions written as explicit lane sums in a fixed order.
5. **Not in exact mode:** `@mod` on floats (Zig lowers it to `frem` plus sign fix-ups, not
   yet probed); `exp`, `log`, `sin`, `cos`, `pow`, which are measured by a
   separate **precision survey** that reports the ulp distribution per operation over a
   bounded domain rather than passing or failing; `@setFloatMode(.optimized)`, which the
   generator never emits; f64, which Metal does not have.
6. **No undefined behaviour.** The only float construct Zig treats as checked undefined
   behaviour in strict mode is `f32 → i32` with an out-of-range or NaN value, so the
   generator clamps to `[-2147483648, 2147483520]` and maps NaN to 0 explicitly (LLVM makes
   that conversion poison on the GPU). Division, `@rem`, `@mod` and `@sqrt` of anything,
   and `f32 → f16` overflow, are all defined. The Debug CPU build keeps catching slips.
7. **Reduction and attribution** work unchanged: a float program is a program, and the
   Apple-assembler step still decides whose bug a finding is.

## Found while writing this

- **In this project, fixed:** LLVM 22 prints a float constant that does not round-trip
  through decimal as `f0x` followed by the bit pattern of its own type. Apple's LLVM-15-era
  reader rejects the spelling, and so did our assembler, so any shader with such a constant
  (`FLT_MIN`, `-2147483648.0`, …) was refused. The assembler now parses the form and the
  splice rewrites it to the legacy spelling in the text it hands to Apple's assembler. The
  first float-mode sweep then found two more LLVM 22 spellings the same way: the special
  values `+qnan`, `-nan(0x3FFFFE)`, `snan`, `inf` (parsed, and rewritten to legacy hex with
  the payload kept exactly; a signalling NaN must not be widened by the hardware, which
  would quiet it), and `llvm.is.fpclass` with an inline `/* (nan inf norm) */` comment,
  which LLVM makes of `fabs(x) < FLT_MIN`. The assembler lowers the class test to integer
  tests on the bit pattern, since Metal has no such intrinsic, and the splice drops the
  comment.
- **Metal, recorded under "Findings" in `docs/crosscheck.md`:** `@round` loses the sign of
  a negative zero; `@rem` is not the exact remainder; an fma with two constant
  multiplicands whose exact product is a denormal is folded and flushed, where the hardware
  fma keeps the intermediate (found by the review of the float-mode generator, which now
  always gives an fma an input-dependent multiplicand, since LLVM folds even a local that
  only ever held constants). All persist through Apple's assembler. `@round` and `@rem`
  are worked around in the assembler since 2026-10-20 (`trunc(x + copysign(0.5 - ulp, x))`
  and a call to Apple's exact `air.fmod.f32`), and the generator emits them again. Probing
  the `frem` workaround in half precision found a seventh finding: `air.fmod.f16` returns
  `+0.0` for an exact `-0.0` remainder (the assembler adds a `copysign` for half).
- **Through Apple's front end** (`zig build crosscheck-msl`, 2026-10-20): the Metal Shading
  Language twins of the reproducers, compiled with `xcrun metal -fno-fast-math`, reproduce
  the `@round` sign loss (`air.round.f32`, 949) and the fma fold (`air.fma.f32`, 223)
  exactly, while MSL's `fmod` (`air.fmod.f32`) is exact on all 4,096 inputs where LLVM's
  `frem` is not: the `@rem` finding is in how `frem` is lowered, and lowering it to
  `air.fmod.f32` is the fix for this project. The fma with its multiplicand read from a
  buffer instead of a constant is exact on every input: the hardware fma keeps the
  intermediate, the compiler's constant fold does not.
- **Documented deviations, normalised by the policy:** single-precision flush to zero
  (inputs, outputs and intermediates), NaN canonicalisation, signalling NaNs quieted by
  `fmin`/`fmax`.

## Precision survey

`zig build survey` (`tools/crosscheck/survey.zig`, kernels in `src/engine/survey_harness.zig`)
runs each transcendental function over sampled domains on the GPU and reports the error in
units of the f32 ulp at an f64 reference computed on the CPU: the share of results that are
correctly rounded (within 0.5 ulp), within 1 ulp, and the worst case. Each function comes in
three variants: the Zig builtin as the assembler lowers it, Apple's precise `air.<f>.f32`
(what `xcrun metal -fno-fast-math` emits) and Apple's `air.fast_<f>.f32`. It is a
measurement, not a check. `-Dsurvey-samples=N` and `-Dseed=N` vary the inputs. Measured on
the M3 with 65,536 samples per row (seed 1), 2026-10-20:

| function | domain | precise `air.<f>`: max ulp, within 1 ulp | fast `air.fast_<f>`: max ulp, within 1 ulp |
| --- | --- | --- | --- |
| exp | [−87, 88] | 1.39, 93.4% | **62.6, 10.8%** (68.6% beyond 4 ulp) |
| exp2 | [−126, 127] | 1.45, 99.99% | 1.21, 100% |
| log | [FLT_MIN, FLT_MAX] | 1.85, 87.3% | 2.90, 87.5% |
| log2 | [FLT_MIN, FLT_MAX] | 2.10, 99.9% | 2.22, 99.9% |
| log10 | [FLT_MIN, FLT_MAX] | 2.25, 98.8% | 2.94, 98.8% |
| sin | [−2π, 2π] | 2.38, 88.4% | 2.48, 88.5% |
| sin | [−1e5, 1e5] | 2.38, 88.4% | 2.45, 88.3% |
| sin | ±[1e5, FLT_MAX] | 2.57, 89.0% | **1.7e7, 4.9%** (returns 0: 94.5% beyond 4 ulp) |
| cos | [−2π, 2π] | 2.54, 88.5% | 2.52, 88.2% |
| cos | ±[1e5, FLT_MAX] | 2.55, 88.9% | **1.7e7, 4.8%** |
| tan | [−2π, 2π] | 3.51, 78.5% | 4.46, 71.3% |
| tan | [−1e5, 1e5] | 3.92, 78.6% | 4.25, 71.7% |
| sqrt | [FLT_MIN, FLT_MAX] | 0.50, 100% (correctly rounded) | 1.59, 97.9% |

What the builtins get: `llvm.exp2`, `llvm.log`, `llvm.log2` and `llvm.sqrt` pass through to
Metal and measure like the precise functions (1.08, 3.03, 1.61 and 0.50 ulp at most).
`llvm.exp` passed through measured like `air.fast_exp` (60.2 ulp at most, 67.6% beyond 4 ulp
over 4,096 samples), so the assembler renames it to `air.exp` now; and the functions whose
`llvm.*` form crashes the compiler (`sin`, `cos`, `tan`, `log10`, `exp10`, `pow`) are renamed
to the precise `air.*` instead of `air.fast_*`, unless the library declares fast math
(`-Dmetal-fast-math=true`), as Apple's own front end chooses. The fast `sin` and `cos`
return 0 for arguments beyond about 1e5; the precise ones reduce the argument correctly up
to FLT_MAX. `pow` has no Zig builtin, so it is not surveyed.

## The fast-math declaration

Every library this project writes carries `!air.compile_options`, copied from Apple's
output, and until 2026-10-20 that included `air.compile.fast_math_enable`, what `xcrun
metal` writes by default. Metal's fast math permits ignoring the sign of zero and assuming
no NaN or infinity, so the declaration could have licensed the `@round` finding. It does
not. `-Dmetal-fast-math=false` writes `air.compile.fast_math_disable`, as `xcrun metal
-fno-fast-math` does, and every probe then gives the same result as before, through our
assembler and through Apple's: `@round` 949, `@rem` 1,067 and the fma fold 223 of 4,096
under seed 1, the contraction probe all match, `fmax` of a signalling NaN still returns the
other operand, and the 42 checks pass. The declaration is informational: Apple's front end
encodes fast math in the IR itself, by choosing `air.fast_round.f32` over `air.round.f32`
and `air.fast_fmod.f32` over `air.fmod.f32` (both seen in `xcrun metal -S -emit-llvm`
output), and the back end reads the instructions, not the flag. The default is now
`fast_math_disable`, because Zig compiles the shaders in strict float mode and the IR
carries no fast-math flags; `-Dmetal-fast-math=true` restores Apple's default wording.
`default.metallib` changed bytes with the new default and passes the 42 checks.

## Decisions taken

1. A flush between dependent operations stays in every generated program: it models the
   GPU's flushing of intermediate results, and it would keep any contraction out of the
   comparison, although none was measured.
2. The sign of zero is significant everywhere except after `@min`/`@max`.
3. The precision survey for `exp`, `log`, `sin`, `cos` and `tan` reports measured ulp
   distributions (above); it has no pass/fail threshold for now.
4. f32 first; f16 as a generated type comes after.
