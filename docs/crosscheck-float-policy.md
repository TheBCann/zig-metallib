# Floats in the crosscheck bench: policy (draft for confirmation)

*Draft, 2026-10-20.* The crosscheck bench compares what the GPU and the CPU compute for the
same Zig program. For integers "the same" means bit-equal. For floats it needs a policy,
because some differences are not bugs: the GPU flushes single-precision denormals,
canonicalises NaNs, and fuses a multiply and an add that the source wrote separately. This
document fixes what the bench normalises, what it requires to match exactly, what it only
measures, and what it already found while being written. Points marked **[decision]** are
the ones to confirm before the generator grows float support.

## What the probes established

Every probe below is a hand-written body run through the real pipeline on the Apple M3
(macOS 26.3, Zig 0.18.0-dev.253), with floats carried as i32 bit patterns and compared
bit for bit over the driver's 4,096 input pairs. "Raw" is the operation alone; "normalised"
applies the rules of the next section inside the program.

| Operation | Raw | Normalised | What the difference was |
| --- | --- | --- | --- |
| `+`, `-`, `*`, `/` on f32 | 1,100–1,500 differ | **all match** | NaN payload/sign, denormals flushed on input and output |
| `@mulAdd` (explicit fma) | 1,424 differ | **all match** | same |
| `x * y + x` written separately | 1,429 differ | 6 differ | Metal fused the two operations into an fma; a flush between them stops it, then all match |
| `@sqrt` | 1,024 differ | **all match** | denormal inputs flushed (`sqrt(-denormal)` is NaN on the CPU, `-0.0` on the GPU) |
| f16 `*`, `+`, `/`, `@sqrt` | 986 differ | **all match** | NaN payloads only: **f16 denormals are kept** |
| `@floor`, `@ceil`, `@trunc` | | **all match** | |
| `@round` | | 949 differ | **finding:** the GPU returns `+0.0` where the result is `-0.0` |
| `@min`, `@max` | 369 differ | 58 differ, then **all match** | which zero is returned is unspecified by IEEE; canonicalising the zero sign after them fixes it |
| comparisons and `if`/`else` | | **all match** | |
| `i32 → f32`, `i16 → f32` | | **all match** | |
| `f32 → i32` (clamped, NaN guarded) | | **all match** | |
| `@rem` on f32 | | 1,067 differ | **finding:** not the exact remainder (0 for 1.02e-10, 8 for 6.91, `-inf` for finite operands) |
| `@exp` | 718 differ | | besides NaN/denormals, a 5-ulp difference in one sample |
| `@sin` | 1,754 differ | | the GPU returns 0 for huge arguments (`sin(1.7e22)`) |
| 4-lane vectors, lane-wise | | **all match** | with the flush between dependent operations |
| `@setFloatMode(.optimized)` | CPU panics | | Zig's Debug build treats float division by zero as undefined in that mode |

Three behaviours of the GPU are systematic and documented by Metal, so they are
normalised rather than reported: single-precision denormals are flushed to zero on input
and output (half precision is not), NaNs are canonicalised, and a separate multiply and
add are contracted into an fma even though the IR carries no `contract` flag.

## The policy

1. **Representation.** Float programs use the existing i32 harness: the generator derives
   float values from the two integer inputs by bit-cast and returns the result's bits.
   Nothing changes in the driver or the kernel wrapper. The driver's edge-value list gains
   float edge patterns: ±0, ±1, ±0.5, ±2, ±FLT_MIN, ±FLT_MAX, ±inf, quiet NaN with
   payloads, the largest denormal, 2^24 and its neighbours, and the integer-conversion
   limits. **[decision]**
2. **Normalisation, written into the generated program** so both sides compute the same
   thing and the comparison stays bit-exact:
   - every f32 input and the result of every f32 operation pass through a flush helper
     (denormal → signed zero), a no-op on the GPU and an emulation of it on the CPU;
   - a NaN becomes the canonical NaN (`0x7fc00000`, f16 `0x7e00`) where a float turns back
     into bits;
   - the result of `@min`/`@max` is passed through `+ 0.0`, which turns `-0.0` into `+0.0`
     and changes nothing else, because IEEE leaves the choice of zero unspecified;
   - everywhere else the sign of zero is significant. **[decision]**
3. **Contraction.** The flush helper between dependent operations is also the contraction
   barrier: with it the GPU and the CPU both compute the unfused result. Fused
   multiply-add is covered explicitly by `@mulAdd`. Metal's default contraction is
   recorded as a documented deviation, not a finding. **[decision]**
4. **Exact mode** (a difference is a finding): `+`, `-`, `*`, `/`, negation, `@abs`,
   `@sqrt`, `@mulAdd`, `@floor`, `@ceil`, `@trunc`, `@min`, `@max`, comparisons feeding
   `if`/`else`, `i32 → f32`, guarded `f32 → i32`, `f32 ↔ f16` and f16 arithmetic, 2- and
   4-lane vectors lane-wise, and reductions written as explicit lane sums in a fixed order.
5. **Not in exact mode:** `@round` and `@rem`/`@mod`, which are findings and would be
   rediscovered on every seed; `exp`, `log`, `sin`, `cos`, `pow`, which are measured by a
   separate **precision survey** that reports the ulp distribution per operation over a
   bounded domain rather than passing or failing; `@setFloatMode(.optimized)`, which the
   generator never emits; f64, which Metal does not have. **[decision: survey, not
   pass/fail, for transcendentals]**
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
  splice rewrites it to the legacy spelling in the text it hands to Apple's assembler.
- **Metal, recorded under "Findings" in `docs/crosscheck.md`:** `@round` loses the sign of
  a negative zero; `@rem` is not the exact remainder. Both persist through Apple's
  assembler.
- **Documented deviations, normalised by the policy:** single-precision flush to zero, NaN
  canonicalisation, contraction of separate multiply and add.

## Open questions

1. Is treating contraction as a documented deviation right, or should the bench report it
   as a conformance finding against Metal? The IR we hand over has no fast-math flags.
2. Should the zero sign stay significant outside `@min`/`@max`? Keeping it is what makes
   the `@round` finding visible.
3. For the precision survey: report the measured ulp distribution only, or also fail above
   a threshold such as 4 ulp for `exp`/`log` and 8 ulp for `sin`/`cos` over `|x| ≤ 2^16`?
4. f16 as a first-class generated type in the first float milestone, or after f32?
