# Bug reports for Apple: seven Metal compiler findings

Drafts for Feedback Assistant, one per finding of the crosscheck bench
([../crosscheck.md](../crosscheck.md#findings)). Each report is self-contained: the
reproducer needs only Xcode's Metal toolchain and `swift`, not Zig or this project. The
`.metal` reproducers are the twins in [`../../tools/crosscheck/known/`](../../tools/crosscheck/known/);
the two `.ll` files, for the findings Metal Shading Language cannot express, are in
[`attachments/`](attachments/), along with the fma control kernel.

| # | Report | Reproduces from | Status |
| --- | --- | --- | --- |
| 1 | [MTLCompilerService fails on `llvm.smul.fix.sat.i32`](01-smul-fix-sat-crash.md) | LLVM IR (`xcrun metal` accepts it) | drafted |
| 2 | [Signed add overflow flag is `(a + b) < a`](02-sadd-overflow-flag.md) | MSL `__builtin_sadd_overflow` | drafted |
| 3 | [Signed subtract overflow flag is `a < b`](03-ssub-overflow-flag.md) | MSL `__builtin_ssub_overflow` | drafted |
| 4 | [`round()` returns `+0.0` for `-0.0` results](04-round-negative-zero.md) | MSL `round`, `-fno-fast-math` | drafted |
| 5 | [`frem` is compiled as the fast `fmod`](05-frem-inexact.md) | LLVM IR (`air.fmod.f32` from MSL is exact) | drafted |
| 6 | [`fma` with two constant multiplicands is folded and flushed](06-fma-constant-product.md) | MSL `fma`, `-fno-fast-math` | drafted |
| 7 | [`fmod` on `half` returns `+0.0` for a `-0.0` remainder](07-fmod-half-negative-zero.md) | MSL `fmod(half, half)`, `-fno-fast-math` | drafted |

Fill in the feedback number (`FB…`) in the Status column once filed, and copy it to the
findings table in `docs/crosscheck.md`.

## Environment the reports were measured on

- macOS 26.3 (25D125), Apple M3 (10-core GPU), Metal 4, target `air64_v28-apple-macosx26.0.0`
- Xcode 26.6 (17F113); `xcrun metal --version`: Apple metal version 32023.883 (metalfe-32023.883)
- Swift 6.3.3 (the runner is a script; `swift run_kernel.swift …`)
- Measured 2026-10-20

## Running a reproducer

```sh
xcrun metal -fno-fast-math ../../tools/crosscheck/known/round_negative_zero.metal -o round.metallib
swift run_kernel.swift round.metallib 0xBE99999A,0 0x80000000,0
```

`run_kernel.swift` loads the library, builds a compute pipeline for `crosscheckKernel`,
runs it once per `a,b` pair (32-bit patterns in hex or decimal) and prints the 32-bit
result of each. For the IR reproducers, assemble with `xcrun metal -Xclang -opaque-pointers
file.ll -o file.metallib` first.

Each kernel has the fixed shape of the bench: `out[gid] = body(in0[gid], in1[gid])` with two
`device const int*` inputs, a `device int*` output and a `constant` count. Floats travel as
bit patterns. The expected values in the reports are what the same body computes on the CPU
(Zig, IEEE 754, with single-precision denormals flushed where Metal flushes them), checked
against the GPU over 4,096 inputs by the bench; the counts "n of 4,096" refer to that run.

## What the bench established beyond each report

- The first six persist when the library is assembled by Apple's `metal` from the same IR, and
  with the library's `air.compile_options` declaring `fast_math_enable` or
  `fast_math_disable`; the compile-options flag changes nothing measured.
- The unsigned `__builtin_uadd_overflow` / `__builtin_usub_overflow` are correct on all
  4,096 inputs; `llvm.smul.with.overflow.i32` is correct; `floor`, `ceil`, `trunc` keep the
  sign of zero.
- MSL's `madsat(a, b, 0)` computes the saturating multiply correctly, and MSL's `fmod`
  under `-fno-fast-math` is the exact remainder in single precision.
- This project works around 1, 4, 5 and 7 in its own assembler (Apple's `madsat` and
  `fmod` intrinsics, a sign-keeping expansion of `round`, a `copysign` after `fmod` on
  half); the reports describe the compiler's behaviour, not the workarounds.
