# `round()` returns `+0.0` where the result is `-0.0`, with `-fno-fast-math`

**Area:** Metal → Metal Compiler (GPU back end). **Type:** incorrect results.
**Environment:** macOS 26.3 (25D125), Apple M3, Xcode 26.6 (17F113), metal 32023.883.

## Summary

With fast math disabled, `round(x)` (`air.round.f32`) returns `+0.0` for every input whose
rounded value is a negative zero: `-0.3`, `-0.49999997`, and `-0.0` itself. IEEE 754's
roundToIntegralTiesToAway and C's `round` return `-0.0` for these; `floor`, `ceil` and
`trunc` on the same GPU keep the sign. Results other than zero are correct (`round(-0.5)`
is `-1.0`).

## Steps to reproduce

```sh
xcrun metal -fno-fast-math round_negative_zero.metal -o round.metallib
swift run_kernel.swift round.metallib 0xBE99999A,0 0x80000000,0 0xBEFFFFFF,0 0xBF000000,0 0xBFC00000,0
```

The kernel body is `round(x)` of the first input taken as a float, with denormals flushed
first (the GPU flushes them anyway) and the result returned as its bit pattern; NaN is
canonicalised. The second input is unused.

## Expected and actual results

| x | expected | GPU |
| --- | --- | --- |
| `0xBE99999A` (−0.3) | `0x80000000` (−0.0) | **`0x00000000` (+0.0)** |
| `0x80000000` (−0.0) | `0x80000000` (−0.0) | **`0x00000000` (+0.0)** |
| `0xBEFFFFFF` (−0.49999997) | `0x80000000` (−0.0) | **`0x00000000` (+0.0)** |
| `0xBF000000` (−0.5) | `0xBF800000` (−1.0) | `0xBF800000` |
| `0xBFC00000` (−1.5) | `0xC0000000` (−2.0) | `0xC0000000` |

Over 4,096 random and edge-value inputs, 949 differ, all of them results that should be
`-0.0`.

## Notes

- `-fno-fast-math` is the mode in which the sign of zero is significant per the Metal
  Shading Language specification; the report is about that mode. (With `-ffast-math`,
  `air.fast_round.f32` is used, and the sign may legitimately be dropped.)
- The front end assumes `round` preserves the sign: `copysign(round(x), x)` is optimised
  to `round(x)` (visible with `xcrun metal -fno-fast-math -S -emit-llvm`), so the two halves
  of the compiler disagree, and the obvious source-level workaround is removed.
- The same result comes from LLVM IR using `llvm.round.f32`, assembled by `xcrun metal`,
  and does not depend on the library's `air.compile_options` fast-math declaration.

## Attachments

- `round_negative_zero.metal`, `run_kernel.swift`.
