# `fmod(half, half)` returns `+0.0` where the remainder is `-0.0`, with `-fno-fast-math`

**Area:** Metal → Metal Compiler (GPU back end). **Type:** incorrect results.
**Environment:** macOS 26.3 (25D125), Apple M3, Xcode 26.6 (17F113), metal 32023.883.

## Summary

`fmod(x, y)` on `half` (`air.fmod.f16` under `-fno-fast-math`) returns `+0.0` whenever x
is negative and the remainder is exactly zero, for example `fmod(-20016.0h, 2.0h)`. C's
`fmod`, IEEE 754's remainder operations, and Metal's own single-precision `fmod` give the
result the sign of x: `-0.0`. Non-zero results are correct: over 4,096 random and edge-value
input pairs, the 9 that differ are all exact-zero remainders of a negative x.

## Steps to reproduce

```sh
xcrun metal -fno-fast-math fmod_half_negative_zero.metal -o fmod_half.metallib
swift run_kernel.swift fmod_half.metallib 0xF4E3,0x40000000 0xE3B8,0xC0000000 0x74E3,0x40000000
```

The kernel takes x from the low 16 bits of the first input and y from the high 16 bits of
the second, computes `fmod(x, y)` on `half`, and returns the result's bit pattern (a NaN
is canonicalised to `0x7E00`).

## Expected and actual results

| x | y | expected (`fmod`) | GPU |
| --- | --- | --- | --- |
| `0xF4E3` (−20016) | `0x4000` (2) | `0x8000` (−0.0) | **`0x0000` (+0.0)** |
| `0xE3B8` (−988) | `0xC000` (−2) | `0x8000` (−0.0) | **`0x0000` (+0.0)** |
| `0x74E3` (+20016) | `0x4000` (2) | `0x0000` (+0.0) | `0x0000` |

## Notes

- Single-precision `fmod` keeps the sign and is exact on all 4,096 inputs in the same
  harness, so this is specific to `half`.
- The LLVM `frem half` instruction, assembled by `xcrun metal -Xclang -opaque-pointers`,
  is lowered differently again and differs from the exact remainder on 672 of 4,096 inputs
  (see the `frem` report for single precision).

## Attachments

- `fmod_half_negative_zero.metal`, `run_kernel.swift`.
