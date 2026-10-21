# `frem` is compiled as the fast `fmod`: inexact remainders and `-inf` for finite operands, while `air.fmod.f32` is exact

**Area:** Metal → Metal Compiler (GPU back end). **Type:** incorrect results.
**Environment:** macOS 26.3 (25D125), Apple M3, Xcode 26.6 (17F113), metal 32023.883.

## Summary

LLVM's `frem` instruction has the semantics of C's `fmod`: the result is exact (the
remainder of two finite floats is always representable). In a Metal compute kernel,
`frem float %x, %y` is lowered to an approximation that behaves like `x - y * trunc(x / y)`
evaluated in single precision: it returns 0 where the remainder is 1.0e-10, 8 where it is
6.9, and `-inf` when `x / y` overflows. Metal Shading Language's `fmod`, compiled with
`-fno-fast-math` to `air.fmod.f32`, is exact on the same inputs; its fast variant
(`air.fast_fmod.f32`) shows the same errors as `frem`. So `frem` is lowered as the fast
function even when the module is not compiled with fast math.

## Steps to reproduce

```sh
xcrun metal -Xclang -opaque-pointers rem_inexact.air.ll -o rem.metallib
swift run_kernel.swift rem.metallib 0x55555555,0x2F33412F 0x4C52B571,0xC413904C 0x5EFD5A65,0x0C5F3564
```

`rem_inexact.air.ll` (attached) is a compute kernel whose body is `frem` of the two inputs
taken as floats, with denormals flushed before and after (the GPU flushes them anyway) and
the result returned as its bit pattern; NaN is canonicalised. It declares
`fast_math_disable`.

## Expected and actual results

| x | y | expected (`fmod(x, y)`) | GPU |
| --- | --- | --- | --- |
| `0x55555555` (1.466015e+13) | `0x2F33412F` (1.630311e-10) | `0x2EE0C242` (1.022085e-10) | **`0x00000000` (0)** |
| `0x4C52B571` (5.523604e+07) | `0xC413904C` (-590.2546) | `0x40DD3800` (6.913086) | **`0x41000000` (8)** |
| `0x5EFD5A65` (9.128008e+18) | `0x0C5F3564` (1.719536e-31) | `0x0C4298BC` (1.499118e-31) | **`0xFF800000` (-inf)** |

Over 4,096 random and edge-value input pairs, 1,067 differ. The kernel with the body
written in MSL as `fmod(x, y)` and compiled with `-fno-fast-math` matches the expected
value on all 4,096 (with `-ffast-math`, 941 differ).

## Notes

- Expected values are C `fmod` computed on the CPU; they are exact by definition and
  representable.
- `-inf` for finite operands cannot be a valid approximation under any precision bound.
- The result does not depend on the library's `air.compile_options` fast-math declaration.

## Attachments

- `rem_inexact.air.ll`, `run_kernel.swift`; for comparison, `rem_inexact.metal` (the MSL
  body with `fmod`).
