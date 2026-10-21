# `fma(c1, c2, x)` with two constant multiplicands is folded to the flushed product: `fma(FLT_MIN, 0.5f, x)` returns `x`

**Area:** Metal → Metal Compiler. **Type:** incorrect results (constant folding under flush-to-zero).
**Environment:** macOS 26.3 (25D125), Apple M3, Xcode 26.6 (17F113), metal 32023.883.

## Summary

When both multiplicands of `fma` are compile-time constants whose exact product is a
single-precision denormal, the compiler folds the product, flushes it to zero, and the fma
degenerates to `x`. The hardware `fma` with the same values arriving at run time keeps the
exact intermediate product, as the definition of fma requires (one rounding of
`c1 * c2 + x`). Two compilations of the same arithmetic therefore give different results.
Measured with `-fno-fast-math`; the same happens with `-ffast-math`.

## Steps to reproduce

```sh
xcrun metal -fno-fast-math fma_constant_product_flush.metal -o fma_const.metallib
xcrun metal -fno-fast-math fma_runtime_product.metal -o fma_runtime.metallib
swift run_kernel.swift fma_const.metallib   0x00800000,0 0x00800001,0 0x00C00000,0 0x80800000,0
swift run_kernel.swift fma_runtime.metallib 0x00800000,0x00800000 0x00800001,0x00800000 0x00C00000,0x00800000 0x80800000,0x00800000
```

The first kernel computes `fma(as_type<float>(0x00800000), as_type<float>(0x3F000000), x)`,
that is `fma(FLT_MIN, 0.5f, x)`, whose exact product is `2^-127`, a denormal. The second
kernel is the control: the same fma, but the first multiplicand is read from the second
input buffer (pass `FLT_MIN` as `b`), so nothing can be folded. Both flush the result and
return its bit pattern.

## Expected and actual results

| x | expected (`fma`, one rounding) | constant kernel (GPU) | run-time kernel (GPU) |
| --- | --- | --- | --- |
| `0x00800000` (FLT_MIN) | `0x00C00000` (1.5 · FLT_MIN) | **`0x00800000`** (x) | `0x00C00000` |
| `0x00800001` | `0x00C00001` | **`0x00800001`** (x) | `0x00C00001` |
| `0x00C00000` (1.5 · FLT_MIN) | `0x01000000` (2 · FLT_MIN) | **`0x00C00000`** (x) | `0x01000000` |
| `0x80800000` (−FLT_MIN) | `0x80000000` (−2^-127, flushed) | **`0x80800000`** (x) | `0x80000000` |

Over 4,096 random and edge-value inputs the constant kernel differs from the reference on
223 of them (every x small enough for `2^-127` to matter); the run-time kernel matches on
all 4,096.

## Notes

- Metal flushes single-precision denormals on inputs and outputs. The exact product inside
  an fma is neither, and `fma(2^-126, 0.5, 2^-126) = 1.5 · 2^-126` is a normal number, which
  the hardware produces. Folding `c1 * c2` as if it were a separate multiply (whose result
  would be flushed) changes the value of the expression.
- Same result from LLVM IR (`llvm.fma.f32` with two constant operands) assembled by
  `xcrun metal -Xclang -opaque-pointers`.

## Attachments

- `fma_constant_product_flush.metal`, `fma_runtime_product.metal`, `run_kernel.swift`.
