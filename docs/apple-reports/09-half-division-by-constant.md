# `half` division by a constant is not correctly rounded, with `-fno-fast-math`

**Area:** Metal → Metal Compiler. **Type:** incorrect results (precision).
**Environment:** macOS 26.3 (25D125), Apple M3, Xcode 26.6 (17F113), metal 32023.883.

## Summary

`x / 33.0h` on `half` differs from the correctly rounded quotient by one ulp on 15 of 4,096
random and edge-value inputs (by `3.0h`: 103, by `10.0h`: 96). Division by a power of two,
division by a run-time divisor, `float` division by the same constants and `half`
multiplication by them are all exact, so the compiler appears to replace the constant
division by a multiplication with the reciprocal, which is not licensed without fast math.

## Steps to reproduce

```sh
xcrun metal -fno-fast-math hdiv_constant.metal -o hdiv.metallib
swift run_kernel.swift hdiv.metallib 0x3ED07A10,0 0x455CEB5C,0 0x3B694F84,0
```

The kernel converts the first input, taken as a float, to `half`, divides by `33.0h`,
widens the quotient to float and returns its bits.

## Expected and actual results

| x (as half) | exact x / 33 | expected (correctly rounded half) | GPU |
| --- | --- | --- | --- |
| `0x3684` (0.407227) | 0.0123402 | `0x2251` (0.01233673) | **`0x2252`** (0.01234436) |
| `0x6AE7` (3534) | 107.0909 | `0x56B1` (107.0625) | **`0x56B2`** (107.125) |
| `0x1B4A` (0.00355911) | 0.000107852 | `0x0711` (0.0001078248) | **`0x0712`** (0.0001078844) |

The GPU values are shown as the float the kernel returns; the half pattern is in brackets.

## Notes

- The Metal Shading Language Specification (section 8.1, Table 8.3) lists half `x / y` as
  correctly rounded without fast math.
- With the divisor read from a buffer instead of a constant, the same division is correctly
  rounded on all 4,096 inputs, as are `half` multiplication by 33 and `float` division by 33.
- Same result from LLVM IR (`fdiv half %x, 0xH5020`) assembled by `xcrun metal -Xclang
  -opaque-pointers`.

## Attachments

- `hdiv_constant.metal`, `run_kernel.swift`.
