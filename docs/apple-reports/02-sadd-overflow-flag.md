# The overflow flag of a signed 32-bit add (`__builtin_sadd_overflow`, `llvm.sadd.with.overflow.i32`) is computed as `(a + b) < a` on the GPU

**Area:** Metal → Metal Compiler (GPU back end). **Type:** incorrect results.
**Environment:** macOS 26.3 (25D125), Apple M3, Xcode 26.6 (17F113), metal 32023.883.

## Summary

In a compute kernel, `__builtin_sadd_overflow(a, b, &r)` returns the wrong flag for about
half of all inputs. The value the GPU produces is `(int)(a + b) < a`, a signed comparison of
the wrapped sum with the first operand, which is the formula for the *unsigned* carry.
Signed overflow happens exactly when `a` and `b` have the same sign and the sum's sign
differs. The unsigned builtin `__builtin_uadd_overflow` is correct.

## Steps to reproduce

```sh
xcrun metal -fno-fast-math add_overflow_bit.metal -o add_overflow_bit.metallib
swift run_kernel.swift add_overflow_bit.metallib 0x80000000,0xFFFFFFFF 0xC0000000,0xC0000000 \
    0x7FFFFFFF,0x80000000 0x80000000,0x80000000 0x7FFFFFFF,1 1,1
```

The kernel body is

```metal
static int body(int a, int b) {
    int r;
    return (int)__builtin_sadd_overflow(a, b, &r);
}
```

## Expected and actual results

| a | b | a + b | expected flag | GPU |
| --- | --- | --- | --- | --- |
| `0x80000000` (INT_MIN) | `0xFFFFFFFF` (−1) | overflows | 1 | **0** |
| `0xC0000000` (−2³⁰) | `0xC0000000` (−2³⁰) | −2³¹, fits | 0 | **1** |
| `0x7FFFFFFF` (INT_MAX) | `0x80000000` (INT_MIN) | −1, fits | 0 | **1** |
| `0x80000000` (INT_MIN) | `0x80000000` (INT_MIN) | overflows | 1 | **0** |
| `0x7FFFFFFF` (INT_MAX) | `1` | overflows | 1 | 1 |
| `1` | `1` | 2 | 0 | 0 |

Every GPU value in the table, and every one of 1,990 wrong results among 4,096 random and
edge-value input pairs, equals `(int)(a + b) < a`.

## Notes

- Same results with the default `-ffast-math`, and from LLVM IR (`llvm.sadd.with.overflow.i32`)
  assembled by `xcrun metal -Xclang -opaque-pointers`, so this is the back end's lowering.
- `__builtin_uadd_overflow` on `uint` is correct on all 4,096 inputs, as is
  `llvm.smul.with.overflow.i32` from IR. The subtract flag has the matching defect
  (separate report: it is computed as `a < b`).

## Attachments

- `add_overflow_bit.metal`, `run_kernel.swift`.
