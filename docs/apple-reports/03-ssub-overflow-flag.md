# The overflow flag of a signed 32-bit subtract (`__builtin_ssub_overflow`, `llvm.ssub.with.overflow.i32`) is computed as `a < b` on the GPU

**Area:** Metal → Metal Compiler (GPU back end). **Type:** incorrect results.
**Environment:** macOS 26.3 (25D125), Apple M3, Xcode 26.6 (17F113), metal 32023.883.

## Summary

In a compute kernel, `__builtin_ssub_overflow(a, b, &r)` returns `a < b` (signed) instead
of the overflow flag of `a - b`. `a < b` is the formula for the *unsigned* borrow; signed
overflow happens exactly when `a` and `b` have different signs and the difference's sign
differs from `a`'s. The unsigned builtin `__builtin_usub_overflow` is correct.

## Steps to reproduce

```sh
xcrun metal -fno-fast-math sub_overflow_bit.metal -o sub_overflow_bit.metallib
swift run_kernel.swift sub_overflow_bit.metallib 0,0x80000000 0x7FFFFFFF,0xFFFFFFFF \
    0xFFFFFFFF,0x7FFFFFFF 3,5 0,1 0x80000000,1
```

The kernel body is

```metal
static int body(int a, int b) {
    int r;
    return (int)__builtin_ssub_overflow(a, b, &r);
}
```

## Expected and actual results

| a | b | a − b | expected flag | GPU |
| --- | --- | --- | --- | --- |
| `0` | `0x80000000` (INT_MIN) | overflows | 1 | **0** |
| `0x7FFFFFFF` (INT_MAX) | `0xFFFFFFFF` (−1) | overflows | 1 | **0** |
| `0xFFFFFFFF` (−1) | `0x7FFFFFFF` (INT_MAX) | INT_MIN, fits | 0 | **1** |
| `3` | `5` | −2 | 0 | **1** |
| `0` | `1` | −1 | 0 | **1** |
| `0x80000000` (INT_MIN) | `1` | overflows | 1 | 1 |

Every GPU value in the table, and every one of 2,103 wrong results among 4,096 random and
edge-value input pairs, equals `a < b` as a signed comparison.

## Notes

- Same results with the default `-ffast-math`, and from LLVM IR (`llvm.ssub.with.overflow.i32`)
  assembled by `xcrun metal -Xclang -opaque-pointers`.
- `__builtin_usub_overflow` on `uint` is correct on all 4,096 inputs. The add flag has the
  matching defect (separate report: it is computed as `(a + b) < a`).

## Attachments

- `sub_overflow_bit.metal`, `run_kernel.swift`.
