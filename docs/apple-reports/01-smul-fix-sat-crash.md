# MTLCompilerService fails ("XPC_ERROR_CONNECTION_INTERRUPTED") building a compute pipeline from a kernel that uses `llvm.smul.fix.sat.i32`

**Area:** Metal → Metal Compiler (GPU back end). **Type:** compiler failure on valid input.
**Environment:** macOS 26.3 (25D125), Apple M3, Xcode 26.6 (17F113), metal 32023.883.

## Summary

A compute kernel whose body is `call i32 @llvm.smul.fix.sat.i32(i32 %a, i32 %b, i32 0)`,
LLVM's saturating 32-bit multiply with a zero scale, is accepted by `xcrun metal` and
packed into a `.metallib`, but `makeComputePipelineState(function:)` fails: the back-end
compiler service is interrupted, and after the framework's retries the error is reported
as `AGXMetalG15G_C0` code 2. The same saturating multiply written in Metal Shading
Language as `madsat(a, b, 0)` builds and computes correctly on every input tested.

## Steps to reproduce

```sh
xcrun metal -Xclang -opaque-pointers mul_sat.air.ll -o mul_sat.metallib    # succeeds
swift run_kernel.swift mul_sat.metallib 3,5
```

`mul_sat.air.ll` (attached, 62 lines) is a compute kernel `crosscheckKernel` that writes
`out[gid] = smul.fix.sat(in0[gid], in1[gid], 0)` for `gid < count`, with the usual
`!air.*` metadata; it declares `fast_math_disable` and `denorms_disable`.

## Expected results

The pipeline builds and the kernel writes 15; or, if the intrinsic is unsupported by the
back end, `xcrun metal` rejects it at assembly time with a diagnostic.

## Actual results

```
FAIL Error Domain=AGXMetalG15G_C0 Code=2 "Compilation failed due to an interrupted connection:
XPC_ERROR_CONNECTION_INTERRUPTED. This error occurred after multiple retries."
```

The same failure happens whether the bitcode comes from Apple's assembler (as above) or
from another bitcode writer, so it is not an encoding issue, and whether the library
declares `fast_math_enable` or `fast_math_disable`.

## Notes

- `madsat(a, b, 0)` in MSL compiles to `air.mad_sat.s.i32` and, run over 4,096 random and
  edge inputs, agrees with the reference saturating multiply on every one. So the operation
  is supported by the hardware path; only the lowering of the LLVM intrinsic fails.
- `llvm.sadd.sat`/`llvm.ssub.sat` (saturating add and subtract) and
  `llvm.smul.with.overflow.i32` compile and run correctly in the same kernel shape.
- The IR is what Zig emits for `a *| b`; the same intrinsic is reachable from clang's
  fixed-point extension, so this is not specific to one producer.

## Attachments

- `mul_sat.air.ll`: the kernel.
- `run_kernel.swift`: loads a `.metallib`, builds the pipeline, runs the kernel on the
  given `a,b` pairs and prints the results.
