# Zig: `@rem(minInt, -1)` and `@mod(minInt, -1)` are specified but lowered as undefined behaviour

Facts measured on 2026-10-20 with Zig 0.18.0-dev.253+16d31ceed, for an upstream issue.

**Specification.** The language reference requires only `denominator != 0` for `@rem` and
`@mod` (changed by [#23635](https://github.com/ziglang/zig/issues/23635) from `> 0`), while
`@divTrunc`, `@divFloor` and `@divCeil` additionally require
`!(numerator == minInt and denominator == -1)`. So `@rem(minInt, -1)` and
`@mod(minInt, -1)` are defined, and the mathematically exact result, 0, is what comptime
evaluation gives:

```zig
comptime @rem(@as(i32, std.math.minInt(i32)), @as(i32, -1)) == 0   // and @mod likewise
```

**Implementation.** The LLVM backend lowers both to a plain `srem` (plus the sign fix-up for
`@mod`) with a division-by-zero check but no `minInt / -1` guard, in Debug as well as in
release modes. LLVM's `srem` is undefined for exactly that operand pair, so:

| target, mode | `@rem(x, y)` with x = minInt, y = -1 read from argv | 
| --- | --- |
| aarch64-macos, every mode | 0 (the hardware `sdiv` wraps, by luck) |
| x86_64-macos (Rosetta), Debug | crash: `idiv` raises `#DE`, exit 134 after a stack trace with no safety message |
| x86_64-macos (Rosetta), ReleaseFast | crash: SIGFPE, exit 136 |
| aarch64-macos, ReleaseSmall, operands visible to LLVM | 83615291 for `@rem` and 83615312 for `@mod`: the folded `srem` is poison |

`std.math.rem` and `std.math.mod` sidestep the case by returning `error.NegativeDenominator`
for every negative denominator.

**Expected.** One of: lower `@rem`/`@mod` with a guard (`if (y == -1) 0 else srem`), or a
safety check in safe modes plus wording like `@divTrunc`'s in the reference. The crosscheck
bench in this repository guards the case itself (`docs/crosscheck.md`, "every generated
program is defined").

**Reproducer.** `rem2.zig`:

```zig
const std = @import("std");
pub fn main(init: std.process.Init) !void {
    const arena = init.arena.allocator();
    const args = try init.minimal.args.toSlice(arena);
    const x = try std.fmt.parseInt(i32, args[1], 10);
    const y = try std.fmt.parseInt(i32, args[2], 10);
    std.debug.print("@rem = {d}, @mod = {d}\n", .{ @rem(x, y), @mod(x, y) });
}
```

```sh
zig build-exe -ODebug -target x86_64-macos rem2.zig && ./rem2 -2147483648 -1   # crashes
```

No existing issue was found for this case (searched the tracker for "rem minInt", "srem
overflow", "mod INT_MIN"; #23635 covers the sign of the denominator only).
