//! Precision survey of the transcendental functions on the GPU
//! (docs/crosscheck-float-policy.md, "Precision survey"). For every kernel of
//! src/engine/survey_harness.zig in its table below (the Zig builtin as the
//! assembler lowers it today, Apple's precise `air.<f>` and Apple's
//! `air.fast_<f>`), and every input domain,
//! it runs the kernel over `--samples` inputs drawn from the seed, computes an
//! f64 reference on the CPU, and reports the error in units of the f32 ulp at
//! the reference: how many results are correctly rounded (within 0.5 ulp),
//! within 1, 2, 4 and 16 ulp, the worst case and where, and whether NaN,
//! infinity and flushed-denormal results agree. It is a measurement, not a
//! pass/fail check: exit 0 after printing, 77 without a Metal device (SKIP), 1
//! when Metal fails, 2 on bad usage.
//!
//! Usage: crosscheck-survey [--seed=<u64>] [--samples=<n>] <survey.metallib>

const std = @import("std");
const objc = @import("objc");
const harness = @import("survey_harness");

extern "c" fn MTLCreateSystemDefaultDevice() ?objc.Object;

const MTLResourceStorageModeShared: u64 = 0;
const MTLCommandBufferStatusError: u64 = 5;

const MTLSize = extern struct { width: u64, height: u64, depth: u64 };
fn size1d(len: u64) MTLSize {
    return .{ .width = len, .height = 1, .depth = 1 };
}

/// Mirrors survey_harness.Params.
const Params = extern struct { count: u32 };

const Domain = struct {
    label: []const u8,
    lo: f64,
    hi: f64,
    /// Draw log-uniformly between lo and hi (both positive).
    log_spaced: bool = false,
    /// Negate half of the draws.
    random_sign: bool = false,
};

const Entry = struct {
    kernel: [:0]const u8,
    what: []const u8,
    ref: *const fn (f64) f64,
    domain: Domain,
};

const d_exp = Domain{ .label = "[-87, 88]", .lo = -87, .hi = 88 };
const d_exp2 = Domain{ .label = "[-126, 127]", .lo = -126, .hi = 127 };
const d_log = Domain{ .label = "[FLT_MIN, FLT_MAX] log", .lo = 1.1754943508222875e-38, .hi = 3.4028234663852886e38, .log_spaced = true };
const d_trig_near = Domain{ .label = "[-2pi, 2pi]", .lo = -6.283185307179586, .hi = 6.283185307179586 };
const d_trig_far = Domain{ .label = "[-1e5, 1e5]", .lo = -1.0e5, .hi = 1.0e5 };
const d_trig_huge = Domain{ .label = "+-[1e5, FLT_MAX] log", .lo = 1.0e5, .hi = 3.4028234663852886e38, .log_spaced = true, .random_sign = true };

fn refExp(x: f64) f64 {
    return @exp(x);
}
fn refExp2(x: f64) f64 {
    return @exp2(x);
}
fn refLog(x: f64) f64 {
    return @log(x);
}
fn refLog2(x: f64) f64 {
    return @log2(x);
}
fn refLog10(x: f64) f64 {
    return @log10(x);
}
fn refSin(x: f64) f64 {
    return @sin(x);
}
fn refCos(x: f64) f64 {
    return @cos(x);
}
fn refTan(x: f64) f64 {
    return @tan(x);
}
fn refSqrt(x: f64) f64 {
    return @sqrt(x);
}

/// The three variants of one function over one domain: Zig's builtin as it
/// compiles today, Apple's precise intrinsic, Apple's fast one.
fn trio(comptime zig_kernel: [:0]const u8, comptime air_kernel: [:0]const u8, comptime fast_kernel: [:0]const u8, comptime zig_what: []const u8, comptime air_what: []const u8, comptime fast_what: []const u8, ref: *const fn (f64) f64, domain: Domain) [3]Entry {
    return .{
        .{ .kernel = zig_kernel, .what = zig_what, .ref = ref, .domain = domain },
        .{ .kernel = air_kernel, .what = air_what, .ref = ref, .domain = domain },
        .{ .kernel = fast_kernel, .what = fast_what, .ref = ref, .domain = domain },
    };
}

const entries = trio("expKernel", "airExpKernel", "fastExpKernel", "@exp (air.exp)", "air.exp", "air.fast_exp", &refExp, d_exp) ++
    trio("exp2Kernel", "airExp2Kernel", "fastExp2Kernel", "@exp2 (llvm.exp2)", "air.exp2", "air.fast_exp2", &refExp2, d_exp2) ++
    trio("logKernel", "airLogKernel", "fastLogKernel", "@log (llvm.log)", "air.log", "air.fast_log", &refLog, d_log) ++
    trio("log2Kernel", "airLog2Kernel", "fastLog2Kernel", "@log2 (llvm.log2)", "air.log2", "air.fast_log2", &refLog2, d_log) ++
    trio("log10Kernel", "airLog10Kernel", "fastLog10Kernel", "@log10 (air.log10)", "air.log10", "air.fast_log10", &refLog10, d_log) ++
    trio("sinKernel", "airSinKernel", "fastSinKernel", "@sin (air.sin)", "air.sin", "air.fast_sin", &refSin, d_trig_near) ++
    trio("sinKernel", "airSinKernel", "fastSinKernel", "@sin (air.sin)", "air.sin", "air.fast_sin", &refSin, d_trig_far) ++
    trio("sinKernel", "airSinKernel", "fastSinKernel", "@sin (air.sin)", "air.sin", "air.fast_sin", &refSin, d_trig_huge) ++
    trio("cosKernel", "airCosKernel", "fastCosKernel", "@cos (air.cos)", "air.cos", "air.fast_cos", &refCos, d_trig_near) ++
    trio("cosKernel", "airCosKernel", "fastCosKernel", "@cos (air.cos)", "air.cos", "air.fast_cos", &refCos, d_trig_far) ++
    trio("cosKernel", "airCosKernel", "fastCosKernel", "@cos (air.cos)", "air.cos", "air.fast_cos", &refCos, d_trig_huge) ++
    trio("tanKernel", "airTanKernel", "fastTanKernel", "@tan (air.tan)", "air.tan", "air.fast_tan", &refTan, d_trig_near) ++
    trio("tanKernel", "airTanKernel", "fastTanKernel", "@tan (air.tan)", "air.tan", "air.fast_tan", &refTan, d_trig_far) ++
    trio("sqrtKernel", "airSqrtKernel", "fastSqrtKernel", "@sqrt (llvm.sqrt)", "air.sqrt", "air.fast_sqrt", &refSqrt, d_log);

/// The manifest's entry-point names, usable at run time (the manifest itself
/// holds types, so it is comptime-only).
const kernel_names: [harness.functions.len][:0]const u8 = blk: {
    var names: [harness.functions.len][:0]const u8 = undefined;
    for (harness.functions, 0..) |f, i| names[i] = f.name;
    break :blk names;
};

// Every kernel named above must exist in the harness manifest.
comptime {
    @setEvalBranchQuota(100_000);
    for (entries) |e| {
        var found = false;
        for (kernel_names) |name| {
            if (std.mem.eql(u8, name, e.kernel)) found = true;
        }
        if (!found) @compileError("survey: kernel not in survey_harness.functions: " ++ e.kernel);
    }
}

fn kernelIndex(name: []const u8) usize {
    for (kernel_names, 0..) |n, i| {
        if (std.mem.eql(u8, n, name)) return i;
    }
    unreachable; // checked at comptime above
}

const Stats = struct {
    total: u32 = 0,
    /// Finite reference and result, compared in ulps.
    finite: u32 = 0,
    exact: u32 = 0,
    le_half: u32 = 0,
    le_1: u32 = 0,
    le_2: u32 = 0,
    le_4: u32 = 0,
    le_16: u32 = 0,
    gt_16: u32 = 0,
    /// NaN or infinite reference: the result must be the same special value.
    special_ok: u32 = 0,
    special_bad: u32 = 0,
    /// Reference below FLT_MIN: the GPU flushes, so any zero or denormal is fine.
    tiny_ok: u32 = 0,
    tiny_bad: u32 = 0,
    max_ulp: f64 = 0,
    max_x: f32 = 0,
    max_got: f32 = 0,
    max_ref: f64 = 0,
};

/// Spacing of f32 values around a normal |v|.
fn ulp32(v: f64) f64 {
    const fr = std.math.frexp(v); // |v| = significand * 2^exponent, significand in [0.5, 1)
    return @exp2(@as(f64, @floatFromInt(fr.exponent - 24)));
}

fn classify(st: *Stats, x: f32, got: f32, ref: f64) void {
    st.total += 1;
    const ref32: f32 = @floatCast(ref);
    if (std.math.isNan(ref) or std.math.isInf(ref32)) {
        const ok = if (std.math.isNan(ref)) std.math.isNan(got) else got == ref32;
        if (ok) st.special_ok += 1 else st.special_bad += 1;
        return;
    }
    if (@abs(ref) < std.math.floatMin(f32)) {
        if (@abs(got) < std.math.floatMin(f32)) st.tiny_ok += 1 else st.tiny_bad += 1;
        return;
    }
    if (std.math.isNan(got) or std.math.isInf(got)) {
        st.special_bad += 1;
        return;
    }
    st.finite += 1;
    const err = @abs(@as(f64, got) - ref) / ulp32(ref);
    if (err == 0) st.exact += 1;
    if (err <= 0.5) {
        st.le_half += 1;
    } else if (err <= 1) {
        st.le_1 += 1;
    } else if (err <= 2) {
        st.le_2 += 1;
    } else if (err <= 4) {
        st.le_4 += 1;
    } else if (err <= 16) {
        st.le_16 += 1;
    } else {
        st.gt_16 += 1;
    }
    if (err > st.max_ulp) {
        st.max_ulp = err;
        st.max_x = x;
        st.max_got = got;
        st.max_ref = ref;
    }
}

/// A few exact values first (those inside the domain, plus NaN and the
/// infinities everywhere), then the domain's draws.
const specials = [_]f32{
    0.0,                    -0.0,                   1.0,               -1.0,               0.5,               2.0,
    1.5707964,              3.1415927,              6.2831855,         88.0,               -88.0,             1.0e-10,
    std.math.floatMin(f32), std.math.floatMax(f32), std.math.inf(f32), -std.math.inf(f32), std.math.nan(f32), 0.1,
};

fn inDomain(d: Domain, s: f32) bool {
    if (std.math.isNan(s) or std.math.isInf(s)) return true;
    const a: f64 = @abs(s);
    if (d.random_sign) return a >= d.lo and a <= d.hi;
    return s >= d.lo and s <= d.hi;
}

fn fill(r: std.Random, buf: []f32, d: Domain) void {
    var next: usize = 0;
    for (specials) |s| {
        if (inDomain(d, s)) {
            buf[next] = s;
            next += 1;
        }
    }
    for (buf[next..]) |*v| {
        const u = r.float(f64);
        var x: f64 = if (d.log_spaced) d.lo * @exp2(u * @log2(d.hi / d.lo)) else d.lo + (d.hi - d.lo) * u;
        if (d.random_sign and r.boolean()) x = -x;
        v.* = @floatCast(x);
    }
}

fn pct(n: u32, total: u32) f64 {
    if (total == 0) return 0;
    return @as(f64, @floatFromInt(n)) * 100.0 / @as(f64, @floatFromInt(total));
}

const SharedBuffer = struct {
    object: objc.Object,
    bytes: []u8,

    fn create(device: objc.Object, len: usize) !SharedBuffer {
        const object = objc.msgSend(?objc.Object, device, "newBufferWithLength:options:", .{ @as(u64, len), MTLResourceStorageModeShared }) orelse return error.NoBuffer;
        const ptr = objc.msgSend([*]u8, object, "contents", .{});
        return .{ .object = object, .bytes = ptr[0..len] };
    }

    fn items(self: SharedBuffer, comptime T: type) []T {
        return @alignCast(std.mem.bytesAsSlice(T, self.bytes));
    }
};

fn errorText(err: ?objc.Object) [*:0]const u8 {
    const e = err orelse return "(no NSError)";
    const desc = objc.msgSend(?objc.Object, e, "description", .{}) orelse return "(no description)";
    return objc.msgSend([*:0]const u8, desc, "UTF8String", .{});
}

pub fn main(init: std.process.Init) !void {
    const arena = init.arena.allocator();
    const io = init.io;

    const args = try init.minimal.args.toSlice(arena);
    var seed: u64 = 1;
    var samples: u32 = 1 << 18;
    var path: ?[]const u8 = null;
    for (args[1..]) |arg| {
        if (std.mem.startsWith(u8, arg, "--seed=")) {
            seed = std.fmt.parseInt(u64, arg["--seed=".len..], 10) catch {
                std.debug.print("crosscheck-survey: bad --seed value\n", .{});
                std.process.exit(2);
            };
        } else if (std.mem.startsWith(u8, arg, "--samples=")) {
            samples = std.fmt.parseInt(u32, arg["--samples=".len..], 10) catch {
                std.debug.print("crosscheck-survey: bad --samples value\n", .{});
                std.process.exit(2);
            };
            if (samples < specials.len) samples = specials.len;
        } else if (std.mem.startsWith(u8, arg, "-")) {
            std.debug.print("crosscheck-survey: unknown option '{s}'\n", .{arg});
            std.process.exit(2);
        } else if (path == null) {
            path = arg;
        } else {
            std.debug.print("crosscheck-survey: unexpected extra argument '{s}'\n", .{arg});
            std.process.exit(2);
        }
    }
    const lib_path = path orelse {
        std.debug.print("usage: crosscheck-survey [--seed=<u64>] [--samples=<n>] <survey.metallib>\n", .{});
        std.process.exit(2);
    };

    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, lib_path, arena, .unlimited);

    const pool = objc.objc_autoreleasePoolPush();
    defer objc.objc_autoreleasePoolPop(pool);

    const device = MTLCreateSystemDefaultDevice() orelse {
        std.debug.print("SKIP no Metal device available; nothing was measured\n", .{});
        std.process.exit(77);
    };
    const device_name = objc.msgSend(?objc.Object, device, "name", .{});
    std.debug.print("info device {s}  seed {d}  samples {d} per row\n", .{ if (device_name) |nm| objc.msgSend([*:0]const u8, nm, "UTF8String", .{}) else "(unnamed)", seed, samples });

    const data = objc.dispatch_data_create(bytes.ptr, bytes.len, null, null) orelse return error.DispatchDataFailed;
    defer objc.dispatch_release(data);
    var err: ?objc.Object = null;
    const library = objc.msgSend(?objc.Object, device, "newLibraryWithData:error:", .{ data, &err }) orelse {
        std.debug.print("FAIL newLibraryWithData: {s}\n", .{errorText(err)});
        std.process.exit(1);
    };

    // One pipeline per kernel, built on first use.
    var pipelines: [harness.functions.len]?objc.Object = @splat(null);

    const in = try SharedBuffer.create(device, @as(usize, samples) * 4);
    const out = try SharedBuffer.create(device, @as(usize, samples) * 4);
    const queue = objc.msgSend(?objc.Object, device, "newCommandQueue", .{}) orelse return error.NoCommandQueue;
    var params = Params{ .count = samples };

    var prng: std.Random.DefaultPrng = .init(seed ^ 0x5851f42d4c957f2d);
    const r = prng.random();

    std.debug.print("{s:<28} {s:<24} {s:>9} {s:>7} {s:>7} {s:>7} {s:>7} {s:>7}  {s}\n", .{ "function", "domain", "max ulp", "<=0.5", "<=1", "<=2", "<=4", ">4", "worst case / specials" });
    for (entries) |e| {
        const idx = kernelIndex(e.kernel);
        if (pipelines[idx] == null) {
            const function = objc.msgSend(?objc.Object, library, "newFunctionWithName:", .{objc.createNSString(e.kernel)}) orelse {
                std.debug.print("FAIL newFunctionWithName: {s} not found\n", .{e.kernel});
                std.process.exit(1);
            };
            pipelines[idx] = objc.msgSend(?objc.Object, device, "newComputePipelineStateWithFunction:error:", .{ function, &err }) orelse {
                std.debug.print("FAIL newComputePipelineState {s}: {s}\n", .{ e.kernel, errorText(err) });
                std.process.exit(1);
            };
        }
        const pso = pipelines[idx].?;

        const inputs = in.items(f32);
        fill(r, inputs, e.domain);
        for (out.items(f32)) |*o| o.* = std.math.nan(f32);

        const command_buffer = objc.msgSend(?objc.Object, queue, "commandBuffer", .{}) orelse return error.NoCommandBuffer;
        const encoder = objc.msgSend(?objc.Object, command_buffer, "computeCommandEncoder", .{}) orelse return error.NoEncoder;
        _ = objc.msgSend(void, encoder, "setComputePipelineState:", .{pso});
        _ = objc.msgSend(void, encoder, "setBuffer:offset:atIndex:", .{ in.object, @as(u64, 0), @as(u64, 0) });
        _ = objc.msgSend(void, encoder, "setBuffer:offset:atIndex:", .{ out.object, @as(u64, 0), @as(u64, 1) });
        _ = objc.msgSend(void, encoder, "setBytes:length:atIndex:", .{ @as(*const anyopaque, @ptrCast(&params)), @as(u64, @sizeOf(Params)), @as(u64, 2) });
        _ = objc.msgSend(void, encoder, "dispatchThreads:threadsPerThreadgroup:", .{ size1d(samples), size1d(64) });
        _ = objc.msgSend(void, encoder, "endEncoding", .{});
        _ = objc.msgSend(void, command_buffer, "commit", .{});
        _ = objc.msgSend(void, command_buffer, "waitUntilCompleted", .{});
        if (objc.msgSend(u64, command_buffer, "status", .{}) == MTLCommandBufferStatusError) {
            std.debug.print("FAIL command buffer ({s}): {s}\n", .{ e.kernel, errorText(objc.msgSend(?objc.Object, command_buffer, "error", .{})) });
            std.process.exit(1);
        }

        var st = Stats{};
        for (inputs, out.items(f32)) |x, got| classify(&st, x, got, e.ref(@as(f64, x)));
        std.debug.print("{s:<28} {s:<24} {d:>9.2} {d:>6.2}% {d:>6.2}% {d:>6.2}% {d:>6.2}% {d:>6.2}%  x={e:.7} got {e:.7} ref {e:.7}; special ok/bad {d}/{d}, tiny ok/bad {d}/{d}\n", .{
            e.what,                     e.domain.label,                      st.max_ulp,
            pct(st.le_half, st.finite), pct(st.le_1, st.finite),             pct(st.le_2, st.finite),
            pct(st.le_4, st.finite),    pct(st.le_16 + st.gt_16, st.finite), st.max_x,
            st.max_got,                 st.max_ref,                          st.special_ok,
            st.special_bad,             st.tiny_ok,                          st.tiny_bad,
        });
    }
}
