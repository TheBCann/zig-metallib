//! spirv-corpus (`zig build spirv-corpus`): the SPIR-V corpus through Metal,
//! and the kernels with known answers executed on the GPU (docs/spirv.md,
//! "The corpus"; SPIRV-DESIGN section 6).
//!
//!   spirv-corpus [--fast-math=<bool>] <corpus dir> <EXPECTED.txt> <zig_kernels.spv>
//!
//! 1. Every `*.spvasm` of the corpus directory is assembled (tools/spirv/
//!    asm.zig), and every entry point of it is lowered and packed on its own
//!    (`spirv_splice.splice` with `only_entry`, as tools/spirv_probe.zig
//!    does), loaded with newLibraryWithData and built into a compute
//!    pipeline. The outcome (`ok`, `refused: <why>`, `Metal: <error>`) is
//!    compared with the `<file>:<entry>` row of EXPECTED.txt, the table
//!    tools/spirv/run_probes.sh also checks (it builds the same libraries in
//!    separate processes; a `Metal:` row matches any Metal failure, the
//!    error texts of the two tools differ). An expected refusal is a pass.
//! 2. `<zig_kernels.spv>` is tools/spirv/zig_kernels.zig compiled by Zig's
//!    self-hosted SPIR-V backend (build.zig); each of its entry points must
//!    build.
//! 3. The kernels with known answers are dispatched with their buffers bound
//!    at the indices of the runtime manifest and every output word is
//!    compared with a CPU computation:
//!      wgsl-collatz:main                  naga's Collatz kernel over 1..4096
//!                                         against `collatz` below;
//!      hand-workgroup-atomics:tg_reverse  a threadgroup reverse (barrier);
//!      hand-workgroup-atomics:atomic_counter, zig_kernels:atomic_counter
//!                                         every old value of a shared
//!                                         counter exactly once;
//!      hand-workgroup-atomics:atomic_minmax64
//!                                         64-bit signed and unsigned atomic
//!                                         min/max extremes;
//!      hand-workgroup-atomics:atomic_fadd float atomic add: the old values
//!                                         are 0, 0.5, ... in some order;
//!      hand-workgroup-atomics:atomic_fxchg_tg
//!                                         threadgroup float store, exchange
//!                                         and load: per workgroup the final
//!                                         value and the old values are 0..64;
//!      hand-workgroup-atomics:sg_reverse  a SIMD-group reverse through
//!                                         threadgroup memory (Subgroup
//!                                         barrier, Workgroup memory scope);
//!      hand-glslang-stage4:reduce         glslang's tree reduction (barrier
//!                                         in a uniform loop) per workgroup;
//!      hand-glslang-stage4:atomics        atomicAdd and atomicCompSwap;
//!      hand-dxc-reflect:main              DXC's -fspv-reflect shape
//!                                         (UserTypeGOOGLE / UserSemantic
//!                                         decorations): in * 3 + 1;
//!      zig_kernels:tg_reverse             Zig's threadgroup reverse;
//!      zig_kernels:control_flow           tools/spirv/zig_kernels_body.zig
//!                                         on the GPU against the same source
//!                                         compiled for the host.
//!
//! 4. The Metal findings: a kernel the lowering refuses because Metal builds
//!    it but computes a wrong answer is built again with
//!    `allow_metal_miscompiles` and executed against its known answer, which
//!    must still DIFFER (`finding reproduces`); a match means a macOS release
//!    fixed the miscompile and the refusal can be lifted:
//!      hand-workgroup-atomics:atomic_fadd_tg
//!                                         threadgroup float add (the final
//!                                         value must be 64; Metal gives 0).
//!
//! Prints one row per kernel (`<file>:<entry>`, the pipeline outcome, the
//! execution outcome or `-`) and a summary; exits 1 on any outcome that
//! differs from EXPECTED.txt, a Zig kernel that does not build, a
//! known-answer kernel that does not match, or a Metal finding that no longer
//! reproduces.

const std = @import("std");
const Io = std.Io;
const Allocator = std.mem.Allocator;
const objc = @import("objc");
const splice = @import("spirv_splice");
const reader = splice.reader;
const asm_ = splice.asm_;
const lower = splice.lower;
const manifest = splice.manifest;
const kernels_body = @import("zig_kernels_body.zig");

extern "c" fn MTLCreateSystemDefaultDevice() ?objc.Object;

const MTLResourceStorageModeShared: u64 = 0;
const MTLCommandBufferStatusError: u64 = 5;
const MTLSize = extern struct { width: u64, height: u64, depth: u64 };

const usage = "usage: spirv-corpus [--fast-math=<bool>] <corpus dir> <EXPECTED.txt> <zig_kernels.spv>\n";

/// The element count of the known-answer kernels' buffers (and of the
/// Zig kernels' `[4096]i32` Blocks).
const n = 4096;

pub fn main(init: std.process.Init) !void {
    const arena = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(arena);
    var rest: []const [:0]const u8 = if (args.len > 0) args[1..] else &.{};
    var fast_math = false;
    if (rest.len > 0 and std.mem.startsWith(u8, rest[0], "--fast-math=")) {
        const v = rest[0]["--fast-math=".len..];
        if (std.mem.eql(u8, v, "true")) fast_math = true else if (!std.mem.eql(u8, v, "false")) {
            std.debug.print(usage, .{});
            std.process.exit(2);
        }
        rest = rest[1..];
    }
    if (rest.len != 3) {
        std.debug.print(usage, .{});
        std.process.exit(2);
    }
    const corpus_dir = rest[0];
    const cwd = Io.Dir.cwd();
    const expected = try parseExpected(arena, try cwd.readFileAlloc(io, rest[1], arena, .unlimited));
    const zig_spv = try cwd.readFileAlloc(io, rest[2], arena, .unlimited);

    const pool = objc.objc_autoreleasePoolPush();
    defer objc.objc_autoreleasePoolPop(pool);
    const device = MTLCreateSystemDefaultDevice() orelse {
        std.debug.print("SKIP no Metal device available\n", .{});
        std.process.exit(77);
    };
    const queue = objc.msgSend(?objc.Object, device, "newCommandQueue", .{}) orelse return error.NoQueue;
    var ctx: Ctx = .{ .gpa = init.gpa, .device = device, .queue = queue, .fast_math = fast_math };

    var buf: [4096]u8 = undefined;
    var out: Io.File.Writer = .initStreaming(.stdout(), io, &buf);
    const w = &out.interface;
    try w.print("{s:<58} {s:<10} {s}\n", .{ "kernel", "pipeline", "execution" });

    // The corpus, in name order.
    var names: std.ArrayList([]const u8) = .empty;
    {
        var dir = try cwd.openDir(io, corpus_dir, .{ .iterate = true });
        defer dir.close(io);
        var it = dir.iterate();
        while (try it.next(io)) |e| {
            if (e.kind == .file and std.mem.endsWith(u8, e.name, ".spvasm")) try names.append(arena, try arena.dupe(u8, e.name));
        }
    }
    std.mem.sort([]const u8, names.items, {}, struct {
        fn lt(_: void, x: []const u8, y: []const u8) bool {
            return std.mem.order(u8, x, y) == .lt;
        }
    }.lt);

    var seen: std.StringHashMapUnmanaged(void) = .empty;
    for (names.items) |name| {
        const stem = name[0 .. name.len - ".spvasm".len];
        const text = try cwd.readFileAlloc(io, try arena.print("{s}/{s}", .{ corpus_dir, name }), arena, .unlimited);
        var adiag: asm_.Diagnostics = .{};
        const bytes = asm_.assemble(arena, text, &adiag) catch |err| {
            try w.print("{s:<58} UNEXPECTED: does not assemble: line {d}: {s} ({t})\n", .{ stem, adiag.line, adiag.message, err });
            ctx.unexpected += 1;
            continue;
        };
        try ctx.module(w, arena, stem, bytes, expected, &seen);
    }
    // Rows of EXPECTED.txt for corpus kernels this run did not see.
    var eit = expected.iterator();
    while (eit.next()) |e| if (!seen.contains(e.key_ptr.*)) {
        try w.print("{s:<58} UNEXPECTED: in EXPECTED.txt but not in the corpus\n", .{e.key_ptr.*});
        ctx.unexpected += 1;
    };
    // Zig's own SPIR-V of tools/spirv/zig_kernels.zig: every entry must build.
    try ctx.module(w, arena, "zig_kernels", zig_spv, null, &seen);

    try w.print("\n{d} kernels: {d} ok, {d} refused as expected; {d} executed with known answers, {d} match; {d} Metal findings reproduce; {d} unexpected\n", .{ ctx.kernels, ctx.ok, ctx.refused, ctx.executed, ctx.matched, ctx.reproduced, ctx.unexpected });
    try w.flush();
    if (ctx.executed != known.len) {
        std.debug.print("spirv-corpus: {d} of the {d} known-answer kernels ran\n", .{ ctx.executed, known.len });
        std.process.exit(1);
    }
    if (ctx.findings_run != findings.len) {
        std.debug.print("spirv-corpus: {d} of the {d} Metal-finding kernels ran\n", .{ ctx.findings_run, findings.len });
        std.process.exit(1);
    }
    if (ctx.unexpected > 0) std.process.exit(1);
}

/// `<file>:<entry>` -> outcome, for the corpus rows of EXPECTED.txt (the
/// probe rows have no `:` in their name).
fn parseExpected(a: Allocator, text: []const u8) !std.StringArrayHashMapUnmanaged([]const u8) {
    var map: std.StringArrayHashMapUnmanaged([]const u8) = .empty;
    var it = std.mem.splitScalar(u8, text, '\n');
    while (it.next()) |line| {
        const tab = std.mem.findScalar(u8, line, '\t') orelse continue;
        const name = line[0..tab];
        if (std.mem.findScalar(u8, name, ':') == null) continue;
        try map.put(a, name, line[tab + 1 ..]);
    }
    return map;
}

/// Whether a pipeline outcome matches EXPECTED.txt's row.
fn sameOutcome(got: []const u8, want: []const u8) bool {
    if (std.mem.startsWith(u8, got, "Metal:") and std.mem.startsWith(u8, want, "Metal:")) return true;
    return std.mem.eql(u8, got, want);
}

const Ctx = struct {
    gpa: Allocator,
    device: objc.Object,
    queue: objc.Object,
    fast_math: bool,
    kernels: usize = 0,
    ok: usize = 0,
    refused: usize = 0,
    executed: usize = 0,
    matched: usize = 0,
    findings_run: usize = 0,
    reproduced: usize = 0,
    unexpected: usize = 0,

    /// Every entry point of one module: lower, build, compare, execute.
    fn module(ctx: *Ctx, w: *Io.Writer, arena: Allocator, stem: []const u8, bytes: []const u8, expected: ?std.StringArrayHashMapUnmanaged([]const u8), seen: *std.StringHashMapUnmanaged(void)) !void {
        var rdiag: reader.Diagnostics = .{};
        var m = reader.read(arena, bytes, &rdiag) catch |err| {
            try w.print("{s:<58} UNEXPECTED: does not read: word {d}: {s} ({t})\n", .{ stem, rdiag.word_offset, rdiag.message, err });
            ctx.unexpected += 1;
            return;
        };
        for (m.entry_points) |ep| {
            var ka = std.heap.ArenaAllocator.init(ctx.gpa);
            defer ka.deinit();
            const a = ka.allocator();
            const inner = objc.objc_autoreleasePoolPush();
            defer objc.objc_autoreleasePoolPop(inner);
            const name = try arena.print("{s}:{s}", .{ stem, ep.name });
            try seen.put(arena, name, {});
            ctx.kernels += 1;
            var lib: ?Library = null;
            const outcome = try ctx.build(a, &m, ep.name, false, &lib);
            defer if (lib) |l| l.release();
            const want: ?[]const u8 = if (expected) |e| e.get(name) else "ok";
            var verdict: []const u8 = "";
            if (want) |wo| {
                if (!sameOutcome(outcome, wo)) {
                    verdict = try a.print("  UNEXPECTED (EXPECTED.txt: {s})", .{wo});
                    ctx.unexpected += 1;
                } else if (std.mem.eql(u8, outcome, "ok")) {
                    ctx.ok += 1;
                } else ctx.refused += 1;
            } else {
                verdict = "  UNEXPECTED (no row in EXPECTED.txt)";
                ctx.unexpected += 1;
            }
            var exec_text: []const u8 = "-";
            for (known) |kn| if (std.mem.eql(u8, kn.name, name)) {
                ctx.executed += 1;
                if (lib) |l| {
                    if (try kn.run(ctx, a, l)) |why| {
                        exec_text = try a.print("MISMATCH: {s}", .{why});
                        ctx.unexpected += 1;
                    } else {
                        exec_text = try a.print("match ({s})", .{kn.what});
                        ctx.matched += 1;
                    }
                } else {
                    exec_text = "NOT RUN: the kernel did not build";
                    ctx.unexpected += 1;
                }
            };
            for (findings) |fd| if (std.mem.eql(u8, fd.name, name)) {
                ctx.findings_run += 1;
                var mlib: ?Library = null;
                const forced = try ctx.build(a, &m, ep.name, true, &mlib);
                defer if (mlib) |l| l.release();
                if (mlib) |l| {
                    if (try fd.run(ctx, a, l)) |why| {
                        exec_text = try a.print("finding reproduces (built with allow_metal_miscompiles): {s}", .{why});
                        ctx.reproduced += 1;
                    } else {
                        exec_text = try a.print("UNEXPECTED: built with allow_metal_miscompiles it now computes {s} right; Metal fixed it, lift the refusal", .{fd.what});
                        ctx.unexpected += 1;
                    }
                } else {
                    exec_text = try a.print("UNEXPECTED: does not build with allow_metal_miscompiles: {s}", .{forced});
                    ctx.unexpected += 1;
                }
            };
            const shown = if (outcome.len > 120) try a.print("{s}...", .{outcome[0..117]}) else outcome;
            try w.print("{s:<58} {s:<10} {s}{s}\n", .{ name, shown, exec_text, verdict });
        }
    }

    /// Lower `entry` alone, pack it, load it and build its pipeline.
    /// Returns the outcome text; `lib` is set when it built.
    /// `allow_miscompiles`: lower what the lowering refuses because Metal
    /// miscompiles it (the finding check).
    fn build(ctx: *Ctx, a: Allocator, m: *const reader.Module, entry: []const u8, allow_miscompiles: bool, lib: *?Library) ![]const u8 {
        var diag: lower.Diagnostics = .{};
        const result = splice.splice(a, m, .{ .fast_math = ctx.fast_math, .only_entry = entry, .allow_metal_miscompiles = allow_miscompiles }, &diag) catch |err| {
            if (err == error.OutOfMemory) return err;
            const why = if (diag.message.len > 0) diag.message else @errorName(err);
            return try a.print("refused: {s}", .{try std.mem.replaceOwned(u8, a, why, "\n", " ")});
        };
        const data = objc.dispatch_data_create(result.image.ptr, result.image.len, null, null) orelse return error.DispatchData;
        defer objc.dispatch_release(data);
        var err: ?objc.Object = null;
        const library = objc.msgSend(?objc.Object, ctx.device, "newLibraryWithData:error:", .{ data, &err }) orelse
            return try a.print("Metal: newLibraryWithData: {s}", .{errorText(err)});
        const name_z = try a.dupeSentinel(u8, entry, 0);
        const function = objc.msgSend(?objc.Object, library, "newFunctionWithName:", .{objc.createNSString(name_z)}) orelse {
            objc.release(library);
            return "Metal: newFunctionWithName: not found";
        };
        const pso = objc.msgSend(?objc.Object, ctx.device, "newComputePipelineStateWithFunction:error:", .{ function, &err }) orelse {
            objc.release(function);
            objc.release(library);
            return try a.print("Metal: newComputePipelineState: {s}", .{errorText(err)});
        };
        lib.* = .{ .library = library, .function = function, .pso = pso, .kernel = result.kernels[0] };
        return "ok";
    }
};

fn errorText(err: ?objc.Object) []const u8 {
    const e = err orelse return "(no NSError)";
    const desc = objc.msgSend(?objc.Object, e, "localizedDescription", .{}) orelse return "(no description)";
    return std.mem.span(objc.msgSend([*:0]const u8, desc, "UTF8String", .{}));
}

/// A built kernel: its pipeline and runtime manifest.
const Library = struct {
    library: objc.Object,
    function: objc.Object,
    pso: objc.Object,
    kernel: manifest.Kernel,

    fn release(l: Library) void {
        objc.release(l.pso);
        objc.release(l.function);
        objc.release(l.library);
    }
};

/// One buffer of a dispatch: bound at `index` when the manifest lists it.
const Binding = struct {
    index: u32,
    /// Initial contents; the contents after the dispatch are written back.
    words: []u32,
};

/// Bind `bindings` at the manifest's indices, dispatch `threads` threads in
/// threadgroups of the manifest's LocalSize, wait, and copy every buffer
/// back into its `words`. Returns an error text, or null.
fn dispatch(ctx: *Ctx, a: Allocator, l: Library, bindings: []const Binding, threads: u32) !?[]const u8 {
    const command_buffer = objc.msgSend(?objc.Object, ctx.queue, "commandBuffer", .{}) orelse return error.NoCommandBuffer;
    const encoder = objc.msgSend(?objc.Object, command_buffer, "computeCommandEncoder", .{}) orelse return error.NoEncoder;
    _ = objc.msgSend(void, encoder, "setComputePipelineState:", .{l.pso});
    const contents = try a.alloc(?[]u32, bindings.len);
    @memset(contents, null);
    var buffers: std.ArrayList(objc.Object) = .empty;
    defer for (buffers.items) |b| objc.release(b);
    for (l.kernel.params) |p| switch (p) {
        .buffer => |mb| {
            const bi = for (bindings, 0..) |b, i| {
                if (b.index == mb.index) break i;
            } else return try a.print("the manifest has buffer {d} ({s}), the test binds nothing there", .{ mb.index, mb.name });
            const b = bindings[bi];
            const len: u64 = @max(4, b.words.len * 4);
            const buf = objc.msgSend(?objc.Object, ctx.device, "newBufferWithLength:options:", .{ len, MTLResourceStorageModeShared }) orelse return error.NoBuffer;
            try buffers.append(a, buf);
            const c: []u32 = @alignCast(std.mem.bytesAsSlice(u32, objc.msgSend([*]u8, buf, "contents", .{})[0..len]));
            @memcpy(c[0..b.words.len], b.words);
            contents[bi] = c;
            _ = objc.msgSend(void, encoder, "setBuffer:offset:atIndex:", .{ buf, @as(u64, 0), @as(u64, b.index) });
        },
        .builtin => {},
    };
    const ls = l.kernel.local_size;
    const grid = MTLSize{ .width = threads, .height = 1, .depth = 1 };
    const group = MTLSize{ .width = ls[0], .height = ls[1], .depth = ls[2] };
    _ = objc.msgSend(void, encoder, "dispatchThreads:threadsPerThreadgroup:", .{ grid, group });
    _ = objc.msgSend(void, encoder, "endEncoding", .{});
    _ = objc.msgSend(void, command_buffer, "commit", .{});
    _ = objc.msgSend(void, command_buffer, "waitUntilCompleted", .{});
    if (objc.msgSend(u64, command_buffer, "status", .{}) == MTLCommandBufferStatusError) return "command buffer error";
    for (bindings, contents) |b, c| if (c) |words| @memcpy(b.words, words[0..b.words.len]);
    return null;
}

// ── The kernels with known answers ──────────────────────────────────────────

const Known = struct {
    name: []const u8,
    /// What was compared, for the table.
    what: []const u8,
    /// Dispatches and compares; returns the first difference, or null.
    run: *const fn (ctx: *Ctx, a: Allocator, l: Library) anyerror!?[]const u8,
};

const known = [_]Known{
    .{ .name = "wgsl-collatz:main", .what = "4096 Collatz counts", .run = runCollatz },
    .{ .name = "hand-workgroup-atomics:tg_reverse", .what = "4096 words", .run = runReverse },
    .{ .name = "hand-workgroup-atomics:atomic_counter", .what = "4095 old values + counter", .run = runHandCounter },
    .{ .name = "hand-workgroup-atomics:atomic_minmax64", .what = "4 64-bit extremes", .run = runMinMax64 },
    .{ .name = "hand-workgroup-atomics:atomic_fadd", .what = "4096 old values + sum", .run = runFAdd },
    .{ .name = "hand-workgroup-atomics:atomic_fxchg_tg", .what = "64 workgroups: final + old values", .run = runFXchgTg },
    .{ .name = "hand-workgroup-atomics:sg_reverse", .what = "4096 words", .run = runSgReverse },
    .{ .name = "hand-glslang-stage4:reduce", .what = "64 workgroup sums", .run = runReduce },
    .{ .name = "hand-glslang-stage4:atomics", .what = "4096 counter + 4096 swap results", .run = runGlslangAtomics },
    .{ .name = "hand-dxc-reflect:main", .what = "4096 words", .run = runDxcReflect },
    .{ .name = "zig_kernels:tg_reverse", .what = "4096 words", .run = runReverse },
    .{ .name = "zig_kernels:atomic_counter", .what = "4096 old values + counter", .run = runZigCounter },
    .{ .name = "zig_kernels:control_flow", .what = "4096 words vs the CPU twin", .run = runControlFlow },
};

/// Kernels the lowering refuses because Metal builds them but computes a
/// wrong answer (docs/spirv.md, "Metal findings made while mapping"); each
/// is built with `allow_metal_miscompiles` and must still mismatch.
const findings = [_]Known{
    .{ .name = "hand-workgroup-atomics:atomic_fadd_tg", .what = "the threadgroup float sum", .run = runFAddTg },
};

/// Deterministic inputs: `len` words from a fixed seed.
fn randomWords(a: Allocator, len: usize, seed: u64) ![]u32 {
    var rng = std.Random.DefaultPrng.init(seed);
    const words = try a.alloc(u32, len);
    for (words) |*x| x.* = rng.random().int(u32);
    return words;
}

fn zeros(a: Allocator, len: usize) ![]u32 {
    const words = try a.alloc(u32, len);
    @memset(words, 0);
    return words;
}

/// naga's `collatz_iterations` (wgpu naga/tests/in/wgsl/collatz.wgsl): the
/// number of steps to reach 1, in wrapping u32 arithmetic.
fn collatz(n_base: u32) u32 {
    var x = n_base;
    var i: u32 = 0;
    while (x > 1) : (i +%= 1) {
        x = if (x % 2 == 0) x / 2 else 3 *% x +% 1;
    }
    return i;
}

fn runCollatz(ctx: *Ctx, a: Allocator, l: Library) !?[]const u8 {
    const v = try a.alloc(u32, n);
    for (v, 0..) |*x, i| x.* = @intCast(i + 1);
    if (try dispatch(ctx, a, l, &.{.{ .index = 0, .words = v }}, n)) |e| return e;
    for (v, 0..) |got, i| {
        const want = collatz(@intCast(i + 1));
        if (got != want) return try a.print("collatz({d}): got {d}, want {d}", .{ i + 1, got, want });
    }
    return null;
}

/// out[gid] = in0[group * 64 + 63 - lid] (hand tg_reverse, zig tg_reverse).
fn runReverse(ctx: *Ctx, a: Allocator, l: Library) !?[]const u8 {
    const in0 = try randomWords(a, n, 1);
    const in1 = try randomWords(a, n, 2);
    const out = try zeros(a, n);
    if (try dispatch(ctx, a, l, &.{ .{ .index = 0, .words = in0 }, .{ .index = 1, .words = in1 }, .{ .index = 2, .words = out } }, n)) |e| return e;
    for (out, 0..) |got, gid| {
        const want = in0[(gid & ~@as(usize, 63)) + 63 - (gid & 63)];
        if (got != want) return try a.print("out[{d}] = 0x{x}, want 0x{x}", .{ gid, got, want });
    }
    return null;
}

/// `words` holds each of 0..len-1 exactly once.
fn permutation(a: Allocator, words: []const u32) !?[]const u8 {
    const hit = try a.alloc(bool, words.len);
    @memset(hit, false);
    for (words, 0..) |x, i| {
        if (x >= words.len) return try a.print("old value {d} at {d} is out of range", .{ x, i });
        if (hit[x]) return try a.print("old value {d} seen twice (second at {d})", .{ x, i });
        hit[x] = true;
    }
    return null;
}

/// hand-workgroup-atomics:atomic_counter: OpAtomicIAdd on out[0] by every
/// thread, the old value to out[gid + 1]; 4095 threads keep it in bounds.
fn runHandCounter(ctx: *Ctx, a: Allocator, l: Library) !?[]const u8 {
    const threads = n - 1;
    const out = try zeros(a, n);
    if (try dispatch(ctx, a, l, &.{.{ .index = 2, .words = out }}, threads)) |e| return e;
    if (out[0] != threads) return try a.print("counter = {d}, want {d}", .{ out[0], threads });
    return permutation(a, out[1..n]);
}

/// zig_kernels:atomic_counter: counter.count += 1 by every thread, the old
/// value to out[gid].
fn runZigCounter(ctx: *Ctx, a: Allocator, l: Library) !?[]const u8 {
    const out = try zeros(a, n);
    const counter = try zeros(a, 1);
    if (try dispatch(ctx, a, l, &.{ .{ .index = 2, .words = out }, .{ .index = 4, .words = counter } }, n)) |e| return e;
    if (counter[0] != n) return try a.print("counter = {d}, want {d}", .{ counter[0], n });
    return permutation(a, out);
}

/// hand-workgroup-atomics:atomic_minmax64: v = (sext(in0) << 32) | in1 per
/// thread into m64 = { SMax, SMin, UMax, UMin }.
fn runMinMax64(ctx: *Ctx, a: Allocator, l: Library) !?[]const u8 {
    const in0 = try randomWords(a, n, 3);
    const in1 = try randomWords(a, n, 4);
    var smax: i64 = std.math.minInt(i64);
    var smin: i64 = std.math.maxInt(i64);
    var umax: u64 = 0;
    var umin: u64 = std.math.maxInt(u64);
    for (in0, in1) |hi, lo| {
        const v: u64 = (@as(u64, @bitCast(@as(i64, @as(i32, @bitCast(hi))))) << 32) | lo;
        const s: i64 = @bitCast(v);
        smax = @max(smax, s);
        smin = @min(smin, s);
        umax = @max(umax, v);
        umin = @min(umin, v);
    }
    const init_vals = [4]u64{ @bitCast(@as(i64, std.math.minInt(i64))), @bitCast(@as(i64, std.math.maxInt(i64))), 0, std.math.maxInt(u64) };
    const m64 = try a.alloc(u32, 8);
    for (init_vals, 0..) |v, i| {
        m64[2 * i] = @truncate(v);
        m64[2 * i + 1] = @truncate(v >> 32);
    }
    if (try dispatch(ctx, a, l, &.{ .{ .index = 0, .words = in0 }, .{ .index = 1, .words = in1 }, .{ .index = 3, .words = m64 } }, n)) |e| return e;
    const want = [4]u64{ @bitCast(smax), @bitCast(smin), umax, umin };
    const what = [4][]const u8{ "SMax", "SMin", "UMax", "UMin" };
    for (want, what, 0..) |wv, wn, i| {
        const got = @as(u64, m64[2 * i]) | (@as(u64, m64[2 * i + 1]) << 32);
        if (got != wv) return try a.print("{s} = 0x{x}, want 0x{x}", .{ wn, got, wv });
    }
    return null;
}

/// hand-workgroup-atomics:atomic_fadd: facc += 0.5 by every thread, the old
/// value's bits to out[gid]; every partial sum is exact.
fn runFAdd(ctx: *Ctx, a: Allocator, l: Library) !?[]const u8 {
    const out = try zeros(a, n);
    const facc = try zeros(a, 1);
    if (try dispatch(ctx, a, l, &.{ .{ .index = 2, .words = out }, .{ .index = 4, .words = facc } }, n)) |e| return e;
    const sum: f32 = @bitCast(facc[0]);
    if (sum != 0.5 * @as(f32, n)) return try a.print("sum = {d}, want {d}", .{ sum, 0.5 * @as(f32, n) });
    const steps = try a.alloc(u32, n);
    for (out, steps, 0..) |bits, *s, i| {
        const x: f32 = @bitCast(bits);
        const twice = x * 2;
        if (twice != @floor(twice) or twice < 0 or twice >= n) return try a.print("old value {d} at {d} is not a multiple of 0.5 below the sum", .{ x, i });
        s.* = @intFromFloat(twice);
    }
    return permutation(a, steps);
}

/// The threadgroup float kernels write out[gid] = old * 128 + final, both
/// small integers held exactly in a float. Per workgroup of 64: every
/// thread saw the same final value, `want_final` if given (the add) else
/// one of 1..64 (the exchange of lid + 1), and the 64 old values are the
/// values 0..64 without the final one (the exchange: 0 then each lid + 1
/// but the last; the add: 0..63).
fn checkTgFloat(a: Allocator, out: []const u32, want_final: ?u32) !?[]const u8 {
    var g: usize = 0;
    while (g < out.len) : (g += 64) {
        const words = out[g..][0..64];
        const final = words[0] % 128;
        if (want_final) |wf| {
            if (final != wf) return try a.print("workgroup {d}: final value {d}, want {d}", .{ g / 64, final, wf });
        } else if (final < 1 or final > 64) return try a.print("workgroup {d}: final value {d}, want one of 1..64", .{ g / 64, final });
        var hit: [65]bool = @splat(false);
        hit[final] = true;
        for (words, 0..) |x, lid| {
            if (x % 128 != final) return try a.print("workgroup {d} thread {d}: final value {d}, thread 0 saw {d}", .{ g / 64, lid, x % 128, final });
            const old = x / 128;
            if (old > 64 or hit[old]) return try a.print("workgroup {d} thread {d}: old value {d} out of range or seen twice", .{ g / 64, lid, old });
            hit[old] = true;
        }
    }
    return null;
}

/// hand-workgroup-atomics:atomic_fxchg_tg: threadgroup float store,
/// exchange, load.
fn runFXchgTg(ctx: *Ctx, a: Allocator, l: Library) !?[]const u8 {
    const out = try zeros(a, n);
    if (try dispatch(ctx, a, l, &.{.{ .index = 2, .words = out }}, n)) |e| return e;
    return checkTgFloat(a, out, null);
}

/// hand-workgroup-atomics:atomic_fadd_tg (a Metal finding): 64 threads add
/// 1.0 to a threadgroup float; the final value must be 64.
fn runFAddTg(ctx: *Ctx, a: Allocator, l: Library) !?[]const u8 {
    const out = try zeros(a, n);
    if (try dispatch(ctx, a, l, &.{.{ .index = 2, .words = out }}, n)) |e| return e;
    return checkTgFloat(a, out, 64);
}

/// hand-workgroup-atomics:sg_reverse: out[gid] = in0 reversed within each
/// SIMD-group, whose width is the pipeline's threadExecutionWidth (Metal
/// numbers a 1-D threadgroup's SIMD-groups by consecutive thread indices).
fn runSgReverse(ctx: *Ctx, a: Allocator, l: Library) !?[]const u8 {
    const width: usize = objc.msgSend(u64, l.pso, "threadExecutionWidth", .{});
    if (width == 0 or 64 % width != 0) return try a.print("threadExecutionWidth {d} does not divide 64", .{width});
    const in0 = try randomWords(a, n, 8);
    const out = try zeros(a, n);
    if (try dispatch(ctx, a, l, &.{ .{ .index = 0, .words = in0 }, .{ .index = 2, .words = out } }, n)) |e| return e;
    for (out, 0..) |got, gid| {
        const want = in0[gid - gid % width + width - 1 - gid % width];
        if (got != want) return try a.print("out[{d}] = 0x{x}, want 0x{x} (SIMD width {d})", .{ gid, got, want, width });
    }
    return null;
}

/// hand-glslang-stage4:reduce: out[g] = the wrapping sum of the 64 inputs
/// of workgroup g.
fn runReduce(ctx: *Ctx, a: Allocator, l: Library) !?[]const u8 {
    const in = try randomWords(a, n, 5);
    const out = try zeros(a, n / 64);
    if (try dispatch(ctx, a, l, &.{ .{ .index = 0, .words = in }, .{ .index = 1, .words = out } }, n)) |e| return e;
    for (out, 0..) |got, g| {
        var want: u32 = 0;
        for (in[g * 64 ..][0..64]) |x| want +%= x;
        if (got != want) return try a.print("sum of workgroup {d} = 0x{x}, want 0x{x}", .{ g, got, want });
    }
    return null;
}

/// hand-dxc-reflect:main (DXC's -fspv-reflect shape, UserTypeGOOGLE on both
/// buffers): output[i] = input[i] * 3 + 1, input at binding 0, output at 1.
fn runDxcReflect(ctx: *Ctx, a: Allocator, l: Library) !?[]const u8 {
    const in = try randomWords(a, n, 6);
    const out = try zeros(a, n);
    if (try dispatch(ctx, a, l, &.{ .{ .index = 0, .words = in }, .{ .index = 1, .words = out } }, n)) |e| return e;
    for (in, out, 0..) |x, got, i| {
        const want = x *% 3 +% 1;
        if (got != want) return try a.print("output[{d}] = 0x{x}, want 0x{x}", .{ i, got, want });
    }
    return null;
}

/// hand-glslang-stage4:atomics: cnt.count ends at n and the old values are a
/// permutation; for each of the 16 slots exactly one thread with gid & 15 ==
/// slot swapped 0 for gid + 1, and every other one saw that value.
fn runGlslangAtomics(ctx: *Ctx, a: Allocator, l: Library) !?[]const u8 {
    const out = try zeros(a, 2 * n);
    const cnt = try zeros(a, 17);
    if (try dispatch(ctx, a, l, &.{ .{ .index = 1, .words = out }, .{ .index = 2, .words = cnt } }, n)) |e| return e;
    if (cnt[0] != n) return try a.print("count = {d}, want {d}", .{ cnt[0], n });
    if (try permutation(a, out[0..n])) |why| return why;
    for (0..16) |slot| {
        const winner = cnt[1 + slot];
        if (winner == 0 or (winner - 1) & 15 != slot or winner > n) return try a.print("slot {d} = {d}: not gid + 1 of a thread of that slot", .{ slot, winner });
        var zero_seen: usize = 0;
        var gid = slot;
        while (gid < n) : (gid += 16) {
            const prev = out[n + gid];
            if (prev == 0) {
                zero_seen += 1;
                if (gid + 1 != winner) return try a.print("thread {d} swapped slot {d} but the slot holds {d}", .{ gid, slot, winner });
            } else if (prev != winner) return try a.print("thread {d} saw {d} in slot {d}, which holds {d}", .{ gid, prev, slot, winner });
        }
        if (zero_seen != 1) return try a.print("{d} threads swapped slot {d}", .{ zero_seen, slot });
    }
    return null;
}

/// zig_kernels:control_flow against zig_kernels_body.zig compiled for the
/// host: out[gid] = body(in0[group * 64 + 63 - lid], in1[gid], scale).
fn runControlFlow(ctx: *Ctx, a: Allocator, l: Library) !?[]const u8 {
    const in0 = try randomWords(a, n, 6);
    const in1 = try randomWords(a, n, 7);
    const out = try zeros(a, n);
    const scale: f32 = 0.75;
    const params = try a.dupe(u32, &.{ n, @bitCast(scale) });
    if (try dispatch(ctx, a, l, &.{ .{ .index = 0, .words = in0 }, .{ .index = 1, .words = in1 }, .{ .index = 2, .words = out }, .{ .index = 3, .words = params } }, n)) |e| return e;
    for (out, 0..) |got, gid| {
        const x: i32 = @bitCast(in0[(gid & ~@as(usize, 63)) + 63 - (gid & 63)]);
        const want: u32 = @bitCast(kernels_body.body(x, @bitCast(in1[gid]), scale));
        if (got != want) return try a.print("out[{d}] = 0x{x}, want 0x{x} (body(0x{x}, 0x{x}))", .{ gid, got, want, @as(u32, @bitCast(x)), in1[gid] });
    }
    return null;
}

// ── Tests (no GPU) ──────────────────────────────────────────────────────────

test "collatz: known iteration counts" {
    try std.testing.expectEqual(0, collatz(1));
    try std.testing.expectEqual(1, collatz(2));
    try std.testing.expectEqual(7, collatz(3));
    try std.testing.expectEqual(111, collatz(27));
}

test "EXPECTED.txt: only the corpus rows, and a Metal row matches any Metal failure" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const map = try parseExpected(arena.allocator(), "glsl_Round_f32\tok\nwgsl-collatz:main\tok\nwgsl-bits:main\trefused: x y\nk:z\tMetal: FAIL kernel z: crash\n");
    try std.testing.expectEqual(3, map.count());
    try std.testing.expectEqualStrings("refused: x y", map.get("wgsl-bits:main").?);
    try std.testing.expect(sameOutcome("Metal: newComputePipelineState: other text", map.get("k:z").?));
    try std.testing.expect(!sameOutcome("ok", map.get("wgsl-bits:main").?));
}

test "the threadgroup float check: the exchange and add answers, and Metal's wrong add" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var out: [128]u32 = undefined;
    // Add: final 64, old values 0..63 in some order.
    for (0..2) |g| for (0..64) |lid| {
        out[g * 64 + lid] = @as(u32, @intCast((lid * 7) % 64)) * 128 + 64;
    };
    try std.testing.expectEqual(null, try checkTgFloat(a, &out, 64));
    // Exchange: final 37; old values 0..64 without 37.
    for (0..2) |g| for (0..64) |lid| {
        const old: u32 = if (lid < 37) @intCast(lid) else @intCast(lid + 1);
        out[g * 64 + lid] = old * 128 + 37;
    };
    try std.testing.expectEqual(null, try checkTgFloat(a, &out, null));
    // What Metal computes for the add: final 0, old values 0..31 twice.
    for (0..2) |g| for (0..64) |lid| {
        out[g * 64 + lid] = @as(u32, @intCast(lid % 32)) * 128;
    };
    try std.testing.expect(try checkTgFloat(a, &out, 64) != null);
}

test "the control-flow body: hand-computed values" {
    // b & 7 == 0: no iteration; s = a; fs < fa is false, so max(a, b).
    try std.testing.expectEqual(@as(i32, 8), kernels_body.body(5, 8, 0.75));
    // One iteration from s = 4 (s & 3 == 0): s = 12; then max(12, 1) = 12.
    try std.testing.expectEqual(@as(i32, 12), kernels_body.body(4, 1, 0.75));
    // s = 7 after the first step breaks the loop: a = 6 (6 & 3 == 2):
    // floor((6 >> 8) * 0.75) = 0, not 7; then 0 & 3 == 0 -> 0 * 3 = 0; ...
    // b = 2 runs two iterations: 0, 0; fs = 0 < fa = 0 is false: max(0, 2).
    try std.testing.expectEqual(@as(i32, 2), kernels_body.body(6, 2, 0.75));
}
