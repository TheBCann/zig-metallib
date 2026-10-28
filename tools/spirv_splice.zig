//! spirv-splice: turn a SPIR-V compute module into a Metal library, with no
//! Apple tooling involved (docs/spirv.md).
//!
//!   module.spv --reader--> Module --lower--> IR text + manifest --assembler--> bitcode --packer--> .metallib
//!
//! tools/spirv/reader.zig decodes the binary, tools/spirv/lower.zig prints
//! the IR text in the dialect tools/air/assembler.zig reads (the one
//! air-splice produces) and builds each kernel's interface at runtime
//! (tools/spirv/manifest.zig); the assembler, metadata and packer are the Zig
//! path's own, so every Metal workaround there applies to SPIR-V too.
//!
//! Usage: spirv-splice [--target=<profile>] [--allow-unverified] [--fast-math=<bool>]
//!                     [--push-constant-index=<n>] [--manifest=<out.txt>] [--dis]
//!                     [--zig-workarounds=<bool>] [--allow-metal-miscompiles]
//!                     <in.spv|in.spvasm> <out.metallib> [out.ll]
//!
//! Option rules are air-splice's: options come before the files, an argument
//! starting with `-` among the files is an error (a misplaced flag must not
//! become an output path), and a deployment target that is not verified is
//! refused unless `--allow-unverified`. `--push-constant-index` is the
//! `[[buffer(n)]]` the PushConstant block is passed in (default 30),
//! `--manifest` writes the kernels' interfaces as text (which buffer index
//! each binding, push constant and parameter takes, the builtins, LocalSize),
//! `--dis` prints the module's disassembly to stdout first.
//! `--zig-workarounds=false` lowers a module from Zig's SPIR-V backend by the
//! specification alone, without undoing that backend's known bugs
//! (lower.zig's module doc, "producer workarounds"); it exists to measure
//! them. `--allow-metal-miscompiles` lowers what the lowering refuses
//! because Metal builds it but computes a wrong answer (threadgroup
//! OpAtomicFAddEXT; docs/spirv.md, "Metal findings made while mapping"),
//! so the finding can be reproduced and handed to Apple's assembler; never
//! for real use. An input that
//! does not start with the SPIR-V magic is taken as text and assembled with
//! tools/spirv/asm.zig. The optional `out.ll` is the IR plus metadata
//! (with the assembler's intrinsic renames and `convergent` attributes
//! spelled out, intrinsics.printForApple), which
//! `xcrun metal -Xclang -opaque-pointers out.ll -o apple.metallib` also
//! accepts: Apple's assembler over the same text, the attribution step.
//!
//! This file is also the root of the SPIR-V lowering test module
//! (`zig build test`): lower.zig, manifest.zig and the CLI tests run here.

const std = @import("std");
const Io = std.Io;
const Allocator = std.mem.Allocator;
pub const reader = @import("spirv/reader.zig");
pub const dis = @import("spirv/dis.zig");
pub const asm_ = @import("spirv/asm.zig");
pub const lower = @import("spirv/lower.zig");
pub const manifest = @import("spirv/manifest.zig");
const assembler = @import("air/assembler.zig");
const metadata = @import("air/metadata.zig");
const metallib = @import("air/metallib.zig");
const target = @import("air/target.zig");
const intrinsics = @import("air/intrinsics.zig");

pub const Cli = struct {
    profile: target.Profile = target.default,
    allow_unverified: bool = false,
    fast_math: bool = metadata.default_compile_options.fast_math,
    push_constant_index: u32 = 30,
    manifest_output: ?[]const u8 = null,
    dis: bool = false,
    zig_workarounds: bool = true,
    allow_metal_miscompiles: bool = false,
    input: []const u8 = "",
    output: []const u8 = "",
    text_output: ?[]const u8 = null,
};

pub const CliError = error{ UnknownOption, BadOptionValue, UnknownTarget, OptionAfterFile, WrongArgCount, UnverifiedTarget };

const usage = "usage: spirv-splice [--target=<profile>] [--allow-unverified] [--fast-math=<bool>] [--push-constant-index=<n>] [--manifest=<out.txt>] [--dis] [--zig-workarounds=<bool>] [--allow-metal-miscompiles] <in.spv|in.spvasm> <out.metallib> [out.ll]\n";

/// Parse the arguments after the program name (see the module doc). On
/// error, `bad` names the offending argument or profile.
pub fn parseCli(args: []const []const u8, bad: *[]const u8) CliError!Cli {
    var cli = Cli{};
    var rest = args;
    while (rest.len > 0 and std.mem.startsWith(u8, rest[0], "--")) : (rest = rest[1..]) {
        const arg = rest[0];
        if (std.mem.startsWith(u8, arg, "--target=")) {
            const name = arg["--target=".len..];
            cli.profile = target.fromName(name) orelse {
                bad.* = name;
                return error.UnknownTarget;
            };
        } else if (std.mem.eql(u8, arg, "--allow-unverified")) {
            cli.allow_unverified = true;
        } else if (std.mem.startsWith(u8, arg, "--fast-math=")) {
            const value = arg["--fast-math=".len..];
            if (std.mem.eql(u8, value, "true")) {
                cli.fast_math = true;
            } else if (std.mem.eql(u8, value, "false")) {
                cli.fast_math = false;
            } else {
                bad.* = arg;
                return error.BadOptionValue;
            }
        } else if (std.mem.startsWith(u8, arg, "--push-constant-index=")) {
            const n = std.fmt.parseInt(u32, arg["--push-constant-index=".len..], 10) catch {
                bad.* = arg;
                return error.BadOptionValue;
            };
            if (n > 30) {
                bad.* = arg;
                return error.BadOptionValue;
            }
            cli.push_constant_index = n;
        } else if (std.mem.startsWith(u8, arg, "--manifest=")) {
            const path = arg["--manifest=".len..];
            if (path.len == 0) {
                bad.* = arg;
                return error.BadOptionValue;
            }
            cli.manifest_output = path;
        } else if (std.mem.eql(u8, arg, "--dis")) {
            cli.dis = true;
        } else if (std.mem.eql(u8, arg, "--allow-metal-miscompiles")) {
            cli.allow_metal_miscompiles = true;
        } else if (std.mem.startsWith(u8, arg, "--zig-workarounds=")) {
            const value = arg["--zig-workarounds=".len..];
            if (std.mem.eql(u8, value, "true")) {
                cli.zig_workarounds = true;
            } else if (std.mem.eql(u8, value, "false")) {
                cli.zig_workarounds = false;
            } else {
                bad.* = arg;
                return error.BadOptionValue;
            }
        } else {
            bad.* = arg;
            return error.UnknownOption;
        }
    }
    for (rest) |arg| {
        if (std.mem.startsWith(u8, arg, "-")) {
            bad.* = arg;
            return error.OptionAfterFile;
        }
    }
    if (rest.len != 2 and rest.len != 3) return error.WrongArgCount;
    if (!cli.profile.verified and !cli.allow_unverified) {
        bad.* = @tagName(cli.profile.name);
        return error.UnverifiedTarget;
    }
    cli.input = rest[0];
    cli.output = rest[1];
    if (rest.len == 3) cli.text_output = rest[2];
    return cli;
}

pub fn isBinary(input: []const u8) bool {
    if (input.len < 4) return false;
    const w = std.mem.readInt(u32, input[0..4], .little);
    return w == reader.spec.magic_number or w == @byteSwap(reader.spec.magic_number);
}

pub const Output = struct {
    /// The .metallib bytes.
    image: []const u8,
    /// IR text plus metadata (the optional `out.ll`).
    text: []const u8,
    kernels: []const manifest.Kernel,
};

pub const Options = struct {
    profile: target.Profile = target.default,
    fast_math: bool = false,
    push_constant_index: u32 = 30,
    only_entry: ?[]const u8 = null,
    zig_workarounds: bool = true,
    /// `lower.Options.allow_metal_miscompiles`: the corpus driver's check
    /// that Metal still miscompiles what the lowering refuses. No CLI flag.
    allow_metal_miscompiles: bool = false,
};

/// Lower, assemble and pack every compute entry point of `m`; everything is
/// allocated in the arena `a`. `error.Unsupported` leaves the reason in
/// `diag` (lowering) or on stderr (the assembler's line message).
pub fn splice(a: Allocator, m: *const reader.Module, opts: Options, diag: *lower.Diagnostics) !Output {
    const lowered = try lower.lower(a, m, .{ .profile = opts.profile, .push_constant_index = opts.push_constant_index, .fast_math = opts.fast_math, .only_entry = opts.only_entry, .zig_workarounds = opts.zig_workarounds, .allow_metal_miscompiles = opts.allow_metal_miscompiles }, diag);
    const compile: metadata.CompileOptions = .{ .fast_math = opts.fast_math };
    const fms = try a.alloc(metadata.FunctionMetadata, lowered.kernels.len);
    const functions = try a.alloc(metallib.Function, lowered.kernels.len);
    for (lowered.kernels, fms, functions) |k, *fm, *func| {
        fm.* = try manifest.functionMetadata(a, k);
        const bitcode = assembler.assemble(a, lowered.ir, .{ .entry = k.name, .fm = fm.*, .check_operand_types = true }, opts.profile, compile) catch |err| {
            diag.message = try a.print("assembling {s}: {t} (the assembler's message is above)", .{ k.name, err });
            return err;
        };
        func.* = .{ .name = k.name, .stage = .kernel, .bitcode = bitcode };
    }
    var text: std.ArrayList(u8) = .empty;
    // Spelled for Apple's front end: the assembler's intrinsic renames and
    // convergent attributes (intrinsics.printForApple).
    try text.appendSlice(a, try intrinsics.printForApple(a, lowered.ir, opts.fast_math));
    try metadata.printManifestWith(a, &text, fms, &.{}, opts.profile, compile);
    return .{
        .image = try metallib.pack(a, functions, "default.metallib", opts.profile),
        .text = text.items,
        .kernels = lowered.kernels,
    };
}

pub fn main(init: std.process.Init) !void {
    const arena = init.arena.allocator();
    assembler.failure_prefix = "spirv-splice";
    const io = init.io;
    const args = try init.minimal.args.toSlice(arena);
    var bad: []const u8 = "";
    const rest: []const []const u8 = if (args.len > 0) args[1..] else &.{};
    const cli = parseCli(rest, &bad) catch |err| {
        switch (err) {
            error.UnknownTarget => {
                std.debug.print("spirv-splice: unknown --target '{s}'; known targets:", .{bad});
                for (target.profiles) |p| std.debug.print(" {t}", .{p.name});
                std.debug.print("\n", .{});
            },
            error.UnknownOption => std.debug.print("spirv-splice: unknown option '{s}'\n", .{bad}),
            error.BadOptionValue => std.debug.print("spirv-splice: bad value in '{s}' (--fast-math and --zig-workarounds take true or false, --push-constant-index 0..30, --manifest a path)\n", .{bad}),
            error.OptionAfterFile => std.debug.print("spirv-splice: '{s}' comes after a file argument; options go before <in.spv>\n", .{bad}),
            error.WrongArgCount => std.debug.print(usage, .{}),
            error.UnverifiedTarget => std.debug.print("spirv-splice: refusing --target={s}: {s}. Pass --allow-unverified to build it anyway\n", .{ bad, target.unverifiedReason(target.fromName(bad).?).? }),
        }
        std.process.exit(1);
    };
    if (target.unverifiedReason(cli.profile)) |why| {
        std.debug.print("spirv-splice: warning: building an UNVERIFIED {t} library: {s}\n", .{ cli.profile.name, why });
    }

    const cwd = Io.Dir.cwd();
    const input = try cwd.readFileAlloc(io, cli.input, arena, .unlimited);
    const bytes: []const u8 = if (isBinary(input)) input else blk: {
        var adiag: asm_.Diagnostics = .{};
        break :blk asm_.assemble(arena, input, &adiag) catch |err| {
            if (err == error.Syntax) {
                std.debug.print("spirv-splice: {s}: line {d}: {s}\n", .{ cli.input, adiag.line, adiag.message });
                std.process.exit(1);
            }
            return err;
        };
    };
    var rdiag: reader.Diagnostics = .{};
    var module = reader.read(arena, bytes, &rdiag) catch |err| {
        if (err == error.Malformed) {
            std.debug.print("spirv-splice: {s}: word {d}: {s}\n", .{ cli.input, rdiag.word_offset, rdiag.message });
            std.process.exit(1);
        }
        return err;
    };
    if (cli.dis) {
        var buf: [8192]u8 = undefined;
        var out: Io.File.Writer = .initStreaming(.stdout(), io, &buf);
        try dis.write(&out.interface, arena, &module, .{ .names = true });
        try out.interface.flush();
    }
    var diag: lower.Diagnostics = .{};
    const result = splice(arena, &module, .{ .profile = cli.profile, .fast_math = cli.fast_math, .push_constant_index = cli.push_constant_index, .zig_workarounds = cli.zig_workarounds, .allow_metal_miscompiles = cli.allow_metal_miscompiles }, &diag) catch |err| {
        switch (err) {
            error.Unsupported, error.EntryPointMissing, error.Syntax, error.UndefinedValue, error.UndefinedType, error.UndefinedBlock, error.UndefinedGlobal => {
                std.debug.print("spirv-splice: {s}: {s}\n", .{ cli.input, if (diag.message.len > 0) diag.message else @errorName(err) });
                std.process.exit(1);
            },
            else => return err,
        }
    };
    if (cli.text_output) |path| try cwd.writeFile(io, .{ .sub_path = path, .data = result.text });
    if (cli.manifest_output) |path| {
        var aw: Io.Writer.Allocating = .init(arena);
        try manifest.write(&aw.writer, result.kernels);
        try cwd.writeFile(io, .{ .sub_path = path, .data = aw.written() });
    }
    try cwd.writeFile(io, .{ .sub_path = cli.output, .data = result.image });
}

// ── Tests ───────────────────────────────────────────────────────────────────

test {
    _ = lower;
    _ = manifest;
    _ = @import("spirv/probes.zig");
    _ = @import("spirv/reference.zig");
    _ = @import("spirv/exec.zig");
}

test "parseCli: options before files, values checked, unverified targets refused" {
    var bad: []const u8 = "";
    const ok = try parseCli(&.{ "--fast-math=true", "--push-constant-index=7", "--manifest=m.txt", "--dis", "a.spv", "b.metallib", "c.ll" }, &bad);
    try std.testing.expect(ok.fast_math);
    try std.testing.expectEqual(7, ok.push_constant_index);
    try std.testing.expectEqualStrings("m.txt", ok.manifest_output.?);
    try std.testing.expect(ok.dis);
    try std.testing.expectEqualStrings("a.spv", ok.input);
    try std.testing.expectEqualStrings("b.metallib", ok.output);
    try std.testing.expectEqualStrings("c.ll", ok.text_output.?);
    const plain = try parseCli(&.{ "a.spv", "b.metallib" }, &bad);
    try std.testing.expectEqual(30, plain.push_constant_index);
    try std.testing.expect(!plain.fast_math);
    try std.testing.expect(plain.zig_workarounds);
    try std.testing.expect(!plain.allow_metal_miscompiles);
    try std.testing.expect((try parseCli(&.{ "--allow-metal-miscompiles", "a.spv", "b.metallib" }, &bad)).allow_metal_miscompiles);
    try std.testing.expectEqual(null, plain.text_output);
    try std.testing.expect(!(try parseCli(&.{ "--zig-workarounds=false", "a.spv", "b.metallib" }, &bad)).zig_workarounds);
    try std.testing.expectError(error.BadOptionValue, parseCli(&.{ "--zig-workarounds=no", "a.spv", "b.metallib" }, &bad));

    try std.testing.expectError(error.OptionAfterFile, parseCli(&.{ "a.spv", "--dis", "b.metallib" }, &bad));
    try std.testing.expectEqualStrings("--dis", bad);
    try std.testing.expectError(error.OptionAfterFile, parseCli(&.{ "a.spv", "b.metallib", "-target=macos15" }, &bad));
    try std.testing.expectError(error.UnknownOption, parseCli(&.{ "--frobnicate", "a.spv", "b.metallib" }, &bad));
    try std.testing.expectError(error.BadOptionValue, parseCli(&.{ "--fast-math=yes", "a.spv", "b.metallib" }, &bad));
    try std.testing.expectError(error.BadOptionValue, parseCli(&.{ "--push-constant-index=31", "a.spv", "b.metallib" }, &bad));
    try std.testing.expectError(error.BadOptionValue, parseCli(&.{ "--push-constant-index=x", "a.spv", "b.metallib" }, &bad));
    try std.testing.expectError(error.BadOptionValue, parseCli(&.{ "--manifest=", "a.spv", "b.metallib" }, &bad));
    try std.testing.expectError(error.WrongArgCount, parseCli(&.{"a.spv"}, &bad));
    try std.testing.expectError(error.WrongArgCount, parseCli(&.{ "a", "b", "c", "d" }, &bad));
    try std.testing.expectError(error.UnknownTarget, parseCli(&.{ "--target=macos99", "a.spv", "b.metallib" }, &bad));
    try std.testing.expectEqualStrings("macos99", bad);
    try std.testing.expectError(error.UnverifiedTarget, parseCli(&.{ "--target=macos15", "a.spv", "b.metallib" }, &bad));
    try std.testing.expectEqualStrings("macos15", bad);
    const unverified = try parseCli(&.{ "--target=macos15", "--allow-unverified", "a.spv", "b.metallib" }, &bad);
    try std.testing.expectEqual(target.Name.macos15, unverified.profile.name);
}

test "splice packs one function per compute entry point and prints the metadata" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var m = try reader.read(a, @embedFile("spirv/testdata/desc.spv"), null);
    var diag: lower.Diagnostics = .{};
    const out = try splice(a, &m, .{}, &diag);
    try std.testing.expectEqualStrings("MTLB", out.image[0..4]);
    try std.testing.expectEqual(1, out.kernels.len);
    try std.testing.expectEqualStrings("crosscheckKernel", out.kernels[0].name);
    try std.testing.expect(std.mem.find(u8, out.text, "!air.kernel = !{") != null);
    try std.testing.expect(std.mem.find(u8, out.text, "ptr @crosscheckKernel, ") != null);
    try std.testing.expect(std.mem.find(u8, out.text, "!\"air.compile.fast_math_disable\"") != null);
    // in0, in1, out by binding, params (Uniform, binding 3), then the id.
    const k = out.kernels[0];
    try std.testing.expectEqual(5, k.params.len);
    for (k.params[0..4], 0..) |p, i| try std.testing.expectEqual(@as(u32, @intCast(i)), p.buffer.index);
    try std.testing.expectEqual(manifest.Space.constant, k.params[3].buffer.space);
    try std.testing.expectEqualStrings("out", k.params[2].buffer.name);
    try std.testing.expectEqual(manifest.Access.read_write, k.params[2].buffer.access);
    var stamp = metallib.readStamp(out.image).?;
    _ = &stamp;
}

test "splice reports a lowering refusal with the instruction" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const text =
        \\OpCapability Shader
        \\OpMemoryModel Logical GLSL450
        \\OpEntryPoint GLCompute %k "k"
        \\%void = OpTypeVoid
        \\%fn = OpTypeFunction %void
        \\%u32 = OpTypeInt 32 0
        \\%c1 = OpConstant %u32 1
        \\%c72 = OpConstant %u32 72
        \\%k = OpFunction %void None %fn
        \\%e = OpLabel
        \\OpControlBarrier %c1 %c1 %c72
        \\OpReturn
        \\OpFunctionEnd
    ;
    const bytes = try asm_.assemble(a, text, null);
    var m = try reader.read(a, bytes, null);
    var diag: lower.Diagnostics = .{};
    try std.testing.expectError(error.Unsupported, splice(a, &m, .{}, &diag));
    try std.testing.expectEqualStrings("entry point 'k': `OpControlBarrier %5 %5 %6`: execution scope 1: Metal synchronises only a threadgroup (Workgroup) or a SIMD-group (Subgroup)", diag.message);
}
