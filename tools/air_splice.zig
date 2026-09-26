//! air-splice: turn the LLVM IR Zig emits for the shader module into a
//! Metal library, with no Apple tooling involved.
//!
//!   zig (nvptx64 target) --.ll--> [text rewrite] --> [assembler] --> [packer] --> .metallib
//!
//! Text rewrite (this file):
//!   * target datalayout / triple -> air64
//!   * drops attribute groups, Zig's metadata, `llvm.lifetime.*` (its
//!     one-argument form segfaults Apple's frontend, r11 finding 21) and
//!     every `define` no entry point reaches (`__keep_*` exports, panic
//!     helpers); helpers that are reached (`noinline` functions,
//!     `define private fastcc ...`) stay, with `private`/`fastcc` kept and
//!     the same attribute stripping as entries (DESIGN.md D4); quoted
//!     names (`@"mod.Vec(4).sum"`, Zig's spelling for helpers inside
//!     generic instantiations) are followed and re-emitted verbatim
//!   * strips attributes / instruction flags newer than Apple's LLVM, and
//!     the `ptx_kernel` / `amdgpu_kernel` conventions (AIR entry points are
//!     plain functions)
//!   * turns Zig's `sret` return of a vertex output struct, or of a
//!     fragment output struct (MRT + depth, D2/D4), into AIR's by-value
//!     packed struct
//!   * renames `@module.name` entry points to `@name`; kernels are exported
//!     under their own name (`define ptx_kernel void @name`) and only lose
//!     the calling convention
//!   * rebuilds every entry header from the manifest (`define <ret> @name(<params>)`,
//!     builtins typed from `Builtin.T`, buffers `ptr addrspace(1|2|3)`,
//!     stage_in fields by their Zig type minus `point_size`) and refuses an
//!     entry whose Zig parameter types disagree with it: a manifest `u16`
//!     builtin over a Zig `u32` parameter would otherwise assemble (the
//!     header says i16, the body still widens an i32) and run with thread
//!     ids wrapping at 65536 (review-b/r2/e4b.check.log); a fragment whose
//!     parameters do not line up with the stage_in fields would get the
//!     wrong varyings
//!   * renumbers nvptx address spaces in the text (4 -> 2, `ptr addrspace(5)`
//!     -> `ptr`) so the debug `.ll` matches what the assembler lowers (D1)
//! Assembler (air/assembler.zig): rewritten text -> bitcode via
//!   std.zig.llvm.Builder, one module per entry point, with `!air.*`
//!   metadata derived from the shader manifest (air/metadata.zig).
//! Packer (air/metallib.zig): bitcode modules -> MTLB container.
//!
//! Usage: air-splice [--target=<profile>] [--allow-unverified] <in.ll> <out.metallib> [out.ll]
//! The optional third output is the rewritten IR plus metadata as text, which
//! `xcrun metal -Xclang -opaque-pointers` also accepts; handy for diffing.

const std = @import("std");
const shader = @import("shader");
const air = shader.air;
const assembler = @import("air/assembler.zig");
const metadata = @import("air/metadata.zig");
const metallib = @import("air/metallib.zig");
const intrinsics = @import("air/intrinsics.zig");
const target = @import("air/target.zig");

const Io = std.Io;
const Allocator = std.mem.Allocator;
const List = std.ArrayList(u8);
const comptimePrint = std.fmt.comptimePrint;

/// Prefix used when renaming LLVM's numbered temporaries (`%7` -> `%__7`).
/// Numbered values must stay sequential, which our edits would break.
const rename_prefix = "__";

/// `air-splice [--target=<profile>] [--allow-unverified] <in.ll> <out.metallib> [out.ll]`
const Cli = struct {
    profile: target.Profile = target.default,
    allow_unverified: bool = false,
    input: []const u8 = "",
    output: []const u8 = "",
    text_output: ?[]const u8 = null,
};

const CliError = error{ UnknownOption, UnknownTarget, OptionAfterFile, WrongArgCount, UnverifiedTarget };

/// Parse the arguments after the program name. Options come before the
/// files, and an argument starting with `-` among the files is an error
/// rather than a file name, so a misplaced flag (`--target=...`, or a
/// single-dash `-target=...`) cannot become an output path and overwrite a
/// file. A profile whose libraries fail `zig build check` is
/// refused unless `--allow-unverified`, rather than producing a library that
/// only fails at pipeline creation. On error, `bad` names the offending
/// argument (the profile name for UnknownTarget / UnverifiedTarget).
fn parseCli(args: []const [:0]const u8, bad: *[]const u8) CliError!Cli {
    var cli = Cli{};
    var rest = args;
    while (rest.len > 0 and std.mem.startsWith(u8, rest[0], "--")) : (rest = rest[1..]) {
        if (std.mem.startsWith(u8, rest[0], "--target=")) {
            const name = rest[0]["--target=".len..];
            cli.profile = target.fromName(name) orelse {
                bad.* = name;
                return error.UnknownTarget;
            };
        } else if (std.mem.eql(u8, rest[0], "--allow-unverified")) {
            cli.allow_unverified = true;
        } else {
            bad.* = rest[0];
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

pub fn main(init: std.process.Init) !void {
    const arena = init.arena.allocator();
    const io = init.io;

    const args = try init.minimal.args.toSlice(arena);
    var bad: []const u8 = "";
    const cli = parseCli(if (args.len > 0) args[1..] else args, &bad) catch |err| {
        switch (err) {
            error.UnknownTarget => {
                std.debug.print("air-splice: unknown --target '{s}'; known targets:", .{bad});
                for (target.profiles) |p| std.debug.print(" {t}", .{p.name});
                std.debug.print("\n", .{});
            },
            error.UnknownOption => std.debug.print("air-splice: unknown option '{s}'\n", .{bad}),
            error.OptionAfterFile => std.debug.print("air-splice: '{s}' comes after a file argument; options go before <in.ll>\n", .{bad}),
            error.WrongArgCount => std.debug.print("usage: air-splice [--target=<profile>] [--allow-unverified] <in.ll> <out.metallib> [out.ll]\n", .{}),
            error.UnverifiedTarget => std.debug.print("air-splice: refusing --target={s}: {s}. Pass --allow-unverified (zig build: -Dallow-unverified-target=true) to build it anyway, e.g. to run zig build check on that macOS\n", .{ bad, target.unverifiedReason(target.fromName(bad).?).? }),
        }
        std.process.exit(1);
    };
    const profile = cli.profile;
    if (target.unverifiedReason(profile)) |why| {
        std.debug.print("air-splice: warning: building an UNVERIFIED {t} library: {s}\n", .{ profile.name, why });
    }

    const cwd = Io.Dir.cwd();
    const input = try cwd.readFileAlloc(io, cli.input, arena, .unlimited);
    const rewritten = convertWith(arena, input, .{ .profile = profile }) catch |err| {
        std.debug.print("air-splice: {s}: {t}\n", .{ cli.input, err });
        return err;
    };

    if (cli.text_output) |path| {
        var text: List = .empty;
        try text.appendSlice(arena, rewritten);
        try metadata.printModule(arena, &text, try samplerGlobals(arena, rewritten), profile);
        try cwd.writeFile(io, .{ .sub_path = path, .data = text.items });
    }

    var functions: [metadata.function_metadata.len]metallib.Function = undefined;
    inline for (metadata.function_metadata, 0..) |fm, i| {
        const bitcode = assembler.assemble(arena, rewritten, .{
            .entry = fm.name,
            .fm = fm,
        }, profile) catch |err| {
            std.debug.print("air-splice: assembling {s}: {t}\n", .{ fm.name, err });
            return err;
        };
        functions[i] = .{
            .name = fm.name,
            .stage = switch (fm.stage) {
                .vertex => .vertex,
                .fragment => .fragment,
                .kernel => .kernel,
            },
            .bitcode = bitcode,
        };
    }
    const image = try metallib.pack(arena, &functions, "default.metallib", profile);
    try cwd.writeFile(io, .{ .sub_path = cli.output, .data = image });
}

// ── Comptime: what the manifest tells us about each entry point ─────────────

const FieldInfo = struct {
    llvm: []const u8,
    offset: usize,
};

/// One entry parameter as the manifest describes it.
const AirParam = struct {
    /// LLVM type in AIR form (`i32`, `<3 x i32>`, `ptr addrspace(1)`).
    ty: []const u8,
    /// Manifest argument name (a stage_in field's name), for diagnostics.
    name: []const u8,
};

const FnInfo = struct {
    name: []const u8,
    stage: air.Stage,
    /// LLVM return type in AIR form.
    air_ret: []const u8,
    /// Struct returns (vertex output, fragment MRT/depth output): fields of
    /// the returned struct, for the sret rewrite. Empty for a vector
    /// fragment return and for kernels.
    fields: []const FieldInfo,
    /// Parameters, one per manifest arg (one per stage_in field), in order.
    /// The header is rebuilt from these (D4) after the IR's own types were
    /// checked against them.
    params: []const AirParam,
};

const fn_infos: []const FnInfo = blk: {
    var infos: [shader.functions.len]FnInfo = undefined;
    for (shader.functions, 0..) |f, i| {
        infos[i] = switch (f.stage) {
            .vertex => .{
                .name = f.name,
                .stage = .vertex,
                .air_ret = packedStructType(f.ret),
                .fields = fieldInfos(f.ret),
                .params = airParams(f),
            },
            .fragment => if (@typeInfo(f.ret) == .@"struct") .{
                .name = f.name,
                .stage = .fragment,
                .air_ret = packedStructType(f.ret),
                .fields = fieldInfos(f.ret),
                .params = airParams(f),
            } else .{
                .name = f.name,
                .stage = .fragment,
                .air_ret = llvmType(f.ret),
                .fields = &.{},
                .params = airParams(f),
            },
            .kernel => .{
                .name = f.name,
                .stage = .kernel,
                .air_ret = "void",
                .fields = &.{},
                .params = airParams(f),
            },
        };
    }
    const final = infos;
    break :blk &final;
};

/// AIR parameters of an entry point, from the manifest: builtins by their
/// Zig type (`u32` -> `i32`, `@Vector(3, u32)` -> `<3 x i32>`, `u16` ->
/// `i16`, `bool` -> `i1`, `@Vector(2, f32)` -> `<2 x float>`), buffers
/// `ptr addrspace(1|2|3)`, textures `ptr addrspace(1)`, and one parameter
/// per stage_in field except `point_size` (metadata.isFragmentInput).
fn airParams(comptime f: air.Function) []const AirParam {
    @setEvalBranchQuota(100000);
    var out: []const AirParam = &.{};
    for (f.args) |arg| {
        switch (arg) {
            .vertex_id => |name| out = out ++ &[_]AirParam{.{ .ty = "i32", .name = name }},
            .builtin => |b| out = out ++ &[_]AirParam{.{ .ty = llvmType(b.T), .name = b.name }},
            .buffer => |b| out = out ++ &[_]AirParam{.{ .ty = comptimePrint("ptr addrspace({d})", .{@backingInt(b.space)}), .name = b.name }},
            .texture => |t| out = out ++ &[_]AirParam{.{ .ty = "ptr addrspace(1)", .name = t.name }},
            .sampler => |s| out = out ++ &[_]AirParam{.{ .ty = "ptr addrspace(2)", .name = s.name }},
            .stage_in => |T| {
                if (f.stage != .fragment) @compileError("'" ++ f.name ++ "': only fragment functions take stage_in");
                const st = @typeInfo(T).@"struct";
                for (st.field_names, st.field_types) |name, ft| {
                    if (!metadata.isFragmentInput(name)) continue;
                    out = out ++ &[_]AirParam{.{ .ty = llvmType(ft), .name = name }};
                }
            },
        }
    }
    return out;
}

fn fieldInfos(comptime T: type) []const FieldInfo {
    const s = @typeInfo(T).@"struct";
    var out: [s.field_names.len]FieldInfo = undefined;
    for (s.field_names, s.field_types, 0..) |name, ft, i| {
        out[i] = .{ .llvm = llvmType(ft), .offset = @offsetOf(T, name) };
    }
    const final = out;
    return &final;
}

/// AIR passes stage outputs as a packed literal struct: `<{ <4 x float>, ... }>`.
fn packedStructType(comptime T: type) []const u8 {
    const s = @typeInfo(T).@"struct";
    var out: []const u8 = "<{ ";
    for (s.field_types, 0..) |ft, i| {
        if (i > 0) out = out ++ ", ";
        out = out ++ llvmType(ft);
    }
    return out ++ " }>";
}

fn llvmType(comptime T: type) []const u8 {
    return switch (@typeInfo(T)) {
        .vector => |v| comptimePrint("<{d} x {s}>", .{ v.len, llvmType(v.child) }),
        .float => |f| switch (f.bits) {
            16 => "half",
            32 => "float",
            64 => "double",
            else => @compileError("unsupported float width in shader type " ++ @typeName(T)),
        },
        .int => |i| comptimePrint("i{d}", .{i.bits}),
        .bool => "i1",
        else => @compileError("unsupported shader type " ++ @typeName(T)),
    };
}

// ── Runtime: the IR text rewrite ──────────────────────────────────────────

const Error = error{
    OutOfMemory,
    MalformedDefine,
    MalformedParam,
    VertexReturnNotSret,
    FragmentReturnIsSret,
    FragmentReturnNotSret,
    FragmentReturnTypeMismatch,
    KernelReturnNotVoid,
    ParamCountMismatch,
    ParamTypeMismatch,
    EntryPointMissing,
};

const Rename = struct { from: []const u8, to: []const u8 };

/// A `define` as read from the input: its header and body lines.
const RawDefine = struct { name: []const u8, header: []const u8, body: []const []const u8 };
/// A module-level `@name = ...` line; its initialiser may name functions.
const RawGlobal = struct { name: []const u8, line: []const u8 };

pub const ConvertOptions = struct {
    /// Fail when a manifest entry has no `define` in the input. The build
    /// wants that; tests feed fixed IR that defines only some entries.
    require_all_entries: bool = true,
    /// Deployment target whose triple the text is stamped with.
    profile: target.Profile = target.default,
};

/// Rewrite Zig's IR into AIR-flavoured IR text. Metadata is not included;
/// `metadata.printModule` appends it for the text output.
pub fn convert(gpa: Allocator, raw: []const u8) Error![]u8 {
    return convertWith(gpa, raw, .{});
}

pub fn convertWith(gpa: Allocator, raw: []const u8, options: ConvertOptions) Error![]u8 {
    const src = try renameNumbered(gpa, raw);

    var out: List = .empty;
    var renames: std.ArrayList(Rename) = .empty;
    var seen: [fn_infos.len]bool = @splat(false);
    var defines: std.ArrayList(RawDefine) = .empty;
    var globals: std.ArrayList(RawGlobal) = .empty;

    var lines = std.mem.splitScalar(u8, src, '\n');
    while (lines.next()) |line| {
        if (std.mem.startsWith(u8, line, "target datalayout")) {
            try out.print(gpa, "target datalayout = \"{s}\"\n", .{target.datalayout});
            continue;
        }
        if (std.mem.startsWith(u8, line, "target triple")) {
            try out.print(gpa, "target triple = \"{s}\"\n", .{options.profile.triple});
            continue;
        }
        if (std.mem.startsWith(u8, line, "; Function Attrs") or
            std.mem.startsWith(u8, line, "attributes #") or
            std.mem.startsWith(u8, line, "; ModuleID") or
            std.mem.startsWith(u8, line, "!"))
        {
            continue;
        }
        if (std.mem.startsWith(u8, line, "define ")) {
            // Emitted after the whole module is read: only the defines an
            // entry point reaches are kept.
            var body: std.ArrayList([]const u8) = .empty;
            while (lines.next()) |body_line| {
                if (std.mem.eql(u8, body_line, "}")) break;
                try body.append(gpa, body_line);
            }
            const d = try parseDefine(line);
            try defines.append(gpa, .{ .name = unquote(d.full_name), .header = line, .body = try body.toOwnedSlice(gpa) });
            continue;
        }
        if (std.mem.startsWith(u8, line, "declare ")) {
            if (isLifetimeIntrinsic(line)) continue;
            // A texture intrinsic is printed with Apple's signature, which Zig
            // cannot express (DESIGN.md D5-12c); the call sites follow in
            // sanitizeBodyLine.
            if (declaredName(line)) |dname| if (intrinsics.textureIntrinsic(dname)) |ti| {
                try out.appendSlice(gpa, try intrinsics.textureDeclareText(gpa, ti));
                try out.append(gpa, '\n');
                continue;
            };
            const cleaned = try stripAttrRefs(gpa, try stripParamAttrWords(gpa, try gpa.dupe(u8, line)));
            try out.appendSlice(gpa, cleaned);
            try out.append(gpa, '\n');
            continue;
        }
        if (line.len > 1 and line[0] == '@') {
            try globals.append(gpa, .{ .name = globalName(line), .line = line });
            // The texture intrinsics load a constexpr sampler word from the
            // constant address space, where the assembler relocates every
            // addrspace-0 constant (D1). The text has to say so too, or Apple's
            // assembler rejects the call whose parameter is `ptr addrspace(2)`.
            // Keyed on the same `[2 x i64]` shape the assembler collects.
            if (find(line, "constant [2 x i64]")) |at| {
                if (find(line, " addrspace(") == null) {
                    try out.appendSlice(gpa, line[0..at]);
                    try out.appendSlice(gpa, "addrspace(2) ");
                    try out.appendSlice(gpa, line[at..]);
                    try out.append(gpa, '\n');
                    continue;
                }
            }
        }
        try out.appendSlice(gpa, line);
        try out.append(gpa, '\n');
    }

    const keep = try reachableDefines(gpa, defines.items, globals.items);
    for (defines.items, keep) |d, reached| {
        if (!reached) continue; // `__keep_*` exports, unreferenced helpers
        try emitFunction(gpa, &out, &renames, &seen, d.header, d.body);
    }

    if (options.require_all_entries) for (seen, 0..) |ok, i| {
        if (!ok) {
            std.debug.print("air-splice: entry point '{s}' not found in IR\n", .{fn_infos[i].name});
            return error.EntryPointMissing;
        }
    };

    // `@my_shader.vertexShader` -> `@vertexShader` everywhere.
    var text: []u8 = try out.toOwnedSlice(gpa);
    for (renames.items) |r| text = try replaceToken(gpa, text, r.from, r.to);
    // Zig's nvptx address spaces the assembler renumbers (D1): `.param` is
    // nvptx 4 = Metal constant 2; `.local` is nvptx 5 = Metal thread 0. The
    // assembler ignores textual types of named values anyway; this keeps the
    // debug `.ll` loadable by `xcrun metal` (Metal knows only spaces 0..3).
    text = try replaceAll(gpa, text, "addrspace(4)", "addrspace(2)");
    text = try replaceAll(gpa, text, "ptr addrspace(5)", "ptr");
    return text;
}

/// Which defines an entry point reaches, following every `@name` reference
/// in function bodies and in the initialisers of the globals they name.
/// A helper only a dropped function calls is dropped with it, so it can
/// never break assembly.
fn reachableDefines(gpa: Allocator, defines: []const RawDefine, globals: []const RawGlobal) Error![]bool {
    const keep = try gpa.alloc(bool, defines.len);
    @memset(keep, false);
    var visited = std.StringHashMap(void).init(gpa);
    var work: std.ArrayList([]const u8) = .empty;
    for (defines) |d| if (isEntryName(d.name)) try work.append(gpa, d.name);
    while (work.pop()) |name| {
        if (visited.contains(name)) continue;
        try visited.put(name, {});
        for (defines, 0..) |d, i| {
            if (!std.mem.eql(u8, d.name, name)) continue;
            keep[i] = true;
            for (d.body) |line| try collectRefs(gpa, &work, line);
        }
        for (globals) |g| {
            if (std.mem.eql(u8, g.name, name)) try collectRefs(gpa, &work, g.line);
        }
    }
    return keep;
}

fn isEntryName(name: []const u8) bool {
    inline for (fn_infos) |info| if (matchesEntry(name, info.name)) return true;
    return false;
}

/// Every `@ident` and `@"quoted"` in `text` (quotes dropped).
fn collectRefs(gpa: Allocator, work: *std.ArrayList([]const u8), text: []const u8) Error!void {
    var i: usize = 0;
    while (i < text.len) : (i += 1) {
        if (text[i] != '@') continue;
        var j = i + 1;
        if (j < text.len and text[j] == '"') {
            const close = quotedEnd(text, j) orelse break;
            try work.append(gpa, text[j + 1 .. close]);
            i = close;
            continue;
        }
        while (j < text.len and isIdentChar(text[j])) j += 1;
        if (j > i + 1) try work.append(gpa, text[i + 1 .. j]);
        i = j - 1;
    }
}

/// `@name = ...` -> `name`; `@"quoted name" = ...` -> `quoted name`.
fn globalName(line: []const u8) []const u8 {
    if (line.len > 1 and line[1] == '"') {
        const close = quotedEnd(line, 1) orelse return line[2..];
        return line[2..close];
    }
    var j: usize = 1;
    while (j < line.len and isIdentChar(line[j])) j += 1;
    return line[1..j];
}

/// Index of the `"` closing the quoted identifier that opens at `open`,
/// honouring `\XX` escapes; null when unterminated.
fn quotedEnd(text: []const u8, open: usize) ?usize {
    var j = open + 1;
    while (j < text.len and text[j] != '"') : (j += 1) {
        if (text[j] == '\\') j += 2;
    }
    return if (j < text.len) j else null;
}

/// `"mod.Vec(4).sum"` -> `mod.Vec(4).sum`; other names unchanged. The
/// assembler decodes `\XX` escapes itself, so both sides compare the same
/// spelling.
fn unquote(name: []const u8) []const u8 {
    if (name.len >= 2 and name[0] == '"' and name[name.len - 1] == '"') return name[1 .. name.len - 1];
    return name;
}

/// `llvm.lifetime.start/end` calls and declarations: no AIR meaning, and
/// the one-argument form crashes Apple's frontend.
fn isLifetimeIntrinsic(line: []const u8) bool {
    return find(line, "@llvm.lifetime.") != null;
}

const Param = struct {
    ty: []const u8,
    name: []const u8,
    sret_ty: ?[]const u8,
};

const Define = struct {
    ret: []const u8,
    full_name: []const u8,
    params: []const u8,
    /// `private` / `internal`, kept on helpers (entries are exported).
    linkage: []const u8,
    /// `fastcc`, kept on helpers so calls and definitions agree.
    fastcc: bool,
};

fn emitFunction(
    gpa: Allocator,
    out: *List,
    renames: *std.ArrayList(Rename),
    seen: *[fn_infos.len]bool,
    header: []const u8,
    body: []const []const u8,
) Error!void {
    const d = try parseDefine(header);
    inline for (fn_infos, 0..) |info, i| {
        if (matchesEntry(unquote(d.full_name), info.name)) {
            seen[i] = true;
            try renames.append(gpa, .{
                .from = try std.fmt.allocPrint(gpa, "@{s}", .{d.full_name}),
                .to = try std.fmt.allocPrint(gpa, "@{s}", .{info.name}),
            });
            return emitShader(gpa, out, info, d, body);
        }
    }
    return emitHelper(gpa, out, d, body);
}

/// `@my_shader.vertexShader` or `@vertexShader` matches entry `vertexShader`.
fn matchesEntry(full: []const u8, name: []const u8) bool {
    if (std.mem.eql(u8, full, name)) return true;
    if (full.len <= name.len) return false;
    return full[full.len - name.len - 1] == '.' and std.mem.endsWith(u8, full, name);
}

/// A non-entry function an entry point reaches: header rebuilt without
/// attributes (`private` and `fastcc` kept, `sret` becomes a plain `ptr`
/// parameter), body sanitised exactly like an entry's.
fn emitHelper(gpa: Allocator, out: *List, d: Define, body: []const []const u8) Error!void {
    const params = try parseParams(gpa, d.params);
    try out.appendSlice(gpa, "define ");
    if (d.linkage.len > 0) try out.print(gpa, "{s} ", .{d.linkage});
    if (d.fastcc) try out.appendSlice(gpa, "fastcc ");
    try out.print(gpa, "{s} @{s}(", .{ d.ret, d.full_name });
    for (params, 0..) |p, i| {
        if (i > 0) try out.appendSlice(gpa, ", ");
        try out.print(gpa, "{s} {s}", .{ p.ty, p.name });
    }
    try out.appendSlice(gpa, ") {\n");
    if (try implicitEntryLabel(gpa, params, body)) |label| try out.print(gpa, "{s}:\n", .{label});
    for (body) |raw_line| {
        if (isLifetimeIntrinsic(raw_line)) continue;
        try out.appendSlice(gpa, try sanitizeBodyLine(gpa, raw_line));
        try out.append(gpa, '\n');
    }
    try out.appendSlice(gpa, "}\n\n");
}

/// LLVM numbers an unnamed entry block right after the parameters and lets
/// phis name it without a label line (`[ %.06, %5 ]` in a function with
/// `%0`..`%4`, the guarded loop of reduceKernel). After `renameNumbered`
/// that reference reads `%__5`, which the assembler resolves by the same
/// numbering rule, but LLVM's own parser (`xcrun metal`, the cross-check)
/// reports "use of undefined value". Returns the label to write in front
/// of the body when it has no label of its own and refers to the entry
/// block by that number; null otherwise (the output is unchanged for every
/// function that never names its entry block).
fn implicitEntryLabel(gpa: Allocator, params: []const Param, body: []const []const u8) Error!?[]u8 {
    for (body) |raw| {
        const line = std.mem.trim(u8, raw, " \t\r");
        if (line.len == 0 or line[0] == ';') continue;
        var i: usize = 0;
        while (i < line.len and (isIdentChar(line[i]) or line[i] == '-')) i += 1;
        if (i > 0 and i < line.len and line[i] == ':') return null;
        break;
    }
    var next: usize = 0;
    for (params) |p| {
        if (!std.mem.startsWith(u8, p.name, "%" ++ rename_prefix)) continue;
        const n = std.fmt.parseInt(usize, p.name["%".len + rename_prefix.len ..], 10) catch continue;
        if (n + 1 > next) next = n + 1;
    }
    const label = try std.fmt.allocPrint(gpa, "{s}{d}", .{ rename_prefix, next });
    const ref = try std.fmt.allocPrint(gpa, "%{s}", .{label});
    for (body) |raw| if (hasToken(raw, ref)) return label;
    return null;
}

/// Whether `s` contains `tok` as a whole token.
fn hasToken(s: []const u8, tok: []const u8) bool {
    var i: usize = 0;
    while (findPos(s, i, tok)) |p| {
        const before_ok = p == 0 or !isIdentChar(s[p - 1]);
        const after_ok = p + tok.len >= s.len or !isIdentChar(s[p + tok.len]);
        if (before_ok and after_ok) return true;
        i = p + 1;
    }
    return false;
}

fn emitShader(
    gpa: Allocator,
    out: *List,
    comptime info: FnInfo,
    d: Define,
    body: []const []const u8,
) Error!void {
    var params = try parseParams(gpa, d.params);
    const all_params = params;
    var sret: ?Param = null;
    if (params.len > 0 and params[0].sret_ty != null) {
        sret = params[0];
        params = params[1..];
    }
    switch (info.stage) {
        .vertex => if (sret == null) {
            std.debug.print("air-splice: {s}: expected Zig to return the vertex output via sret\n", .{info.name});
            return error.VertexReturnNotSret;
        },
        .fragment => if (info.fields.len > 0) {
            // A struct return (MRT / depth, D2): Zig hands it over via sret
            // exactly like a vertex output (research/_results/r01.txt
            // ir_patterns 10, verify-render-bindings/v4_frag.ll).
            if (sret == null) {
                std.debug.print("air-splice: {s}: expected Zig to return the fragment output struct via sret\n", .{info.name});
                return error.FragmentReturnNotSret;
            }
        } else {
            if (sret != null) return error.FragmentReturnIsSret;
            if (!std.mem.eql(u8, d.ret, info.air_ret)) {
                std.debug.print("air-splice: {s}: IR returns '{s}', manifest expects '{s}'\n", .{ info.name, d.ret, info.air_ret });
                return error.FragmentReturnTypeMismatch;
            }
        },
        .kernel => {
            // `define ptx_kernel void @name(...)`: no sret, nothing returned.
            if (sret != null or !std.mem.eql(u8, d.ret, "void")) {
                std.debug.print("air-splice: {s}: kernel IR returns '{s}', expected void\n", .{ info.name, d.ret });
                return error.KernelReturnNotVoid;
            }
        },
    }
    if (params.len != info.params.len) {
        std.debug.print("air-splice: {s}: IR has {d} parameters, manifest lists {d} (stage_in fields except point_size count one each)\n", .{ info.name, params.len, info.params.len });
        return error.ParamCountMismatch;
    }
    // The manifest only re-spells what Zig declared; a disagreement
    // (manifest `u16`, Zig `u32`) would still assemble, because the
    // assembler types operands by their definition, and then run with
    // wrapped thread ids; a fragment whose parameters do not follow the
    // stage_in fields would read the wrong varyings. Refuse it here,
    // where both are known.
    for (params, info.params, 0..) |p, want, i| {
        const got = try airParamType(gpa, p.ty);
        if (!std.mem.eql(u8, got, want.ty) and !(std.mem.eql(u8, got, "ptr") and std.mem.eql(u8, want.ty, "ptr addrspace(2)"))) {
            std.debug.print("air-splice: {s}: parameter {d} ('{s}') is {s} in the IR but the manifest says {s}\n", .{ info.name, i, want.name, got, want.ty });
            return error.ParamTypeMismatch;
        }
    }

    // Header: parameter types from the manifest (equal to the IR's after
    // renumbering, checked above).
    try out.print(gpa, "define {s} @{s}(", .{ info.air_ret, info.name });
    for (params, info.params, 0..) |p, want, i| {
        if (i > 0) try out.appendSlice(gpa, ", ");
        try out.print(gpa, "{s} {s}", .{ want.ty, p.name });
    }
    try out.appendSlice(gpa, ") {\n");

    if (try implicitEntryLabel(gpa, all_params, body)) |label| try out.print(gpa, "{s}:\n", .{label});
    if (sret) |s| try out.print(gpa, "  %sret = alloca {s}, align 16\n", .{s.sret_ty.?});

    var ret_count: usize = 0;
    for (body) |raw_line| {
        if (isLifetimeIntrinsic(raw_line)) continue;
        var line = try sanitizeBodyLine(gpa, raw_line);
        if (sret) |s| {
            line = try replaceToken(gpa, line, s.name, "%sret");
            if (std.mem.eql(u8, std.mem.trim(u8, line, " \t"), "ret void")) {
                try emitSretEpilogue(gpa, out, info, ret_count);
                ret_count += 1;
                continue;
            }
        }
        try out.appendSlice(gpa, line);
        try out.append(gpa, '\n');
    }
    try out.appendSlice(gpa, "}\n\n");
}

/// An IR parameter type in AIR spelling: the same renumbering `convertWith`
/// applies to the whole text (D1: nvptx `.param` 4 -> constant 2, `.local`
/// 5 -> thread 0), so a manifest type compares equal to what Zig declared.
fn airParamType(gpa: Allocator, ty: []const u8) Error![]u8 {
    const t = try replaceAll(gpa, ty, "addrspace(4)", "addrspace(2)");
    return replaceAll(gpa, t, "ptr addrspace(5)", "ptr");
}

/// Replace `ret void` with loads of every field out of the sret slot,
/// assembled into AIR's by-value packed struct, in declaration order (Zig
/// may lay the fields out in another order; `@offsetOf` follows it). Used
/// for vertex outputs and for fragment output structs alike; a discard
/// branch before the stores (verify-render-bindings/v4_frag.ll) does not
/// matter because only the `ret void` line is rewritten.
fn emitSretEpilogue(gpa: Allocator, out: *List, comptime info: FnInfo, k: usize) Error!void {
    inline for (info.fields, 0..) |f, i| {
        if (f.offset == 0) {
            try out.print(gpa, "  %r{d}.f{d} = load {s}, ptr %sret, align 16\n", .{ k, i, f.llvm });
        } else {
            try out.print(gpa, "  %r{d}.p{d} = getelementptr inbounds i8, ptr %sret, i64 {d}\n", .{ k, i, f.offset });
            try out.print(gpa, "  %r{d}.f{d} = load {s}, ptr %r{d}.p{d}, align {d}\n", .{ k, i, f.llvm, k, i, alignOfOffset(f.offset) });
        }
    }
    inline for (info.fields, 0..) |f, i| {
        if (i == 0) {
            try out.print(gpa, "  %r{d}.a0 = insertvalue {s} undef, {s} %r{d}.f0, 0\n", .{ k, info.air_ret, f.llvm, k });
        } else {
            try out.print(gpa, "  %r{d}.a{d} = insertvalue {s} %r{d}.a{d}, {s} %r{d}.f{d}, {d}\n", .{ k, i, info.air_ret, k, i - 1, f.llvm, k, i, i });
        }
    }
    try out.print(gpa, "  ret {s} %r{d}.a{d}\n", .{ info.air_ret, k, info.fields.len -| 1 });
}

/// Largest power of two (up to 16) that divides `offset`; the alloca is 16-aligned.
fn alignOfOffset(offset: usize) usize {
    if (offset % 16 == 0) return 16;
    if (offset % 8 == 0) return 8;
    if (offset % 4 == 0) return 4;
    return 1;
}

// `define private fastcc void @m.f(ptr sret(%T) %__0, i32 %__1) unnamed_addr #1 {`
fn parseDefine(line: []const u8) Error!Define {
    const at = find(line, " @") orelse return error.MalformedDefine;
    const name_start = at + 2;
    // A quoted name may itself contain `(`: `@"mod.Vec(4).sum"(...)`.
    const name_end = if (name_start < line.len and line[name_start] == '"')
        (quotedEnd(line, name_start) orelse return error.MalformedDefine) + 1
    else
        name_start;
    const paren = findPos(line, name_end, "(") orelse return error.MalformedDefine;
    const full_name = line[name_start..paren];

    const brace = findLastScalar(line, '{') orelse return error.MalformedDefine;
    var close = brace;
    while (close > paren and line[close] != ')') close -= 1;
    if (close == paren) return error.MalformedDefine;

    const prefix = line["define ".len..at];
    return .{
        .ret = stripKeywords(prefix),
        .full_name = full_name,
        .params = line[paren + 1 .. close],
        .linkage = if (hasWord(prefix, "private")) "private" else if (hasWord(prefix, "internal")) "internal" else "",
        .fastcc = hasWord(prefix, "fastcc"),
    };
}

fn hasWord(s: []const u8, word: []const u8) bool {
    var it = std.mem.tokenizeScalar(u8, s, ' ');
    while (it.next()) |tok| if (std.mem.eql(u8, tok, word)) return true;
    return false;
}

/// Drop linkage / calling-convention / attribute words, keep the type.
/// LLVM >= 19 also puts `range(i32 lo, hi)` in front of integer return
/// types; the assembler ignores it, but the splice compares the return type
/// against the manifest, so it is stripped here too. `ptx_kernel` /
/// `amdgpu_kernel` go the same way: AIR entry points are plain functions.
fn stripKeywords(prefix: []const u8) []const u8 {
    const keywords = [_][]const u8{
        "private",              "internal",      "external",        "linkonce",           "weak",
        "common",               "appending",     "extern_weak",     "linkonce_odr",       "weak_odr",
        "available_externally", "dso_local",     "dso_preemptable", "hidden",             "protected",
        "default",              "fastcc",        "coldcc",          "ccc",                "tailcc",
        "swiftcc",              "noundef",       "nonnull",         "noalias",            "zeroext",
        "signext",              "inreg",         "unnamed_addr",    "local_unnamed_addr", "ptx_kernel",
        "ptx_device",           "amdgpu_kernel", "dead_on_return",  "returned",
    };
    var rest = std.mem.trim(u8, prefix, " ");
    outer: while (true) {
        if (std.mem.startsWith(u8, rest, "range(") or std.mem.startsWith(u8, rest, "dereferenceable(")) {
            const close = std.mem.findScalar(u8, rest, ')') orelse return rest;
            rest = std.mem.trim(u8, rest[close + 1 ..], " ");
            continue :outer;
        }
        for (keywords) |kw| {
            if (std.mem.startsWith(u8, rest, kw) and (rest.len == kw.len or rest[kw.len] == ' ')) {
                rest = std.mem.trim(u8, rest[kw.len..], " ");
                continue :outer;
            }
        }
        return rest;
    }
}

fn parseParams(gpa: Allocator, raw: []const u8) Error![]Param {
    var params: std.ArrayList(Param) = .empty;
    var depth: usize = 0;
    var start: usize = 0;
    for (raw, 0..) |c, i| {
        switch (c) {
            '<', '(', '[', '{' => depth += 1,
            '>', ')', ']', '}' => depth -= 1,
            ',' => if (depth == 0) {
                try params.append(gpa, try parseParam(raw[start..i]));
                start = i + 1;
            },
            else => {},
        }
    }
    const last = std.mem.trim(u8, raw[start..], " ");
    if (last.len > 0) try params.append(gpa, try parseParam(last));
    return params.toOwnedSlice(gpa);
}

fn parseParam(raw: []const u8) Error!Param {
    const p = std.mem.trim(u8, raw, " ");
    if (p.len == 0) return error.MalformedParam;

    var ty: []const u8 = undefined;
    switch (p[0]) {
        '<', '[', '{' => {
            var depth: usize = 0;
            var end: usize = 0;
            for (p, 0..) |c, i| {
                switch (c) {
                    '<', '[', '{' => depth += 1,
                    '>', ']', '}' => {
                        depth -= 1;
                        if (depth == 0) {
                            end = i + 1;
                            break;
                        }
                    },
                    else => {},
                }
            }
            if (end == 0) return error.MalformedParam;
            ty = p[0..end];
        },
        else => {
            const sp = std.mem.findScalar(u8, p, ' ') orelse return error.MalformedParam;
            ty = p[0..sp];
            if (std.mem.eql(u8, ty, "ptr")) {
                const rest = std.mem.trim(u8, p[sp..], " ");
                if (std.mem.startsWith(u8, rest, "addrspace(")) {
                    const close = std.mem.findScalar(u8, rest, ')') orelse return error.MalformedParam;
                    ty = p[0 .. sp + 1 + close + 1];
                }
            }
        },
    }

    var name: []const u8 = "";
    var it = std.mem.tokenizeScalar(u8, p, ' ');
    while (it.next()) |tok| name = tok;
    if (name.len == 0 or name[0] != '%') return error.MalformedParam;

    var sret_ty: ?[]const u8 = null;
    if (find(p, "sret(")) |s| {
        const start = s + "sret(".len;
        var depth: usize = 1;
        var i = start;
        while (i < p.len) : (i += 1) {
            switch (p[i]) {
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if (depth == 0) break;
                },
                else => {},
            }
        }
        if (i >= p.len) return error.MalformedParam;
        sret_ty = p[start..i];
    }

    return .{ .ty = ty, .name = name, .sret_ty = sret_ty };
}

/// Remove syntax that Apple's LLVM 15-era parser doesn't know.
/// The constexpr sampler globals of the rewritten text, which
/// `!air.sampler_states` must list: a module that samples through one and does
/// not declare it crashes Metal's backend at pipeline creation. The assembler
/// collects the same `[2 x i64]` constants for the bitcode.
fn samplerGlobals(gpa: Allocator, ir: []const u8) Error![]const []const u8 {
    var names: std.ArrayList([]const u8) = .empty;
    var it = std.mem.splitScalar(u8, ir, '\n');
    while (it.next()) |line| {
        if (line.len > 1 and line[0] == '@' and find(line, "constant [2 x i64]") != null) {
            try names.append(gpa, globalName(line));
        }
    }
    return names.toOwnedSlice(gpa);
}

/// The callee of a `declare` line, without the `@`.
fn declaredName(line: []const u8) ?[]const u8 {
    const at = find(line, " @") orelse return null;
    var j = at + 2;
    while (j < line.len and line[j] != '(') j += 1;
    if (j == line.len) return null;
    return line[at + 2 .. j];
}

const Span = struct { start: usize, end: usize };

/// The type that ends just before `at`, skipping the whitespace between them:
/// `<4 x float>`, `{ <4 x float>, i8 }`, `ptr`, `void`.
fn typeSpanBefore(s: []const u8, at: usize) ?Span {
    var end = at;
    while (end > 0 and (s[end - 1] == ' ' or s[end - 1] == '\t')) end -= 1;
    if (end == 0) return null;
    const last = s[end - 1];
    if (last == '>' or last == '}') {
        const open: u8 = if (last == '>') '<' else '{';
        var depth: usize = 0;
        var i = end;
        while (i > 0) {
            i -= 1;
            if (s[i] == last) depth += 1;
            if (s[i] == open) {
                depth -= 1;
                if (depth == 0) return .{ .start = i, .end = end };
            }
        }
        return null;
    }
    var start = end;
    while (start > 0 and s[start - 1] != ' ' and s[start - 1] != '\t') start -= 1;
    return .{ .start = start, .end = end };
}

/// Splits an argument list on the commas that are not inside `<>`, `[]`, `{}`
/// or `()`.
const ArgIterator = struct {
    s: []const u8,
    i: usize = 0,

    fn next(it: *ArgIterator) ?[]const u8 {
        if (it.i >= it.s.len) return null;
        const start = it.i;
        var depth: usize = 0;
        while (it.i < it.s.len) : (it.i += 1) {
            switch (it.s[it.i]) {
                '<', '[', '{', '(' => depth += 1,
                '>', ']', '}', ')' => depth -|= 1,
                ',' => if (depth == 0) {
                    const arg = it.s[start..it.i];
                    it.i += 1;
                    return arg;
                },
                else => {},
            }
        }
        return it.s[start..];
    }
};

/// Apple's texture intrinsics return `{ <colour>, i8 }` and take their sampler
/// in the constant address space; Zig can spell neither, so the text prints
/// Apple's `declare` (`intrinsics.textureDeclareText`) and the call sites are
/// brought in line here: the sampler operand is retyped, and a call that binds
/// the colour becomes a call binding the pair plus an `extractvalue`. The
/// assembler applies the same rule to the bitcode, so the printed IR and the
/// .metallib agree and Apple's own assembler still accepts the text
/// (DESIGN.md D5-12c). Anything shaped unexpectedly is left alone; the
/// assembler reports it with the offending line.
fn rewriteTextureCall(gpa: Allocator, line: []u8) Error![]u8 {
    const at = find(line, "@air.") orelse return line;
    var name_end = at + 1;
    while (name_end < line.len and line[name_end] != '(') name_end += 1;
    if (name_end == line.len) return line;
    const ti = intrinsics.textureIntrinsic(line[at + 1 .. name_end]) orelse return line;
    const args_end = std.mem.lastIndexOfScalar(u8, line, ')') orelse return line;
    if (args_end < name_end) return line;
    const ty = typeSpanBefore(line, at) orelse return line;

    var indent_end: usize = 0;
    while (indent_end < line.len and (line[indent_end] == ' ' or line[indent_end] == '\t')) indent_end += 1;

    // `  %name = tail call <ret> @air...`: the pair is bound to a temporary and
    // unpacked below, so the rest of the body keeps reading `%name`.
    var result: []const u8 = &.{};
    var head_start = indent_end;
    if (line[indent_end] == '%') {
        const eq = findPos(line, indent_end, " = ") orelse return line;
        result = line[indent_end + 1 .. eq];
        head_start = eq + 3;
    }
    const unpack = ti.wraps_status and result.len > 0;

    var out: List = .empty;
    try out.appendSlice(gpa, line[0..indent_end]);
    if (result.len > 0) try out.print(gpa, "%{s}{s} = ", .{ result, if (unpack) ".pair" else "" });
    // Only the return type moves; `tail` and the fast-math flags stay put.
    try out.appendSlice(gpa, line[head_start..ty.start]);
    try out.appendSlice(gpa, ti.ret);
    try out.appendSlice(gpa, line[ty.end..name_end]);
    try out.append(gpa, '(');
    var arg_index: usize = 0;
    var it = ArgIterator{ .s = line[name_end + 1 .. args_end] };
    while (it.next()) |arg| : (arg_index += 1) {
        if (arg_index > 0) try out.appendSlice(gpa, ", ");
        const trimmed = std.mem.trim(u8, arg, " \t");
        if (ti.sampler_param == arg_index) {
            const space = std.mem.lastIndexOfScalar(u8, trimmed, ' ') orelse return line;
            try out.print(gpa, "ptr addrspace(2) {s}", .{trimmed[space + 1 ..]});
        } else try out.appendSlice(gpa, trimmed);
    }
    if (arg_index != ti.params.len) return line;
    try out.appendSlice(gpa, line[args_end..]);
    if (unpack) try out.print(gpa, "\n{s}%{s} = extractvalue {s} %{s}.pair, 0", .{ line[0..indent_end], result, ti.ret, result });
    return out.toOwnedSlice(gpa);
}

fn sanitizeBodyLine(gpa: Allocator, line: []const u8) Error![]u8 {
    var s: []u8 = try gpa.dupe(u8, line);
    if (find(s, "getelementptr") != null) {
        s = try replaceAll(gpa, s, " nuw ", " ");
        s = try replaceAll(gpa, s, " nusw ", " ");
    }
    s = try replaceAll(gpa, s, " nneg ", " ");
    s = try replaceAll(gpa, s, " disjoint ", " ");
    s = try replaceAll(gpa, s, " samesign ", " ");
    // LLVM 19 `trunc nuw`/`nsw`: the assembler drops the flag anyway (D5-9);
    // stripping it here keeps the text output readable by Apple's frontend.
    s = try replaceAll(gpa, s, " trunc nuw nsw ", " trunc ");
    s = try replaceAll(gpa, s, " trunc nuw ", " trunc ");
    s = try replaceAll(gpa, s, " trunc nsw ", " trunc ");
    s = try replaceAll(gpa, s, " captures(none)", "");
    s = try replaceAll(gpa, s, " dead_on_unwind", "");
    s = try replaceAll(gpa, s, " dead_on_return", "");
    s = try stripBalanced(gpa, s, " initializes(");
    s = try stripParamAttrWords(gpa, s);
    s = try stripAttrRefs(gpa, s);
    s = try stripMetadataAttachments(gpa, s);
    s = try rewriteTextureCall(gpa, s);
    return s;
}

/// Removes the parameter attributes Zig puts on extern-fn declarations and
/// call sites (`i16 signext 1`, `i1 zeroext true`, `noundef`) that Apple's
/// AIR declares do not carry (research/_results/r06.txt ir_patterns 11).
fn stripParamAttrWords(gpa: Allocator, s: []u8) Error![]u8 {
    var cur = s;
    for ([_][]const u8{ "signext", "zeroext", "noundef" }) |w| {
        cur = try replaceAll(gpa, cur, try std.mem.concat(gpa, u8, &.{ " ", w, " " }), " ");
        cur = try replaceAll(gpa, cur, try std.mem.concat(gpa, u8, &.{ " ", w, ")" }), ")");
        cur = try replaceAll(gpa, cur, try std.mem.concat(gpa, u8, &.{ " ", w, "," }), ",");
    }
    return cur;
}

/// Removes ` #N` attribute-group references (their groups are dropped).
fn stripAttrRefs(gpa: Allocator, s: []u8) Error![]u8 {
    var out: List = .empty;
    var i: usize = 0;
    while (i < s.len) {
        if (s[i] == ' ' and i + 1 < s.len and s[i + 1] == '#' and i + 2 < s.len and std.ascii.isDigit(s[i + 2])) {
            var j = i + 2;
            while (j < s.len and std.ascii.isDigit(s[j])) j += 1;
            i = j;
            continue;
        }
        try out.append(gpa, s[i]);
        i += 1;
    }
    return out.toOwnedSlice(gpa);
}

/// Removes `, !kind !N` attachments (their nodes are dropped).
fn stripMetadataAttachments(gpa: Allocator, s: []u8) Error![]u8 {
    var out: List = .empty;
    var i: usize = 0;
    while (i < s.len) {
        if (std.mem.startsWith(u8, s[i..], ", !")) {
            var j = i + 3;
            while (j < s.len and (std.ascii.isAlphanumeric(s[j]) or s[j] == '_' or s[j] == '.')) j += 1;
            if (j + 2 < s.len and s[j] == ' ' and s[j + 1] == '!' and std.ascii.isDigit(s[j + 2])) {
                var k = j + 2;
                while (k < s.len and std.ascii.isDigit(s[k])) k += 1;
                i = k;
                continue;
            }
        }
        try out.append(gpa, s[i]);
        i += 1;
    }
    return out.toOwnedSlice(gpa);
}

/// Removes `prefix(...)` including balanced parentheses.
fn stripBalanced(gpa: Allocator, s: []u8, prefix: []const u8) Error![]u8 {
    var cur = s;
    while (find(cur, prefix)) |p| {
        var depth: usize = 0;
        var i = p + prefix.len - 1; // at '('
        while (i < cur.len) : (i += 1) {
            switch (cur[i]) {
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if (depth == 0) break;
                },
                else => {},
            }
        }
        if (i >= cur.len) return cur;
        const next = try std.mem.concat(gpa, u8, &.{ cur[0..p], cur[i + 1 ..] });
        cur = next;
    }
    return cur;
}

/// `%7` -> `%__7`, `7:` -> `__7:`  (LLVM requires numbered values to be sequential).
fn renameNumbered(gpa: Allocator, src: []const u8) Error![]u8 {
    var out: List = .empty;
    var i: usize = 0;
    var line_start = true;
    while (i < src.len) {
        const c = src[i];
        if (c == '%' and i + 1 < src.len and std.ascii.isDigit(src[i + 1])) {
            var j = i + 1;
            while (j < src.len and std.ascii.isDigit(src[j])) j += 1;
            if (j >= src.len or !isIdentChar(src[j])) {
                try out.append(gpa, '%');
                try out.appendSlice(gpa, rename_prefix);
                try out.appendSlice(gpa, src[i + 1 .. j]);
                i = j;
                line_start = false;
                continue;
            }
        }
        if (line_start and std.ascii.isDigit(c)) {
            var j = i;
            while (j < src.len and std.ascii.isDigit(src[j])) j += 1;
            if (j < src.len and src[j] == ':') {
                try out.appendSlice(gpa, rename_prefix);
                try out.appendSlice(gpa, src[i .. j + 1]);
                i = j + 1;
                line_start = false;
                continue;
            }
        }
        try out.append(gpa, c);
        line_start = c == '\n';
        i += 1;
    }
    return out.toOwnedSlice(gpa);
}

fn isIdentChar(c: u8) bool {
    return std.ascii.isAlphanumeric(c) or c == '_' or c == '.' or c == '$';
}

/// Replace whole-token occurrences of `from` with `to`.
fn replaceToken(gpa: Allocator, s: []const u8, from: []const u8, to: []const u8) Error![]u8 {
    var out: List = .empty;
    var i: usize = 0;
    while (i < s.len) {
        if (std.mem.startsWith(u8, s[i..], from)) {
            const before_ok = i == 0 or !isIdentChar(s[i - 1]);
            const after_ok = i + from.len >= s.len or !isIdentChar(s[i + from.len]);
            if (before_ok and after_ok) {
                try out.appendSlice(gpa, to);
                i += from.len;
                continue;
            }
        }
        try out.append(gpa, s[i]);
        i += 1;
    }
    return out.toOwnedSlice(gpa);
}

fn replaceAll(gpa: Allocator, s: []const u8, from: []const u8, to: []const u8) Error![]u8 {
    var out: List = .empty;
    var i: usize = 0;
    while (i < s.len) {
        if (std.mem.startsWith(u8, s[i..], from)) {
            try out.appendSlice(gpa, to);
            i += from.len;
            continue;
        }
        try out.append(gpa, s[i]);
        i += 1;
    }
    return out.toOwnedSlice(gpa);
}

fn find(hay: []const u8, needle: []const u8) ?usize {
    return findPos(hay, 0, needle);
}

fn findPos(hay: []const u8, start: usize, needle: []const u8) ?usize {
    if (needle.len == 0 or hay.len < needle.len) return null;
    var i = start;
    while (i + needle.len <= hay.len) : (i += 1) {
        if (std.mem.eql(u8, hay[i .. i + needle.len], needle)) return i;
    }
    return null;
}

fn findLastScalar(hay: []const u8, c: u8) ?usize {
    var i = hay.len;
    while (i > 0) {
        i -= 1;
        if (hay[i] == c) return i;
    }
    return null;
}

// ── Tests ─────────────────────────────────────────────────────────────────

test {
    _ = assembler;
    _ = metadata;
    _ = metallib;
    _ = target;
}

test "parseDefine strips linkage and finds params" {
    const d = try parseDefine("define private fastcc void @m.f(ptr sret(%m.T) align 16 %__0, i32 %__1) unnamed_addr #1 {");
    try std.testing.expectEqualStrings("void", d.ret);
    try std.testing.expectEqualStrings("m.f", d.full_name);
    try std.testing.expectEqualStrings("ptr sret(%m.T) align 16 %__0, i32 %__1", d.params);
}

test "parseDefine strips range() and other LLVM 19 return attributes, keeps linkage and fastcc" {
    const d = try parseDefine("define private fastcc noundef range(i32 0, 5) i32 @m.maskToInt(<4 x float> %__0) unnamed_addr #0 {");
    try std.testing.expectEqualStrings("i32", d.ret);
    try std.testing.expectEqualStrings("m.maskToInt", d.full_name);
    try std.testing.expectEqualStrings("private", d.linkage);
    try std.testing.expect(d.fastcc);
    const k = try parseDefine("define ptx_kernel void @m.kernel(ptr addrspace(1) %__0) #1 {");
    try std.testing.expectEqualStrings("void", k.ret);
    try std.testing.expectEqualStrings("", k.linkage);
    try std.testing.expect(!k.fastcc);
    const a = try parseDefine("define internal amdgpu_kernel void @k2() {");
    try std.testing.expectEqualStrings("void", a.ret);
    try std.testing.expectEqualStrings("internal", a.linkage);
}

test "D4 helpers: reachable non-entry defines are kept, lifetime dropped, ptx_kernel stripped, unreachable helpers dropped" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();
    const input =
        \\target datalayout = "e-p6:32:32-i64:64"
        \\target triple = "nvptx64-nvidia-cuda12.5.0-unknown"
        \\
        \\%my_shader.VertexOut = type { <4 x float>, <3 x float>, <2 x float>, [8 x i8] }
        \\%calls.Hit = type { <3 x float>, float, i8, [11 x i8] }
        \\
        \\define noundef nonnull ptr @__keep_fragmentShader() local_unnamed_addr #0 {
        \\  ret ptr @my_shader.fragmentShader
        \\}
        \\
        \\define private fastcc void @my_shader.vertexShader(ptr dead_on_unwind noalias writeonly sret(%my_shader.VertexOut) align 16 captures(none) %0, i32 %1, ptr addrspace(1) readonly captures(none) %2) unnamed_addr #1 {
        \\  store <4 x float> zeroinitializer, ptr %0, align 16
        \\  ret void
        \\}
        \\
        \\define private fastcc noundef <4 x float> @my_shader.fragmentShader(<4 x float> %0, <3 x float> %1, <2 x float> %2, ptr addrspace(1) readonly captures(none) %3) unnamed_addr #0 {
        \\  %5 = alloca %calls.Hit, align 16
        \\  call void @llvm.lifetime.start.p0(ptr nonnull %5)
        \\  call fastcc void @calls.intersect(ptr dead_on_unwind noalias nonnull writeonly align 16 captures(none) initializes((0, 12)) %5, <3 x float> %1, <3 x float> %1)
        \\  %6 = tail call fastcc float @calls.lengthSq(<3 x float> %1)
        \\  call void @kernel_like(ptr addrspace(1) %3)
        \\  %7 = load <3 x float>, ptr %5, align 16
        \\  call void @llvm.lifetime.end.p0(ptr nonnull %5)
        \\  %8 = shufflevector <3 x float> %7, <3 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 poison>
        \\  %9 = insertelement <4 x float> %8, float %6, i64 3
        \\  ret <4 x float> %9
        \\}
        \\
        \\define private fastcc noundef range(i32 0, 5) float @calls.lengthSq(<3 x float> returned %0) unnamed_addr #2 {
        \\  %2 = extractelement <3 x float> %0, i64 0
        \\  %3 = fmul float %2, %2
        \\  ret float %3
        \\}
        \\
        \\define private fastcc void @calls.intersect(ptr dead_on_unwind noalias nonnull writeonly sret(%calls.Hit) align 16 captures(none) initializes((0, 12), (16, 21)) %0, <3 x float> %1, <3 x float> %2) unnamed_addr #3 {
        \\  %4 = tail call fastcc float @calls.lengthSq(<3 x float> %1)
        \\  %5 = insertelement <3 x float> %2, float %4, i64 0
        \\  store <3 x float> %5, ptr %0, align 16
        \\  ret void
        \\}
        \\
        \\define ptx_kernel void @kernel_like(ptr addrspace(1) noundef %0) #4 {
        \\  ret void
        \\}
        \\
        \\define private fastcc float @calls.unused(float %0) unnamed_addr #2 {
        \\  %2 = tail call fastcc float @calls.alsoUnused(float %0)
        \\  ret float %2
        \\}
        \\
        \\define private fastcc float @calls.alsoUnused(float %0) unnamed_addr #2 {
        \\  ret float %0
        \\}
        \\
        \\declare void @llvm.lifetime.start.p0(ptr captures(none)) #5
        \\declare void @llvm.lifetime.end.p0(ptr captures(none)) #5
        \\
        \\attributes #0 = { nounwind }
        \\
    ;
    const out = try convertWith(gpa, input, .{ .require_all_entries = false });
    // Reached helpers survive with private/fastcc, without attributes.
    try std.testing.expect(find(out, "define private fastcc float @calls.lengthSq(<3 x float> %__0) {") != null);
    try std.testing.expect(find(out, "define private fastcc void @calls.intersect(ptr %__0, <3 x float> %__1, <3 x float> %__2) {") != null);
    try std.testing.expect(find(out, "  %__4 = tail call fastcc float @calls.lengthSq(<3 x float> %__1)") != null);
    // Kernel convention stripped, function kept (it is reached).
    try std.testing.expect(find(out, "define void @kernel_like(ptr addrspace(1) %__0) {") != null);
    try std.testing.expect(find(out, "ptx_kernel") == null);
    // Lifetime markers gone, calls and declares alike.
    try std.testing.expect(find(out, "llvm.lifetime") == null);
    // Unreachable functions gone, including the helper only they reach.
    try std.testing.expect(find(out, "__keep_fragmentShader") == null);
    try std.testing.expect(find(out, "calls.unused") == null);
    try std.testing.expect(find(out, "calls.alsoUnused") == null);
    try std.testing.expect(find(out, "initializes(") == null);
    try std.testing.expect(find(out, "returned") == null);
    try std.testing.expect(find(out, "range(") == null);
    // The result assembles for the fragment entry with the helpers inside.
    inline for (metadata.function_metadata) |fm| {
        if (comptime std.mem.eql(u8, fm.name, "fragmentShader")) {
            const bc = try assembler.assemble(gpa, out, .{ .entry = fm.name, .fm = fm }, target.default);
            try std.testing.expectEqualStrings("BC\xC0\xDE", bc[0..4]);
        }
    }
}

test "D4 quoted identifiers: generic helpers `@\"mod.Vec(4).sum\"` are followed, kept verbatim and assemble" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();
    const input =
        \\target datalayout = "e-p6:32:32-i64:64"
        \\target triple = "nvptx64-nvidia-cuda12.5.0-unknown"
        \\
        \\%v_helpers.VertexOut = type { <4 x float>, <3 x float>, <2 x float>, [8 x i8] }
        \\
        \\define noundef nonnull ptr @__keep_fragmentShader() local_unnamed_addr #0 {
        \\  ret ptr @v_helpers.fragmentShader
        \\}
        \\
        \\define private fastcc void @v_helpers.vertexShader(ptr dead_on_unwind noalias writeonly sret(%v_helpers.VertexOut) align 16 captures(none) %0, i32 %1, ptr addrspace(1) readonly captures(none) %2) unnamed_addr #1 {
        \\  store <4 x float> zeroinitializer, ptr %0, align 16
        \\  ret void
        \\}
        \\
        \\define private fastcc noundef <4 x float> @v_helpers.fragmentShader(<4 x float> %0, <3 x float> %1, <2 x float> %2, ptr addrspace(1) readonly captures(none) %3) unnamed_addr #0 {
        \\  %5 = tail call fastcc float @v_helpers.lengthSq(<3 x float> %1)
        \\  %6 = tail call fastcc float @"v_helpers.Vec(4).sum"(<4 x float> %0)
        \\  %7 = fadd float %5, %6
        \\  %8 = insertelement <4 x float> %0, float %7, i64 0
        \\  ret <4 x float> %8
        \\}
        \\
        \\define private fastcc float @v_helpers.lengthSq(<3 x float> %0) unnamed_addr #2 {
        \\  %2 = fmul <3 x float> %0, %0
        \\  %3 = tail call fastcc float @"v_helpers.Vec(3).sum"(<3 x float> %2)
        \\  ret float %3
        \\}
        \\
        \\define private fastcc float @"v_helpers.Vec(4).sum"(<4 x float> %0) unnamed_addr #2 {
        \\  %2 = extractelement <4 x float> %0, i64 0
        \\  %3 = extractelement <4 x float> %0, i64 1
        \\  %4 = fadd float %2, %3
        \\  ret float %4
        \\}
        \\
        \\define private fastcc float @"v_helpers.Vec(3).sum"(<3 x float> %0) unnamed_addr #2 {
        \\  %2 = extractelement <3 x float> %0, i64 0
        \\  ret float %2
        \\}
        \\
        \\define private fastcc float @"v_helpers.Vec(2).sum"(<2 x float> %0) unnamed_addr #2 {
        \\  %2 = extractelement <2 x float> %0, i64 0
        \\  ret float %2
        \\}
        \\
        \\attributes #0 = { nounwind }
        \\
    ;
    const out = try convertWith(gpa, input, .{ .require_all_entries = false });
    // Quoted helpers reached directly and transitively are kept, spelled as in the input.
    try std.testing.expect(find(out, "define private fastcc float @\"v_helpers.Vec(4).sum\"(<4 x float> %__0) {") != null);
    try std.testing.expect(find(out, "define private fastcc float @\"v_helpers.Vec(3).sum\"(<3 x float> %__0) {") != null);
    try std.testing.expect(find(out, "tail call fastcc float @\"v_helpers.Vec(4).sum\"(<4 x float> %__0)") != null);
    // The unreached instantiation is dropped.
    try std.testing.expect(find(out, "Vec(2)") == null);
    try std.testing.expect(find(out, "__keep_fragmentShader") == null);
    try std.testing.expectEqualStrings("mod.Vec(4).sum", unquote("\"mod.Vec(4).sum\""));
    try std.testing.expectEqualStrings("plain", unquote("plain"));
    const d = try parseDefine("define private fastcc float @\"v_helpers.Vec(4).sum\"(<4 x float> %0) unnamed_addr #2 {");
    try std.testing.expectEqualStrings("\"v_helpers.Vec(4).sum\"", d.full_name);
    try std.testing.expectEqualStrings("<4 x float> %0", d.params);
    // The assembler resolves the quoted calls.
    inline for (metadata.function_metadata) |fm| {
        if (comptime std.mem.eql(u8, fm.name, "fragmentShader")) {
            const bc = try assembler.assemble(gpa, out, .{ .entry = fm.name, .fm = fm }, target.default);
            try std.testing.expectEqualStrings("BC\xC0\xDE", bc[0..4]);
        }
    }
}

test "D4 quoted named types: the r5_generic shape (quoted `%` type + quoted `@` helper) splices and assembles" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();
    // Trimmed from review-a2/rr/ll/r5_generic.ReleaseFast.ll: a generic
    // `Pair(f32)` built by a noinline helper lands in a quoted named type.
    const input =
        \\target datalayout = "e-p6:32:32-i64:64"
        \\target triple = "nvptx64-nvidia-cuda12.5.0-unknown"
        \\
        \\%r5_generic.VertexOut = type { <4 x float>, <3 x float>, <2 x float>, [8 x i8] }
        \\%"r5_generic.Pair(f32)" = type { float, float }
        \\
        \\define noundef nonnull ptr @__keep_vertexShader() local_unnamed_addr #0 {
        \\  ret ptr @r5_generic.vertexShader
        \\}
        \\
        \\define noundef nonnull ptr @__keep_fragmentShader() local_unnamed_addr #0 {
        \\  ret ptr @r5_generic.fragmentShader
        \\}
        \\
        \\define private fastcc void @r5_generic.vertexShader(ptr dead_on_unwind noalias writeonly sret(%r5_generic.VertexOut) align 16 captures(none) initializes((0, 28), (32, 40)) %0, i32 %1, ptr addrspace(1) readonly captures(none) %2) unnamed_addr #1 {
        \\  store <4 x float> zeroinitializer, ptr %0, align 16
        \\  ret void
        \\}
        \\
        \\define private fastcc <4 x float> @r5_generic.fragmentShader(<4 x float> %0, <3 x float> %1, <2 x float> %2, ptr addrspace(1) readonly align 1 captures(none) %3) unnamed_addr #0 {
        \\  %5 = alloca %"r5_generic.Pair(f32)", align 4
        \\  %6 = extractelement <2 x float> %2, i64 0
        \\  %7 = extractelement <3 x float> %1, i64 0
        \\  call fastcc void @"r5_generic.mk(f32)"(ptr dead_on_unwind noalias writeonly align 4 captures(none) %5, float %6, float %7)
        \\  %.sroa.06.0.copyload = load float, ptr %5, align 4
        \\  %.sroa.27.0..sroa_idx = getelementptr inbounds nuw i8, ptr %5, i64 4
        \\  %.sroa.27.0.copyload = load float, ptr %.sroa.27.0..sroa_idx, align 4
        \\  %8 = fadd float %.sroa.06.0.copyload, %.sroa.27.0.copyload
        \\  %9 = insertelement <4 x float> %0, float %8, i64 0
        \\  ret <4 x float> %9
        \\}
        \\
        \\define private fastcc void @"r5_generic.mk(f32)"(ptr dead_on_unwind noalias nonnull writeonly sret(%"r5_generic.Pair(f32)") align 4 captures(none) initializes((0, 8)) %0, float %1, float %2) unnamed_addr #2 {
        \\  store float %1, ptr %0, align 4
        \\  %4 = getelementptr inbounds nuw i8, ptr %0, i64 4
        \\  store float %2, ptr %4, align 4
        \\  ret void
        \\}
        \\
        \\attributes #0 = { nounwind }
        \\
    ;
    const out = try convertWith(gpa, input, .{ .require_all_entries = false });
    // The quoted type line and the quoted helper are kept verbatim.
    try std.testing.expect(find(out, "%\"r5_generic.Pair(f32)\" = type { float, float }") != null);
    try std.testing.expect(find(out, "%__5 = alloca %\"r5_generic.Pair(f32)\", align 4") != null);
    try std.testing.expect(find(out, "define private fastcc void @\"r5_generic.mk(f32)\"(ptr %__0, float %__1, float %__2) {") != null);
    try std.testing.expect(find(out, "call fastcc void @\"r5_generic.mk(f32)\"(ptr ") != null);
    try std.testing.expect(find(out, "sret(") == null);
    // And both entries assemble.
    inline for (metadata.function_metadata) |fm| {
        if (comptime (std.mem.eql(u8, fm.name, "fragmentShader") or std.mem.eql(u8, fm.name, "vertexShader"))) {
            const bc = try assembler.assemble(gpa, out, .{ .entry = fm.name, .fm = fm }, target.default);
            try std.testing.expectEqualStrings("BC\xC0\xDE", bc[0..4]);
        }
    }
}

test "parseParam handles vectors, address spaces and sret" {
    const a = try parseParam("<4 x float> noundef %__0");
    try std.testing.expectEqualStrings("<4 x float>", a.ty);
    try std.testing.expectEqualStrings("%__0", a.name);

    const b = try parseParam("ptr addrspace(1) readonly align 16 captures(none) %__2");
    try std.testing.expectEqualStrings("ptr addrspace(1)", b.ty);

    const c = try parseParam("ptr dead_on_unwind noalias writeonly sret(%m.VertexOut) align 16 %__0");
    try std.testing.expectEqualStrings("%m.VertexOut", c.sret_ty.?);
}

test "renameNumbered renames numbered values and labels" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const out = try renameNumbered(arena.allocator(), "%4 = zext i32 %1 to i64\n5:\n  ret ptr %4\n");
    try std.testing.expectEqualStrings("%__4 = zext i32 %__1 to i64\n__5:\n  ret ptr %__4\n", out);
}

test "D4 kernels: an entry block that a phi names by number gets an explicit label (xcrun metal cross-check)" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();
    // reduceKernel's guarded loop (review-b/r4/reduce/reduce.nvptx.ll):
    // the loop-carried phi refers to the implicit entry block `%5`.
    const input =
        \\target datalayout = "e-p6:32:32-i64:64"
        \\target triple = "nvptx64-nvidia-cuda12.5.0-unknown"
        \\
        \\declare void @air.wg.barrier(i32, i32)
        \\
        \\define ptx_kernel void @reduceKernel(ptr addrspace(1) readonly %0, ptr addrspace(1) writeonly %1, i32 %2, i32 %3, i32 %4) local_unnamed_addr #1 {
        \\  %.06 = lshr i32 %3, 1
        \\  %.not7 = icmp eq i32 %.06, 0
        \\  br i1 %.not7, label %._crit_edge, label %.lr.ph
        \\
        \\.lr.ph:                                           ; preds = %5, %6
        \\  %.08 = phi i32 [ %.0, %6 ], [ %.06, %5 ]
        \\  br label %6
        \\
        \\._crit_edge:                                      ; preds = %6, %5
        \\  ret void
        \\
        \\6:                                                ; preds = %.lr.ph
        \\  tail call void @air.wg.barrier(i32 2, i32 1) #4
        \\  %.0 = lshr i32 %.08, 1
        \\  %.not = icmp eq i32 %.0, 0
        \\  br i1 %.not, label %._crit_edge, label %.lr.ph
        \\}
        \\
        \\define private fastcc void @my_shader.helper(i32 %0) unnamed_addr #2 {
        \\  br label %2
        \\
        \\2:
        \\  %3 = phi i32 [ %0, %1 ], [ 0, %2 ]
        \\  br label %2
        \\}
        \\
    ;
    const out = try convertWith(gpa, input, .{ .require_all_entries = false });
    try std.testing.expect(find(out, "define void @reduceKernel(ptr addrspace(1) %__0, ptr addrspace(1) %__1, i32 %__2, i32 %__3, i32 %__4) {\n__5:\n  %.06 = lshr i32 %__3, 1") != null);
    try std.testing.expect(find(out, "[ %.06, %__5 ]") != null);
    // The helper is not reached, so it is dropped; the scaleKernel-style
    // bodies that never name their entry block keep no label (see the
    // kernel test above: its output starts with the first instruction).
    try std.testing.expect(find(out, "@my_shader.helper") == null);
    const bc = try assembler.assemble(gpa, out, .{ .entry = "reduceKernel", .fm = reduce_fm }, target.default);
    try std.testing.expectEqualStrings("BC\xC0\xDE", bc[0..4]);
}

const reduce_fm = metadata.functionMetadata(.{
    .name = "reduceKernel",
    .stage = .kernel,
    .ret = void,
    .args = &.{
        .{ .buffer = .{ .index = 0, .T = f32, .name = "in" } },
        .{ .buffer = .{ .index = 1, .T = f32, .name = "out", .access = .read_write } },
        .{ .builtin = .{ .kind = .thread_index_in_threadgroup, .T = u32, .name = "tidx" } },
        .{ .builtin = .{ .kind = .threads_per_threadgroup, .T = u32, .name = "tg_size" } },
        .{ .builtin = .{ .kind = .threadgroup_position_in_grid, .T = u32, .name = "tg_pos" } },
    },
});

test "sanitizeBodyLine strips new attributes" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const out = try sanitizeBodyLine(arena.allocator(), "  %p = getelementptr inbounds nuw i8, ptr %s, i64 16, !tbaa !3");
    try std.testing.expectEqualStrings("  %p = getelementptr inbounds i8, ptr %s, i64 16", out);
    const t = try sanitizeBodyLine(arena.allocator(), "  %__15 = trunc nuw i8 %.sroa.4.0.copyload to i1");
    try std.testing.expectEqualStrings("  %__15 = trunc i8 %.sroa.4.0.copyload to i1", t);
    const t2 = try sanitizeBodyLine(arena.allocator(), "  %x = trunc nuw nsw i64 %y to i32");
    try std.testing.expectEqualStrings("  %x = trunc i64 %y to i32", t2);
}

test "convert rewrites a Zig sret vertex function into AIR form and the result assembles" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();
    const input =
        \\; ModuleID = 'BitcodeBuffer'
        \\target datalayout = "e-p6:32:32-i64:64"
        \\target triple = "nvptx64-nvidia-cuda12.5.0-unknown"
        \\
        \\%my_shader.VertexOut = type { <4 x float>, <3 x float>, <2 x float>, [8 x i8] }
        \\
        \\define noundef nonnull ptr @__keep_vertexShader() local_unnamed_addr #0 {
        \\  ret ptr @my_shader.vertexShader
        \\}
        \\
        \\define private fastcc void @my_shader.vertexShader(ptr dead_on_unwind noalias writeonly sret(%my_shader.VertexOut) align 16 captures(none) initializes((0, 28), (32, 40)) %0, i32 %1, ptr addrspace(1) readonly align 16 captures(none) %2) unnamed_addr #1 {
        \\  %4 = zext i32 %1 to i64
        \\  store <4 x float> zeroinitializer, ptr %0, align 16
        \\  ret void
        \\}
        \\
        \\define private fastcc noundef <4 x float> @my_shader.fragmentShader(<4 x float> %0, <3 x float> %1, <2 x float> %2, ptr addrspace(1) readonly captures(none) %3) unnamed_addr #0 {
        \\  ret <4 x float> zeroinitializer
        \\}
        \\
        \\attributes #0 = { nounwind }
        \\
        \\!llvm.module.flags = !{}
        \\
    ;
    const out = try convertWith(gpa, input, .{ .require_all_entries = false });
    try std.testing.expect(find(out, "target triple = \"air64_v28-apple-macosx26.0.0\"") != null);
    try std.testing.expect(find(out, "@__keep_vertexShader") == null);
    try std.testing.expect(find(out, "@my_shader.vertexShader") == null);
    try std.testing.expect(find(out, "define <{ <4 x float>, <3 x float>, <2 x float> }> @vertexShader(i32 %__1, ptr addrspace(1) %__2) {") != null);
    try std.testing.expect(find(out, "  %sret = alloca %my_shader.VertexOut, align 16") != null);
    try std.testing.expect(find(out, "store <4 x float> zeroinitializer, ptr %sret, align 16") != null);
    try std.testing.expect(find(out, "ret <{ <4 x float>, <3 x float>, <2 x float> }> %r0.a2") != null);
    try std.testing.expect(find(out, "define <4 x float> @fragmentShader(<4 x float> %__0, <3 x float> %__1, <2 x float> %__2, ptr addrspace(1) %__3) {") != null);
    try std.testing.expect(find(out, "captures(none)") == null);
    try std.testing.expect(find(out, "attributes #") == null);

    // The rewritten text must be assemblable and pack into a valid container.
    // Only the manifest entries this fixed IR defines take part, so a larger
    // manifest does not break the test.
    var functions: [metadata.function_metadata.len]metallib.Function = undefined;
    var n: usize = 0;
    inline for (metadata.function_metadata) |fm| {
        if (find(out, "@" ++ fm.name ++ "(") != null) {
            const bc = try assembler.assemble(gpa, out, .{ .entry = fm.name, .fm = fm }, target.default);
            try std.testing.expectEqualStrings("BC\xC0\xDE", bc[0..4]);
            functions[n] = .{ .name = fm.name, .stage = if (fm.stage == .vertex) .vertex else .fragment, .bitcode = bc };
            n += 1;
        }
    }
    try std.testing.expectEqual(2, n);
    const image = try metallib.pack(gpa, functions[0..n], "default.metallib", target.default);
    try std.testing.expectEqualStrings("MTLB", image[0..4]);
}

test "D2/D4 fragment struct return: Zig's sret FragOut becomes a packed literal struct return in field order (r01 findings 7-8)" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();
    // Shape of research/verify-render-bindings/v4_frag.ll for
    // my_shader.fragmentShaderMRT: the discard branch precedes the stores,
    // the FragOut is laid out { <4 x float>, <4 x half>, float, [4 x i8] }.
    const input =
        \\target datalayout = "e-p6:32:32-i64:64"
        \\target triple = "nvptx64-nvidia-cuda12.5.0-unknown"
        \\
        \\%my_shader.FragOut = type { <4 x float>, <4 x half>, float, [4 x i8] }
        \\
        \\define noundef nonnull ptr @__keep_fragmentShaderMRT() local_unnamed_addr #0 {
        \\  ret ptr @my_shader.fragmentShaderMRT
        \\}
        \\
        \\declare void @air.discard_fragment() local_unnamed_addr
        \\
        \\define private fastcc void @my_shader.fragmentShaderMRT(ptr dead_on_unwind noalias writeonly sret(%my_shader.FragOut) align 16 captures(none) initializes((0, 28)) %0, <4 x float> %1, i32 %2, <2 x float> %3, i1 %4, <2 x float> %5, ptr addrspace(4) readonly align 16 captures(none) %6) unnamed_addr #1 {
        \\  %8 = getelementptr inbounds nuw i8, ptr addrspace(4) %6, i64 84
        \\  %9 = load i32, ptr addrspace(4) %8, align 4
        \\  %10 = and i32 %9, 1
        \\  %11 = icmp eq i32 %10, 0
        \\  %12 = extractelement <2 x float> %5, i64 0
        \\  %13 = fcmp ugt float %12, 5.000000e-01
        \\  %14 = select i1 %11, i1 false, i1 %13
        \\  br i1 %14, label %15, label %16
        \\
        \\15:
        \\  tail call void @air.discard_fragment() #3
        \\  br label %16
        \\
        \\16:
        \\  %17 = uitofp i32 %2 to float
        \\  %18 = extractelement <2 x float> %3, i64 0
        \\  %19 = insertelement <4 x float> <float poison, float poison, float poison, float 1.000000e+00>, float %18, i64 0
        \\  %20 = insertelement <4 x float> %19, float %17, i64 2
        \\  %21 = select i1 %4, <4 x float> %20, <4 x float> zeroinitializer
        \\  store <4 x float> %21, ptr %0, align 16
        \\  %22 = getelementptr inbounds nuw i8, ptr %0, i64 16
        \\  %23 = shufflevector <2 x float> %5, <2 x float> poison, <4 x i32> <i32 0, i32 1, i32 poison, i32 poison>
        \\  %24 = fptrunc <4 x float> %23 to <4 x half>
        \\  store <4 x half> %24, ptr %22, align 16
        \\  %25 = getelementptr inbounds nuw i8, ptr %0, i64 24
        \\  %26 = extractelement <4 x float> %1, i64 2
        \\  %27 = fmul float %26, 5.000000e-01
        \\  store float %27, ptr %25, align 8
        \\  ret void
        \\}
        \\
        \\attributes #0 = { nounwind }
        \\
    ;
    const out = try convertWith(gpa, input, .{ .require_all_entries = false });
    // Header: packed struct return, sret gone, stage_in fields / builtins / constant buffer typed from the manifest.
    try std.testing.expect(find(out, "define <{ <4 x float>, <4 x half>, float }> @fragmentShaderMRT(<4 x float> %__1, i32 %__2, <2 x float> %__3, i1 %__4, <2 x float> %__5, ptr addrspace(2) %__6) {") != null);
    try std.testing.expect(find(out, "  %sret = alloca %my_shader.FragOut, align 16") != null);
    try std.testing.expect(find(out, "sret(") == null);
    // The stores now target the alloca; the discard call survives in its branch.
    try std.testing.expect(find(out, "store <4 x float> %__21, ptr %sret, align 16") != null);
    try std.testing.expect(find(out, "  tail call void @air.discard_fragment()\n  br label %__16") != null);
    // Epilogue: loads at the Zig offsets 0 / 16 / 24, inserted in field order.
    try std.testing.expect(find(out, "  %r0.f0 = load <4 x float>, ptr %sret, align 16\n") != null);
    try std.testing.expect(find(out, "  %r0.p1 = getelementptr inbounds i8, ptr %sret, i64 16\n  %r0.f1 = load <4 x half>, ptr %r0.p1, align 16\n") != null);
    try std.testing.expect(find(out, "  %r0.p2 = getelementptr inbounds i8, ptr %sret, i64 24\n  %r0.f2 = load float, ptr %r0.p2, align 8\n") != null);
    try std.testing.expect(find(out, "  %r0.a0 = insertvalue <{ <4 x float>, <4 x half>, float }> undef, <4 x float> %r0.f0, 0\n  %r0.a1 = insertvalue <{ <4 x float>, <4 x half>, float }> %r0.a0, <4 x half> %r0.f1, 1\n  %r0.a2 = insertvalue <{ <4 x float>, <4 x half>, float }> %r0.a1, float %r0.f2, 2\n  ret <{ <4 x float>, <4 x half>, float }> %r0.a2\n") != null);
    try std.testing.expect(find(out, "ret void") == null);
    // It assembles with the MRT metadata.
    inline for (metadata.function_metadata) |fm| {
        if (comptime std.mem.eql(u8, fm.name, "fragmentShaderMRT")) {
            const bc = try assembler.assemble(gpa, out, .{ .entry = fm.name, .fm = fm }, target.default);
            try std.testing.expectEqualStrings("BC\xC0\xDE", bc[0..4]);
        }
    }
    // A fragment struct return that Zig did not pass via sret is refused.
    const by_value =
        \\define private fastcc <4 x float> @my_shader.fragmentShaderMRT(<4 x float> %0, i32 %1, <2 x float> %2, i1 %3, <2 x float> %4, ptr addrspace(4) %5) {
        \\  ret <4 x float> zeroinitializer
        \\}
        \\
    ;
    try std.testing.expectError(error.FragmentReturnNotSret, convertWith(gpa, by_value, .{ .require_all_entries = false }));
}

test "D4 entries: vertex/fragment parameter counts and types are checked against the manifest (stage_in minus point_size)" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();
    // fragmentShaderMRT with a parameter for point_size, which Metal never passes.
    const with_point_size =
        \\define private fastcc void @my_shader.fragmentShaderMRT(ptr sret(%my_shader.FragOut) %0, <4 x float> %1, float %2, i32 %3, <2 x float> %4, i1 %5, <2 x float> %6, ptr addrspace(4) %7) {
        \\  ret void
        \\}
        \\
    ;
    try std.testing.expectError(error.ParamCountMismatch, convertWith(gpa, with_point_size, .{ .require_all_entries = false }));
    // front_facing declared as u32 instead of bool.
    const wrong_builtin =
        \\define private fastcc void @my_shader.fragmentShaderMRT(ptr sret(%my_shader.FragOut) %0, <4 x float> %1, i32 %2, <2 x float> %3, i32 %4, <2 x float> %5, ptr addrspace(4) %6) {
        \\  ret void
        \\}
        \\
    ;
    try std.testing.expectError(error.ParamTypeMismatch, convertWith(gpa, wrong_builtin, .{ .require_all_entries = false }));
    // vertexShaderInstanced missing its base_instance parameter.
    const short_vertex =
        \\define private fastcc void @my_shader.vertexShaderInstanced(ptr sret(%my_shader.VertexOutInstanced) %0, i32 %1, i32 %2, i32 %3, ptr addrspace(1) %4, ptr addrspace(4) %5) {
        \\  ret void
        \\}
        \\
    ;
    try std.testing.expectError(error.ParamCountMismatch, convertWith(gpa, short_vertex, .{ .require_all_entries = false }));
}

test "D4 kernels: `define ptx_kernel void @name` is an entry, header rebuilt from the manifest, addrspace(4) -> (2), signext/zeroext dropped" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();
    // Shape of the IR Zig 0.17.0-dev emits for src/engine/my_shader.zig's
    // kernels (research/_results/r06.txt ir_patterns 8, 10, 11).
    const input =
        \\target datalayout = "e-p6:32:32-i64:64"
        \\target triple = "nvptx64-nvidia-cuda12.5.0-unknown"
        \\
        \\%my_shader.Params = type { i32, float }
        \\
        \\@my_shader.scratch = private unnamed_addr addrspace(3) global [64 x float] undef, align 4
        \\
        \\define ptx_kernel void @scaleKernel(ptr addrspace(1) readonly align 4 captures(none) %0, ptr addrspace(1) writeonly align 4 captures(none) %1, ptr addrspace(4) readonly align 4 captures(none) %2, i32 noundef %3) local_unnamed_addr #0 {
        \\  %5 = load i32, ptr addrspace(4) %2, align 4
        \\  %6 = icmp ult i32 %3, %5
        \\  br i1 %6, label %7, label %14
        \\
        \\7:
        \\  %8 = getelementptr inbounds nuw i8, ptr addrspace(4) %2, i64 4
        \\  %9 = load float, ptr addrspace(4) %8, align 4
        \\  %10 = zext nneg i32 %3 to i64
        \\  %11 = getelementptr inbounds [4 x i8], ptr addrspace(1) %0, i64 %10
        \\  %12 = load float, ptr addrspace(1) %11, align 4
        \\  %13 = fmul float %12, %9
        \\  %15 = getelementptr inbounds [4 x i8], ptr addrspace(1) %1, i64 %10
        \\  store float %13, ptr addrspace(1) %15, align 4
        \\  br label %14
        \\
        \\14:
        \\  ret void
        \\}
        \\
        \\define ptx_kernel void @reverseKernel(ptr addrspace(1) readonly align 4 captures(none) %0, ptr addrspace(1) writeonly align 4 captures(none) %1, i32 %2, i32 %3, i32 %4, i32 %5) local_unnamed_addr #1 {
        \\  %7 = zext i32 %5 to i64
        \\  %8 = getelementptr inbounds [4 x i8], ptr addrspace(3) @my_shader.scratch, i64 %7
        \\  %9 = zext i32 %2 to i64
        \\  %10 = getelementptr inbounds [4 x i8], ptr addrspace(1) %0, i64 %9
        \\  %11 = load float, ptr addrspace(1) %10, align 4
        \\  store float %11, ptr addrspace(3) %8, align 4
        \\  tail call void @air.wg.barrier(i32 2, i32 1) #2
        \\  %12 = load float, ptr addrspace(3) %8, align 4
        \\  %13 = getelementptr inbounds [4 x i8], ptr addrspace(1) %1, i64 %9
        \\  store float %12, ptr addrspace(1) %13, align 4
        \\  ret void
        \\}
        \\
        \\define ptx_kernel void @sumKernel(ptr addrspace(1) readonly align 4 captures(none) %0, ptr addrspace(1) writeonly align 4 captures(none) %1, i32 noundef %2, i32 %3, i32 %4) local_unnamed_addr #1 {
        \\  %6 = zext i32 %2 to i64
        \\  %7 = getelementptr inbounds [4 x i8], ptr addrspace(1) %0, i64 %6
        \\  %8 = load float, ptr addrspace(1) %7, align 4
        \\  %9 = tail call float @air.simd_sum.f32(float %8) #2
        \\  %10 = tail call float @air.simd_shuffle_down.f32(float %8, i16 signext 1) #2
        \\  %11 = fadd float %9, %10
        \\  %12 = zext i32 %4 to i64
        \\  %13 = getelementptr inbounds [4 x i8], ptr addrspace(1) %1, i64 %12
        \\  store float %11, ptr addrspace(1) %13, align 4
        \\  ret void
        \\}
        \\
        \\define ptx_kernel void @countKernel(ptr addrspace(1) align 4 captures(none) %0, ptr addrspace(1) readonly align 4 captures(none) %1, i32 %2) local_unnamed_addr #1 {
        \\  %4 = tail call i32 @air.atomic.global.add.u.i32(ptr addrspace(1) align 4 %0, i32 1, i32 0, i32 2, i1 zeroext true) #2
        \\  ret void
        \\}
        \\
        \\declare void @air.wg.barrier(i32, i32) local_unnamed_addr #2
        \\declare float @air.simd_sum.f32(float) local_unnamed_addr #2
        \\declare float @air.simd_shuffle_down.f32(float, i16 signext) local_unnamed_addr #2
        \\declare i32 @air.atomic.global.add.u.i32(ptr addrspace(1) align 4, i32, i32, i32, i1 zeroext) local_unnamed_addr #2
        \\
        \\attributes #0 = { nounwind }
        \\
    ;
    const out = try convertWith(gpa, input, .{ .require_all_entries = false });
    // Headers rebuilt from the manifest: no ptx_kernel, builtins i32, constant buffer addrspace(2).
    try std.testing.expect(find(out, "define void @scaleKernel(ptr addrspace(1) %__0, ptr addrspace(1) %__1, ptr addrspace(2) %__2, i32 %__3) {") != null);
    try std.testing.expect(find(out, "define void @reverseKernel(ptr addrspace(1) %__0, ptr addrspace(1) %__1, i32 %__2, i32 %__3, i32 %__4, i32 %__5) {") != null);
    try std.testing.expect(find(out, "define void @sumKernel(ptr addrspace(1) %__0, ptr addrspace(1) %__1, i32 %__2, i32 %__3, i32 %__4) {") != null);
    try std.testing.expect(find(out, "define void @countKernel(ptr addrspace(1) %__0, ptr addrspace(1) %__1, i32 %__2) {") != null);
    try std.testing.expect(find(out, "ptx_kernel") == null);
    // nvptx `.param` loads become Metal constant-space loads in the text too.
    try std.testing.expect(find(out, "addrspace(4)") == null);
    try std.testing.expect(find(out, "%__5 = load i32, ptr addrspace(2) %__2, align 4") != null);
    // Threadgroup global and barrier untouched.
    try std.testing.expect(find(out, "@my_shader.scratch = private unnamed_addr addrspace(3) global [64 x float] undef, align 4") != null);
    try std.testing.expect(find(out, "tail call void @air.wg.barrier(i32 2, i32 1)") != null);
    // Zig's extern-fn parameter attributes are gone from declares and calls.
    try std.testing.expect(find(out, "signext") == null);
    try std.testing.expect(find(out, "zeroext") == null);
    try std.testing.expect(find(out, "noundef") == null);
    try std.testing.expect(find(out, "declare float @air.simd_shuffle_down.f32(float, i16)") != null);
    try std.testing.expect(find(out, "@air.simd_shuffle_down.f32(float %__8, i16 1)") != null);
    try std.testing.expect(find(out, "declare i32 @air.atomic.global.add.u.i32(ptr addrspace(1) align 4, i32, i32, i32, i1)") != null);
    try std.testing.expect(find(out, "i32 0, i32 2, i1 true)") != null);
    // Every kernel assembles into bitcode with its !air.kernel metadata and packs.
    var functions: [metadata.function_metadata.len]metallib.Function = undefined;
    var n: usize = 0;
    inline for (metadata.function_metadata) |fm| {
        if (fm.stage == .kernel and find(out, "@" ++ fm.name ++ "(") != null) {
            const bc = try assembler.assemble(gpa, out, .{ .entry = fm.name, .fm = fm }, target.default);
            try std.testing.expectEqualStrings("BC\xC0\xDE", bc[0..4]);
            functions[n] = .{ .name = fm.name, .stage = .kernel, .bitcode = bc };
            n += 1;
        }
    }
    try std.testing.expectEqual(4, n);
    const image = try metallib.pack(gpa, functions[0..n], "default.metallib", target.default);
    try std.testing.expectEqualStrings("MTLB", image[0..4]);
}

test "D4 kernels: parameter count and return type are checked against the manifest" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();
    const too_few =
        \\define ptx_kernel void @countKernel(ptr addrspace(1) %0, i32 %1) {
        \\  ret void
        \\}
        \\
    ;
    try std.testing.expectError(error.ParamCountMismatch, convertWith(gpa, too_few, .{ .require_all_entries = false }));
    const not_void =
        \\define ptx_kernel i32 @countKernel(ptr addrspace(1) %0, ptr addrspace(1) %1, i32 %2) {
        \\  ret i32 0
        \\}
        \\
    ;
    try std.testing.expectError(error.KernelReturnNotVoid, convertWith(gpa, not_void, .{ .require_all_entries = false }));
    // Kernels the manifest lists but the IR lacks are an error for the build.
    try std.testing.expectError(error.EntryPointMissing, convert(gpa, "define void @nothing() {\n  ret void\n}\n"));
}

test "D4 kernels: parameter types are checked against the manifest (review-b E2/E4: u16 vs u32 builtins)" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();
    // countKernel's `gid` is `u32` in the manifest; Zig declared `u16`.
    const narrow =
        \\define ptx_kernel void @countKernel(ptr addrspace(1) %0, ptr addrspace(1) %1, i16 zeroext %2) {
        \\  %4 = zext i16 %2 to i64
        \\  ret void
        \\}
        \\
    ;
    try std.testing.expectError(error.ParamTypeMismatch, convertWith(gpa, narrow, .{ .require_all_entries = false }));
    // tgBufKernel's `tidx` is `u16` in the manifest; Zig declared `u32`.
    const wide =
        \\define ptx_kernel void @tgBufKernel(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(3) %2, <3 x i32> %3, i32 %4, i32 %5) {
        \\  %7 = zext i32 %4 to i64
        \\  ret void
        \\}
        \\
    ;
    try std.testing.expectError(error.ParamTypeMismatch, convertWith(gpa, wide, .{ .require_all_entries = false }));
    // A vector builtin declared scalar, and a buffer in the wrong space.
    const scalar_for_vector =
        \\define ptx_kernel void @tgBufKernel(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(3) %2, i32 %3, i16 %4, i32 %5) {
        \\  ret void
        \\}
        \\
    ;
    try std.testing.expectError(error.ParamTypeMismatch, convertWith(gpa, scalar_for_vector, .{ .require_all_entries = false }));
    const wrong_space =
        \\define ptx_kernel void @tgBufKernel(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, <3 x i32> %3, i16 %4, i32 %5) {
        \\  ret void
        \\}
        \\
    ;
    try std.testing.expectError(error.ParamTypeMismatch, convertWith(gpa, wrong_space, .{ .require_all_entries = false }));
    // The matching declaration (Zig's `.param` space 4 for the constant
    // buffer, `zeroext` on the u16) passes and gets the manifest header.
    const good =
        \\define ptx_kernel void @tgBufKernel(ptr addrspace(1) readonly %0, ptr addrspace(1) writeonly %1, ptr addrspace(3) %2, <3 x i32> %3, i16 zeroext %4, i32 noundef %5) {
        \\  %7 = zext i16 %4 to i64
        \\  ret void
        \\}
        \\
        \\define ptx_kernel void @scaleKernel(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(4) %2, i32 %3) {
        \\  ret void
        \\}
        \\
    ;
    const out = try convertWith(gpa, good, .{ .require_all_entries = false });
    try std.testing.expect(find(out, "define void @tgBufKernel(ptr addrspace(1) %__0, ptr addrspace(1) %__1, ptr addrspace(3) %__2, <3 x i32> %__3, i16 %__4, i32 %__5) {") != null);
    try std.testing.expect(find(out, "define void @scaleKernel(ptr addrspace(1) %__0, ptr addrspace(1) %__1, ptr addrspace(2) %__2, i32 %__3) {") != null);
    try std.testing.expectEqualStrings("ptr addrspace(2)", try airParamType(gpa, "ptr addrspace(4)"));
    try std.testing.expectEqualStrings("ptr", try airParamType(gpa, "ptr addrspace(5)"));
    try std.testing.expectEqualStrings("<3 x i32>", try airParamType(gpa, "<3 x i32>"));
}

test "D5.12c texture calls carry Apple's signature: status pair unpacked, sampler retyped" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();

    // Zig binds the colour vector and passes a generic-space sampler; the
    // printed call binds Apple's `{ <colour>, i8 }` pair and unpacks it.
    const sample = try sanitizeBodyLine(gpa, "  %__6 = tail call <4 x float> @air.sample_texture_2d.v4f32(ptr addrspace(1) readonly %__3, ptr nonnull readonly align 1 @sampler.state, <2 x float> %__2, i1 zeroext true, <2 x i32> zeroinitializer, i1 zeroext false, float 0.000000e+00, float 0.000000e+00, i32 0) #4");
    try std.testing.expectEqualStrings(
        \\  %__6.pair = tail call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) readonly %__3, ptr addrspace(2) @sampler.state, <2 x float> %__2, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
        \\  %__6 = extractvalue { <4 x float>, i8 } %__6.pair, 0
    , sample);

    // No status pair: the result name stays, only the return type moves into
    // the constant address space.
    try std.testing.expectEqualStrings(
        "  %__4 = tail call ptr addrspace(2) @air.get_read_sampler()",
        try sanitizeBodyLine(gpa, "  %__4 = tail call ptr @air.get_read_sampler()"),
    );

    // An intrinsic Zig already spells Apple's way, and one that is not a
    // texture intrinsic at all, are both left alone.
    const write = "  tail call void @air.write_texture_2d.v4f32(ptr addrspace(1) %__1, <2 x i32> %__2, <4 x float> %__6, i32 0, i32 2)";
    try std.testing.expectEqualStrings(write, try sanitizeBodyLine(gpa, write));
    const simd = "  %__11 = tail call float @air.simd_sum.f32(float %__10)";
    try std.testing.expectEqualStrings(simd, try sanitizeBodyLine(gpa, simd));

    try std.testing.expectEqualStrings("air.sample_texture_2d.v4f32", declaredName("declare <4 x float> @air.sample_texture_2d.v4f32(ptr addrspace(1) readonly, ptr) local_unnamed_addr").?);
    try std.testing.expectEqualStrings("llvm.trap", declaredName("declare void @llvm.trap()").?);

    // The sampler word is printed in the address space the intrinsic reads it
    // from, and named for `!air.sampler_states`.
    const samplers = try samplerGlobals(gpa, "@s.state = private unnamed_addr addrspace(2) constant [2 x i64] [i64 1, i64 0], align 8\n@other = private unnamed_addr constant [4 x i8] c\"abc\\00\", align 1\n");
    try std.testing.expectEqual(@as(usize, 1), samplers.len);
    try std.testing.expectEqualStrings("s.state", samplers[0]);
}

test "air-splice command line: options, targets, and a misplaced flag" {
    var bad: []const u8 = "";
    const plain = try parseCli(&.{ "in.ll", "out.metallib" }, &bad);
    try std.testing.expectEqual(target.Name.macos26, plain.profile.name);
    try std.testing.expectEqualStrings("in.ll", plain.input);
    try std.testing.expectEqualStrings("out.metallib", plain.output);
    try std.testing.expect(plain.text_output == null);
    const full = try parseCli(&.{ "--target=macos26", "in.ll", "out.metallib", "out.ll" }, &bad);
    try std.testing.expectEqualStrings("out.ll", full.text_output.?);
    // Unverified targets are refused unless explicitly allowed.
    try std.testing.expectError(error.UnverifiedTarget, parseCli(&.{ "--target=macos15", "in.ll", "out.metallib" }, &bad));
    try std.testing.expectEqualStrings("macos15", bad);
    const allowed = try parseCli(&.{ "--target=macos15", "--allow-unverified", "in.ll", "out.metallib" }, &bad);
    try std.testing.expectEqual(target.Name.macos15, allowed.profile.name);
    const allowed_first = try parseCli(&.{ "--allow-unverified", "--target=macos13", "in.ll", "out.metallib" }, &bad);
    try std.testing.expectEqual(target.Name.macos13, allowed_first.profile.name);
    // A flag after a file would otherwise be taken as a file name.
    try std.testing.expectError(error.OptionAfterFile, parseCli(&.{ "in.ll", "--target=macos15", "out.metallib" }, &bad));
    try std.testing.expectEqualStrings("--target=macos15", bad);
    try std.testing.expectError(error.OptionAfterFile, parseCli(&.{ "in.ll", "out.metallib", "--allow-unverified" }, &bad));
    try std.testing.expectError(error.OptionAfterFile, parseCli(&.{ "in.ll", "-target=macos15", "out.metallib" }, &bad));
    try std.testing.expectEqualStrings("-target=macos15", bad);
    // A bare `--` is not a known option.
    try std.testing.expectError(error.UnknownOption, parseCli(&.{ "--", "in.ll", "out.metallib" }, &bad));
    // A repeated --target: the last one wins.
    const repeated = try parseCli(&.{ "--target=macos15", "--target=macos26", "in.ll", "out.metallib" }, &bad);
    try std.testing.expectEqual(target.Name.macos26, repeated.profile.name);
    try std.testing.expectError(error.WrongArgCount, parseCli(&.{ "in.ll", "out.metallib", "out.ll", "extra" }, &bad));
    try std.testing.expectError(error.UnknownTarget, parseCli(&.{ "--target=macos99", "in.ll", "out.metallib" }, &bad));
    try std.testing.expectError(error.UnknownTarget, parseCli(&.{ "--target=", "in.ll", "out.metallib" }, &bad));
    try std.testing.expectError(error.UnknownOption, parseCli(&.{ "--frobnicate", "in.ll", "out.metallib" }, &bad));
    try std.testing.expectError(error.WrongArgCount, parseCli(&.{"in.ll"}, &bad));
    try std.testing.expectError(error.WrongArgCount, parseCli(&.{}, &bad));
}

test "the printed IR carries the chosen deployment target's triple" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const src = "target datalayout = \"e-p:64:64\"\ntarget triple = \"nvptx64-nvidia-cuda\"\n";
    const out15 = try convertWith(arena.allocator(), src, .{ .require_all_entries = false, .profile = target.get(.macos15) });
    try std.testing.expect(find(out15, "target triple = \"air64_v27-apple-macosx15.0.0\"") != null);
    try std.testing.expect(find(out15, "air64_v28") == null);
    const out26 = try convertWith(arena.allocator(), src, .{ .require_all_entries = false });
    try std.testing.expect(find(out26, "target triple = \"air64_v28-apple-macosx26.0.0\"") != null);
}
