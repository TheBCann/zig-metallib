//! The `!air.*` metadata Apple's driver needs, derived at comptime from the
//! shader manifest (`shader.functions`) and the Zig types it names.
//!
//! The tree is built once as `Node` values. Two consumers walk it:
//!   * `printModule` renders LLVM IR text (for the debug `.ll` output), and
//!   * `lowerModule` builds the same nodes with `std.zig.llvm.Builder` for
//!     the bitcode path.

const std = @import("std");
const shader = @import("shader");
const target = @import("target.zig");
pub const air = shader.air;
const Builder = std.zig.llvm.Builder;
const comptimePrint = std.fmt.comptimePrint;

pub const Node = union(enum) {
    str: []const u8,
    int: i32,
    /// Reference to the entry point the node describes.
    func,
    tuple: []const Node,
};

/// Everything a single entry point's module needs.
pub const FunctionMetadata = struct {
    name: [:0]const u8,
    stage: air.Stage,
    /// `!{ptr @name, outputs, inputs}` — the element of `!air.vertex` / `!air.fragment`.
    node: Node,
    /// Per IR parameter (same numbering as the `i32 <param>` of the input
    /// nodes): which threads of a threadgroup see the same value. Seeds
    /// the assembler's uniformity analysis (tools/air/divergence.zig).
    param_uniformity: []const air.Uniformity,
};

pub const function_metadata: [shader.functions.len]FunctionMetadata = blk: {
    var out: [shader.functions.len]FunctionMetadata = undefined;
    for (shader.functions, 0..) |f, idx| out[idx] = functionMetadata(f);
    break :blk out;
};

/// Metadata for one manifest entry. Tests build their own manifests with
/// it so they do not depend on what `shader.functions` currently lists.
pub fn functionMetadata(comptime f: air.Function) FunctionMetadata {
    return .{ .name = f.name, .stage = f.stage, .node = functionNode(f), .param_uniformity = paramUniformity(f) };
}

/// One entry per IR parameter, in `inputsNode` order: buffers and textures
/// (pointers to memory every thread shares) are threadgroup-uniform, the
/// builtins follow `air.BuiltinKind.uniformity`; `[[vertex_id]]` and every
/// `stage_in` field are per-thread.
pub fn paramUniformity(comptime f: air.Function) []const air.Uniformity {
    var out: []const air.Uniformity = &.{};
    for (f.args) |arg| {
        switch (arg) {
            .vertex_id => out = out ++ &[_]air.Uniformity{.thread},
            .builtin => |bi| out = out ++ &[_]air.Uniformity{bi.kind.uniformity()},
            .stage_in => |T| {
                for (@typeInfo(T).@"struct".field_names) |name| {
                    if (isFragmentInput(name)) out = out ++ &[_]air.Uniformity{.thread};
                }
            },
            else => out = out ++ &[_]air.Uniformity{.threadgroup},
        }
    }
    return out;
}

// ── Fixed module-level metadata ──────────────────────────────────────────────

pub const NamedList = struct { name: []const u8, nodes: []const Node };

/// The module-level named lists for the deployment target `p`, in the order
/// Apple emits them. `air.version`, `air.language_version` and the presence of
/// the frame-pointer flag depend on the target; the limits and compile options
/// are the same for every profile (target.zig). The order is part of the
/// output: it decides the metadata numbering, so moving a list changes the
/// bitcode. Callers pick the profile by name from target.profiles (every
/// Profile in this project comes from that table).
pub fn moduleLists(comptime p: target.Profile) [5]NamedList {
    return .{
        .{
            .name = "llvm.module.flags",
            // Metal-specific limits, copied from metalfe-32023.883 output. Apple
            // leaves out the frame-pointer flag for macOS 13 (target.zig).
            .nodes = &([_]Node{t(&.{ .{ .int = 1 }, .{ .str = "wchar_size" }, .{ .int = 4 } })} ++
                (if (p.frame_pointer_flag) [_]Node{t(&.{ .{ .int = 7 }, .{ .str = "frame-pointer" }, .{ .int = 2 } })} else [_]Node{}) ++
                [_]Node{
                    t(&.{ .{ .int = 7 }, .{ .str = "air.max_device_buffers" }, .{ .int = 31 } }),
                    t(&.{ .{ .int = 7 }, .{ .str = "air.max_constant_buffers" }, .{ .int = 31 } }),
                    t(&.{ .{ .int = 7 }, .{ .str = "air.max_threadgroup_buffers" }, .{ .int = 31 } }),
                    t(&.{ .{ .int = 7 }, .{ .str = "air.max_textures" }, .{ .int = 128 } }),
                    t(&.{ .{ .int = 7 }, .{ .str = "air.max_read_write_textures" }, .{ .int = 8 } }),
                    t(&.{ .{ .int = 7 }, .{ .str = "air.max_samplers" }, .{ .int = 16 } }),
                }),
        },
        .{ .name = "llvm.ident", .nodes = &.{t(&.{.{ .str = "zig air-splice" }})} },
        .{ .name = "air.version", .nodes = &.{t(&.{ .{ .int = 2 }, .{ .int = p.air_minor }, .{ .int = 0 } })} },
        .{ .name = "air.language_version", .nodes = &.{t(&.{ .{ .str = "Metal" }, .{ .int = p.lang_major }, .{ .int = p.lang_minor }, .{ .int = 0 } })} },
        .{ .name = "air.compile_options", .nodes = &.{
            t(&.{.{ .str = "air.compile.denorms_disable" }}),
            t(&.{.{ .str = "air.compile.fast_math_enable" }}),
            t(&.{.{ .str = "air.compile.framebuffer_fetch_enable" }}),
        } },
    };
}

pub fn stageListName(stage: air.Stage) []const u8 {
    return switch (stage) {
        .vertex => "air.vertex",
        .fragment => "air.fragment",
        .kernel => "air.kernel",
    };
}

/// Order in which the per-stage named lists are printed.
const stage_order = [_]air.Stage{ .vertex, .fragment, .kernel };

fn t(comptime nodes: []const Node) Node {
    return .{ .tuple = nodes };
}

fn s(comptime text: []const u8) Node {
    return .{ .str = text };
}

fn i(comptime v: i32) Node {
    return .{ .int = v };
}

// ── Per-function nodes ──────────────────────────────────────────────────────

/// Vertex/fragment: `!{ptr @fn, !outputs, !inputs}`.
/// Kernel: `!{ptr @fn, !{}, !inputs[, !{!"air.max_work_group_size", i32 N}]}`
/// (research/compute-kernels/k1.air.ll; the empty tuple is the outputs slot).
fn functionNode(comptime f: air.Function) Node {
    if (f.stage == .kernel) {
        if (f.ret != void) @compileError("kernel '" ++ f.name ++ "' must have ret = void");
        if (f.max_total_threads_per_threadgroup) |n| {
            return t(&.{ .func, t(&.{}), inputsNode(f), t(&.{ s("air.max_work_group_size"), i(@intCast(n)) }) });
        }
        return t(&.{ .func, t(&.{}), inputsNode(f) });
    }
    if (f.max_total_threads_per_threadgroup != null) @compileError("max_total_threads_per_threadgroup is kernel-only: " ++ f.name);
    return t(&.{ .func, outputsNode(f), inputsNode(f) });
}

/// Vertex outputs, one node per field of the output struct in declaration
/// order (== the packed struct's element order, research/_results/r01.txt
/// metadata_patterns 7-8, s1_uniforms_builtins.metallib.ll lines 64-68):
///   * `position`   -> `!{!"air.position", !"air.arg_type_name", !"float4", !"air.arg_name", !"position"}`
///   * `point_size` -> `!{!"air.point_size", !"air.arg_type_name", !"float", !"air.arg_name", !"point_size"}`
///                     (no generated() name; the field must be `f32`)
///   * anything else -> `!{!"air.vertex_output", !"generated(<mangled>)", !"air.arg_type_name", ..., !"air.arg_name", ...}`
/// Fragment outputs (metadata_patterns 12-13, s2_mrt_depth.metallib.ll
/// lines 54-56): a vector return is one `air.render_target 0 0` without an
/// arg name (the project's original form); a struct return gets one node
/// per field in declaration order, `colorN` ->
/// `!{!"air.render_target", i32 N, i32 0, !"air.arg_type_name", !"float4"|"half4"|"uint4", !"air.arg_name", !"colorN"}`
/// and `depth` -> `!{!"air.depth", !"air.depth_qualifier", !"air.any", !"air.arg_type_name", !"float", !"air.arg_name", !"depth"}`.
fn outputsNode(comptime f: air.Function) Node {
    switch (f.stage) {
        .vertex => {
            const st = @typeInfo(f.ret).@"struct";
            var items: []const Node = &.{};
            for (st.field_names, st.field_types) |name, ft| {
                const node = if (std.mem.eql(u8, name, "position"))
                    t(&.{ s("air.position"), s("air.arg_type_name"), s(mslType(ft)), s("air.arg_name"), s("position") })
                else if (std.mem.eql(u8, name, "point_size")) blk: {
                    if (ft != f32) @compileError("vertex output '" ++ f.name ++ "': the point_size field must be f32, got " ++ @typeName(ft));
                    break :blk t(&.{ s("air.point_size"), s("air.arg_type_name"), s("float"), s("air.arg_name"), s("point_size") });
                } else t(&.{ s("air.vertex_output"), s(varyingName(name, ft)), s("air.arg_type_name"), s(mslType(ft)), s("air.arg_name"), s(name) });
                items = items ++ &[_]Node{node};
            }
            return t(items);
        },
        .fragment => {
            if (@typeInfo(f.ret) != .@"struct") return t(&.{
                t(&.{ s("air.render_target"), i(0), i(0), s("air.arg_type_name"), s(mslType(f.ret)) }),
            });
            const st = @typeInfo(f.ret).@"struct";
            var items: []const Node = &.{};
            for (st.field_names, st.field_types) |name, ft| {
                const node = if (std.mem.eql(u8, name, "depth")) blk: {
                    if (ft != f32) @compileError("fragment output '" ++ f.name ++ "': the depth field must be f32, got " ++ @typeName(ft));
                    break :blk t(&.{ s("air.depth"), s("air.depth_qualifier"), s("air.any"), s("air.arg_type_name"), s("float"), s("air.arg_name"), s("depth") });
                } else blk: {
                    const idx = renderTargetIndex(name) orelse @compileError("fragment output '" ++ f.name ++ "': field '" ++ name ++ "' must be named color<N> or depth");
                    if (@typeInfo(ft) != .vector) @compileError("fragment output '" ++ f.name ++ "': " ++ name ++ " must be a vector, got " ++ @typeName(ft));
                    break :blk t(&.{ s("air.render_target"), i(idx), i(0), s("air.arg_type_name"), s(mslType(ft)), s("air.arg_name"), s(name) });
                };
                items = items ++ &[_]Node{node};
            }
            return t(items);
        },
        .kernel => return t(&.{}),
    }
}

/// `color0` -> 0, `color7` -> 7; null for any other name.
pub fn renderTargetIndex(comptime name: []const u8) ?i32 {
    if (!std.mem.startsWith(u8, name, "color") or name.len == "color".len) return null;
    return std.fmt.parseInt(i32, name["color".len..], 10) catch null;
}

/// Whether a field of the vertex output struct becomes a fragment
/// parameter: every field except `point_size`, which only the rasterizer
/// reads (research/_results/r01.txt ir_patterns 2).
pub fn isFragmentInput(comptime name: []const u8) bool {
    return !std.mem.eql(u8, name, "point_size");
}

/// Integer scalars and vectors are always `[[flat]]` varyings: the Metal
/// compiler emits `air.flat` for them even without the qualifier
/// (research/_results/r01.txt finding 10, s7_int_noflat.metallib.ll line 94).
pub fn isFlatVarying(comptime T: type) bool {
    return switch (@typeInfo(T)) {
        .int => true,
        .vector => |v| @typeInfo(v.child) == .int,
        else => false,
    };
}

fn inputsNode(comptime f: air.Function) Node {
    var items: []const Node = &.{};
    var param: i32 = 0;
    for (f.args) |arg| {
        switch (arg) {
            .vertex_id => |name| {
                items = items ++ &[_]Node{builtinNode(param, .{ .kind = .vertex_id, .T = u32, .name = name })};
                param += 1;
            },
            .builtin => |bi| {
                items = items ++ &[_]Node{builtinNode(param, bi)};
                param += 1;
            },
            .buffer => |b| {
                items = items ++ &[_]Node{bufferNode(param, b)};
                param += 1;
            },
            // One parameter per vertex-output field except `point_size`.
            // `position` is `air.center, air.no_perspective`; integer
            // fields are `air.flat` (research/_results/r01.txt
            // metadata_patterns 9, 17); the rest `air.center, air.perspective`.
            .stage_in => |T| {
                const st = @typeInfo(T).@"struct";
                for (st.field_names, st.field_types) |name, ft| {
                    if (!isFragmentInput(name)) continue;
                    const node = if (std.mem.eql(u8, name, "position"))
                        t(&.{ i(param), s("air.position"), s("air.center"), s("air.no_perspective"), s("air.arg_type_name"), s(mslType(ft)), s("air.arg_name"), s("position") })
                    else if (isFlatVarying(ft))
                        t(&.{ i(param), s("air.fragment_input"), s(varyingName(name, ft)), s("air.flat"), s("air.arg_type_name"), s(mslType(ft)), s("air.arg_name"), s(name) })
                    else
                        t(&.{ i(param), s("air.fragment_input"), s(varyingName(name, ft)), s("air.center"), s("air.perspective"), s("air.arg_type_name"), s(mslType(ft)), s("air.arg_name"), s(name) });
                    items = items ++ &[_]Node{node};
                    param += 1;
                }
            },
            .texture => |tex| {
                items = items ++ &[_]Node{textureNode(param, tex)};
                param += 1;
            },
            .sampler => |sm| {
                items = items ++ &[_]Node{samplerNode(param, sm)};
                param += 1;
            },
        }
    }
    return t(items);
}

/// One texture argument (research/_results/r03.json metadata_patterns 0-1,
/// texture-sampling/argmeta.txt): `!{i32 <param>, !"air.texture",
/// !"air.location_index", i32 <index>, i32 1, !"air.sample"|"air.read"|
/// "air.write"|"air.read_write", !"air.arg_type_name", !"texture2d<float,
/// <access>>", !"air.arg_name", !"<name>"}`. The access string must agree
/// with the `i32 access` the shader passes to the intrinsics; both come
/// from `T.texture_access` (gpu.Texture2D).
fn textureNode(comptime param: i32, comptime tex: air.Texture) Node {
    if (!@hasDecl(tex.T, "texture_access")) @compileError("texture '" ++ tex.name ++ "': T must be gpu.Texture2D(access), got " ++ @typeName(tex.T));
    const access: air.TextureAccess = tex.T.texture_access;
    return t(&.{
        i(param),                s("air.texture"),
        s("air.location_index"), i(@intCast(tex.index)),
        i(1),                    s("air." ++ @tagName(access)),
        s("air.arg_type_name"),  s("texture2d<float, " ++ @tagName(access) ++ ">"),
        s("air.arg_name"),       s(tex.name),
    });
}

/// One `[[sampler(n)]]` argument (r03 metadata_patterns 2): `!{i32 <param>,
/// !"air.sampler", !"air.location_index", i32 <index>, i32 1,
/// !"air.arg_type_name", !"sampler", !"air.arg_name", !"<name>"}`; no
/// access string. The parameter is `ptr addrspace(2)`.
fn samplerNode(comptime param: i32, comptime sm: air.Sampler) Node {
    return t(&.{
        i(param),                s("air.sampler"),
        s("air.location_index"), i(@intCast(sm.index)),
        i(1),                    s("air.arg_type_name"),
        s("sampler"),            s("air.arg_name"),
        s(sm.name),
    });
}

/// One buffer argument, as metalfe describes `device const T*` / `device T*` /
/// `constant T*` / `threadgroup T*` (research/compute-kernels/k1.air.ll):
/// `!{i32 <param>, !"air.buffer", !"air.location_index", i32 <index>, i32 1,
///   !"air.read"|!"air.read_write", !"air.address_space", i32 1|2|3,
///   [!"air.struct_type_info", !N   only for struct element types],
///   !"air.arg_type_size", i32 @sizeOf(T), !"air.arg_type_align_size", i32 @alignOf(T),
///   !"air.arg_type_name", !"<msl name>", !"air.arg_name", !"<name>"}`.
/// Scalar and vector element types carry no struct_type_info and are named
/// with their MSL spelling (`float`, `uint2`, ...); structs by their Zig name.
fn bufferNode(comptime param: i32, comptime b: air.Buffer) Node {
    const is_struct = @typeInfo(b.T) == .@"struct";
    if (b.space == .constant and b.access == .read_write) @compileError("constant buffer '" ++ b.name ++ "' cannot be read_write");
    var items: []const Node = &.{
        i(param),                s("air.buffer"),
        s("air.location_index"), i(@intCast(b.index)),
        i(1),
        s(switch (b.access) {
            .read => "air.read",
            .read_write => "air.read_write",
        }),
        s("air.address_space"),  i(@backingInt(b.space)),
    };
    if (is_struct) items = items ++ &[_]Node{ s("air.struct_type_info"), structTypeInfo(b.T) };
    items = items ++ &[_]Node{
        s("air.arg_type_size"),       i(@sizeOf(b.T)),
        s("air.arg_type_align_size"), i(@alignOf(b.T)),
        s("air.arg_type_name"),       s(if (is_struct) shortTypeName(b.T) else mslType(b.T)),
        s("air.arg_name"),            s(b.name),
    };
    return t(items);
}

/// One built-in input: `!{i32 <param>, !"air.<kind>", !"air.arg_type_name",
/// !"uint"|"uint2"|"uint3"|"ushort"|"bool"|"float2", !"air.arg_name", !"<name>"}`
/// (research/compute-kernels/k1.air.ll, k2.air.ll; the vertex/fragment kinds
/// have the same shape in research/_results/r01.txt).
fn builtinNode(comptime param: i32, comptime bi: air.Builtin) Node {
    return t(&.{ i(param), s("air." ++ @tagName(bi.kind)), s("air.arg_type_name"), s(mslType(bi.T)), s("air.arg_name"), s(bi.name) });
}

/// `air.struct_type_info`: `offset, size, 0, "type", "name"` per field.
fn structTypeInfo(comptime T: type) Node {
    const st = @typeInfo(T).@"struct";
    var items: []const Node = &.{};
    for (st.field_names, st.field_types) |name, ft| {
        items = items ++ &[_]Node{ i(@offsetOf(T, name)), i(@sizeOf(ft)), i(0), s(mslType(ft)), s(name) };
    }
    return t(items);
}

// ── Type naming ─────────────────────────────────────────────────────────────

/// Metal Shading Language spelling of a type, used in reflection metadata.
/// An array of float vectors is a matrix: `[4]@Vector(4, f32)` (four
/// `float4` columns, 64 bytes) spells `float4x4`, as metalfe names the
/// field in `air.struct_type_info` (research/_results/r01.txt
/// metadata_patterns 5, s1_uniforms_builtins.metallib.ll line 77).
pub fn mslType(comptime T: type) []const u8 {
    return switch (@typeInfo(T)) {
        .array => |a| switch (@typeInfo(a.child)) {
            .vector => |v| if (@typeInfo(v.child) == .float)
                comptimePrint("{s}{d}x{d}", .{ mslType(v.child), a.len, v.len })
            else
                @compileError("unsupported shader type " ++ @typeName(T) ++ " (only float matrices spell as arrays of vectors)"),
            else => @compileError("unsupported shader type " ++ @typeName(T)),
        },
        .vector => |v| comptimePrint("{s}{d}", .{ mslType(v.child), v.len }),
        .float => |f| switch (f.bits) {
            16 => "half",
            32 => "float",
            else => @compileError("unsupported float width in shader type " ++ @typeName(T)),
        },
        .int => |n| switch (n.bits) {
            32 => if (n.signedness == .unsigned) "uint" else "int",
            16 => if (n.signedness == .unsigned) "ushort" else "short",
            8 => if (n.signedness == .unsigned) "uchar" else "char",
            else => @compileError("unsupported int width in shader type " ++ @typeName(T)),
        },
        .bool => "bool",
        else => @compileError("unsupported shader type " ++ @typeName(T)),
    };
}

/// `my_shader.VertexIn` -> `VertexIn`
pub fn shortTypeName(comptime T: type) []const u8 {
    const full = @typeName(T);
    var start: usize = 0;
    for (full, 0..) |c, idx| {
        if (c == '.') start = idx + 1;
    }
    return full[start..];
}

/// Apple's name for a varying: `generated(6normalDv3_f)`, i.e. Itanium-
/// mangled `normal` + `Dv3_f` (vector of 3 float); scalars mangle as the
/// bare element code, `generated(6flatIdi)` / `generated(2idj)` /
/// `generated(1af)` / `generated(1dDh)` for int / uint / float / half
/// (research/_results/r01.txt metadata_patterns 8, s6_interp.metallib.ll
/// lines 52-57).
pub fn varyingName(comptime name: []const u8, comptime T: type) []const u8 {
    return switch (@typeInfo(T)) {
        .vector => |v| comptimePrint("generated({d}{s}Dv{d}_{s})", .{ name.len, name, v.len, mangle(v.child) }),
        .float, .int => comptimePrint("generated({d}{s}{s})", .{ name.len, name, mangle(T) }),
        else => @compileError("unsupported varying type " ++ @typeName(T)),
    };
}

/// Itanium mangling of a scalar varying element: i (int), j (uint), f
/// (float), Dh (half).
fn mangle(comptime T: type) []const u8 {
    return switch (@typeInfo(T)) {
        .float => |f| switch (f.bits) {
            16 => "Dh",
            32 => "f",
            else => @compileError("unsupported varying element type " ++ @typeName(T)),
        },
        .int => |n| switch (n.bits) {
            32 => if (n.signedness == .unsigned) "j" else "i",
            else => @compileError("unsupported varying element type " ++ @typeName(T)),
        },
        else => @compileError("unsupported varying element type " ++ @typeName(T)),
    };
}

// ── Consumer 1: LLVM IR text ────────────────────────────────────────────────

/// Append the metadata block for a whole module (all entry points) as text.
/// `samplers` names the constexpr sampler globals (`@name`, already in
/// address space 2 in the text) to list under `!air.sampler_states`.
pub fn printModule(gpa: std.mem.Allocator, out: *std.ArrayList(u8), samplers: []const []const u8, profile: target.Profile) !void {
    return printManifestWith(gpa, out, &function_metadata, samplers, profile);
}

/// `printModule` for an explicit list of entry points (tests use their own),
/// for the default deployment target.
pub fn printManifest(gpa: std.mem.Allocator, out: *std.ArrayList(u8), comptime fms: []const FunctionMetadata) !void {
    return printManifestWith(gpa, out, fms, &.{}, target.default);
}

/// `!air.sampler_states = !{!N, ...}` with `!N = !{!"air.sampler_state",
/// ptr addrspace(2) @<global>}` per constexpr sampler global: required
/// whenever a module samples through one (r03 metadata_patterns 3; leaving
/// it out crashes Metal's backend at pipeline creation, h04_no_sampler_md).
/// Same rule for every stage.
pub const sampler_states_list = "air.sampler_states";
pub const sampler_state_tag = "air.sampler_state";

/// The module-level lists of `profile` as text. `moduleLists` needs the
/// profile at comptime, so this picks the matching one of the known profiles.
fn printModuleLists(gpa: std.mem.Allocator, nodes: *std.ArrayList(u8), named: *std.ArrayList(u8), next: *u32, profile: target.Profile) !void {
    inline for (target.profiles) |p| {
        if (p.name == profile.name) {
            inline for (comptime moduleLists(p)) |list| {
                var refs: std.ArrayList(u8) = .empty;
                inline for (list.nodes) |node| {
                    const id = try printNode(gpa, nodes, next, node, "");
                    try appendRef(gpa, &refs, id);
                }
                try named.print(gpa, "!{s} = !{{{s}}}\n", .{ list.name, refs.items });
            }
        }
    }
}

pub fn printManifestWith(gpa: std.mem.Allocator, out: *std.ArrayList(u8), comptime fms: []const FunctionMetadata, samplers: []const []const u8, profile: target.Profile) !void {
    var next: u32 = 0;
    var nodes: std.ArrayList(u8) = .empty;
    var named: std.ArrayList(u8) = .empty;

    try printModuleLists(gpa, &nodes, &named, &next, profile);
    inline for (stage_order) |stage| {
        var refs: std.ArrayList(u8) = .empty;
        inline for (fms) |fm| {
            if (fm.stage == stage) {
                const id = try printNode(gpa, &nodes, &next, fm.node, fm.name);
                try appendRef(gpa, &refs, id);
            }
        }
        if (refs.items.len > 0) try named.print(gpa, "!{s} = !{{{s}}}\n", .{ stageListName(stage), refs.items });
    }
    if (samplers.len > 0) {
        var refs: std.ArrayList(u8) = .empty;
        for (samplers) |name| {
            const id = next;
            next += 1;
            try nodes.print(gpa, "!{d} = !{{!\"{s}\", ptr addrspace(2) @{s}}}\n", .{ id, sampler_state_tag, name });
            try appendRef(gpa, &refs, id);
        }
        try named.print(gpa, "!{s} = !{{{s}}}\n", .{ sampler_states_list, refs.items });
    }

    try out.append(gpa, '\n');
    try out.appendSlice(gpa, named.items);
    try out.append(gpa, '\n');
    try out.appendSlice(gpa, nodes.items);
}

fn appendRef(gpa: std.mem.Allocator, refs: *std.ArrayList(u8), id: u32) !void {
    if (refs.items.len > 0) try refs.appendSlice(gpa, ", ");
    try refs.print(gpa, "!{d}", .{id});
}

/// Post-order: children get ids before their parent. Returns the node's id.
fn printNode(gpa: std.mem.Allocator, nodes: *std.ArrayList(u8), next: *u32, comptime node: Node, fn_name: []const u8) !u32 {
    switch (node) {
        .tuple => |elems| {
            var body: std.ArrayList(u8) = .empty;
            inline for (elems) |e| {
                if (body.items.len > 0) try body.appendSlice(gpa, ", ");
                switch (e) {
                    .str => |text| try body.print(gpa, "!\"{s}\"", .{text}),
                    .int => |v| try body.print(gpa, "i32 {d}", .{v}),
                    .func => try body.print(gpa, "ptr @{s}", .{fn_name}),
                    .tuple => {
                        const id = try printNode(gpa, nodes, next, e, fn_name);
                        try body.print(gpa, "!{d}", .{id});
                    },
                }
            }
            const id = next.*;
            next.* += 1;
            try nodes.print(gpa, "!{d} = !{{{s}}}\n", .{ id, body.items });
            return id;
        },
        else => @compileError("printNode expects a tuple at the top level"),
    }
}

// ── Consumer 2: std.zig.llvm.Builder ────────────────────────────────────────

/// Add the module-level lists plus the entry-point list for one function,
/// and `!air.sampler_states` for the constexpr sampler globals the
/// assembler saw as sampler operands (`samplers`; empty for a module that
/// samples through bound samplers only).
pub fn lowerModule(b: *Builder, comptime fm: FunctionMetadata, func: Builder.Function.Index, samplers: []const Builder.Global.Index, profile: target.Profile) !void {
    // `moduleLists` needs the profile at comptime: pick the matching one of
    // the few known profiles (only these lists are instantiated per profile,
    // not the entry-point node below).
    inline for (target.profiles) |p| {
        if (p.name == profile.name) {
            inline for (comptime moduleLists(p)) |list| {
                var operands: [list.nodes.len]Builder.Metadata = undefined;
                inline for (list.nodes, 0..) |node, idx| operands[idx] = try lowerNode(b, node, func);
                try b.addNamedMetadata(try b.string(list.name), &operands);
            }
        }
    }
    const fn_node = try lowerNode(b, fm.node, func);
    try b.addNamedMetadata(try b.string(stageListName(fm.stage)), &.{fn_node});
    if (samplers.len > 0) {
        const nodes = try b.gpa.alloc(Builder.Metadata, samplers.len);
        defer b.gpa.free(nodes);
        const tag = (try b.metadataString(sampler_state_tag)).toMetadata();
        for (samplers, nodes) |g, *node| node.* = try b.metadataTuple(&.{ tag, try b.metadataConstant(g.toConst()) });
        try b.addNamedMetadata(try b.string(sampler_states_list), nodes);
    }
}

fn lowerNode(b: *Builder, comptime node: Node, func: Builder.Function.Index) !Builder.Metadata {
    switch (node) {
        .str => |text| return (try b.metadataString(text)).toMetadata(),
        .int => |v| return b.metadataConstant(try b.intConst(.i32, v)),
        .func => return b.metadataConstant(func.toConst(b)),
        .tuple => |elems| {
            var operands: [elems.len]Builder.Metadata = undefined;
            inline for (elems, 0..) |e, idx| operands[idx] = try lowerNode(b, e, func);
            return b.metadataTuple(&operands);
        },
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

const TestParams = extern struct { count: u32, scale: f32 };

const test_kernel_fm = functionMetadata(.{
    .name = "scaleKernel",
    .stage = .kernel,
    .ret = void,
    .args = &.{
        .{ .buffer = .{ .index = 0, .T = f32, .name = "in" } },
        .{ .buffer = .{ .index = 1, .T = f32, .name = "out", .access = .read_write } },
        .{ .buffer = .{ .index = 2, .T = TestParams, .name = "params", .space = .constant } },
        .{ .buffer = .{ .index = 0, .T = @Vector(2, u32), .name = "tile", .space = .threadgroup, .access = .read_write } },
        .{ .builtin = .{ .kind = .thread_position_in_grid, .T = u32, .name = "gid" } },
        .{ .builtin = .{ .kind = .threads_per_grid, .T = @Vector(3, u32), .name = "grid" } },
        .{ .builtin = .{ .kind = .thread_index_in_threadgroup, .T = u16, .name = "tidx" } },
    },
});

const test_kernel_max_fm = functionMetadata(.{
    .name = "countKernel",
    .stage = .kernel,
    .ret = void,
    .max_total_threads_per_threadgroup = 128,
    .args = &.{
        .{ .buffer = .{ .index = 0, .T = u32, .name = "counter", .access = .read_write } },
        .{ .builtin = .{ .kind = .thread_position_in_grid, .T = u32, .name = "gid" } },
    },
});

fn has(hay: []const u8, needle: []const u8) bool {
    return std.mem.find(u8, hay, needle) != null;
}

test {
    // The sampler-state encoder's tests live next to it in gpu.zig.
    _ = shader.gpu;
}

test "kernel manifest prints an !air.kernel list with buffer, builtin and max_work_group_size nodes" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();
    var out: std.ArrayList(u8) = .empty;
    try printManifest(gpa, &out, &.{ test_kernel_fm, test_kernel_max_fm });
    const text = out.items;

    // Named list: one kernel node per entry, no vertex/fragment lists.
    try std.testing.expect(has(text, "!air.kernel = !{!24, !30}\n"));
    try std.testing.expect(!has(text, "!air.vertex"));
    try std.testing.expect(!has(text, "!air.fragment"));
    // Kernel node: {fn, empty outputs, inputs}; the outputs tuple is `!{}`
    // (post-order ids: children before the node).
    try std.testing.expect(has(text, "!14 = !{}\n"));
    try std.testing.expect(has(text, "!23 = !{!15, !16, !18, !19, !20, !21, !22}\n"));
    try std.testing.expect(has(text, "!24 = !{ptr @scaleKernel, !14, !23}\n"));
    // Device buffers: read / read_write, address space 1, no struct_type_info, MSL scalar name.
    try std.testing.expect(has(text, "!{i32 0, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, !\"air.read\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"float\", !\"air.arg_name\", !\"in\"}\n"));
    try std.testing.expect(has(text, "!{i32 1, !\"air.buffer\", !\"air.location_index\", i32 1, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"float\", !\"air.arg_name\", !\"out\"}\n"));
    // Constant struct buffer: address space 2 with struct_type_info and the Zig type name.
    try std.testing.expect(has(text, "!{i32 2, !\"air.buffer\", !\"air.location_index\", i32 2, i32 1, !\"air.read\", !\"air.address_space\", i32 2, !\"air.struct_type_info\", !"));
    try std.testing.expect(has(text, "!\"air.arg_type_size\", i32 8, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"TestParams\", !\"air.arg_name\", !\"params\"}\n"));
    try std.testing.expect(has(text, "= !{i32 0, i32 4, i32 0, !\"uint\", !\"count\", i32 4, i32 4, i32 0, !\"float\", !\"scale\"}\n"));
    // Threadgroup vector buffer: address space 3, `uint2`.
    try std.testing.expect(has(text, "!{i32 3, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, !\"air.read_write\", !\"air.address_space\", i32 3, !\"air.arg_type_size\", i32 8, !\"air.arg_type_align_size\", i32 8, !\"air.arg_type_name\", !\"uint2\", !\"air.arg_name\", !\"tile\"}\n"));
    // Builtins: uint / uint3 / ushort.
    try std.testing.expect(has(text, "!{i32 4, !\"air.thread_position_in_grid\", !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"gid\"}\n"));
    try std.testing.expect(has(text, "!{i32 5, !\"air.threads_per_grid\", !\"air.arg_type_name\", !\"uint3\", !\"air.arg_name\", !\"grid\"}\n"));
    try std.testing.expect(has(text, "!{i32 6, !\"air.thread_index_in_threadgroup\", !\"air.arg_type_name\", !\"ushort\", !\"air.arg_name\", !\"tidx\"}\n"));
    // max_total_threads_per_threadgroup: a fourth element on the kernel node.
    try std.testing.expect(has(text, "!25 = !{}\n"));
    try std.testing.expect(has(text, "!26 = !{i32 0, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"counter\"}\n"));
    try std.testing.expect(has(text, "!29 = !{!\"air.max_work_group_size\", i32 128}\n"));
    try std.testing.expect(has(text, "!30 = !{ptr @countKernel, !25, !28, !29}\n"));
}

const TestVOut = struct { position: @Vector(4, f32) };
const test_vertex_short_fm = functionMetadata(.{ .name = "v", .stage = .vertex, .ret = TestVOut, .args = &.{.{ .vertex_id = "vid" }} });
const test_vertex_long_fm = functionMetadata(.{ .name = "v", .stage = .vertex, .ret = TestVOut, .args = &.{.{ .builtin = .{ .kind = .vertex_id, .T = u32, .name = "vid" } }} });

test "vertex_id shorthand and .builtin vertex_id print the same node" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();
    var a: std.ArrayList(u8) = .empty;
    var b: std.ArrayList(u8) = .empty;
    try printManifest(gpa, &a, &.{test_vertex_short_fm});
    try printManifest(gpa, &b, &.{test_vertex_long_fm});
    try std.testing.expectEqualStrings(a.items, b.items);
    try std.testing.expect(has(a.items, "!{i32 0, !\"air.vertex_id\", !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"vid\"}\n"));
    try std.testing.expect(has(a.items, "!air.vertex = !{"));
}

test "mslType spells kernel builtin and buffer element types" {
    try std.testing.expectEqualStrings("uint", comptime mslType(u32));
    try std.testing.expectEqualStrings("uint2", comptime mslType(@Vector(2, u32)));
    try std.testing.expectEqualStrings("uint3", comptime mslType(@Vector(3, u32)));
    try std.testing.expectEqualStrings("ushort", comptime mslType(u16));
    try std.testing.expectEqualStrings("float4", comptime mslType(@Vector(4, f32)));
    try std.testing.expectEqualStrings("bool", comptime mslType(bool));
    try std.testing.expectEqualStrings("float2", comptime mslType(@Vector(2, f32)));
    try std.testing.expectEqualStrings("half4", comptime mslType(@Vector(4, f16)));
    // r01 metadata_patterns 5 / finding 16: a matrix field spells `float4x4`.
    try std.testing.expectEqualStrings("float4x4", comptime mslType([4]@Vector(4, f32)));
    try std.testing.expectEqualStrings("float3x3", comptime mslType([3]@Vector(3, f32)));
    try std.testing.expectEqualStrings("half2x4", comptime mslType([2]@Vector(4, f16)));
}

test "varyingName mangles vectors and scalars like metalfe (r01 metadata_patterns 8)" {
    try std.testing.expectEqualStrings("generated(6normalDv3_f)", comptime varyingName("normal", @Vector(3, f32)));
    try std.testing.expectEqualStrings("generated(6flatIdi)", comptime varyingName("flatId", i32));
    try std.testing.expectEqualStrings("generated(2idj)", comptime varyingName("id", u32));
    try std.testing.expectEqualStrings("generated(1af)", comptime varyingName("a", f32));
    try std.testing.expectEqualStrings("generated(1dDh)", comptime varyingName("d", f16));
    try std.testing.expectEqualStrings("generated(3idsDv2_j)", comptime varyingName("ids", @Vector(2, u32)));
    try std.testing.expectEqual(@as(?i32, 0), comptime renderTargetIndex("color0"));
    try std.testing.expectEqual(@as(?i32, 3), comptime renderTargetIndex("color3"));
    try std.testing.expectEqual(@as(?i32, null), comptime renderTargetIndex("color"));
    try std.testing.expectEqual(@as(?i32, null), comptime renderTargetIndex("depth"));
    try std.testing.expect(comptime isFlatVarying(u32));
    try std.testing.expect(comptime isFlatVarying(@Vector(2, i32)));
    try std.testing.expect(!comptime isFlatVarying(f32));
    try std.testing.expect(!comptime isFlatVarying(@Vector(3, f32)));
}

// r01 s1_uniforms_builtins: every vertex builtin, a constant matrix
// uniform, point_size and a flat int varying; the fragment side takes
// front_facing and point_coord and returns MRT + depth (s2 / zshape_a).
const TestUniforms = struct { mvp: [4]@Vector(4, f32), tint: @Vector(4, f32), time: f32, flags: u32 };
const TestVOutFull = struct {
    position: @Vector(4, f32),
    point_size: f32,
    flat_id: i32,
    normal: @Vector(3, f32),
    tex: @Vector(2, f32),
    ids: @Vector(2, u32),
    h: f16,
};
const TestFragOut = struct { color0: @Vector(4, f32), color1: @Vector(4, f16), color2: @Vector(4, u32), depth: f32 };

const test_vertex_full_fm = functionMetadata(.{
    .name = "vertexFull",
    .stage = .vertex,
    .ret = TestVOutFull,
    .args = &.{
        .{ .vertex_id = "vid" },
        .{ .builtin = .{ .kind = .instance_id, .T = u32, .name = "iid" } },
        .{ .builtin = .{ .kind = .base_vertex, .T = u32, .name = "bv" } },
        .{ .builtin = .{ .kind = .base_instance, .T = u32, .name = "bi" } },
        .{ .buffer = .{ .index = 1, .T = TestUniforms, .name = "uc", .space = .constant } },
    },
});

const test_fragment_full_fm = functionMetadata(.{
    .name = "fragmentFull",
    .stage = .fragment,
    .ret = TestFragOut,
    .args = &.{
        .{ .stage_in = TestVOutFull },
        .{ .builtin = .{ .kind = .front_facing, .T = bool, .name = "ff" } },
        .{ .builtin = .{ .kind = .point_coord, .T = @Vector(2, f32), .name = "pc" } },
        .{ .builtin = .{ .kind = .sample_id, .T = u32, .name = "sid" } },
        .{ .builtin = .{ .kind = .primitive_id, .T = u32, .name = "pid" } },
    },
});

test "D2 render bindings: vertex builtins, point_size, flat varyings, float4x4 uniform (r01 metadata_patterns 0-2, 5, 7-8)" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();
    var out: std.ArrayList(u8) = .empty;
    try printManifest(gpa, &out, &.{test_vertex_full_fm});
    const text = out.items;
    // Outputs, in field order: position, point_size (no generated name), flat int, vectors, uint2, half.
    try std.testing.expect(has(text, "!{!\"air.position\", !\"air.arg_type_name\", !\"float4\", !\"air.arg_name\", !\"position\"}\n"));
    try std.testing.expect(has(text, "!{!\"air.point_size\", !\"air.arg_type_name\", !\"float\", !\"air.arg_name\", !\"point_size\"}\n"));
    try std.testing.expect(has(text, "!{!\"air.vertex_output\", !\"generated(7flat_idi)\", !\"air.arg_type_name\", !\"int\", !\"air.arg_name\", !\"flat_id\"}\n"));
    try std.testing.expect(has(text, "!{!\"air.vertex_output\", !\"generated(6normalDv3_f)\", !\"air.arg_type_name\", !\"float3\", !\"air.arg_name\", !\"normal\"}\n"));
    try std.testing.expect(has(text, "!{!\"air.vertex_output\", !\"generated(3idsDv2_j)\", !\"air.arg_type_name\", !\"uint2\", !\"air.arg_name\", !\"ids\"}\n"));
    try std.testing.expect(has(text, "!{!\"air.vertex_output\", !\"generated(1hDh)\", !\"air.arg_type_name\", !\"half\", !\"air.arg_name\", !\"h\"}\n"));
    // The vertex-side node carries no interpolation qualifier.
    try std.testing.expect(!has(text, "air.flat"));
    // Builtins: same shape as air.vertex_id, consecutive parameter numbers.
    try std.testing.expect(has(text, "!{i32 0, !\"air.vertex_id\", !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"vid\"}\n"));
    try std.testing.expect(has(text, "!{i32 1, !\"air.instance_id\", !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"iid\"}\n"));
    try std.testing.expect(has(text, "!{i32 2, !\"air.base_vertex\", !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"bv\"}\n"));
    try std.testing.expect(has(text, "!{i32 3, !\"air.base_instance\", !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"bi\"}\n"));
    // Constant struct buffer with a float4x4 field (offset 0, size 64).
    try std.testing.expect(has(text, "= !{i32 0, i32 64, i32 0, !\"float4x4\", !\"mvp\", i32 64, i32 16, i32 0, !\"float4\", !\"tint\", i32 80, i32 4, i32 0, !\"float\", !\"time\", i32 84, i32 4, i32 0, !\"uint\", !\"flags\"}\n"));
    try std.testing.expect(has(text, "!{i32 4, !\"air.buffer\", !\"air.location_index\", i32 1, i32 1, !\"air.read\", !\"air.address_space\", i32 2, !\"air.struct_type_info\", !"));
    try std.testing.expect(has(text, "!\"air.arg_type_size\", i32 96, !\"air.arg_type_align_size\", i32 16, !\"air.arg_type_name\", !\"TestUniforms\", !\"air.arg_name\", !\"uc\"}\n"));
    try std.testing.expect(has(text, "!air.vertex = !{"));
}

test "D2 render bindings: fragment inputs (flat, no point_size, front_facing, point_coord, sample_id, primitive_id) and MRT + depth outputs (r01 metadata_patterns 9-13, 18)" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();
    var out: std.ArrayList(u8) = .empty;
    try printManifest(gpa, &out, &.{test_fragment_full_fm});
    const text = out.items;
    // Outputs in field order: color0 float4, color1 half4, color2 uint4, depth(any).
    try std.testing.expect(has(text, "!{!\"air.render_target\", i32 0, i32 0, !\"air.arg_type_name\", !\"float4\", !\"air.arg_name\", !\"color0\"}\n"));
    try std.testing.expect(has(text, "!{!\"air.render_target\", i32 1, i32 0, !\"air.arg_type_name\", !\"half4\", !\"air.arg_name\", !\"color1\"}\n"));
    try std.testing.expect(has(text, "!{!\"air.render_target\", i32 2, i32 0, !\"air.arg_type_name\", !\"uint4\", !\"air.arg_name\", !\"color2\"}\n"));
    try std.testing.expect(has(text, "!{!\"air.depth\", !\"air.depth_qualifier\", !\"air.any\", !\"air.arg_type_name\", !\"float\", !\"air.arg_name\", !\"depth\"}\n"));
    // Inputs: position, then the varyings minus point_size (parameter numbers stay consecutive).
    try std.testing.expect(has(text, "!{i32 0, !\"air.position\", !\"air.center\", !\"air.no_perspective\", !\"air.arg_type_name\", !\"float4\", !\"air.arg_name\", !\"position\"}\n"));
    try std.testing.expect(!has(text, "point_size"));
    try std.testing.expect(has(text, "!{i32 1, !\"air.fragment_input\", !\"generated(7flat_idi)\", !\"air.flat\", !\"air.arg_type_name\", !\"int\", !\"air.arg_name\", !\"flat_id\"}\n"));
    try std.testing.expect(has(text, "!{i32 2, !\"air.fragment_input\", !\"generated(6normalDv3_f)\", !\"air.center\", !\"air.perspective\", !\"air.arg_type_name\", !\"float3\", !\"air.arg_name\", !\"normal\"}\n"));
    try std.testing.expect(has(text, "!{i32 3, !\"air.fragment_input\", !\"generated(3texDv2_f)\", !\"air.center\", !\"air.perspective\", !\"air.arg_type_name\", !\"float2\", !\"air.arg_name\", !\"tex\"}\n"));
    try std.testing.expect(has(text, "!{i32 4, !\"air.fragment_input\", !\"generated(3idsDv2_j)\", !\"air.flat\", !\"air.arg_type_name\", !\"uint2\", !\"air.arg_name\", !\"ids\"}\n"));
    try std.testing.expect(has(text, "!{i32 5, !\"air.fragment_input\", !\"generated(1hDh)\", !\"air.center\", !\"air.perspective\", !\"air.arg_type_name\", !\"half\", !\"air.arg_name\", !\"h\"}\n"));
    // Fragment builtins.
    try std.testing.expect(has(text, "!{i32 6, !\"air.front_facing\", !\"air.arg_type_name\", !\"bool\", !\"air.arg_name\", !\"ff\"}\n"));
    try std.testing.expect(has(text, "!{i32 7, !\"air.point_coord\", !\"air.arg_type_name\", !\"float2\", !\"air.arg_name\", !\"pc\"}\n"));
    try std.testing.expect(has(text, "!{i32 8, !\"air.sample_id\", !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"sid\"}\n"));
    try std.testing.expect(has(text, "!{i32 9, !\"air.primitive_id\", !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"pid\"}\n"));
    try std.testing.expect(has(text, "!air.fragment = !{"));
    // param_uniformity: one entry per parameter, all of them per-thread.
    try std.testing.expectEqual(10, test_fragment_full_fm.param_uniformity.len);
    for (test_fragment_full_fm.param_uniformity) |u| try std.testing.expectEqual(air.Uniformity.thread, u);
    try std.testing.expectEqual(5, test_vertex_full_fm.param_uniformity.len);
    try std.testing.expectEqual(air.Uniformity.threadgroup, test_vertex_full_fm.param_uniformity[4]);
}

const test_fragment_vec_fm = functionMetadata(.{ .name = "f", .stage = .fragment, .ret = @Vector(4, f32), .args = &.{.{ .stage_in = TestVOut }} });

test "a vector fragment return keeps the single unnamed render_target 0 node" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();
    var out: std.ArrayList(u8) = .empty;
    try printManifest(gpa, &out, &.{test_fragment_vec_fm});
    try std.testing.expect(has(out.items, "!{!\"air.render_target\", i32 0, i32 0, !\"air.arg_type_name\", !\"float4\"}\n"));
    try std.testing.expect(!has(out.items, "air.depth"));
}

// D2/D3 textures and samplers (r03 metadata_patterns 0-3): every access
// kind, a `[[sampler(n)]]` argument in a fragment, a kernel with read +
// write textures, and the `!air.sampler_states` list.
const TestTexSample = shader.gpu.Texture2D(.sample);
const TestTexRead = shader.gpu.Texture2D(.read);
const TestTexWrite = shader.gpu.Texture2D(.write);
const TestTexRW = shader.gpu.Texture2D(.read_write);

const test_textured_fragment_fm = functionMetadata(.{
    .name = "texFrag",
    .stage = .fragment,
    .ret = @Vector(4, f32),
    .args = &.{
        .{ .stage_in = TestVOut },
        .{ .texture = .{ .index = 3, .name = "tex", .T = TestTexSample } },
        .{ .sampler = .{ .index = 2, .name = "s" } },
        .{ .texture = .{ .index = 4, .name = "io", .T = TestTexRW } },
    },
});

const test_texture_kernel_fm = functionMetadata(.{
    .name = "copyTex",
    .stage = .kernel,
    .ret = void,
    .args = &.{
        .{ .texture = .{ .index = 0, .name = "src", .T = TestTexRead } },
        .{ .texture = .{ .index = 1, .name = "dst", .T = TestTexWrite } },
        .{ .builtin = .{ .kind = .thread_position_in_grid, .T = @Vector(2, u32), .name = "gid" } },
    },
});

test "D2 textures and samplers: air.texture nodes per access, air.sampler node, !air.sampler_states (r03 metadata_patterns 0-3)" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();
    var out: std.ArrayList(u8) = .empty;
    try printManifestWith(gpa, &out, &.{ test_textured_fragment_fm, test_texture_kernel_fm }, &.{ "gpu.constexprSampler.Holder.state", "__air_sampler_state.1" }, target.default);
    const text = out.items;
    // Fragment: position, then texture slot 3 (sample), sampler slot 2, read_write texture slot 4.
    try std.testing.expect(has(text, "!{i32 1, !\"air.texture\", !\"air.location_index\", i32 3, i32 1, !\"air.sample\", !\"air.arg_type_name\", !\"texture2d<float, sample>\", !\"air.arg_name\", !\"tex\"}\n"));
    try std.testing.expect(has(text, "!{i32 2, !\"air.sampler\", !\"air.location_index\", i32 2, i32 1, !\"air.arg_type_name\", !\"sampler\", !\"air.arg_name\", !\"s\"}\n"));
    try std.testing.expect(has(text, "!{i32 3, !\"air.texture\", !\"air.location_index\", i32 4, i32 1, !\"air.read_write\", !\"air.arg_type_name\", !\"texture2d<float, read_write>\", !\"air.arg_name\", !\"io\"}\n"));
    // Kernel: read texture 0, write texture 1, uint2 grid position.
    try std.testing.expect(has(text, "!{i32 0, !\"air.texture\", !\"air.location_index\", i32 0, i32 1, !\"air.read\", !\"air.arg_type_name\", !\"texture2d<float, read>\", !\"air.arg_name\", !\"src\"}\n"));
    try std.testing.expect(has(text, "!{i32 1, !\"air.texture\", !\"air.location_index\", i32 1, i32 1, !\"air.write\", !\"air.arg_type_name\", !\"texture2d<float, write>\", !\"air.arg_name\", !\"dst\"}\n"));
    try std.testing.expect(has(text, "!{i32 2, !\"air.thread_position_in_grid\", !\"air.arg_type_name\", !\"uint2\", !\"air.arg_name\", !\"gid\"}\n"));
    try std.testing.expect(has(text, "!air.fragment = !{"));
    try std.testing.expect(has(text, "!air.kernel = !{"));
    // Sampler states: one node per global, referenced from the named list.
    try std.testing.expect(has(text, "!air.sampler_states = !{!"));
    try std.testing.expect(has(text, "= !{!\"air.sampler_state\", ptr addrspace(2) @gpu.constexprSampler.Holder.state}\n"));
    try std.testing.expect(has(text, "= !{!\"air.sampler_state\", ptr addrspace(2) @__air_sampler_state.1}\n"));
    // Textures and samplers are threadgroup-uniform parameters.
    try std.testing.expectEqual(4, test_textured_fragment_fm.param_uniformity.len);
    try std.testing.expectEqual(air.Uniformity.threadgroup, test_textured_fragment_fm.param_uniformity[1]);
    try std.testing.expectEqual(air.Uniformity.threadgroup, test_textured_fragment_fm.param_uniformity[2]);
    try std.testing.expectEqual(air.Uniformity.thread, test_texture_kernel_fm.param_uniformity[2]);

    // Without sampler globals there is no list at all.
    var plain: std.ArrayList(u8) = .empty;
    try printManifest(gpa, &plain, &.{test_textured_fragment_fm});
    try std.testing.expect(!has(plain.items, "air.sampler_states"));
}
