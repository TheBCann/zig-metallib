//! The runtime manifest: one kernel's interface as a SPIR-V module describes
//! it, and its conversion to the `metadata.FunctionMetadata` the assembler
//! attaches (`!air.kernel` node + per-parameter uniformity). The Zig path
//! derives the same structure at comptime from `air.Function` and the Zig
//! types (tools/air/metadata.zig); here the parameters come from the module's
//! decorations (tools/spirv/lower.zig fills a `Kernel`), so nothing about the
//! interface has to be known when spirv-splice is compiled.
//!
//! Parameter order (lower.zig): buffers by location index (descriptor
//! bindings and entry-point pointer/value parameters), then the push
//! constant block, then the builtins in interface order. The text form
//! (`write`, `spirv-splice --manifest=<file>`) is what a host reads to bind
//! buffers and dispatch:
//!
//! ```
//! # spirv-splice manifest 1
//! kernel crosscheckKernel local_size 64 1 1 threadgroup_bytes 0
//!   buffer 0 in0 device read size 16384 align 4 type Buf binding 0 set 0
//!     field 0 4 4096 int data
//!   buffer 3 params constant read size 4 align 4 type Params binding 3 set 0
//!     field 0 4 0 uint count
//!   builtin thread_position_in_grid gid uint3
//! end
//! ```
//!
//! AIR facts (Apple's own output for the MSL twin, `xcrun metal -fno-fast-math`
//! on `struct B { int data[4096]; }; kernel void k(device const B& in0
//! [[buffer(0)]], constant P& params [[buffer(3)]], device R* r [[buffer(4)]],
//! uint3 gid [[thread_position_in_grid]])`, metalfe-32023.883):
//!   * a buffer node is `!{i32 P, !"air.buffer", [!"air.buffer_size", i32 N,]
//!     !"air.location_index", i32 I, i32 1, !"air.read"|!"air.read_write",
//!     !"air.address_space", i32 1|2, [!"air.struct_type_info", !S,]
//!     !"air.arg_type_size", i32 N, !"air.arg_type_align_size", i32 A,
//!     !"air.arg_type_name", !"<name>", !"air.arg_name", !"<name>"}`;
//!     `air.buffer_size` appears for a reference to a struct of fixed size
//!     (`device B&`, `constant P&`), not for a pointer (`device R*`), which
//!     is how a Block ending in a runtime array is described here;
//!   * `air.struct_type_info` is `offset, size, count, "type", "name"` per
//!     member, where an array member gives its ELEMENT size and its length
//!     (`!{i32 0, i32 4, i32 4096, !"int", !"data"}`) and every other member
//!     its size and count 0; a runtime array has count 1 (`float data[1]`);
//!   * builtins are `!{i32 P, !"air.<kind>", !"air.arg_type_name",
//!     !"uint"|"uint3", !"air.arg_name", !"<name>"}` as on the Zig path.

const std = @import("std");
const metadata = @import("../air/metadata.zig");
const air = metadata.air;
const Node = metadata.Node;

/// AIR address space of a buffer parameter.
pub const Space = enum(u8) {
    device = 1,
    constant = 2,
};

pub const Access = enum { read, read_write };

/// Where the host binds a buffer from, in the SPIR-V module's own terms.
pub const Source = union(enum) {
    /// A `StorageBuffer`/`Uniform`/`UniformConstant` variable decorated
    /// `DescriptorSet set` / `Binding binding`; the location index is the
    /// binding.
    binding: struct { set: u32, binding: u32 },
    /// The `PushConstant` block; its location index is
    /// `--push-constant-index`.
    push_constant,
    /// Parameter `n` of the entry function (OpenCL-style kernels and Zig's
    /// pointer-parameter shape): a pointer is the buffer itself, a value is
    /// read from a constant buffer at index n.
    parameter: u32,
    /// A buffer variable without a Binding decoration (Zig's plain
    /// `addrspace(.storage_buffer)` externs): lower.zig gives it the lowest
    /// free index, in variable order.
    unbound,
};

/// One `air.struct_type_info` entry.
pub const Field = struct {
    offset: u32,
    /// The member's size, or its element's size for an array member.
    size: u32,
    /// Array length; 0 for a non-array member, 1 for a runtime array.
    count: u32,
    type_name: []const u8,
    name: []const u8,
};

pub const Buffer = struct {
    /// `[[buffer(index)]]`.
    index: u32,
    source: Source,
    name: []const u8,
    space: Space,
    access: Access,
    /// `air.arg_type_size` / `air.arg_type_align_size`: the pointee.
    size: u32,
    alignment: u32,
    type_name: []const u8,
    /// `air.struct_type_info`; null for a scalar or vector pointee.
    fields: ?[]const Field = null,
    /// `air.buffer_size`: set for a struct of fixed size (MSL `B&`).
    buffer_size: ?u32 = null,
    /// Entry-point value parameter: the shader reads the value from this
    /// constant buffer (the host passes it with setBytes).
    by_value: bool = false,
};

pub const Builtin = struct {
    kind: air.BuiltinKind,
    /// `uint` or `uint<N>`.
    type_name: []const u8,
    name: []const u8,
};

pub const Param = union(enum) {
    buffer: Buffer,
    builtin: Builtin,
};

pub const Kernel = struct {
    /// The OpEntryPoint name, the metallib function name.
    name: [:0]const u8,
    /// OpExecutionMode LocalSize (1 1 1 when absent): the threadgroup size
    /// the host dispatches with; it is not encoded in the library.
    local_size: [3]u32 = .{ 1, 1, 1 },
    params: []const Param,
    /// Bytes of `Workgroup` variables the kernel reaches (static threadgroup
    /// memory; Metal reports it as staticThreadgroupMemoryLength).
    threadgroup_bytes: u32 = 0,
};

/// The `FunctionMetadata` for `k`, allocated in `a` (an arena: the tree is
/// many small slices). Node shapes as in the module doc; uniformity:
/// buffers are threadgroup-uniform, builtins follow `BuiltinKind.uniformity`.
pub fn functionMetadata(a: std.mem.Allocator, k: Kernel) !metadata.FunctionMetadata {
    const inputs = try a.alloc(Node, k.params.len);
    const uniformity = try a.alloc(air.Uniformity, k.params.len);
    for (k.params, inputs, uniformity, 0..) |p, *node, *u, i| {
        const param: i32 = @intCast(i);
        switch (p) {
            .buffer => |b| {
                node.* = try bufferNode(a, param, b);
                u.* = .threadgroup;
            },
            .builtin => |bi| {
                node.* = try tuple(a, &.{
                    .{ .int = param },
                    .{ .str = try std.mem.concat(a, u8, &.{ "air.", @tagName(bi.kind) }) },
                    .{ .str = "air.arg_type_name" },
                    .{ .str = bi.type_name },
                    .{ .str = "air.arg_name" },
                    .{ .str = bi.name },
                });
                u.* = bi.kind.uniformity();
            },
        }
    }
    const node = try tuple(a, &.{ .func, .{ .tuple = &.{} }, .{ .tuple = inputs } });
    return .{ .name = k.name, .stage = .kernel, .node = node, .param_uniformity = uniformity };
}

/// A metadata integer (`i32`): sizes past 2 GiB do not fit Apple's nodes.
fn int32(v: u32) error{Overflow}!i32 {
    return std.math.cast(i32, v) orelse error.Overflow;
}

fn tuple(a: std.mem.Allocator, items: []const Node) !Node {
    return .{ .tuple = try a.dupe(Node, items) };
}

fn bufferNode(a: std.mem.Allocator, param: i32, b: Buffer) !Node {
    var items: std.ArrayList(Node) = .empty;
    try items.appendSlice(a, &.{ .{ .int = param }, .{ .str = "air.buffer" } });
    if (b.buffer_size) |n| try items.appendSlice(a, &.{ .{ .str = "air.buffer_size" }, .{ .int = try int32(n) } });
    try items.appendSlice(a, &.{
        .{ .str = "air.location_index" }, .{ .int = try int32(b.index) },
        .{ .int = 1 },                    .{ .str = if (b.access == .read) "air.read" else "air.read_write" },
        .{ .str = "air.address_space" },  .{ .int = @backingInt(b.space) },
    });
    if (b.fields) |fields| {
        const info = try a.alloc(Node, fields.len * 5);
        for (fields, 0..) |f, i| {
            info[i * 5 ..][0..5].* = .{ .{ .int = try int32(f.offset) }, .{ .int = try int32(f.size) }, .{ .int = try int32(f.count) }, .{ .str = f.type_name }, .{ .str = f.name } };
        }
        try items.appendSlice(a, &.{ .{ .str = "air.struct_type_info" }, .{ .tuple = info } });
    }
    try items.appendSlice(a, &.{
        .{ .str = "air.arg_type_size" },       .{ .int = try int32(b.size) },
        .{ .str = "air.arg_type_align_size" }, .{ .int = try int32(b.alignment) },
        .{ .str = "air.arg_type_name" },       .{ .str = b.type_name },
        .{ .str = "air.arg_name" },            .{ .str = b.name },
    });
    return .{ .tuple = try items.toOwnedSlice(a) };
}

/// The text form of the module doc, one `kernel ... end` block per kernel.
pub fn write(w: *std.Io.Writer, kernels: []const Kernel) std.Io.Writer.Error!void {
    try w.writeAll("# spirv-splice manifest 1\n");
    for (kernels) |k| {
        try w.print("kernel {s} local_size {d} {d} {d} threadgroup_bytes {d}\n", .{ k.name, k.local_size[0], k.local_size[1], k.local_size[2], k.threadgroup_bytes });
        for (k.params) |p| switch (p) {
            .buffer => |b| {
                try w.print("  buffer {d} {s} {t} {t} size {d} align {d} type {s}", .{ b.index, b.name, b.space, b.access, b.size, b.alignment, b.type_name });
                switch (b.source) {
                    .binding => |d| try w.print(" binding {d} set {d}", .{ d.binding, d.set }),
                    .push_constant => try w.writeAll(" push_constant"),
                    .parameter => |n| try w.print(" parameter {d}{s}", .{ n, if (b.by_value) " by_value" else "" }),
                    .unbound => try w.writeAll(" unbound"),
                }
                try w.writeByte('\n');
                if (b.fields) |fields| for (fields) |f| {
                    try w.print("    field {d} {d} {d} {s} {s}\n", .{ f.offset, f.size, f.count, f.type_name, f.name });
                };
            },
            .builtin => |bi| try w.print("  builtin {t} {s} {s}\n", .{ bi.kind, bi.name, bi.type_name }),
        };
        try w.writeAll("end\n");
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

fn has(hay: []const u8, needle: []const u8) bool {
    return std.mem.find(u8, hay, needle) != null;
}

const test_kernel: Kernel = .{
    .name = "crosscheckKernel",
    .local_size = .{ 64, 1, 1 },
    .params = &.{
        .{ .buffer = .{ .index = 0, .source = .{ .binding = .{ .set = 0, .binding = 0 } }, .name = "in0", .space = .device, .access = .read, .size = 16384, .alignment = 4, .type_name = "Buf", .buffer_size = 16384, .fields = &.{.{ .offset = 0, .size = 4, .count = 4096, .type_name = "int", .name = "data" }} } },
        .{ .buffer = .{ .index = 2, .source = .{ .binding = .{ .set = 0, .binding = 2 } }, .name = "out", .space = .device, .access = .read_write, .size = 4, .alignment = 4, .type_name = "Rt", .fields = &.{.{ .offset = 0, .size = 4, .count = 1, .type_name = "int", .name = "data" }} } },
        .{ .buffer = .{ .index = 30, .source = .push_constant, .name = "pc", .space = .constant, .access = .read, .size = 8, .alignment = 4, .type_name = "Push", .buffer_size = 8, .fields = &.{
            .{ .offset = 0, .size = 4, .count = 0, .type_name = "uint", .name = "count" },
            .{ .offset = 4, .size = 4, .count = 0, .type_name = "float", .name = "scale" },
        } } },
        .{ .builtin = .{ .kind = .thread_position_in_grid, .type_name = "uint3", .name = "gid" } },
        .{ .builtin = .{ .kind = .threadgroup_position_in_grid, .type_name = "uint3", .name = "wg" } },
        .{ .builtin = .{ .kind = .simdgroup_index_in_threadgroup, .type_name = "uint", .name = "sg" } },
    },
};

test "functionMetadata: buffer nodes in Apple's shape, builtins, uniformity per parameter" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const fm = try functionMetadata(a, test_kernel);
    try std.testing.expectEqualStrings("crosscheckKernel", fm.name);
    try std.testing.expectEqual(air.Stage.kernel, fm.stage);
    try std.testing.expectEqualSlices(air.Uniformity, &.{ .threadgroup, .threadgroup, .threadgroup, .thread, .threadgroup, .simdgroup }, fm.param_uniformity);

    var out: std.ArrayList(u8) = .empty;
    try metadata.printManifest(a, &out, &.{fm});
    const text = out.items;
    // A fixed-size Block: buffer_size before location_index, struct_type_info
    // with the array member's element size and length (Apple's B&).
    try std.testing.expect(has(text, "!{i32 0, !\"air.buffer\", !\"air.buffer_size\", i32 16384, !\"air.location_index\", i32 0, i32 1, !\"air.read\", !\"air.address_space\", i32 1, !\"air.struct_type_info\", !"));
    try std.testing.expect(has(text, "= !{i32 0, i32 4, i32 4096, !\"int\", !\"data\"}\n"));
    try std.testing.expect(has(text, "!\"air.arg_type_size\", i32 16384, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"Buf\", !\"air.arg_name\", !\"in0\"}\n"));
    // A runtime-array Block: no buffer_size (Apple's R*), count 1.
    try std.testing.expect(has(text, "!{i32 1, !\"air.buffer\", !\"air.location_index\", i32 2, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.struct_type_info\", !"));
    try std.testing.expect(has(text, "= !{i32 0, i32 4, i32 1, !\"int\", !\"data\"}\n"));
    // Push constants: constant space at the push-constant index.
    try std.testing.expect(has(text, "!{i32 2, !\"air.buffer\", !\"air.buffer_size\", i32 8, !\"air.location_index\", i32 30, i32 1, !\"air.read\", !\"air.address_space\", i32 2, !\"air.struct_type_info\", !"));
    try std.testing.expect(has(text, "= !{i32 0, i32 4, i32 0, !\"uint\", !\"count\", i32 4, i32 4, i32 0, !\"float\", !\"scale\"}\n"));
    // Builtins.
    try std.testing.expect(has(text, "!{i32 3, !\"air.thread_position_in_grid\", !\"air.arg_type_name\", !\"uint3\", !\"air.arg_name\", !\"gid\"}\n"));
    try std.testing.expect(has(text, "!{i32 5, !\"air.simdgroup_index_in_threadgroup\", !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"sg\"}\n"));
    // The kernel node: {fn, no outputs, inputs}, no max_work_group_size.
    try std.testing.expect(has(text, "!air.kernel = !{"));
    try std.testing.expect(has(text, "ptr @crosscheckKernel, "));
    try std.testing.expect(!has(text, "air.max_work_group_size"));
}

test "the text form lists every parameter with its source" {
    var aw: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer aw.deinit();
    try write(&aw.writer, &.{test_kernel});
    try std.testing.expectEqualStrings(
        \\# spirv-splice manifest 1
        \\kernel crosscheckKernel local_size 64 1 1 threadgroup_bytes 0
        \\  buffer 0 in0 device read size 16384 align 4 type Buf binding 0 set 0
        \\    field 0 4 4096 int data
        \\  buffer 2 out device read_write size 4 align 4 type Rt binding 2 set 0
        \\    field 0 4 1 int data
        \\  buffer 30 pc constant read size 8 align 4 type Push push_constant
        \\    field 0 4 0 uint count
        \\    field 4 4 0 float scale
        \\  builtin thread_position_in_grid gid uint3
        \\  builtin threadgroup_position_in_grid wg uint3
        \\  builtin simdgroup_index_in_threadgroup sg uint
        \\end
        \\
    , aw.written());
}

test "a quoted entry name prints quoted in the metadata" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const fm = try functionMetadata(a, .{ .name = "my kernel", .params = &.{} });
    var out: std.ArrayList(u8) = .empty;
    try metadata.printManifest(a, &out, &.{fm});
    try std.testing.expect(has(out.items, "ptr @\"my kernel\", "));
}
