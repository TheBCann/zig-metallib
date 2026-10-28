//! SPIR-V binary decoder: bytes in, a `Module` out. A pure function of the
//! bytes (no I/O); every allocation lives in the module's arena.
//!
//! What it decodes (SPIR-V 1.6 unified spec, §2.3 "Physical Layout"):
//!   * the header: magic 0x07230203 in either endianness (a byte-swapped
//!     magic means every word is swapped), version, generator, id bound,
//!     schema (always 0 today);
//!   * the instruction stream: `word count << 16 | opcode` per instruction,
//!     operands decoded by the grammar tables in spec.zig (`Opcode.operands`,
//!     `OperandKind.category/bases`, and `enumerants` below, which adds the
//!     enumerants spec.zig lacks from spec_extra.zig: a vendor decoration
//!     such as DXC's `UserTypeGOOGLE "structuredbuffer:<uint>"` decodes with
//!     its string), so value-enum parameters
//!     (`Aligned 4`, `LocalSize 64 1 1`, `BuiltIn GlobalInvocationId`),
//!     bit-enum parameters, literal strings (nul-terminated UTF-8 packed
//!     little-endian in words), variadic tails, and composite pairs land in
//!     the right slots;
//!   * context-dependent numbers (OpConstant / OpSpecConstant): 1 or 2 words
//!     by the width of the result type (a 64-bit int or double is two words,
//!     low word first); OpSwitch case literals likewise take the selector's
//!     width (§3.49.3 of the spec; the grammar calls both "literal");
//!   * OpExtInst operands after the instruction number come from the GLSL.std.450
//!     or OpenCL.std tables when the set is one of those, else they are ids;
//!     OpSpecConstantOp operands after the opcode come from that opcode's own
//!     grammar (§3.52.12), so a CompositeExtract index or a VectorShuffle
//!     component is a literal, not an id;
//!   * tables: types, constants (scalars as raw u64 bits, composites as member
//!     ids, null/true/false, OpUndef), decorations per id and per member
//!     (decoration groups expanded, see below), names, entry points with
//!     their execution modes (linked once after decoding, linear in the
//!     module however many entry points share a function), extended-instruction
//!     imports, memory model, capabilities, extensions, globals (OpVariable
//!     outside a function), functions with parameters and blocks (label id,
//!     instruction range; the last instruction of a block is its terminator).
//! Decoration groups (§3.32.2: OpDecorationGroup, OpGroupDecorate,
//! OpGroupMemberDecorate; SPIRV-LLVM-Translator and older glslang emit them)
//! are expanded into the decoration table: at each OpGroupDecorate every
//! whole-id decoration recorded on the group id is copied once per target,
//! and at each OpGroupMemberDecorate once per (target, member) pair, so
//! `Module.decoration(target, ...)` sees `NonWritable`, `DescriptorSet 1` or
//! `FuncParamAttr NoAlias` applied through a group exactly as if decorated
//! directly. The group id keeps its own entries. The spec requires every
//! OpDecorate of a group to precede its OpDecorationGroup, so a decoration of
//! an already-declared group (which expansion would miss) is refused, as is
//! an OpGroup*Decorate naming an id that is not a decoration group, and one
//! whose target is a decoration group (before or after that group's
//! OpDecorationGroup; SPIRV-Tools' validator refuses it too, "OpGroupDecorate
//! may not target OpDecorationGroup"): `OpGroupDecorate %g %g` would copy
//! the group's entries onto itself and double the table at every such
//! instruction, so a module of a few hundred bytes could demand gigabytes.
//! What remains is linear per instruction but quadratic in the file (many
//! targets times many group decorations), so the expanded entries are capped
//! at `max_expansion_per_word` per input word; past it the module is refused.
//!
//! OpTypeFloat's optional FPEncoding operand (SPIR-V 1.6 rev 7 grammar:
//! `OpTypeFloat Width [FPEncoding]`; BFloat16KHR from SPV_KHR_bfloat16,
//! Float8E4M3EXT / Float8E5M2EXT from SPV_EXT_float8) is kept in
//! `Type.float.encoding`; null is IEEE 754 binary16/32/64. A bfloat16 is
//! 16 bits wide like a half, so the encoding is the only thing that tells
//! them apart: dis.zig prints such a type's constants as raw `0x` bits,
//! asm.zig accepts only raw bits for them, and lower.zig refuses the type
//! by name (MSL 3.1 has a `bfloat`, but no lowering rule for it has been
//! verified against Metal's compiler; float8 has no MSL type), so neither is
//! ever treated as a half.
//!
//! Every instruction keeps its word offset, so a refusal later (lower.zig)
//! can name it, and `Instruction.operands` is the flattened operand list in
//! grammar order (result type and result id are pulled out).
//!
//! Malformed input (bad magic, odd byte length, zero word count, an
//! instruction running past the end, an unknown opcode, a missing required
//! operand, an unterminated string, an id at or above the bound, a result id
//! defined twice, an instruction outside a block inside a function, a
//! function without OpFunctionEnd, an OpSpecConstantOp naming no opcode, a
//! literal of a type wider than 64 bits, the decoration-group misuse and
//! the expansion cap above, words after an enumerant the grammar does not
//! have, whose parameters are therefore unknown) is refused with `error.Malformed`; the
//! `Diagnostics` the caller passes names the word offset and the reason.
//!
//! Evidence: the Zig SPIR-V backend's output in testdata/*.spv (Vulkan and
//! OpenCL flavours, descriptors, control flow, narrow signed constants) and naga's snapshots in
//! corpus/*.spvasm all decode and round-trip through dis.zig and asm.zig
//! byte for byte (tests below and in asm.zig).

const std = @import("std");
pub const spec = @import("spec.zig");
const spec_extra = @import("spec_extra.zig");

pub const Id = u32;

pub const Diagnostics = struct {
    /// Word offset (0 = the magic number) of the instruction or word that
    /// made the reader give up.
    word_offset: usize = 0,
    message: []const u8 = "",
    buf: [192]u8 = undefined,

    fn set(d: ?*Diagnostics, offset: usize, comptime fmt: []const u8, args: anytype) error{Malformed} {
        if (d) |diag| {
            diag.word_offset = offset;
            diag.message = std.mem.print(&diag.buf, fmt, args) catch diag.buf[0..];
        }
        return error.Malformed;
    }
};

pub const Error = error{ Malformed, OutOfMemory };

/// One decoded operand. Enum parameters, bit-enum parameters and the two
/// halves of a composite pair are flattened into the list in grammar order.
pub const Operand = struct {
    kind: spec.OperandKind,
    value: Value,

    pub const Value = union(enum) {
        id: Id,
        /// literal_integer, literal_ext_inst_integer, literal_spec_constant_op_integer,
        /// value-enum and bit-enum values, and 32-bit context-dependent numbers.
        literal: u32,
        /// A two-word literal: 64-bit context-dependent numbers (OpConstant of a
        /// 64-bit type) and OpSwitch case values of a 64-bit selector.
        literal64: u64,
        string: []const u8,
    };

    pub fn asId(o: Operand) ?Id {
        return switch (o.value) {
            .id => |id| id,
            else => null,
        };
    }

    /// The literal as u64 whether it took one or two words.
    pub fn asInt(o: Operand) ?u64 {
        return switch (o.value) {
            .literal => |v| v,
            .literal64 => |v| v,
            else => null,
        };
    }
};

pub const Instruction = struct {
    /// The raw opcode; `op` is null when spec.zig does not know it (the
    /// reader refuses such modules, so in a decoded module it is always set).
    opcode: u16,
    op: spec.Opcode,
    word_offset: u32,
    word_count: u16,
    result_type: ?Id,
    result: ?Id,
    /// Operands after result type / result id, flattened.
    operands: []const Operand,
    /// For OpExtInst: the set the instruction number belongs to (null when the
    /// set is not GLSL.std.450 / OpenCL.std).
    ext_set: ?spec.InstructionSet = null,

    pub fn operand(inst: Instruction, i: usize) ?Operand {
        return if (i < inst.operands.len) inst.operands[i] else null;
    }

    pub fn idAt(inst: Instruction, i: usize) ?Id {
        const o = inst.operand(i) orelse return null;
        return o.asId();
    }

    pub fn intAt(inst: Instruction, i: usize) ?u64 {
        const o = inst.operand(i) orelse return null;
        return o.asInt();
    }

    pub fn stringAt(inst: Instruction, i: usize) ?[]const u8 {
        const o = inst.operand(i) orelse return null;
        return switch (o.value) {
            .string => |s| s,
            else => null,
        };
    }
};

pub const Type = union(enum) {
    void,
    bool,
    int: struct { width: u32, signed: bool },
    /// `encoding` is OpTypeFloat's optional FPEncoding operand (a
    /// `spec.FPEncoding` value: BFloat16KHR = 0, Float8E4M3EXT,
    /// Float8E5M2EXT, or a value spec.zig does not name); null means IEEE
    /// 754. Every consumer that interprets the bits must check it: a
    /// bfloat16 has `width` 16 like a half.
    float: struct { width: u32, encoding: ?u32 = null },
    vector: struct { elem: Id, len: u32 },
    matrix: struct { column: Id, cols: u32 },
    /// `len` is a constant id (OpConstant or OpSpecConstant).
    array: struct { elem: Id, len: Id },
    runtime_array: struct { elem: Id },
    @"struct": struct { members: []const Id },
    pointer: struct { storage_class: u32, pointee: Id },
    function: struct { ret: Id, params: []const Id },
    forward_pointer: struct { storage_class: u32 },
    /// Images, samplers, sampled images, opaque, event, queue, pipe, ...:
    /// kept by opcode so a refusal can name them.
    other: spec.Opcode,
};

pub const Constant = struct {
    type: Id,
    /// OpSpecConstant*: a specialization constant with its default value.
    spec_constant: bool = false,
    value: Value,

    pub const Value = union(enum) {
        bool: bool,
        /// Raw bits of the context-dependent literal, zero-extended to 64:
        /// the word(s) as they are in the module. SPIR-V 2.2.1 (Literal
        /// Number) has a signed type narrower than its word sign-extended to
        /// the word, and producers do that (Zig's backend writes 0xFFFFFFFF
        /// for an i8 -1, testdata/narrow.spv), so the high bits above the
        /// type's width may be set; `Module.constantBits` truncates them.
        scalar: u64,
        composite: []const Id,
        null,
        undef,
        /// OpSpecConstantOp: the opcode and its operands in that opcode's own
        /// grammar order (ids, and the literal indexes of CompositeExtract /
        /// CompositeInsert / VectorShuffle), result type and id excluded.
        spec_op: struct { opcode: spec.Opcode, operands: []const Operand },
        /// OpConstantSampler and anything else the lowerer refuses.
        other: spec.Opcode,
    };
};

pub const Decoration = struct {
    target: Id,
    /// Member index for OpMemberDecorate, null for OpDecorate.
    member: ?u32,
    decoration: u32,
    /// The decoration's parameters (`Offset 4` -> one literal; `BuiltIn
    /// GlobalInvocationId` -> one value-enum literal; `LinkageAttributes
    /// "name" Export` -> a string and an enum).
    operands: []const Operand,

    pub fn as(d: Decoration) ?spec.Decoration {
        return std.enums.fromInt(spec.Decoration, d.decoration);
    }

    pub fn firstInt(d: Decoration) ?u64 {
        if (d.operands.len == 0) return null;
        return d.operands[0].asInt();
    }
};

pub const ExtInstImport = struct {
    id: Id,
    name: []const u8,
    set: ?spec.InstructionSet,
};

pub const EntryPoint = struct {
    execution_model: u32,
    function: Id,
    name: []const u8,
    interface: []const Id,
    /// Index into `Module.execution_modes` for every OpExecutionMode naming this entry.
    execution_modes: []const u32,

    pub fn model(e: EntryPoint) ?spec.ExecutionModel {
        return std.enums.fromInt(spec.ExecutionModel, e.execution_model);
    }
};

pub const ExecutionMode = struct {
    function: Id,
    mode: u32,
    /// The mode's parameters (`LocalSize 64 1 1` -> three literals).
    operands: []const Operand,

    pub fn as(e: ExecutionMode) ?spec.ExecutionMode {
        return std.enums.fromInt(spec.ExecutionMode, e.mode);
    }
};

pub const Global = struct {
    id: Id,
    /// The pointer type of the variable.
    type: Id,
    storage_class: u32,
    initializer: ?Id,
    /// Index into `Module.instructions`.
    inst: u32,
};

pub const Block = struct {
    label: Id,
    /// Range into `Module.instructions`: the OpLabel is `first`, the
    /// terminator is `end - 1`.
    first: u32,
    end: u32,
};

pub const Parameter = struct { id: Id, type: Id };

pub const Function = struct {
    id: Id,
    result_type: Id,
    control: u32,
    function_type: Id,
    params: []const Parameter,
    blocks: []const Block,
    /// Range into `Module.instructions`: OpFunction at `first`, OpFunctionEnd at `end - 1`.
    first: u32,
    end: u32,
    /// Index into `Module.entry_points` when an OpEntryPoint names this function.
    entry_point: ?u32,
};

pub const MemoryModel = struct { addressing: u32, memory: u32 };

pub const MemberKey = struct { id: Id, member: u32 };

pub const Module = struct {
    arena: std.heap.ArenaAllocator,
    version_major: u8,
    version_minor: u8,
    generator: u32,
    bound: u32,
    schema: u32,
    /// Every instruction of the module in stream order.
    instructions: []const Instruction,
    capabilities: []const u32,
    extensions: []const []const u8,
    ext_inst_imports: []const ExtInstImport,
    memory_model: ?MemoryModel,
    entry_points: []const EntryPoint,
    execution_modes: []const ExecutionMode,
    names: std.AutoHashMapUnmanaged(Id, []const u8),
    member_names: std.AutoHashMapUnmanaged(MemberKey, []const u8),
    decorations: []const Decoration,
    types: std.AutoHashMapUnmanaged(Id, Type),
    constants: std.AutoHashMapUnmanaged(Id, Constant),
    /// Result type of every id that has one (constants, globals, parameters,
    /// labels excluded, instruction results).
    id_types: std.AutoHashMapUnmanaged(Id, Id),
    globals: []const Global,
    functions: []const Function,

    pub fn deinit(m: *Module) void {
        m.arena.deinit();
        m.* = undefined;
    }

    pub fn name(m: *const Module, id: Id) ?[]const u8 {
        return m.names.get(id);
    }

    pub fn memberName(m: *const Module, id: Id, member: u32) ?[]const u8 {
        return m.member_names.get(.{ .id = id, .member = member });
    }

    pub fn typeOf(m: *const Module, id: Id) ?Type {
        return m.types.get(id);
    }

    pub fn constantOf(m: *const Module, id: Id) ?Constant {
        return m.constants.get(id);
    }

    /// The bits of a scalar constant truncated to its type's width (see
    /// `Constant.Value.scalar`): an i8 -1 is 0xFF whether the module wrote
    /// the word as 0xFF or sign-extended it to 0xFFFFFFFF. Null for anything
    /// but an OpConstant / OpSpecConstant of an int or float type.
    pub fn constantBits(m: *const Module, id: Id) ?u64 {
        const c = m.constants.get(id) orelse return null;
        const v = switch (c.value) {
            .scalar => |v| v,
            else => return null,
        };
        const width = m.scalarWidth(c.type) orelse return null;
        return truncateBits(v, width);
    }

    /// The result type id of a value id.
    pub fn resultTypeOf(m: *const Module, id: Id) ?Id {
        return m.id_types.get(id);
    }

    pub fn extSetOf(m: *const Module, id: Id) ?ExtInstImport {
        for (m.ext_inst_imports) |e| if (e.id == id) return e;
        return null;
    }

    pub fn functionById(m: *const Module, id: Id) ?*const Function {
        for (m.functions) |*f| if (f.id == id) return f;
        return null;
    }

    pub fn entryPointByName(m: *const Module, ep_name: []const u8) ?*const EntryPoint {
        for (m.entry_points) |*e| if (std.mem.eql(u8, e.name, ep_name)) return e;
        return null;
    }

    /// The first decoration `dec` on `id` (whole-id decorations only).
    pub fn decoration(m: *const Module, id: Id, dec: spec.Decoration) ?Decoration {
        for (m.decorations) |d| {
            if (d.target == id and d.member == null and d.decoration == @backingInt(dec)) return d;
        }
        return null;
    }

    pub fn memberDecoration(m: *const Module, id: Id, member: u32, dec: spec.Decoration) ?Decoration {
        for (m.decorations) |d| {
            if (d.target == id and d.member != null and d.member.? == member and d.decoration == @backingInt(dec)) return d;
        }
        return null;
    }

    pub fn hasDecoration(m: *const Module, id: Id, dec: spec.Decoration) bool {
        return m.decoration(id, dec) != null;
    }

    /// Width in bits of a scalar int/float type, or of a vector's element.
    pub fn scalarWidth(m: *const Module, type_id: Id) ?u32 {
        const t = m.types.get(type_id) orelse return null;
        return switch (t) {
            .int => |i| i.width,
            .float => |f| f.width,
            .vector => |v| m.scalarWidth(v.elem),
            else => null,
        };
    }
};

const Words = struct {
    words: []const u32,
};

/// Decode a SPIR-V module. The returned module owns an arena; free it with
/// `Module.deinit`. On `error.Malformed`, `diag` (when given) holds the word
/// offset and a message.
pub fn read(gpa: std.mem.Allocator, bytes: []const u8, diag: ?*Diagnostics) Error!Module {
    var arena = std.heap.ArenaAllocator.init(gpa);
    errdefer arena.deinit();
    const a = arena.allocator();

    if (bytes.len % 4 != 0) return Diagnostics.set(diag, 0, "byte length {d} is not a multiple of 4", .{bytes.len});
    if (bytes.len < 20) return Diagnostics.set(diag, 0, "only {d} bytes, shorter than the 5-word header", .{bytes.len});
    const n = bytes.len / 4;
    const words = try a.alloc(u32, n);
    const first = std.mem.readInt(u32, bytes[0..4], .little);
    const endian: std.lang.Endian = if (first == spec.magic_number) .little else if (first == @byteSwap(spec.magic_number)) .big else {
        return Diagnostics.set(diag, 0, "bad magic 0x{x:0>8} (expected 0x07230203)", .{first});
    };
    for (words, 0..) |*w, i| w.* = std.mem.readInt(u32, bytes[i * 4 ..][0..4], endian);

    const bound = words[3];
    var r: Reader = .{
        .a = a,
        .words = words,
        .diag = diag,
        .bound = bound,
    };
    try r.decodeAll();

    return .{
        .arena = arena,
        .version_major = @truncate(words[1] >> 16),
        .version_minor = @truncate(words[1] >> 8),
        .generator = words[2],
        .bound = bound,
        .schema = words[4],
        .instructions = try r.instructions.toOwnedSlice(a),
        .capabilities = try r.capabilities.toOwnedSlice(a),
        .extensions = try r.extensions.toOwnedSlice(a),
        .ext_inst_imports = try r.ext_inst_imports.toOwnedSlice(a),
        .memory_model = r.memory_model,
        .entry_points = try r.entry_points.toOwnedSlice(a),
        .execution_modes = try r.execution_modes.toOwnedSlice(a),
        .names = r.names,
        .member_names = r.member_names,
        .decorations = try r.decorations.toOwnedSlice(a),
        .types = r.types,
        .constants = r.constants,
        .id_types = r.id_types,
        .globals = try r.globals.toOwnedSlice(a),
        .functions = try r.functions.toOwnedSlice(a),
    };
}

const Reader = struct {
    a: std.mem.Allocator,
    words: []const u32,
    diag: ?*Diagnostics,
    bound: u32,

    instructions: std.ArrayList(Instruction) = .empty,
    capabilities: std.ArrayList(u32) = .empty,
    extensions: std.ArrayList([]const u8) = .empty,
    ext_inst_imports: std.ArrayList(ExtInstImport) = .empty,
    memory_model: ?MemoryModel = null,
    entry_points: std.ArrayList(EntryPoint) = .empty,
    execution_modes: std.ArrayList(ExecutionMode) = .empty,
    names: std.AutoHashMapUnmanaged(Id, []const u8) = .empty,
    member_names: std.AutoHashMapUnmanaged(MemberKey, []const u8) = .empty,
    decorations: std.ArrayList(Decoration) = .empty,
    /// Ids declared by OpDecorationGroup so far.
    decoration_groups: std.AutoHashMapUnmanaged(Id, void) = .empty,
    /// Ids named as a target by an OpGroupDecorate / OpGroupMemberDecorate,
    /// so a later OpDecorationGroup of one of them is refused.
    group_targets: std.AutoHashMapUnmanaged(Id, void) = .empty,
    /// Decorations added by group expansion so far (see `max_expansion_per_word`).
    expanded: usize = 0,
    /// The first enumerant of the instruction being decoded that `enumerants`
    /// does not know (for the trailing-words diagnostic).
    unknown_enumerant: ?struct { kind: spec.OperandKind, value: u32, bits: bool } = null,
    types: std.AutoHashMapUnmanaged(Id, Type) = .empty,
    constants: std.AutoHashMapUnmanaged(Id, Constant) = .empty,
    id_types: std.AutoHashMapUnmanaged(Id, Id) = .empty,
    defined: std.AutoHashMapUnmanaged(Id, void) = .empty,
    globals: std.ArrayList(Global) = .empty,
    functions: std.ArrayList(Function) = .empty,

    // Function being decoded.
    cur_fn: ?struct {
        id: Id,
        result_type: Id,
        control: u32,
        function_type: Id,
        first: u32,
        params: std.ArrayList(Parameter),
        blocks: std.ArrayList(Block),
        cur_block: ?struct { label: Id, first: u32 },
    } = null,

    fn fail(r: *Reader, offset: usize, comptime fmt: []const u8, args: anytype) error{Malformed} {
        return Diagnostics.set(r.diag, offset, fmt, args);
    }

    fn decodeAll(r: *Reader) Error!void {
        var pos: usize = 5;
        while (pos < r.words.len) {
            const header = r.words[pos];
            const wc: u16 = @truncate(header >> 16);
            const opcode: u16 = @truncate(header);
            if (wc == 0) return r.fail(pos, "zero word count", .{});
            if (pos + wc > r.words.len) return r.fail(pos, "instruction of {d} words runs past the end of the module ({d} words)", .{ wc, r.words.len });
            const op = std.enums.fromInt(spec.Opcode, opcode) orelse return r.fail(pos, "unknown opcode {d}", .{opcode});
            const body = r.words[pos + 1 .. pos + wc];
            try r.decodeInstruction(pos, op, opcode, wc, body);
            pos += wc;
        }
        if (r.cur_fn != null) return r.fail(r.words.len, "module ends inside a function (no OpFunctionEnd)", .{});
        try r.linkEntryPoints();
    }

    /// Give every entry point the indices of the execution modes naming its
    /// function, in stream order, and every function the index of the
    /// (last) OpEntryPoint naming it. Done once after decoding, in time and
    /// memory linear in the module: the modes are bucketed by function into
    /// one index array, and entry points of the same function share their
    /// bucket. (Copying an entry's list at each OpExecutionMode, as an
    /// earlier version did, cost E * M^2 / 2 words for E entry points of one
    /// function with M modes: gigabytes from a module of tens of kilobytes.)
    fn linkEntryPoints(r: *Reader) Error!void {
        const Bucket = struct { start: u32 = 0, len: u32 = 0, filled: u32 = 0 };
        var buckets: std.AutoHashMapUnmanaged(Id, Bucket) = .empty;
        for (r.execution_modes.items) |em| {
            const gop = try buckets.getOrPut(r.a, em.function);
            if (!gop.found_existing) gop.value_ptr.* = .{};
            gop.value_ptr.len += 1;
        }
        var next: u32 = 0;
        var it = buckets.valueIterator();
        while (it.next()) |b| {
            b.start = next;
            next += b.len;
        }
        const order = try r.a.alloc(u32, r.execution_modes.items.len);
        for (r.execution_modes.items, 0..) |em, k| {
            const b = buckets.getPtr(em.function).?;
            order[b.start + b.filled] = @intCast(k);
            b.filled += 1;
        }
        var entry_of: std.AutoHashMapUnmanaged(Id, u32) = .empty;
        for (r.entry_points.items, 0..) |*e, k| {
            if (buckets.get(e.function)) |b| e.execution_modes = order[b.start..][0..b.len];
            try entry_of.put(r.a, e.function, @intCast(k));
        }
        for (r.functions.items) |*f| f.entry_point = entry_of.get(f.id);
    }

    fn checkId(r: *Reader, offset: usize, id: Id) error{Malformed}!void {
        if (id == 0) return r.fail(offset, "id 0 is not a valid id", .{});
        if (id >= r.bound) return r.fail(offset, "id %{d} is at or above the bound {d}", .{ id, r.bound });
    }

    fn decodeInstruction(r: *Reader, offset: usize, op: spec.Opcode, opcode: u16, wc: u16, body: []const u32) Error!void {
        r.unknown_enumerant = null;
        var ops: std.ArrayList(Operand) = .empty;
        var result_type: ?Id = null;
        var result: ?Id = null;
        var ext_set: ?spec.InstructionSet = null;
        var i: usize = 0; // index into body

        const grammar = op.operands();
        var gi: usize = 0;
        while (gi < grammar.len) : (gi += 1) {
            const g = grammar[gi];
            switch (g.kind) {
                .id_result_type => {
                    if (i >= body.len) return r.fail(offset, "{t}: missing result type", .{op});
                    try r.checkId(offset, body[i]);
                    result_type = body[i];
                    i += 1;
                    continue;
                },
                .id_result => {
                    if (i >= body.len) return r.fail(offset, "{t}: missing result id", .{op});
                    try r.checkId(offset, body[i]);
                    result = body[i];
                    i += 1;
                    continue;
                },
                else => {},
            }
            switch (g.quantifier) {
                .required => {
                    if (i >= body.len) return r.fail(offset, "{t}: missing required operand {d} ({t})", .{ op, gi, g.kind });
                    i = try r.decodeOperand(offset, op, g.kind, body, i, &ops, result_type, &ext_set);
                },
                .optional => {
                    if (i >= body.len) break;
                    i = try r.decodeOperand(offset, op, g.kind, body, i, &ops, result_type, &ext_set);
                },
                .variadic => {
                    if (op == .OpExtInst) {
                        // The operands after the instruction number come from the set's own table.
                        i = try r.decodeExtInstTail(offset, body, i, &ops, ext_set);
                        break;
                    }
                    if (op == .OpSwitch) {
                        i = try r.decodeSwitchTail(offset, body, i, &ops);
                        break;
                    }
                    while (i < body.len) {
                        i = try r.decodeOperand(offset, op, g.kind, body, i, &ops, result_type, &ext_set);
                    }
                },
            }
        }
        if (op == .OpSpecConstantOp) {
            // The grammar ends at the opcode literal; what follows are the
            // operands of that opcode (SPIR-V 3.52.12), decoded with its own
            // grammar minus result type and result id, so CompositeExtract /
            // CompositeInsert / VectorShuffle keep their literal indexes.
            const inner = try r.specConstantOpcode(offset, ops.items[0].value.literal);
            for (inner.operands(), 0..) |g, gj| {
                if (g.kind == .id_result_type or g.kind == .id_result) continue;
                switch (g.quantifier) {
                    .required => {
                        if (i >= body.len) return r.fail(offset, "OpSpecConstantOp {t}: missing required operand {d} ({t})", .{ inner, gj, g.kind });
                        i = try r.decodeOperand(offset, op, g.kind, body, i, &ops, result_type, &ext_set);
                    },
                    .optional => {
                        if (i >= body.len) break;
                        i = try r.decodeOperand(offset, op, g.kind, body, i, &ops, result_type, &ext_set);
                    },
                    .variadic => while (i < body.len) {
                        i = try r.decodeOperand(offset, op, g.kind, body, i, &ops, result_type, &ext_set);
                    },
                }
            }
        }
        if (i < body.len) {
            // An enumerant the grammar tables do not know may well take the
            // words that follow; say so rather than calling them trailing.
            if (r.unknown_enumerant) |u| {
                if (u.bits) return r.fail(offset, "{t}: {t} bit(s) 0x{x} are not in the SPIR-V grammar tables, so their parameters ({d} word(s)) cannot be decoded", .{ op, u.kind, u.value, body.len - i });
                return r.fail(offset, "{t}: {t} enumerant {d} is not in the SPIR-V grammar tables, so its parameters ({d} word(s)) cannot be decoded", .{ op, u.kind, u.value, body.len - i });
            }
            return r.fail(offset, "{t}: {d} trailing word(s) the grammar does not account for", .{ op, body.len - i });
        }

        if (result) |id| {
            if (r.defined.contains(id)) return r.fail(offset, "{t}: result id %{d} is already defined", .{ op, id });
            try r.defined.put(r.a, id, {});
            if (result_type) |t| try r.id_types.put(r.a, id, t);
        }

        const inst: Instruction = .{
            .opcode = opcode,
            .op = op,
            .word_offset = @intCast(offset),
            .word_count = wc,
            .result_type = result_type,
            .result = result,
            .operands = try ops.toOwnedSlice(r.a),
            .ext_set = ext_set,
        };
        const index: u32 = @intCast(r.instructions.items.len);
        try r.instructions.append(r.a, inst);
        try r.record(offset, inst, index);
    }

    /// The opcode an OpSpecConstantOp names, refused when it is no opcode
    /// spec.zig knows (a word >= 0x10000 can never be one).
    fn specConstantOpcode(r: *Reader, offset: usize, lit: u32) error{Malformed}!spec.Opcode {
        const opc = std.math.cast(u16, lit) orelse return r.fail(offset, "OpSpecConstantOp: opcode {d} out of range", .{lit});
        return std.enums.fromInt(spec.Opcode, opc) orelse return r.fail(offset, "OpSpecConstantOp: unknown opcode {d}", .{lit});
    }

    fn decodeOperand(
        r: *Reader,
        offset: usize,
        op: spec.Opcode,
        kind: spec.OperandKind,
        body: []const u32,
        start: usize,
        ops: *std.ArrayList(Operand),
        result_type: ?Id,
        ext_set: *?spec.InstructionSet,
    ) Error!usize {
        var i = start;
        if (i >= body.len) return r.fail(offset, "{t}: operand {t} runs past the instruction", .{ op, kind });
        switch (kind.category()) {
            .id => {
                try r.checkId(offset, body[i]);
                try ops.append(r.a, .{ .kind = kind, .value = .{ .id = body[i] } });
                // OpExtInst's set operand selects the table for the instruction number.
                if (op == .OpExtInst and ops.items.len == 1) {
                    if (r.findExtSet(body[i])) |e| ext_set.* = e.set;
                }
                return i + 1;
            },
            .literal => switch (kind) {
                .literal_string => {
                    const s = try r.decodeString(offset, body, &i);
                    try ops.append(r.a, .{ .kind = kind, .value = .{ .string = s } });
                    return i;
                },
                .literal_context_dependent_number => {
                    const width = if (result_type) |t| (r.widthOf(t) orelse 32) else 32;
                    if (width > 64) return r.fail(offset, "{t}: a {d}-bit literal is wider than the two words this reader decodes", .{ op, width });
                    if (width > 32) {
                        if (i + 1 >= body.len) return r.fail(offset, "{t}: a {d}-bit literal needs two words", .{ op, width });
                        const v: u64 = @as(u64, body[i]) | (@as(u64, body[i + 1]) << 32);
                        try ops.append(r.a, .{ .kind = kind, .value = .{ .literal64 = v } });
                        return i + 2;
                    }
                    try ops.append(r.a, .{ .kind = kind, .value = .{ .literal = body[i] } });
                    return i + 1;
                },
                else => {
                    try ops.append(r.a, .{ .kind = kind, .value = .{ .literal = body[i] } });
                    return i + 1;
                },
            },
            .value_enum => {
                const v = body[i];
                try ops.append(r.a, .{ .kind = kind, .value = .{ .literal = v } });
                i += 1;
                if (enumerant(kind, v)) |e| {
                    for (e.parameters) |p| {
                        i = try r.decodeOperand(offset, op, p.kind, body, i, ops, result_type, ext_set);
                    }
                } else if (r.unknown_enumerant == null) r.unknown_enumerant = .{ .kind = kind, .value = v, .bits = false };
                return i;
            },
            .bit_enum => {
                const v = body[i];
                try ops.append(r.a, .{ .kind = kind, .value = .{ .literal = v } });
                i += 1;
                // Parameters of every set bit, in enumerant (ascending bit) order.
                var known: u32 = 0;
                for (enumerants(kind)) |e| {
                    if (e.value != 0 and (v & e.value) == e.value) {
                        known |= e.value;
                        for (e.parameters) |p| {
                            i = try r.decodeOperand(offset, op, p.kind, body, i, ops, result_type, ext_set);
                        }
                    }
                }
                if (v & ~known != 0 and r.unknown_enumerant == null) r.unknown_enumerant = .{ .kind = kind, .value = v & ~known, .bits = true };
                return i;
            },
            .composite => {
                for (kind.bases()) |b| {
                    i = try r.decodeOperand(offset, op, b, body, i, ops, result_type, ext_set);
                }
                return i;
            },
        }
    }

    fn decodeExtInstTail(r: *Reader, offset: usize, body: []const u32, start: usize, ops: *std.ArrayList(Operand), ext_set: ?spec.InstructionSet) Error!usize {
        var i = start;
        // ops = [set id, instruction number]
        const number: u32 = if (ops.items.len >= 2) ops.items[1].value.literal else 0;
        const table: ?[]const spec.Operand = if (ext_set) |set| set.operands(number) else null;
        if (table) |grammar| {
            var dummy: ?spec.InstructionSet = null;
            for (grammar) |g| {
                switch (g.quantifier) {
                    .required => {
                        if (i >= body.len) return r.fail(offset, "OpExtInst {d}: missing required operand ({t})", .{ number, g.kind });
                        i = try r.decodeOperand(offset, .OpExtInst, g.kind, body, i, ops, null, &dummy);
                    },
                    .optional => {
                        if (i >= body.len) break;
                        i = try r.decodeOperand(offset, .OpExtInst, g.kind, body, i, ops, null, &dummy);
                    },
                    .variadic => while (i < body.len) {
                        i = try r.decodeOperand(offset, .OpExtInst, g.kind, body, i, ops, null, &dummy);
                    },
                }
            }
            if (i < body.len) return r.fail(offset, "OpExtInst {d}: {d} trailing word(s)", .{ number, body.len - i });
            return i;
        }
        while (i < body.len) : (i += 1) {
            try r.checkId(offset, body[i]);
            try ops.append(r.a, .{ .kind = .id_ref, .value = .{ .id = body[i] } });
        }
        return i;
    }

    fn decodeSwitchTail(r: *Reader, offset: usize, body: []const u32, start: usize, ops: *std.ArrayList(Operand)) Error!usize {
        var i = start;
        // ops = [selector, default label]
        const selector = ops.items[0].value.id;
        const width = if (r.id_types.get(selector)) |t| (r.widthOf(t) orelse 32) else 32;
        if (width > 64) return r.fail(offset, "OpSwitch: a {d}-bit selector is wider than the two words this reader decodes", .{width});
        while (i < body.len) {
            if (width > 32) {
                if (i + 2 >= body.len) return r.fail(offset, "OpSwitch: a 64-bit case needs two literal words and a label", .{});
                const v: u64 = @as(u64, body[i]) | (@as(u64, body[i + 1]) << 32);
                try ops.append(r.a, .{ .kind = .literal_integer, .value = .{ .literal64 = v } });
                i += 2;
            } else {
                if (i + 1 >= body.len) return r.fail(offset, "OpSwitch: a case needs a literal and a label", .{});
                try ops.append(r.a, .{ .kind = .literal_integer, .value = .{ .literal = body[i] } });
                i += 1;
            }
            try r.checkId(offset, body[i]);
            try ops.append(r.a, .{ .kind = .id_ref, .value = .{ .id = body[i] } });
            i += 1;
        }
        return i;
    }

    fn decodeString(r: *Reader, offset: usize, body: []const u32, i: *usize) Error![]const u8 {
        var bytes: std.ArrayList(u8) = .empty;
        while (true) {
            if (i.* >= body.len) return r.fail(offset, "unterminated string literal", .{});
            const w = body[i.*];
            i.* += 1;
            var k: u5 = 0;
            while (k < 4) : (k += 1) {
                const b: u8 = @truncate(w >> (@as(u5, @intCast(k)) * 8));
                if (b == 0) return bytes.toOwnedSlice(r.a);
                try bytes.append(r.a, b);
            }
        }
    }

    fn widthOf(r: *Reader, type_id: Id) ?u32 {
        const t = r.types.get(type_id) orelse return null;
        return switch (t) {
            .int => |x| x.width,
            .float => |x| x.width,
            else => null,
        };
    }

    fn findExtSet(r: *Reader, id: Id) ?ExtInstImport {
        for (r.ext_inst_imports.items) |e| if (e.id == id) return e;
        return null;
    }

    fn ids(r: *Reader, operands: []const Operand, from: usize) Error![]const Id {
        var list: std.ArrayList(Id) = .empty;
        for (operands[from..]) |o| {
            if (o.asId()) |id| try list.append(r.a, id);
        }
        return list.toOwnedSlice(r.a);
    }

    /// Fill the module tables from one decoded instruction and track function
    /// / block structure.
    fn record(r: *Reader, offset: usize, inst: Instruction, index: u32) Error!void {
        const op = inst.op;
        const in_fn = r.cur_fn != null;
        if (in_fn) {
            switch (op) {
                .OpFunction => return r.fail(offset, "OpFunction inside a function", .{}),
                .OpFunctionParameter => {
                    const f = &r.cur_fn.?;
                    if (f.blocks.items.len != 0 or f.cur_block != null) return r.fail(offset, "OpFunctionParameter after the first block", .{});
                    try f.params.append(r.a, .{ .id = inst.result.?, .type = inst.result_type.? });
                    return;
                },
                .OpLabel => {
                    const f = &r.cur_fn.?;
                    if (f.cur_block != null) return r.fail(offset, "OpLabel inside a block (the previous block has no terminator)", .{});
                    f.cur_block = .{ .label = inst.result.?, .first = index };
                    return;
                },
                .OpFunctionEnd => {
                    var f = r.cur_fn.?;
                    if (f.cur_block != null) return r.fail(offset, "OpFunctionEnd inside a block (no terminator)", .{});
                    try r.functions.append(r.a, .{
                        .id = f.id,
                        .result_type = f.result_type,
                        .control = f.control,
                        .function_type = f.function_type,
                        .params = try f.params.toOwnedSlice(r.a),
                        .blocks = try f.blocks.toOwnedSlice(r.a),
                        .first = f.first,
                        .end = index + 1,
                        .entry_point = null, // set by `linkEntryPoints`
                    });
                    r.cur_fn = null;
                    return;
                },
                else => {
                    const f = &r.cur_fn.?;
                    // OpLine / OpNoLine may sit between OpFunction and the first OpLabel.
                    if (f.cur_block == null) {
                        if (op == .OpLine or op == .OpNoLine) return;
                        return r.fail(offset, "{t} outside a block (no OpLabel yet)", .{op});
                    }
                    if (inst.result_type) |t| {
                        if (op == .OpVariable) {
                            // Function-scope variables carry no entry in globals.
                            _ = t;
                        }
                    }
                    if (isTerminator(op)) {
                        const b = f.cur_block.?;
                        try f.blocks.append(r.a, .{ .label = b.label, .first = b.first, .end = index + 1 });
                        f.cur_block = null;
                    }
                    return;
                },
            }
        }
        switch (op) {
            .OpCapability => try r.capabilities.append(r.a, inst.operands[0].value.literal),
            .OpExtension => try r.extensions.append(r.a, inst.operands[0].value.string),
            .OpExtInstImport => {
                const s = inst.operands[0].value.string;
                const set: ?spec.InstructionSet = if (std.mem.eql(u8, s, "GLSL.std.450")) .@"GLSL.std.450" else if (std.mem.eql(u8, s, "OpenCL.std")) .@"OpenCL.std" else null;
                try r.ext_inst_imports.append(r.a, .{ .id = inst.result.?, .name = s, .set = set });
            },
            .OpMemoryModel => r.memory_model = .{ .addressing = inst.operands[0].value.literal, .memory = inst.operands[1].value.literal },
            .OpEntryPoint => {
                try r.entry_points.append(r.a, .{
                    .execution_model = inst.operands[0].value.literal,
                    .function = inst.operands[1].value.id,
                    .name = inst.operands[2].value.string,
                    .interface = try r.ids(inst.operands, 3),
                    .execution_modes = &.{},
                });
            },
            .OpExecutionMode, .OpExecutionModeId => {
                // Linked to the entry points once the stream is decoded
                // (`linkEntryPoints`), so the cost stays linear.
                try r.execution_modes.append(r.a, .{
                    .function = inst.operands[0].value.id,
                    .mode = inst.operands[1].value.literal,
                    .operands = inst.operands[2..],
                });
            },
            .OpName => try r.names.put(r.a, inst.operands[0].value.id, inst.operands[1].value.string),
            .OpMemberName => try r.member_names.put(r.a, .{ .id = inst.operands[0].value.id, .member = inst.operands[1].value.literal }, inst.operands[2].value.string),
            .OpDecorate, .OpDecorateId => {
                const target = inst.operands[0].value.id;
                if (r.decoration_groups.contains(target)) return r.fail(offset, "{t} of decoration group %{d} after its OpDecorationGroup (the spec requires the group's decorations to precede it)", .{ op, target });
                try r.decorations.append(r.a, .{
                    .target = target,
                    .member = null,
                    .decoration = inst.operands[1].value.literal,
                    .operands = inst.operands[2..],
                });
            },
            .OpDecorationGroup => {
                const group = inst.result.?;
                if (r.group_targets.contains(group)) return r.fail(offset, "OpDecorationGroup %{d}: an earlier OpGroupDecorate targets it (a decoration group may not be the target of one)", .{group});
                try r.decoration_groups.put(r.a, group, {});
            },
            .OpGroupDecorate => {
                const group = try r.decorationGroup(offset, op, inst.operands[0]);
                for (inst.operands[1..]) |t| {
                    const target = t.asId() orelse return r.fail(offset, "{t}: expected a target id", .{op});
                    try r.expandGroup(offset, op, group, target, null);
                }
            },
            .OpGroupMemberDecorate => {
                const group = try r.decorationGroup(offset, op, inst.operands[0]);
                const pairs = inst.operands[1..];
                if (pairs.len % 2 != 0) return r.fail(offset, "{t}: odd (target, member) operand count", .{op});
                var i: usize = 0;
                while (i < pairs.len) : (i += 2) {
                    const target = pairs[i].asId() orelse return r.fail(offset, "{t}: expected a target id", .{op});
                    const member: u32 = switch (pairs[i + 1].value) {
                        .literal => |v| v,
                        else => return r.fail(offset, "{t}: expected a member index", .{op}),
                    };
                    try r.expandGroup(offset, op, group, target, member);
                }
            },
            .OpMemberDecorate => try r.decorations.append(r.a, .{
                .target = inst.operands[0].value.id,
                .member = inst.operands[1].value.literal,
                .decoration = inst.operands[2].value.literal,
                .operands = inst.operands[3..],
            }),
            .OpTypeVoid => try r.putType(offset, inst.result.?, .void),
            .OpTypeBool => try r.putType(offset, inst.result.?, .bool),
            .OpTypeInt => {
                const width = inst.operands[0].value.literal;
                if (width == 0) return r.fail(offset, "OpTypeInt: width 0", .{});
                try r.putType(offset, inst.result.?, .{ .int = .{ .width = width, .signed = inst.operands[1].value.literal != 0 } });
            },
            .OpTypeFloat => try r.putType(offset, inst.result.?, .{ .float = .{
                .width = inst.operands[0].value.literal,
                .encoding = if (inst.operands.len > 1) inst.operands[1].value.literal else null,
            } }),
            .OpTypeVector => try r.putType(offset, inst.result.?, .{ .vector = .{ .elem = inst.operands[0].value.id, .len = inst.operands[1].value.literal } }),
            .OpTypeMatrix => try r.putType(offset, inst.result.?, .{ .matrix = .{ .column = inst.operands[0].value.id, .cols = inst.operands[1].value.literal } }),
            .OpTypeArray => try r.putType(offset, inst.result.?, .{ .array = .{ .elem = inst.operands[0].value.id, .len = inst.operands[1].value.id } }),
            .OpTypeRuntimeArray => try r.putType(offset, inst.result.?, .{ .runtime_array = .{ .elem = inst.operands[0].value.id } }),
            .OpTypeStruct => try r.putType(offset, inst.result.?, .{ .@"struct" = .{ .members = try r.ids(inst.operands, 0) } }),
            .OpTypePointer => try r.putType(offset, inst.result.?, .{ .pointer = .{ .storage_class = inst.operands[0].value.literal, .pointee = inst.operands[1].value.id } }),
            .OpTypeFunction => try r.putType(offset, inst.result.?, .{ .function = .{ .ret = inst.operands[0].value.id, .params = try r.ids(inst.operands, 1) } }),
            .OpTypeForwardPointer => {
                // Declares the id; the OpTypePointer with that id comes later.
                _ = r.types.get(inst.operands[0].value.id) orelse try r.types.put(r.a, inst.operands[0].value.id, .{ .forward_pointer = .{ .storage_class = inst.operands[1].value.literal } });
            },
            .OpTypeImage, .OpTypeSampler, .OpTypeSampledImage, .OpTypeOpaque, .OpTypeEvent, .OpTypeDeviceEvent, .OpTypeReserveId, .OpTypeQueue, .OpTypePipe, .OpTypeAccelerationStructureKHR, .OpTypeRayQueryKHR, .OpTypeCooperativeMatrixKHR => try r.putType(offset, inst.result.?, .{ .other = op }),
            .OpConstantTrue, .OpSpecConstantTrue => try r.putConstant(inst, .{ .bool = true }, op == .OpSpecConstantTrue),
            .OpConstantFalse, .OpSpecConstantFalse => try r.putConstant(inst, .{ .bool = false }, op == .OpSpecConstantFalse),
            .OpConstant, .OpSpecConstant => try r.putConstant(inst, .{ .scalar = inst.operands[0].asInt().? }, op == .OpSpecConstant),
            .OpConstantComposite, .OpSpecConstantComposite => try r.putConstant(inst, .{ .composite = try r.ids(inst.operands, 0) }, op == .OpSpecConstantComposite),
            .OpConstantNull => try r.putConstant(inst, .null, false),
            .OpUndef => try r.putConstant(inst, .undef, false),
            .OpSpecConstantOp => try r.putConstant(inst, .{ .spec_op = .{ .opcode = try r.specConstantOpcode(offset, inst.operands[0].value.literal), .operands = inst.operands[1..] } }, true),
            .OpConstantSampler => try r.putConstant(inst, .{ .other = op }, false),
            .OpVariable => try r.globals.append(r.a, .{
                .id = inst.result.?,
                .type = inst.result_type.?,
                .storage_class = inst.operands[0].value.literal,
                .initializer = if (inst.operands.len > 1) inst.operands[1].value.id else null,
                .inst = index,
            }),
            .OpFunction => {
                r.cur_fn = .{
                    .id = inst.result.?,
                    .result_type = inst.result_type.?,
                    .control = inst.operands[0].value.literal,
                    .function_type = inst.operands[1].value.id,
                    .first = index,
                    .params = .empty,
                    .blocks = .empty,
                    .cur_block = null,
                };
            },
            .OpFunctionParameter, .OpLabel, .OpFunctionEnd => return r.fail(offset, "{t} outside a function", .{op}),
            .OpNop, .OpSourceContinued, .OpSource, .OpSourceExtension, .OpString, .OpLine, .OpNoLine, .OpModuleProcessed => {},
            else => {
                if (op.class() != .annotation and op.class() != .debug and op.class() != .type_declaration and op.class() != .constant_creation and op.class() != .mode_setting and op.class() != .extension and op.class() != .miscellaneous) {
                    return r.fail(offset, "{t} outside a function", .{op});
                }
            },
        }
    }

    /// The group id an OpGroupDecorate / OpGroupMemberDecorate names, which
    /// must have been declared by an earlier OpDecorationGroup.
    fn decorationGroup(r: *Reader, offset: usize, op: spec.Opcode, operand: Operand) Error!Id {
        const group = operand.asId() orelse return r.fail(offset, "{t}: expected a decoration group id", .{op});
        if (!r.decoration_groups.contains(group)) return r.fail(offset, "{t}: %{d} is not a decoration group declared earlier", .{ op, group });
        return group;
    }

    /// Copy every whole-id decoration on `group` to `target` (or to member
    /// `member` of it). The group's decorations all precede its
    /// OpDecorationGroup, which precedes every use, so the list is complete.
    /// A target that is a decoration group is refused (it would copy the
    /// group onto itself, or two groups onto each other, and grow the table
    /// geometrically), and so is an expansion past the cap.
    fn expandGroup(r: *Reader, offset: usize, op: spec.Opcode, group: Id, target: Id, member: ?u32) Error!void {
        if (r.decoration_groups.contains(target)) return r.fail(offset, "{t}: target %{d} is a decoration group", .{ op, target });
        try r.group_targets.put(r.a, target, {});
        const limit = max_expansion_per_word * r.words.len;
        const n = r.decorations.items.len;
        for (0..n) |i| {
            const d = r.decorations.items[i];
            if (d.target != group or d.member != null) continue;
            if (r.expanded >= limit) return r.fail(offset, "{t}: decoration groups expand to more than {d} decorations ({d} per word of a {d}-word module)", .{ op, limit, max_expansion_per_word, r.words.len });
            r.expanded += 1;
            try r.decorations.append(r.a, .{ .target = target, .member = member, .decoration = d.decoration, .operands = d.operands });
        }
    }

    fn putType(r: *Reader, offset: usize, id: Id, t: Type) Error!void {
        if (r.types.get(id)) |existing| {
            if (existing != .forward_pointer) return r.fail(offset, "type %{d} declared twice", .{id});
        }
        try r.types.put(r.a, id, t);
    }

    fn putConstant(r: *Reader, inst: Instruction, v: Constant.Value, is_spec: bool) Error!void {
        try r.constants.put(r.a, inst.result.?, .{ .type = inst.result_type.?, .spec_constant = is_spec, .value = v });
    }
};

/// The most decorations decoration-group expansion may add per word of the
/// module. A legitimate group carries a handful of decorations (the
/// SPIRV-LLVM-Translator's are one to three: FuncParamAttr, Alignment,
/// NonWritable) and each target costs a word, so real modules stay far
/// below this; an adversarial one (thousands of decorations on one group,
/// thousands of targets) would otherwise grow quadratically in its size.
/// At `@sizeOf(Decoration)` (32 bytes on a 64-bit host) per entry this
/// bounds the expanded table at 512 bytes per input word.
pub const max_expansion_per_word = 16;

/// `v` with every bit at or above `width` cleared (`width` >= 64 keeps all).
pub fn truncateBits(v: u64, width: u32) u64 {
    if (width >= 64) return v;
    return v & ((@as(u64, 1) << @intCast(width)) - 1);
}

/// Block terminators (SPIR-V §2.2.5 and §3.42.17).
pub fn isTerminator(op: spec.Opcode) bool {
    return switch (op) {
        .OpBranch, .OpBranchConditional, .OpSwitch, .OpReturn, .OpReturnValue, .OpKill, .OpUnreachable, .OpTerminateInvocation, .OpTerminateRayKHR, .OpIgnoreIntersectionKHR, .OpEmitMeshTasksEXT => true,
        else => false,
    };
}

/// Every enumerant of a value-enum or bit-enum `kind` in the SPIR-V 1.6
/// rev 7 grammar: spec.zig's vendored table followed by the entries it
/// lacks (spec_extra.zig); a bit enum's table is in ascending bit order,
/// the order in which the parameters of its set bits follow it (SPIR-V
/// 2.2.2, "Instructions": operands of a mask are ordered by bit). Empty for
/// any other kind. Every enumerant lookup of reader, dis and asm goes
/// through here, so no tool sees spec.zig's gaps.
pub fn enumerants(kind: spec.OperandKind) []const spec.Enumerant {
    switch (kind.category()) {
        .value_enum, .bit_enum => {},
        else => return &.{},
    }
    return switch (kind) {
        inline else => |k| comptime mergedEnumerants(k),
    };
}

fn mergedEnumerants(comptime kind: spec.OperandKind) []const spec.Enumerant {
    @setEvalBranchQuota(200_000);
    switch (kind.category()) {
        .value_enum, .bit_enum => {},
        else => return &.{},
    }
    const vendored = kind.enumerants();
    const extra = spec_extra.enumerants(kind);
    var all: [vendored.len + extra.len]spec.Enumerant = undefined;
    for (vendored, 0..) |e, i| all[i] = e;
    for (extra, 0..) |e, i| all[vendored.len + i] = e;
    if (kind.category() == .bit_enum) {
        // Stable insertion sort by value; aliases keep their table order.
        var i: usize = 1;
        while (i < all.len) : (i += 1) {
            const x = all[i];
            var j = i;
            while (j > 0 and all[j - 1].value > x.value) : (j -= 1) all[j] = all[j - 1];
            all[j] = x;
        }
    }
    const final = all;
    return &final;
}

/// The enumerant of `kind` with value `v` (the first one listed when the
/// grammar has aliases), or null.
pub fn enumerant(kind: spec.OperandKind, v: u32) ?spec.Enumerant {
    for (enumerants(kind)) |e| if (e.value == v) return e;
    return null;
}

/// The enumerant of `kind` named `name`, or null.
pub fn enumerantByName(kind: spec.OperandKind, name: []const u8) ?spec.Enumerant {
    for (enumerants(kind)) |e| if (std.mem.eql(u8, e.name, name)) return e;
    return null;
}

// ───────────────────────────── tests ─────────────────────────────

const testing = std.testing;

/// Builds word arrays for the tests: `inst(.OpTypeInt, &.{ 1, 32, 1 })`.
pub const WordBuilder = struct {
    words: std.ArrayList(u32) = .empty,
    a: std.mem.Allocator,

    pub fn init(a: std.mem.Allocator, bound: u32) !WordBuilder {
        var b: WordBuilder = .{ .a = a };
        try b.words.appendSlice(a, &.{ spec.magic_number, 0x00010000, 0, bound, 0 });
        return b;
    }

    pub fn inst(b: *WordBuilder, op: spec.Opcode, operands: []const u32) !void {
        const wc: u32 = @intCast(operands.len + 1);
        try b.words.append(b.a, (wc << 16) | @as(u32, @backingInt(op)));
        try b.words.appendSlice(b.a, operands);
    }

    /// An instruction whose last operand is a string.
    pub fn instStr(b: *WordBuilder, op: spec.Opcode, operands: []const u32, s: []const u8) !void {
        var tail: std.ArrayList(u32) = .empty;
        defer tail.deinit(b.a);
        try tail.appendSlice(b.a, operands);
        try appendString(&tail, b.a, s);
        try b.inst(op, tail.items);
    }

    pub fn bytes(b: *WordBuilder) ![]u8 {
        return b.a.dupe(u8, std.mem.sliceAsBytes(b.words.items));
    }
};

pub fn appendString(list: *std.ArrayList(u32), a: std.mem.Allocator, s: []const u8) !void {
    var i: usize = 0;
    while (i <= s.len) : (i += 4) {
        var w: u32 = 0;
        var k: usize = 0;
        while (k < 4 and i + k < s.len) : (k += 1) w |= @as(u32, s[i + k]) << @intCast(k * 8);
        try list.append(a, w);
    }
}

test "header: version, generator, bound, schema, both endiannesses" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var b = try WordBuilder.init(a, 7);
    b.words.items[1] = 0x00010300; // 1.3
    b.words.items[2] = 0x00290900;
    try b.inst(.OpCapability, &.{@backingInt(spec.Capability.shader)});
    try b.inst(.OpMemoryModel, &.{ 0, 1 });
    const le = try b.bytes();
    var m = try read(testing.allocator, le, null);
    defer m.deinit();
    try testing.expectEqual(@as(u8, 1), m.version_major);
    try testing.expectEqual(@as(u8, 3), m.version_minor);
    try testing.expectEqual(@as(u32, 0x00290900), m.generator);
    try testing.expectEqual(@as(u32, 7), m.bound);
    try testing.expectEqual(@as(u32, 0), m.schema);
    try testing.expectEqual(@as(usize, 1), m.capabilities.len);
    try testing.expectEqual(@as(u32, 1), m.capabilities[0]);
    try testing.expectEqual(@as(u32, 0), m.memory_model.?.addressing);
    try testing.expectEqual(@as(u32, 1), m.memory_model.?.memory);

    // Byte-swapped: every word reversed.
    const be = try a.alloc(u8, le.len);
    var i: usize = 0;
    while (i < le.len) : (i += 4) {
        be[i] = le[i + 3];
        be[i + 1] = le[i + 2];
        be[i + 2] = le[i + 1];
        be[i + 3] = le[i];
    }
    var mb = try read(testing.allocator, be, null);
    defer mb.deinit();
    try testing.expectEqual(@as(u32, 0x00290900), mb.generator);
    try testing.expectEqual(@as(usize, 2), mb.instructions.len);
    try testing.expectEqual(spec.Opcode.OpMemoryModel, mb.instructions[1].op);
}

test "every operand category: ids, literals, strings, value enums with parameters, bit enums with parameters, composites, context-dependent 64-bit numbers" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var b = try WordBuilder.init(a, 40);
    try b.inst(.OpCapability, &.{@backingInt(spec.Capability.shader)});
    try b.inst(.OpCapability, &.{@backingInt(spec.Capability.int64)});
    try b.instStr(.OpExtInstImport, &.{1}, "GLSL.std.450");
    try b.inst(.OpMemoryModel, &.{ 0, 1 });
    try b.instStr(.OpEntryPoint, &.{ 5, 10 }, "main"); // GLCompute %10 "main"
    try b.inst(.OpExecutionMode, &.{ 10, 17, 64, 1, 1 }); // LocalSize 64 1 1
    try b.instStr(.OpName, &.{10}, "main");
    try b.instStr(.OpMemberName, &.{ 20, 0 }, "data");
    try b.inst(.OpDecorate, &.{ 20, 2 }); // Block
    try b.inst(.OpMemberDecorate, &.{ 20, 0, 35, 16 }); // Offset 16
    try b.inst(.OpDecorate, &.{ 21, 11, 28 }); // BuiltIn GlobalInvocationId
    try b.inst(.OpTypeVoid, &.{2});
    try b.inst(.OpTypeInt, &.{ 3, 32, 0 });
    try b.inst(.OpTypeInt, &.{ 4, 64, 1 });
    try b.inst(.OpTypeFloat, &.{ 5, 32 });
    try b.inst(.OpTypeFloat, &.{ 6, 64 });
    try b.inst(.OpConstant, &.{ 3, 7, 42 });
    try b.inst(.OpConstant, &.{ 4, 8, 0xFFFFFFFE, 0xFFFFFFFF }); // i64 -2
    try b.inst(.OpConstant, &.{ 5, 9, @bitCast(@as(f32, 1.5)) });
    const d: u64 = @bitCast(@as(f64, -0.25));
    try b.inst(.OpConstant, &.{ 6, 11, @truncate(d), @truncate(d >> 32) });
    try b.inst(.OpTypeVector, &.{ 12, 3, 3 });
    try b.inst(.OpConstantComposite, &.{ 12, 13, 7, 7, 7 });
    try b.inst(.OpTypeStruct, &.{ 20, 3 });
    try b.inst(.OpTypePointer, &.{ 14, 12, 20 }); // StorageBuffer
    try b.inst(.OpVariable, &.{ 14, 15, 12 });
    try b.inst(.OpTypePointer, &.{ 16, 7, 3 }); // Function
    try b.inst(.OpTypeFunction, &.{ 17, 2 });
    try b.inst(.OpFunction, &.{ 2, 10, 0, 17 });
    try b.inst(.OpLabel, &.{18});
    try b.inst(.OpVariable, &.{ 16, 19, 7 });
    try b.inst(.OpStore, &.{ 19, 7, 2, 4 }); // Aligned 4
    try b.inst(.OpLoad, &.{ 3, 22, 19, 3, 4 }); // Volatile|Aligned 4
    try b.inst(.OpExtInst, &.{ 5, 23, 1, 31, 9 }); // Sqrt %9
    try b.inst(.OpSelectionMerge, &.{ 24, 0 });
    try b.inst(.OpBranch, &.{24});
    try b.inst(.OpLabel, &.{24});
    try b.inst(.OpReturn, &.{});
    try b.inst(.OpFunctionEnd, &.{});
    var m = try read(testing.allocator, try b.bytes(), null);
    defer m.deinit();

    try testing.expectEqualStrings("GLSL.std.450", m.ext_inst_imports[0].name);
    try testing.expectEqual(spec.InstructionSet.@"GLSL.std.450", m.ext_inst_imports[0].set.?);
    try testing.expectEqualStrings("main", m.entry_points[0].name);
    try testing.expectEqual(spec.ExecutionModel.gl_compute, m.entry_points[0].model().?);
    try testing.expectEqual(@as(usize, 1), m.entry_points[0].execution_modes.len);
    const em = m.execution_modes[m.entry_points[0].execution_modes[0]];
    try testing.expectEqual(spec.ExecutionMode.local_size, em.as().?);
    try testing.expectEqual(@as(u64, 64), em.operands[0].asInt().?);
    try testing.expectEqualStrings("main", m.name(10).?);
    try testing.expectEqualStrings("data", m.memberName(20, 0).?);
    try testing.expect(m.hasDecoration(20, .block));
    try testing.expectEqual(@as(u64, 16), m.memberDecoration(20, 0, .offset).?.firstInt().?);
    try testing.expectEqual(@as(u64, @backingInt(spec.BuiltIn.global_invocation_id)), m.decoration(21, .built_in).?.firstInt().?);
    try testing.expectEqual(@as(u32, 64), m.typeOf(4).?.int.width);
    try testing.expect(m.typeOf(4).?.int.signed);
    try testing.expectEqual(@as(u64, 42), m.constantOf(7).?.value.scalar);
    try testing.expectEqual(@as(u64, 0xFFFFFFFFFFFFFFFE), m.constantOf(8).?.value.scalar);
    try testing.expectEqual(@as(u64, @as(u32, @bitCast(@as(f32, 1.5)))), m.constantOf(9).?.value.scalar);
    try testing.expectEqual(d, m.constantOf(11).?.value.scalar);
    try testing.expectEqual(@as(usize, 3), m.constantOf(13).?.value.composite.len);
    try testing.expectEqual(@as(u32, 12), m.globals[0].storage_class);
    try testing.expectEqual(@as(usize, 1), m.functions.len);
    try testing.expectEqual(@as(usize, 2), m.functions[0].blocks.len);
    try testing.expectEqual(@as(u32, 0), m.functions[0].entry_point.?);
    // The OpStore's Aligned parameter was decoded as a literal after the bit enum.
    const store = m.instructions[m.functions[0].blocks[0].first + 2];
    try testing.expectEqual(spec.Opcode.OpStore, store.op);
    try testing.expectEqual(@as(usize, 4), store.operands.len);
    try testing.expectEqual(spec.OperandKind.memory_access, store.operands[2].kind);
    try testing.expectEqual(@as(u64, 4), store.operands[3].asInt().?);
    const load = m.instructions[m.functions[0].blocks[0].first + 3];
    try testing.expectEqual(@as(u32, 3), load.operands[1].value.literal); // Volatile|Aligned
    try testing.expectEqual(@as(u64, 4), load.operands[2].asInt().?);
    const ext = m.instructions[m.functions[0].blocks[0].first + 4];
    try testing.expectEqual(spec.InstructionSet.@"GLSL.std.450", ext.ext_set.?);
    try testing.expectEqual(@as(u32, 31), ext.operands[1].value.literal);
    try testing.expectEqual(@as(Id, 9), ext.operands[2].value.id);
    try testing.expectEqual(@as(Id, 3), m.resultTypeOf(22).?);
}

test "two functions, a phi loop, a 64-bit switch" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var b = try WordBuilder.init(a, 60);
    try b.inst(.OpCapability, &.{1});
    try b.inst(.OpCapability, &.{@backingInt(spec.Capability.int64)});
    try b.inst(.OpMemoryModel, &.{ 0, 1 });
    try b.instStr(.OpEntryPoint, &.{ 5, 10 }, "k");
    try b.inst(.OpTypeVoid, &.{2});
    try b.inst(.OpTypeInt, &.{ 3, 32, 0 });
    try b.inst(.OpTypeInt, &.{ 4, 64, 0 });
    try b.inst(.OpTypeBool, &.{5});
    try b.inst(.OpConstant, &.{ 3, 6, 0 });
    try b.inst(.OpConstant, &.{ 3, 7, 1 });
    try b.inst(.OpConstant, &.{ 3, 8, 10 });
    try b.inst(.OpTypeFunction, &.{ 9, 2 });
    try b.inst(.OpTypeFunction, &.{ 11, 3, 3 });
    // %12 = helper(%13: u32) -> u32 { loop i from 0 to 10 }
    try b.inst(.OpFunction, &.{ 3, 12, 0, 11 });
    try b.inst(.OpFunctionParameter, &.{ 3, 13 });
    try b.inst(.OpLabel, &.{14});
    try b.inst(.OpBranch, &.{15});
    try b.inst(.OpLabel, &.{15});
    try b.inst(.OpPhi, &.{ 3, 16, 6, 14, 17, 18 }); // i = phi [0, %14], [%17, %18]
    try b.inst(.OpLoopMerge, &.{ 19, 18, 0 });
    try b.inst(.OpULessThan, &.{ 5, 20, 16, 8 });
    try b.inst(.OpBranchConditional, &.{ 20, 18, 19 });
    try b.inst(.OpLabel, &.{18});
    try b.inst(.OpIAdd, &.{ 3, 17, 16, 7 });
    try b.inst(.OpBranch, &.{15});
    try b.inst(.OpLabel, &.{19});
    try b.inst(.OpReturnValue, &.{16});
    try b.inst(.OpFunctionEnd, &.{});
    // %10 = k() { switch on a u64 }
    try b.inst(.OpFunction, &.{ 2, 10, 0, 9 });
    try b.inst(.OpLabel, &.{21});
    try b.inst(.OpFunctionCall, &.{ 3, 22, 12, 6 });
    try b.inst(.OpUConvert, &.{ 4, 23, 22 });
    try b.inst(.OpSelectionMerge, &.{ 24, 0 });
    try b.inst(.OpSwitch, &.{ 23, 24, 0, 1, 25, 7, 0, 26 }); // 1<<32 -> %25, 7 -> %26
    try b.inst(.OpLabel, &.{25});
    try b.inst(.OpBranch, &.{24});
    try b.inst(.OpLabel, &.{26});
    try b.inst(.OpBranch, &.{24});
    try b.inst(.OpLabel, &.{24});
    try b.inst(.OpReturn, &.{});
    try b.inst(.OpFunctionEnd, &.{});
    var m = try read(testing.allocator, try b.bytes(), null);
    defer m.deinit();
    try testing.expectEqual(@as(usize, 2), m.functions.len);
    try testing.expectEqual(@as(usize, 1), m.functions[0].params.len);
    try testing.expectEqual(@as(usize, 4), m.functions[0].blocks.len);
    try testing.expectEqual(@as(Id, 15), m.functions[0].blocks[1].label);
    const phi = m.instructions[m.functions[0].blocks[1].first + 1];
    try testing.expectEqual(spec.Opcode.OpPhi, phi.op);
    try testing.expectEqual(@as(usize, 4), phi.operands.len);
    try testing.expectEqual(@as(Id, 17), phi.operands[2].value.id);
    try testing.expectEqual(@as(usize, 5), m.functions[1].blocks[0].end - m.functions[1].blocks[0].first);
    const sw = m.instructions[m.functions[1].blocks[0].end - 1];
    try testing.expectEqual(spec.Opcode.OpSwitch, sw.op);
    try testing.expectEqual(@as(usize, 6), sw.operands.len);
    try testing.expectEqual(@as(u64, 1 << 32), sw.operands[2].value.literal64);
    try testing.expectEqual(@as(Id, 25), sw.operands[3].value.id);
    try testing.expectEqual(@as(u64, 7), sw.operands[4].value.literal64);
    try testing.expect(m.functions[0].entry_point == null);
    try testing.expectEqual(@as(u32, 0), m.functions[1].entry_point.?);
}

fn expectMalformed(bytes: []const u8, expected_offset: usize, needle: []const u8) !void {
    var diag: Diagnostics = .{};
    const r = read(testing.allocator, bytes, &diag);
    if (r) |*m| {
        var mm = m.*;
        mm.deinit();
        std.debug.print("expected Malformed ({s}) but the module decoded\n", .{needle});
        return error.TestUnexpectedResult;
    } else |err| {
        try testing.expectEqual(error.Malformed, err);
        if (std.mem.indexOf(u8, diag.message, needle) == null) {
            std.debug.print("message '{s}' does not contain '{s}'\n", .{ diag.message, needle });
            return error.TestUnexpectedResult;
        }
        try testing.expectEqual(expected_offset, diag.word_offset);
    }
}

test "malformed inputs name the word offset" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try expectMalformed("\x03\x02\x23\x07\x00\x00\x01\x00\x00", 0, "multiple of 4");
    try expectMalformed("\x03\x02\x23\x07\x00\x00\x01\x00", 0, "shorter than");
    try expectMalformed("\x04\x02\x23\x07\x00\x00\x01\x00\x00\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00\x00", 0, "bad magic");
    {
        var b = try WordBuilder.init(a, 8);
        try b.inst(.OpCapability, &.{1});
        try b.words.append(a, 0x00000011); // word count 0
        try expectMalformed(try b.bytes(), 7, "zero word count");
    }
    {
        var b = try WordBuilder.init(a, 8);
        try b.words.append(a, (5 << 16) | 17); // claims 5 words, has 1
        try b.words.append(a, 1);
        try expectMalformed(try b.bytes(), 5, "runs past the end");
    }
    {
        var b = try WordBuilder.init(a, 8);
        try b.words.append(a, (1 << 16) | 9); // opcode 9 does not exist
        try expectMalformed(try b.bytes(), 5, "unknown opcode 9");
    }
    {
        var b = try WordBuilder.init(a, 8);
        try b.inst(.OpTypeInt, &.{ 3, 32 }); // missing signedness
        try expectMalformed(try b.bytes(), 5, "missing required operand");
    }
    {
        var b = try WordBuilder.init(a, 8);
        try b.inst(.OpName, &.{ 3, 0x61616161 }); // "aaaa" without a terminator
        try expectMalformed(try b.bytes(), 5, "unterminated string");
    }
    {
        var b = try WordBuilder.init(a, 3);
        try b.inst(.OpTypeInt, &.{ 3, 32, 0 }); // id 3 >= bound 3
        try expectMalformed(try b.bytes(), 5, "at or above the bound");
    }
    {
        var b = try WordBuilder.init(a, 8);
        try b.inst(.OpTypeInt, &.{ 3, 32, 0 });
        try b.inst(.OpTypeInt, &.{ 3, 64, 0 });
        try expectMalformed(try b.bytes(), 9, "already defined");
    }
    {
        var b = try WordBuilder.init(a, 8);
        try b.inst(.OpTypeVoid, &.{2});
        try b.inst(.OpTypeFunction, &.{ 3, 2 });
        try b.inst(.OpFunction, &.{ 2, 4, 0, 3 });
        try b.inst(.OpReturn, &.{}); // no OpLabel
        try expectMalformed(try b.bytes(), 15, "outside a block");
    }
    {
        var b = try WordBuilder.init(a, 8);
        try b.inst(.OpTypeVoid, &.{2});
        try b.inst(.OpTypeFunction, &.{ 3, 2 });
        try b.inst(.OpFunction, &.{ 2, 4, 0, 3 });
        try b.inst(.OpLabel, &.{5});
        try b.inst(.OpReturn, &.{});
        try expectMalformed(try b.bytes(), 18, "no OpFunctionEnd");
    }
    {
        var b = try WordBuilder.init(a, 8);
        try b.inst(.OpLabel, &.{5});
        try expectMalformed(try b.bytes(), 5, "outside a function");
    }
    {
        var b = try WordBuilder.init(a, 8);
        try b.inst(.OpTypeInt, &.{ 3, 32, 0 });
        try b.inst(.OpIAdd, &.{ 3, 4, 5, 6 });
        try expectMalformed(try b.bytes(), 9, "outside a function");
    }
    {
        var b = try WordBuilder.init(a, 8);
        try b.inst(.OpTypeInt, &.{ 3, 64, 0 });
        try b.inst(.OpConstant, &.{ 3, 4, 1 }); // 64-bit needs two words
        try expectMalformed(try b.bytes(), 9, "two words");
    }
    {
        var b = try WordBuilder.init(a, 8);
        try b.inst(.OpTypeInt, &.{ 3, 32, 0 });
        try b.inst(.OpConstant, &.{ 3, 4, 1, 2 }); // one word too many
        try expectMalformed(try b.bytes(), 9, "trailing word");
    }
    {
        // An OpSpecConstantOp opcode word that no u16 opcode can be, and one
        // that is in range but names nothing.
        var b = try WordBuilder.init(a, 8);
        try b.inst(.OpTypeInt, &.{ 3, 32, 0 });
        try b.inst(.OpConstant, &.{ 3, 5, 1 });
        try b.inst(.OpSpecConstantOp, &.{ 3, 4, 0x10080, 5, 5 });
        try expectMalformed(try b.bytes(), 13, "opcode 65664 out of range");
        var b2 = try WordBuilder.init(a, 8);
        try b2.inst(.OpTypeInt, &.{ 3, 32, 0 });
        try b2.inst(.OpConstant, &.{ 3, 5, 1 });
        try b2.inst(.OpSpecConstantOp, &.{ 3, 4, 9, 5, 5 });
        try expectMalformed(try b2.bytes(), 13, "unknown opcode 9");
        // CompositeExtract needs its composite id.
        var b3 = try WordBuilder.init(a, 8);
        try b3.inst(.OpTypeInt, &.{ 3, 32, 0 });
        try b3.inst(.OpSpecConstantOp, &.{ 3, 4, @backingInt(spec.Opcode.OpCompositeExtract) });
        try expectMalformed(try b3.bytes(), 9, "OpSpecConstantOp OpCompositeExtract: missing required operand");
    }
    {
        // A zero-width integer type.
        var b = try WordBuilder.init(a, 8);
        try b.inst(.OpTypeInt, &.{ 3, 0, 1 });
        try expectMalformed(try b.bytes(), 5, "OpTypeInt: width 0");
    }
    {
        // A literal of a type wider than two words.
        var b = try WordBuilder.init(a, 8);
        try b.inst(.OpTypeInt, &.{ 3, 128, 0 });
        try b.inst(.OpConstant, &.{ 3, 4, 1, 2, 3, 4 });
        try expectMalformed(try b.bytes(), 9, "128-bit literal is wider");
    }
}

test "OpSpecConstantOp operands follow the named opcode's grammar" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var b = try WordBuilder.init(a, 20);
    try b.inst(.OpTypeInt, &.{ 3, 32, 0 });
    try b.inst(.OpTypeVector, &.{ 4, 3, 4 });
    try b.inst(.OpConstant, &.{ 3, 5, 7 });
    try b.inst(.OpSpecConstant, &.{ 3, 6, 9 });
    try b.inst(.OpSpecConstantComposite, &.{ 4, 7, 5, 6, 5, 6 });
    // %8 = CompositeExtract %7 0: a literal index of 0 is not an id.
    try b.inst(.OpSpecConstantOp, &.{ 3, 8, @backingInt(spec.Opcode.OpCompositeExtract), 7, 0 });
    // %9 = VectorShuffle %7 %7 3 2 1 0: literal components, 0 among them.
    try b.inst(.OpSpecConstantOp, &.{ 4, 9, @backingInt(spec.Opcode.OpVectorShuffle), 7, 7, 3, 2, 1, 0 });
    // %10 = CompositeInsert %6 %7 2
    try b.inst(.OpSpecConstantOp, &.{ 4, 10, @backingInt(spec.Opcode.OpCompositeInsert), 6, 7, 2 });
    // %11 = IAdd %5 %6: ids only.
    try b.inst(.OpSpecConstantOp, &.{ 3, 11, @backingInt(spec.Opcode.OpIAdd), 5, 6 });
    var m = try read(testing.allocator, try b.bytes(), null);
    defer m.deinit();
    const ex = m.constantOf(8).?.value.spec_op;
    try testing.expectEqual(spec.Opcode.OpCompositeExtract, ex.opcode);
    try testing.expectEqual(@as(usize, 2), ex.operands.len);
    try testing.expectEqual(@as(Id, 7), ex.operands[0].value.id);
    try testing.expectEqual(spec.OperandKind.literal_integer, ex.operands[1].kind);
    try testing.expectEqual(@as(u64, 0), ex.operands[1].asInt().?);
    const sh = m.constantOf(9).?.value.spec_op;
    try testing.expectEqual(spec.Opcode.OpVectorShuffle, sh.opcode);
    try testing.expectEqual(@as(usize, 6), sh.operands.len);
    try testing.expectEqual(@as(u64, 3), sh.operands[2].asInt().?);
    try testing.expectEqual(@as(u64, 0), sh.operands[5].asInt().?);
    const ins = m.constantOf(10).?.value.spec_op;
    try testing.expectEqual(@as(u64, 2), ins.operands[2].asInt().?);
    const add = m.constantOf(11).?.value.spec_op;
    try testing.expectEqual(spec.Opcode.OpIAdd, add.opcode);
    try testing.expectEqual(@as(Id, 6), add.operands[1].value.id);
    try testing.expect(m.constantOf(11).?.spec_constant);
}

test "decoration groups are expanded onto their targets" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const non_writable = @backingInt(spec.Decoration.non_writable);
    const descriptor_set = @backingInt(spec.Decoration.descriptor_set);
    const binding = @backingInt(spec.Decoration.binding);
    const offset_dec = @backingInt(spec.Decoration.offset);
    const relaxed = @backingInt(spec.Decoration.relaxed_precision);
    var b = try WordBuilder.init(a, 30);
    // %10 = group { NonWritable, DescriptorSet 1 } on the two buffers %4 and %5.
    try b.inst(.OpDecorate, &.{ 10, non_writable });
    try b.inst(.OpDecorate, &.{ 10, descriptor_set, 1 });
    try b.inst(.OpDecorationGroup, &.{10});
    try b.inst(.OpGroupDecorate, &.{ 10, 4, 5 });
    try b.inst(.OpDecorate, &.{ 4, binding, 0 });
    try b.inst(.OpDecorate, &.{ 5, binding, 7 });
    // %11 = group { RelaxedPrecision } on member 0 of %2 and member 2 of %3.
    try b.inst(.OpDecorate, &.{ 11, relaxed });
    try b.inst(.OpDecorationGroup, &.{11});
    try b.inst(.OpGroupMemberDecorate, &.{ 11, 2, 0, 3, 2 });
    try b.inst(.OpMemberDecorate, &.{ 2, 0, offset_dec, 0 });
    try b.inst(.OpTypeInt, &.{ 1, 32, 0 });
    try b.inst(.OpTypeStruct, &.{ 2, 1 });
    try b.inst(.OpTypeStruct, &.{ 3, 1, 1, 1 });
    try b.inst(.OpTypePointer, &.{ 6, 12, 2 }); // StorageBuffer
    try b.inst(.OpVariable, &.{ 6, 4, 12 });
    try b.inst(.OpVariable, &.{ 6, 5, 12 });
    var m = try read(testing.allocator, try b.bytes(), null);
    defer m.deinit();
    for ([_]Id{ 4, 5 }) |v| {
        try testing.expect(m.hasDecoration(v, .non_writable));
        try testing.expectEqual(@as(u64, 1), m.decoration(v, .descriptor_set).?.firstInt().?);
    }
    try testing.expectEqual(@as(u64, 0), m.decoration(4, .binding).?.firstInt().?);
    try testing.expectEqual(@as(u64, 7), m.decoration(5, .binding).?.firstInt().?);
    try testing.expect(m.memberDecoration(2, 0, .relaxed_precision) != null);
    try testing.expect(m.memberDecoration(3, 2, .relaxed_precision) != null);
    try testing.expect(m.memberDecoration(3, 0, .relaxed_precision) == null);
    // The group keeps its own entries; member decorations of a group target are not whole-id ones.
    try testing.expect(m.hasDecoration(10, .non_writable));
    try testing.expect(!m.hasDecoration(2, .relaxed_precision));
}

test "decoration group misuse is malformed" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const non_writable = @backingInt(spec.Decoration.non_writable);
    {
        // Decorating a group after its OpDecorationGroup would be lost by the expansion.
        var b = try WordBuilder.init(a, 8);
        try b.inst(.OpDecorationGroup, &.{3});
        try b.inst(.OpDecorate, &.{ 3, non_writable });
        try expectMalformed(try b.bytes(), 7, "after its OpDecorationGroup");
    }
    {
        var b = try WordBuilder.init(a, 8);
        try b.inst(.OpGroupDecorate, &.{ 3, 4 });
        try expectMalformed(try b.bytes(), 5, "%3 is not a decoration group declared earlier");
    }
    {
        var b = try WordBuilder.init(a, 8);
        try b.inst(.OpDecorationGroup, &.{3});
        try b.inst(.OpGroupMemberDecorate, &.{ 2, 4, 0 });
        try expectMalformed(try b.bytes(), 7, "%2 is not a decoration group declared earlier");
    }
    {
        // A group targeting itself: before the fix every such instruction
        // doubled the decoration table (n = 22 gave 4 194 304 entries from
        // 324 bytes).
        var b = try WordBuilder.init(a, 8);
        try b.inst(.OpDecorate, &.{ 3, non_writable });
        try b.inst(.OpDecorationGroup, &.{3});
        try b.inst(.OpGroupDecorate, &.{ 3, 3 });
        try expectMalformed(try b.bytes(), 10, "OpGroupDecorate: target %3 is a decoration group");
        var bm = try WordBuilder.init(a, 8);
        try bm.inst(.OpDecorate, &.{ 3, non_writable });
        try bm.inst(.OpDecorationGroup, &.{3});
        try bm.inst(.OpGroupMemberDecorate, &.{ 3, 3, 0 });
        try expectMalformed(try bm.bytes(), 10, "OpGroupMemberDecorate: target %3 is a decoration group");
    }
    {
        // Two groups decorating each other (Fibonacci growth before the fix).
        var b = try WordBuilder.init(a, 8);
        try b.inst(.OpDecorate, &.{ 3, non_writable });
        try b.inst(.OpDecorate, &.{ 4, non_writable });
        try b.inst(.OpDecorationGroup, &.{3});
        try b.inst(.OpDecorationGroup, &.{4});
        try b.inst(.OpGroupDecorate, &.{ 3, 4 });
        try expectMalformed(try b.bytes(), 15, "OpGroupDecorate: target %4 is a decoration group");
        // The same with the target's OpDecorationGroup after the use.
        var b2 = try WordBuilder.init(a, 8);
        try b2.inst(.OpDecorate, &.{ 3, non_writable });
        try b2.inst(.OpDecorationGroup, &.{3});
        try b2.inst(.OpGroupDecorate, &.{ 3, 4 });
        try b2.inst(.OpDecorationGroup, &.{4});
        try expectMalformed(try b2.bytes(), 13, "OpDecorationGroup %4: an earlier OpGroupDecorate targets it");
    }
    {
        // Quadratic growth is capped: 40 decorations on one group applied
        // to 400 targets is 16 000 entries from a 529-word module (cap 8464).
        var b = try WordBuilder.init(a, 1000);
        for (0..40) |_| try b.inst(.OpDecorate, &.{ 3, non_writable });
        try b.inst(.OpDecorationGroup, &.{3});
        var targets: [401]u32 = undefined;
        targets[0] = 3;
        for (targets[1..], 0..) |*t, k| t.* = @intCast(10 + k);
        const at = b.words.items.len;
        try b.inst(.OpGroupDecorate, &targets);
        try expectMalformed(try b.bytes(), at, "decoration groups expand to more than");
        // A realistic module (3 decorations, 400 targets) is far below it.
        var ok = try WordBuilder.init(a, 1000);
        for (0..3) |_| try ok.inst(.OpDecorate, &.{ 3, non_writable });
        try ok.inst(.OpDecorationGroup, &.{3});
        try ok.inst(.OpGroupDecorate, &targets);
        var m = try read(testing.allocator, try ok.bytes(), null);
        defer m.deinit();
        try testing.expectEqual(@as(usize, 3 + 3 * 400), m.decorations.len);
        try testing.expect(m.hasDecoration(409, .non_writable));
    }
}

test "OpTypeFloat keeps its FPEncoding: a bfloat16 is not a half" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var b = try WordBuilder.init(a, 8);
    try b.inst(.OpTypeFloat, &.{ 1, 16, @backingInt(spec.FPEncoding.b_float16khr) }); // BFloat16KHR = 0
    try b.inst(.OpTypeFloat, &.{ 2, 16 });
    try b.inst(.OpTypeFloat, &.{ 3, 8, @backingInt(spec.FPEncoding.float8e4m3ext) });
    try b.inst(.OpConstant, &.{ 1, 4, 0x3f80 });
    var m = try read(testing.allocator, try b.bytes(), null);
    defer m.deinit();
    try testing.expectEqual(@as(u32, 16), m.typeOf(1).?.float.width);
    try testing.expectEqual(@as(?u32, 0), m.typeOf(1).?.float.encoding);
    try testing.expectEqual(@as(?u32, null), m.typeOf(2).?.float.encoding);
    try testing.expectEqual(@as(?u32, 4214), m.typeOf(3).?.float.encoding);
    try testing.expectEqual(@as(u64, 0x3f80), m.constantBits(4).?);
}

test "narrow signed constants: the word is sign-extended, constantBits truncates" {
    // testdata/narrow.zig: `out8[gid] = -1` (i8), `out16[gid] = -2` (i16),
    // `outu8[gid] = 200` (u8); Zig writes the signed ones as 0xFFFFFFFF and
    // 0xFFFFFFFE (SPIR-V 2.2.1), the unsigned one as 0xC8.
    const bin = @embedFile("testdata/narrow.spv");
    var m = try read(testing.allocator, bin, null);
    defer m.deinit();
    var seen: u32 = 0;
    var it = m.constants.iterator();
    while (it.next()) |e| {
        const c = e.value_ptr.*;
        if (c.value != .scalar) continue;
        const t = m.typeOf(c.type).?;
        if (t != .int) continue;
        if (t.int.width == 8 and t.int.signed) {
            try testing.expectEqual(@as(u64, 0xFFFFFFFF), c.value.scalar);
            try testing.expectEqual(@as(u64, 0xFF), m.constantBits(e.key_ptr.*).?);
            seen |= 1;
        } else if (t.int.width == 16 and t.int.signed) {
            try testing.expectEqual(@as(u64, 0xFFFFFFFE), c.value.scalar);
            try testing.expectEqual(@as(u64, 0xFFFE), m.constantBits(e.key_ptr.*).?);
            seen |= 2;
        } else if (t.int.width == 8 and !t.int.signed and c.value.scalar == 200) {
            try testing.expectEqual(@as(u64, 200), m.constantBits(e.key_ptr.*).?);
            seen |= 4;
        }
    }
    try testing.expectEqual(@as(u32, 7), seen);
    try testing.expectEqual(@as(u64, 0xFF), truncateBits(0xFFFFFFFF, 8));
    try testing.expectEqual(@as(u64, 0xFFFFFFFFFFFFFFFF), truncateBits(0xFFFFFFFFFFFFFFFF, 64));
}

test "the Zig SPIR-V backend's binaries decode: types, globals, decorations, functions" {
    const desc = @embedFile("testdata/desc.spv");
    var m = try read(testing.allocator, desc, null);
    defer m.deinit();
    try testing.expectEqual(@as(u32, 0x00290900), m.generator); // Zig's generator id 41, version 9
    try testing.expectEqualStrings("crosscheckKernel", m.entry_points[0].name);
    try testing.expectEqual(@as(usize, 1), m.entry_points[0].interface.len);
    const gid = m.entry_points[0].interface[0];
    try testing.expectEqualStrings("global_invocation_id", m.name(gid).?);
    try testing.expectEqual(@as(u64, @backingInt(spec.BuiltIn.global_invocation_id)), m.decoration(gid, .built_in).?.firstInt().?);
    try testing.expectEqual(@as(usize, 5), m.globals.len);
    var bindings: u32 = 0;
    for (m.globals) |g| {
        if (m.decoration(g.id, .binding)) |d| bindings |= @as(u32, 1) << @intCast(d.firstInt().?);
    }
    try testing.expectEqual(@as(u32, 0b1111), bindings);
    try testing.expectEqual(@as(usize, 2), m.functions.len);
    try testing.expect(m.functions[0].entry_point != null);
    try testing.expect(m.functions[1].entry_point == null);
    try testing.expectEqual(@as(usize, 2), m.functions[1].params.len);
    // LocalSize 64 1 1
    const em = m.execution_modes[m.entry_points[0].execution_modes[0]];
    try testing.expectEqual(@as(u64, 64), em.operands[0].asInt().?);
    try testing.expectEqual(@as(u64, 1), em.operands[2].asInt().?);
    // The Block struct wrapping [4096]i32 with ArrayStride 4.
    const out_ptr = m.typeOf(m.globals[2].type).?.pointer;
    try testing.expectEqual(@as(u32, @backingInt(spec.StorageClass.storage_buffer)), out_ptr.storage_class);
    const block = m.typeOf(out_ptr.pointee).?.@"struct";
    try testing.expectEqual(@as(usize, 1), block.members.len);
    const arr = m.typeOf(block.members[0]).?.array;
    try testing.expectEqual(@as(u64, 4096), m.constantOf(arr.len).?.value.scalar);
    try testing.expectEqual(@as(u64, 4), m.decoration(block.members[0], .array_stride).?.firstInt().?);

    const cf = @embedFile("testdata/cf.spv");
    var mc = try read(testing.allocator, cf, null);
    defer mc.deinit();
    try testing.expectEqualStrings("GLSL.std.450", mc.ext_inst_imports[0].name);
    var switches: usize = 0;
    var phis: usize = 0;
    var ext: usize = 0;
    for (mc.instructions) |inst| switch (inst.op) {
        .OpSwitch => switches += 1,
        .OpPhi => phis += 1,
        .OpExtInst => {
            ext += 1;
            try testing.expectEqual(spec.InstructionSet.@"GLSL.std.450", inst.ext_set.?);
        },
        else => {},
    };
    try testing.expectEqual(@as(usize, 1), switches);
    try testing.expect(phis >= 9);
    try testing.expect(ext >= 4);
    // The Workgroup array and the Uniform block are globals.
    var workgroup = false;
    var uniform = false;
    for (mc.globals) |g| {
        if (g.storage_class == @backingInt(spec.StorageClass.workgroup)) workgroup = true;
        if (g.storage_class == @backingInt(spec.StorageClass.uniform)) uniform = true;
    }
    try testing.expect(workgroup and uniform);

    const kv = @embedFile("testdata/k-spirv64-vulkan.spv");
    var mk = try read(testing.allocator, kv, null);
    defer mk.deinit();
    try testing.expectEqual(@as(usize, 4), mk.functions[0].params.len);
    const ko = @embedFile("testdata/k-spirv64-opencl.spv");
    var mo = try read(testing.allocator, ko, null);
    defer mo.deinit();
    try testing.expectEqual(spec.ExecutionModel.kernel, mo.entry_points[0].model().?);
    try testing.expectEqual(@as(u32, @backingInt(spec.AddressingModel.physical64)), mo.memory_model.?.addressing);
}

test "execution modes: linear in entry points times modes (many entry points of one function, many modes)" {
    // 300 OpEntryPoint of one function and 300 OpExecutionMode of it: the
    // per-mode copy of every entry's list that this reader once made needed
    // E * M^2 / 2 words (54 MB here; gigabytes at E = M = 1200, a module of
    // about 80 KB). Linking once after decoding needs a few hundred KB, so
    // the read must succeed inside a 4 MB budget.
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const count = 300;
    var b = try WordBuilder.init(a, 20);
    try b.inst(.OpCapability, &.{@backingInt(spec.Capability.shader)});
    try b.inst(.OpMemoryModel, &.{ 0, 1 });
    var name_buf: [16]u8 = undefined;
    for (0..count) |k| try b.instStr(.OpEntryPoint, &.{ 5, 10 }, try std.fmt.bufPrint(&name_buf, "k{d}", .{k}));
    try b.instStr(.OpEntryPoint, &.{ 5, 11 }, "other");
    for (0..count) |k| try b.inst(.OpExecutionMode, &.{ 10, 17, @intCast(k + 1), 1, 1 }); // LocalSize k+1 1 1
    try b.inst(.OpExecutionMode, &.{ 11, 17, 7, 1, 1 });
    try b.inst(.OpTypeVoid, &.{1});
    try b.inst(.OpTypeFunction, &.{ 2, 1 });
    for ([_]u32{ 10, 11 }) |fid| {
        try b.inst(.OpFunction, &.{ 1, fid, 0, 2 });
        try b.inst(.OpLabel, &.{fid + 2});
        try b.inst(.OpReturn, &.{});
        try b.inst(.OpFunctionEnd, &.{});
    }
    const bytes = try b.bytes();

    const budget = try testing.allocator.alloc(u8, 4 << 20);
    defer testing.allocator.free(budget);
    var fba = std.heap.FixedBufferAllocator.init(budget);
    var m = try read(fba.allocator(), bytes, null);
    defer m.deinit();
    try testing.expectEqual(@as(usize, count + 1), m.entry_points.len);
    for (m.entry_points[0..count]) |e| {
        try testing.expectEqual(@as(usize, count), e.execution_modes.len);
        // In stream order.
        for (e.execution_modes, 0..) |mi, k| try testing.expectEqual(@as(u64, k + 1), m.execution_modes[mi].operands[0].asInt().?);
    }
    const other = m.entry_points[count];
    try testing.expectEqual(@as(usize, 1), other.execution_modes.len);
    try testing.expectEqual(@as(u64, 7), m.execution_modes[other.execution_modes[0]].operands[0].asInt().?);
    try testing.expectEqual(@as(u32, 11), m.functions[1].id);
    try testing.expectEqual(@as(?u32, count), m.functions[1].entry_point);
    try testing.expectEqual(@as(?u32, count - 1), m.functions[0].entry_point); // the last naming it
}

test "enumerants: spec_extra adds the grammar's missing values without shadowing a vendored one" {
    inline for (comptime std.enums.values(spec.OperandKind)) |kind| {
        switch (comptime kind.category()) {
            .value_enum, .bit_enum => {
                const vendored = kind.enumerants();
                const extra = spec_extra.enumerants(kind);
                for (extra) |x| {
                    for (vendored) |v| try testing.expect(v.value != x.value);
                    if (kind.category() == .bit_enum) try testing.expect(x.value != 0 and std.math.isPowerOfTwo(x.value));
                }
                const all = enumerants(kind);
                try testing.expectEqual(vendored.len + extra.len, all.len);
                if (kind.category() == .bit_enum) {
                    for (all[1..], 0..) |e, i| try testing.expect(all[i].value <= e.value);
                }
            },
            else => try testing.expectEqual(@as(usize, 0), spec_extra.enumerants(kind).len),
        }
    }
    try testing.expectEqual(@as(u32, 5636), enumerantByName(.decoration, "UserTypeGOOGLE").?.value);
    try testing.expectEqualStrings("UserTypeGOOGLE", enumerant(.decoration, 5636).?.name);
    try testing.expectEqualStrings("UserSemantic", enumerant(.decoration, 5635).?.name); // vendored still first
    try testing.expectEqual(@as(u32, 43), enumerantByName(.capability, "Sampled1D").?.value);
    try testing.expectEqual(@as(u32, 44), enumerantByName(.capability, "Image1D").?.value);
    try testing.expectEqual(@as(u32, 4468), enumerantByName(.capability, "RoundingModeRTZ").?.value);
    try testing.expectEqual(@as(u32, 1), enumerantByName(.fp_denorm_mode, "FlushToZero").?.value);
}

test "decorations the vendored tables lack: DXC's UserTypeGOOGLE decodes with its string; an unknown one is named" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    {
        // dxc -spirv -fspv-reflect: OpDecorateString %3 UserTypeGOOGLE "structuredbuffer:<uint>"
        var b = try WordBuilder.init(a, 8);
        try b.inst(.OpCapability, &.{@backingInt(spec.Capability.shader)});
        try b.instStr(.OpExtension, &.{}, "SPV_GOOGLE_user_type");
        try b.inst(.OpMemoryModel, &.{ 0, 1 });
        try b.instStr(.OpDecorateString, &.{ 3, 5636 }, "structuredbuffer:<uint>");
        try b.instStr(.OpDecorateString, &.{ 3, 5635 }, "SV_Target");
        var m = try read(testing.allocator, try b.bytes(), null);
        defer m.deinit();
        for (m.instructions[3..5], [_]u32{ 5636, 5635 }, [_][]const u8{ "structuredbuffer:<uint>", "SV_Target" }) |inst, dec, str| {
            try testing.expectEqual(spec.Opcode.OpDecorateString, inst.op);
            try testing.expectEqual(@as(usize, 3), inst.operands.len);
            try testing.expectEqual(@as(?Id, 3), inst.operands[0].asId());
            try testing.expectEqual(@as(u64, dec), inst.operands[1].asInt().?);
            try testing.expectEqualStrings(str, inst.operands[2].value.string);
        }
        try testing.expectEqual(@as(?spec.Decoration, null), std.enums.fromInt(spec.Decoration, 5636)); // the Zig enum lacks it
    }
    {
        // A decoration in no grammar, with a parameter word: the reader
        // cannot know the word's kind and says why, rather than "trailing".
        var b = try WordBuilder.init(a, 8);
        try b.inst(.OpMemoryModel, &.{ 0, 1 });
        try b.inst(.OpDecorate, &.{ 3, 9999, 1 });
        try expectMalformed(try b.bytes(), 8, "decoration enumerant 9999 is not in the SPIR-V grammar tables");
    }
    {
        // Without parameter words an unknown enumerant still decodes.
        var b = try WordBuilder.init(a, 8);
        try b.inst(.OpCapability, &.{9999});
        var m = try read(testing.allocator, try b.bytes(), null);
        defer m.deinit();
        try testing.expectEqual(@as(u32, 9999), m.capabilities[0]);
    }
    {
        // A vendor bit with a parameter: MemoryAccess AliasScopeINTELMask (0x10000) %5.
        var b = try WordBuilder.init(a, 8);
        try b.inst(.OpMemoryModel, &.{ 0, 1 });
        try b.inst(.OpTypeInt, &.{ 1, 32, 0 });
        try b.inst(.OpTypeVoid, &.{2});
        try b.inst(.OpTypeFunction, &.{ 3, 2 });
        try b.inst(.OpFunction, &.{ 2, 4, 0, 3 });
        try b.inst(.OpLabel, &.{5});
        try b.inst(.OpStore, &.{ 6, 7, 0x10002, 4, 5 }); // Aligned 4 | AliasScopeINTELMask %5
        try b.inst(.OpReturn, &.{});
        try b.inst(.OpFunctionEnd, &.{});
        var m = try read(testing.allocator, try b.bytes(), null);
        defer m.deinit();
        const st = m.instructions[6];
        try testing.expectEqual(spec.Opcode.OpStore, st.op);
        try testing.expectEqual(@as(usize, 5), st.operands.len);
        try testing.expectEqual(@as(u64, 4), st.operands[3].asInt().?); // Aligned's literal first (bit 0x2)
        try testing.expectEqual(@as(?Id, 5), st.operands[4].asId()); // then the INTEL bit's id
    }
}
