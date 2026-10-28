//! SPIR-V text assembler, spirv-as style, for the subset the corpus uses:
//! the naga snapshots in corpus/*.spvasm, the Zig binaries disassembled by
//! dis.zig, and the hand-written tests. It exists so the test corpus is text,
//! not binaries, and so the lowering's corpus step can carry its modules in
//! git as readable files.
//!
//! Syntax (one instruction per line; whitespace separates operands):
//!   `%id = OpXxx operands` or `OpXxx operands`; `;` starts a comment.
//!   * ids: `%N` (digits only) is the numeric id N verbatim; `%name` is a
//!     symbol, numbered in order of first appearance after the largest
//!     numeric id in the file, so a disassembly with numeric ids assembles
//!     back to the identical binary and a named one gets stable ids. Forward
//!     references are fine (phi operands, branch targets, OpTypeForwardPointer).
//!   * enumerants by their Khronos name (`Logical GLSL450`, `GLCompute`,
//!     `StorageBuffer`, `Aligned 4`, `BuiltIn GlobalInvocationId`) or by
//!     number; bit enums joined with `|` (`Volatile|Aligned 16`), `None` (or
//!     `Relaxed`) for zero; enumerant parameters follow in grammar order.
//!   * integers in decimal or `0x` hex, negative where the type is signed
//!     (`OpConstant %i32 -128`, OpSwitch cases of a signed selector); a
//!     64-bit type takes a 64-bit literal (two words, low first). A signed
//!     type narrower than its word is sign-extended to the word, as SPIR-V
//!     2.2.1 requires and Zig's backend does (`OpConstant %i8 -1` is the word
//!     0xFFFFFFFF, testdata/narrow.spv); an unsigned one is zero-extended.
//!     OpTypeInt widths outside 1..64 are refused.
//!   * floats for OpConstant of a float type: decimal (`1.5`, `-2147483600`,
//!     `3.4028235e38`), `inf`/`-inf`/`nan`/`-nan`, or `0x` raw bits (what
//!     dis.zig prints for a NaN with a payload); rounded to the type's width
//!     (`OpTypeFloat 16` packs IEEE half into the low 16 bits). A float
//!     type with an FPEncoding (`OpTypeFloat 16 BFloat16KHR`, the float8
//!     encodings) takes raw `0x` bits only (`OpConstant %bf16 0x3f80` is
//!     1.0); a decimal for it is a syntax error, since rounding it as IEEE
//!     would encode a different value.
//!   * strings in double quotes with `\"`, `\\`, `\n`, `\t` escapes; raw
//!     newlines inside a string are kept (multi-line OpSource).
//!   * `OpExtInst %ty %set Sqrt %x` takes the GLSL.std.450 / OpenCL.std name
//!     or the number; the set is the OpExtInstImport the id names.
//!   * `OpSpecConstantOp %ty IAdd %a %b` takes the opcode without `Op` (or
//!     with it, or the number) and then that opcode's own operands, so
//!     `OpSpecConstantOp %u32 CompositeExtract %v 0` takes a literal index.
//!   * header comments `; Version: 1.3`, `; Generator: 0x00290900` (a
//!     non-numeric generator such as naga's `rspirv` reads as 0), `; Bound: N`
//!     (used when larger than max id + 1) and `; Schema: 0` set the header;
//!     the defaults are 1.0, 0, max id + 1 and 0.
//! Every instruction is encoded from spec.zig's grammar tables (the same
//! ones the reader decodes with), so the operand layout of every opcode in
//! the grammar is covered, not a hand-picked list. Unknown opcodes,
//! enumerants, malformed numbers, missing or surplus operands and bad ids are
//! `error.Syntax` with the line number and a message in `Diagnostics`.
//!
//! Round trips (tests below): assemble -> read -> dis -> assemble gives the
//! same bytes for every corpus file, and dis -> assemble reproduces the five
//! Zig binaries in testdata/ byte for byte.

const std = @import("std");
const reader = @import("reader.zig");
const dis = @import("dis.zig");
const spec = reader.spec;
const Id = reader.Id;

pub const Diagnostics = struct {
    line: usize = 0,
    message: []const u8 = "",
    buf: [192]u8 = undefined,

    fn set(d: ?*Diagnostics, line: usize, comptime fmt: []const u8, args: anytype) error{Syntax} {
        if (d) |diag| {
            diag.line = line;
            diag.message = std.mem.print(&diag.buf, fmt, args) catch diag.buf[0..];
        }
        return error.Syntax;
    }
};

pub const Error = error{ Syntax, OutOfMemory };

/// Assemble `text` into a little-endian SPIR-V module. Caller frees with `gpa`.
pub fn assemble(gpa: std.mem.Allocator, text: []const u8, diag: ?*Diagnostics) Error![]u8 {
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const a = arena_state.allocator();

    var asmr: Assembler = .{ .a = a, .diag = diag };
    try asmr.tokenize(text);
    try asmr.assignNumericIds();
    try asmr.encodeAll();

    var out: std.ArrayList(u32) = .empty;
    const bound = @max(asmr.next_id, asmr.header_bound);
    try out.appendSlice(a, &.{ spec.magic_number, (@as(u32, asmr.version_major) << 16) | (@as(u32, asmr.version_minor) << 8), asmr.generator, bound, asmr.schema });
    try out.appendSlice(a, asmr.words.items);
    return gpa.dupe(u8, std.mem.sliceAsBytes(out.items));
}

const Token = struct {
    kind: enum { id, eq, string, word },
    text: []const u8,
};

const Statement = struct {
    line: usize,
    tokens: []const Token,
};

const ScalarType = union(enum) {
    int: struct { width: u32, signed: bool },
    /// `encoding`: OpTypeFloat's FPEncoding operand, null for IEEE 754.
    float: struct { width: u32, encoding: ?u32 = null },
};

const Assembler = struct {
    a: std.mem.Allocator,
    diag: ?*Diagnostics,
    statements: std.ArrayList(Statement) = .empty,
    words: std.ArrayList(u32) = .empty,
    version_major: u8 = 1,
    version_minor: u8 = 0,
    generator: u32 = 0,
    header_bound: u32 = 0,
    schema: u32 = 0,
    /// Symbolic names -> ids.
    symbols: std.StringHashMapUnmanaged(Id) = .empty,
    next_id: u32 = 1,
    /// Line of the statement being encoded, for idOf's diagnostics.
    line: usize = 0,
    /// Scalar types by id, for context-dependent literals.
    scalar_types: std.AutoHashMapUnmanaged(Id, ScalarType) = .empty,
    /// Result type of every id with one, for OpSwitch selectors.
    result_types: std.AutoHashMapUnmanaged(Id, Id) = .empty,
    /// OpExtInstImport ids -> set.
    ext_sets: std.AutoHashMapUnmanaged(Id, ?spec.InstructionSet) = .empty,

    fn fail(s: *Assembler, line: usize, comptime fmt: []const u8, args: anytype) error{Syntax} {
        return Diagnostics.set(s.diag, line, fmt, args);
    }

    fn tokenize(s: *Assembler, text: []const u8) Error!void {
        var i: usize = 0;
        var line: usize = 1;
        var cur: std.ArrayList(Token) = .empty;
        var cur_line: usize = 1;
        while (i < text.len) {
            const c = text[i];
            if (c == '\n') {
                if (cur.items.len > 0) {
                    try s.statements.append(s.a, .{ .line = cur_line, .tokens = try cur.toOwnedSlice(s.a) });
                    cur = .empty;
                }
                line += 1;
                i += 1;
                continue;
            }
            if (std.ascii.isWhitespace(c)) {
                i += 1;
                continue;
            }
            if (c == ';') {
                const end = std.mem.indexOfScalarPos(u8, text, i, '\n') orelse text.len;
                try s.headerComment(text[i + 1 .. end], line);
                i = end;
                continue;
            }
            if (cur.items.len == 0) cur_line = line;
            if (c == '"') {
                var str: std.ArrayList(u8) = .empty;
                i += 1;
                while (true) {
                    if (i >= text.len) return s.fail(cur_line, "unterminated string", .{});
                    const d = text[i];
                    if (d == '"') {
                        i += 1;
                        break;
                    }
                    if (d == '\\') {
                        if (i + 1 >= text.len) return s.fail(cur_line, "unterminated string", .{});
                        const e = text[i + 1];
                        try str.append(s.a, switch (e) {
                            'n' => '\n',
                            't' => '\t',
                            else => e,
                        });
                        i += 2;
                        continue;
                    }
                    if (d == '\n') line += 1;
                    try str.append(s.a, d);
                    i += 1;
                }
                try cur.append(s.a, .{ .kind = .string, .text = try str.toOwnedSlice(s.a) });
                continue;
            }
            if (c == '=') {
                try cur.append(s.a, .{ .kind = .eq, .text = "=" });
                i += 1;
                continue;
            }
            const start = i;
            while (i < text.len and !std.ascii.isWhitespace(text[i]) and text[i] != '"' and text[i] != ';' and text[i] != '=') i += 1;
            const t = text[start..i];
            if (t[0] == '%') {
                if (t.len == 1) return s.fail(line, "empty id '%'", .{});
                for (t[1..]) |ch| if (!(std.ascii.isAlphanumeric(ch) or ch == '_' or ch == '.')) return s.fail(line, "bad character in id '{s}'", .{t});
                try cur.append(s.a, .{ .kind = .id, .text = t[1..] });
            } else {
                try cur.append(s.a, .{ .kind = .word, .text = t });
            }
        }
        if (cur.items.len > 0) try s.statements.append(s.a, .{ .line = cur_line, .tokens = try cur.toOwnedSlice(s.a) });
    }

    fn headerComment(s: *Assembler, comment: []const u8, line: usize) Error!void {
        const c = std.mem.trim(u8, comment, " \t\r");
        if (std.mem.startsWith(u8, c, "Version:")) {
            const v = std.mem.trim(u8, c["Version:".len..], " \t");
            const dot = std.mem.indexOfScalar(u8, v, '.') orelse return s.fail(line, "bad '; Version:' comment", .{});
            s.version_major = std.fmt.parseInt(u8, v[0..dot], 10) catch return s.fail(line, "bad '; Version:' comment", .{});
            s.version_minor = std.fmt.parseInt(u8, v[dot + 1 ..], 10) catch return s.fail(line, "bad '; Version:' comment", .{});
        } else if (std.mem.startsWith(u8, c, "Generator:")) {
            const v = std.mem.trim(u8, c["Generator:".len..], " \t");
            // Only a number is honoured; naga's `rspirv` and spirv-dis' prose read as 0.
            const end = std.mem.indexOfAny(u8, v, " \t;(") orelse v.len;
            s.generator = std.fmt.parseInt(u32, v[0..end], 0) catch 0;
        } else if (std.mem.startsWith(u8, c, "Bound:")) {
            const v = std.mem.trim(u8, c["Bound:".len..], " \t");
            s.header_bound = std.fmt.parseInt(u32, v, 10) catch return s.fail(line, "bad '; Bound:' comment", .{});
        } else if (std.mem.startsWith(u8, c, "Schema:")) {
            const v = std.mem.trim(u8, c["Schema:".len..], " \t");
            s.schema = std.fmt.parseInt(u32, v, 10) catch return s.fail(line, "bad '; Schema:' comment", .{});
        }
    }

    /// Symbolic ids start after the largest numeric id in the file. Every id
    /// must be below the header's bound, which is itself a u32, so the
    /// largest usable id is maxInt(u32) - 1 (SPIR-V 2.3: 0 < id < Bound).
    fn assignNumericIds(s: *Assembler) Error!void {
        var max: u32 = 0;
        for (s.statements.items) |st| for (st.tokens) |t| {
            if (t.kind == .id and isNumeric(t.text)) {
                const n = std.fmt.parseInt(u32, t.text, 10) catch return s.fail(st.line, "numeric id %{s} is out of range", .{t.text});
                if (n == 0) return s.fail(st.line, "%0 is not a valid id", .{});
                if (n == std.math.maxInt(u32)) return s.fail(st.line, "numeric id %{d} leaves no room for the bound", .{n});
                max = @max(max, n);
            }
        };
        s.next_id = max + 1;
    }

    fn isNumeric(t: []const u8) bool {
        for (t) |c| if (!std.ascii.isDigit(c)) return false;
        return true;
    }

    fn idOf(s: *Assembler, tok: Token) Error!Id {
        if (isNumeric(tok.text)) return std.fmt.parseInt(u32, tok.text, 10) catch unreachable;
        const gop = try s.symbols.getOrPut(s.a, tok.text);
        if (!gop.found_existing) {
            // next_id becomes the bound once this id is taken; it must stay a u32.
            if (s.next_id == std.math.maxInt(u32)) {
                _ = s.symbols.remove(tok.text);
                return s.fail(s.line, "symbolic id %{s} leaves no room for the bound (ids are exhausted)", .{tok.text});
            }
            gop.value_ptr.* = s.next_id;
            s.next_id += 1;
        }
        return gop.value_ptr.*;
    }

    fn encodeAll(s: *Assembler) Error!void {
        for (s.statements.items) |st| try s.encodeStatement(st);
    }

    const Cursor = struct {
        tokens: []const Token,
        i: usize = 0,

        fn next(c: *Cursor) ?Token {
            if (c.i >= c.tokens.len) return null;
            c.i += 1;
            return c.tokens[c.i - 1];
        }

        fn done(c: *Cursor) bool {
            return c.i >= c.tokens.len;
        }
    };

    fn encodeStatement(s: *Assembler, st: Statement) Error!void {
        s.line = st.line;
        var cur: Cursor = .{ .tokens = st.tokens };
        var lhs: ?Token = null;
        if (st.tokens.len >= 2 and st.tokens[0].kind == .id and st.tokens[1].kind == .eq) {
            lhs = st.tokens[0];
            cur.i = 2;
        }
        const op_tok = cur.next() orelse return s.fail(st.line, "missing opcode", .{});
        if (op_tok.kind != .word) return s.fail(st.line, "expected an opcode", .{});
        const op = opcodeByName(op_tok.text) orelse return s.fail(st.line, "unknown opcode '{s}'", .{op_tok.text});

        const start = s.words.items.len;
        try s.words.append(s.a, 0); // header, patched below
        var result_type: ?Id = null;
        var result: ?Id = null;
        var ext_set: ?spec.InstructionSet = null;
        var ext_number: u32 = 0;
        var lhs_used = false;

        const grammar = op.operands();
        for (grammar, 0..) |g, gi| {
            switch (g.kind) {
                .id_result_type => {
                    const t = cur.next() orelse return s.fail(st.line, "{t}: missing result type", .{op});
                    if (t.kind != .id) return s.fail(st.line, "{t}: expected a result type id, got '{s}'", .{ op, t.text });
                    result_type = try s.idOf(t);
                    try s.words.append(s.a, result_type.?);
                    continue;
                },
                .id_result => {
                    const l = lhs orelse return s.fail(st.line, "{t} needs a result id ('%x = {t} ...')", .{ op, op });
                    result = try s.idOf(l);
                    lhs_used = true;
                    try s.words.append(s.a, result.?);
                    continue;
                },
                else => {},
            }
            switch (g.quantifier) {
                .required => {
                    if (cur.done()) return s.fail(st.line, "{t}: missing operand {d} ({t})", .{ op, gi, g.kind });
                    try s.encodeOperand(st.line, op, g.kind, &cur, result_type, &ext_set, &ext_number);
                },
                .optional => {
                    if (cur.done()) break;
                    try s.encodeOperand(st.line, op, g.kind, &cur, result_type, &ext_set, &ext_number);
                },
                .variadic => {
                    if (op == .OpExtInst) {
                        try s.encodeExtInstTail(st.line, &cur, ext_set, ext_number);
                        break;
                    }
                    if (op == .OpSwitch) {
                        try s.encodeSwitchTail(st.line, &cur);
                        break;
                    }
                    while (!cur.done()) try s.encodeOperand(st.line, op, g.kind, &cur, result_type, &ext_set, &ext_number);
                },
            }
        }
        if (op == .OpSpecConstantOp) {
            // The operands of the named opcode follow, in its own grammar
            // (SPIR-V 3.52.12) minus result type and result id: ids, and the
            // literal indexes of CompositeExtract / CompositeInsert / VectorShuffle.
            const inner = try s.specConstantOpcode(st.line, s.words.items[start + 3]);
            for (inner.operands(), 0..) |g, gi| {
                if (g.kind == .id_result_type or g.kind == .id_result) continue;
                switch (g.quantifier) {
                    .required => {
                        if (cur.done()) return s.fail(st.line, "OpSpecConstantOp {t}: missing operand {d} ({t})", .{ inner, gi, g.kind });
                        try s.encodeOperand(st.line, op, g.kind, &cur, result_type, &ext_set, &ext_number);
                    },
                    .optional => {
                        if (cur.done()) break;
                        try s.encodeOperand(st.line, op, g.kind, &cur, result_type, &ext_set, &ext_number);
                    },
                    .variadic => while (!cur.done()) try s.encodeOperand(st.line, op, g.kind, &cur, result_type, &ext_set, &ext_number),
                }
            }
        }
        if (lhs != null and !lhs_used) return s.fail(st.line, "{t} has no result id", .{op});
        if (!cur.done()) return s.fail(st.line, "{t}: surplus operand '{s}'", .{ op, cur.tokens[cur.i].text });

        const wc = s.words.items.len - start;
        if (wc > 0xFFFF) return s.fail(st.line, "{t}: instruction longer than 65535 words", .{op});
        s.words.items[start] = (@as(u32, @intCast(wc)) << 16) | @as(u32, @backingInt(op));

        // Bookkeeping for later literals.
        if (result) |id| {
            if (result_type) |t| try s.result_types.put(s.a, id, t);
            switch (op) {
                .OpTypeInt => {
                    const width = s.words.items[start + 2];
                    if (width == 0 or width > 64) return s.fail(st.line, "OpTypeInt: width {d} is outside 1..64", .{width});
                    try s.scalar_types.put(s.a, id, .{ .int = .{ .width = width, .signed = s.words.items[start + 3] != 0 } });
                },
                .OpTypeFloat => try s.scalar_types.put(s.a, id, .{ .float = .{
                    .width = s.words.items[start + 2],
                    .encoding = if (wc > 3) s.words.items[start + 3] else null,
                } }),
                .OpExtInstImport => {
                    const name = st.tokens[3].text;
                    const set: ?spec.InstructionSet = if (std.mem.eql(u8, name, "GLSL.std.450")) .@"GLSL.std.450" else if (std.mem.eql(u8, name, "OpenCL.std")) .@"OpenCL.std" else null;
                    try s.ext_sets.put(s.a, id, set);
                },
                else => {},
            }
        }
    }

    /// The opcode an OpSpecConstantOp names (already encoded as a word by
    /// the literal_spec_constant_op_integer path, which resolves names).
    fn specConstantOpcode(s: *Assembler, line: usize, word: u32) Error!spec.Opcode {
        const opc = std.math.cast(u16, word) orelse return s.fail(line, "OpSpecConstantOp: opcode {d} out of range", .{word});
        return std.enums.fromInt(spec.Opcode, opc) orelse return s.fail(line, "OpSpecConstantOp: unknown opcode {d}", .{word});
    }

    fn encodeOperand(
        s: *Assembler,
        line: usize,
        op: spec.Opcode,
        kind: spec.OperandKind,
        cur: *Cursor,
        result_type: ?Id,
        ext_set: *?spec.InstructionSet,
        ext_number: *u32,
    ) Error!void {
        const tok = cur.next() orelse return s.fail(line, "{t}: missing operand ({t})", .{ op, kind });
        switch (kind.category()) {
            .id => {
                if (tok.kind != .id) return s.fail(line, "{t}: expected an id for {t}, got '{s}'", .{ op, kind, tok.text });
                const id = try s.idOf(tok);
                try s.words.append(s.a, id);
                if (op == .OpExtInst and kind == .id_ref and ext_number.* == 0 and ext_set.* == null) {
                    // The first id_ref after the result is the set.
                    if (s.ext_sets.get(id)) |set| ext_set.* = set;
                }
            },
            .literal => switch (kind) {
                .literal_string => {
                    if (tok.kind != .string) return s.fail(line, "{t}: expected a string, got '{s}'", .{ op, tok.text });
                    try reader.appendString(&s.words, s.a, tok.text);
                },
                .literal_context_dependent_number => {
                    if (tok.kind != .word) return s.fail(line, "{t}: expected a number, got '{s}'", .{ op, tok.text });
                    const ty: ?ScalarType = if (result_type) |t| s.scalar_types.get(t) else null;
                    try s.encodeNumber(line, op, tok.text, ty);
                },
                .literal_ext_inst_integer => {
                    if (tok.kind != .word) return s.fail(line, "OpExtInst: expected an instruction name or number, got '{s}'", .{tok.text});
                    const n: u32 = if (std.fmt.parseInt(u32, tok.text, 0)) |v| v else |_| blk: {
                        const set = ext_set.* orelse return s.fail(line, "OpExtInst: '{s}' needs a known set (GLSL.std.450 or OpenCL.std) to resolve the name", .{tok.text});
                        break :blk extInstByName(set, tok.text) orelse return s.fail(line, "OpExtInst: unknown {t} instruction '{s}'", .{ set, tok.text });
                    };
                    ext_number.* = n;
                    try s.words.append(s.a, n);
                },
                .literal_spec_constant_op_integer => {
                    if (tok.kind != .word) return s.fail(line, "OpSpecConstantOp: expected an opcode, got '{s}'", .{tok.text});
                    const n: u32 = if (std.fmt.parseInt(u32, tok.text, 0)) |v| v else |_| blk: {
                        if (opcodeByName(tok.text)) |o| break :blk @backingInt(o);
                        var buf: [64]u8 = undefined;
                        const with = std.mem.print(&buf, "Op{s}", .{tok.text}) catch return s.fail(line, "OpSpecConstantOp: unknown opcode '{s}'", .{tok.text});
                        break :blk @backingInt(opcodeByName(with) orelse return s.fail(line, "OpSpecConstantOp: unknown opcode '{s}'", .{tok.text}));
                    };
                    try s.words.append(s.a, n);
                },
                else => {
                    if (tok.kind != .word) return s.fail(line, "{t}: expected an integer for {t}, got '{s}'", .{ op, kind, tok.text });
                    const v = parseIntLiteral(tok.text, 32, true) orelse return s.fail(line, "{t}: bad integer literal '{s}'", .{ op, tok.text });
                    try s.words.append(s.a, @truncate(v));
                },
            },
            .value_enum => {
                if (tok.kind != .word) return s.fail(line, "{t}: expected a {t} enumerant, got '{s}'", .{ op, kind, tok.text });
                const e: ?spec.Enumerant = reader.enumerantByName(kind, tok.text);
                const v: u32 = if (e) |en| en.value else (std.fmt.parseInt(u32, tok.text, 0) catch return s.fail(line, "{t}: unknown {t} enumerant '{s}'", .{ op, kind, tok.text }));
                try s.words.append(s.a, v);
                const en = e orelse reader.enumerant(kind, v);
                if (en) |x| for (x.parameters) |p| {
                    if (cur.done()) return s.fail(line, "{t}: {s} needs a {t} parameter", .{ op, x.name, p.kind });
                    try s.encodeOperand(line, op, p.kind, cur, result_type, ext_set, ext_number);
                };
            },
            .bit_enum => {
                if (tok.kind != .word) return s.fail(line, "{t}: expected {t} flags, got '{s}'", .{ op, kind, tok.text });
                var v: u32 = 0;
                var it = std.mem.splitScalar(u8, tok.text, '|');
                while (it.next()) |part| {
                    if (std.mem.eql(u8, part, "None") or std.mem.eql(u8, part, "Relaxed")) continue;
                    if (reader.enumerantByName(kind, part)) |e| {
                        v |= e.value;
                    } else {
                        v |= std.fmt.parseInt(u32, part, 0) catch return s.fail(line, "{t}: unknown {t} flag '{s}'", .{ op, kind, part });
                    }
                }
                try s.words.append(s.a, v);
                for (reader.enumerants(kind)) |e| {
                    if (e.value != 0 and (v & e.value) == e.value) {
                        for (e.parameters) |p| {
                            if (cur.done()) return s.fail(line, "{t}: {s} needs a {t} parameter", .{ op, e.name, p.kind });
                            try s.encodeOperand(line, op, p.kind, cur, result_type, ext_set, ext_number);
                        }
                    }
                }
            },
            .composite => {
                cur.i -= 1;
                for (kind.bases()) |b| {
                    if (cur.done()) return s.fail(line, "{t}: incomplete {t} pair", .{ op, kind });
                    try s.encodeOperand(line, op, b, cur, result_type, ext_set, ext_number);
                }
            },
        }
    }

    fn encodeExtInstTail(s: *Assembler, line: usize, cur: *Cursor, ext_set: ?spec.InstructionSet, number: u32) Error!void {
        const table: ?[]const spec.Operand = if (ext_set) |set| set.operands(number) else null;
        var dummy_set: ?spec.InstructionSet = null;
        var dummy_n: u32 = 0;
        if (table) |grammar| {
            for (grammar) |g| switch (g.quantifier) {
                .required => {
                    if (cur.done()) return s.fail(line, "OpExtInst {d}: missing operand ({t})", .{ number, g.kind });
                    try s.encodeOperand(line, .OpExtInst, g.kind, cur, null, &dummy_set, &dummy_n);
                },
                .optional => {
                    if (cur.done()) break;
                    try s.encodeOperand(line, .OpExtInst, g.kind, cur, null, &dummy_set, &dummy_n);
                },
                .variadic => while (!cur.done()) try s.encodeOperand(line, .OpExtInst, g.kind, cur, null, &dummy_set, &dummy_n),
            };
            return;
        }
        while (cur.next()) |t| {
            if (t.kind != .id) return s.fail(line, "OpExtInst: expected an id operand, got '{s}'", .{t.text});
            try s.words.append(s.a, try s.idOf(t));
        }
    }

    fn encodeSwitchTail(s: *Assembler, line: usize, cur: *Cursor) Error!void {
        // words so far: header, selector, default
        const start = s.words.items.len - 2;
        const selector = s.words.items[start];
        var width: u32 = 32;
        var signed = false;
        if (s.result_types.get(selector)) |t| if (s.scalar_types.get(t)) |st| switch (st) {
            .int => |i| {
                width = i.width;
                signed = i.signed;
            },
            else => {},
        };
        while (!cur.done()) {
            const lit = cur.next().?;
            if (lit.kind != .word) return s.fail(line, "OpSwitch: expected a case literal, got '{s}'", .{lit.text});
            const v = signExtendToWord(parseIntLiteral(lit.text, width, signed) orelse return s.fail(line, "OpSwitch: bad case literal '{s}' for a {d}-bit {s} selector", .{ lit.text, width, if (signed) "signed" else "unsigned" }), width, signed);
            if (width > 32) {
                try s.words.append(s.a, @truncate(v));
                try s.words.append(s.a, @truncate(v >> 32));
            } else {
                try s.words.append(s.a, @truncate(v));
            }
            const label = cur.next() orelse return s.fail(line, "OpSwitch: case {s} has no label", .{lit.text});
            if (label.kind != .id) return s.fail(line, "OpSwitch: expected a label id, got '{s}'", .{label.text});
            try s.words.append(s.a, try s.idOf(label));
        }
    }

    fn encodeNumber(s: *Assembler, line: usize, op: spec.Opcode, text: []const u8, ty: ?ScalarType) Error!void {
        const t: ScalarType = ty orelse .{ .int = .{ .width = 32, .signed = false } };
        switch (t) {
            .int => |i| {
                const v = signExtendToWord(parseIntLiteral(text, i.width, i.signed) orelse return s.fail(line, "{t}: '{s}' is not a {d}-bit {s} integer", .{ op, text, i.width, if (i.signed) "signed" else "unsigned" }), i.width, i.signed);
                if (i.width > 32) {
                    try s.words.append(s.a, @truncate(v));
                    try s.words.append(s.a, @truncate(v >> 32));
                } else try s.words.append(s.a, @truncate(v));
            },
            .float => |f| {
                // A bfloat16 / float8 literal is raw bits only: reading a
                // decimal as IEEE would encode the wrong value silently.
                const bits = if (f.encoding) |e|
                    parseRawFloatBits(text, f.width) orelse return s.fail(line, "{t}: '{s}': a float with FPEncoding {s} takes raw `0x` bits only", .{ op, text, if (reader.enumerant(.fp_encoding, e)) |en| en.name else "?" })
                else
                    parseFloatLiteral(text, f.width) orelse return s.fail(line, "{t}: '{s}' is not a {d}-bit float literal", .{ op, text, f.width });
                if (f.width > 32) {
                    try s.words.append(s.a, @truncate(bits));
                    try s.words.append(s.a, @truncate(bits >> 32));
                } else try s.words.append(s.a, @truncate(bits));
            },
        }
    }
};

/// The bits of an integer literal of `width` bits (1..64, else null;
/// zero-extended to u64); negatives need `signed` and must fit, positives
/// must fit the width. The word a module stores is `signExtendToWord` of this.
pub fn parseIntLiteral(text: []const u8, width: u32, signed: bool) ?u64 {
    if (width == 0 or width > 64) return null;
    const bits: u7 = @intCast(width);
    const mask: u64 = if (bits == 64) std.math.maxInt(u64) else (@as(u64, 1) << @as(u6, @intCast(bits))) - 1;
    if (text.len > 0 and text[0] == '-') {
        if (!signed) return null;
        const mag = std.fmt.parseInt(u64, text[1..], 0) catch return null;
        const limit: u64 = @as(u64, 1) << @as(u6, @intCast(bits - 1));
        if (mag > limit) return null;
        return (~mag +% 1) & mask;
    }
    const v = std.fmt.parseInt(u64, text, 0) catch return null;
    if (v > mask) return null;
    return v;
}

/// The value `v` of a `width`-bit integer as the word(s) a module stores it
/// in: a signed type narrower than its 32- or 64-bit word is sign-extended to
/// the word (SPIR-V 2.2.1 "Literal Number"; Zig's backend writes 0xFFFFFFFF
/// for an i8 -1, testdata/narrow.spv), an unsigned one is zero-extended.
pub fn signExtendToWord(v: u64, width: u32, signed: bool) u64 {
    if (!signed or width == 32 or width >= 64) return v;
    const sign_bit = @as(u64, 1) << @intCast(width - 1);
    if (v & sign_bit == 0) return v;
    const word_mask: u64 = if (width < 32) 0xFFFF_FFFF else std.math.maxInt(u64);
    const value_mask = (@as(u64, 1) << @intCast(width)) - 1;
    return (v | ~value_mask) & word_mask;
}

/// IEEE bits of a float literal rounded to `width` (16, 32 or 64), in the
/// low bits of the result. `0x...` without a binary exponent or dot is the
/// raw bit pattern (how dis.zig prints NaN payloads).
pub fn parseFloatLiteral(text: []const u8, width: u32) ?u64 {
    const negative = text.len > 0 and text[0] == '-';
    const body = if (negative) text[1..] else text;
    // `nan` / `inf` by hand: std.fmt.parseFloat drops the sign of a NaN, and
    // dis.zig prints the negative canonical quiet NaN as `-nan`.
    const is_nan = std.ascii.eqlIgnoreCase(body, "nan");
    const is_inf = std.ascii.eqlIgnoreCase(body, "inf") or std.ascii.eqlIgnoreCase(body, "infinity");
    if (is_nan or is_inf) {
        const bits: u64 = switch (width) {
            16 => if (is_nan) 0x7e00 else 0x7c00,
            32 => if (is_nan) 0x7fc00000 else 0x7f800000,
            64 => if (is_nan) 0x7ff8000000000000 else 0x7ff0000000000000,
            else => return null,
        };
        const sign: u64 = if (negative) @as(u64, 1) << @intCast(width - 1) else 0;
        return bits | sign;
    }
    if (std.mem.startsWith(u8, body, "0x") and std.mem.indexOfAny(u8, body, "pP.") == null) {
        return parseRawFloatBits(text, width);
    }
    switch (width) {
        16 => {
            const x = std.fmt.parseFloat(f16, text) catch return null;
            return @as(u16, @bitCast(x));
        },
        32 => {
            const x = std.fmt.parseFloat(f32, text) catch return null;
            return @as(u32, @bitCast(x));
        },
        64 => {
            const x = std.fmt.parseFloat(f64, text) catch return null;
            return @as(u64, @bitCast(x));
        },
        else => return null,
    }
}

/// A raw `0x` bit pattern that fits `width` bits (1..64), else null: the
/// only literal a float type with an FPEncoding takes, and how dis.zig
/// prints NaN payloads.
pub fn parseRawFloatBits(text: []const u8, width: u32) ?u64 {
    if (width == 0 or width > 64) return null;
    if (!std.mem.startsWith(u8, text, "0x") or std.mem.indexOfAny(u8, text, "pP.") != null) return null;
    const v = std.fmt.parseInt(u64, text, 0) catch return null;
    const limit: u64 = if (width == 64) std.math.maxInt(u64) else (@as(u64, 1) << @intCast(width)) - 1;
    if (v > limit) return null;
    return v;
}

pub fn opcodeByName(name: []const u8) ?spec.Opcode {
    @setEvalBranchQuota(20000);
    return std.meta.stringToEnum(spec.Opcode, name);
}

pub fn extInstByName(set: spec.InstructionSet, name: []const u8) ?u32 {
    @setEvalBranchQuota(20000);
    return switch (set) {
        .core => null,
        .@"GLSL.std.450" => if (std.meta.stringToEnum(spec.GlslOpcode, name)) |o| @backingInt(o) else null,
        .@"OpenCL.std" => if (std.meta.stringToEnum(spec.OpenClOpcode, name)) |o| @backingInt(o) else null,
    };
}

// ───────────────────────────── tests ─────────────────────────────

const testing = std.testing;

test {
    _ = reader;
    _ = dis;
}

/// Names of every corpus file, for the round-trip tests and the corpus step.
pub const corpus_files = [_][]const u8{
    "wgsl-6772-unpack-expr-accesses.spvasm",
    "wgsl-abstract-types-const.spvasm",
    "wgsl-abstract-types-operators.spvasm",
    "wgsl-access.spvasm",
    "wgsl-atomicCompareExchange-int64.spvasm",
    "wgsl-atomicCompareExchange.spvasm",
    "wgsl-atomicOps-float32.spvasm",
    "wgsl-atomicOps-int64-min-max.spvasm",
    "wgsl-atomicOps-int64.spvasm",
    "wgsl-atomicOps.spvasm",
    "wgsl-atomicTexture-int64.spvasm",
    "wgsl-atomicTexture.spvasm",
    "wgsl-bitcast.spvasm",
    "wgsl-bits.spvasm",
    "wgsl-boids.spvasm",
    "wgsl-collatz.spvasm",
    "zig-vulkan-params.spvasm",
    "zig-opencl-params.spvasm",
    "zig-crosscheck.spvasm",
    "zig-control-flow.spvasm",
    "zig-narrow-ints.spvasm",
    "hand-glsl-std-450.spvasm",
    "hand-opencl-std.spvasm",
    "hand-scalars.spvasm",
    "hand-vectors.spvasm",
    "hand-control-flow.spvasm",
    "hand-workgroup-atomics.spvasm",
    "hand-glslang-stage4.spvasm",
    "hand-dxc-reflect.spvasm",
};

const corpus_text = blk: {
    var t: [corpus_files.len][]const u8 = undefined;
    for (corpus_files, 0..) |name, i| t[i] = @embedFile("corpus/" ++ name);
    break :blk t;
};

fn expectSyntax(text: []const u8, line: usize, needle: []const u8) !void {
    var diag: Diagnostics = .{};
    const r = assemble(testing.allocator, text, &diag);
    if (r) |bytes| {
        testing.allocator.free(bytes);
        std.debug.print("expected Syntax ({s}) but it assembled\n", .{needle});
        return error.TestUnexpectedResult;
    } else |err| {
        try testing.expectEqual(error.Syntax, err);
        if (std.mem.indexOf(u8, diag.message, needle) == null) {
            std.debug.print("message '{s}' does not contain '{s}'\n", .{ diag.message, needle });
            return error.TestUnexpectedResult;
        }
        try testing.expectEqual(line, diag.line);
    }
}

test "numbers, enums, strings and ids encode as the reader expects" {
    const text =
        \\; Version: 1.3
        \\; Generator: 0x00290900
        \\; Bound: 100
        \\OpCapability Shader
        \\OpCapability Int64
        \\%glsl = OpExtInstImport "GLSL.std.450"
        \\OpMemoryModel Logical GLSL450
        \\OpEntryPoint GLCompute %main "main" %gid
        \\OpExecutionMode %main LocalSize 64 1 1
        \\%src = OpString "k.zig"
        \\OpSource Zig 2304 %src "a \"b\"\\
        \\c"
        \\OpName %main "main"
        \\OpName %glsl "glsl"
        \\OpName %gid "gid"
        \\OpName %block "block"
        \\OpName %u32 "u32"
        \\OpName %i64 "i64"
        \\OpName %f32 "f32"
        \\OpName %f16 "f16"
        \\OpName %f64 "f64"
        \\OpName %c1 "c1"
        \\OpName %c2 "c2"
        \\OpName %c3 "c3"
        \\OpName %c4 "c4"
        \\OpName %c5 "c5"
        \\OpName %c6 "c6"
        \\OpName %c7 "c7"
        \\OpName %c8 "c8"
        \\OpName %sc "sc"
        \\OpName %v "v"
        \\OpName %l "l"
        \\OpName %sq "sq"
        \\OpName %sq2 "sq2"
        \\OpName %merge "merge"
        \\OpName %pin "pin"
        \\OpDecorate %gid BuiltIn GlobalInvocationId
        \\OpMemberDecorate %block 0 Offset 16
        \\%void = OpTypeVoid
        \\%u32 = OpTypeInt 32 0
        \\%i64 = OpTypeInt 64 1
        \\%f32 = OpTypeFloat 32
        \\%f16 = OpTypeFloat 16
        \\%f64 = OpTypeFloat 64
        \\%c1 = OpConstant %u32 4294967295
        \\%c2 = OpConstant %i64 -2
        \\%c3 = OpConstant %f32 1.5
        \\%c4 = OpConstant %f16 0.5
        \\%c5 = OpConstant %f64 -0.25
        \\%c6 = OpConstant %f32 0x7fc00001
        \\%c7 = OpConstant %f32 -inf
        \\%c8 = OpConstant %u32 0x10
        \\%block = OpTypeStruct %u32
        \\%fn = OpTypeFunction %void
        \\%pf = OpTypePointer Function %u32
        \\%sc = OpSpecConstantOp %u32 IAdd %c1 %c1
        \\%main = OpFunction %void None %fn
        \\%entry = OpLabel
        \\%v = OpVariable %pf Function
        \\OpStore %v %c1 Aligned 4
        \\%l = OpLoad %u32 %v Volatile|Aligned 4
        \\%sq = OpExtInst %f32 %glsl Sqrt %c3
        \\%sq2 = OpExtInst %f32 %glsl 31 %c3
        \\OpSelectionMerge %merge None
        \\OpSwitch %l %merge 5 %merge 0x10 %merge
        \\%merge = OpLabel
        \\OpReturn
        \\OpFunctionEnd
        \\%gid = OpVariable %pin Input
        \\%pin = OpTypePointer Input %u32
        \\
    ;
    var diag: Diagnostics = .{};
    const bytes = assemble(testing.allocator, text, &diag) catch |err| {
        std.debug.print("line {d}: {s}\n", .{ diag.line, diag.message });
        return err;
    };
    defer testing.allocator.free(bytes);
    var m = try reader.read(testing.allocator, bytes, null);
    defer m.deinit();
    try testing.expectEqual(@as(u8, 1), m.version_major);
    try testing.expectEqual(@as(u8, 3), m.version_minor);
    try testing.expectEqual(@as(u32, 0x00290900), m.generator);
    try testing.expectEqual(@as(u32, 100), m.bound);
    try testing.expectEqualStrings("main", m.entry_points[0].name);
    try testing.expectEqual(@as(usize, 1), m.entry_points[0].interface.len);
    const em = m.execution_modes[0];
    try testing.expectEqual(@as(u64, 64), em.operands[0].asInt().?);
    // The multi-line string with escapes.
    try testing.expectEqualStrings("a \"b\"\\\nc", m.instructions[7].operands[3].value.string);
    const out = try dis.disassemble(testing.allocator, &m, .{ .names = true, .header = false });
    defer testing.allocator.free(out);
    const has = struct {
        fn f(t: []const u8, n: []const u8) bool {
            return std.mem.indexOf(u8, t, n) != null;
        }
    }.f;
    try testing.expect(has(out, "%c1 = OpConstant %u32 4294967295\n"));
    try testing.expect(has(out, "%c2 = OpConstant %i64 -2\n"));
    try testing.expect(has(out, "%c3 = OpConstant %f32 1.5\n"));
    try testing.expect(has(out, "%c4 = OpConstant %f16 0.5\n"));
    try testing.expect(has(out, "%c5 = OpConstant %f64 -0.25\n"));
    try testing.expect(has(out, "%c6 = OpConstant %f32 0x7fc00001\n"));
    try testing.expect(has(out, "%c7 = OpConstant %f32 -inf\n"));
    try testing.expect(has(out, "%c8 = OpConstant %u32 16\n"));
    try testing.expect(has(out, "%sc = OpSpecConstantOp %u32 IAdd %c1 %c1\n"));
    try testing.expect(has(out, "OpStore %v %c1 Aligned 4\n"));
    try testing.expect(has(out, "%l = OpLoad %u32 %v Volatile|Aligned 4\n"));
    try testing.expect(has(out, "%sq = OpExtInst %f32 %glsl Sqrt %c3\n"));
    try testing.expect(has(out, "%sq2 = OpExtInst %f32 %glsl Sqrt %c3\n"));
    try testing.expect(has(out, "OpSwitch %l %merge 5 %merge 16 %merge\n"));
    try testing.expect(has(out, "OpDecorate %gid BuiltIn GlobalInvocationId\n"));
    try testing.expect(has(out, "OpMemberDecorate %block 0 Offset 16\n"));
    // Forward references: %gid and %pin were used before their definitions.
    try testing.expect(has(out, "%gid = OpVariable %pin Input\n"));
}

test "numeric ids are kept, symbols follow the largest numeric id" {
    const text =
        \\%7 = OpTypeInt 32 0
        \\%x = OpTypeFloat 32
        \\%3 = OpTypeVector %7 2
        \\%y = OpTypeVector %x 2
        \\
    ;
    const bytes = try assemble(testing.allocator, text, null);
    defer testing.allocator.free(bytes);
    var m = try reader.read(testing.allocator, bytes, null);
    defer m.deinit();
    try testing.expectEqual(@as(u32, 10), m.bound);
    try testing.expectEqual(@as(Id, 7), m.instructions[0].result.?);
    try testing.expectEqual(@as(Id, 8), m.instructions[1].result.?);
    try testing.expectEqual(@as(Id, 3), m.instructions[2].result.?);
    try testing.expectEqual(@as(Id, 9), m.instructions[3].result.?);
    try testing.expectEqual(@as(Id, 8), m.instructions[3].operands[0].value.id);
}

test "syntax errors name the line" {
    try expectSyntax("OpCapability Shader\nOpFoo 1\n", 2, "unknown opcode 'OpFoo'");
    try expectSyntax("%1 = OpTypeInt 32\n", 1, "missing operand");
    try expectSyntax("%1 = OpTypeInt 32 0 7\n", 1, "surplus operand");
    try expectSyntax("OpTypeInt 32 0\n", 1, "needs a result id");
    try expectSyntax("%1 = OpCapability Shader\n", 1, "has no result id");
    try expectSyntax("OpCapability Shadr\n", 1, "unknown capability enumerant 'Shadr'");
    try expectSyntax("%1 = OpTypeInt 32 0\n%2 = OpConstant %1 -1\n", 2, "not a 32-bit unsigned integer");
    try expectSyntax("%1 = OpTypeInt 32 0\n%2 = OpConstant %1 4294967296\n", 2, "not a 32-bit unsigned integer");
    try expectSyntax("%1 = OpTypeFloat 32\n%2 = OpConstant %1 abc\n", 2, "not a 32-bit float literal");
    try expectSyntax("OpName %1 \"unterminated\n", 1, "unterminated string");
    try expectSyntax("%1 = OpTypeInt 32 0\n%2 = OpConstant %1 1\n%3 = OpTypeInt 32 0\n%v = OpVariable %3 Function\nOpStore %v %2 Aligned\n", 5, "Aligned needs a literal_integer parameter");
    try expectSyntax("%0 = OpTypeInt 32 0\n", 1, "%0 is not a valid id");
    try expectSyntax("%a-b = OpTypeInt 32 0\n", 1, "bad character in id");
    try expectSyntax("%1 = OpExtInstImport \"GLSL.std.450\"\n%2 = OpTypeFloat 32\n%3 = OpExtInst %2 %1 Sqrtt %2\n", 3, "unknown GLSL.std.450 instruction 'Sqrtt'");
    try expectSyntax("%1 = OpTypeInt 32 0\n%2 = OpSpecConstantOp %1 Nonsense\n", 2, "unknown opcode 'Nonsense'");
    try expectSyntax("%1 = OpTypeInt 32 0\n%2 = OpSpecConstantOp %1 65664 %1 %1\n", 2, "opcode 65664 out of range");
    try expectSyntax("%1 = OpTypeInt 32 0\n%2 = OpSpecConstantOp %1 CompositeExtract %1 %1\n", 2, "expected an integer for literal_integer, got '1'");
    try expectSyntax("%1 = OpTypeInt 32 0\n%2 = OpSpecConstantOp %1 IAdd %1\n", 2, "OpSpecConstantOp OpIAdd: missing operand");
    try expectSyntax("%1 = OpTypeInt 0 1\n%2 = OpConstant %1 -1\n", 1, "width 0 is outside 1..64");
    try expectSyntax("%1 = OpTypeInt 65 0\n", 1, "width 65 is outside 1..64");
    try expectSyntax("%1 = OpTypeInt 8 1\n%2 = OpConstant %1 -129\n", 2, "not a 8-bit signed integer");
    try expectSyntax("%1 = OpTypeInt 8 0\n%2 = OpConstant %1 256\n", 2, "not a 8-bit unsigned integer");
    // Ids at the top of the u32 range: the bound (largest id + 1) must still be a u32.
    try expectSyntax("OpCapability Shader\nOpMemoryModel Logical GLSL450\n%4294967295 = OpTypeVoid\n", 3, "numeric id %4294967295 leaves no room for the bound");
    try expectSyntax("OpCapability Shader\nOpMemoryModel Logical GLSL450\n%4294967294 = OpTypeVoid\n%a = OpTypeBool\n%b = OpTypeFloat 32\n", 4, "symbolic id %a leaves no room for the bound");
    try expectSyntax("%4294967296 = OpTypeVoid\n", 1, "numeric id %4294967296 is out of range");
}

test "decoration groups reach their targets through assemble and read" {
    // A Vulkan buffer decorated through a group (NonWritable, DescriptorSet 1)
    // and OpenCL kernel parameters decorated through one (FuncParamAttr
    // NoAlias), the shapes SPIRV-LLVM-Translator and older glslang emit.
    const text =
        \\OpCapability Shader
        \\OpMemoryModel Logical GLSL450
        \\OpEntryPoint GLCompute %main "main"
        \\OpExecutionMode %main LocalSize 64 1 1
        \\OpDecorate %grp NonWritable
        \\OpDecorate %grp DescriptorSet 1
        \\%grp = OpDecorationGroup
        \\OpGroupDecorate %grp %inbuf
        \\OpDecorate %inbuf Binding 0
        \\OpDecorate %Buf Block
        \\OpMemberDecorate %Buf 0 Offset 0
        \\OpDecorate %fpgrp FuncParamAttr NoAlias
        \\%fpgrp = OpDecorationGroup
        \\OpGroupDecorate %fpgrp %pout %pin
        \\%void = OpTypeVoid
        \\%u32 = OpTypeInt 32 0
        \\%Buf = OpTypeStruct %u32
        \\%pBuf = OpTypePointer StorageBuffer %Buf
        \\%inbuf = OpVariable %pBuf StorageBuffer
        \\%pu32 = OpTypePointer StorageBuffer %u32
        \\%fnty = OpTypeFunction %void
        \\%hty = OpTypeFunction %void %pu32 %pu32
        \\%main = OpFunction %void None %fnty
        \\%l0 = OpLabel
        \\OpReturn
        \\OpFunctionEnd
        \\%helper = OpFunction %void None %hty
        \\%pout = OpFunctionParameter %pu32
        \\%pin = OpFunctionParameter %pu32
        \\%l1 = OpLabel
        \\OpReturn
        \\OpFunctionEnd
        \\
    ;
    const bytes = try assemble(testing.allocator, text, null);
    defer testing.allocator.free(bytes);
    var m = try reader.read(testing.allocator, bytes, null);
    defer m.deinit();
    const inbuf = m.globals[0].id;
    try testing.expect(m.hasDecoration(inbuf, .non_writable));
    try testing.expectEqual(@as(u64, 1), m.decoration(inbuf, .descriptor_set).?.firstInt().?);
    try testing.expectEqual(@as(u64, 0), m.decoration(inbuf, .binding).?.firstInt().?);
    const helper = m.functions[1];
    try testing.expectEqual(@as(usize, 2), helper.params.len);
    for (helper.params) |p| {
        const d = m.decoration(p.id, .func_param_attr) orelse return error.TestUnexpectedResult;
        try testing.expectEqual(@as(u64, @backingInt(spec.FunctionParameterAttribute.no_alias)), d.firstInt().?);
    }
}

test "the largest usable id assembles and reads back" {
    const bytes = try assemble(testing.allocator, "OpCapability Shader\nOpMemoryModel Logical GLSL450\n%4294967293 = OpTypeVoid\n%a = OpTypeBool\n", null);
    defer testing.allocator.free(bytes);
    var m = try reader.read(testing.allocator, bytes, null);
    defer m.deinit();
    try testing.expectEqual(@as(u32, std.math.maxInt(u32)), m.bound);
}

test "narrow signed literals are sign-extended to the word, unsigned ones are not" {
    const text =
        \\%i8 = OpTypeInt 8 1
        \\%i16 = OpTypeInt 16 1
        \\%u8 = OpTypeInt 8 0
        \\%i48 = OpTypeInt 48 1
        \\%u48 = OpTypeInt 48 0
        \\%a = OpConstant %i8 -1
        \\%b = OpConstant %i16 -2
        \\%c = OpConstant %u8 200
        \\%d = OpConstant %i8 127
        \\%e = OpConstant %i8 -128
        \\%f = OpConstant %i48 -1
        \\%g = OpConstant %u48 0xFFFFFFFFFFFF
        \\%h = OpConstant %i16 0x7FFF
        \\%bool = OpTypeBool
        \\%fn = OpTypeFunction %i8
        \\%fun = OpFunction %i8 None %fn
        \\%l = OpLabel
        \\OpSwitch %a %l -1 %l 5 %l
        \\OpFunctionEnd
        \\
    ;
    var diag: Diagnostics = .{};
    const bytes = assemble(testing.allocator, text, &diag) catch |err| {
        std.debug.print("line {d}: {s}\n", .{ diag.line, diag.message });
        return err;
    };
    defer testing.allocator.free(bytes);
    var m = try reader.read(testing.allocator, bytes, null);
    defer m.deinit();
    // Symbolic ids are numbered in order of first appearance: %i8=1 %i16=2 %u8=3 %i48=4 %u48=5 %a=6 ...
    try testing.expectEqual(@as(u64, 0xFFFFFFFF), m.constantOf(6).?.value.scalar);
    try testing.expectEqual(@as(u64, 0xFF), m.constantBits(6).?);
    try testing.expectEqual(@as(u64, 0xFFFFFFFE), m.constantOf(7).?.value.scalar);
    try testing.expectEqual(@as(u64, 200), m.constantOf(8).?.value.scalar);
    try testing.expectEqual(@as(u64, 127), m.constantOf(9).?.value.scalar);
    try testing.expectEqual(@as(u64, 0xFFFFFF80), m.constantOf(10).?.value.scalar);
    try testing.expectEqual(@as(u64, 0xFFFFFFFFFFFFFFFF), m.constantOf(11).?.value.scalar);
    try testing.expectEqual(@as(u64, 0xFFFFFFFFFFFF), m.constantBits(11).?);
    try testing.expectEqual(@as(u64, 0xFFFFFFFFFFFF), m.constantOf(12).?.value.scalar);
    try testing.expectEqual(@as(u64, 0x7FFF), m.constantOf(13).?.value.scalar);
    // The OpSwitch case literal of the i8 selector is sign-extended too.
    const sw = m.instructions[m.instructions.len - 2];
    try testing.expectEqual(spec.Opcode.OpSwitch, sw.op);
    try testing.expectEqual(@as(u64, 0xFFFFFFFF), sw.operands[2].asInt().?);
    try testing.expectEqual(@as(u64, 5), sw.operands[4].asInt().?);
    // And the disassembly prints the values, so it re-assembles to the same bytes.
    const out = try dis.disassemble(testing.allocator, &m, .{});
    defer testing.allocator.free(out);
    try testing.expect(std.mem.indexOf(u8, out, "%6 = OpConstant %1 -1\n") != null);
    try testing.expect(std.mem.indexOf(u8, out, "%7 = OpConstant %2 -2\n") != null);
    try testing.expect(std.mem.indexOf(u8, out, "%10 = OpConstant %1 -128\n") != null);
    try testing.expect(std.mem.indexOf(u8, out, "%11 = OpConstant %4 -1\n") != null);
    try testing.expect(std.mem.indexOf(u8, out, "OpSwitch %6 %17 -1 %17 5 %17\n") != null);
    const again = try assemble(testing.allocator, out, null);
    defer testing.allocator.free(again);
    try testing.expectEqualSlices(u8, bytes, again);
    try testing.expectEqual(@as(u64, 0xFFFFFFFF), signExtendToWord(0xFF, 8, true));
    try testing.expectEqual(@as(u64, 0x7F), signExtendToWord(0x7F, 8, true));
    try testing.expectEqual(@as(u64, 0xFF), signExtendToWord(0xFF, 8, false));
    try testing.expectEqual(@as(u64, 0xFFFFFFFF), signExtendToWord(0xFFFFFFFF, 32, true));
    try testing.expectEqual(@as(u64, 0x8000_0000), signExtendToWord(0x8000_0000, 33, true));
    try testing.expectEqual(@as(u64, 0xFFFF_FFFF_0000_0000), signExtendToWord(0x1_0000_0000, 33, true));
}

test "nan and inf literals keep their sign and width" {
    const text =
        \\%f16 = OpTypeFloat 16
        \\%f32 = OpTypeFloat 32
        \\%f64 = OpTypeFloat 64
        \\%a = OpConstant %f32 -nan
        \\%b = OpConstant %f32 nan
        \\%c = OpConstant %f16 -nan
        \\%d = OpConstant %f64 -nan
        \\%e = OpConstant %f16 -inf
        \\%f = OpConstant %f64 inf
        \\%g = OpConstant %f32 0xffc00001
        \\
    ;
    const bytes = try assemble(testing.allocator, text, null);
    defer testing.allocator.free(bytes);
    var m = try reader.read(testing.allocator, bytes, null);
    defer m.deinit();
    try testing.expectEqual(@as(u64, 0xffc00000), m.constantOf(4).?.value.scalar);
    try testing.expectEqual(@as(u64, 0x7fc00000), m.constantOf(5).?.value.scalar);
    try testing.expectEqual(@as(u64, 0xfe00), m.constantOf(6).?.value.scalar);
    try testing.expectEqual(@as(u64, 0xfff8000000000000), m.constantOf(7).?.value.scalar);
    try testing.expectEqual(@as(u64, 0xfc00), m.constantOf(8).?.value.scalar);
    try testing.expectEqual(@as(u64, 0x7ff0000000000000), m.constantOf(9).?.value.scalar);
    try testing.expectEqual(@as(u64, 0xffc00001), m.constantOf(10).?.value.scalar);
    const out = try dis.disassemble(testing.allocator, &m, .{});
    defer testing.allocator.free(out);
    try testing.expect(std.mem.indexOf(u8, out, "%4 = OpConstant %2 -nan\n") != null);
    try testing.expect(std.mem.indexOf(u8, out, "%6 = OpConstant %1 -nan\n") != null);
    try testing.expect(std.mem.indexOf(u8, out, "%7 = OpConstant %3 -nan\n") != null);
    try testing.expect(std.mem.indexOf(u8, out, "%8 = OpConstant %1 -inf\n") != null);
    try testing.expect(std.mem.indexOf(u8, out, "%10 = OpConstant %2 0xffc00001\n") != null);
    const again = try assemble(testing.allocator, out, null);
    defer testing.allocator.free(again);
    try testing.expectEqualSlices(u8, bytes, again);
}

test "an FPEncoding float takes raw bits only and round-trips through dis" {
    // The review's bf.spvasm: `OpConstant %bf 1.0` used to assemble to the
    // half 0x3C00 (bfloat16 0.0078125) and dis printed 0x3f80 as 1.875.
    const text =
        \\%bf = OpTypeFloat 16 BFloat16KHR
        \\%f16 = OpTypeFloat 16
        \\%f8 = OpTypeFloat 8 Float8E4M3EXT
        \\%c = OpConstant %bf 0x3f80
        \\%h = OpConstant %f16 1.0
        \\%e = OpConstant %f8 0x38
        \\
    ;
    const bytes = try assemble(testing.allocator, text, null);
    defer testing.allocator.free(bytes);
    var m = try reader.read(testing.allocator, bytes, null);
    defer m.deinit();
    try testing.expectEqual(@as(?u32, 0), m.typeOf(1).?.float.encoding);
    try testing.expectEqual(@as(u64, 0x3f80), m.constantOf(4).?.value.scalar);
    try testing.expectEqual(@as(u64, 0x3c00), m.constantOf(5).?.value.scalar);
    try testing.expectEqual(@as(u64, 0x38), m.constantOf(6).?.value.scalar);
    const out = try dis.disassemble(testing.allocator, &m, .{});
    defer testing.allocator.free(out);
    try testing.expect(std.mem.indexOf(u8, out, "%4 = OpConstant %1 0x3f80\n") != null);
    try testing.expect(std.mem.indexOf(u8, out, "%5 = OpConstant %2 1\n") != null);
    const again = try assemble(testing.allocator, out, null);
    defer testing.allocator.free(again);
    try testing.expectEqualSlices(u8, bytes, again);

    try expectSyntax("%bf = OpTypeFloat 16 BFloat16KHR\n%d = OpConstant %bf 1.0\n", 2, "FPEncoding BFloat16KHR takes raw `0x` bits only");
    try expectSyntax("%bf = OpTypeFloat 16 BFloat16KHR\n%d = OpConstant %bf nan\n", 2, "takes raw `0x` bits only");
    try expectSyntax("%bf = OpTypeFloat 16 BFloat16KHR\n%d = OpConstant %bf 0x13f80\n", 2, "takes raw `0x` bits only");
    try expectSyntax("%f8 = OpTypeFloat 8 Float8E5M2EXT\n%d = OpConstant %f8 0x100\n", 2, "FPEncoding Float8E5M2EXT");
}

test "OpSpecConstantOp takes the named opcode's operands, literal indexes included" {
    const text =
        \\%u32 = OpTypeInt 32 0
        \\%v4 = OpTypeVector %u32 4
        \\%one = OpConstant %u32 1
        \\%two = OpSpecConstant %u32 2
        \\%vec = OpSpecConstantComposite %v4 %one %two %one %two
        \\%x = OpSpecConstantOp %u32 CompositeExtract %vec 0
        \\%y = OpSpecConstantOp %v4 VectorShuffle %vec %vec 3 2 1 0
        \\%z = OpSpecConstantOp %v4 CompositeInsert %two %vec 2
        \\%w = OpSpecConstantOp %u32 IAdd %x %two
        \\%q = OpSpecConstantOp %u32 OpISub %w %one
        \\
    ;
    var diag: Diagnostics = .{};
    const bytes = assemble(testing.allocator, text, &diag) catch |err| {
        std.debug.print("line {d}: {s}\n", .{ diag.line, diag.message });
        return err;
    };
    defer testing.allocator.free(bytes);
    var m = try reader.read(testing.allocator, bytes, null);
    defer m.deinit();
    const x = m.constantOf(6).?.value.spec_op;
    try testing.expectEqual(spec.Opcode.OpCompositeExtract, x.opcode);
    try testing.expectEqual(@as(u64, 0), x.operands[1].asInt().?);
    const y = m.constantOf(7).?.value.spec_op;
    try testing.expectEqual(@as(usize, 6), y.operands.len);
    try testing.expectEqual(@as(u64, 3), y.operands[2].asInt().?);
    const out = try dis.disassemble(testing.allocator, &m, .{});
    defer testing.allocator.free(out);
    try testing.expect(std.mem.indexOf(u8, out, "%6 = OpSpecConstantOp %1 CompositeExtract %5 0\n") != null);
    try testing.expect(std.mem.indexOf(u8, out, "%7 = OpSpecConstantOp %2 VectorShuffle %5 %5 3 2 1 0\n") != null);
    try testing.expect(std.mem.indexOf(u8, out, "%8 = OpSpecConstantOp %2 CompositeInsert %4 %5 2\n") != null);
    try testing.expect(std.mem.indexOf(u8, out, "%10 = OpSpecConstantOp %1 ISub %9 %3\n") != null);
    const again = try assemble(testing.allocator, out, null);
    defer testing.allocator.free(again);
    try testing.expectEqualSlices(u8, bytes, again);
}

test "assemble -> read -> dis -> assemble is the identity on every corpus file" {
    for (corpus_files, corpus_text) |name, text| {
        var diag: Diagnostics = .{};
        const bin1 = assemble(testing.allocator, text, &diag) catch |err| {
            std.debug.print("{s}:{d}: {s}\n", .{ name, diag.line, diag.message });
            return err;
        };
        defer testing.allocator.free(bin1);
        var rd: reader.Diagnostics = .{};
        var m = reader.read(testing.allocator, bin1, &rd) catch |err| {
            std.debug.print("{s}: word {d}: {s}\n", .{ name, rd.word_offset, rd.message });
            return err;
        };
        defer m.deinit();
        try testing.expect(m.instructions.len > 0);
        // Numeric ids: the text must assemble back to the same bytes.
        const text2 = try dis.disassemble(testing.allocator, &m, .{});
        defer testing.allocator.free(text2);
        const bin2 = assemble(testing.allocator, text2, &diag) catch |err| {
            std.debug.print("{s} (re-assembling its disassembly):{d}: {s}\n", .{ name, diag.line, diag.message });
            return err;
        };
        defer testing.allocator.free(bin2);
        if (!std.mem.eql(u8, bin1, bin2)) {
            std.debug.print("{s}: round trip differs ({d} vs {d} bytes)\n", .{ name, bin1.len, bin2.len });
            return error.TestUnexpectedResult;
        }
        // Named ids: the text must assemble and decode to the same instruction stream.
        const text3 = try dis.disassemble(testing.allocator, &m, .{ .names = true });
        defer testing.allocator.free(text3);
        const bin3 = assemble(testing.allocator, text3, &diag) catch |err| {
            std.debug.print("{s} (re-assembling its named disassembly):{d}: {s}\n", .{ name, diag.line, diag.message });
            return err;
        };
        defer testing.allocator.free(bin3);
        var m3 = try reader.read(testing.allocator, bin3, null);
        defer m3.deinit();
        try testing.expectEqual(m.instructions.len, m3.instructions.len);
        try testing.expectEqual(m.functions.len, m3.functions.len);
        // (The bound may grow: an id that was numeric in the text but carries a
        // unique OpName gets a fresh symbolic id above the numeric ones.)
        for (m.instructions, m3.instructions) |x, y| {
            try testing.expectEqual(x.op, y.op);
            try testing.expectEqual(x.word_count, y.word_count);
        }
    }
}

test "the zig-*.spvasm corpus files are the current disassembly of testdata/*.spv" {
    // Pins the corpus text to the binaries (and so, through the header
    // commands, to the sources): a regenerated binary with a stale corpus
    // file fails here. The text after the comment header is exactly what
    // `zig build spirv-dis -- <bin>` prints (names on, header on).
    const pairs = [_]struct { []const u8, []const u8 }{
        .{ @embedFile("testdata/k-spirv64-vulkan.spv"), @embedFile("corpus/zig-vulkan-params.spvasm") },
        .{ @embedFile("testdata/k-spirv64-opencl.spv"), @embedFile("corpus/zig-opencl-params.spvasm") },
        .{ @embedFile("testdata/desc.spv"), @embedFile("corpus/zig-crosscheck.spvasm") },
        .{ @embedFile("testdata/cf.spv"), @embedFile("corpus/zig-control-flow.spvasm") },
        .{ @embedFile("testdata/narrow.spv"), @embedFile("corpus/zig-narrow-ints.spvasm") },
    };
    for (pairs) |pair| {
        var m = try reader.read(testing.allocator, pair[0], null);
        defer m.deinit();
        const text = try dis.disassemble(testing.allocator, &m, .{ .names = true });
        defer testing.allocator.free(text);
        const start = std.mem.indexOf(u8, pair[1], "; SPIR-V\n").?;
        try testing.expectEqualStrings(text, pair[1][start..]);
    }
}

test "dis -> assemble reproduces the Zig binaries byte for byte" {
    const bins = [_][]const u8{
        @embedFile("testdata/k-spirv64-vulkan.spv"),
        @embedFile("testdata/k-spirv64-opencl.spv"),
        @embedFile("testdata/desc.spv"),
        @embedFile("testdata/cf.spv"),
        @embedFile("testdata/narrow.spv"),
    };
    for (bins) |bin| {
        var m = try reader.read(testing.allocator, bin, null);
        defer m.deinit();
        const text = try dis.disassemble(testing.allocator, &m, .{});
        defer testing.allocator.free(text);
        var diag: Diagnostics = .{};
        const again = assemble(testing.allocator, text, &diag) catch |err| {
            std.debug.print("line {d}: {s}\n", .{ diag.line, diag.message });
            return err;
        };
        defer testing.allocator.free(again);
        try testing.expectEqualSlices(u8, bin, again);
    }
}
