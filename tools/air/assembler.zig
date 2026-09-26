//! LLVM IR text -> bitcode, for the subset of IR Zig emits for shader code.
//!
//! Input is the text produced by air-splice's rewrite pass: opaque pointers,
//! AIR target, no attribute groups, no metadata. Output is a bitcode module
//! containing one entry point (Apple packages one module per function), every
//! `define` that entry point reaches through calls, and only the globals and
//! declarations those functions name: a global enters the module the first
//! time something refers to it, so a table that only another entry point
//! uses can neither bloat this module nor break its assembly.
//!
//! Supported: integer/float/vector/array/struct types, named struct types,
//! globals, declarations, and these instructions: ret, br, switch, phi,
//! select, icmp, fcmp, integer and float arithmetic, fneg, freeze, casts,
//! getelementptr, load, store (plain and atomic), atomicrmw, cmpxchg, alloca,
//! insertvalue, extractvalue, insertelement, extractelement, shufflevector,
//! call, unreachable. Constants: literals, aggregates, `splat (T c)`,
//! `c"..."` byte strings and `getelementptr` constant expressions.
//! Anything else is reported as `error.Unsupported` with the offending line.
//!
//! AIR facts baked in here (DESIGN.md D1/D5, research r06/r11/r13):
//!   * Metal knows address spaces 0 (thread), 1 (device), 2 (constant) and
//!     3 (threadgroup). Zig's nvptx IR also uses 4 (param) and 5 (local);
//!     both crash the runtime compiler, so types are renumbered 4 -> 2 and
//!     5 -> 0 while parsing.
//!   * A module-level `constant` global left in address space 0 fails
//!     pipeline creation with "Undefined symbols: ___anon_N"; such globals
//!     are relocated to address space 2. A mutable (`global`) one in address
//!     space 0 fails the same way (`Undefined symbols: _name`, r11 open
//!     question 2 / review-a1/fix1check/v_mutglobal.check.log) and has no
//!     Metal address space to move to: Metal's only writable program-scope
//!     storage is device or threadgroup memory. Refused with a hint.
//!   * To keep that consistent without text surgery, named operands (`%x`,
//!     `@g`) always take the type they were defined with; the textual type
//!     annotation is only used to parse literal constants. GEP results
//!     inherit the base pointer's address space.
//!   * An aggregate or vector that stores the address of a relocated
//!     constant (`[4 x ptr] [ptr @tab_a, ...]`, `<2 x ptr>`, `{ ptr, i64 }`
//!     slices, whether as a constant or built with `insertvalue` /
//!     `insertelement`) is refused: the Builder asserts (rather than
//!     reports) that the element types match, and a `load ptr` from such a
//!     table would yield a thread-space pointer into constant memory, which
//!     Metal cannot dereference.
//!   * `llvm.memcpy/memmove/memset.pX.pY` are re-declared under the name
//!     matching the actual operand address spaces when relocation changed
//!     them. Any other argument/parameter mismatch is an error.
//!   * LLVM >= 19 instruction flags (`trunc nuw`, `zext nneg`, `or disjoint`,
//!     `icmp samesign`, GEP `nuw`/`nusw`) are dropped: the Builder would
//!     encode them as records Apple's LLVM-15-era reader rejects
//!     ("Failed to materializeAll").
//!   * Native `atomicrmw`, `cmpxchg`, `load atomic` and `store atomic` are
//!     accepted by Metal; `fence` crashes its runtime compiler.
//!   * Convergent calls (`air.wg.barrier`, `air.simdgroup.barrier`, every
//!     `air.simd_*` collective, and any helper that transitively makes one;
//!     intrinsics.isConvergent) must sit in a block every thread they
//!     synchronise executes: the whole threadgroup for `air.wg.barrier`,
//!     the whole SIMD-group for the SIMD-group calls (divergence.Scope).
//!     Zig cannot mark the extern declarations `convergent`, so LLVM's
//!     jump threading duplicates a barrier that follows `if (tidx == 0)
//!     x = 0;` into both arms of the branch; Metal's compiler accepts the
//!     result and the threadgroup silently desynchronises
//!     (review-b/fable-hazard/hazard.check.log), and a duplicated
//!     `simd_sum` adds up only the lanes of one arm
//!     (review-b/r4/helper/check.log). divergence.zig decides which threads
//!     reach a block together from the manifest's builtins (a uniformity
//!     analysis over threadgroup / SIMD-group / thread levels, with
//!     post-dominance and loop reasoning), so a loop bounded by
//!     `threads_per_threadgroup` may hold a barrier and `if (sgid == 0)
//!     simd_sum(...)` is fine, while `if (gid >= n) return; barrier` and
//!     `if (sgid == 0) barrier` are refused.
//!   * Intrinsics go through intrinsics.zig (DESIGN.md D5-12): `llvm.sin`
//!     and friends are called under their `air.fast_*` name (the `llvm.*`
//!     declaration never enters the module), `llvm.vector.reduce.*` is
//!     expanded into an `extractelement` chain (an op with no expansion is
//!     refused), `bitcast <N x i1>` to an integer is packed lane by lane
//!     and `bitcast iN` to `<N x i1>` is unpacked with vector shifts; the
//!     original forms all crash the GPU backend.
//!   * Quoted identifiers (`@"mod.Vec(4).sum"`, Zig's spelling for
//!     helpers inside generic instantiations) are read wherever a global
//!     name may appear; the name enters the module without the quotes.
//!   * `addrspacecast` is unverified on Metal, `llvm.ctpop.i4` crashes the
//!     GPU backend, and `llvm.nvvm.*` has no AIR equivalent: all are
//!     rejected with a hint.

const std = @import("std");
const Allocator = std.mem.Allocator;
const Builder = std.zig.llvm.Builder;
const Type = Builder.Type;
const Value = Builder.Value;
const Constant = Builder.Constant;
const WipFunction = Builder.WipFunction;
const metadata = @import("metadata.zig");
const intrinsics = @import("intrinsics.zig");
const divergence = @import("divergence.zig");
const air_target = @import("target.zig");

pub const Error = error{
    OutOfMemory,
    Syntax,
    Unsupported,
    UndefinedValue,
    UndefinedType,
    UndefinedBlock,
    UndefinedGlobal,
    EntryPointMissing,
};

pub const Options = struct {
    /// Function to assemble. Other `define`s in the text are included only
    /// when this one reaches them through calls.
    entry: []const u8,
    /// Metadata to attach for that entry point.
    fm: metadata.FunctionMetadata,
};

/// Prefix air-splice puts in front of LLVM's numbered temporaries (`%7` ->
/// `%__7`). The assembler needs it to name the implicit entry block, which
/// LLVM numbers right after the parameters and phis refer to by number.
pub const numbered_prefix = "__";

/// Metal's `constant` address space; relocation target for addrspace-0
/// constant globals.
const constant_space: Builder.AddrSpace = @fromBackingInt(@intCast(2));

/// Assemble one entry point into a bitcode module for the deployment target
/// `profile` (triple and `!air.version` / `!air.language_version`). Returns
/// the raw bitcode bytes (no darwin wrapper).
pub fn assemble(gpa: Allocator, ir: []const u8, comptime opts: Options, profile: air_target.Profile) Error![]u8 {
    var b = try Builder.init(.{ .allocator = gpa, .strip = true, .name = "air-splice" });
    defer b.deinit();
    try build(&b, gpa, ir, opts, profile);
    const words = try b.toBitcode(gpa, .{ .name = "zig air-splice", .version = .{ .major = 0, .minor = 1, .patch = 0 } });
    defer gpa.free(words);
    return gpa.dupe(u8, std.mem.sliceAsBytes(words));
}

/// Populate `b` with the module for `opts.entry`.
fn build(b: *Builder, gpa: Allocator, ir: []const u8, comptime opts: Options, profile: air_target.Profile) Error!void {
    b.source_filename = try b.string(opts.entry);
    b.target_triple = try b.string(profile.triple);
    b.data_layout.deinit(gpa);
    b.data_layout = try Builder.DataLayout.parseString(try b.string(air_target.datalayout), b);

    var mod = ModuleState{ .gpa = gpa, .b = b };
    defer mod.deinit();

    // Pass 1: named types, global names, declarations and function headers.
    // Function bodies and global initialisers are kept as text and lowered
    // on demand in pass 2.
    var lines = std.mem.splitScalar(u8, ir, '\n');
    var line_no: usize = 0;
    while (lines.next()) |raw| {
        line_no += 1;
        const line = std.mem.trim(u8, raw, " \t\r");
        if (line.len == 0 or line[0] == ';' or line[0] == '!') continue;
        if (std.mem.startsWith(u8, line, "define ")) {
            const header_line = line_no;
            var body: std.ArrayList([]const u8) = .empty;
            errdefer body.deinit(gpa);
            while (lines.next()) |l| {
                line_no += 1;
                if (std.mem.eql(u8, std.mem.trim(u8, l, " \t\r"), "}")) break;
                try body.append(gpa, l);
            }
            try mod.registerDefine(line, header_line, try body.toOwnedSlice(gpa));
            continue;
        }
        if (std.mem.startsWith(u8, line, "target ") or std.mem.startsWith(u8, line, "source_filename")) continue;
        if (line[0] == '%') {
            // %name = type { ... }
            var c = Cursor{ .s = line, .line = line_no };
            try c.expect("%");
            const name = try mod.quotedOrPlainName(&c);
            try c.expect("=");
            try c.expectWord("type");
            try mod.named_types.append(gpa, .{ .name = name, .body = c.rest() });
            continue;
        }
        if (line[0] == '@') {
            try mod.registerGlobal(line, line_no);
            continue;
        }
        if (std.mem.startsWith(u8, line, "declare ")) {
            try mod.declare(line, line_no);
            continue;
        }
        return fail(line_no, line, "unrecognised top-level construct", error.Syntax);
    }

    // Pass 2: the entry point, then every define it reaches through calls.
    const entry = mod.findDefine(opts.entry) orelse return error.EntryPointMissing;
    if (mod.defines.items[entry].body == null) return error.EntryPointMissing;
    try mod.analyseDivergence(entry, opts.fm.param_uniformity);
    try mod.queue(entry);
    var i: usize = 0;
    while (i < mod.worklist.items.len) : (i += 1) try mod.lowerDefine(mod.worklist.items[i]);
    try metadata.lowerModule(b, opts.fm, mod.defines.items[entry].func.?, mod.samplers.items, profile);
}

fn fail(line_no: usize, line: []const u8, msg: []const u8, err: Error) Error {
    std.debug.print("air-splice: line {d}: {s}\n    {s}\n", .{ line_no, msg, line });
    return err;
}

/// Metal accepts only address spaces 0..3; Zig's nvptx IR also emits
/// 4 (param -> Metal constant) and 5 (local -> Metal thread).
fn mapAddrSpace(n: u24) Builder.AddrSpace {
    return @fromBackingInt(@intCast(switch (n) {
        4 => 2,
        5 => 0,
        else => n,
    }));
}

// ── Module state ────────────────────────────────────────────────────────────

const NamedType = struct { name: []const u8, body: []const u8, resolved: ?Type = null };

/// A `define` or `declare` seen in pass 1. The Builder function is created
/// on first reference, so helpers and intrinsics the entry point never
/// reaches leave no trace in the module (a body-less `private` function
/// would even be an invalid declaration).
const Define = struct {
    name: []const u8,
    ret_ty: Type,
    fn_ty: Type,
    linkage: Builder.Linkage,
    call_conv: Builder.CallConv,
    param_names: []const []const u8,
    /// null for a `declare`.
    body: ?[]const []const u8,
    header_line: usize,
    func: ?Builder.Function.Index = null,
    queued: bool = false,
};

const Global = struct {
    name: []const u8,
    index: Builder.Global.Index,
    fn_ty: ?Type,
    call_conv: Builder.CallConv = .ccc,
    /// Index into `ModuleState.defines` when this is a function with a body.
    define: ?usize = null,
};

/// A `@name = ... constant|global ...` line seen in pass 1. Like functions,
/// a variable is parsed and added to the Builder on first reference, so a
/// global the entry point never touches costs nothing and cannot fail.
const PendingGlobal = struct {
    name: []const u8,
    line: []const u8,
    line_no: usize,
    state: enum { pending, defining, done } = .pending,
};

const ModuleState = struct {
    gpa: Allocator,
    b: *Builder,
    named_types: std.ArrayList(NamedType) = .empty,
    globals: std.ArrayList(Global) = .empty,
    /// Variables not yet referenced (see `PendingGlobal`).
    pending_globals: std.ArrayList(PendingGlobal) = .empty,
    defines: std.ArrayList(Define) = .empty,
    /// Defines to lower, in discovery order.
    worklist: std.ArrayList(usize) = .empty,
    /// Names allocated for re-declared intrinsics.
    owned_names: std.ArrayList([]u8) = .empty,
    samplers: std.ArrayList(std.zig.llvm.Builder.Global.Index) = .empty,
    /// Uniformity analysis results, parallel to `defines`
    /// (`analyseDivergence`).
    funcs: []divergence.Func = &.{},

    fn deinit(m: *ModuleState) void {
        for (m.funcs) |*f| f.deinit(m.gpa);
        m.gpa.free(m.funcs);
        for (m.defines.items) |d| {
            m.gpa.free(d.param_names);
            if (d.body) |body| m.gpa.free(body);
        }
        for (m.owned_names.items) |n| m.gpa.free(n);
        m.owned_names.deinit(m.gpa);
        m.samplers.deinit(m.gpa);
        m.worklist.deinit(m.gpa);
        m.defines.deinit(m.gpa);
        m.named_types.deinit(m.gpa);
        m.pending_globals.deinit(m.gpa);
        m.globals.deinit(m.gpa);
    }

    /// The identifier after `@`: a plain name, or a quoted one
    /// (`@"mod.Vec(4).sum"`).
    fn globalName(m: *ModuleState, c: *Cursor) Error![]const u8 {
        return m.quotedOrPlainName(c);
    }

    /// An identifier after `@` or `%`: a plain name, or a quoted one whose
    /// quotes are dropped and whose `\XX` escapes are decoded (the decoded
    /// copy lives in `owned_names`). Zig quotes every name that carries a
    /// generic instantiation, for functions (`@"mod.Vec(4).sum"`) and for
    /// named types alike (`%"mod.Pair(f32)" = type { float, float }`), so
    /// every definition and reference is compared on the unquoted, decoded
    /// spelling.
    fn quotedOrPlainName(m: *ModuleState, c: *Cursor) Error![]const u8 {
        c.skipWs();
        if (c.i >= c.s.len or c.s[c.i] != '"') return c.word();
        const start = c.i + 1;
        var j = start;
        var escaped = false;
        while (j < c.s.len and c.s[j] != '"') : (j += 1) {
            if (c.s[j] == '\\') {
                escaped = true;
                j += 2;
            }
        }
        if (j >= c.s.len) return fail(c.line, c.s, "unterminated quoted identifier", error.Syntax);
        c.i = j + 1;
        const raw = c.s[start..j];
        if (!escaped) return raw;
        var out = try m.gpa.alloc(u8, raw.len);
        errdefer m.gpa.free(out);
        var n: usize = 0;
        var k: usize = 0;
        while (k < raw.len) : (k += 1) {
            if (raw[k] == '\\' and k + 2 < raw.len) {
                if (std.fmt.parseInt(u8, raw[k + 1 .. k + 3], 16)) |byte| {
                    out[n] = byte;
                    n += 1;
                    k += 2;
                    continue;
                } else |_| {}
            }
            out[n] = raw[k];
            n += 1;
        }
        try m.owned_names.append(m.gpa, out);
        return out[0..n];
    }

    fn findDefine(m: *ModuleState, name: []const u8) ?usize {
        for (m.defines.items, 0..) |d, i| if (std.mem.eql(u8, d.name, name)) return i;
        return null;
    }

    /// Between the passes: which threads reach every block of every
    /// function together and which functions execute a convergent call
    /// (divergence.zig). The entry's parameters are seeded from the
    /// manifest (`FunctionMetadata.param_uniformity`, indexed like the IR
    /// parameters; a parameter past the manifest is taken as per-thread).
    fn analyseDivergence(m: *ModuleState, entry: usize, param_uniformity: []const divergence.Level) Error!void {
        const funcs = try m.gpa.alloc(divergence.Func, m.defines.items.len);
        var n: usize = 0;
        errdefer {
            for (funcs[0..n]) |*f| f.deinit(m.gpa);
            m.gpa.free(funcs);
        }
        for (m.defines.items) |d| {
            funcs[n] = try divergence.Func.init(m.gpa, d.name, d.param_names, d.body);
            n += 1;
        }
        for (funcs[entry].param_level, 0..) |*pl, i| pl.* = if (i < param_uniformity.len) param_uniformity[i] else .thread;
        try divergence.analyse(m.gpa, funcs);
        m.funcs = funcs;
    }

    /// The set of threads a call to `name` synchronises or communicates
    /// across: that of the convergent intrinsic, or the widest one a define
    /// reaches through calls (`.none` for everything else).
    fn convergentScope(m: *ModuleState, name: []const u8) divergence.Scope {
        const scope = divergence.callScope(name);
        const di = m.findDefine(name) orelse return scope;
        return if (di < m.funcs.len) scope.join(m.funcs[di].convergent) else scope;
    }

    /// Variables and functions something has referred to. A variable,
    /// define or declare enters the Builder the first time it is named;
    /// a variable's initialiser may in turn name further globals.
    fn findGlobal(m: *ModuleState, name: []const u8) Error!?Global {
        for (m.globals.items) |g| if (std.mem.eql(u8, g.name, name)) return g;
        for (m.pending_globals.items) |*pg| if (std.mem.eql(u8, pg.name, name)) {
            switch (pg.state) {
                .pending => {},
                .defining => return fail(pg.line_no, pg.line, "global initialiser refers back to the global being defined", error.Unsupported),
                .done => unreachable, // it is in `globals`
            }
            pg.state = .defining;
            try m.defineGlobal(pg.line, pg.line_no);
            pg.state = .done;
            for (m.globals.items) |g| if (std.mem.eql(u8, g.name, name)) return g;
            unreachable;
        };
        const di = m.findDefine(name) orelse return null;
        const d = &m.defines.items[di];
        const func = try m.b.addFunction(d.fn_ty, try m.b.strtabString(d.name), .default);
        func.setLinkage(d.linkage, m.b);
        func.setCallConv(d.call_conv, m.b);
        d.func = func;
        const g = Global{ .name = d.name, .index = func.ptr(m.b).global, .fn_ty = d.fn_ty, .call_conv = d.call_conv, .define = di };
        try m.globals.append(m.gpa, g);
        return g;
    }

    const Callee = struct { fn_ty: Type, call_conv: Builder.CallConv, define: ?usize };

    /// Type information for a function without adding it to the Builder,
    /// so a call that ends up re-declared (memcpy renaming) leaves no stray
    /// declaration behind.
    fn calleeInfo(m: *ModuleState, name: []const u8, c: *Cursor) Error!Callee {
        for (m.globals.items) |g| if (std.mem.eql(u8, g.name, name)) {
            const fn_ty = g.fn_ty orelse return fail(c.line, c.s, "call target is not a function", error.Syntax);
            return .{ .fn_ty = fn_ty, .call_conv = g.call_conv, .define = g.define };
        };
        const di = m.findDefine(name) orelse return fail(c.line, c.s, "call to undeclared function", error.UndefinedGlobal);
        const d = m.defines.items[di];
        return .{ .fn_ty = d.fn_ty, .call_conv = d.call_conv, .define = di };
    }

    fn queue(m: *ModuleState, di: usize) Error!void {
        const d = &m.defines.items[di];
        if (d.queued or d.body == null) return;
        d.queued = true;
        if (d.func == null) _ = try m.findGlobal(d.name);
        try m.worklist.append(m.gpa, di);
    }

    /// Pass 1 entry for a variable line: only the name is read now.
    fn registerGlobal(m: *ModuleState, line: []const u8, line_no: usize) Error!void {
        var c = Cursor{ .s = line, .line = line_no };
        try c.expect("@");
        const name = try m.globalName(&c);
        if (name.len == 0) return fail(line_no, line, "global without a name", error.Syntax);
        try m.pending_globals.append(m.gpa, .{ .name = name, .line = line, .line_no = line_no });
    }

    // @name = [linkage] [visibility] [unnamed_addr] [addrspace(N)] constant|global TYPE INIT [, align N]
    fn defineGlobal(m: *ModuleState, line: []const u8, line_no: usize) Error!void {
        var c = Cursor{ .s = line, .line = line_no };
        try c.expect("@");
        const name = try m.globalName(&c);
        try c.expect("=");
        var linkage: Builder.Linkage = .external;
        var addr_space: Builder.AddrSpace = .default;
        var is_const = false;
        while (true) {
            c.skipWs();
            const w = c.peekWord();
            if (std.mem.eql(u8, w, "constant")) {
                _ = c.word();
                is_const = true;
                break;
            } else if (std.mem.eql(u8, w, "global")) {
                _ = c.word();
                break;
            } else if (std.mem.eql(u8, w, "addrspace")) {
                _ = c.word();
                try c.expect("(");
                addr_space = mapAddrSpace(try c.int(u24));
                try c.expect(")");
            } else if (std.meta.stringToEnum(Builder.Linkage, w)) |l| {
                _ = c.word();
                linkage = l;
            } else if (w.len > 0 and isIgnorableGlobalWord(w)) {
                _ = c.word();
            } else return fail(line_no, line, "unexpected token in global definition", error.Syntax);
        }
        // Zig emits comptime tables, sampler words and strings as addrspace-0
        // constants; Metal links constant data only from address space 2.
        if (is_const and addr_space == .default) addr_space = constant_space;
        // A `var` at module scope (`@name = private global float ...`, or a
        // nvptx addrspace(5) local) assembles and loads, then pipeline
        // creation fails with "Undefined symbols: _name": Metal has no
        // writable storage at program scope except device/threadgroup memory.
        if (!is_const and addr_space == .default)
            return fail(line_no, line, "module-level mutable globals have no Metal address space (pipeline creation fails with 'Undefined symbols'); use a local, a device buffer or threadgroup memory (`var x: T addrspace(.shared)`)", error.Unsupported);
        const ty = try m.parseType(&c);
        const init = try m.parseConst(&c, ty);
        const variable = try m.b.addVariable(try m.b.strtabString(name), ty, addr_space);
        if (is_const) variable.setMutability(.constant, m.b);
        try variable.setInitializer(init, m.b);
        variable.ptr(m.b).global.setLinkage(linkage, m.b);
        c.skipWs();
        while (c.eat(",")) {
            c.skipWs();
            if (c.eatWord("align")) {
                variable.setAlignment(Builder.Alignment.fromByteUnits(try c.int(u64)), m.b);
            } else break; // section, comdat, metadata: irrelevant here
        }
        try m.globals.append(m.gpa, .{ .name = name, .index = variable.ptr(m.b).global, .fn_ty = null });
        if (std.mem.indexOf(u8, line, "[2 x i64]") != null) try m.samplers.append(m.gpa, variable.ptr(m.b).global);
    }

    // declare [cc] [ret attrs] RET @name(TY [attrs], ...) [attrs]
    fn declare(m: *ModuleState, line: []const u8, line_no: usize) Error!void {
        var c = Cursor{ .s = line, .line = line_no };
        try c.expectWord("declare");
        var call_conv: Builder.CallConv = .ccc;
        while (true) {
            const w = c.peekWord();
            if (w.len == 0) break;
            if (eatCallConv(&c)) |cc| {
                call_conv = cc;
                continue;
            }
            if (isIgnorableGlobalWord(w)) {
                _ = c.word();
                continue;
            }
            if (skipAttrs(&c)) continue;
            break;
        }
        const ret = try m.parseType(&c);
        try c.expect("@");
        const name = try m.globalName(&c);
        try c.expect("(");
        var params: std.ArrayList(Type) = .empty;
        defer params.deinit(m.gpa);
        c.skipWs();
        if (!c.eat(")")) {
            while (true) {
                if (c.eat("...")) return fail(line_no, line, "variadic declarations are not supported", error.Unsupported);
                try params.append(m.gpa, try m.parseType(&c));
                _ = skipAttrs(&c);
                c.skipWs();
                if (c.eat(",")) continue;
                try c.expect(")");
                break;
            }
        }
        // A texture intrinsic keeps Apple's signature rather than the one Zig
        // can spell (DESIGN.md D5-12c): an `extern struct` may not hold a
        // vector, so Zig declares the colour alone instead of
        // `{ <colour>, i8 }`, and Zig rejects `addrspace(.constant)` on nvptx,
        // so its sampler pointer arrives in the generic space. The declaration
        // is authoritative from here on; `lowerCall` adapts the call sites.
        var ret_ty = ret;
        var fn_ty = try m.b.fnType(ret, params.items, .normal);
        if (intrinsics.textureIntrinsic(name)) |ti| {
            if (ti.params.len != params.items.len) {
                return fail(line_no, line, "texture intrinsic declared with a different number of parameters than Apple's signature", error.Unsupported);
            }
            fn_ty = try m.textureFnType(ti, line_no);
            ret_ty = fn_ty.functionReturn(m.b);
        }
        try m.defines.append(m.gpa, .{
            .name = name,
            .ret_ty = ret_ty,
            .fn_ty = fn_ty,
            .linkage = .external,
            .call_conv = call_conv,
            .param_names = try m.gpa.alloc([]const u8, 0),
            .body = null,
            .header_line = line_no,
        });
    }

    // define [linkage] [visibility] [cc] [ret attrs] RET @name(TY [attrs] %p, ...) [attrs] {
    fn registerDefine(m: *ModuleState, line: []const u8, line_no: usize, body: []const []const u8) Error!void {
        const gpa = m.gpa;
        var c = Cursor{ .s = line, .line = line_no };
        try c.expectWord("define");
        var linkage: Builder.Linkage = .external;
        var call_conv: Builder.CallConv = .ccc;
        while (true) {
            const w = c.peekWord();
            if (w.len == 0) break;
            if (std.meta.stringToEnum(Builder.Linkage, w)) |l| {
                _ = c.word();
                linkage = l;
                continue;
            }
            if (eatCallConv(&c)) |cc| {
                call_conv = cc;
                continue;
            }
            if (isIgnorableGlobalWord(w)) {
                _ = c.word();
                continue;
            }
            if (skipAttrs(&c)) continue;
            break;
        }
        const ret_ty = try m.parseType(&c);
        try c.expect("@");
        const name = try m.globalName(&c);
        try c.expect("(");
        var param_types: std.ArrayList(Type) = .empty;
        defer param_types.deinit(gpa);
        var param_names: std.ArrayList([]const u8) = .empty;
        errdefer param_names.deinit(gpa);
        c.skipWs();
        if (!c.eat(")")) {
            while (true) {
                if (c.eat("...")) return fail(line_no, line, "variadic functions are not supported", error.Unsupported);
                try param_types.append(gpa, try m.parseType(&c));
                _ = skipAttrs(&c);
                try c.expect("%");
                try param_names.append(gpa, try m.quotedOrPlainName(&c));
                c.skipWs();
                if (c.eat(",")) continue;
                try c.expect(")");
                break;
            }
        }
        // Whatever follows the parameter list (unnamed_addr, #N, section)
        // is irrelevant to the bitcode we emit.
        const fn_ty = try m.b.fnType(ret_ty, param_types.items, .normal);
        try m.defines.append(gpa, .{
            .name = name,
            .ret_ty = ret_ty,
            .fn_ty = fn_ty,
            .linkage = linkage,
            .call_conv = call_conv,
            .param_names = try param_names.toOwnedSlice(gpa),
            .body = body,
            .header_line = line_no,
        });
    }

    fn lowerDefine(m: *ModuleState, di: usize) Error!void {
        const gpa = m.gpa;
        const d = m.defines.items[di];
        const body = d.body.?;
        var f = FunctionState{
            .m = m,
            .ret_ty = d.ret_ty,
            .wip = try WipFunction.init(m.b, .{ .function = d.func.?, .strip = true }),
            .line_level = if (di < m.funcs.len) m.funcs[di].line_level else &.{},
        };
        defer f.deinit();
        for (d.param_names, 0..) |pname, i| {
            try f.locals.append(gpa, .{ .name = pname, .value = f.wip.arg(@intCast(i)) });
        }
        try f.createBlocks(body, d.param_names);
        try f.lowerBody(body, d.header_line);
        try f.resolvePhis();
        try f.wip.finish();
    }

    /// Apple's signature for a texture intrinsic, parsed from the IR text in
    /// `intrinsics.texture_intrinsics` into Builder types (`{ <4 x float>, i8 }`
    /// returns, `ptr addrspace(1)` textures, `ptr addrspace(2)` samplers).
    fn textureFnType(m: *ModuleState, ti: *const intrinsics.TextureIntrinsic, line_no: usize) Error!Type {
        var rc = Cursor{ .s = ti.ret, .line = line_no };
        const ret = try m.parseType(&rc);
        var params: std.ArrayList(Type) = .empty;
        defer params.deinit(m.gpa);
        for (ti.params) |p| {
            var pc = Cursor{ .s = p, .line = line_no };
            try params.append(m.gpa, try m.parseType(&pc));
        }
        return m.b.fnType(ret, params.items, .normal);
    }

    /// `llvm.memcpy.p0.p0.i64` called with a relocated constant source must
    /// become `llvm.memcpy.p0.p2.i64`: the name and the declaration both
    /// carry the operand address spaces.
    fn memIntrinsicFor(m: *ModuleState, name: []const u8, fn_ty: Type, args: []const Value, wip: *WipFunction, c: *Cursor) Error!Global {
        const b = m.b;
        const gpa = m.gpa;
        const params = fn_ty.functionParameters(b);
        const new_params = try gpa.alloc(Type, params.len);
        defer gpa.free(new_params);
        var spaces: [4]u32 = undefined;
        var n_ptr: usize = 0;
        for (params, args, 0..) |p, a, i| {
            const at = a.typeOfWip(wip);
            if (p.isPointer(b) and at.isPointer(b)) {
                if (n_ptr == spaces.len) return fail(c.line, c.s, "too many pointer operands for a memory intrinsic", error.Unsupported);
                new_params[i] = at;
                spaces[n_ptr] = @backingInt(at.pointerAddrSpace(b));
                n_ptr += 1;
            } else if (p == at) {
                new_params[i] = p;
            } else return fail(c.line, c.s, "argument type does not match the intrinsic declaration", error.Unsupported);
        }
        const new_name = try intrinsics.memIntrinsicName(gpa, name, spaces[0..n_ptr]);
        defer gpa.free(new_name);
        const new_ty = try b.fnType(fn_ty.functionReturn(b), new_params, .normal);
        return m.intrinsicGlobal(new_name, new_ty);
    }

    /// The function `name` with type `fn_ty`, declared on first use. This
    /// is how renamed intrinsics (`air.fast_sin.f32`, `llvm.maxnum.f32`,
    /// `air.any.v4i1`, re-suffixed memcpy) enter the module: the text may
    /// declare them too, but the Builder sees only one declaration.
    fn intrinsicGlobal(m: *ModuleState, name: []const u8, fn_ty: Type) Error!Global {
        if (try m.findGlobal(name)) |g| {
            if (g.fn_ty != fn_ty) return error.Unsupported;
            return g;
        }
        const owned = try m.gpa.dupe(u8, name);
        try m.owned_names.append(m.gpa, owned);
        const func = try m.b.addFunction(fn_ty, try m.b.strtabString(owned), .default);
        const g = Global{ .name = owned, .index = func.ptr(m.b).global, .fn_ty = fn_ty };
        try m.globals.append(m.gpa, g);
        return g;
    }

    /// `intrinsics.lowerReduce` asks for its helper declarations through
    /// this adapter.
    const Declarer = struct {
        m: *ModuleState,
        pub fn declare(d: *Declarer, name: []const u8, fn_ty: Type) intrinsics.Error!Value {
            const g = d.m.intrinsicGlobal(name, fn_ty) catch |err| switch (err) {
                error.OutOfMemory => return error.OutOfMemory,
                else => return error.Unsupported,
            };
            return g.index.toValue();
        }
    };

    // ── types ───────────────────────────────────────────────────────────

    fn parseType(m: *ModuleState, c: *Cursor) Error!Type {
        c.skipWs();
        if (c.eat("<{")) {
            const fields = try m.parseTypeList(c, "}>");
            defer m.gpa.free(fields);
            return m.b.structType(.@"packed", fields);
        }
        if (c.eat("<")) {
            const len = try c.int(u32);
            try c.expectWord("x");
            const child = try m.parseType(c);
            try c.expect(">");
            return m.b.vectorType(.normal, len, child);
        }
        if (c.eat("[")) {
            const len = try c.int(u64);
            try c.expectWord("x");
            const child = try m.parseType(c);
            try c.expect("]");
            return m.b.arrayType(len, child);
        }
        if (c.eat("{")) {
            const fields = try m.parseTypeList(c, "}");
            defer m.gpa.free(fields);
            return m.b.structType(.normal, fields);
        }
        if (c.eat("%")) {
            const name = try m.quotedOrPlainName(c);
            for (m.named_types.items) |*nt| {
                if (std.mem.eql(u8, nt.name, name)) {
                    if (nt.resolved) |t| return t;
                    var bc = Cursor{ .s = nt.body, .line = c.line };
                    const t = try m.parseType(&bc);
                    nt.resolved = t;
                    return t;
                }
            }
            return fail(c.line, c.s, "unknown named type", error.UndefinedType);
        }
        const w = c.word();
        if (std.mem.eql(u8, w, "void")) return .void;
        if (std.mem.eql(u8, w, "float")) return .float;
        if (std.mem.eql(u8, w, "half")) return .half;
        if (std.mem.eql(u8, w, "double")) return .double;
        if (std.mem.eql(u8, w, "ptr")) {
            if (c.eatWord("addrspace")) {
                try c.expect("(");
                const n = try c.int(u24);
                try c.expect(")");
                return m.b.ptrType(mapAddrSpace(n));
            }
            return m.b.ptrType(.default);
        }
        if (w.len > 1 and w[0] == 'i') {
            if (std.fmt.parseInt(u24, w[1..], 10)) |bits| return m.b.intType(bits) else |_| {}
        }
        return fail(c.line, c.s, "unknown type", error.Syntax);
    }

    fn parseTypeList(m: *ModuleState, c: *Cursor, close: []const u8) Error![]Type {
        var list: std.ArrayList(Type) = .empty;
        errdefer list.deinit(m.gpa);
        c.skipWs();
        if (c.eat(close)) return list.toOwnedSlice(m.gpa);
        while (true) {
            try list.append(m.gpa, try m.parseType(c));
            c.skipWs();
            if (c.eat(",")) continue;
            try c.expect(close);
            break;
        }
        return list.toOwnedSlice(m.gpa);
    }

    // ── constants ───────────────────────────────────────────────────────

    fn parseConst(m: *ModuleState, c: *Cursor, ty: Type) Error!Constant {
        const b = m.b;
        c.skipWs();
        if (c.eat("<{")) {
            const vals = try m.parseConstList(c, "}>");
            defer m.gpa.free(vals);
            try m.checkAggregate(c, ty, vals, .@"struct");
            return b.structConst(ty, vals);
        }
        if (c.eat("{")) {
            const vals = try m.parseConstList(c, "}");
            defer m.gpa.free(vals);
            try m.checkAggregate(c, ty, vals, .@"struct");
            return b.structConst(ty, vals);
        }
        if (c.eat("<")) {
            const vals = try m.parseConstList(c, ">");
            defer m.gpa.free(vals);
            if (!ty.isVector(b) or ty.vectorLen(b) != vals.len) return fail(c.line, c.s, "vector literal does not match its type", error.Syntax);
            for (vals) |val| try m.checkElement(c, ty.childType(b), val.typeOf(b));
            return b.vectorConst(ty, vals);
        }
        if (c.eat("[")) {
            const vals = try m.parseConstList(c, "]");
            defer m.gpa.free(vals);
            try m.checkAggregate(c, ty, vals, .array);
            return b.arrayConst(ty, vals);
        }
        if (c.eat("@")) {
            const name = try m.globalName(c);
            const g = (try m.findGlobal(name)) orelse return fail(c.line, c.s, "unknown global", error.UndefinedGlobal);
            return g.index.toConst();
        }
        if (c.i + 1 < c.s.len and c.s[c.i] == 'c' and c.s[c.i + 1] == '"') return m.parseCString(c);
        // `splat (T c)`: LLVM >= 19 spelling of a uniform vector.
        if (c.eatWord("splat")) {
            try c.expect("(");
            const ety = try m.parseType(c);
            const val = try m.parseConst(c, ety);
            try c.expect(")");
            if (!ty.isVector(b)) return fail(c.line, c.s, "splat of a non-vector type", error.Syntax);
            return b.splatConst(ty, val);
        }
        // `getelementptr [inbounds] [nuw|nusw] (T, ptr BASE, IDX...)`
        if (c.eatWord("getelementptr")) {
            var inbounds = false;
            while (true) {
                if (c.eatWord("inbounds")) inbounds = true else if (c.eatWord("nuw") or c.eatWord("nusw")) {} else break;
            }
            try c.expect("(");
            const src_ty = try m.parseType(c);
            try c.expect(",");
            const list = try m.parseConstList(c, ")");
            defer m.gpa.free(list);
            if (list.len == 0) return fail(c.line, c.s, "getelementptr without a base", error.Syntax);
            return if (inbounds)
                b.gepConst(.inbounds, src_ty, list[0], null, list[1..])
            else
                b.gepConst(.normal, src_ty, list[0], null, list[1..]);
        }
        const w = c.numberOrWord();
        if (std.mem.eql(u8, w, "zeroinitializer")) return b.zeroInitConst(ty);
        if (std.mem.eql(u8, w, "undef")) return b.undefConst(ty);
        if (std.mem.eql(u8, w, "poison")) return b.poisonConst(ty);
        if (std.mem.eql(u8, w, "null")) return b.nullConst(ty);
        if (std.mem.eql(u8, w, "true")) return b.intConst(ty, 1);
        if (std.mem.eql(u8, w, "false")) return b.intConst(ty, 0);
        if (w.len == 0) return fail(c.line, c.s, "expected a constant", error.Syntax);

        if (ty == .float or ty == .half or ty == .double) {
            const value = try parseFloatLiteral(w, c);
            return switch (ty) {
                .float => b.floatConst(@floatCast(value)),
                .half => b.halfConst(@floatCast(value)),
                else => b.doubleConst(value),
            };
        }
        if (std.fmt.parseInt(i64, w, 10)) |v| return b.intConst(ty, v) else |_| {}
        if (std.fmt.parseInt(u64, w, 10)) |v| return b.intConst(ty, v) else |_| {}
        return fail(c.line, c.s, "unsupported constant", error.Unsupported);
    }

    /// The Builder asserts (rather than reports) that every element of an
    /// aggregate constant has exactly the aggregate's element type, so the
    /// check happens here. After constant relocation a `ptr @g` element is
    /// really `ptr addrspace(2)`, which no longer matches a textual
    /// `[N x ptr]` or `{ ptr, i64 }`. Rebuilding the aggregate type would
    /// not help: the shader then does `load ptr` from the table and
    /// dereferences a thread-space pointer into constant memory. Refused
    /// with a hint instead (DESIGN.md D1).
    fn checkAggregate(m: *ModuleState, c: *Cursor, ty: Type, vals: []const Constant, kind: enum { @"struct", array }) Error!void {
        const b = m.b;
        const shape_ok = switch (kind) {
            .@"struct" => ty.isStruct(b),
            .array => switch (ty.tag(b)) {
                .array, .small_array => true,
                else => false,
            },
        };
        if (!shape_ok) return fail(c.line, c.s, "aggregate literal does not match its type", error.Syntax);
        if (ty.aggregateLen(b) != vals.len) return fail(c.line, c.s, "aggregate literal has the wrong number of elements", error.Syntax);
        for (vals, 0..) |val, idx| {
            const got = val.typeOf(b);
            const want = switch (kind) {
                .@"struct" => ty.structFields(b)[idx],
                .array => ty.childType(b),
            };
            try m.checkElement(c, want, got);
        }
    }

    /// One element of an aggregate/vector, literal or built by
    /// `insertvalue`/`insertelement`, against the slot it goes into. Two
    /// different pointer types can only mean a relocated constant's address
    /// (`ptr addrspace(2)`) in a textual `ptr` slot; anything else is a
    /// malformed module.
    fn checkElement(m: *ModuleState, c: *Cursor, want: Type, got: Type) Error!void {
        const b = m.b;
        if (got == want) return;
        if (got.isPointer(b) and want.isPointer(b))
            return fail(c.line, c.s, "value holds a pointer to a relocated addrspace-0 constant, which Metal cannot load through a thread-space ptr; index the tables directly instead of storing or returning their addresses (slices of comptime tables included)", error.Unsupported);
        return fail(c.line, c.s, "element type does not match the aggregate slot it is inserted into", error.Syntax);
    }

    /// `Type.childTypeAt` without its `unreachable`s: the element type the
    /// `insertvalue`/`extractvalue` indices address, or null when an index
    /// runs off the aggregate.
    fn aggregateChildType(m: *ModuleState, ty: Type, indices: []const u32) ?Type {
        const b = m.b;
        var cur = ty;
        for (indices) |idx| {
            if (cur.isStruct(b)) {
                const fields = cur.structFields(b);
                if (idx >= fields.len) return null;
                cur = fields[idx];
            } else switch (cur.tag(b)) {
                .array, .small_array => {
                    if (idx >= cur.aggregateLen(b)) return null;
                    cur = cur.childType(b);
                },
                else => return null,
            }
        }
        return cur;
    }

    /// `c"..."` with `\XX` hex escapes -> `[N x i8]` string constant.
    fn parseCString(m: *ModuleState, c: *Cursor) Error!Constant {
        try c.expect("c\"");
        var bytes: std.ArrayList(u8) = .empty;
        defer bytes.deinit(m.gpa);
        while (c.i < c.s.len and c.s[c.i] != '"') {
            if (c.s[c.i] == '\\') {
                if (c.i + 2 >= c.s.len) return fail(c.line, c.s, "truncated string escape", error.Syntax);
                const v = std.fmt.parseInt(u8, c.s[c.i + 1 .. c.i + 3], 16) catch return fail(c.line, c.s, "bad string escape", error.Syntax);
                try bytes.append(m.gpa, v);
                c.i += 3;
            } else {
                try bytes.append(m.gpa, c.s[c.i]);
                c.i += 1;
            }
        }
        try c.expect("\"");
        return m.b.stringConst(try m.b.string(bytes.items));
    }

    /// `TYPE CONST, TYPE CONST, ...` up to `close`.
    fn parseConstList(m: *ModuleState, c: *Cursor, close: []const u8) Error![]Constant {
        var list: std.ArrayList(Constant) = .empty;
        errdefer list.deinit(m.gpa);
        while (true) {
            c.skipWs();
            if (c.eat(close)) break;
            if (c.i >= c.s.len) return fail(c.line, c.s, "unterminated constant list", error.Syntax);
            const ety = try m.parseType(c);
            try list.append(m.gpa, try m.parseConst(c, ety));
            c.skipWs();
            _ = c.eat(",");
        }
        return list.toOwnedSlice(m.gpa);
    }
};

/// LLVM prints `1.000000e+00`, `-0.5`, or hex IEEE-754 doubles `0x3FF0...`
/// (and `0xH3C00` for half).
fn parseFloatLiteral(w: []const u8, c: *Cursor) Error!f64 {
    if (std.mem.startsWith(u8, w, "0xH")) {
        const bits = std.fmt.parseInt(u16, w[3..], 16) catch return fail(c.line, c.s, "bad half literal", error.Syntax);
        return @floatCast(@as(f16, @bitCast(bits)));
    }
    if (std.mem.startsWith(u8, w, "0x")) {
        if (w.len > 2 and std.ascii.isAlphabetic(w[2]) and !std.ascii.isHex(w[2]))
            return fail(c.line, c.s, "unsupported float literal kind", error.Unsupported);
        const bits = std.fmt.parseInt(u64, w[2..], 16) catch return fail(c.line, c.s, "bad hex float literal", error.Syntax);
        return @bitCast(bits);
    }
    return std.fmt.parseFloat(f64, w) catch fail(c.line, c.s, "bad float literal", error.Syntax);
}

fn allDigits(s: []const u8) bool {
    if (s.len == 0) return false;
    for (s) |ch| if (!std.ascii.isDigit(ch)) return false;
    return true;
}

fn isIgnorableGlobalWord(w: []const u8) bool {
    const words = [_][]const u8{
        "unnamed_addr", "local_unnamed_addr", "dso_local",    "dso_preemptable",        "hidden",
        "protected",    "default",            "thread_local", "externally_initialized",
    };
    for (words) |x| if (std.mem.eql(u8, x, w)) return true;
    return false;
}

/// Skips parameter / return attributes wherever they may appear: after a
/// type in declarations, define headers and call arguments, and before the
/// return type of a define/declare/call. Bare words, `align N`, and the
/// parenthesised forms (`range(i32 0, 5)`, `captures(none)`, `sret(%T)`,
/// ...) are all consumed. Returns whether anything was skipped.
fn skipAttrs(c: *Cursor) bool {
    const bare = [_][]const u8{
        "noundef",    "nonnull",      "noalias",        "readonly",       "writeonly", "readnone",
        "nocapture",  "signext",      "zeroext",        "inreg",          "nofree",    "nest",
        "returned",   "immarg",       "nosync",         "cold",           "noinline",  "nounwind",
        "willreturn", "mustprogress", "dead_on_unwind", "dead_on_return", "writable",  "swiftself",
        "swifterror", "allocalign",   "allocptr",       "nonlazybind",
    };
    const parenthesised = [_][]const u8{
        "range",           "captures",                "initializes", "sret",        "byval",        "byref",
        "dereferenceable", "dereferenceable_or_null", "nofpclass",   "elementtype", "preallocated", "inalloca",
    };
    var any = false;
    while (true) {
        c.skipWs();
        const w = c.peekWord();
        if (w.len == 0) return any;
        if (std.mem.eql(u8, w, "align")) {
            _ = c.word();
            c.skipWs();
            if (c.eat("(")) c.skipBalancedParens() else _ = c.numberOrWord();
            any = true;
            continue;
        }
        var matched = false;
        for (bare) |x| if (std.mem.eql(u8, x, w)) {
            matched = true;
            break;
        };
        if (matched) {
            _ = c.word();
            any = true;
            continue;
        }
        for (parenthesised) |x| if (std.mem.eql(u8, x, w)) {
            matched = true;
            break;
        };
        if (!matched) return any;
        _ = c.word();
        c.skipWs();
        if (c.eat("(")) c.skipBalancedParens();
        any = true;
    }
}

/// Calling-convention words that may precede the return type of a
/// define/declare/call. Kernel conventions carry no meaning in AIR (entry
/// points are plain functions), so they map to the C convention.
fn eatCallConv(c: *Cursor) ?Builder.CallConv {
    const table = [_]struct { name: []const u8, cc: Builder.CallConv }{
        .{ .name = "ccc", .cc = .ccc },         .{ .name = "fastcc", .cc = .fastcc },
        .{ .name = "coldcc", .cc = .coldcc },   .{ .name = "tailcc", .cc = .tailcc },
        .{ .name = "swiftcc", .cc = .swiftcc }, .{ .name = "ptx_kernel", .cc = .ccc },
        .{ .name = "ptx_device", .cc = .ccc },  .{ .name = "amdgpu_kernel", .cc = .ccc },
    };
    for (table) |t| if (c.eatWord(t.name)) return t.cc;
    return null;
}

// ── Function state ──────────────────────────────────────────────────────────

const Local = struct { name: []const u8, value: Value };
/// `WipFunction.Block.Index` is private in the Builder; recover it from the
/// return type of `WipFunction.block`.
const BlockIndex = @typeInfo(@typeInfo(@TypeOf(WipFunction.block)).@"fn".return_type.?).error_union.payload;
const BlockRef = struct { name: []const u8, index: BlockIndex, incoming: u32 };
const PendingPhi = struct {
    phi: WipFunction.WipPhi,
    ty: Type,
    /// Each incoming is `(value text, block name)`.
    incoming: []const [2][]const u8,
    /// Predecessor count of the block holding the phi; the Builder requires
    /// exactly one incoming per predecessor edge.
    block_incoming: u32,
    line_no: usize,
    line: []const u8,
};
const SwitchCase = struct { val: Constant, dest: BlockIndex };

const FunctionState = struct {
    m: *ModuleState,
    ret_ty: Type,
    wip: WipFunction,
    locals: std.ArrayList(Local) = .empty,
    blocks: std.ArrayList(BlockRef) = .empty,
    phis: std.ArrayList(PendingPhi) = .empty,
    /// Index into `blocks` of the block being lowered.
    current: usize = 0,
    entry_name: ?[]u8 = null,
    /// Per body line, from the uniformity analysis: which threads reach
    /// the line's block together (empty when no analysis ran).
    line_level: []const divergence.Level = &.{},
    /// Body line index of the instruction being lowered.
    line_idx: usize = 0,

    fn deinit(f: *FunctionState) void {
        const gpa = f.m.gpa;
        for (f.phis.items) |p| gpa.free(p.incoming);
        f.phis.deinit(gpa);
        f.blocks.deinit(gpa);
        f.locals.deinit(gpa);
        if (f.entry_name) |n| gpa.free(n);
        f.wip.deinit();
    }

    fn lookup(f: *FunctionState, name: []const u8, c: *Cursor) Error!Value {
        for (f.locals.items) |l| if (std.mem.eql(u8, l.name, name)) return l.value;
        return fail(c.line, c.s, "use of undefined value", error.UndefinedValue);
    }

    fn define(f: *FunctionState, name: []const u8, value: Value) Error!void {
        try f.locals.append(f.m.gpa, .{ .name = name, .value = value });
    }

    fn block(f: *FunctionState, name: []const u8, c: *Cursor) Error!BlockIndex {
        for (f.blocks.items) |blk| if (std.mem.eql(u8, blk.name, name)) return blk.index;
        return fail(c.line, c.s, "branch to undefined block", error.UndefinedBlock);
    }

    /// Labels end with ':' and start a line. Predecessor counts come from a
    /// scan of all branch targets (`br`, `switch` cases included), which the
    /// Builder needs for phi storage. (Control-flow facts for the
    /// convergent-call check come from divergence.zig, not from here.)
    fn createBlocks(f: *FunctionState, body: []const []const u8, param_names: []const []const u8) Error!void {
        const gpa = f.m.gpa;
        var names: std.ArrayList([]const u8) = .empty;
        defer names.deinit(gpa);
        var counts: std.ArrayList(u32) = .empty;
        defer counts.deinit(gpa);

        // The entry block is implicit unless the body opens with a label.
        // LLVM numbers it right after the parameters, and phis name it that
        // way (`[ %x, %__3 ]` for a function with `%__0`..`%__2`).
        var first_is_label = false;
        for (body) |raw| {
            const line = std.mem.trim(u8, raw, " \t\r");
            if (line.len == 0 or line[0] == ';') continue;
            first_is_label = labelName(line) != null;
            break;
        }
        if (!first_is_label) {
            var next: usize = 0;
            for (param_names) |p| {
                if (std.mem.startsWith(u8, p, numbered_prefix) and allDigits(p[numbered_prefix.len..])) {
                    const n = std.fmt.parseInt(usize, p[numbered_prefix.len..], 10) catch continue;
                    if (n + 1 > next) next = n + 1;
                }
            }
            f.entry_name = try std.fmt.allocPrint(gpa, "{s}{d}", .{ numbered_prefix, next });
            try names.append(gpa, f.entry_name.?);
            try counts.append(gpa, 0);
        }
        for (body) |raw| {
            const line = std.mem.trim(u8, raw, " \t\r");
            if (labelName(line)) |name| {
                try names.append(gpa, name);
                try counts.append(gpa, 0);
            }
        }
        // Every `label %x` in a block's lines is a branch target (phis
        // spell their blocks `[ %v, %bb ]`, so only terminators match).
        for (body) |raw| {
            const line = std.mem.trim(u8, raw, " \t\r");
            if (line.len == 0 or line[0] == ';' or labelName(line) != null) continue;
            var pos: usize = 0;
            while (findPos(line, pos, "label %")) |p| {
                var c = Cursor{ .s = line, .line = 0 };
                c.i = p + "label %".len;
                const target = c.word();
                pos = c.i;
                for (names.items, 0..) |n, i| if (std.mem.eql(u8, n, target)) {
                    counts.items[i] += 1;
                };
            }
        }
        for (names.items, counts.items) |name, count| {
            const index = try f.wip.block(count, name);
            try f.blocks.append(gpa, .{ .name = name, .index = index, .incoming = count });
        }
        f.current = 0;
        f.wip.cursor = .{ .block = f.blocks.items[0].index };
    }

    fn lowerBody(f: *FunctionState, body: []const []const u8, header_line: usize) Error!void {
        const gpa = f.m.gpa;
        var idx: usize = 0;
        while (idx < body.len) : (idx += 1) {
            const line_no = header_line + idx + 1;
            const line = std.mem.trim(u8, body[idx], " \t\r");
            if (line.len == 0 or line[0] == ';') continue;
            if (labelName(line)) |name| {
                var c = Cursor{ .s = line, .line = line_no };
                for (f.blocks.items, 0..) |blk, i| if (std.mem.eql(u8, blk.name, name)) {
                    f.current = i;
                };
                f.wip.cursor = .{ .block = try f.block(name, &c) };
                continue;
            }
            f.line_idx = idx;
            if (std.mem.startsWith(u8, line, "switch ")) {
                // The case list spans following lines up to a lone `]`.
                var joined: std.ArrayList(u8) = .empty;
                defer joined.deinit(gpa);
                try joined.appendSlice(gpa, line);
                while (idx + 1 < body.len) {
                    idx += 1;
                    const l = std.mem.trim(u8, body[idx], " \t\r");
                    try joined.append(gpa, ' ');
                    try joined.appendSlice(gpa, l);
                    if (std.mem.startsWith(u8, l, "]")) break;
                }
                var c = Cursor{ .s = joined.items, .line = line_no };
                try f.lowerInstruction(&c);
                continue;
            }
            var c = Cursor{ .s = line, .line = line_no };
            try f.lowerInstruction(&c);
        }
    }

    fn lowerInstruction(f: *FunctionState, c: *Cursor) Error!void {
        const m = f.m;
        const b = m.b;
        const wip = &f.wip;

        var result: ?[]const u8 = null;
        c.skipWs();
        if (c.eat("%")) {
            result = try m.quotedOrPlainName(c);
            try c.expect("=");
        }
        c.skipWs();
        const op = c.word();

        if (std.mem.eql(u8, op, "ret")) {
            c.skipWs();
            if (c.eatWord("void")) {
                _ = try wip.retVoid();
            } else {
                const val = try f.operand(c);
                const got = val.typeOfWip(wip);
                if (got != f.ret_ty) {
                    // A helper declared `ptr` handing back a relocated table
                    // (`ret ptr @tab`) is the D1 case; refuse with the hint.
                    if (got.isPointer(b) and f.ret_ty.isPointer(b)) try m.checkElement(c, f.ret_ty, got);
                    return fail(c.line, c.s, "returned value does not match the function's return type", error.Syntax);
                }
                _ = try wip.ret(val);
            }
            return;
        }
        if (std.mem.eql(u8, op, "br")) {
            c.skipWs();
            if (c.eatWord("label")) {
                try c.expect("%");
                _ = try wip.br(try f.block(c.word(), c));
                return;
            }
            const cond = try f.operand(c);
            try c.expect(",");
            try c.expectWord("label");
            try c.expect("%");
            const then = try f.block(c.word(), c);
            try c.expect(",");
            try c.expectWord("label");
            try c.expect("%");
            const els = try f.block(c.word(), c);
            _ = try wip.brCond(cond, then, els, .none);
            return;
        }
        if (std.mem.eql(u8, op, "switch")) return f.lowerSwitch(c);
        if (std.mem.eql(u8, op, "unreachable")) {
            _ = try wip.@"unreachable"();
            return;
        }
        if (std.mem.eql(u8, op, "phi")) {
            _ = skipFastMath(c);
            var ty = try m.parseType(c);
            var incoming: std.ArrayList([2][]const u8) = .empty;
            errdefer incoming.deinit(m.gpa);
            while (true) {
                c.skipWs();
                if (!c.eat("[")) break;
                c.skipWs();
                const val_start = c.i;
                // Value runs up to the top-level ','.
                var depth: usize = 0;
                while (c.i < c.s.len) : (c.i += 1) {
                    switch (c.s[c.i]) {
                        '<', '(', '[', '{' => depth += 1,
                        '>', ')', ']', '}' => depth -= 1,
                        ',' => if (depth == 0) break,
                        else => {},
                    }
                }
                const val_text = std.mem.trim(u8, c.s[val_start..c.i], " ");
                try c.expect(",");
                try c.expect("%");
                const blk = c.word();
                try c.expect("]");
                try incoming.append(m.gpa, .{ val_text, blk });
                c.skipWs();
                _ = c.eat(",");
            }
            // D1: named operands carry their real type, so a `phi ptr` over
            // relocated constants is really a phi of `ptr addrspace(2)`.
            if (ty.isPointer(b)) ty = try f.phiPointerType(ty, incoming.items, c.line);
            const phi = try wip.phi(ty, "");
            try f.phis.append(m.gpa, .{
                .phi = phi,
                .ty = ty,
                .incoming = try incoming.toOwnedSlice(m.gpa),
                .block_incoming = f.blocks.items[f.current].incoming,
                .line_no = c.line,
                .line = c.s,
            });
            try f.define(result orelse return fail(c.line, c.s, "phi without result", error.Syntax), phi.toValue());
            return;
        }
        if (std.mem.eql(u8, op, "select")) {
            const fast = skipFastMath(c);
            const cond = try f.operand(c);
            try c.expect(",");
            const lhs = try f.operand(c);
            try c.expect(",");
            const rhs = try f.operand(c);
            if (lhs.typeOfWip(wip) != rhs.typeOfWip(wip)) return fail(c.line, c.s, "select operands have different types", error.Syntax);
            try f.define(result orelse return fail(c.line, c.s, "select without result", error.Syntax), try wip.select(fast, cond, lhs, rhs, ""));
            return;
        }
        if (std.mem.eql(u8, op, "icmp")) {
            c.skipWs();
            _ = c.eatWord("samesign"); // LLVM >= 19 flag
            const pred = c.word();
            const cond = std.meta.stringToEnum(Builder.IntegerCondition, pred) orelse return fail(c.line, c.s, "bad icmp predicate", error.Syntax);
            const lhs = try f.operand(c);
            try c.expect(",");
            const rhs = try f.operandOfType(c, lhs.typeOfWip(wip));
            if (lhs.typeOfWip(wip) != rhs.typeOfWip(wip)) return fail(c.line, c.s, "icmp operands have different types", error.Syntax);
            try f.define(result orelse return fail(c.line, c.s, "icmp without result", error.Syntax), try wip.icmp(cond, lhs, rhs, ""));
            return;
        }
        if (std.mem.eql(u8, op, "fcmp")) {
            const fast = skipFastMath(c);
            c.skipWs();
            const pred = c.word();
            const cond = std.meta.stringToEnum(Builder.FloatCondition, pred) orelse return fail(c.line, c.s, "unsupported fcmp predicate", error.Unsupported);
            const lhs = try f.operand(c);
            try c.expect(",");
            const rhs = try f.operandOfType(c, lhs.typeOfWip(wip));
            if (lhs.typeOfWip(wip) != rhs.typeOfWip(wip)) return fail(c.line, c.s, "fcmp operands have different types", error.Syntax);
            try f.define(result orelse return fail(c.line, c.s, "fcmp without result", error.Syntax), try wip.fcmp(fast, cond, lhs, rhs, ""));
            return;
        }
        if (std.mem.eql(u8, op, "freeze")) {
            // The Builder has no freeze; Metal accepts the value unchanged.
            const val = try f.operand(c);
            try f.define(result orelse return fail(c.line, c.s, "freeze without result", error.Syntax), val);
            return;
        }
        if (binaryTag(op, c)) |tag| {
            const lhs = try f.operand(c);
            try c.expect(",");
            const rhs = try f.operandOfType(c, lhs.typeOfWip(wip));
            if (lhs.typeOfWip(wip) != rhs.typeOfWip(wip)) return fail(c.line, c.s, "arithmetic operands have different types", error.Syntax);
            try f.define(result orelse return fail(c.line, c.s, "arithmetic without result", error.Syntax), try wip.bin(tag, lhs, rhs, ""));
            return;
        }
        if (std.mem.eql(u8, op, "fneg")) {
            const fast = skipFastMath(c);
            const val = try f.operand(c);
            const ty = val.typeOfWip(wip);
            // fneg x == fsub -0.0, x for every non-NaN payload we care about.
            const neg_zero = try negativeZero(b, ty);
            const tag: Builder.Function.Instruction.Tag = if (fast == .fast) .@"fsub fast" else .fsub;
            try f.define(result orelse return fail(c.line, c.s, "fneg without result", error.Syntax), try wip.bin(tag, neg_zero, val, ""));
            return;
        }
        if (std.mem.eql(u8, op, "addrspacecast")) {
            return fail(c.line, c.s, "addrspacecast is not supported: its semantics on Metal are unverified; keep each pointer in one address space", error.Unsupported);
        }
        if (std.mem.eql(u8, op, "fence")) {
            return fail(c.line, c.s, "fence is not supported: a native fence crashes Metal's compiler; use gpu.atomicFence (air.atomic.fence)", error.Unsupported);
        }
        if (castTag(op, c)) |tag| {
            const val = try f.operand(c);
            try c.expectWord("to");
            const ty = try m.parseType(c);
            if (tag == .bitcast) {
                // Zig's `@bitCast` of a bool vector, and what LLVM folds
                // `@reduce(.Or)` into: crashes the GPU backend as a bitcast,
                // packs fine lane by lane (intrinsics.zig).
                const from = val.typeOfWip(wip);
                if (from.isVector(b) and from.scalarType(b) == .i1 and !ty.isVector(b)) {
                    const packed_bits = intrinsics.lowerBoolVectorBitcast(b, wip, val, ty) catch |err| switch (err) {
                        error.OutOfMemory => return error.OutOfMemory,
                        error.Unsupported => return fail(c.line, c.s, "bool-vector bitcast: unsupported shape (more than 32 lanes or a non-integer target)", error.Unsupported),
                    };
                    try f.define(result orelse return fail(c.line, c.s, "cast without result", error.Syntax), packed_bits);
                    return;
                }
                // The reverse form, `@as(@Vector(N, bool), @bitCast(uN))`,
                // crashes the same way (review-a2/h_bitcast_rev.ll).
                if (!from.isVector(b) and from.isInteger(b) and ty.isVector(b) and ty.childType(b) == .i1) {
                    const lanes = intrinsics.lowerIntToBoolVector(b, wip, val, ty) catch |err| switch (err) {
                        error.OutOfMemory => return error.OutOfMemory,
                        error.Unsupported => return fail(c.line, c.s, "integer to bool-vector bitcast: unsupported shape (more than 32 lanes)", error.Unsupported),
                    };
                    try f.define(result orelse return fail(c.line, c.s, "cast without result", error.Syntax), lanes);
                    return;
                }
            }
            try f.define(result orelse return fail(c.line, c.s, "cast without result", error.Syntax), try wip.cast(tag, val, ty, ""));
            return;
        }
        if (std.mem.eql(u8, op, "getelementptr")) {
            var kind: Builder.Constant.GetElementPtr.Kind = .normal;
            while (true) {
                if (c.eatWord("inbounds")) kind = .inbounds else if (c.eatWord("nuw") or c.eatWord("nusw")) {} else break;
            }
            const ty = try m.parseType(c);
            try c.expect(",");
            const base = try f.operand(c);
            if (!base.typeOfWip(wip).isPointer(b)) return fail(c.line, c.s, "getelementptr base is not a pointer", error.Syntax);
            var indices: std.ArrayList(Value) = .empty;
            defer indices.deinit(m.gpa);
            while (true) {
                c.skipWs();
                if (!c.eat(",")) break;
                try indices.append(m.gpa, try f.operand(c));
            }
            try f.define(result orelse return fail(c.line, c.s, "gep without result", error.Syntax), try wip.gep(kind, ty, base, indices.items, ""));
            return;
        }
        if (std.mem.eql(u8, op, "load")) {
            const atomic = c.eatWord("atomic");
            const kind: Builder.MemoryAccessKind = if (c.eatWord("volatile")) .@"volatile" else .normal;
            const ty = try m.parseType(c);
            try c.expect(",");
            const ptr = try f.operand(c);
            if (!ptr.typeOfWip(wip).isPointer(b)) return fail(c.line, c.s, "load from a non-pointer", error.Syntax);
            const name = result orelse return fail(c.line, c.s, "load without result", error.Syntax);
            if (atomic) {
                skipSyncScope(c);
                const ordering = try parseOrdering(c);
                const alignment = try trailingAlign(c);
                try f.define(name, try wip.loadAtomic(kind, ty, ptr, .system, ordering, alignment, ""));
            } else {
                const alignment = try trailingAlign(c);
                try f.define(name, try wip.load(kind, ty, ptr, alignment, ""));
            }
            return;
        }
        if (std.mem.eql(u8, op, "store")) {
            const atomic = c.eatWord("atomic");
            const kind: Builder.MemoryAccessKind = if (c.eatWord("volatile")) .@"volatile" else .normal;
            const val = try f.operand(c);
            try c.expect(",");
            const ptr = try f.operand(c);
            if (!ptr.typeOfWip(wip).isPointer(b)) return fail(c.line, c.s, "store to a non-pointer", error.Syntax);
            if (atomic) {
                skipSyncScope(c);
                const ordering = try parseOrdering(c);
                const alignment = try trailingAlign(c);
                _ = try wip.storeAtomic(kind, val, ptr, .system, ordering, alignment);
            } else {
                const alignment = try trailingAlign(c);
                _ = try wip.store(kind, val, ptr, alignment);
            }
            return;
        }
        if (std.mem.eql(u8, op, "atomicrmw")) {
            // atomicrmw [volatile] OP ptr %p, T %v [syncscope("...")] ORDERING[, align N]
            const kind: Builder.MemoryAccessKind = if (c.eatWord("volatile")) .@"volatile" else .normal;
            c.skipWs();
            const opname = c.word();
            const operation = std.meta.stringToEnum(Builder.Function.Instruction.AtomicRmw.Operation, opname) orelse
                return fail(c.line, c.s, "unsupported atomicrmw operation", error.Unsupported);
            if (operation == .none) return fail(c.line, c.s, "unsupported atomicrmw operation", error.Unsupported);
            const ptr = try f.operand(c);
            if (!ptr.typeOfWip(wip).isPointer(b)) return fail(c.line, c.s, "atomicrmw on a non-pointer", error.Syntax);
            try c.expect(",");
            const val = try f.operand(c);
            skipSyncScope(c);
            const ordering = try parseOrdering(c);
            const alignment = try trailingAlign(c);
            try f.define(result orelse return fail(c.line, c.s, "atomicrmw without result", error.Syntax), try wip.atomicrmw(kind, operation, ptr, val, .system, ordering, alignment, ""));
            return;
        }
        if (std.mem.eql(u8, op, "cmpxchg")) {
            // cmpxchg [weak] [volatile] ptr %p, T %cmp, T %new [syncscope] SUCCESS FAILURE[, align N]
            const weak = c.eatWord("weak");
            const kind: Builder.MemoryAccessKind = if (c.eatWord("volatile")) .@"volatile" else .normal;
            const ptr = try f.operand(c);
            if (!ptr.typeOfWip(wip).isPointer(b)) return fail(c.line, c.s, "cmpxchg on a non-pointer", error.Syntax);
            try c.expect(",");
            const cmp = try f.operand(c);
            try c.expect(",");
            const new = try f.operand(c);
            if (cmp.typeOfWip(wip) != new.typeOfWip(wip)) return fail(c.line, c.s, "cmpxchg operands have different types", error.Syntax);
            skipSyncScope(c);
            const success = try parseOrdering(c);
            const failure = try parseOrdering(c);
            const alignment = try trailingAlign(c);
            const cx_kind: Builder.Function.Instruction.CmpXchg.Kind = if (weak) .weak else .strong;
            try f.define(result orelse return fail(c.line, c.s, "cmpxchg without result", error.Syntax), try wip.cmpxchg(cx_kind, kind, ptr, cmp, new, .system, success, failure, alignment, ""));
            return;
        }
        if (std.mem.eql(u8, op, "alloca")) {
            const ty = try m.parseType(c);
            var len: Value = (try b.intConst(.i32, 1)).toValue();
            var alignment: Builder.Alignment = .default;
            var addr_space: Builder.AddrSpace = .default;
            while (true) {
                c.skipWs();
                if (!c.eat(",")) break;
                c.skipWs();
                if (c.eatWord("align")) {
                    alignment = Builder.Alignment.fromByteUnits(try c.int(u64));
                } else if (c.eatWord("addrspace")) {
                    try c.expect("(");
                    addr_space = mapAddrSpace(try c.int(u24));
                    try c.expect(")");
                } else {
                    len = try f.operand(c);
                }
            }
            try f.define(result orelse return fail(c.line, c.s, "alloca without result", error.Syntax), try wip.alloca(.normal, ty, len, alignment, addr_space, ""));
            return;
        }
        if (std.mem.eql(u8, op, "insertvalue")) {
            const agg = try f.operand(c);
            try c.expect(",");
            const elem = try f.operand(c);
            const indices = try f.indexList(c);
            defer m.gpa.free(indices);
            // The Builder asserts the slot type; a `{ ptr, i64 }` slice of
            // a relocated table is the case that reaches here (D1).
            const slot = m.aggregateChildType(agg.typeOfWip(wip), indices) orelse return fail(c.line, c.s, "insertvalue indices do not address an element of the aggregate", error.Syntax);
            try m.checkElement(c, slot, elem.typeOfWip(wip));
            try f.define(result orelse return fail(c.line, c.s, "insertvalue without result", error.Syntax), try wip.insertValue(agg, elem, indices, ""));
            return;
        }
        if (std.mem.eql(u8, op, "extractvalue")) {
            const agg = try f.operand(c);
            const indices = try f.indexList(c);
            defer m.gpa.free(indices);
            if (m.aggregateChildType(agg.typeOfWip(wip), indices) == null) return fail(c.line, c.s, "extractvalue indices do not address an element of the aggregate", error.Syntax);
            try f.define(result orelse return fail(c.line, c.s, "extractvalue without result", error.Syntax), try wip.extractValue(agg, indices, ""));
            return;
        }
        if (std.mem.eql(u8, op, "insertelement")) {
            const vec = try f.operand(c);
            try c.expect(",");
            const elem = try f.operand(c);
            try c.expect(",");
            const index = try f.operand(c);
            const vec_ty = vec.typeOfWip(wip);
            if (!vec_ty.isVector(b)) return fail(c.line, c.s, "insertelement into a non-vector value", error.Syntax);
            try m.checkElement(c, vec_ty.childType(b), elem.typeOfWip(wip));
            if (!index.typeOfWip(wip).isInteger(b)) return fail(c.line, c.s, "insertelement index is not an integer", error.Syntax);
            try f.define(result orelse return fail(c.line, c.s, "insertelement without result", error.Syntax), try wip.insertElement(vec, elem, index, ""));
            return;
        }
        if (std.mem.eql(u8, op, "extractelement")) {
            const vec = try f.operand(c);
            try c.expect(",");
            const index = try f.operand(c);
            try f.define(result orelse return fail(c.line, c.s, "extractelement without result", error.Syntax), try wip.extractElement(vec, index, ""));
            return;
        }
        if (std.mem.eql(u8, op, "shufflevector")) {
            const lhs = try f.operand(c);
            try c.expect(",");
            const rhs = try f.operand(c);
            try c.expect(",");
            const mask = try f.operand(c);
            try f.define(result orelse return fail(c.line, c.s, "shufflevector without result", error.Syntax), try wip.shuffleVector(lhs, rhs, mask, ""));
            return;
        }
        if (std.mem.eql(u8, op, "tail") or std.mem.eql(u8, op, "musttail") or std.mem.eql(u8, op, "notail") or std.mem.eql(u8, op, "call")) {
            return f.lowerCall(c, op, result);
        }
        return fail(c.line, c.s, "unsupported instruction", error.Unsupported);
    }

    /// `switch T %v, label %default [ T c, label %bb ... ]` (already joined
    /// into one line by `lowerBody`). Duplicate targets are ordinary edges.
    fn lowerSwitch(f: *FunctionState, c: *Cursor) Error!void {
        const m = f.m;
        const gpa = m.gpa;
        const val = try f.operand(c);
        try c.expect(",");
        try c.expectWord("label");
        try c.expect("%");
        const default = try f.block(c.word(), c);
        try c.expect("[");
        var cases: std.ArrayList(SwitchCase) = .empty;
        defer cases.deinit(gpa);
        while (true) {
            c.skipWs();
            if (c.eat("]")) break;
            if (c.i >= c.s.len) return fail(c.line, c.s, "unterminated switch", error.Syntax);
            const cty = try m.parseType(c);
            const cval = try m.parseConst(c, cty);
            if (cval.typeOf(m.b) != val.typeOfWip(&f.wip)) return fail(c.line, c.s, "switch case type does not match the value", error.Syntax);
            try c.expect(",");
            try c.expectWord("label");
            try c.expect("%");
            const dest = try f.block(c.word(), c);
            try cases.append(gpa, .{ .val = cval, .dest = dest });
        }
        var sw = try f.wip.@"switch"(val, default, @intCast(cases.items.len), .none);
        for (cases.items) |cs| try sw.addCase(cs.val, cs.dest, &f.wip);
        sw.finish(&f.wip);
    }

    /// `[tail|musttail|notail] call [fast-math] [cc] [ret attrs] RET [(fn type)] @callee(args) [bundle]`
    fn lowerCall(f: *FunctionState, c: *Cursor, op: []const u8, result: ?[]const u8) Error!void {
        const m = f.m;
        const b = m.b;
        const wip = &f.wip;
        var kind: Builder.Function.Instruction.Call.Kind = .normal;
        if (!std.mem.eql(u8, op, "call")) {
            kind = if (std.mem.eql(u8, op, "tail")) .tail else if (std.mem.eql(u8, op, "musttail")) .musttail else .notail;
            c.skipWs();
            if (!std.mem.eql(u8, c.word(), "call")) return fail(c.line, c.s, "expected call", error.Syntax);
        }
        const fast = skipFastMath(c);
        if (fast == .fast) kind = switch (kind) {
            .normal => .fast,
            .tail => .tail_fast,
            .musttail => .musttail_fast,
            .notail => .notail_fast,
            else => kind,
        };
        const explicit_cc = eatCallConv(c);
        _ = skipAttrs(c);
        // The declaration is authoritative; the textual return type only says
        // whether a texture intrinsic's caller wants the colour or the pair.
        const text_ret = try m.parseType(c);
        c.skipWs();
        if (c.eat("(")) {
            // Explicit function type `ret (params) @callee(...)`: skip it.
            c.skipBalancedParens();
            c.skipWs();
        }
        try c.expect("@");
        const callee_name = try m.globalName(c);
        if (f.line_idx < f.line_level.len) {
            const scope = m.convergentScope(callee_name);
            const level = f.line_level[f.line_idx];
            if (!scope.allows(level)) {
                return fail(c.line, c.s, switch (scope) {
                    .threadgroup => switch (level) {
                        .simdgroup => "threadgroup barrier in SIMD-group-divergent control flow: this block runs for some SIMD-groups of the threadgroup only (a branch on simdgroup_index_in_threadgroup or on a simd_sum-style result). A SIMD-group call may sit here, a threadgroup barrier (or a helper that makes one) may not: move the barrier before the branch, see gpu.threadgroupBarrier",
                        else => "convergent call in divergent control flow: this block runs for a subset of the threadgroup (a branch on a per-thread id such as thread_index_in_threadgroup, a per-thread early return, or a loop some threads leave early). Zig cannot mark barriers, SIMD-group calls and the helpers that make them `convergent`, so LLVM duplicates the call that follows `if (tidx == 0) x = 0;` into both arms and Metal runs the copies as different barriers / partial SIMD sums; make the store before the call unconditional or move the conditional work after it, see gpu.threadgroupBarrier",
                    },
                    else => "SIMD-group call in divergent control flow: this block runs for a subset of the SIMD-group (a branch on a per-thread id such as thread_index_in_simdgroup or thread_position_in_grid, a per-thread early return, or a loop some threads leave early). Zig cannot mark simd_* calls, simdgroup_barrier and the helpers that make them `convergent`, so LLVM duplicates the call into the arms of the branch and Metal sums only the lanes of each arm; a branch on simdgroup_index_in_threadgroup is fine, a per-thread one is not, see gpu.simdSum",
                }, error.Unsupported);
            }
        }
        if (std.mem.startsWith(u8, callee_name, "llvm.nvvm.")) {
            return fail(c.line, c.s, "llvm.nvvm.* intrinsics have no AIR equivalent; thread ids are explicit kernel parameters described in the manifest", error.Unsupported);
        }
        if (std.mem.eql(u8, callee_name, "llvm.ctpop.i4")) {
            return fail(c.line, c.s, "llvm.ctpop.i4 crashes Metal's compiler; zext the bool vector to <N x i32> and add the lanes instead", error.Unsupported);
        }
        const info = try m.calleeInfo(callee_name, c);
        var fn_ty = info.fn_ty;
        try c.expect("(");
        var args: std.ArrayList(Value) = .empty;
        defer args.deinit(m.gpa);
        c.skipWs();
        if (!c.eat(")")) {
            while (true) {
                try args.append(m.gpa, try f.operand(c));
                c.skipWs();
                if (c.eat(",")) continue;
                try c.expect(")");
                break;
            }
        }
        // Operand bundles (`[ "cold"() ]`) carry nothing Metal needs.
        c.skipWs();
        if (c.eat("[")) c.i = c.s.len;

        // Intrinsics Metal's backend cannot take in their llvm.* form
        // (DESIGN.md D5-12a/12b): call the air.* equivalent, or expand.
        const lowering = try intrinsics.classify(m.gpa, callee_name);
        defer lowering.deinit(m.gpa);
        switch (lowering) {
            .pass => {},
            .refuse => |hint| return fail(c.line, c.s, hint, error.Unsupported),
            .reduce => |reduce_op| {
                var declarer = ModuleState.Declarer{ .m = m };
                const value = intrinsics.lowerReduce(b, wip, &declarer, reduce_op, args.items) catch |err| switch (err) {
                    error.OutOfMemory => return error.OutOfMemory,
                    error.Unsupported => return fail(c.line, c.s, "llvm.vector.reduce operand shape does not match the intrinsic (or its helper declaration conflicts)", error.Unsupported),
                };
                if (result) |r| try f.define(r, value);
                return;
            },
            .rename => |rename| {
                const new_ty = intrinsics.renamedFnType(b, rename, fn_ty) catch |err| switch (err) {
                    error.OutOfMemory => return error.OutOfMemory,
                    error.Unsupported => return fail(c.line, c.s, "intrinsic declaration has an unexpected shape", error.Unsupported),
                };
                const converted = try m.gpa.alloc(Value, args.items.len);
                defer m.gpa.free(converted);
                intrinsics.renamedArgs(b, wip, rename, args.items, converted) catch |err| switch (err) {
                    error.OutOfMemory => return error.OutOfMemory,
                    error.Unsupported => return fail(c.line, c.s, "intrinsic arguments have an unexpected shape", error.Unsupported),
                };
                @memcpy(args.items, converted);
                fn_ty = new_ty;
                const callee = m.intrinsicGlobal(rename.name, new_ty) catch |err| switch (err) {
                    error.Unsupported => return fail(c.line, c.s, "renamed intrinsic is already declared with another type", error.Unsupported),
                    else => |e| return e,
                };
                const value = try wip.call(kind, .ccc, .none, fn_ty, callee.index.toValue(), args.items, "");
                if (result) |r| try f.define(r, value);
                return;
            },
        }

        // Texture intrinsics were registered with Apple's signature, so the
        // declaration decides the call's type and the colour is unpacked from
        // the `{ <colour>, i8 }` pair Zig cannot name (DESIGN.md D5-12c).
        if (intrinsics.textureIntrinsic(callee_name)) |ti| {
            const callee = (try m.findGlobal(callee_name)) orelse
                return fail(c.line, c.s, "call to an undeclared texture intrinsic", error.UndefinedGlobal);
            const tex_params = fn_ty.functionParameters(b);
            if (tex_params.len != args.items.len) {
                return fail(c.line, c.s, "argument count does not match Apple's signature for this texture intrinsic", error.Syntax);
            }
            for (tex_params, args.items) |p, a| {
                if (p != a.typeOfWip(wip)) return fail(c.line, c.s, "texture intrinsic argument does not match Apple's signature: a texture is `ptr addrspace(1)` (a `.global` pointer in Zig), a sampler `ptr addrspace(2)` (gpu.constexprSampler or a manifest `.sampler` argument)", error.Unsupported);
            }
            const value = try wip.call(kind, .ccc, .none, fn_ty, callee.index.toValue(), args.items, "");
            if (result) |r| {
                // Zig's call binds the colour vector alone; IR that asks for
                // the pair keeps it and extracts the colour itself.
                if (ti.wraps_status and text_ret != fn_ty.functionReturn(b)) {
                    try f.define(r, try wip.extractValue(value, &[_]u32{0}, ""));
                } else try f.define(r, value);
            }
            return;
        }

        // Named operands carry their real types, so a relocated global or a
        // renumbered pointer may no longer match the declaration.
        const params = fn_ty.functionParameters(b);
        if (params.len != args.items.len) return fail(c.line, c.s, "argument count does not match the declaration", error.Syntax);
        var mismatch = false;
        for (params, args.items) |p, a| {
            if (p != a.typeOfWip(wip)) mismatch = true;
        }
        var callee: Global = undefined;
        if (mismatch) {
            if (intrinsics.isMemIntrinsic(callee_name)) {
                callee = try m.memIntrinsicFor(callee_name, fn_ty, args.items, wip, c);
                fn_ty = callee.fn_ty.?;
            } else return fail(c.line, c.s, "argument types do not match the callee's declaration (pointer address space?)", error.Unsupported);
        } else {
            callee = (try m.findGlobal(callee_name)) orelse return fail(c.line, c.s, "call to undeclared function", error.UndefinedGlobal);
        }
        if (callee.define) |di| try m.queue(di);
        const call_conv = explicit_cc orelse callee.call_conv;
        const value = try wip.call(kind, call_conv, .none, fn_ty, callee.index.toValue(), args.items, "");
        if (result) |r| try f.define(r, value);
    }

    /// `TYPE VALUE` where VALUE is a local or a constant.
    fn operand(f: *FunctionState, c: *Cursor) Error!Value {
        const ty = try f.m.parseType(c);
        return f.operandOfType(c, ty);
    }

    /// Named values (`%x`, `@g`) keep the type they were defined with; `ty`
    /// only shapes literal constants.
    fn operandOfType(f: *FunctionState, c: *Cursor, ty: Type) Error!Value {
        _ = skipAttrs(c);
        c.skipWs();
        if (c.eat("%")) return f.lookup(try f.m.quotedOrPlainName(c), c);
        return (try f.m.parseConst(c, ty)).toValue();
    }

    /// `, N, N...` for insertvalue / extractvalue.
    fn indexList(f: *FunctionState, c: *Cursor) Error![]u32 {
        var list: std.ArrayList(u32) = .empty;
        errdefer list.deinit(f.m.gpa);
        while (true) {
            c.skipWs();
            if (!c.eat(",")) break;
            try list.append(f.m.gpa, try c.int(u32));
        }
        if (list.items.len == 0) return fail(c.line, c.s, "expected aggregate indices", error.Syntax);
        return list.toOwnedSlice(f.m.gpa);
    }

    /// The pointer type a `phi ptr` really has: the type its incoming
    /// values agree on, looking at the ones resolvable before the phi is
    /// created (globals, `getelementptr` constant expressions over globals,
    /// and locals defined earlier in the function). `null`/`undef`/`poison`
    /// take whatever the others decide. When nothing resolves (all
    /// incomings are loop-carried) or the resolvable values disagree, the
    /// textual type stays and `resolvePhis` reports any mismatch.
    fn phiPointerType(f: *FunctionState, textual: Type, incoming: []const [2][]const u8, line_no: usize) Error!Type {
        const b = f.m.b;
        var found: ?Type = null;
        for (incoming) |inc| {
            const text = inc[0];
            if (text.len < 2) continue;
            var vc = Cursor{ .s = text, .line = line_no };
            const real: Type = switch (text[0]) {
                '@' => blk: {
                    _ = vc.eat("@");
                    const g = (try f.m.findGlobal(try f.m.globalName(&vc))) orelse continue;
                    break :blk g.index.toConst().typeOf(b);
                },
                '%' => blk: {
                    _ = vc.eat("%");
                    const name = try f.m.quotedOrPlainName(&vc);
                    for (f.locals.items) |l| if (std.mem.eql(u8, l.name, name)) break :blk l.value.typeOfWip(&f.wip);
                    continue;
                },
                // `getelementptr inbounds (i8, ptr @tab, i64 16)`: a
                // sub-slice of a relocated table; the constant inherits the
                // base's address space. Building it here is harmless, the
                // Builder interns constants.
                'g' => blk: {
                    if (!std.mem.startsWith(u8, text, "getelementptr")) continue;
                    break :blk (try f.m.parseConst(&vc, textual)).typeOf(b);
                },
                else => continue,
            };
            if (!real.isPointer(b)) return textual;
            if (found) |t| {
                if (t != real) return textual;
            } else found = real;
        }
        return found orelse textual;
    }

    fn resolvePhis(f: *FunctionState) Error!void {
        const gpa = f.m.gpa;
        for (f.phis.items) |p| {
            if (p.incoming.len != p.block_incoming) {
                std.debug.print("air-splice: line {d}: phi lists {d} incoming values but its block has {d} predecessor edges\n    {s}\n", .{ p.line_no, p.incoming.len, p.block_incoming, p.line });
                return error.Syntax;
            }
            var vals: std.ArrayList(Value) = .empty;
            defer vals.deinit(gpa);
            var blocks: std.ArrayList(BlockIndex) = .empty;
            defer blocks.deinit(gpa);
            for (p.incoming) |inc| {
                var vc = Cursor{ .s = inc[0], .line = p.line_no };
                const v = try f.operandOfType(&vc, p.ty);
                if (v.typeOfWip(&f.wip) != p.ty) return fail(p.line_no, p.line, "phi incoming value does not match the phi type", error.Syntax);
                try vals.append(gpa, v);
                var bc = Cursor{ .s = p.line, .line = p.line_no };
                try blocks.append(gpa, try f.block(inc[1], &bc));
            }
            p.phi.finish(vals.items, blocks.items, &f.wip);
        }
    }
};

fn labelName(line: []const u8) ?[]const u8 {
    if (line.len < 2) return null;
    var i: usize = 0;
    while (i < line.len and isIdentChar(line[i])) i += 1;
    if (i == 0 or i >= line.len or line[i] != ':') return null;
    return line[0..i];
}

fn trailingAlign(c: *Cursor) Error!Builder.Alignment {
    c.skipWs();
    if (c.eat(",")) {
        c.skipWs();
        if (c.eatWord("align")) return Builder.Alignment.fromByteUnits(try c.int(u64));
    }
    return .default;
}

/// `syncscope("...")` is ignored: Metal's scopes come from the address
/// space, and the Builder records the system scope.
fn skipSyncScope(c: *Cursor) void {
    if (c.eatWord("syncscope")) {
        c.skipWs();
        if (c.eat("(")) c.skipBalancedParens();
    }
}

fn parseOrdering(c: *Cursor) Error!Builder.AtomicOrdering {
    c.skipWs();
    const w = c.word();
    const ordering = std.meta.stringToEnum(Builder.AtomicOrdering, w) orelse return fail(c.line, c.s, "expected an atomic ordering", error.Syntax);
    if (ordering == .none) return fail(c.line, c.s, "expected an atomic ordering", error.Syntax);
    return ordering;
}

/// Consumes fast-math flags; any of them maps to the Builder's `fast` kind.
fn skipFastMath(c: *Cursor) Builder.FastMathKind {
    const flags = [_][]const u8{ "fast", "nnan", "ninf", "nsz", "arcp", "contract", "afn", "reassoc" };
    var fast: Builder.FastMathKind = .normal;
    while (true) {
        c.skipWs();
        const w = c.peekWord();
        var matched = false;
        for (flags) |x| if (std.mem.eql(u8, x, w)) {
            matched = true;
            break;
        };
        if (!matched) return fast;
        _ = c.word();
        fast = .fast;
    }
}

fn binaryTag(op: []const u8, c: *Cursor) ?Builder.Function.Instruction.Tag {
    const T = Builder.Function.Instruction.Tag;
    // nsw/nuw on add/sub/mul/shl predate LLVM 19 and Metal accepts them.
    // `WipFunction.bin` takes a single wrap flag, so `nuw nsw` keeps `nuw`.
    const int_ops = [_]struct { name: []const u8, plain: T, nsw: T, nuw: T }{
        .{ .name = "add", .plain = .add, .nsw = .@"add nsw", .nuw = .@"add nuw" },
        .{ .name = "sub", .plain = .sub, .nsw = .@"sub nsw", .nuw = .@"sub nuw" },
        .{ .name = "mul", .plain = .mul, .nsw = .@"mul nsw", .nuw = .@"mul nuw" },
        .{ .name = "shl", .plain = .shl, .nsw = .@"shl nsw", .nuw = .@"shl nuw" },
    };
    for (int_ops) |io| if (std.mem.eql(u8, op, io.name)) {
        var nsw = false;
        var nuw = false;
        while (true) {
            if (c.eatWord("nsw")) nsw = true else if (c.eatWord("nuw")) nuw = true else break;
        }
        return if (nuw) io.nuw else if (nsw) io.nsw else io.plain;
    };
    const exact_ops = [_]struct { name: []const u8, plain: T, exact: T }{
        .{ .name = "udiv", .plain = .udiv, .exact = .@"udiv exact" },
        .{ .name = "sdiv", .plain = .sdiv, .exact = .@"sdiv exact" },
        .{ .name = "lshr", .plain = .lshr, .exact = .@"lshr exact" },
        .{ .name = "ashr", .plain = .ashr, .exact = .@"ashr exact" },
    };
    for (exact_ops) |eo| if (std.mem.eql(u8, op, eo.name)) {
        return if (c.eatWord("exact")) eo.exact else eo.plain;
    };
    if (std.mem.eql(u8, op, "or")) {
        _ = c.eatWord("disjoint"); // LLVM >= 19 flag
        return .@"or";
    }
    const plain_ops = [_]struct { name: []const u8, tag: T }{
        .{ .name = "urem", .tag = .urem },  .{ .name = "srem", .tag = .srem },
        .{ .name = "and", .tag = .@"and" }, .{ .name = "xor", .tag = .xor },
    };
    for (plain_ops) |po| if (std.mem.eql(u8, op, po.name)) return po.tag;
    const float_ops = [_]struct { name: []const u8, plain: T, fast: T }{
        .{ .name = "fadd", .plain = .fadd, .fast = .@"fadd fast" },
        .{ .name = "fsub", .plain = .fsub, .fast = .@"fsub fast" },
        .{ .name = "fmul", .plain = .fmul, .fast = .@"fmul fast" },
        .{ .name = "fdiv", .plain = .fdiv, .fast = .@"fdiv fast" },
        .{ .name = "frem", .plain = .frem, .fast = .@"frem fast" },
    };
    for (float_ops) |fo| if (std.mem.eql(u8, op, fo.name)) {
        return if (skipFastMath(c) == .fast) fo.fast else fo.plain;
    };
    return null;
}

/// Cast mnemonics. LLVM >= 19 flags (`trunc nuw`, `zext nneg`, ...) are
/// dropped: the Builder would encode them as flag records that Apple's
/// reader rejects with "Failed to materializeAll".
fn castTag(op: []const u8, c: *Cursor) ?Builder.Function.Instruction.Tag {
    const T = Builder.Function.Instruction.Tag;
    const casts = [_]struct { name: []const u8, tag: T }{
        .{ .name = "trunc", .tag = .trunc },       .{ .name = "zext", .tag = .zext },
        .{ .name = "sext", .tag = .sext },         .{ .name = "fptrunc", .tag = .fptrunc },
        .{ .name = "fpext", .tag = .fpext },       .{ .name = "fptoui", .tag = .fptoui },
        .{ .name = "fptosi", .tag = .fptosi },     .{ .name = "uitofp", .tag = .uitofp },
        .{ .name = "sitofp", .tag = .sitofp },     .{ .name = "ptrtoint", .tag = .ptrtoint },
        .{ .name = "inttoptr", .tag = .inttoptr }, .{ .name = "bitcast", .tag = .bitcast },
    };
    for (casts) |ct| if (std.mem.eql(u8, op, ct.name)) {
        while (c.eatWord("nuw") or c.eatWord("nsw") or c.eatWord("nneg")) {}
        return ct.tag;
    };
    return null;
}

/// `-0.0` of a float or float-vector type, for lowering `fneg`.
fn negativeZero(b: *Builder, ty: Type) Error!Value {
    const scalar = ty.scalarType(b);
    const zero: Constant = switch (scalar) {
        .float => try b.floatConst(-0.0),
        .half => try b.halfConst(-0.0),
        .double => try b.doubleConst(-0.0),
        else => return error.Unsupported,
    };
    if (ty == scalar) return zero.toValue();
    return b.splatValue(ty, zero);
}

/// LLVM identifiers: `[-a-zA-Z$._][-a-zA-Z$._0-9]*` (loop unrolling emits
/// `._crit_edge.loopexit.unr-lcssa`).
fn isIdentChar(ch: u8) bool {
    return std.ascii.isAlphanumeric(ch) or ch == '_' or ch == '.' or ch == '$' or ch == '-';
}

fn findPos(hay: []const u8, start: usize, needle: []const u8) ?usize {
    if (needle.len == 0 or hay.len < needle.len) return null;
    var i = start;
    while (i + needle.len <= hay.len) : (i += 1) {
        if (std.mem.eql(u8, hay[i .. i + needle.len], needle)) return i;
    }
    return null;
}

// ── Cursor ──────────────────────────────────────────────────────────────────

const Cursor = struct {
    s: []const u8,
    i: usize = 0,
    line: usize,

    fn skipWs(c: *Cursor) void {
        while (c.i < c.s.len and (c.s[c.i] == ' ' or c.s[c.i] == '\t')) c.i += 1;
    }

    fn rest(c: *Cursor) []const u8 {
        c.skipWs();
        return c.s[c.i..];
    }

    /// Consume `lit` if it is next (after whitespace).
    fn eat(c: *Cursor, lit: []const u8) bool {
        c.skipWs();
        if (std.mem.startsWith(u8, c.s[c.i..], lit)) {
            c.i += lit.len;
            return true;
        }
        return false;
    }

    /// Consume `word` only as a whole identifier.
    fn eatWord(c: *Cursor, word_lit: []const u8) bool {
        c.skipWs();
        if (std.mem.startsWith(u8, c.s[c.i..], word_lit)) {
            const end = c.i + word_lit.len;
            if (end >= c.s.len or !isIdentChar(c.s[end])) {
                c.i = end;
                return true;
            }
        }
        return false;
    }

    fn expect(c: *Cursor, lit: []const u8) Error!void {
        if (!c.eat(lit)) {
            std.debug.print("air-splice: line {d}: expected '{s}' at column {d}\n    {s}\n", .{ c.line, lit, c.i, c.s });
            return error.Syntax;
        }
    }

    fn expectWord(c: *Cursor, word_lit: []const u8) Error!void {
        if (!c.eatWord(word_lit)) {
            std.debug.print("air-splice: line {d}: expected '{s}' at column {d}\n    {s}\n", .{ c.line, word_lit, c.i, c.s });
            return error.Syntax;
        }
    }

    /// Identifier characters: letters, digits, `_`, `.`, `$`, `-`.
    fn word(c: *Cursor) []const u8 {
        c.skipWs();
        const start = c.i;
        while (c.i < c.s.len and isIdentChar(c.s[c.i])) c.i += 1;
        return c.s[start..c.i];
    }

    fn peekWord(c: *Cursor) []const u8 {
        c.skipWs();
        var j = c.i;
        while (j < c.s.len and isIdentChar(c.s[j])) j += 1;
        return c.s[c.i..j];
    }

    /// Like `word` but also accepts a leading `+` (number literals).
    fn numberOrWord(c: *Cursor) []const u8 {
        c.skipWs();
        const start = c.i;
        if (c.i < c.s.len and c.s[c.i] == '+') c.i += 1;
        while (c.i < c.s.len and (isIdentChar(c.s[c.i]) or c.s[c.i] == '+')) c.i += 1;
        return c.s[start..c.i];
    }

    fn int(c: *Cursor, comptime T: type) Error!T {
        const w = c.numberOrWord();
        return std.fmt.parseInt(T, w, 10) catch {
            std.debug.print("air-splice: line {d}: expected integer, found '{s}'\n    {s}\n", .{ c.line, w, c.s });
            return error.Syntax;
        };
    }

    /// Skips to just past the `)` matching an already consumed `(`.
    fn skipBalancedParens(c: *Cursor) void {
        var depth: usize = 1;
        while (c.i < c.s.len and depth > 0) : (c.i += 1) {
            if (c.s[c.i] == '(') depth += 1;
            if (c.s[c.i] == ')') depth -= 1;
        }
    }
};

// ── Tests ───────────────────────────────────────────────────────────────────

const test_header =
    \\target datalayout = "e-p:64:64:64"
    \\target triple = "air64_v28-apple-macosx26.0.0"
    \\
;

/// The tests carry a manifest of their own, built through the same
/// `metadata.functionMetadata` path as the real one, so they keep working
/// whatever `shader.functions` lists at the time.
const TestVertexIn = struct { position: @Vector(4, f32), normal: @Vector(4, f32), texCoords: @Vector(2, f32) };
const TestVertexOut = struct { position: @Vector(4, f32), normal: @Vector(3, f32), texCoords: @Vector(2, f32) };
const vertex_fm = metadata.functionMetadata(.{
    .name = "vertexShader",
    .stage = .vertex,
    .ret = TestVertexOut,
    .args = &.{
        .{ .vertex_id = "vertexID" },
        .{ .buffer = .{ .index = 0, .T = TestVertexIn, .name = "vertices" } },
    },
});
const fragment_fm = metadata.functionMetadata(.{
    .name = "fragmentShader",
    .stage = .fragment,
    .ret = @Vector(4, f32),
    .args = &.{
        .{ .stage_in = TestVertexOut },
        // Test-only use of the shader module: a texture argument names its Zig
        // type so the metadata can read the access from `T.texture_access`.
        .{ .texture = .{ .index = 0, .name = "colorTexture", .T = @import("shader").gpu.Texture2D(.sample) } },
    },
});
/// `reduceKernel(in, out, tidx, tg_size, params)`: `tidx` is per-thread,
/// `tg_size` and the constant buffer are uniform (param_uniformity =
/// {threadgroup, threadgroup, thread, threadgroup, threadgroup}).
const kernel_fm = metadata.functionMetadata(.{
    .name = "reduceKernel",
    .stage = .kernel,
    .ret = void,
    .args = &.{
        .{ .buffer = .{ .index = 0, .T = f32, .name = "in" } },
        .{ .buffer = .{ .index = 1, .T = f32, .name = "out", .access = .read_write } },
        .{ .builtin = .{ .kind = .thread_index_in_threadgroup, .T = u32, .name = "tidx" } },
        .{ .builtin = .{ .kind = .threads_per_threadgroup, .T = u32, .name = "tg_size" } },
        .{ .buffer = .{ .index = 2, .T = u32, .name = "params", .space = .constant } },
    },
});

/// `twoStageKernel(in, out, gid, lane, sgid, nsg, tg_pos)` of the sample
/// shader: `gid` and `lane` are per-thread, `sgid` per-SIMD-group, the
/// rest uniform.
const simd_kernel_fm = metadata.functionMetadata(.{
    .name = "twoStageKernel",
    .stage = .kernel,
    .ret = void,
    .args = &.{
        .{ .buffer = .{ .index = 0, .T = f32, .name = "in" } },
        .{ .buffer = .{ .index = 1, .T = f32, .name = "out", .access = .read_write } },
        .{ .builtin = .{ .kind = .thread_position_in_grid, .T = u32, .name = "gid" } },
        .{ .builtin = .{ .kind = .thread_index_in_simdgroup, .T = u32, .name = "lane" } },
        .{ .builtin = .{ .kind = .simdgroup_index_in_threadgroup, .T = u32, .name = "sgid" } },
        .{ .builtin = .{ .kind = .simdgroups_per_threadgroup, .T = u32, .name = "nsg" } },
        .{ .builtin = .{ .kind = .threadgroup_position_in_grid, .T = u32, .name = "tg_pos" } },
    },
});

fn has(hay: []const u8, needle: []const u8) bool {
    return findPos(hay, 0, needle) != null;
}

test {
    _ = intrinsics;
    _ = divergence;
}

/// Assembles `module` (which must define `@fragmentShader`), checks the
/// bitcode header and returns the Builder's own rendering of the module, so
/// tests can see what was built.
fn testRoundTrip(gpa: Allocator, module: []const u8) ![]u8 {
    return testRoundTripAs(gpa, module, "fragmentShader", fragment_fm);
}

fn testRoundTripAs(gpa: Allocator, module: []const u8, comptime entry: []const u8, comptime fm: metadata.FunctionMetadata) ![]u8 {
    var b = try Builder.init(.{ .allocator = gpa, .strip = true, .name = "air-splice" });
    defer b.deinit();
    try build(&b, gpa, module, .{ .entry = entry, .fm = fm }, air_target.default);
    const words = try b.toBitcode(gpa, .{ .name = "zig air-splice", .version = .{ .major = 0, .minor = 1, .patch = 0 } });
    defer gpa.free(words);
    const bytes = std.mem.sliceAsBytes(words);
    try std.testing.expect(bytes.len > 64);
    try std.testing.expectEqualStrings("BC\xC0\xDE", bytes[0..4]);
    var aw: std.Io.Writer.Allocating = .init(gpa);
    defer aw.deinit();
    try b.print(&aw.writer);
    return aw.toOwnedSlice();
}

fn testFails(gpa: Allocator, module: []const u8) !void {
    return testFailsAs(gpa, module, "fragmentShader", fragment_fm);
}

fn testFailsAs(gpa: Allocator, module: []const u8, comptime entry: []const u8, comptime fm: metadata.FunctionMetadata) !void {
    var b = try Builder.init(.{ .allocator = gpa, .strip = true, .name = "air-splice" });
    defer b.deinit();
    try std.testing.expectError(error.Unsupported, build(&b, gpa, module, .{ .entry = entry, .fm = fm }, air_target.default));
}

test "assembles the converted vertex/fragment pair into bitcode" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();
    const ir =
        \\target datalayout = "e-p:64:64:64"
        \\target triple = "air64_v28-apple-macosx26.0.0"
        \\
        \\%my_shader.VertexOut = type { <4 x float>, <3 x float>, <2 x float>, [8 x i8] }
        \\
        \\define <{ <4 x float>, <3 x float>, <2 x float> }> @vertexShader(i32 %__1, ptr addrspace(1) %__2) {
        \\  %sret = alloca %my_shader.VertexOut, align 16
        \\  %__4 = zext i32 %__1 to i64
        \\  %__5 = getelementptr inbounds [48 x i8], ptr addrspace(1) %__2, i64 %__4
        \\  %a = load <4 x float>, ptr addrspace(1) %__5, align 16
        \\  %p1 = getelementptr inbounds i8, ptr addrspace(1) %__5, i64 16
        \\  %n4 = load <4 x float>, ptr addrspace(1) %p1, align 16
        \\  %n3 = shufflevector <4 x float> %n4, <4 x float> poison, <3 x i32> <i32 0, i32 1, i32 2>
        \\  store <4 x float> %a, ptr %sret, align 16
        \\  %s1 = getelementptr inbounds i8, ptr %sret, i64 16
        \\  store <3 x float> %n3, ptr %s1, align 16
        \\  %r0.f0 = load <4 x float>, ptr %sret, align 16
        \\  %r0.f1 = load <3 x float>, ptr %s1, align 16
        \\  %r0.a0 = insertvalue <{ <4 x float>, <3 x float>, <2 x float> }> undef, <4 x float> %r0.f0, 0
        \\  %r0.a1 = insertvalue <{ <4 x float>, <3 x float>, <2 x float> }> %r0.a0, <3 x float> %r0.f1, 1
        \\  %r0.a2 = insertvalue <{ <4 x float>, <3 x float>, <2 x float> }> %r0.a1, <2 x float> zeroinitializer, 2
        \\  ret <{ <4 x float>, <3 x float>, <2 x float> }> %r0.a2
        \\}
        \\
        \\define <4 x float> @fragmentShader(<4 x float> %__0, <3 x float> %__1, <2 x float> %__2, ptr addrspace(1) %__3) {
        \\  %c = fcmp fast ogt float 1.000000e+00, 0x3FE0000000000000
        \\  br i1 %c, label %__yes, label %__no
        \\__yes:
        \\  br label %__join
        \\__no:
        \\  br label %__join
        \\__join:
        \\  %v = phi <4 x float> [ <float 1.000000e+00, float 0.000000e+00, float 0.000000e+00, float 1.000000e+00>, %__yes ], [ zeroinitializer, %__no ]
        \\  %w = fneg <4 x float> %v
        \\  %x = fadd fast <4 x float> %w, %v
        \\  ret <4 x float> %x
        \\}
        \\
    ;
    inline for (.{ vertex_fm, fragment_fm }) |fm| {
        const bc = try assemble(gpa, ir, .{
            .entry = fm.name,
            .fm = fm,
        }, air_target.default);
        try std.testing.expect(bc.len > 64);
        try std.testing.expectEqualStrings("BC\xC0\xDE", bc[0..4]);
    }
    // One module per entry point: the fragment module must not carry the
    // vertex function.
    const text = try testRoundTrip(gpa, ir);
    try std.testing.expect(has(text, "@fragmentShader"));
    try std.testing.expect(!has(text, "@vertexShader"));
}

test "D5.1 splat constants in instructions and global initialisers" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();
    const text = try testRoundTrip(gpa, test_header ++
        \\@palette = private unnamed_addr constant [2 x <4 x float>] [<4 x float> <float 1.000000e+00, float 0.000000e+00, float 0.000000e+00, float 1.000000e+00>, <4 x float> splat (float 1.000000e+00)], align 16
        \\
        \\define <4 x float> @fragmentShader(<4 x float> %__0, <4 x half> %__1) {
        \\  %__3 = fmul <4 x float> %__0, splat (float 2.000000e+00)
        \\  %__4 = fsub <4 x float> splat (float 1.000000e+00), %__3
        \\  %__5 = fadd <4 x half> %__1, splat (half 0xH3800)
        \\  %__6 = fcmp oge <4 x float> %__4, splat (float 5.000000e-01)
        \\  %__7 = load <4 x float>, ptr addrspace(2) getelementptr inbounds ([2 x <4 x float>], ptr addrspace(2) @palette, i64 0, i64 1), align 16
        \\  %__8 = select <4 x i1> %__6, <4 x float> %__4, <4 x float> %__7
        \\  ret <4 x float> %__8
        \\}
        \\
    );
    // The Builder prints floats as hex doubles and expands splats.
    try std.testing.expect(has(text, "fmul <4 x float> %0, <float 0x4000000000000000, float 0x4000000000000000, float 0x4000000000000000, float 0x4000000000000000>"));
    try std.testing.expect(has(text, "fadd <4 x half> %1, <half 0xH3800, half 0xH3800, half 0xH3800, half 0xH3800>"));
    try std.testing.expect(has(text, "@palette = private unnamed_addr addrspace(2) constant") or has(text, "@palette = private addrspace(2) constant"));
}

test "D5.2 multi-line switch with duplicate targets and a phi in the join block" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();
    const text = try testRoundTrip(gpa, test_header ++
        \\define <4 x float> @fragmentShader(<4 x float> %__0, i32 %__1) {
        \\  switch i32 %__1, label %__9 [
        \\    i32 0, label %__3
        \\    i32 1, label %__5
        \\    i32 5, label %__7
        \\    i32 6, label %__7
        \\  ]
        \\
        \\__3:                                                ; preds = %__2, %__9, %__7, %__5
        \\  %__4 = phi <4 x float> [ zeroinitializer, %__9 ], [ %__6, %__5 ], [ %__8, %__7 ], [ %__0, %__2 ]
        \\  ret <4 x float> %__4
        \\
        \\__5:                                                ; preds = %__2
        \\  %__6 = fmul <4 x float> %__0, splat (float 2.000000e+00)
        \\  br label %__3
        \\
        \\__7:                                                ; preds = %__2, %__2
        \\  %__8 = fdiv <4 x float> %__0, splat (float 3.000000e+00)
        \\  br label %__3
        \\
        \\__9:                                                ; preds = %__2
        \\  br label %__3
        \\}
        \\
    );
    try std.testing.expect(has(text, "switch i32"));
    try std.testing.expect(has(text, "i32 6, label"));
    try std.testing.expect(has(text, "phi <4 x float>"));
}

test "D5.3 freeze binds the result to its operand" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();
    const text = try testRoundTrip(gpa, test_header ++
        \\define i32 @fragmentShader(i32 %__0, i32 %__1) {
        \\  %.frozen = freeze i32 %__0
        \\  %.frozen1 = freeze i32 %__1
        \\  %__3 = udiv i32 %.frozen, %.frozen1
        \\  %__4 = mul i32 %__3, %.frozen1
        \\  %.decomposed = sub i32 %.frozen, %__4
        \\  %__5 = add nuw i32 %.decomposed, %__3
        \\  ret i32 %__5
        \\}
        \\
    );
    try std.testing.expect(!has(text, "freeze"));
    try std.testing.expect(has(text, "udiv i32"));
    try std.testing.expect(has(text, "add nuw i32"));
}

test "D5.4 range/noundef/immarg attributes on define, declare and call result positions" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();
    const text = try testRoundTrip(gpa, test_header ++
        \\declare noundef range(i32 0, 33) i32 @llvm.ctlz.i32(i32 noundef, i1 immarg)
        \\declare void @llvm.assume(i1 noundef)
        \\
        \\define noundef range(i32 0, 67564) i32 @fragmentShader(i32 noundef %__0, i32 returned signext %__1, ptr dead_on_return zeroext %__2) {
        \\  %__4 = tail call range(i32 0, 33) i32 @llvm.ctlz.i32(i32 %__0, i1 false)
        \\  %__5 = icmp ne i32 %__4, 0
        \\  tail call void @llvm.assume(i1 %__5)
        \\  %__6 = add nuw nsw i32 %__4, %__1
        \\  ret i32 %__6
        \\}
        \\
    );
    try std.testing.expect(has(text, "@llvm.ctlz.i32"));
    try std.testing.expect(has(text, "@llvm.assume"));
    try std.testing.expect(!has(text, "range("));
}

test "D5.5 identifiers containing '-' as labels and values" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();
    const text = try testRoundTrip(gpa, test_header ++
        \\define float @fragmentShader(ptr addrspace(1) %__0, i32 %__1) {
        \\  %niter.ncmp-7 = icmp eq i32 %__1, 0
        \\  br i1 %niter.ncmp-7, label %._crit_edge.loopexit.unr-lcssa, label %.lr.ph
        \\
        \\.lr.ph:                                           ; preds = %__2
        \\  %v = load float, ptr addrspace(1) %__0, align 4
        \\  br label %._crit_edge.loopexit.unr-lcssa
        \\
        \\._crit_edge.loopexit.unr-lcssa:                   ; preds = %.lr.ph, %__2
        \\  %r = phi float [ %v, %.lr.ph ], [ -1.000000e+00, %__2 ]
        \\  ret float %r
        \\}
        \\
    );
    try std.testing.expect(has(text, "phi float"));
    // LLVM spells -1.0 either way depending on the printer.
    try std.testing.expect(has(text, "-1.000000e+00") or has(text, "0xBFF0000000000000"));
}

test "D5.6 c-string initialisers with escapes, nested arrays" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();
    const text = try testRoundTrip(gpa, test_header ++
        \\@lut.bayer = private unnamed_addr constant [4 x [4 x i8]] [[4 x i8] c"\00\08\02\0A", [4 x i8] c"\0C\04\0E\06", [4 x i8] c"\03\0B\01\09", [4 x i8] c"\0F\07\0D\05"], align 1
        \\@msg = private unnamed_addr constant [6 x i8] c"hi\5C\22!\00", align 1
        \\
        \\define i32 @fragmentShader(i32 %__0, i32 %__1) {
        \\  %__3 = and i32 %__1, 3
        \\  %__4 = zext i32 %__3 to i64
        \\  %__5 = getelementptr inbounds [4 x i8], ptr @lut.bayer, i64 %__4
        \\  %__6 = and i32 %__0, 3
        \\  %__7 = zext i32 %__6 to i64
        \\  %__8 = getelementptr inbounds [1 x i8], ptr %__5, i64 %__7
        \\  %__9 = load i8, ptr %__8, align 1
        \\  %__10 = load i8, ptr @msg, align 1
        \\  %__11 = add i8 %__9, %__10
        \\  %__12 = zext i8 %__11 to i32
        \\  ret i32 %__12
        \\}
        \\
    );
    try std.testing.expect(has(text, "c\"\\00\\08\\02\\0A\""));
    try std.testing.expect(has(text, "@lut.bayer = private unnamed_addr addrspace(2) constant [4 x [4 x i8]]") or has(text, "@lut.bayer = private addrspace(2) constant [4 x [4 x i8]]"));
    // GEPs through the relocated global follow its address space.
    try std.testing.expect(has(text, "getelementptr inbounds [4 x i8], ptr addrspace(2) @lut.bayer"));
    try std.testing.expect(has(text, "load i8, ptr addrspace(2)"));
}

test "D5.7 getelementptr constant expressions as operands" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();
    const text = try testRoundTrip(gpa, test_header ++
        \\@shared.tile = private unnamed_addr addrspace(3) global [256 x float] undef, align 4
        \\@lut = private unnamed_addr constant [4 x float] [float 1.000000e+00, float 2.000000e+00, float 3.000000e+00, float 4.000000e+00], align 4
        \\
        \\define float @fragmentShader(i32 %__0) {
        \\  %__2 = load float, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shared.tile, i64 4), align 4
        \\  %__3 = load float, ptr getelementptr inbounds (i8, ptr @lut, i64 8), align 4
        \\  %__4 = fadd float %__2, %__3
        \\  store float %__4, ptr addrspace(3) getelementptr (i8, ptr addrspace(3) @shared.tile, i64 12), align 4
        \\  ret float %__4
        \\}
        \\
    );
    try std.testing.expect(has(text, "getelementptr inbounds (i8, ptr addrspace(3) @shared.tile, i64 4)"));
    try std.testing.expect(has(text, "getelementptr inbounds (i8, ptr addrspace(2) @lut, i64 8)"));
    try std.testing.expect(has(text, "addrspace(3) global [256 x float] undef"));
}

test "D5.8 non-entry defines are registered and lowered when reached; call fastcc" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();
    const text = try testRoundTrip(gpa, test_header ++
        \\%calls.Hit = type { <3 x float>, float, i8, [11 x i8] }
        \\
        \\define <4 x float> @fragmentShader(<3 x float> %__0, <3 x float> %__1) {
        \\  %__3 = alloca %calls.Hit, align 16
        \\  call fastcc void @calls.intersect(ptr noalias writeonly align 16 %__3, <3 x float> %__0, <3 x float> %__1)
        \\  %__4 = tail call fastcc float @calls.lengthSq(<3 x float> %__1)
        \\  %__5 = load <3 x float>, ptr %__3, align 16
        \\  %__6 = shufflevector <3 x float> %__5, <3 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 poison>
        \\  %__7 = insertelement <4 x float> %__6, float %__4, i64 3
        \\  ret <4 x float> %__7
        \\}
        \\
        \\define private fastcc float @calls.lengthSq(<3 x float> %__0) unnamed_addr #5 {
        \\  %__2 = extractelement <3 x float> %__0, i64 0
        \\  %__3 = fmul float %__2, %__2
        \\  ret float %__3
        \\}
        \\
        \\define private fastcc void @calls.intersect(ptr dead_on_unwind noalias nonnull writeonly align 16 captures(none) initializes((0, 12), (16, 21)) %__0, <3 x float> %__1, <3 x float> %__2) unnamed_addr #3 {
        \\  %__4 = tail call fastcc float @calls.lengthSq(<3 x float> %__1)
        \\  %__5 = insertelement <3 x float> %__2, float %__4, i64 0
        \\  store <3 x float> %__5, ptr %__0, align 16
        \\  ret void
        \\}
        \\
        \\define private fastcc float @calls.unused(float %__0) unnamed_addr #5 {
        \\  ret float %__0
        \\}
        \\
    );
    try std.testing.expect(has(text, "define private fastcc float @calls.lengthSq"));
    try std.testing.expect(has(text, "define private fastcc void @calls.intersect"));
    try std.testing.expect(has(text, "call fastcc void @calls.intersect"));
    try std.testing.expect(has(text, "tail call fastcc float @calls.lengthSq"));
    // Unreached helpers leave nothing behind, not even a declaration.
    try std.testing.expect(!has(text, "calls.unused"));
}

test "D5.8 quoted identifiers: `@\"q.Vec(4).sum\"` defines and calls resolve, `\\XX` escapes decode" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();
    const text = try testRoundTrip(gpa, test_header ++
        \\@"tab le" = private unnamed_addr constant [2 x float] [float 1.0, float 2.0], align 4
        \\
        \\define private fastcc float @"q.Vec(4).sum"(<4 x float> %__0) {
        \\  %__2 = extractelement <4 x float> %__0, i64 0
        \\  %__3 = load float, ptr @"tab le", align 4
        \\  %__4 = fadd float %__2, %__3
        \\  ret float %__4
        \\}
        \\
        \\define private fastcc float @"esc\28x\29"(float %__0) {
        \\  ret float %__0
        \\}
        \\
        \\define <4 x float> @fragmentShader(<4 x float> %__0, <3 x float> %__1, <2 x float> %__2, ptr addrspace(1) %__3) {
        \\  %__5 = tail call fastcc float @"q.Vec(4).sum"(<4 x float> %__0)
        \\  %__6 = tail call fastcc float @"esc\28x\29"(float %__5)
        \\  %__7 = insertelement <4 x float> %__0, float %__6, i64 0
        \\  ret <4 x float> %__7
        \\}
        \\
    );
    try std.testing.expect(has(text, "define private fastcc float @\"q.Vec(4).sum\"("));
    try std.testing.expect(has(text, "call fastcc float @\"q.Vec(4).sum\"("));
    // Escapes decoded once: the module holds `esc(x)`, not `esc\28x\29`.
    try std.testing.expect(has(text, "@\"esc(x)\"("));
    try std.testing.expect(!has(text, "esc\\28"));
    try std.testing.expect(has(text, "@\"tab le\" = private addrspace(2) constant"));
}

test "D5.8 quoted named types: `%\"q.Pair(f32)\"` definitions and references resolve, escapes decode" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();
    // The shape Zig emits for a generic struct instantiation that goes
    // through memory (review-a2/r5_generic.zig): a quoted named type used
    // by alloca/gep/load, filled by a quoted helper through an sret-like
    // pointer. `%"esc\28x\29"` checks that `\XX` escapes decode in type
    // names too, and `%"loc al"` that quoted local values resolve.
    const text = try testRoundTrip(gpa, test_header ++
        \\%"q.Pair(f32)" = type { float, float }
        \\%"esc\28x\29" = type { <2 x float>, %"q.Pair(f32)" }
        \\
        \\define private fastcc void @"q.mk(f32)"(ptr noalias nonnull writeonly align 4 captures(none) initializes((0, 8)) %__0, float %__1, float %__2) unnamed_addr #2 {
        \\  store float %__1, ptr %__0, align 4
        \\  %__4 = getelementptr inbounds nuw i8, ptr %__0, i64 4
        \\  store float %__2, ptr %__4, align 4
        \\  ret void
        \\}
        \\
        \\define <4 x float> @fragmentShader(<4 x float> %__0, <3 x float> %__1, <2 x float> %__2, ptr addrspace(1) %__3) {
        \\  %__5 = alloca %"q.Pair(f32)", align 4
        \\  %"loc al" = alloca %"esc\28x\29", align 8
        \\  %__6 = extractelement <2 x float> %__2, i64 0
        \\  %__7 = extractelement <3 x float> %__1, i64 0
        \\  call fastcc void @"q.mk(f32)"(ptr noalias writeonly align 4 captures(none) %__5, float %__6, float %__7)
        \\  %__8 = load float, ptr %__5, align 4
        \\  %__9 = getelementptr inbounds nuw %"q.Pair(f32)", ptr %__5, i64 0, i32 1
        \\  %__10 = load float, ptr %__9, align 4
        \\  %__11 = getelementptr inbounds %"esc\28x\29", ptr %"loc al", i64 0, i32 1, i32 0
        \\  store float %__10, ptr %__11, align 4
        \\  %__12 = load float, ptr %__11, align 4
        \\  %__13 = fadd float %__8, %__12
        \\  %__14 = insertelement <4 x float> %__0, float %__13, i64 0
        \\  ret <4 x float> %__14
        \\}
        \\
    );
    // Named types are resolved structurally; the alloca/gep carry the body.
    try std.testing.expect(has(text, "alloca { float, float }"));
    try std.testing.expect(has(text, "getelementptr inbounds { float, float }, ptr %5, i64 0, i32 1"));
    try std.testing.expect(has(text, "alloca { <2 x float>, { float, float } }"));
    try std.testing.expect(has(text, "define private fastcc void @\"q.mk(f32)\"("));
    try std.testing.expect(has(text, "call fastcc void @\"q.mk(f32)\"("));
    // Local names are stripped in the rendering; the gep through the quoted
    // local `%"loc al"` (the second alloca) shows it resolved.
    try std.testing.expect(has(text, "getelementptr inbounds { <2 x float>, { float, float } }, ptr %6, i64 0, i32 1, i32 0"));
    try std.testing.expect(!has(text, "esc\\28"));

    // A reference to a quoted type that was never defined is a clean
    // UndefinedType, not a syntax error at the wrong construct.
    var b = try Builder.init(.{ .allocator = gpa, .strip = true, .name = "air-splice" });
    defer b.deinit();
    try std.testing.expectError(error.UndefinedType, build(&b, gpa, test_header ++
        \\define <4 x float> @fragmentShader(<4 x float> %__0, <3 x float> %__1, <2 x float> %__2, ptr addrspace(1) %__3) {
        \\  %__5 = alloca %"q.Missing(f32)", align 4
        \\  ret <4 x float> %__0
        \\}
        \\
    , .{ .entry = "fragmentShader", .fm = fragment_fm }, air_target.default));
}

test "D5.9 LLVM 19 flags on casts and binary ops are dropped" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();
    const text = try testRoundTrip(gpa, test_header ++
        \\define i32 @fragmentShader(i32 %__0, i32 %__1, ptr addrspace(1) %__2) {
        \\  %__4 = trunc nuw nsw i32 %__0 to i16
        \\  %__5 = trunc nuw i32 %__1 to i8
        \\  %__6 = trunc nsw i32 %__1 to i6
        \\  %__7 = zext nneg i16 %__4 to i64
        \\  %__8 = or disjoint i32 %__0, 1
        \\  %__9 = icmp samesign ult i32 %__8, %__1
        \\  %__10 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %__2, i64 %__7
        \\  %__11 = load i32, ptr addrspace(1) %__10, align 4
        \\  %__12 = uitofp nneg i8 %__5 to float
        \\  %__13 = fptoui float %__12 to i32
        \\  %__14 = zext i6 %__6 to i32
        \\  %__15 = select i1 %__9, i32 %__11, i32 %__13
        \\  %__16 = add i32 %__15, %__14
        \\  ret i32 %__16
        \\}
        \\
    );
    try std.testing.expect(has(text, "trunc i32"));
    try std.testing.expect(!has(text, "trunc nuw"));
    try std.testing.expect(!has(text, "trunc nsw"));
    try std.testing.expect(!has(text, "nneg"));
    try std.testing.expect(!has(text, "disjoint"));
    try std.testing.expect(!has(text, "samesign"));
    try std.testing.expect(has(text, "getelementptr inbounds [4 x i8]"));
}

test "D1/D5.10 address spaces: renumbering, constant relocation, value-typed operands, memcpy rename" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();
    const text = try testRoundTrip(gpa, test_header ++
        \\@__anon_796 = private unnamed_addr constant [4 x float] [float 1.000000e+00, float 2.000000e+00, float 3.000000e+00, float 4.000000e+00], align 4
        \\@gvar = private unnamed_addr addrspace(1) global float 0.000000e+00, align 4
        \\@tile = private unnamed_addr addrspace(3) global [64 x float] zeroinitializer, align 4
        \\@counter = private unnamed_addr addrspace(3) global i32 0, align 4
        \\
        \\declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg)
        \\
        \\define float @fragmentShader(i32 %__0, ptr addrspace(4) %__1, ptr addrspace(5) %__2) {
        \\  %__4 = alloca [4 x float], align 16
        \\  call void @llvm.memcpy.p0.p0.i64(ptr align 16 %__4, ptr align 4 @__anon_796, i64 16, i1 false)
        \\  %__5 = and i32 %__0, 3
        \\  %__6 = zext i32 %__5 to i64
        \\  %__7 = getelementptr inbounds nuw [4 x i8], ptr @__anon_796, i64 %__6
        \\  %__8 = load float, ptr %__7, align 4
        \\  %__9 = load float, ptr addrspace(4) %__1, align 4
        \\  store float %__9, ptr addrspace(5) %__2, align 4
        \\  %__10 = getelementptr inbounds [4 x i8], ptr addrspace(5) %__2, i64 %__6
        \\  store float %__8, ptr addrspace(5) %__10, align 4
        \\  %__11 = load float, ptr addrspace(1) @gvar, align 4
        \\  store float %__11, ptr addrspace(1) @gvar, align 4
        \\  %__12 = getelementptr inbounds [4 x i8], ptr addrspace(3) @tile, i64 %__6
        \\  store float %__11, ptr addrspace(3) %__12, align 4
        \\  %__13 = load i32, ptr addrspace(3) @counter, align 4
        \\  %__14 = uitofp i32 %__13 to float
        \\  %__15 = fadd float %__8, %__14
        \\  %__16 = getelementptr inbounds [4 x i8], ptr %__4, i64 %__6
        \\  %__17 = load float, ptr %__16, align 4
        \\  %__18 = fadd float %__15, %__17
        \\  ret float %__18
        \\}
        \\
    );
    try std.testing.expect(!has(text, "addrspace(4)"));
    try std.testing.expect(!has(text, "addrspace(5)"));
    try std.testing.expect(has(text, "addrspace(2) constant [4 x float]"));
    try std.testing.expect(has(text, "@gvar = private unnamed_addr addrspace(1) global float") or has(text, "@gvar = private addrspace(1) global float"));
    try std.testing.expect(has(text, "ptr addrspace(2) %"));
    try std.testing.expect(has(text, "getelementptr inbounds [4 x i8], ptr %"));
    try std.testing.expect(has(text, "getelementptr inbounds [4 x i8], ptr addrspace(2) @__anon_796"));
    try std.testing.expect(has(text, "@llvm.memcpy.p0.p2.i64(ptr"));
    try std.testing.expect(has(text, "ptr addrspace(2) @__anon_796, i64 16"));
    try std.testing.expect(!has(text, "@llvm.memcpy.p0.p0.i64"));

    // Any other address-space mismatch is refused rather than emitted.
    try testFails(gpa, test_header ++
        \\@__anon_1 = private unnamed_addr constant [4 x float] zeroinitializer, align 4
        \\declare float @takes_thread_ptr(ptr)
        \\define float @fragmentShader(i32 %__0) {
        \\  %__2 = call float @takes_thread_ptr(ptr @__anon_1)
        \\  ret float %__2
        \\}
        \\
    );
}

test "D5.11 atomicrmw, cmpxchg, load/store atomic; fence is refused" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();
    const text = try testRoundTrip(gpa, test_header ++
        \\@s = private unnamed_addr addrspace(3) global i32 0, align 4
        \\
        \\define i32 @fragmentShader(ptr addrspace(1) %__0, i32 %__1, ptr addrspace(1) %__2) {
        \\  %a = atomicrmw add ptr addrspace(1) %__0, i32 1 monotonic, align 4
        \\  %b = atomicrmw umax ptr addrspace(1) %__0, i32 %__1 seq_cst, align 4
        \\  %c = atomicrmw xchg ptr addrspace(3) @s, i32 %__1 syncscope("singlethread") monotonic, align 4
        \\  %d = atomicrmw fadd ptr addrspace(1) %__2, float 1.000000e+00 monotonic, align 4
        \\  %e = atomicrmw volatile sub ptr addrspace(1) %__0, i32 %__1 acq_rel, align 4
        \\  %f = cmpxchg weak ptr addrspace(1) %__0, i32 0, i32 %__1 monotonic monotonic, align 4
        \\  %g = cmpxchg ptr addrspace(1) %__0, i32 %__1, i32 0 acquire monotonic, align 4
        \\  %h = extractvalue { i32, i1 } %f, 0
        \\  %i = extractvalue { i32, i1 } %g, 1
        \\  %l = load atomic i32, ptr addrspace(1) %__0 monotonic, align 4
        \\  store atomic i32 %l, ptr addrspace(3) @s release, align 4
        \\  %j = zext i1 %i to i32
        \\  %k = add i32 %a, %b
        \\  %m = add i32 %k, %c
        \\  %n = add i32 %m, %e
        \\  %o = add i32 %n, %h
        \\  %p = add i32 %o, %j
        \\  %q = fptoui float %d to i32
        \\  %r = add i32 %p, %q
        \\  ret i32 %r
        \\}
        \\
    );
    try std.testing.expect(has(text, "atomicrmw add ptr addrspace(1)"));
    try std.testing.expect(has(text, "atomicrmw umax"));
    try std.testing.expect(has(text, "seq_cst"));
    try std.testing.expect(has(text, "atomicrmw xchg ptr addrspace(3) @s"));
    try std.testing.expect(has(text, "atomicrmw fadd"));
    try std.testing.expect(has(text, "atomicrmw volatile sub"));
    try std.testing.expect(has(text, "cmpxchg weak ptr addrspace(1)"));
    try std.testing.expect(has(text, "cmpxchg ptr addrspace(1)"));
    try std.testing.expect(has(text, "load atomic i32"));
    try std.testing.expect(has(text, "store atomic i32"));

    try testFails(gpa, test_header ++
        \\define void @fragmentShader(ptr addrspace(1) %__0) {
        \\  fence seq_cst
        \\  ret void
        \\}
        \\
    );
}

test "D5.12 intrinsic table: renames declare the air.* name only, powi exponent, reduce expansion, bool-vector bitcast" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();
    const text = try testRoundTrip(gpa, test_header ++
        \\declare float @llvm.sin.f32(float)
        \\declare <4 x float> @llvm.cos.v4f32(<4 x float>)
        \\declare half @llvm.tan.f16(half)
        \\declare float @llvm.powi.f32.i32(float, i32)
        \\declare float @llvm.nearbyint.f32(float)
        \\declare float @llvm.maximum.f32(float, float)
        \\declare float @llvm.sqrt.f32(float)
        \\declare float @llvm.vector.reduce.fadd.v4f32(float, <4 x float>)
        \\declare float @llvm.vector.reduce.fmax.v3f32(<3 x float>)
        \\declare i32 @llvm.vector.reduce.umin.v4i32(<4 x i32>)
        \\declare i1 @llvm.vector.reduce.or.v4i1(<4 x i1>)
        \\
        \\define <4 x float> @fragmentShader(<4 x float> %__0, <3 x float> %__1, <2 x float> %__2, i32 %__3) {
        \\  %__5 = extractelement <2 x float> %__2, i64 0
        \\  %__6 = tail call float @llvm.sin.f32(float %__5)
        \\  %__7 = tail call <4 x float> @llvm.cos.v4f32(<4 x float> %__0)
        \\  %__8 = fptrunc float %__6 to half
        \\  %__9 = tail call half @llvm.tan.f16(half %__8)
        \\  %__10 = tail call float @llvm.powi.f32.i32(float %__5, i32 %__3)
        \\  %__11 = tail call float @llvm.nearbyint.f32(float %__10)
        \\  %__12 = tail call float @llvm.maximum.f32(float %__11, float %__6)
        \\  %__13 = tail call float @llvm.sqrt.f32(float %__12)
        \\  %__14 = fmul <4 x float> %__0, %__0
        \\  %__15 = tail call float @llvm.vector.reduce.fadd.v4f32(float -0.000000e+00, <4 x float> %__14)
        \\  %__16 = tail call float @llvm.vector.reduce.fmax.v3f32(<3 x float> %__1)
        \\  %__17 = fptoui <4 x float> %__0 to <4 x i32>
        \\  %__18 = tail call i32 @llvm.vector.reduce.umin.v4i32(<4 x i32> %__17)
        \\  %__19 = fcmp olt <4 x float> %__0, zeroinitializer
        \\  %__20 = tail call i1 @llvm.vector.reduce.or.v4i1(<4 x i1> %__19)
        \\  %__21 = bitcast <4 x i1> %__19 to i4
        \\  %__22 = icmp ne i4 %__21, 0
        \\  %__23 = and i1 %__20, %__22
        \\  %__24 = uitofp i1 %__23 to float
        \\  %__25 = uitofp i32 %__18 to float
        \\  %__26 = fadd float %__15, %__16
        \\  %__27 = fadd float %__26, %__13
        \\  %__28 = fadd float %__27, %__24
        \\  %__29 = fadd float %__28, %__25
        \\  %__30 = fpext half %__9 to float
        \\  %__31 = fadd float %__29, %__30
        \\  %__32 = insertelement <4 x float> %__7, float %__31, i64 0
        \\  %__33 = fptoui float %__5 to i4
        \\  %__34 = bitcast i4 %__33 to <4 x i1>
        \\  %__35 = select <4 x i1> %__34, <4 x float> %__32, <4 x float> zeroinitializer
        \\  ret <4 x float> %__35
        \\}
        \\
    );
    // Renamed: the air.* names are declared and called, the llvm.* ones are gone.
    try std.testing.expect(has(text, "declare float @air.fast_sin.f32("));
    try std.testing.expect(has(text, "call float @air.fast_sin.f32(float"));
    try std.testing.expect(has(text, "declare <4 x float> @air.fast_cos.v4f32("));
    try std.testing.expect(has(text, "declare half @air.fast_tan.f16("));
    try std.testing.expect(has(text, "declare float @air.fast_pow.f32(float %0, float %1)"));
    try std.testing.expect(has(text, "sitofp i32 %3 to float"));
    try std.testing.expect(has(text, "call float @llvm.rint.f32("));
    try std.testing.expect(has(text, "call float @air.fast_fmax.f32("));
    try std.testing.expect(!has(text, "llvm.sin."));
    try std.testing.expect(!has(text, "llvm.cos."));
    try std.testing.expect(!has(text, "llvm.tan."));
    try std.testing.expect(!has(text, "llvm.powi."));
    try std.testing.expect(!has(text, "llvm.nearbyint."));
    try std.testing.expect(!has(text, "llvm.maximum."));
    // Accepted unchanged.
    try std.testing.expect(has(text, "call float @llvm.sqrt.f32("));
    // Reduces expanded: no llvm.vector.reduce survives.
    try std.testing.expect(!has(text, "vector.reduce"));
    try std.testing.expect(has(text, "call float @llvm.maxnum.f32("));
    try std.testing.expect(has(text, "icmp ult i32"));
    try std.testing.expect(has(text, "call i1 @air.any.v4i1(<4 x i1>"));
    // Bool-vector bitcast packed lane by lane.
    try std.testing.expect(!has(text, "bitcast"));
    try std.testing.expect(has(text, "zext <4 x i1>"));
    try std.testing.expect(has(text, "shl i32"));
    try std.testing.expect(has(text, "to i4"));
    // Integer to bool-vector bitcast unpacked with vector shifts.
    try std.testing.expect(has(text, "zext i4 %"));
    try std.testing.expect(has(text, "lshr <4 x i32> %"));
    try std.testing.expect(has(text, "<i32 0, i32 1, i32 2, i32 3>"));
    try std.testing.expect(has(text, "icmp ne <4 x i32> %"));
    try std.testing.expect(has(text, "select <4 x i1> %"));
}

test "D5.12c texture intrinsics: Apple's signature replaces Zig's, bare-vector calls get extractvalue, get_read_sampler returns a constant-space sampler" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();
    const text = try testRoundTrip(gpa, test_header ++
        \\@sampler.state = private unnamed_addr constant [2 x i64] [i64 34901797601020489, i64 0], align 8
        \\declare <4 x float> @air.sample_texture_2d.v4f32(ptr addrspace(1) readonly, ptr nonnull readonly align 1, <2 x float>, i1 zeroext, <2 x i32>, i1 zeroext, float, float, i32)
        \\declare ptr @air.get_read_sampler()
        \\declare <4 x float> @air.read_texture_2d.v4f32(ptr addrspace(1) readonly, ptr nonnull readonly align 1, <2 x i32>, <2 x i32>, i32, i32)
        \\declare i32 @air.get_width_texture_2d(ptr addrspace(1) readonly, i32)
        \\
        \\define <4 x float> @fragmentShader(<4 x float> %__0, <3 x float> %__1, <2 x float> %__2, ptr addrspace(1) %__3, ptr addrspace(2) %__4) {
        \\  %__6 = tail call <4 x float> @air.sample_texture_2d.v4f32(ptr addrspace(1) readonly %__3, ptr nonnull readonly align 1 @sampler.state, <2 x float> %__2, i1 zeroext true, <2 x i32> zeroinitializer, i1 zeroext false, float 0.000000e+00, float 0.000000e+00, i32 0)
        \\  %__7 = call { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) %__3, ptr addrspace(2) %__4, <2 x float> %__2, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.000000e+00, float 0.000000e+00, i32 0)
        \\  %__8 = extractvalue { <4 x float>, i8 } %__7, 0
        \\  %__9 = tail call ptr @air.get_read_sampler()
        \\  %__10 = fptoui <2 x float> %__2 to <2 x i32>
        \\  %__11 = tail call <4 x float> @air.read_texture_2d.v4f32(ptr addrspace(1) readonly %__3, ptr nonnull readonly align 1 %__9, <2 x i32> %__10, <2 x i32> zeroinitializer, i32 0, i32 1)
        \\  %__12 = tail call i32 @air.get_width_texture_2d(ptr addrspace(1) readonly %__3, i32 0)
        \\  %__13 = uitofp i32 %__12 to float
        \\  %__14 = fadd <4 x float> %__6, %__8
        \\  %__15 = fadd <4 x float> %__14, %__11
        \\  %__16 = insertelement <4 x float> %__15, float %__13, i64 3
        \\  ret <4 x float> %__16
        \\}
        \\
    );
    // Apple's signature replaces the one Zig can express: the colour comes
    // back paired with the residency byte and the sampler is a constant-space
    // pointer (research/_results/r03.txt ir_patterns 0, 9, 11).
    try std.testing.expect(has(text, "declare { <4 x float>, i8 } @air.sample_texture_2d.v4f32(ptr addrspace(1) %0, ptr addrspace(2) %1,"));
    try std.testing.expect(!has(text, "declare <4 x float> @air.sample_texture_2d"));
    try std.testing.expect(has(text, "declare ptr addrspace(2) @air.get_read_sampler()"));
    try std.testing.expect(has(text, "declare { <4 x float>, i8 } @air.read_texture_2d.v4f32(ptr addrspace(1) %0, ptr addrspace(2) %1,"));
    // The constexpr sampler word is relocated to the constant space (D1) and
    // handed to the intrinsic from there.
    try std.testing.expect(has(text, "@sampler.state = private addrspace(2) constant [2 x i64]"));
    try std.testing.expect(has(text, "@air.sample_texture_2d.v4f32(ptr addrspace(1) %3, ptr addrspace(2) @sampler.state,"));
    // Two Zig-shaped calls each gain one `extractvalue 0`; the call that asks
    // for the pair itself keeps its own and does not get a second.
    try std.testing.expect(std.mem.count(u8, text, "extractvalue { <4 x float>, i8 }") == 3);
    // Signatures Zig already spells the canonical way pass through untouched.
    try std.testing.expect(has(text, "call i32 @air.get_width_texture_2d(ptr addrspace(1) %3, i32 0)"));
}

test "D5.13 diagnostics: llvm.nvvm.*, addrspacecast, llvm.ctpop.i4, unexpandable llvm.vector.reduce" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();
    try testFails(gpa, test_header ++
        \\declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x()
        \\define i32 @fragmentShader(ptr addrspace(1) %__0) {
        \\  %__2 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
        \\  ret i32 %__2
        \\}
        \\
    );
    try testFails(gpa, test_header ++
        \\define ptr @fragmentShader(ptr addrspace(1) %__0) {
        \\  %__2 = addrspacecast ptr addrspace(1) %__0 to ptr
        \\  ret ptr %__2
        \\}
        \\
    );
    try testFails(gpa, test_header ++
        \\declare i4 @llvm.ctpop.i4(i4)
        \\define i32 @fragmentShader(i4 %__0) {
        \\  %__2 = tail call range(i4 0, 5) i4 @llvm.ctpop.i4(i4 %__0)
        \\  %__3 = zext i4 %__2 to i32
        \\  ret i32 %__3
        \\}
        \\
    );
    // A reduce op without a scalar expansion is refused, never passed
    // through (every llvm.vector.reduce.* crashes Metal's compiler).
    try testFails(gpa, test_header ++
        \\declare float @llvm.vector.reduce.fmaximum.v4f32(<4 x float>)
        \\define float @fragmentShader(<4 x float> %__0) {
        \\  %__2 = tail call float @llvm.vector.reduce.fmaximum.v4f32(<4 x float> %__0)
        \\  ret float %__2
        \\}
        \\
    );
}

test "D5.13 diagnostics: barriers in divergent control flow (jump-threaded barrier hazard)" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();
    const decls =
        \\@tg_counter = internal unnamed_addr addrspace(3) global i32 undef, align 4
        \\declare void @air.wg.barrier(i32, i32)
        \\declare void @air.simdgroup.barrier(i32, i32)
        \\declare i32 @air.atomic.local.add.u.i32(ptr addrspace(3) align 4, i32, i32, i32, i1)
        \\
    ;
    // The shape LLVM made of `if (tidx == 0) tg_counter = 0; barrier; ...`
    // (review-b/fable-hazard/hazard.shader.air.ll): the barriers sit in
    // both arms of the branch, neither of which every thread passes.
    try testFails(gpa, test_header ++ decls ++
        \\define void @fragmentShader(ptr addrspace(1) %__0, i32 %__1) {
        \\  %__3 = icmp eq i32 %__1, 0
        \\  br i1 %__3, label %__4, label %.critedge
        \\
        \\__4:
        \\  store i32 0, ptr addrspace(3) @tg_counter, align 4
        \\  tail call void @air.wg.barrier(i32 2, i32 1)
        \\  %__5 = tail call i32 @air.atomic.local.add.u.i32(ptr addrspace(3) align 4 @tg_counter, i32 1, i32 0, i32 1, i1 true)
        \\  br label %__6
        \\
        \\.critedge:
        \\  tail call void @air.wg.barrier(i32 2, i32 1)
        \\  %__7 = tail call i32 @air.atomic.local.add.u.i32(ptr addrspace(3) align 4 @tg_counter, i32 1, i32 0, i32 1, i1 true)
        \\  br label %__6
        \\
        \\__6:
        \\  ret void
        \\}
        \\
    );
    // A simdgroup barrier behind a divergent early return is refused too.
    try testFails(gpa, test_header ++ decls ++
        \\define void @fragmentShader(ptr addrspace(1) %__0, i32 %__1) {
        \\  %__3 = icmp ugt i32 %__1, 7
        \\  br i1 %__3, label %__4, label %__5
        \\
        \\__4:
        \\  ret void
        \\
        \\__5:
        \\  tail call void @air.simdgroup.barrier(i32 2, i32 4)
        \\  ret void
        \\}
        \\
    );
    // Accepted: a barrier in the entry block, in the join block after a
    // diamond, and in the body of a rotated (do-while) loop: every thread
    // that finishes the kernel passes through each of those blocks.
    const text = try testRoundTrip(gpa, test_header ++ decls ++
        \\define void @fragmentShader(ptr addrspace(1) %__0, i32 %__1) {
        \\  store i32 0, ptr addrspace(3) @tg_counter, align 4
        \\  tail call void @air.wg.barrier(i32 2, i32 1)
        \\  %__3 = icmp eq i32 %__1, 0
        \\  br i1 %__3, label %__4, label %__5
        \\
        \\__4:
        \\  %__6 = tail call i32 @air.atomic.local.add.u.i32(ptr addrspace(3) align 4 @tg_counter, i32 1, i32 0, i32 1, i1 true)
        \\  br label %__5
        \\
        \\__5:
        \\  tail call void @air.wg.barrier(i32 2, i32 1)
        \\  br label %loop
        \\
        \\loop:
        \\  %i = phi i32 [ 0, %__5 ], [ %next, %loop ]
        \\  tail call void @air.simdgroup.barrier(i32 2, i32 4)
        \\  %next = add i32 %i, 1
        \\  %done = icmp eq i32 %next, 4
        \\  br i1 %done, label %exit, label %loop
        \\
        \\exit:
        \\  ret void
        \\}
        \\
    );
    try std.testing.expect(has(text, "@air.wg.barrier"));
    try std.testing.expect(has(text, "@air.simdgroup.barrier"));
}

test "D5.13 diagnostics: a barrier hidden in a noinline helper is checked at the helper's call site" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();
    // review-b/r4/helper/zig-out/bin/shader.air.ll lines 130-177: the
    // kernel's own blocks hold no barrier, only `syncTg()` calls that LLVM
    // threaded into both arms of `if (tidx == 0)`; the helper's barrier
    // trivially post-dominates the helper's entry.
    const decls =
        \\@tg_counter2 = internal unnamed_addr addrspace(3) global i32 undef, align 4
        \\declare void @air.wg.barrier(i32, i32)
        \\declare i32 @air.atomic.local.add.u.i32(ptr addrspace(3) align 4, i32, i32, i32, i1)
        \\
        \\define private fastcc void @my_shader.syncTg() {
        \\  tail call void @air.wg.barrier(i32 2, i32 1)
        \\  ret void
        \\}
        \\
    ;
    try testFails(gpa, test_header ++ decls ++
        \\define void @fragmentShader(ptr addrspace(1) %__0, i32 %__1) {
        \\  %__3 = icmp eq i32 %__1, 0
        \\  br i1 %__3, label %__4, label %.critedge
        \\
        \\__4:
        \\  store i32 0, ptr addrspace(3) @tg_counter2, align 4
        \\  tail call fastcc void @my_shader.syncTg()
        \\  %__5 = tail call i32 @air.atomic.local.add.u.i32(ptr addrspace(3) align 4 @tg_counter2, i32 1, i32 0, i32 1, i1 true)
        \\  tail call fastcc void @my_shader.syncTg()
        \\  br label %__6
        \\
        \\.critedge:
        \\  tail call fastcc void @my_shader.syncTg()
        \\  %__7 = tail call i32 @air.atomic.local.add.u.i32(ptr addrspace(3) align 4 @tg_counter2, i32 1, i32 0, i32 1, i1 true)
        \\  tail call fastcc void @my_shader.syncTg()
        \\  br label %__6
        \\
        \\__6:
        \\  ret void
        \\}
        \\
    );
    // The same helper in uniform control flow is fine, and stays a call.
    const text = try testRoundTrip(gpa, test_header ++ decls ++
        \\define void @fragmentShader(ptr addrspace(1) %__0, i32 %__1) {
        \\  store i32 0, ptr addrspace(3) @tg_counter2, align 4
        \\  tail call fastcc void @my_shader.syncTg()
        \\  %__3 = icmp eq i32 %__1, 0
        \\  br i1 %__3, label %__4, label %__6
        \\
        \\__4:
        \\  %__5 = tail call i32 @air.atomic.local.add.u.i32(ptr addrspace(3) align 4 @tg_counter2, i32 1, i32 0, i32 1, i1 true)
        \\  br label %__6
        \\
        \\__6:
        \\  tail call fastcc void @my_shader.syncTg()
        \\  ret void
        \\}
        \\
    );
    try std.testing.expect(has(text, "call fastcc void @my_shader.syncTg()"));
}

test "D5.13 diagnostics: SIMD-group collectives are convergent too (jump-threaded simd_sum)" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();
    // review-b/r4/helper/zig-out/bin/shader.air.ll lines 79-103, from
    // `if (lane == 0) out[sgid] = -1; s = simdSum(in[gid]); if (lane == 0)
    // out[sgid] = s;`: each arm sums only its own lanes (out = 1 and 33
    // instead of 528 and 1552).
    const decls =
        \\declare float @air.simd_sum.f32(float)
        \\
    ;
    try testFails(gpa, test_header ++ decls ++
        \\define void @fragmentShader(ptr addrspace(1) %__0, i32 %__1) {
        \\  %__6 = icmp eq i32 %__1, 0
        \\  br i1 %__6, label %__7, label %.critedge
        \\
        \\__7:
        \\  store float -1.0, ptr addrspace(1) %__0, align 4
        \\  %__13 = tail call float @air.simd_sum.f32(float 2.0)
        \\  store float %__13, ptr addrspace(1) %__0, align 4
        \\  br label %__9
        \\
        \\.critedge:
        \\  %__17 = tail call float @air.simd_sum.f32(float 2.0)
        \\  br label %__9
        \\
        \\__9:
        \\  ret void
        \\}
        \\
    );
    // The straight-line form (sumKernel, simdOpsKernel) is accepted.
    const text = try testRoundTrip(gpa, test_header ++ decls ++
        \\define void @fragmentShader(ptr addrspace(1) %__0, i32 %__1) {
        \\  %__13 = tail call float @air.simd_sum.f32(float 2.0)
        \\  %__6 = icmp eq i32 %__1, 0
        \\  br i1 %__6, label %__7, label %__9
        \\
        \\__7:
        \\  store float %__13, ptr addrspace(1) %__0, align 4
        \\  br label %__9
        \\
        \\__9:
        \\  ret void
        \\}
        \\
    );
    try std.testing.expect(has(text, "@air.simd_sum.f32"));
}

test "D5.13 uniformity: a barrier in a loop bounded by threads_per_threadgroup or a constant buffer is accepted" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();
    const decls =
        \\@red = internal unnamed_addr addrspace(3) global [64 x float] undef, align 4
        \\declare void @air.wg.barrier(i32, i32)
        \\
    ;
    // review-b/r4/reduce/reduce.nvptx.ll: LLVM guarded the runtime-bounded
    // tree reduction (`br i1 %.not7, label %._crit_edge, label %.lr.ph`),
    // the barrier sits in the latch, and the `if (tidx < s)` branch inside
    // the loop rejoins before it. Every branch the barrier depends on is
    // uniform (tg_size), so it is accepted.
    const text = try testRoundTripAs(gpa, test_header ++ decls ++
        \\define void @reduceKernel(ptr addrspace(1) %__0, ptr addrspace(1) %__1, i32 %__2, i32 %__3, ptr addrspace(2) %__4) {
        \\  %__6 = zext i32 %__2 to i64
        \\  %__7 = getelementptr inbounds [4 x i8], ptr addrspace(3) @red, i64 %__6
        \\  %__11 = getelementptr inbounds [4 x i8], ptr addrspace(1) %__0, i64 %__6
        \\  %__12 = load float, ptr addrspace(1) %__11, align 4
        \\  store float %__12, ptr addrspace(3) %__7, align 4
        \\  tail call void @air.wg.barrier(i32 2, i32 1)
        \\  %.06 = lshr i32 %__3, 1
        \\  %.not7 = icmp eq i32 %.06, 0
        \\  br i1 %.not7, label %._crit_edge, label %.lr.ph
        \\
        \\.lr.ph:
        \\  %.08 = phi i32 [ %.0, %__15 ], [ %.06, %__5 ]
        \\  %__13 = icmp ult i32 %__2, %.08
        \\  br i1 %__13, label %__16, label %__15
        \\
        \\._crit_edge:
        \\  %__14 = icmp eq i32 %__2, 0
        \\  br i1 %__14, label %__24, label %__23
        \\
        \\__15:
        \\  tail call void @air.wg.barrier(i32 2, i32 1)
        \\  %.0 = lshr i32 %.08, 1
        \\  %.not = icmp eq i32 %.0, 0
        \\  br i1 %.not, label %._crit_edge, label %.lr.ph
        \\
        \\__16:
        \\  %__17 = add i32 %.08, %__2
        \\  %__18 = zext i32 %__17 to i64
        \\  %__19 = getelementptr inbounds [4 x i8], ptr addrspace(3) @red, i64 %__18
        \\  %__20 = load float, ptr addrspace(3) %__7, align 4
        \\  %__21 = load float, ptr addrspace(3) %__19, align 4
        \\  %__22 = fadd float %__20, %__21
        \\  store float %__22, ptr addrspace(3) %__7, align 4
        \\  br label %__15
        \\
        \\__23:
        \\  ret void
        \\
        \\__24:
        \\  %__27 = load float, ptr addrspace(3) @red, align 4
        \\  store float %__27, ptr addrspace(1) %__1, align 4
        \\  br label %__23
        \\}
        \\
    , "reduceKernel", kernel_fm);
    try std.testing.expect(has(text, "@air.wg.barrier"));
    // A uniform early return (constant-buffer field) before a barrier is
    // fine; the same shape on the per-thread id is not.
    const uniform_exit = try testRoundTripAs(gpa, test_header ++ decls ++
        \\define void @reduceKernel(ptr addrspace(1) %__0, ptr addrspace(1) %__1, i32 %__2, i32 %__3, ptr addrspace(2) %__4) {
        \\  %__6 = load i32, ptr addrspace(2) %__4, align 4
        \\  %__7 = icmp eq i32 %__6, 0
        \\  br i1 %__7, label %__8, label %__9
        \\
        \\__8:
        \\  ret void
        \\
        \\__9:
        \\  tail call void @air.wg.barrier(i32 2, i32 1)
        \\  ret void
        \\}
        \\
    , "reduceKernel", kernel_fm);
    try std.testing.expect(has(uniform_exit, "@air.wg.barrier"));
    try testFailsAs(gpa, test_header ++ decls ++
        \\define void @reduceKernel(ptr addrspace(1) %__0, ptr addrspace(1) %__1, i32 %__2, i32 %__3, ptr addrspace(2) %__4) {
        \\  %__7 = icmp ugt i32 %__2, %__3
        \\  br i1 %__7, label %__8, label %__9
        \\
        \\__8:
        \\  ret void
        \\
        \\__9:
        \\  tail call void @air.wg.barrier(i32 2, i32 1)
        \\  ret void
        \\}
        \\
    , "reduceKernel", kernel_fm);
    // A loop some threads leave early (`if (tidx == i) break;`) is
    // temporally divergent: the barrier inside is refused.
    try testFailsAs(gpa, test_header ++ decls ++
        \\define void @reduceKernel(ptr addrspace(1) %__0, ptr addrspace(1) %__1, i32 %__2, i32 %__3, ptr addrspace(2) %__4) {
        \\  br label %loop
        \\
        \\loop:
        \\  %i = phi i32 [ 0, %__5 ], [ %next, %latch ]
        \\  %stop = icmp eq i32 %i, %__2
        \\  br i1 %stop, label %exit, label %body
        \\
        \\body:
        \\  tail call void @air.wg.barrier(i32 2, i32 1)
        \\  br label %latch
        \\
        \\latch:
        \\  %next = add i32 %i, 1
        \\  %done = icmp eq i32 %next, %__3
        \\  br i1 %done, label %exit, label %loop
        \\
        \\exit:
        \\  ret void
        \\}
        \\
    , "reduceKernel", kernel_fm);
}

test "D5.13 diagnostics: SIMD-group calls under a SIMD-group-uniform branch are accepted, threadgroup barriers there and SIMD-group calls under a per-thread branch are not" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();
    const decls =
        \\@partial = internal unnamed_addr addrspace(3) global [32 x float] undef, align 4
        \\declare void @air.wg.barrier(i32, i32)
        \\declare float @air.simd_sum.f32(float)
        \\
    ;
    // review2/v1/.zig-cache/o/*/shader.ll @twoStageKernel: the canonical
    // two-stage reduction. The second `simd_sum` sits under
    // `if (sgid == 0)`, uniform within every SIMD-group
    // (review2/v1_nocheck/check.log: dispatches correctly).
    const text = try testRoundTripAs(gpa, test_header ++ decls ++
        \\define void @twoStageKernel(ptr addrspace(1) %__0, ptr addrspace(1) %__1, i32 %__2, i32 %__3, i32 %__4, i32 %__5, i32 %__6) {
        \\  %__8 = zext i32 %__2 to i64
        \\  %__9 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %__0, i64 %__8
        \\  %__10 = load float, ptr addrspace(1) %__9, align 4
        \\  %__11 = tail call float @air.simd_sum.f32(float %__10)
        \\  %__12 = icmp eq i32 %__3, 0
        \\  br i1 %__12, label %__15, label %__13
        \\
        \\__13:
        \\  tail call void @air.wg.barrier(i32 2, i32 1)
        \\  %__14 = icmp eq i32 %__4, 0
        \\  br i1 %__14, label %__19, label %__18
        \\
        \\__15:
        \\  %__16 = zext i32 %__4 to i64
        \\  %__17 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @partial, i64 %__16
        \\  store float %__11, ptr addrspace(3) %__17, align 4
        \\  br label %__13
        \\
        \\__18:
        \\  ret void
        \\
        \\__19:
        \\  %__20 = icmp ult i32 %__3, %__5
        \\  br i1 %__20, label %__24, label %__21
        \\
        \\__21:
        \\  %__22 = phi float [ %__27, %__24 ], [ 0.000000e+00, %__19 ]
        \\  %__23 = tail call float @air.simd_sum.f32(float %__22)
        \\  br i1 %__12, label %__28, label %__18
        \\
        \\__24:
        \\  %__25 = zext i32 %__3 to i64
        \\  %__26 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @partial, i64 %__25
        \\  %__27 = load float, ptr addrspace(3) %__26, align 4
        \\  br label %__21
        \\
        \\__28:
        \\  %__29 = zext i32 %__6 to i64
        \\  %__30 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %__1, i64 %__29
        \\  store float %__23, ptr addrspace(1) %__30, align 4
        \\  br label %__18
        \\}
        \\
    , "twoStageKernel", simd_kernel_fm);
    try std.testing.expect(has(text, "@air.simd_sum.f32"));
    // A threadgroup barrier under `if (sgid == 0)` synchronises a subset
    // of the threadgroup: refused.
    try testFailsAs(gpa, test_header ++ decls ++
        \\define void @twoStageKernel(ptr addrspace(1) %__0, ptr addrspace(1) %__1, i32 %__2, i32 %__3, i32 %__4, i32 %__5, i32 %__6) {
        \\  %__8 = icmp eq i32 %__4, 0
        \\  br i1 %__8, label %__9, label %__10
        \\
        \\__9:
        \\  tail call void @air.wg.barrier(i32 2, i32 1)
        \\  br label %__10
        \\
        \\__10:
        \\  ret void
        \\}
        \\
    , "twoStageKernel", simd_kernel_fm);
    // A `simd_sum` under `if (lane == 0)` sums one lane: refused.
    try testFailsAs(gpa, test_header ++ decls ++
        \\define void @twoStageKernel(ptr addrspace(1) %__0, ptr addrspace(1) %__1, i32 %__2, i32 %__3, i32 %__4, i32 %__5, i32 %__6) {
        \\  %__8 = icmp eq i32 %__3, 0
        \\  br i1 %__8, label %__9, label %__10
        \\
        \\__9:
        \\  %__11 = tail call float @air.simd_sum.f32(float 1.0)
        \\  br label %__10
        \\
        \\__10:
        \\  ret void
        \\}
        \\
    , "twoStageKernel", simd_kernel_fm);
}

test "D5.13 diagnostics: a trapping safety check (ReleaseSafe bounds check) before a barrier is accepted" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();
    // review2/v3/.zig-cache/o/*/shader.ll @reverseKernel at ReleaseSafe:
    // `scratch[tidx]` is bounds-checked and the failing path is
    // `llvm.trap(); unreachable`; the barrier after it must stay accepted
    // (review2/v3_nocheck/check.log: every dispatch correct).
    const decls =
        \\@red = internal unnamed_addr addrspace(3) global [64 x float] undef, align 4
        \\declare void @air.wg.barrier(i32, i32)
        \\declare void @llvm.trap()
        \\
    ;
    const text = try testRoundTripAs(gpa, test_header ++ decls ++
        \\define void @reduceKernel(ptr addrspace(1) %__0, ptr addrspace(1) %__1, i32 %__2, i32 %__3, ptr addrspace(2) %__4) {
        \\  %__7 = icmp ult i32 %__2, 64
        \\  br i1 %__7, label %__8, label %__13
        \\
        \\__8:
        \\  %__9 = zext i32 %__2 to i64
        \\  %__10 = getelementptr inbounds [4 x i8], ptr addrspace(3) @red, i64 %__9
        \\  store float 1.0, ptr addrspace(3) %__10, align 4
        \\  tail call void @air.wg.barrier(i32 2, i32 1)
        \\  ret void
        \\
        \\__13:
        \\  tail call void @llvm.trap()
        \\  unreachable
        \\}
        \\
    , "reduceKernel", kernel_fm);
    try std.testing.expect(has(text, "@air.wg.barrier"));
}

test "D5.13 diagnostics: a helper returning constants from a per-thread `ret` makes its caller's branch divergent" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();
    // review2/v2/edited/shader.ll: `ticket()` returns 1 to the thread that
    // won the atomic and 2 to the others; the caller's barriers were
    // jump-threaded into both arms of `if (k == 1)` and the old rule
    // accepted them (`FAIL dispatch multiRetKernel: 8 mismatches`).
    const decls =
        \\@tg_count2 = internal unnamed_addr addrspace(3) global i32 undef, align 4
        \\declare void @air.wg.barrier(i32, i32)
        \\declare i32 @air.atomic.global.add.u.i32(ptr addrspace(1), i32, i32, i32, i1)
        \\
        \\define private fastcc i32 @my_shader.ticket(ptr addrspace(1) %__0, ptr addrspace(1) %__1) {
        \\  %__3 = tail call i32 @air.atomic.global.add.u.i32(ptr addrspace(1) %__0, i32 1, i32 0, i32 2, i1 true)
        \\  %__4 = icmp eq i32 %__3, 0
        \\  br i1 %__4, label %__5, label %common.ret
        \\
        \\common.ret:
        \\  ret i32 2
        \\
        \\__5:
        \\  store i32 7, ptr addrspace(1) %__1, align 4
        \\  ret i32 1
        \\}
        \\
    ;
    try testFailsAs(gpa, test_header ++ decls ++
        \\define void @reduceKernel(ptr addrspace(1) %__0, ptr addrspace(1) %__1, i32 %__2, i32 %__3, ptr addrspace(2) %__4) {
        \\  %__7 = tail call fastcc i32 @my_shader.ticket(ptr addrspace(1) %__0, ptr addrspace(1) %__1)
        \\  %__8 = icmp eq i32 %__7, 1
        \\  br i1 %__8, label %__9, label %.critedge
        \\
        \\__9:
        \\  store i32 0, ptr addrspace(3) @tg_count2, align 4
        \\  tail call void @air.wg.barrier(i32 2, i32 1)
        \\  br label %__14
        \\
        \\.critedge:
        \\  tail call void @air.wg.barrier(i32 2, i32 1)
        \\  br label %__14
        \\
        \\__14:
        \\  ret void
        \\}
        \\
    , "reduceKernel", kernel_fm);
}

test "D5.14 operand bundles after call arguments are ignored" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();
    const text = try testRoundTrip(gpa, test_header ++
        \\declare void @llvm.assume(i1 noundef)
        \\declare void @llvm.trap()
        \\
        \\define i32 @fragmentShader(i32 %__0) {
        \\  %.not = icmp eq i32 %__0, 0
        \\  br i1 %.not, label %__3, label %__4
        \\
        \\__3:                                                ; preds = %__1
        \\  call void @llvm.assume(i1 true) [ "cold"() ]
        \\  tail call void @llvm.trap()
        \\  unreachable
        \\
        \\__4:                                                ; preds = %__1
        \\  ret i32 %__0
        \\}
        \\
    );
    try std.testing.expect(has(text, "call void @llvm.assume(i1 true)"));
    try std.testing.expect(has(text, "@llvm.trap()"));
    try std.testing.expect(has(text, "unreachable"));
}

test "literals: negative integers, i1 true/false, hex floats, nested packed struct initialisers" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();
    const text = try testRoundTrip(gpa, test_header ++
        \\@feat = private unnamed_addr constant { [2 x i64], i8 } { [2 x i64] [i64 -9223372036854775808, i64 -1], i8 31 }, align 8
        \\@pk = private unnamed_addr constant <{ float, <{ i32, i8 }>, [3 x i8] }> <{ float 0x3F70101020000000, <{ i32, i8 }> <{ i32 -1, i8 3 }>, [3 x i8] undef }>, align 4
        \\@hv = private unnamed_addr constant <4 x half> <half 0xH3C00, half 0xH0000, half 0xHBC00, half 0xH3800>, align 8
        \\
        \\define i64 @fragmentShader(i32 %__0, i1 %__1) {
        \\  %__3 = select i1 %__1, i1 true, i1 false
        \\  %__4 = xor i32 %__0, -1
        \\  %__5 = icmp ugt i32 %__4, -2147483648
        \\  %__6 = and i1 %__3, %__5
        \\  %__7 = zext i1 %__6 to i64
        \\  %__8 = load i64, ptr @feat, align 8
        \\  %__9 = add i64 %__8, %__7
        \\  %__10 = load float, ptr @pk, align 4
        \\  %__11 = fptosi float %__10 to i64
        \\  %__12 = add i64 %__9, %__11
        \\  %__13 = load <4 x half>, ptr @hv, align 8
        \\  %__14 = extractelement <4 x half> %__13, i32 2
        \\  %__15 = fptosi half %__14 to i64
        \\  %__16 = add i64 %__12, %__15
        \\  ret i64 %__16
        \\}
        \\
    );
    try std.testing.expect(has(text, "i64 -9223372036854775808"));
    try std.testing.expect(has(text, "0x3F70101020000000"));
    try std.testing.expect(has(text, "0xHBC00"));
    try std.testing.expect(has(text, "<{ i32 -1, i8 3 }>"));
}

test "D1 aggregate constants holding pointers to relocated constants are refused, not asserted" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();
    // `const tables = [2]*const [4]f32{ &a, &b }` at ReleaseFast.
    try testFails(gpa, test_header ++
        \\@a = private unnamed_addr constant [4 x float] [float 1.000000e+00, float 2.000000e+00, float 3.000000e+00, float 4.000000e+00], align 4
        \\@b = private unnamed_addr constant [4 x float] [float 5.000000e+00, float 6.000000e+00, float 7.000000e+00, float 8.000000e+00], align 4
        \\@tab = private unnamed_addr constant [2 x ptr] [ptr @a, ptr @b], align 8
        \\define float @fragmentShader(i32 %__0) {
        \\  %__2 = and i32 %__0, 1
        \\  %__3 = zext i32 %__2 to i64
        \\  %__4 = getelementptr inbounds [8 x i8], ptr @tab, i64 %__3
        \\  %__5 = load ptr, ptr %__4, align 8
        \\  %__6 = load float, ptr %__5, align 4
        \\  ret float %__6
        \\}
        \\
    );
    // A comptime slice table: `[2][]const u8{ "ab", "cde" }`.
    try testFails(gpa, test_header ++
        \\@s1 = private unnamed_addr constant [3 x i8] c"ab\00", align 1
        \\@s2 = private unnamed_addr constant [4 x i8] c"cde\00", align 1
        \\@names = private unnamed_addr constant [2 x { ptr, i64 }] [{ ptr, i64 } { ptr @s1, i64 2 }, { ptr, i64 } { ptr @s2, i64 3 }], align 8
        \\define i32 @fragmentShader(i32 %__0) {
        \\  %__2 = and i32 %__0, 1
        \\  %__3 = zext i32 %__2 to i64
        \\  %__4 = getelementptr inbounds [16 x i8], ptr @names, i64 %__3
        \\  %.unpack = load ptr, ptr %__4, align 8
        \\  %__5 = load i8, ptr %.unpack, align 1
        \\  %__6 = zext i8 %__5 to i32
        \\  ret i32 %__6
        \\}
        \\
    );
    // A single slice constant `{ ptr, i64 }` pointing at a relocated string.
    try testFails(gpa, test_header ++
        \\@s = private unnamed_addr constant [2 x i8] c"hi", align 1
        \\@tbl = private unnamed_addr constant { ptr, i64 } { ptr @s, i64 2 }, align 8
        \\define i32 @fragmentShader(i32 %__0) {
        \\  %__2 = load ptr, ptr @tbl, align 8
        \\  %__3 = load i8, ptr %__2, align 1
        \\  %__4 = zext i8 %__3 to i32
        \\  ret i32 %__4
        \\}
        \\
    );
}

test "D1 globals are defined on first reference: an untouched pointer table cannot break the module" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();
    // The same [2 x ptr] / [2 x { ptr, i64 }] tables as above, but only the
    // vertex stage (not in this module) would use them: the fragment entry
    // reads @a directly, so the module assembles and carries only @a.
    const text = try testRoundTrip(gpa, test_header ++
        \\@a = private unnamed_addr constant [4 x float] [float 1.000000e+00, float 2.000000e+00, float 3.000000e+00, float 4.000000e+00], align 4
        \\@b = private unnamed_addr constant [4 x float] [float 5.000000e+00, float 6.000000e+00, float 7.000000e+00, float 8.000000e+00], align 4
        \\@tab = private unnamed_addr constant [2 x ptr] [ptr @a, ptr @b], align 8
        \\@s1 = private unnamed_addr constant [3 x i8] c"ab\00", align 1
        \\@names = private unnamed_addr constant [1 x { ptr, i64 }] [{ ptr, i64 } { ptr @s1, i64 2 }], align 8
        \\@unused_tile = private unnamed_addr addrspace(3) global [64 x float] undef, align 4
        \\define float @fragmentShader(i32 %__0) {
        \\  %__2 = and i32 %__0, 3
        \\  %__3 = zext i32 %__2 to i64
        \\  %__4 = getelementptr inbounds [4 x i8], ptr @a, i64 %__3
        \\  %__5 = load float, ptr %__4, align 4
        \\  ret float %__5
        \\}
        \\
    );
    try std.testing.expect(has(text, "@a = private unnamed_addr addrspace(2) constant [4 x float]") or has(text, "@a = private addrspace(2) constant [4 x float]"));
    try std.testing.expect(has(text, "getelementptr inbounds [4 x i8], ptr addrspace(2) @a"));
    try std.testing.expect(!has(text, "@b ="));
    try std.testing.expect(!has(text, "@tab"));
    try std.testing.expect(!has(text, "@names"));
    try std.testing.expect(!has(text, "@s1"));
    try std.testing.expect(!has(text, "@unused_tile"));

    // A global referenced only through another global's initialiser is
    // pulled in transitively (gep constant expression into a string).
    const text2 = try testRoundTrip(gpa, test_header ++
        \\@str = private unnamed_addr constant [8 x i8] c"abcdefgh", align 1
        \\@lut = private unnamed_addr constant [4 x float] [float 1.000000e+00, float 2.000000e+00, float 3.000000e+00, float 4.000000e+00], align 4
        \\define float @fragmentShader(i32 %__0) {
        \\  %__2 = load i8, ptr getelementptr inbounds (i8, ptr @str, i64 3), align 1
        \\  %__3 = uitofp i8 %__2 to float
        \\  ret float %__3
        \\}
        \\
    );
    try std.testing.expect(has(text2, "@str = private unnamed_addr addrspace(2) constant [8 x i8]") or has(text2, "@str = private addrspace(2) constant [8 x i8]"));
    try std.testing.expect(has(text2, "getelementptr inbounds (i8, ptr addrspace(2) @str, i64 3)"));
    try std.testing.expect(!has(text2, "@lut"));
}

test "D1 phi over relocated constant pointers takes the constant address space" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();
    // `const t = if (c) &tableA else &tableB;` when LLVM emits a phi rather
    // than a select.
    const text = try testRoundTrip(gpa, test_header ++
        \\@a = private unnamed_addr constant [4 x float] [float 1.000000e+00, float 2.000000e+00, float 3.000000e+00, float 4.000000e+00], align 4
        \\@b = private unnamed_addr constant [4 x float] [float 5.000000e+00, float 6.000000e+00, float 7.000000e+00, float 8.000000e+00], align 4
        \\define float @fragmentShader(i32 %__0, i1 %__1) {
        \\  br i1 %__1, label %__3, label %__4
        \\__3:
        \\  br label %__5
        \\__4:
        \\  br label %__5
        \\__5:
        \\  %t = phi ptr [ @a, %__3 ], [ @b, %__4 ]
        \\  %__6 = and i32 %__0, 3
        \\  %__7 = zext i32 %__6 to i64
        \\  %__8 = getelementptr inbounds [4 x i8], ptr %t, i64 %__7
        \\  %__9 = load float, ptr %__8, align 4
        \\  ret float %__9
        \\}
        \\
    );
    // The Builder strips value names, so match on shapes only.
    try std.testing.expect(has(text, "phi ptr addrspace(2) [ @a, %"));
    try std.testing.expect(has(text, "], [ @b, %"));
    try std.testing.expect(has(text, "getelementptr inbounds [4 x i8], ptr addrspace(2) %"));
    try std.testing.expect(has(text, "load float, ptr addrspace(2) %"));

    // A loop-carried pointer phi whose only pre-defined incoming is a GEP
    // of a relocated constant: the earlier local fixes the address space,
    // the back-edge value is checked when the phi is resolved.
    const text2 = try testRoundTrip(gpa, test_header ++
        \\@a = private unnamed_addr constant [4 x float] [float 1.000000e+00, float 2.000000e+00, float 3.000000e+00, float 4.000000e+00], align 4
        \\define float @fragmentShader(i32 %__0) {
        \\  %start = getelementptr inbounds [4 x i8], ptr @a, i64 0
        \\  br label %loop
        \\loop:
        \\  %p = phi ptr [ %start, %__1 ], [ %next, %loop ]
        \\  %i = phi i32 [ 0, %__1 ], [ %i.next, %loop ]
        \\  %v = load float, ptr %p, align 4
        \\  %next = getelementptr inbounds i8, ptr %p, i64 4
        \\  %i.next = add i32 %i, 1
        \\  %done = icmp eq i32 %i.next, 4
        \\  br i1 %done, label %exit, label %loop
        \\exit:
        \\  ret float %v
        \\}
        \\
    );
    try std.testing.expect(has(text2, "phi ptr addrspace(2) [ %"));
    try std.testing.expect(has(text2, "getelementptr inbounds i8, ptr addrspace(2) %"));
    try std.testing.expect(has(text2, ", i64 4"));
}

test "D1 insertvalue/insertelement/vector literals holding relocated pointers are refused, not asserted" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();
    // `noinline fn pick(c: bool, len: u32) []const f32 { const p = if (c) &tableA else &tableB; return p[0..len]; }`
    // at ReleaseFast: a select over the two tables, then insertvalue into
    // a `{ ptr, i64 }` slice (review-a1/fix1check/v_retslice2.ll).
    try testFails(gpa, test_header ++
        \\@s1 = private unnamed_addr constant [3 x i8] c"ab\00", align 1
        \\@s2 = private unnamed_addr constant [4 x i8] c"cde\00", align 1
        \\define private fastcc { ptr, i64 } @pick(i1 %__0) {
        \\  %__2 = select i1 %__0, ptr @s1, ptr @s2
        \\  %__3 = select i1 %__0, i64 2, i64 3
        \\  %__4 = insertvalue { ptr, i64 } poison, ptr %__2, 0
        \\  %__5 = insertvalue { ptr, i64 } %__4, i64 %__3, 1
        \\  ret { ptr, i64 } %__5
        \\}
        \\define i32 @fragmentShader(i32 %__0, i1 %__1) {
        \\  %__3 = call fastcc { ptr, i64 } @pick(i1 %__1)
        \\  %__4 = extractvalue { ptr, i64 } %__3, 0
        \\  %__5 = load i8, ptr %__4, align 1
        \\  %__6 = zext i8 %__5 to i32
        \\  ret i32 %__6
        \\}
        \\
    );
    // The same slice built directly from the global.
    try testFails(gpa, test_header ++
        \\@s = private unnamed_addr constant [2 x i8] c"hi", align 1
        \\define i32 @fragmentShader(i32 %__0) {
        \\  %__2 = insertvalue { ptr, i64 } poison, ptr @s, 0
        \\  %__3 = extractvalue { ptr, i64 } %__2, 0
        \\  %__4 = load i8, ptr %__3, align 1
        \\  %__5 = zext i8 %__4 to i32
        \\  ret i32 %__5
        \\}
        \\
    );
    // A vector-of-pointer constant (`<2 x ptr> <ptr @a, ptr @b>`).
    try testFails(gpa, test_header ++
        \\@a = private unnamed_addr constant [4 x float] [float 1.0, float 2.0, float 3.0, float 4.0], align 4
        \\@b = private unnamed_addr constant [4 x float] [float 5.0, float 6.0, float 7.0, float 8.0], align 4
        \\@vt = private unnamed_addr constant <2 x ptr> <ptr @a, ptr @b>, align 16
        \\define float @fragmentShader(i32 %__0) {
        \\  %__2 = and i32 %__0, 1
        \\  %__3 = zext i32 %__2 to i64
        \\  %__4 = getelementptr inbounds [8 x i8], ptr @vt, i64 %__3
        \\  %__5 = load ptr, ptr %__4, align 8
        \\  %__6 = load float, ptr %__5, align 4
        \\  ret float %__6
        \\}
        \\
    );
    // insertelement of a relocated pointer into a `<2 x ptr>`.
    try testFails(gpa, test_header ++
        \\@a = private unnamed_addr constant [4 x float] [float 1.0, float 2.0, float 3.0, float 4.0], align 4
        \\define float @fragmentShader(i32 %__0) {
        \\  %__2 = insertelement <2 x ptr> poison, ptr @a, i64 0
        \\  %__3 = extractelement <2 x ptr> %__2, i64 0
        \\  %__4 = load float, ptr %__3, align 4
        \\  ret float %__4
        \\}
        \\
    );
    // Well-typed insertvalue/insertelement still assemble: a slice of a
    // thread-space alloca, and a float lane.
    const text = try testRoundTrip(gpa, test_header ++
        \\define float @fragmentShader(i32 %__0) {
        \\  %buf = alloca [4 x float], align 16
        \\  %s0 = insertvalue { ptr, i64 } poison, ptr %buf, 0
        \\  %s1 = insertvalue { ptr, i64 } %s0, i64 4, 1
        \\  %p = extractvalue { ptr, i64 } %s1, 0
        \\  %f = uitofp i32 %__0 to float
        \\  store float %f, ptr %p, align 4
        \\  %v = insertelement <2 x float> poison, float %f, i64 1
        \\  %l = extractelement <2 x float> %v, i64 1
        \\  ret float %l
        \\}
        \\
    );
    try std.testing.expect(has(text, "insertvalue { ptr, i64 } poison, ptr %"));
    try std.testing.expect(has(text, "insertelement <2 x float> poison, float %"));
}

test "D1 mutable module-level globals in address space 0 are refused with a hint" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();
    // `var acc: f32 = 0.25;` read/written from the fragment stage
    // (review-a1/fix1check/v_mutglobal.ll): Metal loads the library and
    // then fails pipeline creation with "Undefined symbols: _acc".
    try testFails(gpa, test_header ++
        \\@acc = private unnamed_addr global float 2.500000e-01, align 4
        \\define float @fragmentShader(float %__0) {
        \\  %__2 = load volatile float, ptr @acc, align 4
        \\  %__3 = fadd float %__2, %__0
        \\  store volatile float %__3, ptr @acc, align 4
        \\  ret float %__3
        \\}
        \\
    );
    // nvptx addrspace(5) (local) renumbers to 0 and is the same case.
    try testFails(gpa, test_header ++
        \\@lbuf = private unnamed_addr addrspace(5) global [4 x float] undef, align 4
        \\define float @fragmentShader(i32 %__0) {
        \\  %__2 = load float, ptr addrspace(5) @lbuf, align 4
        \\  ret float %__2
        \\}
        \\
    );
    // Unreferenced, such a global does not break the module (defined on
    // first reference); device/threadgroup mutable globals are fine.
    const text = try testRoundTrip(gpa, test_header ++
        \\@acc = private unnamed_addr global float 2.500000e-01, align 4
        \\@gvar = private unnamed_addr addrspace(1) global float 0.000000e+00, align 4
        \\@tile = private unnamed_addr addrspace(3) global [64 x float] undef, align 4
        \\define float @fragmentShader(float %__0) {
        \\  store float %__0, ptr addrspace(1) @gvar, align 4
        \\  store float %__0, ptr addrspace(3) @tile, align 4
        \\  %__2 = load float, ptr addrspace(3) @tile, align 4
        \\  ret float %__2
        \\}
        \\
    );
    try std.testing.expect(!has(text, "@acc"));
    try std.testing.expect(has(text, "addrspace(1) global float"));
    try std.testing.expect(has(text, "addrspace(3) global [64 x float] undef"));
}

test "D1 phi over getelementptr constant expressions into relocated tables" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();
    // `tableA[4..8]` vs `tableB[2..6]` joined by a phi when SimplifyCFG
    // does not form a select (review-a1/fix1check/p_phi_gep.ll).
    const text = try testRoundTrip(gpa, test_header ++
        \\@a = private unnamed_addr constant [8 x float] [float 1.0, float 2.0, float 3.0, float 4.0, float 5.0, float 6.0, float 7.0, float 8.0], align 4
        \\@b = private unnamed_addr constant [8 x float] [float 9.0, float 8.0, float 7.0, float 6.0, float 5.0, float 4.0, float 3.0, float 2.0], align 4
        \\define float @fragmentShader(i32 %__0, i1 %__1) {
        \\  br i1 %__1, label %__3, label %__4
        \\__3:
        \\  br label %__5
        \\__4:
        \\  br label %__5
        \\__5:
        \\  %t = phi ptr [ getelementptr inbounds (i8, ptr @a, i64 16), %__3 ], [ getelementptr inbounds (i8, ptr @b, i64 8), %__4 ]
        \\  %__6 = and i32 %__0, 3
        \\  %__7 = zext i32 %__6 to i64
        \\  %__8 = getelementptr inbounds [4 x i8], ptr %t, i64 %__7
        \\  %__9 = load float, ptr %__8, align 4
        \\  ret float %__9
        \\}
        \\
    );
    try std.testing.expect(has(text, "phi ptr addrspace(2) [ getelementptr inbounds (i8, ptr addrspace(2) @a, i64 16), %"));
    try std.testing.expect(has(text, "], [ getelementptr inbounds (i8, ptr addrspace(2) @b, i64 8), %"));
    try std.testing.expect(has(text, "load float, ptr addrspace(2) %"));

    // A `null` incoming takes the address space the other incomings fix.
    const text2 = try testRoundTrip(gpa, test_header ++
        \\@a = private unnamed_addr constant [8 x float] [float 1.0, float 2.0, float 3.0, float 4.0, float 5.0, float 6.0, float 7.0, float 8.0], align 4
        \\define float @fragmentShader(i32 %__0, i1 %__1) {
        \\  br i1 %__1, label %__3, label %__4
        \\__3:
        \\  br label %__5
        \\__4:
        \\  br label %__5
        \\__5:
        \\  %t = phi ptr [ null, %__3 ], [ getelementptr inbounds (i8, ptr @a, i64 16), %__4 ]
        \\  %isnull = icmp eq ptr %t, null
        \\  br i1 %isnull, label %__6, label %__7
        \\__6:
        \\  ret float 0.0
        \\__7:
        \\  %__8 = load float, ptr %t, align 4
        \\  ret float %__8
        \\}
        \\
    );
    try std.testing.expect(has(text2, "phi ptr addrspace(2) [ null, %"));
    try std.testing.expect(has(text2, "icmp eq ptr addrspace(2) %"));
}

test "the bitcode's triple and module metadata follow the deployment target (literal values from Apple's output)" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();
    const module = test_header ++
        \\define <4 x float> @fragmentShader(<4 x float> %__0, <3 x float> %__1, <2 x float> %__2, ptr addrspace(1) %__3) {
        \\  ret <4 x float> %__0
        \\}
        \\
    ;
    // `xcrun metal -mmacosx-version-min=<os> -std=<metal>` disassemblies
    // (macOS 26.5 SDK): triple, !air.version, !air.language_version, and the
    // frame-pointer module flag that macOS 13 output does not carry.
    const Expect = struct { name: air_target.Name, triple: []const u8, air: []const u8, metal: []const u8, frame_pointer: bool };
    const apple = [_]Expect{
        .{ .name = .macos26, .triple = "target triple = \"air64_v28-apple-macosx26.0.0\"", .air = "i32 2, i32 8, i32 0", .metal = "!\"Metal\", i32 4, i32 0, i32 0", .frame_pointer = true },
        .{ .name = .macos15, .triple = "target triple = \"air64_v27-apple-macosx15.0.0\"", .air = "i32 2, i32 7, i32 0", .metal = "!\"Metal\", i32 3, i32 2, i32 0", .frame_pointer = true },
        .{ .name = .macos13, .triple = "target triple = \"air64_v25-apple-macosx13.0.0\"", .air = "i32 2, i32 5, i32 0", .metal = "!\"Metal\", i32 3, i32 0, i32 0", .frame_pointer = false },
    };
    for (apple) |e| {
        var b = try Builder.init(.{ .allocator = gpa, .strip = true, .name = "air-splice" });
        defer b.deinit();
        try build(&b, gpa, module, .{ .entry = "fragmentShader", .fm = fragment_fm }, air_target.get(e.name));
        var aw: std.Io.Writer.Allocating = .init(gpa);
        try b.print(&aw.writer);
        const text = aw.written();
        try std.testing.expect(has(text, e.triple));
        try std.testing.expect(has(text, e.air));
        try std.testing.expect(has(text, e.metal));
        try std.testing.expectEqual(e.frame_pointer, has(text, "!\"frame-pointer\""));
        try std.testing.expect(has(text, "!\"air.compile.fast_math_enable\""));
        try std.testing.expect(has(text, "!\"air.max_samplers\", i32 16"));
        // No other profile's versions leak in.
        for (apple) |other| {
            if (other.name != e.name) try std.testing.expect(!has(text, other.metal));
        }
    }
}
