//! Uniformity analysis for the assembler's convergent-call check
//! (DESIGN.md D5-13, review-b rounds 1 and 2, B re-review fix round 1).
//!
//! Apple's compiler declares the barriers and every SIMD-group collective
//! `convergent` (research/_results/r10.json claim 2). Zig cannot put that
//! attribute on an `extern fn`, so LLVM freely duplicates such a call into
//! the arms of a branch (jump threading through the block that follows
//! `if (tidx == 0) x = 0;`) or unswitches a loop around it. Metal's runtime
//! compiler accepts the result and the GPU runs the copies as different
//! barriers (review-b/fable-hazard/hazard.check.log: 264 of 512 threads
//! counted) or sums only the lanes that took the same arm
//! (review-b/r4/helper/check.log: `out[0]=1 out[1]=33 want 528 1552`).
//!
//! Two optimizers can duplicate such a call. Zig's LLVM does it before the
//! assembler sees the IR; the IR keeps no trace of the duplication, so the
//! assembler decides from the CFG alone which threads reach a call site
//! together, and refuses a call that only some of them reach. Metal's own
//! optimizer does it after: it jump-threads `if (lid == 0) ...; barrier;
//! ...; barrier; if (lid == 0) ...`, which this analysis accepts (every
//! barrier is in a join block), into both arms unless the barrier is
//! declared `convergent`. The SPIR-V front end hands Metal unoptimized IR of
//! exactly that shape (glslang's workgroup counters and histograms), and so
//! does a Debug Zig shader; the assembler therefore also writes Apple's
//! `convergent` attribute sets on the declarations, the calls and every
//! helper this module finds reaching one (`Func.convergent`;
//! assembler.ModuleState.attributesFor), which is what stops Metal (exec
//! tests barrier_workgroup_counter and barrier_shared_flag, metallib-check's
//! barrierJoinKernel under -Dshader-optimize=debug: wrong without the
//! attribute, right with it). A purely
//! structural rule (the block must post-dominate the entry) refused every
//! barrier inside a loop with a runtime trip count, including the canonical
//! tree reduction `while (s > 0) : (s >>= 1)` (review-b/r4/reduce/
//! metallib.log). This module replaces it with a uniformity analysis in the
//! style of LLVM's UniformityAnalysis, on the text the splice hands to the
//! assembler, over a two-level lattice (`Level` = `air.Uniformity`):
//! `.threadgroup` (every thread agrees / arrives together) < `.simdgroup`
//! (every thread of one SIMD-group agrees; SIMD-groups may differ) <
//! `.thread` (per-thread). The second level exists for the canonical
//! two-stage reduction `simd_sum` per SIMD-group, then `if (sgid == 0)
//! { t = simd_sum(partial[lane]) }`: the branch on
//! `simdgroup_index_in_threadgroup` is uniform within every SIMD-group, and
//! MSL defines the SIMD-group functions over the active lanes, so the second
//! `simd_sum` is legal and runs correctly (review2/v1_nocheck/check.log)
//! while a threadgroup barrier in the same block would not be.
//!
//!   * Values: a kernel parameter's level comes from the manifest
//!     (`air.BuiltinKind.uniformity`: `thread_position_in_grid`,
//!     `thread_index_in_simdgroup`, ... are `.thread`, `simdgroup_index_in_
//!     threadgroup` is `.simdgroup`, buffer pointers and the sizes and
//!     positions of the threadgroup are `.threadgroup`). Constants and
//!     globals are uniform. An instruction's result is as divergent as its
//!     most divergent operand; an `alloca` whose address escapes (below),
//!     `atomicrmw`/`cmpxchg`, `air.atomic.*`, the SIMD-group shuffles and prefix
//!     operations are `.thread` whatever their operands, and the SIMD-group
//!     reductions (`air.simd_sum`, `_max`, `_min`, `_product`, `_and`,
//!     `_or`, `_xor`, `_any`, `_all`, `_broadcast_first`) are `.simdgroup`
//!     whatever their operands. A `load` is as divergent as its address,
//!     except a `load atomic`, which is `.thread` (another thread may store
//!     between two loads), and a load from a private variable (below). A
//!     `phi` is as divergent as its incoming values and as the branches
//!     whose paths its block joins. Helpers (`define private fastcc`) take
//!     the level of their arguments over every call site (an
//!     interprocedural fixpoint over a worklist, `analyse`); a call returns
//!     the join of its arguments' levels and the callee's return level.
//!   * Private variables: every `alloca` is a variable, and so is a `ptr`
//!     parameter of a helper that some caller passes a variable to. Pointers
//!     into it are the variable itself and the `getelementptr`s derived from
//!     it (a struct member, an array element, a vector lane). While every
//!     such pointer is only the address of a plain `load` or `store`, the
//!     base of a `getelementptr`, or an argument of a helper (a `define`)
//!     whose parameter does not escape in turn (or of `llvm.lifetime.*`),
//!     the variable holds what was stored in it: its address names the
//!     same variable in every thread (`.threadgroup`), a store raises its
//!     content by the stored value, the address's indices and the block the
//!     store sits in (a store that only some threads execute makes the
//!     variable per-thread, as a phi joining divergent paths would be), and
//!     a load returns that content joined with the address's indices. A
//!     helper reads its callers' variables through the parameter
//!     (`Func.param_mem`, joined over the call sites) and what it stores
//!     through it lands in them (`Func.param_store`, joined with the
//!     calling block). This is mem2reg and SROA's view of the variable
//!     without the rewrite: the SPIR-V front end (tools/spirv/lower.zig)
//!     keeps every OpVariable Function as an `alloca`; glslang and Zig's
//!     SPIR-V backend keep a loop counter in one, glslang -V copies every
//!     helper argument into a `param` variable and passes it by pointer
//!     (`uint tiles = ceilDiv(pc.n, 64u);`), and reads a local struct or
//!     vector through OpAccessChain (`P p = P(pc.n); ... p.n`), so a tree
//!     reduction bounded by the workgroup size or a tiled loop bounded by a
//!     push constant, with a barrier inside, is accepted as on the Zig
//!     path, where LLVM has promoted all of these to SSA values (review
//!     round 2: such kernels were refused; exec tests uniform_bound_*). A
//!     variable whose address escapes (stored, compared, selected, passed
//!     to an intrinsic such as `llvm.memcpy` or an external function, or to
//!     a helper whose parameter escapes) is not followed: its address and
//!     everything loaded through it are `.thread`.
//!   * Control: a conditional `br` or `switch` on a value of level L > 0 is
//!     a divergent branch X of level L. Every block reachable from X's
//!     successors without passing a block that post-dominates X is raised
//!     to L (a subset of the threads that reached X runs it). A block that
//!     ends in `unreachable` (Zig safety checks: `llvm.trap` +
//!     `unreachable`, `@panic` under `std.debug.no_panic`) never returns,
//!     so it does not cut post-dominance: the block after a bounds check on
//!     a per-thread index post-dominates the check, as in LLVM's handling
//!     of no-return paths. When a path from X reaches the header of a loop
//!     containing X while another leaves the loop, the threads leave or
//!     re-enter that loop at different iterations ("temporal divergence"):
//!     every block of the loop is raised to L and every value defined in it
//!     too. A branch whose arms all return to the header (or all leave)
//!     reconverges there instead; only the header's phis are raised.
//!   * A `ret` in a block of level L returns a value of level at least L,
//!     even a constant: a helper `if (atomic(p) == 0) { ret 1 } ret 2`
//!     returns a per-thread value although both operands are constants
//!     (review2/v2/edited/shader.ll: accepting it let the caller's
//!     `if (k == 1)` pass as uniform and both jump-threaded barriers
//!     through; `FAIL dispatch multiRetKernel: 8 mismatches`).
//!   * Convergent calls carry a `Scope`: `air.wg.barrier` synchronises the
//!     threadgroup and is refused in any block above `.threadgroup`;
//!     `air.simdgroup.barrier`, `air.simd_*` and `air.quad_*` communicate
//!     within a SIMD-group and are refused only in `.thread` blocks. A
//!     helper's `Func.convergent` is the widest scope it reaches through
//!     calls, so a barrier hidden in a noinline helper is checked at the
//!     helper's call site exactly like the barrier itself (review-b/r4/
//!     helper: `syncTg()` jump-threaded into both arms).
//!
//! Accepted by this rule: barriers in straight-line code, in the join block
//! after a per-thread branch, in a loop whose trip count is a uniform value
//! (`threads_per_threadgroup / 2`, a constant-buffer field, a constant) even
//! when LLVM guards or rotates it, after a uniform early return, and after a
//! safety check that traps; SIMD-group calls under `if (sgid == 0)`.
//! Refused: both arms of a jump-threaded diamond, a barrier behind
//! `if (gid >= n) return;`, a loop some threads `break` out of on a
//! per-thread condition, both copies of an unswitched loop, a threadgroup
//! barrier under `if (sgid == 0)`, a `simd_sum` under `if (lane == 0)`. As
//! in MSL, a call the analysis cannot prove convergent is undefined
//! behaviour on the GPU; the analysis errs on the side of refusing.
//!
//! The analysis never fails: anything it does not understand is treated as
//! per-thread (values) or as an ordinary instruction (control flow). Its
//! cost is near-linear in the module: post-dominance and loop membership
//! are answered from trees in O(1) (`Control`), a divergent branch walks
//! only the blocks it makes divergent, and functions are found by name in
//! a hash map (review round 2: the former n x n tables and linear lookups
//! took 17 s for a 1000-block chain, 111 s for 1600 sequential ifs and
//! 113 s for 20 000 helpers; now 0.04 s, 0.3 s and 2.6 s, debug build).

const std = @import("std");
const Allocator = std.mem.Allocator;
const intrinsics = @import("intrinsics.zig");

pub const Error = error{OutOfMemory};

/// The lattice (`src/engine/air.zig`): `.threadgroup` < `.simdgroup` <
/// `.thread`.
pub const Level = @import("shader").air.Uniformity;

/// The set of threads a convergent call needs to arrive together.
pub const Scope = enum(u2) {
    /// Not convergent.
    none = 0,
    /// Communicates within a SIMD-group (`air.simd_*`, `air.quad_*`,
    /// `air.simdgroup.barrier`): every active lane of the SIMD-group must
    /// reach it, so its block may be `.simdgroup` but not `.thread`.
    simdgroup = 1,
    /// Synchronises the threadgroup (`air.wg.barrier`): its block must be
    /// `.threadgroup`.
    threadgroup = 2,

    pub fn join(a: Scope, b: Scope) Scope {
        return @fromBackingInt(@intCast(@max(@backingInt(a), @backingInt(b))));
    }

    /// Whether a call of this scope may sit in a block of level `l`.
    pub fn allows(s: Scope, l: Level) bool {
        return switch (s) {
            .none => true,
            .simdgroup => l != .thread,
            .threadgroup => l == .threadgroup,
        };
    }
};

/// The scope of a convergent intrinsic (`.none` for everything else,
/// including helpers: those carry their own `Func.convergent`).
pub fn callScope(name: []const u8) Scope {
    if (std.mem.eql(u8, name, "air.wg.barrier")) return .threadgroup;
    if (intrinsics.isConvergent(name)) return .simdgroup;
    return .none;
}

/// One `define` (or `declare`, with `body == null`) of the module, with the
/// analysis results. Indices into the caller's list must match the
/// assembler's `defines`.
pub const Func = struct {
    name: []const u8,
    param_names: []const []const u8,
    /// Body lines between the `define` header and the closing `}`.
    body: ?[]const []const u8,
    /// Per parameter. The caller seeds the entry point's from the manifest;
    /// every other function starts uniform and picks up divergence from
    /// its call sites.
    param_level: []Level,
    /// Per parameter: the level of the private variable a `ptr` argument
    /// points to, joined over the call sites (module doc, "Private
    /// variables"); null while no call site passes a private pointer.
    param_mem: []?Level,
    /// Per parameter: the level of what this function stores through the
    /// pointer, itself or through its callees; null if it never does.
    param_store: []?Level,
    /// Per parameter: this function lets the pointer escape (anything but
    /// a load, a store, a getelementptr or a helper call uses it), so the
    /// callers' variable can no longer be followed.
    param_escapes: []bool,
    /// How the value this function returns may differ between threads.
    ret_level: Level = .threadgroup,
    /// The widest scope of a convergent call it executes, directly or
    /// through a callee.
    convergent: Scope = .none,
    /// Per body line: which threads reach the block holding it together.
    /// Empty for a `declare`.
    line_level: []Level,

    pub fn init(gpa: Allocator, name: []const u8, param_names: []const []const u8, body: ?[]const []const u8) Error!Func {
        const n = param_names.len;
        const pl = try gpa.alloc(Level, n);
        errdefer gpa.free(pl);
        @memset(pl, .threadgroup);
        const pm = try gpa.alloc(?Level, n);
        errdefer gpa.free(pm);
        @memset(pm, null);
        const ps = try gpa.alloc(?Level, n);
        errdefer gpa.free(ps);
        @memset(ps, null);
        const pe = try gpa.alloc(bool, n);
        errdefer gpa.free(pe);
        @memset(pe, false);
        const ll = try gpa.alloc(Level, if (body) |b| b.len else 0);
        @memset(ll, .threadgroup);
        return .{ .name = name, .param_names = param_names, .body = body, .param_level = pl, .param_mem = pm, .param_store = ps, .param_escapes = pe, .line_level = ll };
    }

    pub fn deinit(f: *Func, gpa: Allocator) void {
        gpa.free(f.param_level);
        gpa.free(f.param_mem);
        gpa.free(f.param_store);
        gpa.free(f.param_escapes);
        gpa.free(f.line_level);
    }
};

/// Run the analysis over every function to a fixpoint. Only the entry
/// point's `param_level` needs to be seeded; the results of functions
/// nothing reaches are meaningless but harmless (their parameters stay
/// uniform, so no divergence flows out of them).
///
/// A worklist, not rounds over the whole module: a function is analysed
/// again only when an input of it rose (a parameter's level or private
/// content, from a call site) or an output of a callee did (its return
/// level, convergence scope, or what it stores through or lets escape
/// from a pointer parameter). Callees are found by name in a hash map. A
/// module of F helpers and a call chain of depth D therefore costs about
/// one analysis per function and change instead of D rounds of F scans
/// (review finding: 20 000 helpers took 113 s, a 10 000-deep chain 21 s).
pub fn analyse(gpa: Allocator, funcs: []Func) Error!void {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    var m = try Module.init(arena.allocator(), funcs);
    for (funcs, 0..) |_, i| try m.enqueue(@intCast(i));
    while (m.head < m.queue.items.len) {
        const fi = m.queue.items[m.head];
        m.head += 1;
        if (m.head == m.queue.items.len) {
            m.queue.clearRetainingCapacity();
            m.head = 0;
        }
        m.queued[fi] = false;
        try analyseOne(gpa, &m, fi);
    }
}

/// The module-wide bookkeeping of `analyse`.
const Module = struct {
    a: Allocator,
    funcs: []Func,
    /// Function name -> index (the first of a name, as a linear search
    /// would find it).
    index: std.StringHashMapUnmanaged(u32) = .empty,
    /// Per function: the distinct functions seen calling it.
    callers: []std.ArrayList(u32),
    edges: std.AutoHashMapUnmanaged(u64, void) = .empty,
    queued: []bool,
    queue: std.ArrayList(u32) = .empty,
    head: usize = 0,

    fn init(a: Allocator, funcs: []Func) Error!Module {
        var m = Module{ .a = a, .funcs = funcs, .callers = try a.alloc(std.ArrayList(u32), funcs.len), .queued = try a.alloc(bool, funcs.len) };
        @memset(m.callers, .empty);
        @memset(m.queued, false);
        for (funcs, 0..) |f, i| {
            const gop = try m.index.getOrPut(a, f.name);
            if (!gop.found_existing) gop.value_ptr.* = @intCast(i);
        }
        return m;
    }

    fn find(m: *const Module, name: []const u8) ?u32 {
        return m.index.get(name);
    }

    fn enqueue(m: *Module, fi: u32) Error!void {
        if (m.queued[fi] or m.funcs[fi].body == null) return;
        m.queued[fi] = true;
        try m.queue.append(m.a, fi);
    }

    fn addCaller(m: *Module, callee: u32, caller: u32) Error!void {
        const gop = try m.edges.getOrPut(m.a, (@as(u64, callee) << 32) | caller);
        if (!gop.found_existing) try m.callers[callee].append(m.a, caller);
    }

    /// An output of `fi` rose: its callers read it.
    fn outputsChanged(m: *Module, fi: u32) Error!void {
        for (m.callers[fi].items) |c| try m.enqueue(c);
    }
};

/// The SIMD-group reductions: the result is the same for every lane of the
/// SIMD-group whatever the lanes passed in (`air.simd_sum.f32`,
/// `air.simd_any`, `air.simd_broadcast_first.f32`, ...). `simd_broadcast`
/// with a lane operand, the shuffles and the prefix operations are not:
/// their result is per lane.
fn isSimdReduction(name: []const u8) bool {
    const prefix = "air.simd_";
    if (!std.mem.startsWith(u8, name, prefix)) return false;
    const rest = name[prefix.len..];
    const op = rest[0 .. std.mem.findScalar(u8, rest, '.') orelse rest.len];
    const ops = [_][]const u8{ "sum", "product", "min", "max", "and", "or", "xor", "any", "all", "broadcast_first" };
    for (ops) |o| if (std.mem.eql(u8, op, o)) return true;
    return false;
}

/// The level of the value a call to `name` returns even with uniform
/// arguments: atomics return the previous value each thread saw, and the
/// SIMD-group collectives differ at least between the SIMD-groups of a
/// threadgroup.
fn callResultLevel(name: []const u8) Level {
    if (std.mem.startsWith(u8, name, "air.atomic.")) return .thread;
    if (intrinsics.isConvergent(name)) return if (isSimdReduction(name)) .simdgroup else .thread;
    return .threadgroup;
}

// ── Per-function analysis ───────────────────────────────────────────────────

/// A private variable: an `alloca` of this function, or a pointer
/// parameter some caller passes a private variable to (module doc).
const Root = struct {
    /// The `%name` of the alloca or parameter.
    name: []const u8,
    /// The parameter index, for a parameter.
    param: ?u32 = null,
    /// Its address reaches something other than a load, a store, a
    /// getelementptr or a helper call: it is not followed.
    escaped: bool = false,
    /// What the callers stored in it (a parameter's `Func.param_mem`).
    content: Level = .threadgroup,
    /// What this function (and its callees) store into it; null if
    /// nothing.
    written: ?Level = null,

    fn level(r: Root) Level {
        return r.content.join(r.written orelse .threadgroup);
    }
};

const none: u32 = std.math.maxInt(u32);

const Analyser = struct {
    a: Allocator,
    mod: *Module,
    fi: u32,
    body: []const []const u8,
    cfg: Cfg,
    /// Level of every value above `.threadgroup` (function-local names).
    div: std.StringHashMapUnmanaged(Level) = .empty,
    block_level: []Level,
    /// Per block: the level of the divergent branches whose paths it joins
    /// (its phis are at least this divergent).
    join: []Level,
    /// Per block: the level its terminator was last expanded at
    /// (`.threadgroup` = not yet; a branch is re-expanded when its
    /// condition rises).
    branch_level: []Level,
    /// Lazily built control-flow facts.
    ctl: ?Control = null,
    ret_level: Level = .threadgroup,
    convergent: Scope = .none,
    local_changed: bool = false,
    /// Private variables (`findRoots`).
    roots: std.ArrayList(Root) = .empty,
    /// Pointer name (a root or a getelementptr from one) -> root index.
    root_of: std.StringHashMapUnmanaged(u32) = .empty,
    /// Scratch for `expandBranch`: per block the epoch it was last
    /// reached in and the join count of that epoch, per loop the epoch
    /// its header was last hit in.
    reached: []u32,
    join_epoch: []u32,
    join_count: []u32,
    hit_epoch: []u32 = &.{},
    epoch: u32 = 0,
    region: std.ArrayList(u32) = .empty,
    hits: std.ArrayList(u32) = .empty,
    stack: std.ArrayList(u32) = .empty,

    fn init(a: Allocator, mod: *Module, fi: u32) Error!Analyser {
        const body = mod.funcs[fi].body.?;
        const cfg = try Cfg.build(a, body);
        const n = cfg.n;
        const s = Analyser{
            .a = a,
            .mod = mod,
            .fi = fi,
            .body = body,
            .cfg = cfg,
            .block_level = try a.alloc(Level, n),
            .join = try a.alloc(Level, n),
            .branch_level = try a.alloc(Level, n),
            .reached = try a.alloc(u32, n),
            .join_epoch = try a.alloc(u32, n),
            .join_count = try a.alloc(u32, n),
        };
        @memset(s.block_level, .threadgroup);
        @memset(s.join, .threadgroup);
        @memset(s.branch_level, .threadgroup);
        @memset(s.reached, 0);
        @memset(s.join_epoch, 0);
        return s;
    }

    fn levelOfName(s: *Analyser, name: []const u8) Level {
        return s.div.get(name) orelse .threadgroup;
    }

    fn raiseValue(s: *Analyser, name: []const u8, level: Level) Error!void {
        if (!level.above(s.levelOfName(name))) return;
        try s.div.put(s.a, name, level);
        s.local_changed = true;
    }

    fn raiseBlock(s: *Analyser, b: usize, level: Level) void {
        if (!level.above(s.block_level[b])) return;
        s.block_level[b] = level;
        s.local_changed = true;
    }

    /// The followed (non-escaping) private variable `name` points into.
    fn trackedRoot(s: *Analyser, name: []const u8) ?*Root {
        const r = &s.roots.items[s.root_of.get(name) orelse return null];
        return if (r.escaped) null else r;
    }

    fn raiseWritten(s: *Analyser, r: *Root, level: Level) void {
        if (r.written) |w| if (!level.above(w)) return;
        r.written = level;
        s.local_changed = true;
    }

    fn run(s: *Analyser) Error!void {
        const f = &s.mod.funcs[s.fi];
        for (f.param_names, f.param_level) |p, l| if (l != .threadgroup) try s.div.put(s.a, p, l);
        try s.findRoots();
        // An escaped variable is not followed: its address, every pointer
        // derived from it and every load through one is per-thread.
        for (s.roots.items) |r| if (r.escaped) try s.raiseValue(r.name, .thread);
        s.local_changed = true;
        while (s.local_changed) {
            s.local_changed = false;
            for (s.body, 0..) |raw, li| try s.visitLine(raw, li);
        }
        for (f.line_level, 0..) |*ll, li| ll.* = s.block_level[s.cfg.line_block[li]];
        var changed = false;
        if (s.ret_level.above(f.ret_level)) {
            f.ret_level = s.ret_level;
            changed = true;
        }
        if (@backingInt(s.convergent) > @backingInt(f.convergent)) {
            f.convergent = s.convergent;
            changed = true;
        }
        for (s.roots.items) |r| {
            const pi = r.param orelse continue;
            if (r.escaped and !f.param_escapes[pi]) {
                f.param_escapes[pi] = true;
                changed = true;
            }
            if (r.written) |w| if (f.param_store[pi] == null or w.above(f.param_store[pi].?)) {
                f.param_store[pi] = w;
                changed = true;
            };
        }
        if (changed) try s.mod.outputsChanged(s.fi);
    }

    fn visitLine(s: *Analyser, raw: []const u8, li: usize) Error!void {
        const line = instructionText(raw) orelse return;
        const blk = s.cfg.line_block[li];
        var c = Scanner{ .s = line };
        var result: ?[]const u8 = null;
        if (c.peek() == '%') {
            _ = c.next();
            const name = c.name();
            c.skipWs();
            if (!c.eat('=')) return;
            result = name;
        }
        c.skipWs();
        var op = c.word();
        if (std.mem.eql(u8, op, "tail") or std.mem.eql(u8, op, "musttail") or std.mem.eql(u8, op, "notail")) {
            c.skipWs();
            op = c.word();
        }
        const rest = c.rest();
        var level: Level = undefined;
        if (std.mem.eql(u8, op, "phi")) {
            level = s.levelOf(rest).join(s.join[blk]);
        } else if (std.mem.eql(u8, op, "alloca")) {
            // The address of a followed variable names the same variable
            // in every thread; what it holds is tracked in its `Root`.
            level = if (result != null and s.trackedRoot(result.?) != null) .threadgroup else .thread;
        } else if (std.mem.eql(u8, op, "atomicrmw") or std.mem.eql(u8, op, "cmpxchg")) {
            level = .thread;
        } else if (std.mem.eql(u8, op, "load")) {
            if (startsWithWord(rest, "atomic")) {
                level = .thread;
            } else if (lastName(rest)) |ptr| {
                level = s.levelOf(rest);
                if (s.trackedRoot(ptr.name)) |r| level = level.join(r.level());
            } else level = s.levelOf(rest);
        } else if (std.mem.eql(u8, op, "store")) {
            if (startsWithWord(rest, "atomic")) return;
            const ptr = lastName(rest) orelse return;
            const r = s.trackedRoot(ptr.name) orelse return;
            // The value, the element the address selects, and which
            // threads store at all.
            s.raiseWritten(r, s.levelOf(rest[0..ptr.start]).join(s.levelOfName(ptr.name)).join(s.block_level[blk]));
            return;
        } else if (std.mem.eql(u8, op, "call")) {
            level = try s.visitCall(rest, blk);
        } else if (std.mem.eql(u8, op, "br") or std.mem.eql(u8, op, "switch")) {
            const l = s.levelOf(rest);
            if (l.above(s.branch_level[blk])) {
                s.branch_level[blk] = l;
                try s.expandBranch(blk, l);
                s.local_changed = true;
            }
            return;
        } else if (std.mem.eql(u8, op, "ret")) {
            // A return only a subset of the threads reaches returns a
            // per-subset value even when the operand is a constant.
            const l = s.levelOf(rest).join(s.block_level[blk]);
            if (l.above(s.ret_level)) s.ret_level = l;
            return;
        } else {
            level = s.levelOf(rest);
        }
        if (result) |r| try s.raiseValue(r, level);
    }

    /// `[fast-math] [cc] [attrs] RET [(fn type)] @callee(args) [bundle]`:
    /// the callee is the first `@`; nothing before it may contain one.
    /// Returns the level of the result.
    fn visitCall(s: *Analyser, rest: []const u8, blk: u32) Error!Level {
        const at = std.mem.findScalar(u8, rest, '@') orelse return s.levelOf(rest);
        var c = Scanner{ .s = rest, .i = at + 1 };
        const callee = c.name();
        c.skipWs();
        var args_level: Level = .threadgroup;
        const callee_index = s.mod.find(callee);
        if (callee_index) |ci| try s.mod.addCaller(ci, s.fi);
        s.convergent = s.convergent.join(callScope(callee));
        if (callee_index) |ci| s.convergent = s.convergent.join(s.mod.funcs[ci].convergent);
        if (c.eat('(')) {
            var arg_index: usize = 0;
            while (true) {
                c.skipWs();
                if (c.eat(')') or c.atEnd()) break;
                const arg = c.argument();
                const l = s.levelOf(arg);
                args_level = args_level.join(l);
                if (callee_index) |ci| {
                    const cf = &s.mod.funcs[ci];
                    if (arg_index < cf.param_level.len) {
                        if (l.above(cf.param_level[arg_index])) {
                            cf.param_level[arg_index] = l;
                            try s.mod.enqueue(ci);
                        }
                        if (cf.body != null and isPrivatePointer(arg)) try s.passPointer(cf, ci, arg_index, arg, l, blk);
                    }
                }
                arg_index += 1;
                c.skipWs();
                if (!c.eat(',')) {
                    _ = c.eat(')');
                    break;
                }
            }
        }
        if (callee_index) |ci| {
            const cf = s.mod.funcs[ci];
            if (cf.body != null) return args_level.join(cf.ret_level);
        }
        if (isSimdReduction(callee)) return .simdgroup;
        return args_level.join(callResultLevel(callee));
    }

    /// A private pointer passed to helper `cf`'s parameter `pi`: the
    /// helper reads the variable's content through it, and what it
    /// stores through it lands in the variable (for the threads that make
    /// the call). A pointer this function does not follow carries
    /// unknown, per-thread content.
    fn passPointer(s: *Analyser, cf: *Func, ci: u32, pi: usize, arg: []const u8, l: Level, blk: u32) Error!void {
        const name = lastName(arg);
        const root: ?*Root = if (name) |n| s.trackedRoot(n.name) else null;
        const content: Level = if (root) |r| r.level() else if (name != null) .thread else .threadgroup;
        if (cf.param_mem[pi] == null or content.above(cf.param_mem[pi].?)) {
            cf.param_mem[pi] = content;
            try s.mod.enqueue(ci);
        }
        if (root) |r| if (cf.param_store[pi]) |st| s.raiseWritten(r, st.join(l).join(s.block_level[blk]));
    }

    /// Collect the private variables (module doc): every `alloca`, every
    /// pointer parameter a caller passes a private variable to, and the
    /// `getelementptr`s derived from them; then mark a variable escaped
    /// when a pointer into it is used other than as the address of a
    /// plain `load` or `store`, the base of a `getelementptr`, or an
    /// argument of a helper whose parameter does not escape.
    fn findRoots(s: *Analyser) Error!void {
        const f = s.mod.funcs[s.fi];
        for (f.param_names, f.param_mem, 0..) |p, pm, i| if (pm) |content| {
            try s.addRoot(.{ .name = p, .param = @intCast(i), .content = content });
        };
        // getelementptr result -> its base operand.
        var gep_base: std.StringHashMapUnmanaged([]const u8) = .empty;
        for (s.body) |raw| {
            const line = instructionText(raw) orelse continue;
            var c = Scanner{ .s = line };
            if (c.peek() != '%') continue;
            _ = c.next();
            const name = c.name();
            c.skipWs();
            if (!c.eat('=')) continue;
            c.skipWs();
            const op = c.word();
            if (std.mem.eql(u8, op, "alloca")) {
                try s.addRoot(.{ .name = name });
            } else if (std.mem.eql(u8, op, "getelementptr")) {
                const base = gepBase(c.rest()) orelse continue;
                try gep_base.put(s.a, name, base.name);
            }
        }
        if (s.roots.items.len == 0) return;
        // Derived pointers, whatever the textual order of the chain; each
        // getelementptr is resolved once (`no_root`: not derived from a
        // variable, such as every device-buffer address).
        var no_root: std.StringHashMapUnmanaged(void) = .empty;
        var it = gep_base.iterator();
        var path: std.ArrayList([]const u8) = .empty;
        while (it.next()) |e| {
            if (s.root_of.contains(e.key_ptr.*) or no_root.contains(e.key_ptr.*)) continue;
            path.clearRetainingCapacity();
            var cur = e.key_ptr.*;
            const root: ?u32 = while (true) {
                if (s.root_of.get(cur)) |r| break r;
                if (no_root.contains(cur)) break null;
                const base = gep_base.get(cur) orelse break null;
                try path.append(s.a, cur);
                if (path.items.len > gep_base.count()) break null; // a cycle: not SSA
                cur = base;
            };
            if (root) |r| {
                for (path.items) |p| try s.root_of.put(s.a, p, r);
            } else for (path.items) |p| try no_root.put(s.a, p, {});
        }
        // Escapes.
        for (s.body) |raw| {
            const line = instructionText(raw) orelse continue;
            var c = Scanner{ .s = line };
            if (c.peek() == '%') {
                const save = c.i;
                _ = c.next();
                _ = c.name();
                c.skipWs();
                if (!c.eat('=')) c.i = save;
            }
            c.skipWs();
            var op = c.word();
            if (std.mem.eql(u8, op, "tail") or std.mem.eql(u8, op, "musttail") or std.mem.eql(u8, op, "notail")) {
                c.skipWs();
                op = c.word();
            }
            if (std.mem.eql(u8, op, "alloca")) continue;
            const rest = c.rest();
            if (std.mem.eql(u8, op, "call")) {
                try s.callEscapes(rest);
                continue;
            }
            const plain = !startsWithWord(rest, "atomic");
            const last = lastName(rest);
            const is_mem = plain and (std.mem.eql(u8, op, "load") or std.mem.eql(u8, op, "store"));
            const base: ?NameAt = if (std.mem.eql(u8, op, "getelementptr")) gepBase(rest) else null;
            var r = Scanner{ .s = rest };
            while (!r.atEnd()) {
                const at = r.i;
                if (r.next() != '%') continue;
                const ri = s.root_of.get(r.name()) orelse continue;
                const ok = (is_mem and last != null and last.?.start == at) or (base != null and base.?.start == at);
                if (!ok) s.roots.items[ri].escaped = true;
            }
        }
    }

    fn addRoot(s: *Analyser, root: Root) Error!void {
        const gop = try s.root_of.getOrPut(s.a, root.name);
        if (gop.found_existing) return;
        gop.value_ptr.* = @intCast(s.roots.items.len);
        try s.roots.append(s.a, root);
    }

    /// The escape rule for a call: a private pointer may be an argument
    /// of a helper (a `define`) whose parameter does not escape, and of
    /// `llvm.lifetime.*`; anything else (`llvm.memcpy`, an intrinsic, an
    /// external function) lets it escape.
    fn callEscapes(s: *Analyser, rest: []const u8) Error!void {
        const at = std.mem.findScalar(u8, rest, '@') orelse return s.escapeAll(rest);
        var c = Scanner{ .s = rest, .i = at + 1 };
        const callee = c.name();
        c.skipWs();
        const ci = s.mod.find(callee);
        if (ci) |i| try s.mod.addCaller(i, s.fi);
        const harmless = std.mem.startsWith(u8, callee, "llvm.lifetime.");
        if (!c.eat('(')) return;
        var arg_index: usize = 0;
        while (true) {
            c.skipWs();
            if (c.eat(')') or c.atEnd()) break;
            const arg = c.argument();
            if (!harmless) {
                const pass = if (ci) |i| blk: {
                    const cf = s.mod.funcs[i];
                    break :blk cf.body != null and arg_index < cf.param_escapes.len and !cf.param_escapes[arg_index] and isPrivatePointer(arg);
                } else false;
                const last = lastName(arg);
                var r = Scanner{ .s = arg };
                while (!r.atEnd()) {
                    const p = r.i;
                    if (r.next() != '%') continue;
                    const ri = s.root_of.get(r.name()) orelse continue;
                    if (!(pass and last != null and last.?.start == p)) s.roots.items[ri].escaped = true;
                }
            }
            arg_index += 1;
            c.skipWs();
            if (!c.eat(',')) break;
        }
    }

    fn escapeAll(s: *Analyser, text: []const u8) void {
        var r = Scanner{ .s = text };
        while (!r.atEnd()) {
            if (r.next() != '%') continue;
            if (s.root_of.get(r.name())) |ri| s.roots.items[ri].escaped = true;
        }
    }

    /// The join of the levels of every `%name` in `text`. Block labels and
    /// named types share the spelling but never match a value.
    fn levelOf(s: *Analyser, text: []const u8) Level {
        var level: Level = .threadgroup;
        var c = Scanner{ .s = text };
        while (!c.atEnd()) {
            if (c.next() != '%') continue;
            level = level.join(s.levelOfName(c.name()));
            if (level == .thread) break;
        }
        return level;
    }

    /// Block `x` ends in a branch on a value of `level`: raise the blocks
    /// that only a subset of the threads reaches, the loops the subset
    /// leaves or re-enters at its own pace, and the joins whose phis merge
    /// values from different subsets. Costs O(region + loops hit), with
    /// O(1) post-dominance and loop-membership queries (`Control`).
    fn expandBranch(s: *Analyser, x: u32, level: Level) Error!void {
        const a = s.a;
        const ctl = try s.control();
        s.epoch += 1;
        const ep = s.epoch;
        s.region.clearRetainingCapacity();
        s.hits.clearRetainingCapacity();
        s.stack.clearRetainingCapacity();
        for (s.cfg.succs[x]) |succ| try s.stack.append(a, succ);
        while (s.stack.pop()) |b| {
            if (s.reached[b] == ep) continue;
            // The region ends at the headers of the loops around `x`: the
            // threads arriving there start another iteration. (Checked
            // before post-dominance: `x` post-dominates itself, but reaching
            // it again through a back edge is exactly this case.)
            const li = ctl.header_loop[b];
            if (li != none and ctl.loops[li].members.isSet(x)) {
                if (s.hit_epoch[li] != ep) {
                    s.hit_epoch[li] = ep;
                    try s.hits.append(a, li);
                }
                continue;
            }
            // A block every path from `x` passes reconverges the threads.
            if (ctl.postDominates(b, x)) continue;
            s.reached[b] = ep;
            try s.region.append(a, b);
            s.raiseBlock(b, level);
            for (s.cfg.succs[b]) |succ| if (s.reached[succ] != ep) try s.stack.append(a, succ);
        }
        // Temporal divergence: some of the threads that passed `x` start
        // the next iteration of a loop while others leave it (an edge out
        // of the loop from `x` or from the region; a path that ends in a
        // trap leaves nothing behind and does not count). Every block of
        // that loop then runs for threads at different iterations, and
        // every value it defines is seen at different iterations after it.
        for (s.hits.items) |li| {
            const l = ctl.loops[li];
            var leaves = false;
            for (s.cfg.succs[x]) |v| if (!l.members.isSet(v) and !s.cfg.traps[v]) {
                leaves = true;
            };
            for (s.region.items) |u| for (s.cfg.succs[u]) |v| if (!l.members.isSet(v) and !s.cfg.traps[v]) {
                leaves = true;
            };
            if (!leaves) continue;
            for (l.list) |i| if (s.reached[i] != ep) {
                s.reached[i] = ep;
                try s.region.append(a, i);
                s.raiseBlock(i, level);
                try s.raiseBlockValues(i, level);
            };
        }
        // Joins: two or more predecessors among the branch and the region.
        if (s.reached[x] != ep) s.countJoins(x, ep, level);
        for (s.region.items) |u| s.countJoins(u, ep, level);
    }

    fn countJoins(s: *Analyser, u: u32, ep: u32, level: Level) void {
        for (s.cfg.succs[u]) |j| {
            if (s.join_epoch[j] != ep) {
                s.join_epoch[j] = ep;
                s.join_count[j] = 0;
            }
            s.join_count[j] += 1;
            if (s.join_count[j] == 2 and level.above(s.join[j])) {
                s.join[j] = level;
                s.local_changed = true;
            }
        }
    }

    /// Every value defined in block `b` (temporal divergence: threads see
    /// the loop's values at different iterations).
    fn raiseBlockValues(s: *Analyser, b: u32, level: Level) Error!void {
        for (s.cfg.blockLines(b)) |li| {
            const line = instructionText(s.body[li]) orelse continue;
            var c = Scanner{ .s = line };
            if (c.peek() != '%') continue;
            _ = c.next();
            const name = c.name();
            c.skipWs();
            if (c.eat('=')) try s.raiseValue(name, level);
        }
    }

    fn control(s: *Analyser) Error!*const Control {
        if (s.ctl == null) {
            s.ctl = try Control.build(s.a, s.cfg);
            s.hit_epoch = try s.a.alloc(u32, s.ctl.?.loops.len);
            @memset(s.hit_epoch, 0);
        }
        return &s.ctl.?;
    }
};

/// A natural loop: its header, its blocks as a set and as a list.
const Loop = struct { header: u32, members: std.DynamicBitSetUnmanaged, list: []u32 };

/// The control-flow facts `expandBranch` needs, in O(N + E) space: the
/// post-dominator tree and the dominator tree (Cooper, Harvey and
/// Kennedy's iterative algorithm, "A Simple, Fast Dominance Algorithm"),
/// each numbered in DFS pre/post order so "a (post-)dominates b" is two
/// comparisons, and the natural loops. The former n x n tables cost O(n^2)
/// memory and an O(n^3) fixpoint (review finding: a 1000-block chain behind
/// one divergent branch took 17 s, 1600 sequential ifs 111 s).
const Control = struct {
    /// The block has a path to a return. Post-dominance is the greatest
    /// fixpoint of `pdom(b) = {b} ∪ ⋂ pdom(succ)` with a returning block's
    /// set `{b}`, so a block with no path to a return (it traps or loops
    /// forever) is post-dominated by every block and post-dominates
    /// nothing that returns: a no-return path does not cut post-dominance
    /// (the block after `if (i >= 64) trap` post-dominates the check).
    to_exit: []bool,
    /// DFS numbering of the post-dominator tree, rooted at a virtual exit
    /// joining the returns.
    pd_pre: []u32,
    pd_post: []u32,
    loops: []Loop,
    /// Per block: the index of the loop it heads, or `none`.
    header_loop: []u32,

    /// Whether `b` post-dominates `x`.
    fn postDominates(c: *const Control, b: u32, x: u32) bool {
        if (!c.to_exit[x]) return true;
        if (!c.to_exit[b]) return false;
        return c.pd_pre[b] <= c.pd_pre[x] and c.pd_post[x] <= c.pd_post[b];
    }

    fn build(a: Allocator, cfg: Cfg) Error!Control {
        const n = cfg.n;
        // Post-dominators: the reverse graph plus a virtual exit `n`.
        const v: u32 = @intCast(n);
        var exits: std.ArrayList(u32) = .empty;
        for (0..n) |b| if (cfg.reach[b] and cfg.succs[b].len == 0 and !cfg.traps[b]) try exits.append(a, @intCast(b));
        const rsucc = try a.alloc([]const u32, n + 1);
        const rpred = try a.alloc([]const u32, n + 1);
        for (0..n) |b| {
            rsucc[b] = cfg.preds[b];
            const is_exit = cfg.reach[b] and cfg.succs[b].len == 0 and !cfg.traps[b];
            if (is_exit) {
                const p = try a.alloc(u32, 1);
                p[0] = v;
                rpred[b] = p;
            } else rpred[b] = cfg.succs[b];
        }
        rsucc[n] = exits.items;
        rpred[n] = &.{};
        const pd = try dominatorTree(a, n + 1, v, rsucc, rpred);
        const to_exit = try a.alloc(bool, n);
        for (to_exit, 0..) |*t, b| t.* = pd.pre[b] != none;

        // Dominators from the entry, for the back edges.
        const succs_c = try a.alloc([]const u32, n);
        const preds_c = try a.alloc([]const u32, n);
        for (0..n) |b| {
            succs_c[b] = cfg.succs[b];
            preds_c[b] = cfg.preds[b];
        }
        const dt = try dominatorTree(a, n, 0, succs_c, preds_c);
        const header_loop = try a.alloc(u32, n);
        @memset(header_loop, none);
        var loops: std.ArrayList(Loop) = .empty;
        var stack: std.ArrayList(u32) = .empty;
        for (0..n) |u| {
            if (!cfg.reach[u] or dt.pre[u] == none) continue;
            for (cfg.succs[u]) |h| {
                // A back edge `u -> h`: `h` dominates `u`.
                if (!(dt.pre[h] <= dt.pre[u] and dt.post[u] <= dt.post[h])) continue;
                if (header_loop[h] == none) {
                    header_loop[h] = @intCast(loops.items.len);
                    var members = try std.DynamicBitSetUnmanaged.initEmpty(a, n);
                    members.set(h);
                    const list = try a.alloc(u32, 1);
                    list[0] = h;
                    try loops.append(a, .{ .header = h, .members = members, .list = list });
                }
                const loop = &loops.items[header_loop[h]];
                var list: std.ArrayList(u32) = .fromOwnedSlice(loop.list);
                stack.clearRetainingCapacity();
                try stack.append(a, @intCast(u));
                while (stack.pop()) |b| {
                    if (loop.members.isSet(b)) continue;
                    loop.members.set(b);
                    try list.append(a, b);
                    for (cfg.preds[b]) |p| try stack.append(a, p);
                }
                loop.list = list.items;
            }
        }
        return .{ .to_exit = to_exit, .pd_pre = pd.pre, .pd_post = pd.post, .loops = loops.items, .header_loop = header_loop };
    }
};

/// A (post-)dominator tree numbered in DFS pre/post order: `a` dominates
/// `b` iff `pre[a] <= pre[b] and post[b] <= post[a]`; `pre[b] == none`
/// for a node the root does not reach.
const DomTree = struct { pre: []u32, post: []u32 };

/// Cooper-Harvey-Kennedy over `succ` from `root` (`pred` is its reverse,
/// possibly naming nodes the root does not reach).
fn dominatorTree(a: Allocator, n: usize, root: u32, succ: []const []const u32, pred: []const []const u32) Error!DomTree {
    // Postorder of a DFS from the root.
    const po = try a.alloc(u32, n);
    @memset(po, none);
    var order: std.ArrayList(u32) = .empty;
    {
        const Frame = struct { node: u32, next: u32 };
        var stack: std.ArrayList(Frame) = .empty;
        const seen = try a.alloc(bool, n);
        @memset(seen, false);
        seen[root] = true;
        try stack.append(a, .{ .node = root, .next = 0 });
        while (stack.items.len > 0) {
            const top = &stack.items[stack.items.len - 1];
            const ss = succ[top.node];
            if (top.next < ss.len) {
                const w = ss[top.next];
                top.next += 1;
                if (!seen[w]) {
                    seen[w] = true;
                    try stack.append(a, .{ .node = w, .next = 0 });
                }
                continue;
            }
            po[top.node] = @intCast(order.items.len);
            try order.append(a, top.node);
            _ = stack.pop();
        }
    }
    const idom = try a.alloc(u32, n);
    @memset(idom, none);
    idom[root] = root;
    var changed = true;
    while (changed) {
        changed = false;
        var k = order.items.len;
        while (k > 0) {
            k -= 1;
            const b = order.items[k];
            if (b == root) continue;
            var new: u32 = none;
            for (pred[b]) |p| {
                if (po[p] == none or idom[p] == none) continue;
                if (new == none) {
                    new = p;
                    continue;
                }
                var x = p;
                var y = new;
                while (x != y) {
                    while (po[x] < po[y]) x = idom[x];
                    while (po[y] < po[x]) y = idom[y];
                }
                new = x;
            }
            if (new != none and idom[b] != new) {
                idom[b] = new;
                changed = true;
            }
        }
    }
    // Children lists, then the pre/post numbering.
    const child_count = try a.alloc(u32, n + 1);
    @memset(child_count, 0);
    for (order.items) |b| if (b != root) {
        child_count[idom[b] + 1] += 1;
    };
    for (1..n + 1) |i| child_count[i] += child_count[i - 1];
    const children = try a.alloc(u32, order.items.len);
    const fill = try a.alloc(u32, n);
    @memcpy(fill, child_count[0..n]);
    for (order.items) |b| if (b != root) {
        children[fill[idom[b]]] = b;
        fill[idom[b]] += 1;
    };
    const pre = try a.alloc(u32, n);
    const post = try a.alloc(u32, n);
    @memset(pre, none);
    @memset(post, none);
    const Frame = struct { node: u32, next: u32 };
    var stack: std.ArrayList(Frame) = .empty;
    var pre_n: u32 = 0;
    var post_n: u32 = 0;
    pre[root] = pre_n;
    pre_n += 1;
    try stack.append(a, .{ .node = root, .next = child_count[root] });
    while (stack.items.len > 0) {
        const top = &stack.items[stack.items.len - 1];
        if (top.next < child_count[top.node + 1]) {
            const w = children[top.next];
            top.next += 1;
            pre[w] = pre_n;
            pre_n += 1;
            try stack.append(a, .{ .node = w, .next = child_count[w] });
            continue;
        }
        post[top.node] = post_n;
        post_n += 1;
        _ = stack.pop();
    }
    return .{ .pre = pre, .post = post };
}

fn analyseOne(gpa: Allocator, mod: *Module, fi: u32) Error!void {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    var s = try Analyser.init(arena.allocator(), mod, fi);
    try s.run();
}

// ── CFG from the text ───────────────────────────────────────────────────────

const Cfg = struct {
    n: usize,
    /// Block index of every body line (label lines belong to their block).
    line_block: []u32,
    /// Distinct successors per block (`label %x` targets of its lines);
    /// empty for a block the entry does not reach.
    succs: [][]u32,
    preds: [][]u32,
    reach: []bool,
    /// The block ends in `unreachable` (a trap: no thread returns from it).
    traps: []bool,
    /// The body lines of block `b` are `lines[line_start[b]..line_start[b + 1]]`.
    line_start: []u32,
    lines: []u32,

    fn blockLines(cfg: Cfg, b: u32) []const u32 {
        return cfg.lines[cfg.line_start[b]..cfg.line_start[b + 1]];
    }

    /// Labels are looked up by name in a hash map and duplicate edges are
    /// caught with a per-block stamp, so a function of N blocks and E
    /// branch targets costs O(N + E) (review finding: a 20 000-block chain
    /// spent most of a minute here and in the assembler's block lookup).
    fn build(a: Allocator, body: []const []const u8) Error!Cfg {
        var names: std.ArrayList([]const u8) = .empty;
        var index: std.StringHashMapUnmanaged(u32) = .empty;
        // The entry block is implicit unless the body opens with a label.
        var first = true;
        for (body) |raw| {
            const line = std.mem.trim(u8, raw, " \t\r");
            if (line.len == 0 or line[0] == ';') continue;
            if (labelName(line)) |name| {
                const gop = try index.getOrPut(a, name);
                if (!gop.found_existing) gop.value_ptr.* = @intCast(names.items.len);
                try names.append(a, name);
            } else if (first) {
                try names.append(a, "");
            }
            first = false;
        }
        if (names.items.len == 0) try names.append(a, "");
        const n = names.items.len;
        // `edge_stamp[t] == cur + 1`: block `cur` already has an edge to `t`.
        const edge_stamp = try a.alloc(u32, n);
        @memset(edge_stamp, 0);
        const line_block = try a.alloc(u32, body.len);
        const edges = try a.alloc(std.ArrayList(u32), n);
        for (edges) |*e| e.* = .empty;
        const traps = try a.alloc(bool, n);
        @memset(traps, false);
        var cur: u32 = 0;
        for (body, 0..) |raw, li| {
            const line = std.mem.trim(u8, raw, " \t\r");
            if (labelName(line)) |name| {
                if (index.get(name)) |i| cur = i;
                line_block[li] = cur;
                continue;
            }
            line_block[li] = cur;
            const text = instructionText(raw) orelse continue;
            if (std.mem.eql(u8, text, "unreachable")) traps[cur] = true;
            var pos: usize = 0;
            while (std.mem.findPos(u8, text, pos, "label %")) |p| {
                var c = Scanner{ .s = text, .i = p + "label %".len };
                const target = c.name();
                pos = c.i;
                const i = index.get(target) orelse continue;
                if (edge_stamp[i] == cur + 1) continue;
                edge_stamp[i] = cur + 1;
                try edges[cur].append(a, i);
            }
        }
        const reach = try a.alloc(bool, n);
        @memset(reach, false);
        {
            var stack: std.ArrayList(u32) = .empty;
            try stack.append(a, 0);
            reach[0] = true;
            while (stack.pop()) |b| for (edges[b].items) |succ| if (!reach[succ]) {
                reach[succ] = true;
                try stack.append(a, succ);
            };
        }
        const succs = try a.alloc([]u32, n);
        const pred_lists = try a.alloc(std.ArrayList(u32), n);
        for (pred_lists) |*p| p.* = .empty;
        for (0..n) |b| {
            succs[b] = if (reach[b]) edges[b].items else edges[b].items[0..0];
            for (succs[b]) |succ| try pred_lists[succ].append(a, @intCast(b));
        }
        const preds = try a.alloc([]u32, n);
        for (0..n) |b| preds[b] = pred_lists[b].items;
        const line_start = try a.alloc(u32, n + 1);
        @memset(line_start, 0);
        for (line_block) |b| line_start[b + 1] += 1;
        for (1..n + 1) |b| line_start[b] += line_start[b - 1];
        const fill = try a.alloc(u32, n);
        @memcpy(fill, line_start[0..n]);
        const lines = try a.alloc(u32, body.len);
        for (line_block, 0..) |b, li| {
            lines[fill[b]] = @intCast(li);
            fill[b] += 1;
        }
        return .{ .n = n, .line_block = line_block, .succs = succs, .preds = preds, .reach = reach, .traps = traps, .line_start = line_start, .lines = lines };
    }
};

/// Whether `text` (after optional blanks) starts with the word `w`.
fn startsWithWord(text: []const u8, w: []const u8) bool {
    const t = std.mem.trimStart(u8, text, " \t");
    return std.mem.startsWith(u8, t, w) and (t.len == w.len or !isIdentChar(t[w.len]));
}

const NameAt = struct { name: []const u8, start: usize };

/// The last `%name` of `text` and the offset of its `%`.
fn lastName(text: []const u8) ?NameAt {
    var c = Scanner{ .s = text };
    var found: ?NameAt = null;
    while (!c.atEnd()) {
        const at = c.i;
        if (c.next() != '%') continue;
        found = .{ .name = c.name(), .start = at };
    }
    return found;
}

/// The base pointer of `getelementptr [flags] T, ptr %base, indices...`
/// (the operand after the first top-level comma) and its offset in `rest`.
fn gepBase(rest: []const u8) ?NameAt {
    var depth: usize = 0;
    var i: usize = 0;
    while (i < rest.len) : (i += 1) switch (rest[i]) {
        '(', '[', '{', '<' => depth += 1,
        ')', ']', '}', '>' => depth -|= 1,
        ',' => if (depth == 0) break,
        else => {},
    };
    if (i >= rest.len) return null;
    var c = Scanner{ .s = rest, .i = i + 1 };
    const arg = c.argument();
    const n = lastName(arg) orelse return null;
    return .{ .name = n.name, .start = i + 1 + n.start };
}

/// Whether a call argument is a thread-space pointer (`ptr %x`, `ptr
/// noundef %x`; Zig's local space 5 is Metal's thread space too).
fn isPrivatePointer(arg: []const u8) bool {
    const t = std.mem.trimStart(u8, arg, " \t");
    if (!startsWithWord(t, "ptr")) return false;
    const after = std.mem.trimStart(u8, t[3..], " \t");
    if (!std.mem.startsWith(u8, after, "addrspace(")) return true;
    return std.mem.startsWith(u8, after, "addrspace(0)") or std.mem.startsWith(u8, after, "addrspace(5)");
}

/// The instruction on a body line: trimmed, with a trailing comment
/// removed; null for blank, comment and label lines.
fn instructionText(raw: []const u8) ?[]const u8 {
    var line = std.mem.trim(u8, raw, " \t\r");
    if (line.len == 0 or line[0] == ';') return null;
    if (labelName(line) != null) return null;
    if (std.mem.findScalar(u8, line, ';')) |semi| line = std.mem.trimEnd(u8, line[0..semi], " \t");
    return line;
}

fn labelName(line: []const u8) ?[]const u8 {
    if (line.len < 2) return null;
    var i: usize = 0;
    while (i < line.len and isIdentChar(line[i])) i += 1;
    if (i == 0 or i >= line.len or line[i] != ':') return null;
    return line[0..i];
}

fn isIdentChar(ch: u8) bool {
    return std.ascii.isAlphanumeric(ch) or ch == '_' or ch == '.' or ch == '$' or ch == '-';
}

const Scanner = struct {
    s: []const u8,
    i: usize = 0,

    fn atEnd(c: *const Scanner) bool {
        return c.i >= c.s.len;
    }
    fn peek(c: *const Scanner) u8 {
        return if (c.i < c.s.len) c.s[c.i] else 0;
    }
    fn next(c: *Scanner) u8 {
        const ch = c.s[c.i];
        c.i += 1;
        return ch;
    }
    fn skipWs(c: *Scanner) void {
        while (c.i < c.s.len and (c.s[c.i] == ' ' or c.s[c.i] == '\t')) c.i += 1;
    }
    fn eat(c: *Scanner, ch: u8) bool {
        if (c.i < c.s.len and c.s[c.i] == ch) {
            c.i += 1;
            return true;
        }
        return false;
    }
    fn rest(c: *const Scanner) []const u8 {
        return c.s[c.i..];
    }
    fn word(c: *Scanner) []const u8 {
        const start = c.i;
        while (c.i < c.s.len and isIdentChar(c.s[c.i])) c.i += 1;
        return c.s[start..c.i];
    }
    /// An identifier after `%` or `@`: plain, or quoted (returned without
    /// the quotes; `\XX` escapes are left as written, the spelling only
    /// has to be consistent within one function).
    fn name(c: *Scanner) []const u8 {
        if (c.eat('"')) {
            const start = c.i;
            while (c.i < c.s.len and c.s[c.i] != '"') : (c.i += 1) {
                if (c.s[c.i] == '\\') c.i += 1;
            }
            const end = @min(c.i, c.s.len);
            _ = c.eat('"');
            return c.s[start..end];
        }
        return c.word();
    }
    /// One call argument: text up to the next top-level `,` or `)`.
    fn argument(c: *Scanner) []const u8 {
        const start = c.i;
        var depth: usize = 0;
        while (c.i < c.s.len) : (c.i += 1) {
            switch (c.s[c.i]) {
                '(', '[', '{', '<' => depth += 1,
                ']', '}', '>' => depth -|= 1,
                ')' => {
                    if (depth == 0) break;
                    depth -= 1;
                },
                ',' => if (depth == 0) break,
                '"' => {
                    c.i += 1;
                    while (c.i < c.s.len and c.s[c.i] != '"') : (c.i += 1) {
                        if (c.s[c.i] == '\\') c.i += 1;
                    }
                },
                else => {},
            }
        }
        return c.s[start..@min(c.i, c.s.len)];
    }
};

// ── Tests ───────────────────────────────────────────────────────────────────

/// Split a function body (the lines between `{` and `}`) and analyse it as
/// the entry with the given divergent parameters, plus optional helpers.
const TestModule = struct {
    funcs: std.ArrayList(Func) = .empty,

    fn add(t: *TestModule, gpa: Allocator, name: []const u8, params: []const []const u8, body: []const u8) !usize {
        var lines: std.ArrayList([]const u8) = .empty;
        var it = std.mem.splitScalar(u8, body, '\n');
        while (it.next()) |l| try lines.append(gpa, l);
        try t.funcs.append(gpa, try Func.init(gpa, name, params, lines.items));
        return t.funcs.items.len - 1;
    }
};

fn lineOf(f: Func, needle: []const u8) usize {
    for (f.body.?, 0..) |l, i| if (std.mem.find(u8, l, needle) != null) return i;
    unreachable;
}

/// The level of the block holding the first line containing `needle`.
fn levelAt(f: Func, needle: []const u8) Level {
    return f.line_level[lineOf(f, needle)];
}

const expectEqual = std.testing.expectEqual;

test "values: per-thread parameters flow through arithmetic, loads and phis; uniform ones do not" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();
    var t = TestModule{};
    const fi = try t.add(gpa, "k", &.{ "in", "tidx", "size" },
        \\  %a = add i32 %size, 1
        \\  %b = mul i32 %tidx, 2
        \\  %p = getelementptr float, ptr addrspace(1) %in, i32 %a
        \\  %q = getelementptr float, ptr addrspace(1) %in, i32 %b
        \\  %u = load float, ptr addrspace(1) %p
        \\  %d = load float, ptr addrspace(1) %q
        \\  %c = icmp eq i32 %a, 4
        \\  br i1 %c, label %x, label %y
        \\x:
        \\  br label %y
        \\y:
        \\  %ph = phi float [ %u, %x ], [ 0.0, %0 ]
        \\  ret void
    );
    t.funcs.items[fi].param_level[1] = .thread;
    try analyse(gpa, t.funcs.items);
    // Re-run one function to inspect the value set through a fresh Analyser.
    var a2 = std.heap.ArenaAllocator.init(gpa);
    defer a2.deinit();
    const a = a2.allocator();
    var mod = try Module.init(a, t.funcs.items);
    var s = try Analyser.init(a, &mod, @intCast(fi));
    try s.run();
    try expectEqual(.threadgroup, s.levelOfName("a"));
    try expectEqual(.thread, s.levelOfName("b"));
    try expectEqual(.threadgroup, s.levelOfName("p"));
    try expectEqual(.thread, s.levelOfName("q"));
    try expectEqual(.threadgroup, s.levelOfName("u"));
    try expectEqual(.thread, s.levelOfName("d"));
    try expectEqual(.threadgroup, s.levelOfName("c"));
    try expectEqual(.threadgroup, s.levelOfName("ph"));
    // The uniform branch made no block divergent.
    for (t.funcs.items[fi].line_level) |ll| try expectEqual(.threadgroup, ll);
}

test "control: jump-threaded diamond arms are divergent, the join after them is not" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();
    var t = TestModule{};
    const fi = try t.add(gpa, "k", &.{ "vals", "tidx" },
        \\  %__3 = icmp eq i32 %__1, 0
        \\  br i1 %__3, label %__4, label %.critedge
        \\__4:
        \\  store i32 0, ptr addrspace(3) @tg_counter, align 4
        \\  tail call void @air.wg.barrier(i32 2, i32 1)
        \\  br label %__6
        \\.critedge:
        \\  tail call void @air.wg.barrier(i32 2, i32 1)
        \\  br label %__6
        \\__6:
        \\  %m = phi i32 [ 1, %__4 ], [ 2, %.critedge ]
        \\  tail call void @air.wg.barrier(i32 2, i32 1)
        \\  ret void
    );
    // The test's parameter names are `%__0`/`%__1` in the body.
    t.funcs.items[fi].param_names = &.{ "__0", "__1" };
    t.funcs.items[fi].param_level[1] = .thread;
    try analyse(gpa, t.funcs.items);
    const f = t.funcs.items[fi];
    try expectEqual(.threadgroup, f.convergent);
    try expectEqual(.thread, levelAt(f, "store i32 0"));
    try expectEqual(.thread, levelAt(f, "__4:"));
    try expectEqual(.thread, f.line_level[lineOf(f, ".critedge:") + 1]);
    try expectEqual(.threadgroup, levelAt(f, "%m = phi"));
    try expectEqual(.threadgroup, levelAt(f, "ret void"));
    try std.testing.expect(!Scope.threadgroup.allows(levelAt(f, "store i32 0")));
    try std.testing.expect(Scope.threadgroup.allows(levelAt(f, "%m = phi")));
}

test "control: a guarded loop with a uniform trip count is uniform even with a per-thread branch inside" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();
    var t = TestModule{};
    // review-b/r4/reduce/reduce.nvptx.ll, renamed the way the splice does.
    const fi = try t.add(gpa, "reduceKernel", &.{ "__0", "__1", "__2", "__3", "__4" },
        \\  %__6 = zext i32 %__2 to i64
        \\  %__7 = getelementptr inbounds [4 x i8], ptr addrspace(3) @red, i64 %__6
        \\  tail call void @air.wg.barrier(i32 2, i32 1)
        \\  %.06 = lshr i32 %__3, 1
        \\  %.not7 = icmp eq i32 %.06, 0
        \\  br i1 %.not7, label %._crit_edge, label %.lr.ph
        \\.lr.ph:                                           ; preds = %__5, %__15
        \\  %.08 = phi i32 [ %.0, %__15 ], [ %.06, %__5 ]
        \\  %__13 = icmp ult i32 %__2, %.08
        \\  br i1 %__13, label %__16, label %__15
        \\._crit_edge:                                      ; preds = %__15, %__5
        \\  %__14 = icmp eq i32 %__2, 0
        \\  br i1 %__14, label %__24, label %__23
        \\__15:                                             ; preds = %.lr.ph, %__16
        \\  tail call void @air.wg.barrier(i32 2, i32 1)
        \\  %.0 = lshr i32 %.08, 1
        \\  %.not = icmp eq i32 %.0, 0
        \\  br i1 %.not, label %._crit_edge, label %.lr.ph
        \\__16:                                             ; preds = %.lr.ph
        \\  %__20 = load float, ptr addrspace(3) %__7, align 4
        \\  store float %__20, ptr addrspace(3) %__7, align 4
        \\  br label %__15
        \\__23:                                             ; preds = %._crit_edge, %__24
        \\  ret void
        \\__24:                                             ; preds = %._crit_edge
        \\  %__27 = load float, ptr addrspace(3) @red, align 4
        \\  store float %__27, ptr addrspace(1) %__1, align 4
        \\  br label %__23
    );
    t.funcs.items[fi].param_level[2] = .thread; // tidx
    try analyse(gpa, t.funcs.items);
    const f = t.funcs.items[fi];
    try expectEqual(.threadgroup, levelAt(f, "%.08 = phi"));
    try expectEqual(.threadgroup, levelAt(f, "%.0 = lshr"));
    try expectEqual(.thread, levelAt(f, "%__20 = load"));
    try expectEqual(.thread, levelAt(f, "%__27 = load"));
    try expectEqual(.threadgroup, levelAt(f, "%__14 = icmp"));
    try expectEqual(.threadgroup, levelAt(f, "ret void"));
}

test "private variables: a loop counter in an alloca is as uniform as what is stored in it" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();
    // glslang's -O0 shape (and Zig's SPIR-V backend's) as tools/spirv/
    // lower.zig prints it: the tree reduction's counter `s` lives in an
    // alloca; `lid` too, stored from the per-thread builtin.
    const reduce =
        \\  %lid = alloca i32, align 4
        \\  %s = alloca i32, align 4
        \\  %t = extractelement <3 x i32> %__0, i32 0
        \\  store i32 %t, ptr %lid, align 4
        \\  store i32 32, ptr %s, align 4
        \\  br label %head
        \\head:
        \\  %sv = load i32, ptr %s, align 4
        \\  %more = icmp ugt i32 %sv, 0
        \\  br i1 %more, label %body, label %exit
        \\body:
        \\  %l = load i32, ptr %lid, align 4
        \\  %lt = icmp ult i32 %l, %sv
        \\  br i1 %lt, label %add, label %join
        \\add:
        \\  store i32 1, ptr addrspace(3) @tile, align 4
        \\  br label %join
        \\join:
        \\  call void @air.wg.barrier(i32 2, i32 1)
        \\  %s2 = load i32, ptr %s, align 4
        \\  %s3 = lshr i32 %s2, STEP
        \\  store i32 %s3, ptr %s, align 4
        \\  br label %head
        \\exit:
        \\  ret void
    ;
    {
        var t = TestModule{};
        const fi = try t.add(gpa, "k", &.{"__0"}, try std.mem.replaceOwned(u8, gpa, reduce, "STEP", "1"));
        t.funcs.items[fi].param_level[0] = .thread;
        try analyse(gpa, t.funcs.items);
        const f = t.funcs.items[fi];
        try expectEqual(.threadgroup, levelAt(f, "%more = icmp"));
        try expectEqual(.thread, levelAt(f, "store i32 1, ptr addrspace(3)"));
        try expectEqual(.threadgroup, levelAt(f, "call void @air.wg.barrier"));
        try expectEqual(Scope.threadgroup, f.convergent);
    }
    {
        // The step is per-thread (`s >>= lid`): the loop's exit is too.
        var t = TestModule{};
        const fi = try t.add(gpa, "k", &.{"__0"}, try std.mem.replaceOwned(u8, gpa, reduce, "lshr i32 %s2, STEP", "lshr i32 %s2, %l"));
        t.funcs.items[fi].param_level[0] = .thread;
        try analyse(gpa, t.funcs.items);
        try expectEqual(.thread, levelAt(t.funcs.items[fi], "call void @air.wg.barrier"));
    }
    {
        // A store only some threads execute makes the variable per-thread.
        var t = TestModule{};
        const fi = try t.add(gpa, "k", &.{"__0"}, try std.mem.replaceOwned(u8, gpa, reduce, "store i32 1, ptr addrspace(3) @tile, align 4", "store i32 0, ptr %s, align 4"));
        t.funcs.items[fi].param_level[0] = .thread;
        try analyse(gpa, t.funcs.items);
        try expectEqual(.thread, levelAt(t.funcs.items[fi], "call void @air.wg.barrier"));
    }
    {
        // An escaping alloca (its address passed to an external call or an
        // intrinsic, compared, used as an index, or stored) keeps the
        // per-thread level of its address. (A getelementptr from it is a
        // derived pointer, not an escape: "private variables: ..." below.)
        for ([_][]const u8{ "call void @f(ptr %s)", "%g = icmp eq ptr %s, null", "%g = getelementptr i32, ptr addrspace(3) @tile, i64 0, ptr %s", "store ptr %s, ptr %lid, align 8", "call void @llvm.memcpy.p0.p0.i64(ptr %s, ptr %lid, i64 4, i1 false)" }) |esc| {
            var t = TestModule{};
            const body = try std.mem.replaceOwned(u8, gpa, reduce, "  br label %head\nhead:", try std.fmt.allocPrint(gpa, "  {s}\n  br label %head\nhead:", .{esc}));
            const fi = try t.add(gpa, "k", &.{"__0"}, try std.mem.replaceOwned(u8, gpa, body, "STEP", "1"));
            t.funcs.items[fi].param_level[0] = .thread;
            try analyse(gpa, t.funcs.items);
            try expectEqual(.thread, levelAt(t.funcs.items[fi], "call void @air.wg.barrier"));
        }
    }
    {
        // `load atomic` is per-thread whatever its address.
        var t = TestModule{};
        const fi = try t.add(gpa, "k", &.{"__0"},
            \\  %v = load atomic i32, ptr addrspace(3) @c monotonic, align 4
            \\  %z = icmp eq i32 %v, 0
            \\  br i1 %z, label %a, label %b
            \\a:
            \\  br label %b
            \\b:
            \\  ret void
        );
        try analyse(gpa, t.funcs.items);
        try expectEqual(.thread, t.funcs.items[fi].line_level[lineOf(t.funcs.items[fi], "a:") + 1]);
    }
}

test "control: a loop some threads leave early is divergent everywhere inside; the block after it is not" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();
    var t = TestModule{};
    const fi = try t.add(gpa, "k", &.{ "__0", "__1" },
        \\  br label %loop
        \\loop:
        \\  %i = phi i32 [ 0, %__2 ], [ %next, %latch ]
        \\  %stop = icmp eq i32 %i, %__1
        \\  br i1 %stop, label %exit, label %body
        \\body:
        \\  tail call void @air.wg.barrier(i32 2, i32 1)
        \\  br label %latch
        \\latch:
        \\  %next = add i32 %i, 1
        \\  %done = icmp eq i32 %next, 8
        \\  br i1 %done, label %exit, label %loop
        \\exit:
        \\  tail call void @air.wg.barrier(i32 2, i32 1)
        \\  ret void
    );
    t.funcs.items[fi].param_level[1] = .thread;
    try analyse(gpa, t.funcs.items);
    const f = t.funcs.items[fi];
    try expectEqual(.thread, levelAt(f, "%i = phi"));
    try expectEqual(.thread, f.line_level[lineOf(f, "body:") + 1]);
    try expectEqual(.thread, levelAt(f, "%next = add"));
    try expectEqual(.threadgroup, f.line_level[lineOf(f, "exit:") + 1]);
    try expectEqual(.threadgroup, levelAt(f, "br label %loop"));
}

test "control: a loop some SIMD-groups leave early is per-SIMD-group inside" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();
    var t = TestModule{};
    // `for (0..8) |i| { if (i == sgid) break; barrier; simd_sum }`
    const fi = try t.add(gpa, "k", &.{ "__0", "__1" },
        \\  br label %loop
        \\loop:
        \\  %i = phi i32 [ 0, %__2 ], [ %next, %latch ]
        \\  %stop = icmp eq i32 %i, %__1
        \\  br i1 %stop, label %exit, label %body
        \\body:
        \\  tail call void @air.wg.barrier(i32 2, i32 1)
        \\  %s = tail call float @air.simd_sum.f32(float 1.0)
        \\  br label %latch
        \\latch:
        \\  %next = add i32 %i, 1
        \\  %done = icmp eq i32 %next, 8
        \\  br i1 %done, label %exit, label %loop
        \\exit:
        \\  ret void
    );
    t.funcs.items[fi].param_level[1] = .simdgroup;
    try analyse(gpa, t.funcs.items);
    const f = t.funcs.items[fi];
    try expectEqual(.simdgroup, levelAt(f, "%i = phi"));
    try expectEqual(.simdgroup, levelAt(f, "@air.wg.barrier"));
    try expectEqual(.simdgroup, levelAt(f, "%next = add"));
    try expectEqual(.threadgroup, levelAt(f, "ret void"));
    try std.testing.expect(!Scope.threadgroup.allows(levelAt(f, "@air.wg.barrier")));
    try std.testing.expect(Scope.simdgroup.allows(levelAt(f, "%s = tail call")));
}

test "helpers: convergence scope and argument levels propagate over call edges" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();
    var t = TestModule{};
    const k = try t.add(gpa, "k", &.{ "__0", "__1" },
        \\  %__3 = icmp eq i32 %__1, 0
        \\  br i1 %__3, label %__4, label %.critedge
        \\__4:
        \\  tail call fastcc void @"m.syncTg"()
        \\  %__8 = tail call fastcc i32 @"m.twice"(i32 %__1)
        \\  br label %__6
        \\.critedge:
        \\  tail call fastcc void @"m.syncTg"()
        \\  br label %__6
        \\__6:
        \\  %__9 = tail call fastcc i32 @"m.twice"(i32 4)
        \\  %__10 = tail call fastcc i32 @"m.inner"(i32 4)
        \\  %__11 = tail call fastcc float @"m.simdOnly"(float 1.0)
        \\  ret void
    );
    const sync = try t.add(gpa, "m.syncTg", &.{},
        \\  tail call fastcc void @"m.inner"()
        \\  ret void
    );
    const inner = try t.add(gpa, "m.inner", &.{},
        \\  tail call void @air.wg.barrier(i32 2, i32 1)
        \\  ret void
    );
    const twice = try t.add(gpa, "m.twice", &.{"__0"},
        \\  %__2 = shl i32 %__0, 1
        \\  ret i32 %__2
    );
    const simd_only = try t.add(gpa, "m.simdOnly", &.{"__0"},
        \\  %__2 = tail call float @air.simd_sum.f32(float %__0)
        \\  ret float %__2
    );
    t.funcs.items[k].param_level[1] = .thread;
    try analyse(gpa, t.funcs.items);
    const fs = t.funcs.items;
    try expectEqual(.threadgroup, fs[inner].convergent);
    try expectEqual(.threadgroup, fs[sync].convergent);
    try expectEqual(.threadgroup, fs[k].convergent);
    try expectEqual(.none, fs[twice].convergent);
    try expectEqual(.simdgroup, fs[simd_only].convergent);
    try expectEqual(.simdgroup, fs[simd_only].ret_level);
    try expectEqual(.thread, fs[twice].param_level[0]);
    try expectEqual(.thread, fs[twice].ret_level);
    try expectEqual(.thread, fs[k].line_level[lineOf(fs[k], "__4:") + 1]);
    try expectEqual(.thread, fs[k].line_level[lineOf(fs[k], ".critedge:") + 1]);
    try expectEqual(.threadgroup, levelAt(fs[k], "%__9 ="));
}

test "helpers: a constant returned from a block only some threads reach is a per-thread return" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();
    var t = TestModule{};
    // review2/v2/edited/shader.ll: `ticket()` returns 1 for the thread that
    // won the atomic and 2 for every other one; both `ret` operands are
    // constants. The caller branches on the result and LLVM jump-threaded
    // its barriers into both arms (`FAIL dispatch multiRetKernel: 8
    // mismatches`).
    const k = try t.add(gpa, "multiRetKernel", &.{ "__0", "__1", "__2", "__3" },
        \\  %__5 = zext i32 %__3 to i64
        \\  %__6 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %__0, i64 %__5
        \\  %__7 = tail call fastcc i32 @my_shader.ticket(ptr addrspace(1) %__6, ptr addrspace(1) %__2)
        \\  %__8 = icmp eq i32 %__7, 1
        \\  br i1 %__8, label %__9, label %.critedge
        \\__9:
        \\  store i32 0, ptr addrspace(3) @my_shader.tg_count2, align 4
        \\  tail call void @air.wg.barrier(i32 2, i32 1)
        \\  br label %__14
        \\.critedge:
        \\  tail call void @air.wg.barrier(i32 2, i32 1)
        \\  br label %__14
        \\__14:
        \\  ret void
    );
    const ticket = try t.add(gpa, "my_shader.ticket", &.{ "__0", "__1" },
        \\  %__3 = tail call i32 @air.atomic.global.add.u.i32(ptr addrspace(1) %__0, i32 1, i32 0, i32 2, i1 true)
        \\  %__4 = icmp eq i32 %__3, 0
        \\  br i1 %__4, label %__5, label %common.ret
        \\common.ret:
        \\  ret i32 2
        \\__5:
        \\  store i32 7, ptr addrspace(1) %__1, align 4
        \\  ret i32 1
    );
    // Even with uniform arguments (`%__0` is only the pointer here).
    t.funcs.items[k].param_level[3] = .threadgroup;
    try analyse(gpa, t.funcs.items);
    const fs = t.funcs.items;
    try expectEqual(.thread, fs[ticket].ret_level);
    try expectEqual(.thread, levelAt(fs[k], "store i32 0"));
    try expectEqual(.thread, fs[k].line_level[lineOf(fs[k], ".critedge:") + 1]);
    try expectEqual(.threadgroup, levelAt(fs[k], "ret void"));
    // A helper whose only per-thread `ret` returns void changes nothing.
    var t2 = TestModule{};
    const v = try t2.add(gpa, "m.v", &.{"__0"},
        \\  %__2 = icmp eq i32 %__0, 0
        \\  br i1 %__2, label %a, label %b
        \\a:
        \\  ret void
        \\b:
        \\  ret void
    );
    t2.funcs.items[v].param_level[0] = .thread;
    try analyse(gpa, t2.funcs.items);
    try expectEqual(.thread, t2.funcs.items[v].ret_level);
}

test "calls: SIMD-group reductions are per-SIMD-group, shuffles and atomics per-thread, other calls follow their arguments" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();
    var t = TestModule{};
    const fi = try t.add(gpa, "k", &.{ "__0", "__1" },
        \\  %s = tail call float @air.simd_sum.f32(float %__1)
        \\  %sh = tail call float @air.simd_shuffle_down.f32(float 1.0, i16 1)
        \\  %o = tail call i32 @air.atomic.global.add.u.i32(ptr addrspace(1) %__0, i32 1, i32 0, i32 2, i1 true)
        \\  %r = tail call float @llvm.sqrt.f32(float 2.0)
        \\  %c1 = fcmp ogt float %s, 0.0
        \\  br i1 %c1, label %a, label %b
        \\a:
        \\  tail call void @air.wg.barrier(i32 2, i32 1)
        \\  %s2 = tail call float @air.simd_sum.f32(float %sh)
        \\  br label %b
        \\b:
        \\  %c2 = fcmp ogt float %r, 0.0
        \\  br i1 %c2, label %c, label %d
        \\c:
        \\  tail call void @air.wg.barrier(i32 2, i32 1)
        \\  br label %d
        \\d:
        \\  %c3 = fcmp ogt float %sh, 0.0
        \\  br i1 %c3, label %e, label %f
        \\e:
        \\  %s3 = tail call float @air.simd_sum.f32(float 1.0)
        \\  br label %f
        \\f:
        \\  ret void
    );
    t.funcs.items[fi].param_level[1] = .thread;
    try analyse(gpa, t.funcs.items);
    const f = t.funcs.items[fi];
    // A reduction of per-thread lanes is per-SIMD-group; a branch on it
    // admits SIMD-group calls, not threadgroup barriers.
    try expectEqual(.simdgroup, f.line_level[lineOf(f, "a:") + 1]);
    try std.testing.expect(!Scope.threadgroup.allows(levelAt(f, "%s2 =")));
    try std.testing.expect(Scope.simdgroup.allows(levelAt(f, "%s2 =")));
    try expectEqual(.threadgroup, f.line_level[lineOf(f, "c:") + 1]);
    try expectEqual(.threadgroup, levelAt(f, "%c2 = fcmp"));
    // A branch on a shuffle result is per-thread: no SIMD-group call.
    try expectEqual(.thread, levelAt(f, "%s3 ="));
    try std.testing.expect(!Scope.simdgroup.allows(levelAt(f, "%s3 =")));
}

test "two-stage reduction: simd_sum under `if (sgid == 0)` is per-SIMD-group, the barrier before it uniform" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();
    var t = TestModule{};
    // review2/v1/.zig-cache/o/*/shader.ll @twoStageKernel (in, out, gid,
    // lane, sgid, nsg, tg_pos), renamed the way the splice does.
    const fi = try t.add(gpa, "twoStageKernel", &.{ "__0", "__1", "__2", "__3", "__4", "__5", "__6" },
        \\  %__8 = zext i32 %__2 to i64
        \\  %__9 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %__0, i64 %__8
        \\  %__10 = load float, ptr addrspace(1) %__9, align 4
        \\  %__11 = tail call float @air.simd_sum.f32(float %__10)
        \\  %__12 = icmp eq i32 %__3, 0
        \\  br i1 %__12, label %__15, label %__13
        \\__13:
        \\  tail call void @air.wg.barrier(i32 2, i32 1)
        \\  %__14 = icmp eq i32 %__4, 0
        \\  br i1 %__14, label %__19, label %__18
        \\__15:
        \\  %__16 = zext i32 %__4 to i64
        \\  %__17 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @my_shader.partial, i64 %__16
        \\  store float %__11, ptr addrspace(3) %__17, align 4
        \\  br label %__13
        \\__18:
        \\  ret void
        \\__19:
        \\  %__20 = icmp ult i32 %__3, %__5
        \\  br i1 %__20, label %__24, label %__21
        \\__21:
        \\  %__22 = phi float [ %__27, %__24 ], [ 0.000000e+00, %__19 ]
        \\  %__23 = tail call float @air.simd_sum.f32(float %__22)
        \\  br i1 %__12, label %__28, label %__18
        \\__24:
        \\  %__25 = zext i32 %__3 to i64
        \\  %__26 = getelementptr inbounds nuw [4 x i8], ptr addrspace(3) @my_shader.partial, i64 %__25
        \\  %__27 = load float, ptr addrspace(3) %__26, align 4
        \\  br label %__21
        \\__28:
        \\  %__29 = zext i32 %__6 to i64
        \\  %__30 = getelementptr inbounds nuw [4 x i8], ptr addrspace(1) %__1, i64 %__29
        \\  store float %__23, ptr addrspace(1) %__30, align 4
        \\  br label %__18
    );
    const pl = t.funcs.items[fi].param_level;
    pl[2] = .thread; // gid
    pl[3] = .thread; // lane
    pl[4] = .simdgroup; // sgid
    try analyse(gpa, t.funcs.items);
    const f = t.funcs.items[fi];
    try expectEqual(.threadgroup, levelAt(f, "%__11 = tail call"));
    try expectEqual(.thread, levelAt(f, "store float %__11"));
    try expectEqual(.threadgroup, levelAt(f, "@air.wg.barrier"));
    try expectEqual(.threadgroup, levelAt(f, "%__14 = icmp"));
    try expectEqual(.simdgroup, levelAt(f, "%__20 = icmp"));
    try expectEqual(.simdgroup, levelAt(f, "%__23 = tail call"));
    try expectEqual(.thread, levelAt(f, "%__27 = load"));
    try expectEqual(.thread, levelAt(f, "store float %__23"));
    try expectEqual(.threadgroup, levelAt(f, "ret void"));
    try std.testing.expect(Scope.simdgroup.allows(levelAt(f, "%__23 = tail call")));
    try std.testing.expect(!Scope.threadgroup.allows(levelAt(f, "%__23 = tail call")));
    try std.testing.expect(Scope.threadgroup.allows(levelAt(f, "@air.wg.barrier")));
}

test "control: a block that traps (`unreachable`) does not cut post-dominance; the code after a safety check stays uniform" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();
    var t = TestModule{};
    // review2/v3/.zig-cache/o/*/shader.ll @reverseKernel at ReleaseSafe:
    // `scratch[tidx]` bounds-checked, the failing path is
    // `llvm.trap(); unreachable`.
    const fi = try t.add(gpa, "reverseKernel", &.{ "__0", "__1", "__2" },
        \\  %__4 = icmp ult i32 %__2, 64
        \\  br i1 %__4, label %__5, label %__8
        \\__5:
        \\  %__6 = zext i32 %__2 to i64
        \\  %__7 = getelementptr inbounds [4 x i8], ptr addrspace(3) @scratch, i64 %__6
        \\  store float 1.0, ptr addrspace(3) %__7, align 4
        \\  tail call void @air.wg.barrier(i32 2, i32 1)
        \\  ret void
        \\__8:
        \\  tail call void @llvm.trap()
        \\  unreachable
    );
    t.funcs.items[fi].param_level[2] = .thread;
    try analyse(gpa, t.funcs.items);
    const f = t.funcs.items[fi];
    try expectEqual(.threadgroup, levelAt(f, "@air.wg.barrier"));
    try expectEqual(.threadgroup, levelAt(f, "ret void"));
    try expectEqual(.thread, levelAt(f, "@llvm.trap"));
    // The same check inside a uniform loop: the loop stays uniform, the
    // trap edge is not a per-thread exit.
    var t2 = TestModule{};
    const li = try t2.add(gpa, "k", &.{ "__0", "__1" },
        \\  br label %loop
        \\loop:
        \\  %i = phi i32 [ 0, %__2 ], [ %next, %latch ]
        \\  %inb = icmp ult i32 %__1, 64
        \\  br i1 %inb, label %body, label %trap
        \\trap:
        \\  tail call void @llvm.trap()
        \\  unreachable
        \\body:
        \\  tail call void @air.wg.barrier(i32 2, i32 1)
        \\  br label %latch
        \\latch:
        \\  %next = add i32 %i, 1
        \\  %done = icmp eq i32 %next, %__0
        \\  br i1 %done, label %exit, label %loop
        \\exit:
        \\  ret void
    );
    t2.funcs.items[li].param_level[1] = .thread;
    try analyse(gpa, t2.funcs.items);
    const g = t2.funcs.items[li];
    try expectEqual(.threadgroup, levelAt(g, "%i = phi"));
    try expectEqual(.threadgroup, levelAt(g, "@air.wg.barrier"));
    try expectEqual(.threadgroup, levelAt(g, "%next = add"));
    try expectEqual(.thread, levelAt(g, "@llvm.trap"));
}

test "private variables: glslang passes helper arguments by pointer; the helper reads the caller's variable" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();
    // review round 2 (find-real-world-1/t8.spvasm), as tools/spirv/lower.zig
    // prints glslang -V's `uint tiles = ceilDiv(pc.n, 64u); for (t = 0; t <
    // tiles; t++) { ...; barrier(); }`: every argument is copied into a
    // Function variable and passed by pointer.
    const kernel =
        \\  %tiles = alloca i32, align 4
        \\  %param = alloca i32, align 4
        \\  %param_0 = alloca i32, align 4
        \\  %t = alloca i32, align 4
        \\  %lid = extractelement <3 x i32> %__0, i32 0
        \\  %n = load i32, ptr addrspace(2) %__1, align 4
        \\  store i32 SRC, ptr %param, align 4
        \\  store i32 64, ptr %param_0, align 4
        \\  CALL
        \\  store i32 0, ptr %t, align 4
        \\  br label %head
        \\head:
        \\  %tv = load i32, ptr %t, align 4
        \\  %tn = load i32, ptr %tiles, align 4
        \\  %more = icmp ult i32 %tv, %tn
        \\  br i1 %more, label %body, label %exit
        \\body:
        \\  call void @air.wg.barrier(i32 2, i32 1)
        \\  %t2 = add i32 %tv, 1
        \\  store i32 %t2, ptr %t, align 4
        \\  br label %head
        \\exit:
        \\  ret void
    ;
    const ceil_div =
        \\  %x = load i32, ptr %a, align 4
        \\  %y = load i32, ptr %b, align 4
        \\  %s = add i32 %x, %y
        \\  %s1 = sub i32 %s, 1
        \\  %d = udiv i32 %s1, %y
        \\  ESCAPE
        \\  ret i32 %d
    ;
    const call_ceil = "%r = call i32 @ceilDiv(ptr %param, ptr %param_0)\n  store i32 %r, ptr %tiles, align 4";
    const Case = struct { src: []const u8, call: []const u8, escape: []const u8 = "", want: Level };
    const cases = [_]Case{
        // A uniform push constant through the helper: accepted.
        .{ .src = "%n", .call = call_ceil, .want = .threadgroup },
        // The thread id through the helper: refused.
        .{ .src = "%lid", .call = call_ceil, .want = .thread },
        // The helper lets the pointer escape (an external call): the
        // variable is no longer followed.
        .{ .src = "%n", .call = call_ceil, .escape = "call void @ext(ptr %a)", .want = .thread },
        // `out` parameters: what the helper stores through the pointer
        // lands in the caller's variable.
        .{ .src = "%n", .call = "call void @put(ptr %tiles, i32 %n)", .want = .threadgroup },
        .{ .src = "%n", .call = "call void @put(ptr %tiles, i32 %lid)", .want = .thread },
        // ... also for the threads that make the call: a uniform value
        // stored by a helper only thread 0 calls.
        .{ .src = "%n", .call = "store i32 4, ptr %tiles, align 4\n  %z = icmp eq i32 %lid, 0\n  br i1 %z, label %one, label %rest\none:\n  call void @put(ptr %tiles, i32 %n)\n  br label %rest\nrest:", .want = .thread },
        // A helper that stores what it read back through another pointer.
        .{ .src = "%lid", .call = "call void @copy(ptr %tiles, ptr %param)", .want = .thread },
        .{ .src = "%n", .call = "call void @copy(ptr %tiles, ptr %param)", .want = .threadgroup },
    };
    for (cases) |cs| {
        var t = TestModule{};
        var body = try std.mem.replaceOwned(u8, gpa, kernel, "SRC", cs.src);
        body = try std.mem.replaceOwned(u8, gpa, body, "CALL", cs.call);
        const k = try t.add(gpa, "main", &.{ "__0", "__1" }, body);
        _ = try t.add(gpa, "ceilDiv", &.{ "a", "b" }, try std.mem.replaceOwned(u8, gpa, ceil_div, "ESCAPE", cs.escape));
        _ = try t.add(gpa, "put", &.{ "o", "v" },
            \\  store i32 %v, ptr %o, align 4
            \\  ret void
        );
        _ = try t.add(gpa, "copy", &.{ "dst", "src" },
            \\  %v = load i32, ptr %src, align 4
            \\  store i32 %v, ptr %dst, align 4
            \\  ret void
        );
        t.funcs.items[k].param_level[0] = .thread;
        try analyse(gpa, t.funcs.items);
        expectEqual(cs.want, levelAt(t.funcs.items[k], "@air.wg.barrier")) catch |err| {
            std.debug.print("case src={s} call={s} escape={s}\n", .{ cs.src, cs.call, cs.escape });
            return err;
        };
    }
}

test "private variables: struct members and vector lanes read through getelementptr" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();
    // find-real-world-1/t8_struct and t8_vec: `P p = P(pc.n); ... p.n` and
    // `uvec2 r = uvec2(pc.n, 64u); ... r.x`, read through OpAccessChain.
    const kernel =
        \\  %p = alloca { i32, i32 }, align 4
        \\  %arr = alloca [64 x i32], align 4
        \\  %t = alloca i32, align 4
        \\  %lid = extractelement <3 x i32> %__0, i32 0
        \\  %n = load i32, ptr addrspace(2) %__1, align 4
        \\  %v0 = insertvalue { i32, i32 } undef, i32 SRC, 0
        \\  %v1 = insertvalue { i32, i32 } %v0, i32 64, 1
        \\  store { i32, i32 } %v1, ptr %p, align 4
        \\  EXTRA
        \\  store i32 0, ptr %t, align 4
        \\  br label %head
        \\head:
        \\  %pn = getelementptr { i32, i32 }, ptr %p, i64 0, i32 0
        \\  %nn = load i32, ptr %pn, align 4
        \\  %tv = load i32, ptr %t, align 4
        \\  %more = icmp ult i32 %tv, %nn
        \\  br i1 %more, label %body, label %exit
        \\body:
        \\  call void @air.wg.barrier(i32 2, i32 1)
        \\  %t2 = add i32 %tv, 1
        \\  store i32 %t2, ptr %t, align 4
        \\  br label %head
        \\exit:
        \\  ret void
    ;
    const Case = struct { src: []const u8, extra: []const u8 = "", want: Level };
    const cases = [_]Case{
        .{ .src = "%n", .want = .threadgroup },
        .{ .src = "%lid", .want = .thread },
        // A per-thread member stored through its own getelementptr.
        .{ .src = "%n", .extra = "%q = getelementptr { i32, i32 }, ptr %p, i64 0, i32 0\n  store i32 %lid, ptr %q, align 4", .want = .thread },
        // A uniform value stored at a per-thread index: the copies differ.
        .{ .src = "%n", .extra = "%e = getelementptr [64 x i32], ptr %arr, i64 0, i32 %lid\n  store i32 1, ptr %e, align 4\n  %e0 = getelementptr [64 x i32], ptr %arr, i64 0, i32 0\n  %a0 = load i32, ptr %e0, align 4\n  store i32 %a0, ptr %t, align 4", .want = .thread },
        // A uniform store to a uniform element stays uniform.
        .{ .src = "%n", .extra = "%e = getelementptr [64 x i32], ptr %arr, i64 0, i32 3\n  store i32 1, ptr %e, align 4\n  %e0 = getelementptr [64 x i32], ptr %arr, i64 0, i32 3\n  %a0 = load i32, ptr %e0, align 4\n  store i32 %a0, ptr %t, align 4", .want = .threadgroup },
        // A derived pointer that escapes takes its variable with it.
        .{ .src = "%n", .extra = "%q = getelementptr { i32, i32 }, ptr %p, i64 0, i32 0\n  call void @ext(ptr %q)", .want = .thread },
    };
    for (cases) |cs| {
        var t = TestModule{};
        var body = try std.mem.replaceOwned(u8, gpa, kernel, "SRC", cs.src);
        body = try std.mem.replaceOwned(u8, gpa, body, "EXTRA", cs.extra);
        const k = try t.add(gpa, "main", &.{ "__0", "__1" }, body);
        t.funcs.items[k].param_level[0] = .thread;
        try analyse(gpa, t.funcs.items);
        expectEqual(cs.want, levelAt(t.funcs.items[k], "@air.wg.barrier")) catch |err| {
            std.debug.print("case src={s} extra={s}\n", .{ cs.src, cs.extra });
            return err;
        };
    }
}

test "scaling: thousands of blocks behind divergent branches, thousands of helpers, deep call chains" {
    // Review round 2: the n x n post-dominator fixpoint took 17 s for a
    // 1000-block chain behind one divergent branch and 111 s for 1600
    // sequential divergent ifs; a call chain needed one round over the
    // whole module per level. These sizes would not finish with those
    // algorithms; here they take milliseconds.
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();
    const n = 20000;
    {
        // One divergent branch, then a chain of n blocks back to a join.
        var text: std.Io.Writer.Allocating = .init(gpa);
        const w = &text.writer;
        try w.writeAll("  %c = icmp eq i32 %__0, 0\n  br i1 %c, label %b0, label %end\n");
        for (0..n) |i| try w.print("b{d}:\n  br label %{s}{d}\n", .{ i, if (i + 1 < n) "b" else "end", if (i + 1 < n) i + 1 else 0 });
        try w.writeAll("end0:\n  br label %end\nend:\n  call void @air.wg.barrier(i32 2, i32 1)\n  ret void");
        var t = TestModule{};
        const k = try t.add(gpa, "k", &.{"__0"}, text.written());
        t.funcs.items[k].param_level[0] = .thread;
        try analyse(gpa, t.funcs.items);
        const f = t.funcs.items[k];
        try expectEqual(.thread, f.line_level[lineOf(f, "b0:") + 1]);
        try expectEqual(.thread, f.line_level[lineOf(f, "end0:") + 1]);
        try expectEqual(.threadgroup, levelAt(f, "@air.wg.barrier"));
    }
    {
        // n sequential `if (x == i) {}` on a per-thread x, with a
        // barrier in every join.
        var text: std.Io.Writer.Allocating = .init(gpa);
        const w = &text.writer;
        try w.writeAll("  br label %j0\n");
        for (0..n) |i| try w.print("j{d}:\n  call void @air.wg.barrier(i32 2, i32 1)\n  %c{d} = icmp eq i32 %__0, {d}\n  br i1 %c{d}, label %a{d}, label %j{d}\na{d}:\n  store i32 1, ptr addrspace(3) @x, align 4\n  br label %j{d}\n", .{ i, i, i, i, i, i + 1, i, i + 1 });
        try w.print("j{d}:\n  ret void", .{n});
        var t = TestModule{};
        const k = try t.add(gpa, "k", &.{"__0"}, text.written());
        t.funcs.items[k].param_level[0] = .thread;
        try analyse(gpa, t.funcs.items);
        const f = t.funcs.items[k];
        for (f.body.?, f.line_level) |l, lv| {
            if (std.mem.find(u8, l, "@air.wg.barrier") != null) try expectEqual(.threadgroup, lv);
            if (std.mem.find(u8, l, "store i32 1") != null) try expectEqual(.thread, lv);
        }
    }
    {
        // n uniform loops in sequence, each with a per-thread branch and a
        // barrier inside.
        var text: std.Io.Writer.Allocating = .init(gpa);
        const w = &text.writer;
        const loops = 2000;
        try w.writeAll("  br label %h0\n");
        for (0..loops) |i| try w.print(
            \\h{d}:
            \\  %i{d} = phi i32 [ 0, %{s}{d} ], [ %n{d}, %l{d} ]
            \\  %c{d} = icmp eq i32 %__0, %i{d}
            \\  br i1 %c{d}, label %a{d}, label %l{d}
            \\a{d}:
            \\  store i32 1, ptr addrspace(3) @x, align 4
            \\  br label %l{d}
            \\l{d}:
            \\  call void @air.wg.barrier(i32 2, i32 1)
            \\  %n{d} = add i32 %i{d}, 1
            \\  %d{d} = icmp eq i32 %n{d}, 8
            \\  br i1 %d{d}, label %h{d}, label %h{d}
            \\
        , .{ i, i, if (i == 0) "__1" else "l", if (i == 0) 0 else i - 1, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i + 1, i });
        try w.print("h{d}:\n  ret void", .{loops});
        var t = TestModule{};
        const k = try t.add(gpa, "k", &.{"__0"}, text.written());
        t.funcs.items[k].param_level[0] = .thread;
        try analyse(gpa, t.funcs.items);
        const f = t.funcs.items[k];
        for (f.body.?, f.line_level) |l, lv| {
            if (std.mem.find(u8, l, "@air.wg.barrier") != null) try expectEqual(.threadgroup, lv);
            if (std.mem.find(u8, l, "store i32 1") != null) try expectEqual(.thread, lv);
        }
    }
    {
        // A flat module of n helpers and a call chain n deep, the chain's
        // last helper making a barrier; every helper defined after its
        // caller, so the convergence scope flows against the order.
        var t = TestModule{};
        var calls: std.Io.Writer.Allocating = .init(gpa);
        for (0..n) |i| try calls.writer.print("  %r{d} = call i32 @h{d}(i32 %__0)\n", .{ i, i });
        try calls.writer.writeAll("  call void @c0()\n  ret void");
        const k = try t.add(gpa, "k", &.{"__0"}, calls.written());
        for (0..n) |i| _ = try t.add(gpa, try std.fmt.allocPrint(gpa, "h{d}", .{i}), &.{"x"}, "  %y = add i32 %x, 1\n  ret i32 %y");
        for (0..n) |i| _ = try t.add(gpa, try std.fmt.allocPrint(gpa, "c{d}", .{i}), &.{}, if (i + 1 < n)
            try std.fmt.allocPrint(gpa, "  call void @c{d}()\n  ret void", .{i + 1})
        else
            "  call void @air.wg.barrier(i32 2, i32 1)\n  ret void");
        t.funcs.items[k].param_level[0] = .thread;
        try analyse(gpa, t.funcs.items);
        try expectEqual(Scope.threadgroup, t.funcs.items[k].convergent);
        try expectEqual(Scope.threadgroup, t.funcs.items[1 + n].convergent);
        try expectEqual(Level.thread, t.funcs.items[n].ret_level);
    }
}
