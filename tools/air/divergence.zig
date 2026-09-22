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
//! The IR keeps no trace of the duplication, so the assembler decides from
//! the CFG alone which threads reach a call site together. A purely
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
//!     most divergent operand; `alloca` (per-thread storage), `atomicrmw`/
//!     `cmpxchg`, `air.atomic.*`, the SIMD-group shuffles and prefix
//!     operations are `.thread` whatever their operands, and the SIMD-group
//!     reductions (`air.simd_sum`, `_max`, `_min`, `_product`, `_and`,
//!     `_or`, `_xor`, `_any`, `_all`, `_broadcast_first`) are `.simdgroup`
//!     whatever their operands. A `load` is as divergent as its address. A
//!     `phi` is as divergent as its incoming values and as the branches
//!     whose paths its block joins. Helpers (`define private fastcc`) take
//!     the level of their arguments over every call site (interprocedural
//!     fixpoint); a call returns the join of its arguments' levels and the
//!     callee's return level.
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
//! per-thread (values) or as an ordinary instruction (control flow).

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
    /// How the value this function returns may differ between threads.
    ret_level: Level = .threadgroup,
    /// The widest scope of a convergent call it executes, directly or
    /// through a callee.
    convergent: Scope = .none,
    /// Per body line: which threads reach the block holding it together.
    /// Empty for a `declare`.
    line_level: []Level,

    pub fn init(gpa: Allocator, name: []const u8, param_names: []const []const u8, body: ?[]const []const u8) Error!Func {
        const pl = try gpa.alloc(Level, param_names.len);
        errdefer gpa.free(pl);
        @memset(pl, .threadgroup);
        const ll = try gpa.alloc(Level, if (body) |b| b.len else 0);
        @memset(ll, .threadgroup);
        return .{ .name = name, .param_names = param_names, .body = body, .param_level = pl, .line_level = ll };
    }

    pub fn deinit(f: *Func, gpa: Allocator) void {
        gpa.free(f.param_level);
        gpa.free(f.line_level);
    }
};

/// Run the analysis over every function to a fixpoint. Only the entry
/// point's `param_level` needs to be seeded; the results of functions
/// nothing reaches are meaningless but harmless (their parameters stay
/// uniform, so no divergence flows out of them).
pub fn analyse(gpa: Allocator, funcs: []Func) Error!void {
    var changed = true;
    while (changed) {
        changed = false;
        for (funcs, 0..) |*f, i| {
            if (f.body == null) continue;
            if (try analyseOne(gpa, funcs, i)) changed = true;
        }
    }
}

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

const Analyser = struct {
    a: Allocator,
    funcs: []Func,
    fi: usize,
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
    pdom: ?[]bool = null,
    loops: ?[]Loop = null,
    ret_level: Level = .threadgroup,
    convergent: Scope = .none,
    inter_changed: bool = false,
    local_changed: bool = false,

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

    fn run(s: *Analyser) Error!void {
        const f = &s.funcs[s.fi];
        for (f.param_names, f.param_level) |p, l| if (l != .threadgroup) try s.div.put(s.a, p, l);
        s.local_changed = true;
        while (s.local_changed) {
            s.local_changed = false;
            for (s.body, 0..) |raw, li| try s.visitLine(raw, li);
        }
        for (f.line_level, 0..) |*ll, li| ll.* = s.block_level[s.cfg.line_block[li]];
        if (s.ret_level.above(f.ret_level)) {
            f.ret_level = s.ret_level;
            s.inter_changed = true;
        }
        if (@backingInt(s.convergent) > @backingInt(f.convergent)) {
            f.convergent = s.convergent;
            s.inter_changed = true;
        }
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
        } else if (std.mem.eql(u8, op, "alloca") or std.mem.eql(u8, op, "atomicrmw") or std.mem.eql(u8, op, "cmpxchg")) {
            level = .thread;
        } else if (std.mem.eql(u8, op, "call")) {
            level = try s.visitCall(rest);
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
    fn visitCall(s: *Analyser, rest: []const u8) Error!Level {
        const at = std.mem.findScalar(u8, rest, '@') orelse return s.levelOf(rest);
        var c = Scanner{ .s = rest, .i = at + 1 };
        const callee = c.name();
        c.skipWs();
        var args_level: Level = .threadgroup;
        var callee_index: ?usize = null;
        for (s.funcs, 0..) |cf, i| if (std.mem.eql(u8, cf.name, callee)) {
            callee_index = i;
            break;
        };
        s.convergent = s.convergent.join(callScope(callee));
        if (callee_index) |ci| s.convergent = s.convergent.join(s.funcs[ci].convergent);
        if (c.eat('(')) {
            var arg_index: usize = 0;
            while (true) {
                c.skipWs();
                if (c.eat(')') or c.atEnd()) break;
                const arg = c.argument();
                const l = s.levelOf(arg);
                args_level = args_level.join(l);
                if (callee_index) |ci| {
                    const cf = &s.funcs[ci];
                    if (arg_index < cf.param_level.len and l.above(cf.param_level[arg_index])) {
                        cf.param_level[arg_index] = l;
                        s.inter_changed = true;
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
            const cf = s.funcs[ci];
            if (cf.body != null) return args_level.join(cf.ret_level);
        }
        if (isSimdReduction(callee)) return .simdgroup;
        return args_level.join(callResultLevel(callee));
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
    /// values from different subsets.
    fn expandBranch(s: *Analyser, x: u32, level: Level) Error!void {
        const a = s.a;
        const n = s.cfg.n;
        const pdom = try s.postDominators();
        const loops = try s.naturalLoops();
        const reached = try a.alloc(bool, n);
        @memset(reached, false);
        // Headers of loops containing `x` that a path from `x`'s
        // successors reaches without passing a post-dominator of `x`.
        const header_hit = try a.alloc(bool, loops.len);
        @memset(header_hit, false);
        var stack: std.ArrayList(u32) = .empty;
        for (s.cfg.succs[x]) |succ| try stack.append(a, succ);
        while (stack.pop()) |b| {
            if (reached[b]) continue;
            // The region ends at the headers of the loops around `x`: the
            // threads arriving there start another iteration. (Checked
            // before post-dominance: `x` post-dominates itself, but reaching
            // it again through a back edge is exactly this case.)
            var is_header = false;
            for (loops, 0..) |l, li| if (l.header == b and l.members[x]) {
                header_hit[li] = true;
                is_header = true;
            };
            if (is_header) continue;
            // A block every path from `x` passes reconverges the threads.
            if (pdom[@as(usize, x) * n + b]) continue;
            reached[b] = true;
            s.raiseBlock(b, level);
            for (s.cfg.succs[b]) |succ| if (!reached[succ]) try stack.append(a, succ);
        }
        // Temporal divergence: some of the threads that passed `x` start
        // the next iteration of a loop while others leave it (an edge out
        // of the loop from `x` or from the region; a path that ends in a
        // trap leaves nothing behind and does not count). Every block of
        // that loop then runs for threads at different iterations, and
        // every value it defines is seen at different iterations after it.
        for (loops, header_hit) |l, hit| {
            if (!hit) continue;
            var leaves = false;
            for (0..n) |u| {
                if (u != x and !reached[u]) continue;
                for (s.cfg.succs[u]) |v| if (!l.members[v] and !s.cfg.traps[v]) {
                    leaves = true;
                };
            }
            if (!leaves) continue;
            for (l.members, 0..) |m, i| if (m and !reached[i]) {
                reached[i] = true;
                s.raiseBlock(i, level);
                try s.raiseBlockValues(@intCast(i), level);
            };
        }
        // Joins: two or more predecessors among the branch and the region.
        for (0..n) |j| {
            var count: usize = 0;
            for (s.cfg.preds[j]) |p| if (p == x or reached[p]) {
                count += 1;
            };
            if (count >= 2 and level.above(s.join[j])) {
                s.join[j] = level;
                s.local_changed = true;
            }
        }
    }

    /// Every value defined in block `b` (temporal divergence: threads see
    /// the loop's values at different iterations).
    fn raiseBlockValues(s: *Analyser, b: u32, level: Level) Error!void {
        for (s.body, 0..) |raw, li| {
            if (s.cfg.line_block[li] != b) continue;
            const line = instructionText(raw) orelse continue;
            var c = Scanner{ .s = line };
            if (c.peek() != '%') continue;
            _ = c.next();
            const name = c.name();
            c.skipWs();
            if (c.eat('=')) try s.raiseValue(name, level);
        }
    }

    /// `pdom[b * n + a]`: block `a` post-dominates block `b`. Greatest
    /// fixpoint of `pdom(b) = {b} ∪ ⋂ pdom(succ)`, so a block every
    /// terminating path passes post-dominates even through a loop. A
    /// no-return block (`unreachable`) has the full row: no path through it
    /// reaches the exit, so it does not cut post-dominance (the block
    /// after `if (i >= 64) trap` post-dominates the check).
    fn postDominators(s: *Analyser) Error![]bool {
        if (s.pdom) |p| return p;
        const n = s.cfg.n;
        const tab = try s.a.alloc(bool, n * n);
        for (0..n) |b| {
            const row = tab[b * n .. (b + 1) * n];
            if (s.cfg.succs[b].len == 0 and !s.cfg.traps[b]) {
                @memset(row, false);
                row[b] = true;
            } else @memset(row, true);
        }
        var changed = true;
        while (changed) {
            changed = false;
            for (0..n) |b| {
                if (s.cfg.succs[b].len == 0) continue;
                for (0..n) |v| {
                    if (!tab[b * n + v] or v == b) continue;
                    var keep = true;
                    for (s.cfg.succs[b]) |succ| if (!tab[@as(usize, succ) * n + v]) {
                        keep = false;
                        break;
                    };
                    if (!keep) {
                        tab[b * n + v] = false;
                        changed = true;
                    }
                }
            }
        }
        s.pdom = tab;
        return tab;
    }

    /// Natural loops from the back edges of the dominator tree (one entry
    /// per header, the loops of one header merged).
    fn naturalLoops(s: *Analyser) Error![]Loop {
        if (s.loops) |l| return l;
        const a = s.a;
        const n = s.cfg.n;
        // dom[b * n + v]: v dominates b.
        const dom = try a.alloc(bool, n * n);
        @memset(dom, true);
        @memset(dom[0..n], false);
        dom[0] = true;
        var changed = true;
        while (changed) {
            changed = false;
            for (1..n) |b| {
                if (!s.cfg.reach[b]) continue;
                for (0..n) |v| {
                    if (!dom[b * n + v] or v == b) continue;
                    var keep = true;
                    for (s.cfg.preds[b]) |p| if (!dom[@as(usize, p) * n + v]) {
                        keep = false;
                        break;
                    };
                    if (!keep) {
                        dom[b * n + v] = false;
                        changed = true;
                    }
                }
            }
        }
        var loops: std.ArrayList(Loop) = .empty;
        for (0..n) |u| {
            if (!s.cfg.reach[u]) continue;
            for (s.cfg.succs[u]) |h| {
                if (!dom[u * n + h]) continue;
                var loop: *Loop = undefined;
                var found = false;
                for (loops.items) |*l| if (l.header == h) {
                    loop = l;
                    found = true;
                    break;
                };
                if (!found) {
                    const members = try a.alloc(bool, n);
                    @memset(members, false);
                    members[h] = true;
                    try loops.append(a, .{ .header = h, .members = members });
                    loop = &loops.items[loops.items.len - 1];
                }
                var stack: std.ArrayList(u32) = .empty;
                try stack.append(a, @intCast(u));
                while (stack.pop()) |b| {
                    if (loop.members[b]) continue;
                    loop.members[b] = true;
                    for (s.cfg.preds[b]) |p| try stack.append(a, p);
                }
            }
        }
        s.loops = loops.items;
        return loops.items;
    }
};

const Loop = struct { header: u32, members: []bool };

fn analyseOne(gpa: Allocator, funcs: []Func, fi: usize) Error!bool {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const body = funcs[fi].body.?;
    const cfg = try Cfg.build(a, body);
    const block_level = try a.alloc(Level, cfg.n);
    @memset(block_level, .threadgroup);
    const join = try a.alloc(Level, cfg.n);
    @memset(join, .threadgroup);
    const branch_level = try a.alloc(Level, cfg.n);
    @memset(branch_level, .threadgroup);
    var s = Analyser{
        .a = a,
        .funcs = funcs,
        .fi = fi,
        .body = body,
        .cfg = cfg,
        .block_level = block_level,
        .join = join,
        .branch_level = branch_level,
    };
    try s.run();
    return s.inter_changed;
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

    fn build(a: Allocator, body: []const []const u8) Error!Cfg {
        var names: std.ArrayList([]const u8) = .empty;
        // The entry block is implicit unless the body opens with a label.
        var first = true;
        for (body) |raw| {
            const line = std.mem.trim(u8, raw, " \t\r");
            if (line.len == 0 or line[0] == ';') continue;
            if (labelName(line)) |name| {
                try names.append(a, name);
            } else if (first) {
                try names.append(a, "");
            }
            first = false;
        }
        if (names.items.len == 0) try names.append(a, "");
        const n = names.items.len;
        const line_block = try a.alloc(u32, body.len);
        const edges = try a.alloc(std.ArrayList(u32), n);
        for (edges) |*e| e.* = .empty;
        const traps = try a.alloc(bool, n);
        @memset(traps, false);
        var cur: u32 = 0;
        for (body, 0..) |raw, li| {
            const line = std.mem.trim(u8, raw, " \t\r");
            if (labelName(line)) |name| {
                for (names.items, 0..) |nm, i| if (std.mem.eql(u8, nm, name)) {
                    cur = @intCast(i);
                };
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
                for (names.items, 0..) |nm, i| if (std.mem.eql(u8, nm, target)) {
                    var dup = false;
                    for (edges[cur].items) |e| if (e == i) {
                        dup = true;
                    };
                    if (!dup) try edges[cur].append(a, @intCast(i));
                };
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
        return .{ .n = n, .line_block = line_block, .succs = succs, .preds = preds, .reach = reach, .traps = traps };
    }
};

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
    const body = t.funcs.items[fi].body.?;
    const cfg = try Cfg.build(a, body);
    var s = Analyser{ .a = a, .funcs = t.funcs.items, .fi = fi, .body = body, .cfg = cfg, .block_level = try a.alloc(Level, cfg.n), .join = try a.alloc(Level, cfg.n), .branch_level = try a.alloc(Level, cfg.n) };
    @memset(s.block_level, .threadgroup);
    @memset(s.join, .threadgroup);
    @memset(s.branch_level, .threadgroup);
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
