//! crosscheck-reduce: shrink a failing seed to a minimal program.
//!
//! The generator builds a syntax tree before printing (program.zig), so a
//! failing program can be shrunk structurally: delete statements, hoist the
//! body of an `if`, `for` or `switch` into its parent, shorten loops, replace
//! expressions by their children or by a leaf. Each candidate is printed and
//! judged by the real pipeline (`zig build crosscheck -Dcrosscheck-body=...`),
//! classified exactly as tools/crosscheck/sweep.sh does, and kept only when it
//! fails the same way as the original: same bucket, and for a refusal, a Metal
//! failure or a compile error the same message with the numbers blanked.
//! Candidates stay valid Zig by construction: the printer declares a local
//! `const` when it is never assigned and discards it when it is never read,
//! loop control hoisted out of its loop is dropped, and a candidate that is
//! still invalid simply lands in another bucket and is not kept.
//!
//! Output: <out>/<seed>/original.zig and reduced.zig, plus a line per kept
//! change. The candidates are built in <out>/<seed>/cache, which is deleted
//! afterwards (hundreds of builds would otherwise fill the project's cache and
//! the disk). Exit 0 when a reduced program was written, 3 when the seed does
//! not fail at all, 2 on bad usage.
//!
//! Usage: crosscheck-reduce --seed=<u64> --repo=<dir> --zig=<zig exe> --out=<dir>
//!        [--include-known-findings] [--floats] [--max-candidates=<n>]

const std = @import("std");
const program = @import("program.zig");
const Io = std.Io;
const Allocator = std.mem.Allocator;
const Expr = program.Expr;
const Cond = program.Cond;
const Stmt = program.Stmt;
const Program = program.Program;

// ── Classification, mirroring sweep.sh ──────────────────────────────────────

pub const Bucket = enum { match, mismatch, refused, metal, skip, err };

pub const Signature = struct {
    bucket: Bucket,
    /// The failure message with digit runs blanked, so a line number or a
    /// count does not make two instances of one failure look different.
    key: []const u8,

    pub fn eql(x: Signature, y: Signature) bool {
        return x.bucket == y.bucket and std.mem.eql(u8, x.key, y.key);
    }
};

fn blankDigits(gpa: Allocator, s: []const u8) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    var i: usize = 0;
    while (i < s.len) : (i += 1) {
        if (std.ascii.isDigit(s[i])) {
            while (i + 1 < s.len and std.ascii.isDigit(s[i + 1])) i += 1;
            try out.append(gpa, '#');
        } else try out.append(gpa, s[i]);
    }
    return out.toOwnedSlice(gpa);
}

/// The bucket of one crosscheck run from its combined output, with the same
/// precedence as sweep.sh: match, mismatch, skip, Metal, refused, else error.
pub fn classify(gpa: Allocator, out: []const u8) !Signature {
    var metal: ?[]const u8 = null;
    var refused: ?[]const u8 = null;
    var first_error: ?[]const u8 = null;
    var saw_match = false;
    var saw_mismatch = false;
    var saw_skip = false;
    var it = std.mem.splitScalar(u8, out, '\n');
    while (it.next()) |raw| {
        const line = std.mem.trimEnd(u8, raw, "\r");
        if (std.mem.startsWith(u8, line, "ok  crosscheck")) saw_match = true;
        if (std.mem.startsWith(u8, line, "FAIL crosscheck") and std.mem.indexOf(u8, line, "mismatched") != null) saw_mismatch = true;
        if (std.mem.startsWith(u8, line, "SKIP ")) saw_skip = true;
        if (metal == null) for ([_][]const u8{ "FAIL newLibraryWithData", "FAIL newFunctionWithName", "FAIL newComputePipelineState", "FAIL command buffer" }) |prefix| {
            if (std.mem.startsWith(u8, line, prefix)) metal = line;
        };
        if (refused == null and std.mem.startsWith(u8, line, "air-splice: ")) {
            const rest = line["air-splice: ".len..];
            if (std.mem.startsWith(u8, rest, "line ")) {
                // "line 174: use of undefined value" -> the message only.
                if (std.mem.indexOf(u8, rest, ": ")) |colon| refused = rest[colon + 2 ..] else refused = rest;
            } else if (std.mem.indexOf(u8, rest, "Unsupported") != null) refused = rest;
        }
        if (first_error == null) if (std.mem.indexOf(u8, line, "error: ")) |at| {
            first_error = line[at..];
        };
    }
    if (saw_match) return .{ .bucket = .match, .key = "" };
    if (saw_mismatch) return .{ .bucket = .mismatch, .key = "" };
    if (saw_skip) return .{ .bucket = .skip, .key = "" };
    if (metal) |m| return .{ .bucket = .metal, .key = try blankDigits(gpa, m) };
    if (refused) |r| return .{ .bucket = .refused, .key = try blankDigits(gpa, r) };
    return .{ .bucket = .err, .key = try blankDigits(gpa, first_error orelse "") };
}

// ── The judge: does a candidate still fail the same way? ─────────────────────

pub const Judge = struct {
    ctx: *anyopaque,
    func: *const fn (ctx: *anyopaque, text: []const u8) anyerror!bool,

    pub fn stillFails(j: Judge, text: []const u8) anyerror!bool {
        return j.func(j.ctx, text);
    }
};

/// Runs the real pipeline on a candidate body file. Every candidate is a new
/// body, so every one of them adds a kernel object, a library and a driver
/// build to the cache; a reduction runs hundreds. They go to a cache directory
/// of their own (`cache_dir`), deleted when the reduction ends, so the
/// project's cache and the disk stay the size they were.
const PipelineJudge = struct {
    gpa: Allocator,
    io: Io,
    zig_exe: []const u8,
    repo: []const u8,
    candidate_path: []const u8,
    cache_dir: []const u8,
    /// Extra `zig build` options every candidate run needs, e.g. the float
    /// mode's driver inputs (`-Dcrosscheck-floats=true`).
    build_args: []const []const u8,
    target: Signature,
    runs: u32 = 0,

    fn run(ctx: *anyopaque, text: []const u8) anyerror!bool {
        const j: *PipelineJudge = @ptrCast(@alignCast(ctx));
        const sig = try j.evaluate(text);
        return sig.eql(j.target);
    }

    fn evaluate(j: *PipelineJudge, text: []const u8) !Signature {
        try Io.Dir.cwd().writeFile(j.io, .{ .sub_path = j.candidate_path, .data = text });
        const body_arg = try std.mem.concat(j.gpa, u8, &.{ "-Dcrosscheck-body=", j.candidate_path });
        var argv: std.ArrayList([]const u8) = .empty;
        try argv.appendSlice(j.gpa, &.{ j.zig_exe, "build", "--cache-dir", j.cache_dir, "crosscheck", body_arg });
        try argv.appendSlice(j.gpa, j.build_args);
        const result = try std.process.run(j.gpa, j.io, .{
            .argv = argv.items,
            .cwd = .{ .path = j.repo },
        });
        j.runs += 1;
        const combined = try std.mem.concat(j.gpa, u8, &.{ result.stderr, "\n", result.stdout });
        return classify(j.gpa, combined);
    }
};

// ── The engine ───────────────────────────────────────────────────────────────

pub const Engine = struct {
    gpa: Allocator,
    judge: Judge,
    prog: *Program,
    seed: u64,
    tried: u32 = 0,
    kept: u32 = 0,
    max_candidates: u32,
    verbose: bool,
    /// Set when the budget ran out: every later trial is rejected unjudged
    /// (so the caller reverts it) and `run` stops after the current pass.
    exhausted: bool = false,

    const ListSlot = struct { list: *[]Stmt, in_loop: bool };
    const ExprSlot = struct { p: **Expr, rt: bool };

    pub fn header(e: *Engine, buf: *std.Io.Writer.Allocating) !void {
        try buf.writer.print("// crosscheck-reduce candidate for seed {d}.\npub const seed: u64 = {d};\n\n", .{ e.seed, e.seed });
    }

    /// Renders the current program and asks the judge. Returns whether the
    /// change that produced it is kept.
    fn accept(e: *Engine, what: []const u8) !bool {
        if (e.exhausted or e.tried >= e.max_candidates) {
            e.exhausted = true;
            return false;
        }
        var hb: std.Io.Writer.Allocating = .init(e.gpa);
        try e.header(&hb);
        const text = try program.render(e.gpa, e.prog, hb.written());
        e.tried += 1;
        const keep = try e.judge.stillFails(text);
        if (keep) {
            e.kept += 1;
            if (e.verbose) std.debug.print("    kept: {s} -> {d} statements, {d} nodes (candidate {d})\n", .{ what, program.countStmts(e.prog.stmts), program.countNodes(e.prog), e.tried });
        }
        return keep;
    }

    pub fn run(e: *Engine) !void {
        while (true) {
            var changed = false;
            if (try e.passStatements()) changed = true;
            if (e.exhausted) return error.CandidateBudgetExhausted;
            if (try e.passHoist()) changed = true;
            if (e.exhausted) return error.CandidateBudgetExhausted;
            if (try e.passSimplify()) changed = true;
            if (e.exhausted) return error.CandidateBudgetExhausted;
            if (try e.passExprs()) changed = true;
            if (e.exhausted) return error.CandidateBudgetExhausted;
            if (try e.passLocals()) changed = true;
            if (e.exhausted) return error.CandidateBudgetExhausted;
            if (!changed) return;
        }
    }

    // -- statement lists ------------------------------------------------------

    fn collectLists(e: *Engine, out: *std.ArrayList(ListSlot)) !void {
        try out.append(e.gpa, .{ .list = &e.prog.stmts, .in_loop = false });
        try e.collectListsIn(e.prog.stmts, false, out);
    }

    fn collectListsIn(e: *Engine, list: []Stmt, in_loop: bool, out: *std.ArrayList(ListSlot)) anyerror!void {
        for (list) |*s| switch (s.*) {
            .if_ => |*n| {
                try out.append(e.gpa, .{ .list = &n.then, .in_loop = in_loop });
                try e.collectListsIn(n.then, in_loop, out);
                if (n.els) |*els| {
                    try out.append(e.gpa, .{ .list = els, .in_loop = in_loop });
                    try e.collectListsIn(els.*, in_loop, out);
                }
            },
            .for_ => |*n| {
                try out.append(e.gpa, .{ .list = &n.body, .in_loop = true });
                try e.collectListsIn(n.body, true, out);
            },
            .switch_ => |*n| for (&n.arms) |*arm| {
                try out.append(e.gpa, .{ .list = arm, .in_loop = in_loop });
                try e.collectListsIn(arm.*, in_loop, out);
            },
            else => {},
        };
    }

    fn without(e: *Engine, list: []Stmt, i: usize) ![]Stmt {
        const out = try e.gpa.alloc(Stmt, list.len - 1);
        @memcpy(out[0..i], list[0..i]);
        @memcpy(out[i..], list[i + 1 ..]);
        return out;
    }

    /// `list` with element `i` replaced by `inner`.
    fn spliced(e: *Engine, list: []Stmt, i: usize, inner: []const Stmt) ![]Stmt {
        const out = try e.gpa.alloc(Stmt, list.len - 1 + inner.len);
        @memcpy(out[0..i], list[0..i]);
        @memcpy(out[i .. i + inner.len], inner);
        @memcpy(out[i + inner.len ..], list[i + 1 ..]);
        return out;
    }

    /// Drops `break`/`continue` that would leave their loop when a loop body
    /// is hoisted into a list outside any loop; nested loops keep theirs.
    fn stripLoopControl(e: *Engine, list: []const Stmt) anyerror![]Stmt {
        var out: std.ArrayList(Stmt) = .empty;
        for (list) |s| switch (s) {
            .brk, .cont => {},
            .if_ => |n| {
                var m = n;
                m.then = try e.stripLoopControl(n.then);
                if (n.els) |els| m.els = try e.stripLoopControl(els);
                try out.append(e.gpa, .{ .if_ = m });
            },
            .switch_ => |n| {
                var m = n;
                for (&m.arms, n.arms) |*dst, arm| dst.* = try e.stripLoopControl(arm);
                try out.append(e.gpa, .{ .switch_ = m });
            },
            else => try out.append(e.gpa, s),
        };
        return out.toOwnedSlice(e.gpa);
    }

    /// Empty whole lists, then delete single statements.
    fn passStatements(e: *Engine) !bool {
        var changed = false;
        var slots: std.ArrayList(ListSlot) = .empty;
        var k: usize = 0;
        while (true) {
            slots.clearRetainingCapacity();
            try e.collectLists(&slots);
            if (k >= slots.items.len) break;
            const slot = slots.items[k];
            const old = slot.list.*;
            if (old.len > 0) {
                slot.list.* = &.{};
                if (try e.accept("emptied a block")) {
                    changed = true;
                    continue;
                }
                slot.list.* = old;
            }
            var i: usize = 0;
            var removed_any = false;
            while (i < slot.list.*.len) {
                const before = slot.list.*;
                slot.list.* = try e.without(before, i);
                if (try e.accept("removed a statement")) {
                    changed = true;
                    removed_any = true;
                    // The list changed; nested slots are stale, re-collect.
                    break;
                }
                slot.list.* = before;
                i += 1;
            }
            if (!removed_any) k += 1;
        }
        return changed;
    }

    /// Replace an `if`, `for` or `switch` by one of its blocks.
    fn passHoist(e: *Engine) !bool {
        var changed = false;
        var slots: std.ArrayList(ListSlot) = .empty;
        var k: usize = 0;
        outer: while (true) {
            slots.clearRetainingCapacity();
            try e.collectLists(&slots);
            if (k >= slots.items.len) break;
            const slot = slots.items[k];
            const list = slot.list.*;
            for (list, 0..) |s, i| {
                var inners: [4]?[]const Stmt = .{ null, null, null, null };
                switch (s) {
                    .if_ => |n| {
                        inners[0] = n.then;
                        inners[1] = n.els;
                    },
                    .for_ => |n| inners[0] = if (slot.in_loop) n.body else try e.stripLoopControl(n.body),
                    .switch_ => |n| for (n.arms, 0..) |arm, a| {
                        if (arm.len > 0) inners[a] = arm;
                    },
                    else => continue,
                }
                for (inners) |maybe| {
                    const inner = maybe orelse continue;
                    slot.list.* = try e.spliced(list, i, inner);
                    if (try e.accept("hoisted a block into its parent")) {
                        changed = true;
                        continue :outer;
                    }
                    slot.list.* = list;
                }
            }
            k += 1;
        }
        return changed;
    }

    /// Smaller loops, no `else`, single comparisons.
    fn passSimplify(e: *Engine) !bool {
        var changed = false;
        var slots: std.ArrayList(ListSlot) = .empty;
        slots.clearRetainingCapacity();
        try e.collectLists(&slots);
        for (slots.items) |slot| {
            for (slot.list.*) |*s| switch (s.*) {
                .if_ => |*n| {
                    if (n.els != null) {
                        const old = n.els;
                        n.els = null;
                        if (try e.accept("dropped an else")) changed = true else n.els = old;
                    }
                    if (try e.simplifyCond(n.c)) changed = true;
                },
                .for_ => |*n| {
                    switch (n.bound) {
                        .runtime => {
                            const old = n.bound;
                            n.bound = .{ .constant = 2 };
                            if (try e.accept("made a loop bound constant")) changed = true else n.bound = old;
                        },
                        .constant => |c| if (c > 1) {
                            n.bound = .{ .constant = 1 };
                            if (try e.accept("shortened a loop to one iteration")) changed = true else n.bound = .{ .constant = c };
                        },
                    }
                    if (n.idx != null) {
                        const old = n.idx;
                        n.idx = null;
                        if (try e.accept("dropped a loop counter")) changed = true else n.idx = old;
                    }
                },
                .brk, .cont => |c| if (try e.simplifyCond(c)) {
                    changed = true;
                },
                else => {},
            };
        }
        return changed;
    }

    /// Delete locals that nothing reads or assigns any more, renumbering the
    /// rest. (A read local is left alone: deleting it would change meaning.)
    fn passLocals(e: *Engine) !bool {
        var changed = false;
        var i: u8 = 0;
        while (i < e.prog.nvars) {
            var usage: program.Usage = .{};
            usage.program(e.prog);
            if (usage.read[i] or usage.assigned[i]) {
                i += 1;
                continue;
            }
            const saved_inits = e.prog.inits;
            const saved_nvars = e.prog.nvars;
            try e.removeLocal(i);
            if (try e.accept("dropped an unused local")) {
                changed = true;
                continue; // the next local now has index i
            }
            // Undo: put the local back and renumber upwards again.
            e.renumber(e.prog.stmts, i, 1);
            e.renumberExpr(e.prog.ret, i, 1);
            for (saved_inits, 0..) |init, k| if (k != i) e.renumberExpr(init, i, 1);
            e.prog.inits = saved_inits;
            e.prog.nvars = saved_nvars;
            i += 1;
        }
        return changed;
    }

    fn removeLocal(e: *Engine, i: u8) !void {
        const inits = try e.gpa.alloc(*Expr, e.prog.inits.len - 1);
        @memcpy(inits[0..i], e.prog.inits[0..i]);
        @memcpy(inits[i..], e.prog.inits[i + 1 ..]);
        e.prog.inits = inits;
        e.prog.nvars -= 1;
        for (inits) |init| e.renumberExpr(init, i, -1);
        e.renumber(e.prog.stmts, i, -1);
        e.renumberExpr(e.prog.ret, i, -1);
    }

    /// Shift every reference to a local above `at` by `delta` (-1 after a
    /// deletion, +1 to undo one).
    fn renumber(e: *Engine, list: []Stmt, at: u8, delta: i8) void {
        for (list) |*s| switch (s.*) {
            .assign => |*n| {
                if (n.v > at or (delta > 0 and n.v >= at)) n.v = @intCast(@as(i16, n.v) + delta);
                e.renumberExpr(n.e, at, delta);
            },
            .if_ => |*n| {
                e.renumberCond(n.c, at, delta);
                e.renumber(n.then, at, delta);
                if (n.els) |els| e.renumber(els, at, delta);
            },
            .for_ => |*n| {
                switch (n.bound) {
                    .runtime => |b| e.renumberExpr(b, at, delta),
                    .constant => {},
                }
                if (n.idx) |*v| if (v.* > at or (delta > 0 and v.* >= at)) {
                    v.* = @intCast(@as(i16, v.*) + delta);
                };
                e.renumber(n.body, at, delta);
            },
            .switch_ => |*n| {
                e.renumberExpr(n.sel, at, delta);
                for (n.arms) |arm| e.renumber(arm, at, delta);
            },
            .brk, .cont => |c| e.renumberCond(c, at, delta),
        };
    }

    fn renumberCond(e: *Engine, c: *Cond, at: u8, delta: i8) void {
        e.renumberExpr(c.x, at, delta);
        e.renumberExpr(c.y, at, delta);
        if (c.rest) |r| e.renumberCond(r.c, at, delta);
    }

    fn renumberExpr(e: *Engine, x: *Expr, at: u8, delta: i8) void {
        switch (x.*) {
            .v => |*v| if (v.* > at or (delta > 0 and v.* >= at)) {
                v.* = @intCast(@as(i16, v.*) + delta);
            },
            .un => |n| e.renumberExpr(n.x, at, delta),
            .fun => |n| e.renumberExpr(n.x, at, delta),
            .bin => |n| {
                e.renumberExpr(n.x, at, delta);
                e.renumberExpr(n.y, at, delta);
            },
            .fbin => |n| {
                e.renumberExpr(n.x, at, delta);
                e.renumberExpr(n.y, at, delta);
            },
            .ffma => |n| {
                e.renumberExpr(n.x, at, delta);
                e.renumberExpr(n.y, at, delta);
                e.renumberExpr(n.z, at, delta);
            },
            .hfma => |n| {
                e.renumberExpr(n.x, at, delta);
                e.renumberExpr(n.y, at, delta);
                e.renumberExpr(n.z, at, delta);
            },
            .hun => |n| e.renumberExpr(n.x, at, delta),
            .hbin => |n| {
                e.renumberExpr(n.x, at, delta);
                e.renumberExpr(n.y, at, delta);
            },
            .select => |n| {
                e.renumberCond(n.c, at, delta);
                e.renumberExpr(n.x, at, delta);
                e.renumberExpr(n.y, at, delta);
            },
            .fsel => |n| {
                e.renumberCond(n.c, at, delta);
                e.renumberExpr(n.x, at, delta);
                e.renumberExpr(n.y, at, delta);
            },
            .fin, .fromint, .fbits, .ftoi => |inner| e.renumberExpr(inner, at, delta),
            else => {},
        }
    }

    /// Drop the second comparison, or keep only the second comparison.
    fn simplifyCond(e: *Engine, c: *Cond) !bool {
        const rest = c.rest orelse return false;
        c.rest = null;
        if (try e.accept("dropped a second comparison")) return true;
        c.rest = rest;
        const saved = c.*;
        c.* = rest.c.*;
        if (try e.accept("kept only the second comparison")) return true;
        c.* = saved;
        return false;
    }

    // -- expressions ----------------------------------------------------------

    fn collectExprs(e: *Engine, out: *std.ArrayList(ExprSlot)) !void {
        for (e.prog.inits) |*init| {
            try out.append(e.gpa, .{ .p = init, .rt = false });
            try e.collectChildren(init.*, false, out);
        }
        try e.collectStmtExprs(e.prog.stmts, out);
        try out.append(e.gpa, .{ .p = &e.prog.ret, .rt = false });
        try e.collectChildren(e.prog.ret, false, out);
    }

    fn collectChildren(e: *Engine, x: *Expr, rt: bool, out: *std.ArrayList(ExprSlot)) anyerror!void {
        switch (x.*) {
            .un => |*n| {
                try out.append(e.gpa, .{ .p = &n.x, .rt = rt });
                try e.collectChildren(n.x, rt, out);
            },
            .bin => |*n| {
                try out.append(e.gpa, .{ .p = &n.x, .rt = rt });
                try e.collectChildren(n.x, rt, out);
                try out.append(e.gpa, .{ .p = &n.y, .rt = rt });
                try e.collectChildren(n.y, rt, out);
            },
            .select => |*n| {
                try e.collectCond(n.c, out);
                try out.append(e.gpa, .{ .p = &n.x, .rt = rt });
                try e.collectChildren(n.x, rt, out);
                try out.append(e.gpa, .{ .p = &n.y, .rt = rt });
                try e.collectChildren(n.y, rt, out);
            },
            .fun => |*n| {
                try out.append(e.gpa, .{ .p = &n.x, .rt = rt });
                try e.collectChildren(n.x, rt, out);
            },
            .fbin => |*n| {
                try out.append(e.gpa, .{ .p = &n.x, .rt = rt });
                try e.collectChildren(n.x, rt, out);
                try out.append(e.gpa, .{ .p = &n.y, .rt = rt });
                try e.collectChildren(n.y, rt, out);
            },
            .ffma => |*n| {
                try out.append(e.gpa, .{ .p = &n.x, .rt = rt });
                try e.collectChildren(n.x, rt, out);
                try out.append(e.gpa, .{ .p = &n.y, .rt = rt });
                try e.collectChildren(n.y, rt, out);
                try out.append(e.gpa, .{ .p = &n.z, .rt = rt });
                try e.collectChildren(n.z, rt, out);
            },
            .hfma => |*n| {
                try out.append(e.gpa, .{ .p = &n.x, .rt = rt });
                try e.collectChildren(n.x, rt, out);
                try out.append(e.gpa, .{ .p = &n.y, .rt = rt });
                try e.collectChildren(n.y, rt, out);
                try out.append(e.gpa, .{ .p = &n.z, .rt = rt });
                try e.collectChildren(n.z, rt, out);
            },
            .hun => |*n| {
                try out.append(e.gpa, .{ .p = &n.x, .rt = rt });
                try e.collectChildren(n.x, rt, out);
            },
            .hbin => |*n| {
                try out.append(e.gpa, .{ .p = &n.x, .rt = rt });
                try e.collectChildren(n.x, rt, out);
                try out.append(e.gpa, .{ .p = &n.y, .rt = rt });
                try e.collectChildren(n.y, rt, out);
            },
            .fsel => |*n| {
                try e.collectCond(n.c, out);
                try out.append(e.gpa, .{ .p = &n.x, .rt = rt });
                try e.collectChildren(n.x, rt, out);
                try out.append(e.gpa, .{ .p = &n.y, .rt = rt });
                try e.collectChildren(n.y, rt, out);
            },
            .fin, .fromint, .fbits, .ftoi => |*inner| {
                try out.append(e.gpa, .{ .p = inner, .rt = rt });
                try e.collectChildren(inner.*, rt, out);
            },
            else => {},
        }
    }

    fn collectCond(e: *Engine, c: *Cond, out: *std.ArrayList(ExprSlot)) anyerror!void {
        try out.append(e.gpa, .{ .p = &c.x, .rt = true });
        try e.collectChildren(c.x, true, out);
        try out.append(e.gpa, .{ .p = &c.y, .rt = true });
        try e.collectChildren(c.y, true, out);
        if (c.rest) |r| try e.collectCond(r.c, out);
    }

    fn collectStmtExprs(e: *Engine, list: []Stmt, out: *std.ArrayList(ExprSlot)) anyerror!void {
        for (list) |*s| switch (s.*) {
            .assign => |*n| {
                try out.append(e.gpa, .{ .p = &n.e, .rt = false });
                try e.collectChildren(n.e, false, out);
            },
            .if_ => |*n| {
                try e.collectCond(n.c, out);
                try e.collectStmtExprs(n.then, out);
                if (n.els) |els| try e.collectStmtExprs(els, out);
            },
            .for_ => |*n| {
                switch (n.bound) {
                    .runtime => |*b| {
                        try out.append(e.gpa, .{ .p = b, .rt = false });
                        try e.collectChildren(b.*, false, out);
                    },
                    .constant => {},
                }
                try e.collectStmtExprs(n.body, out);
            },
            .switch_ => |*n| {
                try out.append(e.gpa, .{ .p = &n.sel, .rt = false });
                try e.collectChildren(n.sel, false, out);
                for (n.arms) |arm| try e.collectStmtExprs(arm, out);
            },
            .brk, .cont => |c| try e.collectCond(c, out),
        };
    }

    fn leaf(e: *Engine, v: Expr) !*Expr {
        const p = try e.gpa.create(Expr);
        p.* = v;
        return p;
    }

    /// Replace an expression by a child, then by `a`, `b`, 0 or 1 (constants
    /// only where a comptime-known value is allowed).
    fn passExprs(e: *Engine) !bool {
        var changed = false;
        var slots: std.ArrayList(ExprSlot) = .empty;
        var k: usize = 0;
        outer: while (true) {
            slots.clearRetainingCapacity();
            try e.collectExprs(&slots);
            if (k >= slots.items.len) break;
            const slot = slots.items[k];
            const old = slot.p.*;
            const ty = program.exprType(old);
            // A leaf is as small as an expression gets. The only leaf-to-leaf
            // change allowed is towards `a` (for floats, `fin(a)`), the
            // smallest leaf: it frees locals for passLocals, and because it is
            // one-way it cannot go on forever.
            const int_leaf = switch (old.*) {
                .a, .b, .v, .constant => true,
                else => false,
            };
            const float_leaf = switch (old.*) {
                .fconst => true,
                .fin => |x| switch (x.*) {
                    .a, .b, .v, .constant => true,
                    else => false,
                },
                else => false,
            };
            var candidates: [7]?*Expr = .{ null, null, null, null, null, null, null };
            if (int_leaf or float_leaf) {
                if (int_leaf and old.* != .a) candidates[0] = try e.leaf(.a);
                if (old.* == .fin and old.fin.* != .a) candidates[0] = try e.leaf(.{ .fin = try e.leaf(.a) });
                for (candidates) |maybe| {
                    const cand = maybe orelse continue;
                    slot.p.* = cand;
                    if (try e.accept("replaced a leaf by a")) {
                        changed = true;
                        continue :outer;
                    }
                    slot.p.* = old;
                }
                k += 1;
                continue;
            }
            // Children of the slot's own type first (a bridge like fbits or
            // fin has a child of the other type, which cannot replace it).
            switch (old.*) {
                .un => |n| candidates[0] = n.x,
                .bin => |n| {
                    candidates[0] = n.x;
                    candidates[1] = n.y;
                },
                .select => |n| {
                    candidates[0] = n.x;
                    candidates[1] = n.y;
                },
                .fun => |n| candidates[0] = n.x,
                .fbin => |n| {
                    candidates[0] = n.x;
                    candidates[1] = n.y;
                },
                .ffma => |n| {
                    candidates[0] = n.x;
                    candidates[1] = n.y;
                    candidates[2] = n.z;
                },
                .hfma => |n| {
                    candidates[0] = n.x;
                    candidates[1] = n.y;
                    candidates[2] = n.z;
                },
                .hun => |n| candidates[0] = n.x,
                .hbin => |n| {
                    candidates[0] = n.x;
                    candidates[1] = n.y;
                },
                .fsel => |n| {
                    candidates[0] = n.x;
                    candidates[1] = n.y;
                },
                else => {},
            }
            switch (ty) {
                .int => {
                    candidates[3] = try e.leaf(.a);
                    candidates[4] = try e.leaf(.b);
                    if (!slot.rt) {
                        candidates[5] = try e.leaf(.{ .constant = 0 });
                        candidates[6] = try e.leaf(.{ .constant = 1 });
                    }
                },
                .float => {
                    candidates[3] = try e.leaf(.{ .fin = try e.leaf(.a) });
                    candidates[4] = try e.leaf(.{ .fin = try e.leaf(.b) });
                    if (!slot.rt) {
                        candidates[5] = try e.leaf(.{ .fconst = 0 });
                        candidates[6] = try e.leaf(.{ .fconst = 0x3f800000 });
                    }
                },
            }
            for (candidates) |maybe| {
                const cand = maybe orelse continue;
                slot.p.* = cand;
                if (try e.accept("simplified an expression")) {
                    changed = true;
                    continue :outer;
                }
                slot.p.* = old;
            }
            k += 1;
        }
        return changed;
    }
};

// ── CLI ──────────────────────────────────────────────────────────────────────

pub fn main(init: std.process.Init) !void {
    const arena = init.arena.allocator();
    const io = init.io;

    const args = try init.minimal.args.toSlice(arena);
    var seed: ?u64 = null;
    var repo: ?[]const u8 = null;
    var zig_exe: []const u8 = "zig";
    var out_dir: ?[]const u8 = null;
    var include_known = false;
    var floats = false;
    var max_candidates: u32 = 3000;
    for (args[1..]) |arg| {
        if (std.mem.startsWith(u8, arg, "--seed=")) {
            seed = std.fmt.parseInt(u64, arg["--seed=".len..], 10) catch null;
        } else if (std.mem.startsWith(u8, arg, "--repo=")) {
            repo = arg["--repo=".len..];
        } else if (std.mem.startsWith(u8, arg, "--zig=")) {
            zig_exe = arg["--zig=".len..];
        } else if (std.mem.startsWith(u8, arg, "--out=")) {
            out_dir = arg["--out=".len..];
        } else if (std.mem.startsWith(u8, arg, "--max-candidates=")) {
            max_candidates = std.fmt.parseInt(u32, arg["--max-candidates=".len..], 10) catch max_candidates;
        } else if (std.mem.eql(u8, arg, "--include-known-findings")) {
            include_known = true;
        } else if (std.mem.eql(u8, arg, "--floats")) {
            floats = true;
        } else {
            std.debug.print("crosscheck-reduce: unknown argument '{s}'\n", .{arg});
            std.process.exit(2);
        }
    }
    const sd = seed orelse {
        std.debug.print("usage: crosscheck-reduce --seed=<u64> --repo=<dir> --zig=<zig> --out=<dir> [--include-known-findings] [--max-candidates=<n>]\n", .{});
        std.process.exit(2);
    };
    const repo_path = repo orelse ".";
    // `--out` is relative to the repository unless absolute.
    const out_base = out_dir orelse "zig-out/crosscheck/reduce";
    const out_root = if (std.fs.path.isAbsolute(out_base)) out_base else try std.fs.path.join(arena, &.{ repo_path, out_base });
    const dir = try std.fmt.allocPrint(arena, "{s}/{d}", .{ out_root, sd });
    try Io.Dir.cwd().createDirPath(io, dir);

    var prog = try program.generate(arena, sd, .{ .include_known = include_known, .floats = floats });
    const known_note: []const u8 = if (include_known and floats) " -Dcrosscheck-known=true -Dcrosscheck-floats=true" else if (include_known) " -Dcrosscheck-known=true" else if (floats) " -Dcrosscheck-floats=true" else "";
    // The judge runs every candidate under the failing seed's input stream
    // and mode; the reduced file's hint repeats both.
    const mode_note: []const u8 = if (floats) " -Dcrosscheck-floats=true" else "";
    const seed_arg = try std.fmt.allocPrint(arena, "-Dseed={d}", .{sd});
    const build_args: []const []const u8 = if (floats) &.{ seed_arg, "-Dcrosscheck-floats=true" } else &.{seed_arg};
    const original_header = try std.fmt.allocPrint(arena,
        \\// The program of `zig build crosscheck -Dseed={d}{s}`, as crosscheck-reduce
        \\// started from it.
        \\pub const seed: u64 = {d};
        \\
        \\
    , .{ sd, known_note, sd });
    const original = try program.render(arena, &prog, original_header);
    const original_path = try std.fmt.allocPrint(arena, "{s}/original.zig", .{dir});
    try Io.Dir.cwd().writeFile(io, .{ .sub_path = original_path, .data = original });
    const stmts0 = program.countStmts(prog.stmts);
    const nodes0 = program.countNodes(&prog);

    var judge: PipelineJudge = .{
        .gpa = arena,
        .io = io,
        .zig_exe = zig_exe,
        .repo = repo_path,
        .candidate_path = try std.fmt.allocPrint(arena, "{s}/candidate.zig", .{dir}),
        .cache_dir = try std.fmt.allocPrint(arena, "{s}/cache", .{dir}),
        .build_args = build_args,
        .target = .{ .bucket = .match, .key = "" },
    };
    defer Io.Dir.cwd().deleteTree(io, judge.cache_dir) catch {};
    judge.target = try judge.evaluate(original);
    std.debug.print("info reduce seed {d}{s}: {d} statements, {d} nodes; failure: {t} {s}\n", .{ sd, known_note, stmts0, nodes0, judge.target.bucket, judge.target.key });
    if (judge.target.bucket == .match) {
        std.debug.print("nothing to reduce: seed {d} matches\n", .{sd});
        Io.Dir.cwd().deleteTree(io, judge.cache_dir) catch {}; // exit skips the defer
        std.process.exit(3);
    }
    if (judge.target.bucket == .skip) {
        std.debug.print("SKIP no Metal device; cannot judge candidates\n", .{});
        Io.Dir.cwd().deleteTree(io, judge.cache_dir) catch {};
        std.process.exit(77);
    }

    var engine: Engine = .{
        .gpa = arena,
        .judge = .{ .ctx = @ptrCast(&judge), .func = PipelineJudge.run },
        .prog = &prog,
        .seed = sd,
        .max_candidates = max_candidates,
        .verbose = true,
    };
    engine.run() catch |err| switch (err) {
        error.CandidateBudgetExhausted => std.debug.print("    candidate budget ({d}) exhausted; writing the best so far\n", .{max_candidates}),
        else => return err,
    };

    const reduced_header = try std.fmt.allocPrint(arena,
        \\// Reduced by crosscheck-reduce from `zig build crosscheck -Dseed={d}{s}`:
        \\// {d} statements / {d} nodes -> {d} / {d}, {d} candidates, {d} kept.
        \\// Failure: {t} {s}
        \\//
        \\//   zig build crosscheck -Dseed={d}{s} -Dcrosscheck-body=<this file>
        \\pub const seed: u64 = {d};
        \\
        \\
    , .{ sd, known_note, stmts0, nodes0, program.countStmts(prog.stmts), program.countNodes(&prog), engine.tried, engine.kept, judge.target.bucket, judge.target.key, sd, mode_note, sd });
    const reduced = try program.render(arena, &prog, reduced_header);
    const reduced_path = try std.fmt.allocPrint(arena, "{s}/reduced.zig", .{dir});
    try Io.Dir.cwd().writeFile(io, .{ .sub_path = reduced_path, .data = reduced });
    std.debug.print("ok  reduced seed {d}: {d} statements / {d} nodes -> {d} / {d} after {d} candidates ({d} kept)\n    {s}\n", .{ sd, stmts0, nodes0, program.countStmts(prog.stmts), program.countNodes(&prog), engine.tried, engine.kept, reduced_path });
}

// ── Tests ────────────────────────────────────────────────────────────────────

test "classify follows sweep.sh's buckets and blanks numbers in the key" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();
    try std.testing.expectEqual(Bucket.match, (try classify(gpa, "info device Apple M3\nok  crosscheck seed 5: 4096/4096 match\n")).bucket);
    try std.testing.expectEqual(Bucket.mismatch, (try classify(gpa, "FAIL crosscheck seed 5: 12/4096 mismatched (reproduce: x)\n")).bucket);
    const metal = try classify(gpa, "ok  newLibraryWithData (1234 bytes)\nFAIL newComputePipelineState crosscheckKernel: Error Domain=AGXMetalG15G_C0 Code=2 \"Compilation failed\"\n");
    try std.testing.expectEqual(Bucket.metal, metal.bucket);
    try std.testing.expectEqualStrings("FAIL newComputePipelineState crosscheckKernel: Error Domain=AGXMetalG#G_C# Code=# \"Compilation failed\"", metal.key);
    const refused = try classify(gpa, "air-splice: line 174: use of undefined value\n    %__100 = add i64 %x, 1\nair-splice: assembling crosscheckKernel: UndefinedValue\n");
    try std.testing.expectEqual(Bucket.refused, refused.bucket);
    try std.testing.expectEqualStrings("use of undefined value", refused.key);
    const err = try classify(gpa, "crosscheck_body.zig:111:32: error: @byteSwap requires the number of bits to be evenly divisible by 8, but u31 has 31 bits\n");
    try std.testing.expectEqual(Bucket.err, err.bucket);
    try std.testing.expectEqualStrings("error: @byteSwap requires the number of bits to be evenly divisible by #, but u# has # bits", err.key);
    // Same failure, different line number: equal signatures.
    const r2 = try classify(gpa, "air-splice: line 9: use of undefined value\n");
    try std.testing.expect(refused.eql(r2));
}

const TextProperty = struct {
    needle: []const u8,
    fn judge(ctx: *anyopaque, text: []const u8) anyerror!bool {
        const t: *TextProperty = @ptrCast(@alignCast(ctx));
        return std.mem.indexOf(u8, bodyOf(text), t.needle) != null;
    }
};

/// The `pub fn body` part of a rendered file, after the shared helpers (which
/// would otherwise satisfy any needle naming one of them).
fn bodyOf(text: []const u8) []const u8 {
    return text[std.mem.indexOf(u8, text, "pub fn body") orelse 0 ..];
}

test "the engine shrinks a program to a few nodes that keep a textual property" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();
    // A seed whose program uses divTrunc somewhere; the property is cheap to
    // judge, so the engine's mechanics are exercised, not the pipeline.
    var seed: u64 = 1;
    var prog: Program = undefined;
    while (true) : (seed += 1) {
        prog = try program.generate(gpa, seed, .{});
        const text = try program.render(gpa, &prog, "pub const seed: u64 = 0;\n\n");
        if (std.mem.indexOf(u8, bodyOf(text), "divTrunc(") != null) break;
    }
    const stmts0 = program.countStmts(prog.stmts);
    const nodes0 = program.countNodes(&prog);
    var prop: TextProperty = .{ .needle = "divTrunc(" };
    var engine: Engine = .{
        .gpa = gpa,
        .judge = .{ .ctx = @ptrCast(&prop), .func = TextProperty.judge },
        .prog = &prog,
        .seed = seed,
        .max_candidates = 100_000,
        .verbose = false,
    };
    try engine.run();
    const reduced = try program.render(gpa, &prog, "pub const seed: u64 = 0;\n\n");
    try std.testing.expect(std.mem.indexOf(u8, bodyOf(reduced), "divTrunc(") != null);
    // The smallest program with the property is one `divTrunc` of two leaves,
    // in an initialiser, the result, or a single assignment; every other
    // expression is a leaf, and every local nothing references is gone.
    try std.testing.expect(program.countStmts(prog.stmts) <= 1);
    try std.testing.expect(program.countNodes(&prog) <= @as(u32, prog.nvars) + 4);
    try std.testing.expect(program.countNodes(&prog) < nodes0 or stmts0 == 0);
    try std.testing.expectEqual(@as(u8, 0), prog.nvars);
    // Every local is declared validly: never assigned ones became const.
    var usage: program.Usage = .{};
    usage.program(&prog);
    for (0..prog.nvars) |i| if (!usage.assigned[i]) {
        const decl = try std.fmt.allocPrint(gpa, "const v{d}: i32 = ", .{i});
        try std.testing.expect(std.mem.indexOf(u8, reduced, decl) != null);
    };
}

test "the engine shrinks a float program, keeping types consistent" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();
    var seed: u64 = 1;
    var prog: Program = undefined;
    while (true) : (seed += 1) {
        prog = try program.generate(gpa, seed, .{ .floats = true });
        const text = try program.render(gpa, &prog, "pub const seed: u64 = 0;\n\n");
        if (std.mem.indexOf(u8, bodyOf(text), "fsqrt(") != null) break;
    }
    var prop: TextProperty = .{ .needle = "fsqrt(" };
    var engine: Engine = .{
        .gpa = gpa,
        .judge = .{ .ctx = @ptrCast(&prop), .func = TextProperty.judge },
        .prog = &prog,
        .seed = seed,
        .max_candidates = 100_000,
        .verbose = false,
    };
    try engine.run();
    const reduced = try program.render(gpa, &prog, "pub const seed: u64 = 0;\n\n");
    try std.testing.expect(std.mem.indexOf(u8, bodyOf(reduced), "fsqrt(") != null);
    // Smallest under the engine's moves: one fsqrt(fin(a)) kept alive by a
    // bits/toInt bridge or a single comparison, a leaf result, and no local
    // unless the needle started in an assignment: the engine has no move that
    // lifts a local's value into the result, so then one `v0 = bits(fsqrt(
    // fin(a)))` plus `return v0` remains.
    try std.testing.expect(program.countStmts(prog.stmts) <= 1);
    try std.testing.expect(program.countNodes(&prog) <= 6);
    try std.testing.expect(prog.nvars <= 1);
}

test "running out of candidates leaves the program exactly as last kept" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();
    var prog = try program.generate(gpa, 3, .{});
    const before = try program.render(gpa, &prog, "");
    // A judge that keeps nothing, and a budget of one: the single trial is
    // rejected unjudged and reverted, and run() reports the exhaustion.
    var prop: TextProperty = .{ .needle = "this text appears in no program" };
    var engine: Engine = .{
        .gpa = gpa,
        .judge = .{ .ctx = @ptrCast(&prop), .func = TextProperty.judge },
        .prog = &prog,
        .seed = 3,
        .max_candidates = 1,
        .verbose = false,
    };
    try std.testing.expectError(error.CandidateBudgetExhausted, engine.run());
    try std.testing.expectEqualStrings(before, try program.render(gpa, &prog, ""));
}

test "hoisting a loop body out of its loop drops break and continue, nested loops keep theirs" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();
    var prog = try program.generate(gpa, 1, .{});
    var engine: Engine = .{ .gpa = gpa, .judge = undefined, .prog = &prog, .seed = 1, .max_candidates = 0, .verbose = false };
    const c = try gpa.create(Cond);
    c.* = .{ .cmp = .eq, .unsigned = false, .x = try engine.leaf(.a), .y = try engine.leaf(.b) };
    const inner_body = try gpa.dupe(Stmt, &.{ .{ .brk = c }, .{ .assign = .{ .v = 0, .e = try engine.leaf(.a) } } });
    const body = try gpa.dupe(Stmt, &.{
        .{ .cont = c },
        .{ .for_ = .{ .bound = .{ .constant = 2 }, .idx = null, .body = inner_body } },
        .{ .if_ = .{ .c = c, .then = try gpa.dupe(Stmt, &.{.{ .brk = c }}), .els = null } },
    });
    const stripped = try engine.stripLoopControl(body);
    try std.testing.expectEqual(@as(usize, 2), stripped.len);
    try std.testing.expect(stripped[0] == .for_);
    try std.testing.expectEqual(@as(usize, 2), stripped[0].for_.body.len); // nested loop untouched
    try std.testing.expect(stripped[1] == .if_);
    try std.testing.expectEqual(@as(usize, 0), stripped[1].if_.then.len);
}
