//! CPU reference for the probe kernels of probes.zig: what each kernel must
//! compute, lane by lane, so that tools/spirv_probe_run.zig can dispatch the
//! kernel on the GPU and compare. It covers the instructions whose lowering
//! is more than a rename (SMod/FMod, bit fields, carries, the 64-bit high
//! product, FindSMsb, sign, clamp, mix, step, smoothstep, cross, ...), every
//! exact integer and conversion instruction, and the transcendental mappings
//! to within a loose tolerance (a wrong function, Asin for Acos, is far
//! outside it; the functions' precision is Apple's, measured by `zig build
//! survey`).
//!
//! `prepare` first fixes up generated inputs so they stay inside the
//! instruction's defined domain (no division by zero, shift amounts below
//! the width, `offset + count <= width`, float-to-int inputs in range, ...),
//! then returns the expected lanes and which of them are defined. Where a
//! specification defines the edges, the edge inputs stay: OpenCL's powr
//! (±0, ±1, ±inf, NaN and negative x, `powrRef`) and its length, distance
//! and normalize (zero vectors, infinite and NaN elements, `oclGeometry`). Floats are
//! compared bit-exactly for exact operations (any NaN equals any NaN; an f32
//! subnormal result may come back flushed to zero, the library declares
//! `denorms_disable`), and within a relative tolerance for `.approx` ones.

const std = @import("std");
const probes = @import("probes.zig");
const Probe = probes.Probe;

pub const Kind = enum { int, uint, float, bool };

pub const Elem = struct {
    kind: Kind,
    bits: u8,
    lanes: u8,

    /// Bytes one element occupies in a buffer (AIR allocation: a 3-lane
    /// vector takes four lanes).
    pub fn stride(e: Elem) u32 {
        const lanes: u32 = if (e.lanes == 3) 4 else e.lanes;
        return lanes * @max(1, e.bits / 8);
    }
};

pub fn elemOf(t: []const u8) Elem {
    var lanes: u8 = 1;
    var s = t;
    if (s.len > 2 and s[0] == 'v' and std.ascii.isDigit(s[1])) {
        lanes = s[1] - '0';
        s = s[2..];
    }
    if (std.mem.eql(u8, s, "bool")) return .{ .kind = .bool, .bits = 8, .lanes = lanes };
    const kind: Kind = switch (s[0]) {
        'i' => .int,
        'u' => .uint,
        else => .float,
    };
    return .{ .kind = kind, .bits = std.fmt.parseInt(u8, s[1..], 10) catch unreachable, .lanes = lanes };
}

/// Element type of the stored result (probes.storeType).
pub fn storeElem(p: Probe) Elem {
    if (std.mem.startsWith(u8, p.result, "S_")) return elemOf(p.result[2..]);
    const e = elemOf(p.result);
    if (e.kind == .bool) return .{ .kind = .uint, .bits = 32, .lanes = e.lanes };
    return e;
}

pub const Check = enum { exact, approx };

pub const Expect = struct {
    lanes: [4]u64 = .{ 0, 0, 0, 0 },
    defined: [4]bool = .{ true, true, true, true },
};

pub const Inputs = [3][4]u64;

const Inst = struct { group: []const u8, inst: []const u8 };

/// `op_SMod_i32` -> ("op", "SMod"); `ocl_s_mul_hi_u64` -> ("ocl", "s_mul_hi");
/// `conv_UConvert_u8_u32` -> ("conv", "UConvert").
fn parseName(name: []const u8) Inst {
    const first = std.mem.findScalar(u8, name, '_').?;
    const group = name[0..first];
    var rest = name[first + 1 ..];
    if (std.mem.eql(u8, group, "conv")) return .{ .group = group, .inst = rest[0..std.mem.findScalar(u8, rest, '_').?] };
    if (std.mem.eql(u8, group, "vec") or std.mem.eql(u8, group, "comp")) return .{ .group = group, .inst = rest };
    const last = std.mem.lastIndexOfScalar(u8, rest, '_').?;
    rest = rest[0..last];
    // `op_Select_scalar_cond_v4f32` keeps its descriptive tail.
    return .{ .group = group, .inst = rest };
}

const approx_insts = [_][]const u8{
    // GLSL.std.450
    "Sin",       "Cos",   "Tan",   "Asin",       "Acos",   "Atan",        "Sinh",      "Cosh",  "Tanh", "Asinh",      "Acosh",  "Atanh",
    "Exp",       "Log",   "Exp2",  "Log2",       "Sqrt",   "InverseSqrt", "Atan2",     "Pow",   "FMix", "SmoothStep", "Length", "Distance",
    "Normalize", "Cross", "FDiv",
    // OpenCL.std
     "sqrt",       "rsqrt",  "sin",         "cos",       "tan",   "asin", "acos",       "atan",   "sinh",
    "cosh",      "tanh",  "asinh", "acosh",      "atanh",  "exp",         "exp2",      "exp10", "log",  "log2",       "log10",  "atan2",
    "pow",       "powr",  "mix",   "smoothstep", "length", "distance",    "normalize", "cross", "mad",
};

/// How the kernel's result is compared, or null when the reference does not
/// model the probe (it is then only built, not run).
pub fn checkOf(p: Probe) ?Check {
    // Barriers and atomics (probes.stage4): built only; their results
    // depend on the other threads, and the corpus driver
    // (tools/spirv/corpus_driver.zig) runs whole kernels with known answers.
    if (std.mem.startsWith(u8, p.name, "atom_") or std.mem.startsWith(u8, p.name, "sync_")) return null;
    const n = parseName(p.name);
    for (approx_insts) |a| if (std.mem.eql(u8, a, n.inst)) return .approx;
    if (std.mem.startsWith(u8, n.inst, "Dot_")) return .approx;
    return .exact;
}

fn mask(w: u32) u64 {
    return if (w >= 64) ~@as(u64, 0) else (@as(u64, 1) << @intCast(w)) - 1;
}

fn sext(v: u64, w: u32) i64 {
    const sh: u6 = @intCast(64 - w);
    return @as(i64, @bitCast(v << sh)) >> sh;
}

fn T(comptime bits: u8) type {
    return if (bits == 16) f16 else f32;
}

fn fl(comptime B: u8, v: u64) T(B) {
    return @bitCast(@as(@Int(.unsigned, B), @truncate(v)));
}

fn bitsOf(x: anytype) u64 {
    return @as(@Int(.unsigned, @bitSizeOf(@TypeOf(x))), @bitCast(x));
}

/// Like `clampF`, but zeros, ±1, infinities, NaNs and (when `lo` > 0)
/// negative values stay: the edge inputs of an OpenCL function that
/// defines its results there.
fn clampFinite(comptime B: u8, v: *u64, lo: f64, hi: f64) void {
    const x: f64 = fl(B, v.*);
    if (std.math.isNan(x) or std.math.isInf(x) or x == 0 or @abs(x) == 1 or (lo > 0 and x < 0)) return;
    clampF(B, v, lo, hi);
}

/// OpenCL C's powr (3.0, "Additional Requirements Beyond C99"): pow for
/// x >= 0, NaN for x < 0, for (±0, ±0), (+inf, ±0), (+1, ±inf) and NaN
/// operands; ±0 is +0 (so powr(-0, 3) is +0 and powr(-0, -3) +inf).
fn powrRef(x: f64, y: f64) f64 {
    const nan = std.math.nan(f64);
    const inf = std.math.inf(f64);
    if (std.math.isNan(x) or std.math.isNan(y) or x < 0) return nan;
    if (x == 0) return if (y == 0) nan else if (y < 0) inf else 0;
    if (std.math.isInf(x)) return if (y == 0) nan else if (y < 0) 0 else inf;
    if (x == 1) return if (std.math.isInf(y)) nan else 1;
    return std.math.pow(f64, x, y);
}

/// Replace `v` (a float of `B` bits) when it is NaN, infinite or outside
/// [lo, hi] by a value inside the range derived from its bits.
fn clampF(comptime B: u8, v: *u64, lo: f64, hi: f64) void {
    const x: f64 = fl(B, v.*);
    if (std.math.isNan(x) or x < lo or x > hi) {
        const frac = @as(f64, @floatFromInt(v.* % 1000)) / 1000.0;
        v.* = bitsOf(@as(T(B), @floatCast(lo + (hi - lo) * frac)));
    }
}

fn isSubnormalOrZero(comptime B: u8, v: u64) bool {
    const x: f64 = @abs(@as(f64, fl(B, v)));
    return x < (if (B == 16) 6.103515625e-05 else 1.1754943508222875e-38);
}

/// Fix `ins` in place for `p` and return the expected result, or null when
/// the probe is not modelled.
pub fn prepare(p: Probe, ins: *Inputs) ?Expect {
    const n = parseName(p.name);
    const e0 = elemOf(p.operands[0]);
    const out = storeElem(p);
    var exp: Expect = .{};
    for (exp.defined[out.lanes..]) |*d| d.* = false;

    // Whole-element instructions first.
    if (std.mem.eql(u8, n.group, "vec")) return vecInst(n.inst, ins, out);
    if (std.mem.eql(u8, n.group, "comp")) return compInst(n.inst, ins, out);
    if (std.mem.eql(u8, n.inst, "Bitcast")) {
        // Same bytes, other lanes.
        var bytes: [32]u8 = undefined;
        const e = e0;
        for (0..e.lanes) |l| std.mem.writeInt(u64, bytes[l * 8 ..][0..8], ins[0][l], .little);
        var packed_bytes: [16]u8 = @splat(0);
        const eb: u32 = e.bits / 8;
        for (0..e.lanes) |l| @memcpy(packed_bytes[l * eb ..][0..eb], bytes[l * 8 ..][0..eb]);
        if (e.kind == .float) {
            // A NaN may change payload through a float register; keep inputs numeric.
            for (0..e.lanes) |l| {
                if (std.math.isNan(@as(f64, if (e.bits == 16) fl(16, ins[0][l]) else fl(32, ins[0][l])))) return prepare(p, fixNan(ins, e));
            }
        }
        const ob: u32 = out.bits / 8;
        for (0..out.lanes) |l| {
            var v: u64 = 0;
            for (0..ob) |k| v |= @as(u64, packed_bytes[l * ob + k]) << @intCast(8 * k);
            exp.lanes[l] = v;
        }
        if (out.kind == .float) for (0..out.lanes) |l| {
            if (std.math.isNan(@as(f64, if (out.bits == 16) fl(16, exp.lanes[l]) else fl(32, exp.lanes[l])))) exp.defined[l] = false;
        };
        return exp;
    }
    if (std.mem.eql(u8, n.inst, "Length") or std.mem.eql(u8, n.inst, "length") or std.mem.eql(u8, n.inst, "Distance") or std.mem.eql(u8, n.inst, "distance") or std.mem.eql(u8, n.inst, "Normalize") or std.mem.eql(u8, n.inst, "normalize") or std.mem.eql(u8, n.inst, "Cross") or std.mem.eql(u8, n.inst, "cross")) {
        return geometry(n.inst, ins, e0, out);
    }
    if (std.mem.startsWith(u8, n.inst, "All") or std.mem.startsWith(u8, n.inst, "Any") or std.mem.eql(u8, n.inst, "Select_scalar_cond")) return boolReduce(n.inst, ins, e0, out);

    for (0..out.lanes) |l| {
        var lane: [3]u64 = .{ ins[0][l], ins[1][l], ins[2][l] };
        // Scalar operands of a vector instruction (none in the lane-wise probes) keep lane 0.
        for (p.operands, 0..) |t, i| if (elemOf(t).lanes == 1) {
            lane[i] = ins[i][0];
        };
        const r = laneInst(p, n, &lane) orelse return null;
        for (p.operands, 0..) |t, i| if (elemOf(t).lanes != 1 or l == 0) {
            ins[i][l] = lane[i];
        };
        switch (r) {
            .value => |v| exp.lanes[l] = v,
            .undefined => exp.defined[l] = false,
        }
    }
    return exp;
}

fn fixNan(ins: *Inputs, e: Elem) *Inputs {
    for (0..e.lanes) |l| ins[0][l] = if (e.bits == 16) bitsOf(@as(f16, 1.5)) else bitsOf(@as(f32, 1.5));
    return ins;
}

const LaneResult = union(enum) { value: u64, undefined };

fn laneInst(p: Probe, n: Inst, lane: *[3]u64) ?LaneResult {
    const e = elemOf(p.operands[0]);
    const i = n.inst;
    if (std.mem.eql(u8, n.group, "conv")) return conversion(p, i, lane);
    if (e.kind == .float) {
        return switch (e.bits) {
            16 => floatInst(16, p, i, lane),
            else => floatInst(32, p, i, lane),
        };
    }
    return intInst(p, i, lane);
}

fn eq(a: []const u8, b: []const u8) bool {
    return std.mem.eql(u8, a, b);
}

fn anyOf(a: []const u8, list: []const []const u8) bool {
    for (list) |b| if (eq(a, b)) return true;
    return false;
}

fn intInst(p: Probe, i: []const u8, lane: *[3]u64) ?LaneResult {
    const e = elemOf(p.operands[0]);
    const w: u32 = e.bits;
    const m = mask(w);
    const a = lane[0] & m;
    var b = lane[1] & m;
    var c = lane[2] & m;
    const sa = sext(a, w);
    const min_s: u64 = @as(u64, 1) << @intCast(w - 1);
    var r: u64 = 0;
    if (anyOf(i, &.{ "SDiv", "SRem", "SMod" })) {
        if (b == 0 or (a == min_s and b == m)) b = 3;
    }
    if (anyOf(i, &.{ "UDiv", "UMod" }) and b == 0) b = 7;
    if (anyOf(i, &.{ "ShiftLeftLogical", "ShiftRightLogical", "ShiftRightArithmetic" })) b %= w;
    if (anyOf(i, &.{ "BitFieldUExtract", "BitFieldSExtract" })) {
        b %= w + 1;
        c %= w - b + 1;
    }
    if (eq(i, "BitFieldInsert")) c %= w / 2 + 1;
    if (anyOf(i, &.{ "SClamp", "s_clamp" }) and sext(b, w) > sext(c, w)) std.mem.swap(u64, &b, &c);
    if (anyOf(i, &.{ "UClamp", "u_clamp" }) and b > c) std.mem.swap(u64, &b, &c);
    lane.* = .{ a, b, c };
    const sb = sext(b, w);
    const sc = sext(c, w);
    const bool_r = struct {
        fn f(x: bool) LaneResult {
            return .{ .value = @intFromBool(x) };
        }
    }.f;
    if (eq(i, "IAdd")) r = a +% b else if (eq(i, "ISub")) r = a -% b else if (eq(i, "IMul")) r = a *% b else if (eq(i, "SDiv")) r = @bitCast(@divTrunc(sa, sb)) else if (eq(i, "UDiv")) r = a / b else if (eq(i, "SRem")) r = @bitCast(@rem(sa, sb)) else if (eq(i, "SMod")) r = @bitCast(@mod(sa, sb)) else if (eq(i, "UMod")) r = a % b else if (eq(i, "BitwiseAnd")) r = a & b else if (eq(i, "BitwiseOr")) r = a | b else if (eq(i, "BitwiseXor")) r = a ^ b else if (eq(i, "ShiftLeftLogical")) r = a << @intCast(b) else if (eq(i, "ShiftRightLogical")) r = a >> @intCast(b) else if (eq(i, "ShiftRightArithmetic")) r = @bitCast(sa >> @intCast(b)) else if (eq(i, "SNegate")) r = 0 -% a else if (eq(i, "Not")) r = ~a else if (anyOf(i, &.{ "BitCount", "popcount" })) r = @popCount(a) else if (eq(i, "BitReverse")) r = @bitReverse(a) >> @intCast(64 - w) else if (eq(i, "IEqual")) return bool_r(a == b) else if (eq(i, "INotEqual")) return bool_r(a != b) else if (eq(i, "UGreaterThan")) return bool_r(a > b) else if (eq(i, "SGreaterThan")) return bool_r(sa > sb) else if (eq(i, "UGreaterThanEqual")) return bool_r(a >= b) else if (eq(i, "SGreaterThanEqual")) return bool_r(sa >= sb) else if (eq(i, "ULessThan")) return bool_r(a < b) else if (eq(i, "SLessThan")) return bool_r(sa < sb) else if (eq(i, "ULessThanEqual")) return bool_r(a <= b) else if (eq(i, "SLessThanEqual")) return bool_r(sa <= sb) else if (eq(i, "LogicalAnd")) return bool_r(sa < sb and a == b) else if (eq(i, "LogicalOr")) return bool_r(sa < sb or a == b) else if (eq(i, "LogicalEqual")) return bool_r((sa < sb) == (a == b)) else if (eq(i, "LogicalNotEqual")) return bool_r((sa < sb) != (a == b)) else if (eq(i, "LogicalNot")) return bool_r(!(sa < sb)) else if (eq(i, "Select")) r = if (sa < sb) a else b else if (anyOf(i, &.{ "SAbs", "s_abs" })) r = if (sa < 0) 0 -% a else a else if (eq(i, "SSign")) r = @bitCast(@as(i64, std.math.sign(sa))) else if (eq(i, "FindILsb")) r = if (a == 0) m else @ctz(a) else if (eq(i, "FindUMsb")) r = if (a == 0) m else 63 - @clz(a) else if (eq(i, "FindSMsb")) {
        const y = if (sa < 0) ~a & m else a;
        r = if (y == 0) m else 63 - @clz(y);
    } else if (eq(i, "clz")) r = @clz(a) - (64 - w) else if (eq(i, "ctz")) r = if (a == 0) w else @ctz(a) else if (anyOf(i, &.{ "SMin", "s_min" })) r = if (sa < sb) a else b else if (anyOf(i, &.{ "SMax", "s_max" })) r = if (sa > sb) a else b else if (anyOf(i, &.{ "UMin", "u_min" })) r = @min(a, b) else if (anyOf(i, &.{ "UMax", "u_max" })) r = @max(a, b) else if (anyOf(i, &.{ "SClamp", "s_clamp" })) r = @bitCast(@min(@max(sa, sb), sc)) else if (anyOf(i, &.{ "UClamp", "u_clamp" })) r = @min(@max(a, b), c) else if (anyOf(i, &.{ "IAddCarry", "ISubBorrow", "UMulExtended", "SMulExtended", "s_mul_hi", "u_mul_hi" })) {
        var lo: u64 = 0;
        var hi: u64 = 0;
        if (eq(i, "IAddCarry")) {
            lo = (a +% b) & m;
            hi = @intFromBool(lo < a);
        } else if (eq(i, "ISubBorrow")) {
            lo = (a -% b) & m;
            hi = @intFromBool(a < b);
        } else if (anyOf(i, &.{ "UMulExtended", "u_mul_hi" })) {
            const prod = @as(u128, a) * @as(u128, b);
            lo = @truncate(prod);
            hi = @truncate(prod >> @intCast(w));
        } else {
            const prod = @as(i128, sa) * @as(i128, sb);
            lo = @truncate(@as(u128, @bitCast(prod)));
            hi = @truncate(@as(u128, @bitCast(prod)) >> @intCast(w));
        }
        r = if (std.mem.startsWith(u8, p.result, "S_")) (lo ^ hi) else hi;
    } else if (eq(i, "BitFieldUExtract")) r = if (c == 0) 0 else (a >> @intCast(b)) & mask(@intCast(c)) else if (eq(i, "BitFieldSExtract")) r = if (c == 0) 0 else @bitCast(sext((a >> @intCast(b)) & mask(@intCast(c)), @intCast(c))) else if (eq(i, "BitFieldInsert")) {
        const fm = if (c == 0) 0 else (mask(@intCast(c)) << @intCast(c)) & m;
        r = (a & ~fm) | ((b << @intCast(c)) & fm);
    } else return null;
    return .{ .value = r & mask(storeElem(p).bits) };
}

fn floatInst(comptime B: u8, p: Probe, i: []const u8, lane: *[3]u64) ?LaneResult {
    const F = T(B);
    const big: f64 = if (B == 16) 1000 else 1e6;
    // Domains.
    if (anyOf(i, &.{ "Asin", "Acos", "Atanh", "asin", "acos", "atanh" })) clampF(B, &lane[0], -0.95, 0.95);
    if (anyOf(i, &.{ "Acosh", "acosh" })) clampF(B, &lane[0], 1.05, 50);
    if (anyOf(i, &.{ "Log", "Log2", "Sqrt", "InverseSqrt", "log", "log2", "log10", "sqrt", "rsqrt" })) clampF(B, &lane[0], 0.01, big);
    if (anyOf(i, &.{ "Exp", "Exp2", "Sinh", "Cosh", "exp", "exp2", "exp10", "sinh", "cosh" })) clampF(B, &lane[0], -4, 4);
    if (anyOf(i, &.{ "Sin", "Cos", "sin", "cos", "Tan", "tan", "Atan", "atan", "Tanh", "tanh", "Asinh", "asinh" })) clampF(B, &lane[0], -1.4, 1.4);
    if (anyOf(i, &.{ "Pow", "pow" })) {
        clampF(B, &lane[0], 0.1, 10);
        clampF(B, &lane[1], -3, 3);
    }
    if (eq(i, "powr")) {
        // OpenCL defines powr at the edge inputs (±0, ±1, ±inf, NaN,
        // negative x: `powrRef`), so only finite magnitudes are kept in range.
        clampFinite(B, &lane[0], 0.1, 10);
        clampFinite(B, &lane[1], -3, 3);
    }
    if (anyOf(i, &.{ "Atan2", "atan2" })) {
        clampF(B, &lane[0], -10, 10);
        clampF(B, &lane[1], 0.5, 10);
    }
    if (anyOf(i, &.{ "FMix", "mix", "SmoothStep", "smoothstep", "FDiv", "mad" })) {
        clampF(B, &lane[0], -100, 100);
        clampF(B, &lane[1], -100, 100);
        clampF(B, &lane[2], 0.1, 0.9);
    }
    if (anyOf(i, &.{ "SmoothStep", "smoothstep" })) {
        clampF(B, &lane[0], -10, 0);
        clampF(B, &lane[1], 1, 10);
        clampF(B, &lane[2], -12, 12);
    }
    if (eq(i, "FDiv")) clampF(B, &lane[1], 0.5, 50);
    if (anyOf(i, &.{ "FMin", "FMax", "NMin", "NMax", "fmin", "fmax" })) {
        // minnum(+0, -0) may return either zero.
        if (fl(B, lane[0]) == 0 and fl(B, lane[1]) == 0) lane[1] = bitsOf(@as(F, 1));
    }
    if (anyOf(i, &.{ "FClamp", "NClamp", "fclamp" })) {
        clampF(B, &lane[1], -100, 100);
        clampF(B, &lane[2], -100, 100);
        if (fl(B, lane[1]) > fl(B, lane[2])) std.mem.swap(u64, &lane[1], &lane[2]);
        if (fl(B, lane[0]) == 0) lane[0] = bitsOf(@as(F, 0.25));
        if (fl(B, lane[1]) == 0) lane[1] = bitsOf(@as(F, -0.5));
        if (fl(B, lane[2]) == 0) lane[2] = bitsOf(@as(F, 0.5));
    }
    if (anyOf(i, &.{ "Ldexp", "ldexp" })) {
        clampF(B, &lane[0], -1000, 1000);
        lane[1] = @as(u64, @bitCast(@rem(sext(lane[1] & mask(32), 32), 9))) & mask(32);
    }
    if (B == 32) {
        // Keep subnormal inputs out: the library flushes them.
        for (lane, 0..) |*v, k| {
            if (k < p.operands.len and elemOf(p.operands[k]).kind != .float) continue;
            if (isSubnormalOrZero(32, v.*) and fl(32, v.*) != 0) v.* = bitsOf(@as(f32, 0.75));
        }
    }
    const x = fl(B, lane[0]);
    const y = fl(B, lane[1]);
    const z = fl(B, lane[2]);
    const xd: f64 = x;
    const yd: f64 = y;
    const zd: f64 = z;
    const b = struct {
        fn f(v: bool) LaneResult {
            return .{ .value = @intFromBool(v) };
        }
    }.f;
    const nan = std.math.isNan;
    const r: F = blk: {
        if (eq(i, "FAdd")) break :blk x + y;
        if (eq(i, "FSub")) break :blk x - y;
        if (eq(i, "FMul")) break :blk x * y;
        if (eq(i, "FDiv")) break :blk x / y;
        if (anyOf(i, &.{ "FRem", "fmod" })) break :blk @rem(x, y);
        if (eq(i, "FMod")) {
            const rr = @rem(x, y);
            break :blk if (rr != 0 and ((rr < 0) != (y < 0))) rr + y else rr;
        }
        if (eq(i, "FNegate")) break :blk -x;
        if (eq(i, "FOrdEqual")) return b(x == y);
        if (eq(i, "FUnordEqual")) return b(nan(x) or nan(y) or x == y);
        if (eq(i, "FOrdNotEqual")) return b(!nan(x) and !nan(y) and x != y);
        if (eq(i, "FUnordNotEqual")) return b(x != y);
        if (eq(i, "FOrdLessThan")) return b(x < y);
        if (eq(i, "FUnordLessThan")) return b(nan(x) or nan(y) or x < y);
        if (eq(i, "FOrdGreaterThan")) return b(x > y);
        if (eq(i, "FUnordGreaterThan")) return b(nan(x) or nan(y) or x > y);
        if (eq(i, "FOrdLessThanEqual")) return b(x <= y);
        if (eq(i, "FUnordLessThanEqual")) return b(nan(x) or nan(y) or x <= y);
        if (eq(i, "FOrdGreaterThanEqual")) return b(x >= y);
        if (eq(i, "FUnordGreaterThanEqual")) return b(nan(x) or nan(y) or x >= y);
        if (eq(i, "Ordered")) return b(!nan(x) and !nan(y));
        if (eq(i, "Unordered")) return b(nan(x) or nan(y));
        if (eq(i, "IsNan")) return b(nan(x));
        if (eq(i, "IsInf")) return b(std.math.isInf(x));
        if (eq(i, "IsFinite")) return b(!nan(x) and !std.math.isInf(x));
        if (anyOf(i, &.{ "FAbs", "fabs" })) break :blk @abs(x);
        if (anyOf(i, &.{ "Floor", "floor" })) break :blk @floor(x);
        if (anyOf(i, &.{ "Ceil", "ceil" })) break :blk @ceil(x);
        if (anyOf(i, &.{ "Trunc", "trunc" })) break :blk @trunc(x);
        if (anyOf(i, &.{ "Round", "round" })) break :blk @round(x);
        if (anyOf(i, &.{ "RoundEven", "rint" })) {
            const rr = @round(x);
            if (@abs(x - @trunc(x)) == 0.5 and @rem(rr, 2) != 0) break :blk rr - std.math.sign(x);
            break :blk rr;
        }
        if (eq(i, "Fract")) break :blk x - @floor(x);
        if (eq(i, "FSign")) {
            if (nan(x)) return .undefined;
            break :blk if (x > 0) 1 else if (x < 0) -1 else x;
        }
        if (eq(i, "sign")) break :blk if (nan(x)) 0 else if (x > 0) 1 else if (x < 0) -1 else x;
        if (anyOf(i, &.{ "Step", "step" })) break :blk if (y < x) 0 else 1;
        if (anyOf(i, &.{ "FMin", "NMin", "fmin" })) break :blk if (nan(x)) y else if (nan(y)) x else @min(x, y);
        if (anyOf(i, &.{ "FMax", "NMax", "fmax" })) break :blk if (nan(x)) y else if (nan(y)) x else @max(x, y);
        if (anyOf(i, &.{ "FClamp", "NClamp", "fclamp" })) {
            if (nan(x)) return .undefined;
            break :blk @min(@max(x, y), z);
        }
        if (anyOf(i, &.{ "Fma", "fma" })) break :blk @mulAdd(F, x, y, z);
        if (eq(i, "mad")) break :blk x * y + z;
        if (eq(i, "copysign")) break :blk std.math.copysign(x, y);
        if (anyOf(i, &.{ "Radians", "radians" })) break :blk x * @as(F, @floatCast(std.math.pi / 180.0));
        if (anyOf(i, &.{ "Degrees", "degrees" })) break :blk x * @as(F, @floatCast(180.0 / std.math.pi));
        if (eq(i, "FMix")) break :blk x * (1 - z) + y * z;
        if (eq(i, "mix")) break :blk x + (y - x) * z;
        if (anyOf(i, &.{ "SmoothStep", "smoothstep" })) {
            const t = std.math.clamp((zd - xd) / (yd - xd), 0, 1);
            break :blk @floatCast(t * t * (3 - 2 * t));
        }
        if (anyOf(i, &.{ "Ldexp", "ldexp" })) break :blk @floatCast(std.math.ldexp(xd, @as(i32, @truncate(sext(lane[1], 32)))));
        const d: f64 = if (anyOf(i, &.{ "Sin", "sin" })) @sin(xd) else if (anyOf(i, &.{ "Cos", "cos" })) @cos(xd) else if (anyOf(i, &.{ "Tan", "tan" })) @tan(xd) else if (anyOf(i, &.{ "Asin", "asin" })) std.math.asin(xd) else if (anyOf(i, &.{ "Acos", "acos" })) std.math.acos(xd) else if (anyOf(i, &.{ "Atan", "atan" })) std.math.atan(xd) else if (anyOf(i, &.{ "Sinh", "sinh" })) std.math.sinh(xd) else if (anyOf(i, &.{ "Cosh", "cosh" })) std.math.cosh(xd) else if (anyOf(i, &.{ "Tanh", "tanh" })) std.math.tanh(xd) else if (anyOf(i, &.{ "Asinh", "asinh" })) std.math.asinh(xd) else if (anyOf(i, &.{ "Acosh", "acosh" })) std.math.acosh(xd) else if (anyOf(i, &.{ "Atanh", "atanh" })) std.math.atanh(xd) else if (anyOf(i, &.{ "Exp", "exp" })) @exp(xd) else if (anyOf(i, &.{ "Exp2", "exp2" })) @exp2(xd) else if (eq(i, "exp10")) std.math.pow(f64, 10, xd) else if (anyOf(i, &.{ "Log", "log" })) @log(xd) else if (anyOf(i, &.{ "Log2", "log2" })) @log2(xd) else if (eq(i, "log10")) @log10(xd) else if (anyOf(i, &.{ "Sqrt", "sqrt" })) @sqrt(xd) else if (anyOf(i, &.{ "InverseSqrt", "rsqrt" })) 1 / @sqrt(xd) else if (anyOf(i, &.{ "Atan2", "atan2" })) std.math.atan2(xd, yd) else if (anyOf(i, &.{ "Pow", "pow" })) std.math.pow(f64, xd, yd) else if (eq(i, "powr")) powrRef(xd, yd) else return null;
        break :blk @floatCast(d);
    };
    return .{ .value = bitsOf(r) };
}

/// Round-half-even of an f64 holding an f16/f32 value.
fn rint(x: f64) f64 {
    const lo = @floor(x);
    const d = x - lo;
    if (d < 0.5) return lo;
    if (d > 0.5) return lo + 1;
    return if (@mod(lo, 2) == 0) lo else lo + 1;
}

/// The SaturatedConversion / FPRoundingMode probes
/// (`conv_ConvertFToSSatRte_f32_i8`): `null` when `i` is neither.
fn decoratedConversion(p: Probe, i: []const u8, lane: *[3]u64) ?LaneResult {
    const from = elemOf(p.operands[0]);
    const to = elemOf(p.result);
    var base = i;
    var mode: u8 = 'z';
    for ([_][]const u8{ "Rte", "Rtp", "Rtn" }) |sfx| if (std.mem.endsWith(u8, base, sfx)) {
        mode = sfx[2];
        base = base[0 .. base.len - 3];
    };
    const sat = std.mem.endsWith(u8, base, "Sat");
    if (sat) base = base[0 .. base.len - 3];
    if (!sat and mode == 'z') return null;
    const v = lane[0];
    if (eq(base, "UConvert")) return .{ .value = @min(v & mask(from.bits), mask(to.bits)) };
    if (eq(base, "SConvert")) {
        const x = sext(v & mask(from.bits), from.bits);
        const hi: i64 = @intCast(mask(to.bits - 1));
        return .{ .value = @as(u64, @bitCast(std.math.clamp(x, -hi - 1, hi))) & mask(to.bits) };
    }
    const signed = eq(base, "ConvertFToS");
    if (!signed and !eq(base, "ConvertFToU")) return null;
    if (!sat) {
        // Keep the rounded value in range (out of range is undefined).
        const hi: f64 = if (signed) @min(@as(f64, @floatFromInt(@as(u64, 1) << @intCast(to.bits - 1))) - 2, 60000) else @min(@as(f64, @floatFromInt(mask(to.bits))) - 1, 60000);
        const lo: f64 = if (signed) -hi else 0;
        if (from.bits == 16) clampF(16, &lane[0], lo, hi) else clampF(32, &lane[0], lo, hi);
    }
    const x: f64 = if (from.bits == 16) fl(16, lane[0]) else fl(32, lane[0]);
    if (std.math.isNan(x)) return .{ .value = 0 };
    const t: f64 = switch (mode) {
        'e' => rint(x),
        'p' => @ceil(x),
        'n' => @floor(x),
        else => @trunc(x),
    };
    const top: f64 = std.math.pow(f64, 2, @floatFromInt(if (signed) to.bits - 1 else to.bits));
    const max: u64 = if (signed) mask(to.bits - 1) else mask(to.bits);
    if (t >= top) return .{ .value = max };
    if (signed and t < -top) return .{ .value = (~max) & mask(to.bits) };
    if (!signed and t < 0) return .{ .value = 0 };
    if (signed) return .{ .value = @as(u64, @bitCast(@as(i64, @intFromFloat(t)))) & mask(to.bits) };
    return .{ .value = @as(u64, @intFromFloat(t)) & mask(to.bits) };
}

fn conversion(p: Probe, i: []const u8, lane: *[3]u64) ?LaneResult {
    if (decoratedConversion(p, i, lane)) |r| return r;
    const from = elemOf(p.operands[0]);
    const to = elemOf(p.result);
    var v = lane[0];
    if (eq(i, "UConvert")) return .{ .value = (v & mask(from.bits)) & mask(to.bits) };
    if (eq(i, "SConvert")) return .{ .value = @as(u64, @bitCast(sext(v & mask(from.bits), from.bits))) & mask(to.bits) };
    if (eq(i, "FConvert")) {
        if (from.bits == 16) {
            const x = fl(16, v);
            if (std.math.isNan(x)) return .undefined;
            return .{ .value = bitsOf(@as(f32, x)) };
        }
        const x = fl(32, v);
        if (std.math.isNan(x)) return .undefined;
        if (isSubnormalOrZero(32, v) and x != 0) {
            lane[0] = bitsOf(@as(f32, 0.75));
            v = lane[0];
        }
        return .{ .value = bitsOf(@as(f16, @floatCast(fl(32, v)))) };
    }
    if (eq(i, "ConvertFToS") or eq(i, "ConvertFToU")) {
        const signed = eq(i, "ConvertFToS");
        const hi: f64 = if (signed) @min(@as(f64, @floatFromInt(@as(u64, 1) << @intCast(to.bits - 1))) - 1, 60000) else @min(@as(f64, @floatFromInt(mask(to.bits))), 60000);
        const lo: f64 = if (signed) -hi else 0;
        if (from.bits == 16) clampF(16, &lane[0], lo, hi) else clampF(32, &lane[0], lo, hi);
        const x: f64 = if (from.bits == 16) fl(16, lane[0]) else fl(32, lane[0]);
        const t = @trunc(x);
        const iv: i64 = @intFromFloat(t);
        return .{ .value = @as(u64, @bitCast(iv)) & mask(to.bits) };
    }
    if (eq(i, "ConvertSToF") or eq(i, "ConvertUToF")) {
        const signed = eq(i, "ConvertSToF");
        const iv: f64 = if (signed) @floatFromInt(sext(v & mask(from.bits), from.bits)) else @floatFromInt(v & mask(from.bits));
        if (to.bits == 16) {
            // Integers convert with one rounding; f64 holds every 32-bit value exactly.
            if (!signed and from.bits == 64) return .undefined;
            if (signed and from.bits == 64) return .undefined;
            return .{ .value = bitsOf(@as(f16, @floatCast(iv))) };
        }
        if (from.bits == 64) {
            const r: f32 = if (signed) @floatFromInt(sext(v, 64)) else @floatFromInt(v);
            return .{ .value = bitsOf(r) };
        }
        return .{ .value = bitsOf(@as(f32, @floatCast(iv))) };
    }
    return null;
}

/// The composite-select probes (probes.zig, `comp_Select_*`).
fn compInst(i: []const u8, ins: *Inputs, out: Elem) ?Expect {
    var exp: Expect = .{};
    for (exp.defined[out.lanes..]) |*d| d.* = false;
    const m32 = mask(32);
    if (eq(i, "Select_struct_u32_v4u32")) {
        const a = ins[0][0] & m32;
        const c = ins[2][0] & m32;
        if (a < c) {
            exp.lanes = .{ a, ins[1][1] & m32, ins[1][2] & m32, ins[1][3] & m32 };
        } else {
            exp.lanes = .{ c, (ins[1][1] *% 2) & m32, (ins[1][2] *% 2) & m32, (ins[1][3] *% 2) & m32 };
        }
    } else if (eq(i, "Select_array_u32")) {
        const a = ins[0][0] & m32;
        const b = ins[1][0] & m32;
        const c = ins[2][0] & m32;
        exp.lanes = if (a < c) .{ a, b, 0, 0 } else .{ b, c, 0, 0 };
    } else if (eq(i, "Select_nested_bool_v3u32")) {
        const b = ins[1][0] & m32;
        const c = ins[2][0] & m32;
        if (b < c) {
            exp.lanes = .{ ins[0][0] & m32, ins[0][1] & m32, ins[0][2] & m32, (b + 1) & m32 };
        } else {
            exp.lanes = .{ (ins[0][0] *% 2) & m32, (ins[0][1] *% 2) & m32, (ins[0][2] *% 2) & m32, c };
        }
    } else return null;
    return exp;
}

fn vecInst(i: []const u8, ins: *Inputs, out: Elem) ?Expect {
    var exp: Expect = .{};
    for (exp.defined[out.lanes..]) |*d| d.* = false;
    const f = struct {
        fn g(v: u64) f32 {
            return fl(32, v);
        }
    }.g;
    if (eq(i, "VectorShuffle_v4f32")) {
        exp.lanes = .{ ins[1][3], ins[0][0], 0, ins[0][2] };
        exp.defined[2] = false;
    } else if (eq(i, "VectorShuffle_v2f32_v4f32")) {
        exp.lanes = .{ ins[1][3], ins[0][1], ins[0][0], 0 };
    } else if (eq(i, "VectorExtractDynamic_v4f32")) {
        ins[1][0] %= 4;
        exp.lanes[0] = ins[0][ins[1][0]];
    } else if (eq(i, "VectorInsertDynamic_v4f32")) {
        ins[2][0] %= 4;
        exp.lanes = ins[0];
        exp.lanes[ins[2][0]] = ins[1][0];
    } else if (eq(i, "VectorTimesScalar_v4f32")) {
        for (0..4) |l| exp.lanes[l] = bitsOf(f(ins[0][l]) * f(ins[1][0]));
    } else if (eq(i, "VectorTimesScalar_v3f16")) {
        for (0..3) |l| exp.lanes[l] = bitsOf(fl(16, ins[0][l]) * fl(16, ins[1][0]));
    } else if (std.mem.startsWith(u8, i, "Dot_")) {
        const e = elemOf(i[4..]);
        var sum: f64 = 0;
        for (0..e.lanes) |l| {
            if (e.bits == 16) {
                clampF(16, &ins[0][l], -10, 10);
                clampF(16, &ins[1][l], -10, 10);
                sum += @as(f64, fl(16, ins[0][l])) * @as(f64, fl(16, ins[1][l]));
            } else {
                clampF(32, &ins[0][l], -1000, 1000);
                clampF(32, &ins[1][l], -1000, 1000);
                sum += @as(f64, f(ins[0][l])) * @as(f64, f(ins[1][l]));
            }
        }
        exp.lanes[0] = if (e.bits == 16) bitsOf(@as(f16, @floatCast(sum))) else bitsOf(@as(f32, @floatCast(sum)));
    } else if (eq(i, "CompositeConstruct_v4f32")) {
        exp.lanes = .{ ins[0][0], ins[0][1], ins[1][0], ins[1][0] };
    } else if (eq(i, "CompositeExtractInsert_v3u8")) {
        exp.lanes = .{ (ins[0][2] +% ins[1][0]) & 0xff, ins[0][1] & 0xff, ins[0][2] & 0xff, 0 };
    } else return null;
    // Float lanes that are NaN compare as NaN; nothing else to fix.
    return exp;
}

fn geometry(i: []const u8, ins: *Inputs, e: Elem, out: Elem) ?Expect {
    if (anyOf(i, &.{ "length", "distance", "normalize" })) return oclGeometry(i, ins, e, out);
    var exp: Expect = .{};
    for (exp.defined[out.lanes..]) |*d| d.* = false;
    var a: [4]f64 = undefined;
    var b: [4]f64 = undefined;
    for (0..e.lanes) |l| {
        if (e.bits == 16) {
            clampF(16, &ins[0][l], -10, 10);
            clampF(16, &ins[1][l], -10, 10);
            a[l] = fl(16, ins[0][l]);
            b[l] = fl(16, ins[1][l]);
        } else {
            clampF(32, &ins[0][l], -100, 100);
            clampF(32, &ins[1][l], -100, 100);
            a[l] = fl(32, ins[0][l]);
            b[l] = fl(32, ins[1][l]);
        }
        // Keep vectors away from zero length.
        if (@abs(a[l]) < 0.1) {
            a[l] = 0.5;
            ins[0][l] = if (e.bits == 16) bitsOf(@as(f16, 0.5)) else bitsOf(@as(f32, 0.5));
        }
    }
    const n = e.lanes;
    var r: [4]f64 = .{ 0, 0, 0, 0 };
    if (eq(i, "Length") or eq(i, "length")) {
        var s: f64 = 0;
        for (0..n) |l| s += a[l] * a[l];
        r[0] = @sqrt(s);
    } else if (eq(i, "Distance") or eq(i, "distance")) {
        var s: f64 = 0;
        for (0..n) |l| s += (a[l] - b[l]) * (a[l] - b[l]);
        r[0] = @sqrt(s);
    } else if (eq(i, "Normalize") or eq(i, "normalize")) {
        var s: f64 = 0;
        for (0..n) |l| s += a[l] * a[l];
        for (0..n) |l| r[l] = a[l] / @sqrt(s);
    } else {
        r[0] = a[1] * b[2] - a[2] * b[1];
        r[1] = a[2] * b[0] - a[0] * b[2];
        r[2] = a[0] * b[1] - a[1] * b[0];
        r[3] = 0;
    }
    for (0..out.lanes) |l| exp.lanes[l] = if (out.bits == 16) bitsOf(@as(f16, @floatCast(r[l]))) else bitsOf(@as(f32, @floatCast(r[l])));
    return exp;
}

/// OpenCL C's length, distance and normalize on unclamped inputs (the edge
/// values included): the exact result in f64, so a sum of squares that
/// overflows f32 is no excuse, and normalize's special cases (3.0,
/// geometric functions): v when every element is zero, all NaN when one is,
/// infinite elements as copysign(1, e) and the finite ones as 0 * e.
fn oclGeometry(i: []const u8, ins: *Inputs, e: Elem, out: Elem) ?Expect {
    var exp: Expect = .{};
    for (exp.defined[out.lanes..]) |*d| d.* = false;
    const n = e.lanes;
    var a: [4]f64 = undefined;
    for (0..n) |l| {
        if (e.bits == 32 and isSubnormalOrZero(32, ins[0][l]) and fl(32, ins[0][l]) != 0) ins[0][l] = bitsOf(@as(f32, 0.75));
        if (e.bits == 32 and isSubnormalOrZero(32, ins[1][l]) and fl(32, ins[1][l]) != 0) ins[1][l] = bitsOf(@as(f32, 0.75));
        const x: f64 = if (e.bits == 16) fl(16, ins[0][l]) else fl(32, ins[0][l]);
        const y: f64 = if (e.bits == 16) fl(16, ins[1][l]) else fl(32, ins[1][l]);
        a[l] = if (eq(i, "distance")) x - y else x;
        // The kernel subtracts in the element type.
        if (eq(i, "distance")) a[l] = if (e.bits == 16) @as(f16, @floatCast(a[l])) else @as(f32, @floatCast(a[l]));
    }
    var r: [4]f64 = .{ 0, 0, 0, 0 };
    if (eq(i, "normalize")) {
        var any_nan = false;
        var any_inf = false;
        var all_zero = true;
        for (a[0..n]) |v| {
            any_nan = any_nan or std.math.isNan(v);
            any_inf = any_inf or std.math.isInf(v);
            all_zero = all_zero and v == 0;
        }
        if (any_nan) {
            for (0..n) |l| r[l] = std.math.nan(f64);
        } else if (all_zero) {
            for (0..n) |l| r[l] = a[l];
        } else {
            var v = a;
            if (any_inf) for (v[0..n]) |*c| {
                c.* = if (std.math.isInf(c.*)) std.math.copysign(@as(f64, 1), c.*) else 0 * c.*;
            };
            var s: f64 = 0;
            for (v[0..n]) |c| s += c * c;
            for (0..n) |l| r[l] = v[l] / @sqrt(s);
        }
    } else {
        var s: f64 = 0;
        for (a[0..n]) |c| s += c * c;
        r[0] = @sqrt(s);
    }
    for (0..out.lanes) |l| exp.lanes[l] = if (out.bits == 16) bitsOf(@as(f16, @floatCast(r[l]))) else bitsOf(@as(f32, @floatCast(r[l])));
    return exp;
}

fn boolReduce(i: []const u8, ins: *Inputs, e: Elem, out: Elem) ?Expect {
    var exp: Expect = .{};
    for (exp.defined[out.lanes..]) |*d| d.* = false;
    if (eq(i, "Select_scalar_cond")) {
        exp.lanes = ins[0];
        for (0..4) |l| if (std.math.isNan(fl(32, ins[0][l]))) {
            exp.defined[l] = false;
        };
        return exp;
    }
    var any = false;
    var all = true;
    for (0..e.lanes) |l| {
        const lt = sext(ins[0][l] & mask(32), 32) < sext(ins[1][l] & mask(32), 32);
        any = any or lt;
        all = all and lt;
    }
    exp.lanes[0] = @intFromBool(if (std.mem.startsWith(u8, i, "Any")) any else all);
    return exp;
}

/// Whether `got` matches the expected lane.
pub fn laneMatches(out: Elem, check: Check, want: u64, got: u64) bool {
    if (out.kind != .float) return (want & mask(out.bits)) == (got & mask(out.bits));
    const w: f64 = if (out.bits == 16) fl(16, want) else fl(32, want);
    const g: f64 = if (out.bits == 16) fl(16, got) else fl(32, got);
    if (std.math.isNan(w) or std.math.isNan(g)) return std.math.isNan(w) and std.math.isNan(g);
    if (check == .exact) {
        if ((want & mask(out.bits)) == (got & mask(out.bits))) return true;
        // An f32 subnormal result may be flushed to a zero of either sign.
        if (out.bits == 32 and isSubnormalOrZero(32, want) and g == 0) return true;
        return false;
    }
    if (std.math.isInf(w) or std.math.isInf(g)) return w == g;
    const tol: f64 = if (out.bits == 16) 2e-2 else 2e-4;
    return @abs(w - g) <= tol * @max(1.0, @abs(w));
}

// ── Inputs ──────────────────────────────────────────────────────────────────

const int_edges = [_]u64{ 0, 1, ~@as(u64, 0), 2, 3, 0x7f, 0x80, 0x7fff, 0x8000, 0x7fffffff, 0x80000000, 0xffffffff, 0x7fffffffffffffff, 0x8000000000000000, 31, 32 };
const float_edges = [_]f64{ 0, -0.0, 1, -1, 0.5, -2.5, std.math.inf(f64), -std.math.inf(f64), std.math.nan(f64), 3, 7.25, -1e5, 1e-3, 1.5, -0.5, 2.5 };

/// Input bits for element `t` of thread `thread`, lane `lane`, operand `op`.
pub fn inputBits(e: Elem, op: usize, thread: usize, lane: usize, rng: *std.Random.DefaultPrng) u64 {
    const k = (thread * 4 + lane + op * 5) % 64;
    const r = rng.random().int(u64);
    switch (e.kind) {
        .int, .uint, .bool => {
            const v = if (k < int_edges.len) int_edges[k] else r;
            return v & mask(e.bits);
        },
        .float => {
            const x: f64 = if (k < float_edges.len) float_edges[k] else (@as(f64, @floatFromInt(r % 200001)) - 100000.0) / 64.0;
            return if (e.bits == 16) bitsOf(@as(f16, @floatCast(std.math.clamp(x, -60000, 60000)))) else bitsOf(@as(f32, @floatCast(x)));
        },
    }
}

test "reference: a few hand-computed lanes" {
    const p_smod: Probe = .{ .name = "op_SMod_i32", .operands = &.{ "i32", "i32" }, .result = "i32", .body = "" };
    var ins: Inputs = .{ .{ @as(u64, @bitCast(@as(i64, -7))) & mask(32), 0, 0, 0 }, .{ 3, 0, 0, 0 }, .{ 0, 0, 0, 0 } };
    const e = prepare(p_smod, &ins).?;
    try std.testing.expectEqual(@as(u64, 2), e.lanes[0]);
    const p_hi: Probe = .{ .name = "op_UMulExtended_u64", .operands = &.{ "u64", "u64" }, .result = "S_u64", .body = "" };
    var ins2: Inputs = .{ .{ ~@as(u64, 0), 0, 0, 0 }, .{ ~@as(u64, 0), 0, 0, 0 }, .{ 0, 0, 0, 0 } };
    // (2^64-1)^2 = 2^128 - 2^65 + 1: lo 1, hi 2^64 - 2; xor = 2^64 - 1.
    try std.testing.expectEqual(~@as(u64, 0), prepare(p_hi, &ins2).?.lanes[0]);
    const p_msb: Probe = .{ .name = "glsl_FindSMsb_i32", .operands = &.{"i32"}, .result = "i32", .body = "" };
    var ins3: Inputs = .{ .{ @as(u64, @bitCast(@as(i64, -2))) & mask(32), 0, 0, 0 }, .{ 0, 0, 0, 0 }, .{ 0, 0, 0, 0 } };
    try std.testing.expectEqual(@as(u64, 0), prepare(p_msb, &ins3).?.lanes[0]);
    try std.testing.expect(laneMatches(.{ .kind = .float, .bits = 32, .lanes = 1 }, .approx, bitsOf(@as(f32, 1.0)), bitsOf(@as(f32, 1.00001))));
    try std.testing.expect(!laneMatches(.{ .kind = .float, .bits = 32, .lanes = 1 }, .exact, bitsOf(@as(f32, 1.0)), bitsOf(@as(f32, 1.00001))));
}
