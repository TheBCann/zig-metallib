//! MTLB (.metallib) container writer.
//!
//! Layout, reverse-engineered from `xcrun metal` output and verified by
//! repacking Apple's own bitcode and loading the result through
//! `newLibraryWithData:` (see tools/metallib_check.zig):
//!
//!   header (88 bytes)
//!   function list      u32 count, then per function: u32 size, tags..., ENDT
//!   header extension   HDYN, RLST, UUID, ENDT (format 2.9, macOS 26); the
//!                      older formats have RLST, UUID, ENDT only
//!   public metadata    per function: u32 8, "ENDT"
//!   private metadata   per function: u32 8, "ENDT"
//!   bitcode            per function: darwin bitcode wrapper + LLVM bitcode,
//!                      zero-padded to a multiple of 16 bytes (metal-objdump
//!                      rejects unpadded blobs with "unknown magic"; the
//!                      wrapper's own size field stays the unpadded length)
//!   dynamic header     NAME "default.metallib\0", ENDT (format 2.9 only)
//!   reflection list    u32 count = 0 (the runtime ignores it; metal-objdump
//!                      refuses to open a library without the count)
//!
//! Every tag is `4-char id, u16 length, payload`. All integers little-endian.

const std = @import("std");
const target = @import("target.zig");
const Allocator = std.mem.Allocator;
const Sha256 = std.crypto.hash.sha2.Sha256;

pub const Stage = enum(u8) { vertex = 0, fragment = 1, kernel = 2 };

pub const Function = struct {
    name: []const u8,
    stage: Stage,
    /// Raw LLVM bitcode (no wrapper) for a module that defines just this function.
    bitcode: []const u8,
};

const header_len = 88;
const wrapper_len = 20;
/// Every wrapped blob is padded to this alignment; MDSZ, HASH and the
/// bitcode section size all describe the padded bytes (Apple's layout).
const blob_align = 16;
const bitcode_wrapper_magic: u32 = 0x0B17C0DE;

/// Serialise `functions` into a complete .metallib image stamped for the
/// deployment target `profile` (header words and each function's `VERS`).
pub fn pack(gpa: Allocator, functions: []const Function, library_name: []const u8, profile: target.Profile) Allocator.Error![]u8 {
    var fn_list: std.ArrayList(u8) = .empty;
    var pub_md: std.ArrayList(u8) = .empty;
    var priv_md: std.ArrayList(u8) = .empty;
    var bitcode: std.ArrayList(u8) = .empty;

    try appendInt(&fn_list, gpa, u32, @intCast(functions.len));
    for (functions) |f| {
        const blob_off = bitcode.items.len;
        try appendWrappedBitcode(&bitcode, gpa, f.bitcode);
        const padded_len = std.mem.alignForward(usize, bitcode.items.len - blob_off, blob_align);
        try bitcode.appendNTimes(gpa, 0, blob_off + padded_len - bitcode.items.len);
        const blob = bitcode.items[blob_off..];

        var digest: [Sha256.digest_length]u8 = undefined;
        Sha256.hash(blob, &digest, .{});

        var entry: std.ArrayList(u8) = .empty;
        try appendTagZ(&entry, gpa, "NAME", f.name);
        try appendTag(&entry, gpa, "TYPE", &.{@backingInt(f.stage)});
        try appendTag(&entry, gpa, "HASH", &digest);
        var offt: [24]u8 = undefined;
        std.mem.writeInt(u64, offt[0..8], pub_md.items.len, .little);
        std.mem.writeInt(u64, offt[8..16], priv_md.items.len, .little);
        std.mem.writeInt(u64, offt[16..24], blob_off, .little);
        try appendTag(&entry, gpa, "OFFT", &offt);
        // AIR version then Metal language version, 4 x u16: 2, 8, 4, 0 for
        // macOS 26 (metalfe-32023.883 output; see target.zig).
        var vers: [8]u8 = undefined;
        std.mem.writeInt(u16, vers[0..2], 2, .little);
        std.mem.writeInt(u16, vers[2..4], profile.air_minor, .little);
        std.mem.writeInt(u16, vers[4..6], profile.lang_major, .little);
        std.mem.writeInt(u16, vers[6..8], profile.lang_minor, .little);
        try appendTag(&entry, gpa, "VERS", &vers);
        // MDSZ = the padded blob length.
        var mdsz: [8]u8 = undefined;
        std.mem.writeInt(u64, &mdsz, blob.len, .little);
        try appendTag(&entry, gpa, "MDSZ", &mdsz);
        try appendTag(&entry, gpa, "RFLT", &@as([8]u8, @splat(0)));
        try entry.appendSlice(gpa, "ENDT");

        // Entry size counts its own 4-byte size field.
        try appendInt(&fn_list, gpa, u32, @intCast(4 + entry.items.len));
        try fn_list.appendSlice(gpa, entry.items);

        try appendInt(&pub_md, gpa, u32, 8);
        try pub_md.appendSlice(gpa, "ENDT");
        try appendInt(&priv_md, gpa, u32, 8);
        try priv_md.appendSlice(gpa, "ENDT");
    }

    // Only format 2.9 (macOS 26) has the dynamic header and the HDYN tag that
    // points at it; for the older targets Apple's reflection list follows the
    // bitcode directly and the extension is RLST, UUID, ENDT (target.zig).
    var dyn: std.ArrayList(u8) = .empty;
    if (profile.dynamic_header) {
        try appendTagZ(&dyn, gpa, "NAME", library_name);
        try dyn.appendSlice(gpa, "ENDT");
    }

    const ext_tags: usize = if (profile.dynamic_header) 3 else 2; // [HDYN] RLST UUID
    const ext_len = (4 + 2 + 16) * ext_tags + 4; // ... ENDT
    const fn_list_off = header_len;
    const ext_off = fn_list_off + fn_list.items.len;
    const pub_off = ext_off + ext_len;
    const priv_off = pub_off + pub_md.items.len;
    const bc_off = priv_off + priv_md.items.len;
    const dyn_off = bc_off + bitcode.items.len;
    const rlst_off = dyn_off + dyn.items.len;
    const rlst_len = 4; // u32 entry count, zero entries
    const total = rlst_off + rlst_len;

    // Deterministic UUID: hash of every function blob, so identical shaders
    // produce identical libraries (the runtime uses it as a cache key).
    var uuid: [Sha256.digest_length]u8 = undefined;
    Sha256.hash(bitcode.items, &uuid, .{});

    var out: std.ArrayList(u8) = .empty;
    try out.ensureTotalCapacity(gpa, total);

    // Header. 0x8001 and 0x8100 are copied from Apple's output and are the
    // same for every deployment target (0x8001 looks like platform + flags).
    // The format version's minor half and the deployment target after it
    // follow the target: 2.9 / 26.0 for macOS 26, 2.8 / 15.0 for macOS 15,
    // 2.7 / 14.0 and 2.7 / 13.0 below that. The target is two u16s, major then
    // minor (a 26.1 target writes 26, 1), and it is the target's version, not
    // the SDK's: the macOS 26.5 SDK writes 15, 0 when targeting macOS 15.
    try out.appendSlice(gpa, "MTLB");
    try appendInt(&out, gpa, u16, 0x8001);
    try appendInt(&out, gpa, u16, 2);
    try appendInt(&out, gpa, u16, profile.container_minor);
    try appendInt(&out, gpa, u16, 0x8100);
    try appendInt(&out, gpa, u16, profile.macos_major);
    try appendInt(&out, gpa, u16, 0);
    try appendInt(&out, gpa, u64, total);
    try appendInt(&out, gpa, u64, fn_list_off);
    // Apple's size excludes the last entry's ENDT; the reader stops at ENDT anyway.
    try appendInt(&out, gpa, u64, fn_list.items.len - 4);
    try appendInt(&out, gpa, u64, pub_off);
    try appendInt(&out, gpa, u64, pub_md.items.len);
    try appendInt(&out, gpa, u64, priv_off);
    try appendInt(&out, gpa, u64, priv_md.items.len);
    try appendInt(&out, gpa, u64, bc_off);
    try appendInt(&out, gpa, u64, bitcode.items.len);
    std.debug.assert(out.items.len == header_len);

    try out.appendSlice(gpa, fn_list.items);

    var off_size: [16]u8 = undefined;
    if (profile.dynamic_header) {
        std.mem.writeInt(u64, off_size[0..8], dyn_off, .little);
        std.mem.writeInt(u64, off_size[8..16], dyn.items.len, .little);
        try appendTag(&out, gpa, "HDYN", &off_size);
    }
    std.mem.writeInt(u64, off_size[0..8], rlst_off, .little);
    std.mem.writeInt(u64, off_size[8..16], rlst_len, .little);
    try appendTag(&out, gpa, "RLST", &off_size);
    try appendTag(&out, gpa, "UUID", uuid[0..16]);
    try out.appendSlice(gpa, "ENDT");
    std.debug.assert(out.items.len == pub_off);

    try out.appendSlice(gpa, pub_md.items);
    try out.appendSlice(gpa, priv_md.items);
    try out.appendSlice(gpa, bitcode.items);
    try out.appendSlice(gpa, dyn.items);
    try appendInt(&out, gpa, u32, 0);
    std.debug.assert(out.items.len == total);

    return out.toOwnedSlice(gpa);
}

/// What a container header says about the library: the format version's
/// minor half, the deployment target, and the first function's `VERS`
/// (AIR major, AIR minor, Metal major, Metal minor), which is what the
/// library was actually compiled for.
pub const Stamp = struct {
    container_minor: u16,
    macos: target.Version,
    vers: ?[4]u16,
};

/// Decode `bytes`' header, or null when it is not an MTLB container.
pub fn readStamp(bytes: []const u8) ?Stamp {
    if (bytes.len < header_len or !std.mem.eql(u8, bytes[0..4], "MTLB")) return null;
    var stamp = Stamp{
        .container_minor = std.mem.readInt(u16, bytes[8..10], .little),
        .macos = .{
            .major = std.mem.readInt(u16, bytes[12..14], .little),
            .minor = std.mem.readInt(u16, bytes[14..16], .little),
        },
        .vers = null,
    };
    // First function-list entry: u32 count, u32 entry size, then tags.
    // The offset comes from the file: compare without adding to it, so a
    // hostile value cannot overflow.
    const list_off = std.mem.readInt(u64, bytes[24..32], .little);
    if (list_off > bytes.len or bytes.len - list_off < 8) return stamp;
    const entry_size = std.mem.readInt(u32, bytes[list_off + 4 ..][0..4], .little);
    const end = @min(bytes.len, list_off + 4 + entry_size);
    var i: usize = list_off + 8;
    while (i + 6 <= end) {
        const id = bytes[i..][0..4];
        if (std.mem.eql(u8, id, "ENDT")) break;
        const len = std.mem.readInt(u16, bytes[i + 4 ..][0..2], .little);
        if (std.mem.eql(u8, id, "VERS") and len == 8 and i + 14 <= end) {
            var v: [4]u16 = undefined;
            for (&v, 0..) |*x, k| x.* = std.mem.readInt(u16, bytes[i + 6 + 2 * k ..][0..2], .little);
            stamp.vers = v;
            break;
        }
        i += 6 + len;
    }
    return stamp;
}

/// Darwin bitcode wrapper: magic, version, header size, bitcode size
/// (unpadded), cputype.
fn appendWrappedBitcode(list: *std.ArrayList(u8), gpa: Allocator, bc: []const u8) Allocator.Error!void {
    try appendInt(list, gpa, u32, bitcode_wrapper_magic);
    try appendInt(list, gpa, u32, 0);
    try appendInt(list, gpa, u32, wrapper_len);
    try appendInt(list, gpa, u32, @intCast(bc.len));
    try appendInt(list, gpa, u32, 0xFFFF_FFFF);
    try list.appendSlice(gpa, bc);
}

fn appendTag(list: *std.ArrayList(u8), gpa: Allocator, id: *const [4]u8, payload: []const u8) Allocator.Error!void {
    try list.appendSlice(gpa, id);
    try appendInt(list, gpa, u16, @intCast(payload.len));
    try list.appendSlice(gpa, payload);
}

/// Tag whose payload is a NUL-terminated string.
fn appendTagZ(list: *std.ArrayList(u8), gpa: Allocator, id: *const [4]u8, s: []const u8) Allocator.Error!void {
    try list.appendSlice(gpa, id);
    try appendInt(list, gpa, u16, @intCast(s.len + 1));
    try list.appendSlice(gpa, s);
    try list.append(gpa, 0);
}

fn appendInt(list: *std.ArrayList(u8), gpa: Allocator, comptime T: type, value: T) Allocator.Error!void {
    var buf: [@sizeOf(T)]u8 = undefined;
    std.mem.writeInt(T, &buf, value, .little);
    try list.appendSlice(gpa, &buf);
}

test "pack produces a well-formed container" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const gpa = arena.allocator();

    const bc_v = "BC\xC0\xDEvertex-bitcode";
    const bc_f = "BC\xC0\xDEfragment-bitcode!";
    const image = try pack(gpa, &.{
        .{ .name = "vertexShader", .stage = .vertex, .bitcode = bc_v },
        .{ .name = "fragmentShader", .stage = .fragment, .bitcode = bc_f },
    }, "default.metallib", target.default);

    try std.testing.expectEqualStrings("MTLB", image[0..4]);
    const file_size = std.mem.readInt(u64, image[16..24], .little);
    try std.testing.expectEqual(image.len, file_size);
    const bc_off = std.mem.readInt(u64, image[72..80], .little);
    const bc_size = std.mem.readInt(u64, image[80..88], .little);
    // Each blob (wrapper + bitcode) is padded to 16 bytes.
    const blob_v = std.mem.alignForward(usize, wrapper_len + bc_v.len, blob_align);
    const blob_f = std.mem.alignForward(usize, wrapper_len + bc_f.len, blob_align);
    try std.testing.expect(blob_v != wrapper_len + bc_v.len); // the fixture really exercises padding
    try std.testing.expectEqual(blob_v + blob_f, bc_size);
    try std.testing.expectEqual(0, bc_size % blob_align);
    // First blob starts with the darwin wrapper magic, records the unpadded
    // bitcode length, holds the vertex bitcode and is zero-padded.
    try std.testing.expectEqual(bitcode_wrapper_magic, std.mem.readInt(u32, image[bc_off..][0..4], .little));
    try std.testing.expectEqual(bc_v.len, std.mem.readInt(u32, image[bc_off + 12 ..][0..4], .little));
    try std.testing.expectEqualStrings(bc_v, image[bc_off + wrapper_len ..][0..bc_v.len]);
    for (image[bc_off + wrapper_len + bc_v.len .. bc_off + blob_v]) |byte| try std.testing.expectEqual(0, byte);
    // The second blob starts at a 16-byte boundary and its OFFT says so.
    try std.testing.expectEqual(bitcode_wrapper_magic, std.mem.readInt(u32, image[bc_off + blob_v ..][0..4], .little));
    // Function count and first entry name.
    try std.testing.expectEqual(2, std.mem.readInt(u32, image[88..92], .little));
    try std.testing.expectEqualStrings("NAME", image[96..100]);
    try std.testing.expectEqualStrings("vertexShader", image[102..114]);
    // HASH covers the padded first blob; MDSZ is its padded length; the
    // second entry's OFFT bitcode offset is the padded first blob length.
    var digest: [Sha256.digest_length]u8 = undefined;
    Sha256.hash(image[bc_off..][0..blob_v], &digest, .{});
    const hash_pos = std.mem.find(u8, image, "HASH").? + 6;
    try std.testing.expectEqualSlices(u8, &digest, image[hash_pos..][0..32]);
    const mdsz_pos = std.mem.find(u8, image, "MDSZ").? + 6;
    const mdsz = std.mem.readInt(u64, image[mdsz_pos..][0..8], .little);
    try std.testing.expectEqual(blob_v, mdsz);
    try std.testing.expectEqual(0, mdsz % 16);
    // The wrapper's own length field is the unpadded bitcode length, so
    // MDSZ - (wrapper + bitcode) is the zero padding (0..15 bytes).
    const wrapper_bc_len = std.mem.readInt(u32, image[bc_off + 12 ..][0..4], .little);
    try std.testing.expect(mdsz - (wrapper_len + wrapper_bc_len) < 16);
    // The hash is not the hash of the unpadded bytes.
    var unpadded_digest: [Sha256.digest_length]u8 = undefined;
    Sha256.hash(image[bc_off..][0 .. wrapper_len + bc_v.len], &unpadded_digest, .{});
    try std.testing.expect(!std.mem.eql(u8, &unpadded_digest, image[hash_pos..][0..32]));
    const offt2 = std.mem.findPos(u8, image, std.mem.find(u8, image, "fragmentShader").?, "OFFT").? + 6;
    try std.testing.expectEqual(blob_v, std.mem.readInt(u64, image[offt2 + 16 ..][0..8], .little));
    // Reflection list: last four bytes, zero entries, referenced from RLST.
    const rlst_pos = std.mem.find(u8, image, "RLST").? + 6;
    const rlst_off = std.mem.readInt(u64, image[rlst_pos..][0..8], .little);
    const rlst_len = std.mem.readInt(u64, image[rlst_pos + 8 ..][0..8], .little);
    try std.testing.expectEqual(image.len - 4, rlst_off);
    try std.testing.expectEqual(4, rlst_len);
    try std.testing.expectEqual(0, std.mem.readInt(u32, image[image.len - 4 ..][0..4], .little));
}

test "each deployment target writes the header, extension and VERS Apple's compiler writes" {
    // Decoded from `xcrun metal -mmacosx-version-min=<os> -std=<metal>` output
    // (macOS 26.5 SDK, metalfe-32023.883): the u16s at 4, 6, 8 and 10, the
    // target (u16 major, u16 minor) at 12, each function's VERS tag, and
    // whether the header extension carries HDYN. For macOS 15, 14 and 13 Apple
    // writes no HDYN and no dynamic header: the reflection list starts where
    // the bitcode ends.
    const Expect = struct { name: target.Name, header: [4]u16, macos: [2]u16, vers: [4]u16, hdyn: bool };
    const apple = [_]Expect{
        .{ .name = .macos26, .header = .{ 0x8001, 2, 9, 0x8100 }, .macos = .{ 26, 0 }, .vers = .{ 2, 8, 4, 0 }, .hdyn = true },
        .{ .name = .macos15, .header = .{ 0x8001, 2, 8, 0x8100 }, .macos = .{ 15, 0 }, .vers = .{ 2, 7, 3, 2 }, .hdyn = false },
        .{ .name = .macos14, .header = .{ 0x8001, 2, 7, 0x8100 }, .macos = .{ 14, 0 }, .vers = .{ 2, 6, 3, 1 }, .hdyn = false },
        .{ .name = .macos13, .header = .{ 0x8001, 2, 7, 0x8100 }, .macos = .{ 13, 0 }, .vers = .{ 2, 5, 3, 0 }, .hdyn = false },
    };
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    for (apple) |e| {
        const image = try pack(arena.allocator(), &.{.{ .name = "k", .stage = .kernel, .bitcode = "BC\xC0\xDE" }}, "default.metallib", target.get(e.name));
        for (e.header, 0..) |word, i| try std.testing.expectEqual(word, std.mem.readInt(u16, image[4 + 2 * i ..][0..2], .little));
        try std.testing.expectEqual(e.macos[0], std.mem.readInt(u16, image[12..14], .little));
        try std.testing.expectEqual(e.macos[1], std.mem.readInt(u16, image[14..16], .little));
        const vers = std.mem.find(u8, image, "VERS").? + 6;
        for (e.vers, 0..) |v, i| try std.testing.expectEqual(v, std.mem.readInt(u16, image[vers + 2 * i ..][0..2], .little));
        // HDYN and the dynamic header only for format 2.9; otherwise the
        // reflection list follows the bitcode directly.
        try std.testing.expectEqual(e.hdyn, std.mem.find(u8, image, "HDYN") != null);
        try std.testing.expectEqual(e.hdyn, std.mem.find(u8, image, "default.metallib") != null);
        const bc_end = std.mem.readInt(u64, image[72..80], .little) + std.mem.readInt(u64, image[80..88], .little);
        const rlst_off = std.mem.readInt(u64, image[std.mem.find(u8, image, "RLST").? + 6 ..][0..8], .little);
        if (e.hdyn) try std.testing.expect(rlst_off > bc_end) else try std.testing.expectEqual(bc_end, rlst_off);
        try std.testing.expectEqual(image.len - 4, rlst_off);
        // The public metadata starts right after the extension's ENDT.
        const pub_off = std.mem.readInt(u64, image[40..48], .little);
        try std.testing.expectEqualStrings("ENDT", image[pub_off - 4 .. pub_off]);
    }
}

test "readStamp decodes what pack wrote and a minor deployment target, and rejects non-containers" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    for (target.profiles) |p| {
        const image = try pack(arena.allocator(), &.{.{ .name = "k", .stage = .kernel, .bitcode = "BC\xC0\xDE" }}, "default.metallib", p);
        const stamp = readStamp(image).?;
        try std.testing.expectEqual(p.container_minor, stamp.container_minor);
        try std.testing.expectEqual(target.Version{ .major = p.macos_major, .minor = 0 }, stamp.macos);
        try std.testing.expectEqual([4]u16{ 2, p.air_minor, p.lang_major, p.lang_minor }, stamp.vers.?);
    }
    // A 26.1 target, as Apple writes it for -mmacosx-version-min=26.1.
    const image = try pack(arena.allocator(), &.{.{ .name = "k", .stage = .kernel, .bitcode = "BC\xC0\xDE" }}, "default.metallib", target.default);
    image[14] = 1;
    try std.testing.expectEqual(target.Version{ .major = 26, .minor = 1 }, readStamp(image).?.macos);
    // Long enough to pass the length check, so the magic comparison decides.
    const not_mtlb: [header_len]u8 = @splat('x');
    try std.testing.expect(readStamp(&not_mtlb) == null);
    try std.testing.expect(readStamp("MTLB") == null); // truncated header
    // A function-list offset near u64 max must not overflow: the stamp comes
    // back without a VERS instead of crashing the caller.
    const hostile = try arena.allocator().dupe(u8, image);
    std.mem.writeInt(u64, hostile[24..32], std.math.maxInt(u64), .little);
    try std.testing.expect(readStamp(hostile).?.vers == null);
    std.mem.writeInt(u64, hostile[24..32], std.math.maxInt(u64) - 7, .little);
    try std.testing.expect(readStamp(hostile).?.vers == null);
}
