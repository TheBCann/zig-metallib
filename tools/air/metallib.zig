//! MTLB (.metallib) container writer.
//!
//! Layout, reverse-engineered from `xcrun metal` output and verified by
//! repacking Apple's own bitcode and loading the result through
//! `newLibraryWithData:` (see tools/metallib_check.zig):
//!
//!   header (88 bytes)
//!   function list      u32 count, then per function: u32 size, tags..., ENDT
//!   header extension   HDYN, RLST, UUID, ENDT
//!   public metadata    per function: u32 8, "ENDT"
//!   private metadata   per function: u32 8, "ENDT"
//!   bitcode            per function: darwin bitcode wrapper + LLVM bitcode,
//!                      zero-padded to a multiple of 16 bytes (metal-objdump
//!                      rejects unpadded blobs with "unknown magic"; the
//!                      wrapper's own size field stays the unpadded length)
//!   dynamic header     NAME "default.metallib\0", ENDT
//!   reflection list    u32 count = 0 (the runtime ignores it; metal-objdump
//!                      refuses to open a library without the count)
//!
//! Every tag is `4-char id, u16 length, payload`. All integers little-endian.

const std = @import("std");
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

/// Serialise `functions` into a complete .metallib image.
pub fn pack(gpa: Allocator, functions: []const Function, library_name: []const u8) Allocator.Error![]u8 {
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
        try appendTag(&entry, gpa, "TYPE", &.{@intFromEnum(f.stage)});
        try appendTag(&entry, gpa, "HASH", &digest);
        var offt: [24]u8 = undefined;
        std.mem.writeInt(u64, offt[0..8], pub_md.items.len, .little);
        std.mem.writeInt(u64, offt[8..16], priv_md.items.len, .little);
        std.mem.writeInt(u64, offt[16..24], blob_off, .little);
        try appendTag(&entry, gpa, "OFFT", &offt);
        // air.version 2.8, Metal language 4.0 — matches metalfe-32023.883 output.
        try appendTag(&entry, gpa, "VERS", &.{ 2, 0, 8, 0, 4, 0, 0, 0 });
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

    var dyn: std.ArrayList(u8) = .empty;
    try appendTagZ(&dyn, gpa, "NAME", library_name);
    try dyn.appendSlice(gpa, "ENDT");

    const ext_len = (4 + 2 + 16) * 3 + 4; // HDYN RLST UUID ENDT
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

    // Header. The u16/u32 fields are copied from Apple's output; their exact
    // meaning is undocumented (0x8001 looks like platform+flags, 2.9 a format
    // version, 26 the macOS SDK major).
    try out.appendSlice(gpa, "MTLB");
    try appendInt(&out, gpa, u16, 0x8001);
    try appendInt(&out, gpa, u16, 2);
    try appendInt(&out, gpa, u16, 9);
    try appendInt(&out, gpa, u16, 0x8100);
    try appendInt(&out, gpa, u32, 26);
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
    std.mem.writeInt(u64, off_size[0..8], dyn_off, .little);
    std.mem.writeInt(u64, off_size[8..16], dyn.items.len, .little);
    try appendTag(&out, gpa, "HDYN", &off_size);
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
    }, "default.metallib");

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
