//! Deployment-target profiles for the emitted `.metallib`.
//!
//! A Metal library is stamped for the oldest macOS it must load on, much like
//! `-mmacosx-version-min` for a binary. Measured on macOS 26.3: libraries that
//! target 26.0, 26.4 and 26.9 load, a macOS 27.0 target is rejected
//! ("Unsupported target triple"). So Metal refuses a newer macOS *major* than
//! the running one and accepts newer minors of the same major, and it decides
//! from the bitcode's target triple, not the container header: a 26 triple
//! with its header forged to 27 loads, a 27 triple forged to 26 does not.
//!
//! The profiles below were measured by compiling one probe shader with
//! Apple's compiler (`xcrun metal -mmacosx-version-min=<os> -std=<metal>`,
//! macOS 26.5 SDK, metalfe-32023.883, run on macOS 26.3) for each target and
//! decoding the result. These values differ between the targets (not each of
//! them at every step: header @8 is 7 for both macOS 14 and 13). The Metal
//! language version is the highest each target supports, which is what the
//! profile emits; it is a choice, not dictated by the target (a macOS 26 build
//! with -std=metal3.2 carries AIR 2.8 / Metal 3.2):
//!
//!   target    triple                          header @8  header @12  AIR   Metal
//!   macos26   air64_v28-apple-macosx26.0.0        9       26 (.0)    2.8   4.0
//!   macos15   air64_v27-apple-macosx15.0.0        8       15 (.0)    2.7   3.2
//!   macos14   air64_v26-apple-macosx14.0.0        7       14 (.0)    2.6   3.1
//!   macos13   air64_v25-apple-macosx13.0.0        7       13 (.0)    2.5   3.0
//!
//! (header @12 is the target's macOS major as a u16, followed by its minor as a
//! u16 at @14). Two structural differences come on top:
//!   * only macOS 26 (container format 2.9) has the `HDYN` tag in the header
//!     extension and the dynamic-header section after the bitcode; the older
//!     formats have an extension of just `RLST` and `UUID`, and the reflection
//!     list follows the bitcode directly (`dynamic_header`);
//!   * macOS 13 output has no `frame-pointer` module flag (`frame_pointer_flag`),
//!     writes `undef` where the newer targets write `poison`, drops `noundef`
//!     parameter attributes and uses an older function-attribute set. Only the
//!     flag is reproduced: this project emits no attribute groups, and its
//!     `poison` is one more reason the macos13 profile is unverified.
//! No profile reproduces Apple's `SDK Version` module flag or its real
//! reflection list (this project writes an empty one, see metallib.zig); the
//! macOS 26 library passes every check without either.
//! The datalayout, the binding limits, the compile options and the signatures
//! of the two intrinsics the probe calls (`air.simd_sum.f32`,
//! `air.sample_texture_2d.v4f32`) were identical across all four; after
//! normalizing the fields above, the 26 disassembly differs from the 15 and 14
//! ones only in function offsets. All four Apple-built probe libraries load and
//! build pipelines on macOS 26.
//!
//! One difference is NOT captured by the profile: Apple's compiler emits
//! typed-pointer bitcode for every target (including macOS 26), while this
//! project's assembler emits opaque pointers, the only kind
//! std.zig.llvm.Builder can write. For AIR 2.8 that is fine: macOS 26 reads it
//! directly and every check passes. For the older profiles it is not. macOS 26
//! loads an older-AIR library through an upgrader, and pipeline creation fails
//! with "Failed to upgrade function bitcode" on our output with the macOS 15,
//! 14 and 13 stamps, while Apple's typed-pointer libraries for the same targets
//! upgrade and build pipelines. Apple's own compiler reproduces the failure
//! when told to emit opaque pointers for those targets, so the cause is opaque
//! pointers under a pre-2.8 AIR version, not the stamp values. Hence
//! `verified`: only macOS 26 is. Whether the macOS 13-15 runtimes read opaque
//! pointers natively (without the upgrader) is still unknown and needs one of
//! those systems.
//!
//! This file imports only `std`, so build.zig can import it to offer the
//! profile names as a build option.

const std = @import("std");

pub const Name = enum { macos26, macos15, macos14, macos13 };

pub const Profile = struct {
    name: Name,
    /// macOS major version the library targets: the triple's `macosx<N>.0.0`
    /// and the container header's u16 at offset 12 (the minor at 14 is 0).
    macos_major: u16,
    /// AIR version `2.<air_minor>`; the triple's `air64_v<20 + air_minor>`,
    /// `!air.version` and the first half of each function's `VERS` tag.
    air_minor: u16,
    /// Metal Shading Language version: `!air.language_version` and the second
    /// half of `VERS`.
    lang_major: u16,
    lang_minor: u16,
    /// Container header u16 at offset 8 (the format version's minor half).
    container_minor: u16,
    triple: []const u8,
    /// Whether the header extension carries `HDYN` and a dynamic-header section
    /// (`NAME <library>` + `ENDT`) follows the bitcode: format 2.9 only.
    dynamic_header: bool = false,
    /// Whether `llvm.module.flags` carries `frame-pointer 2` (not on macOS 13).
    frame_pointer_flag: bool = true,
    /// Whether libraries built for this profile pass `zig build check`. The
    /// older profiles reproduce Apple's layout, but their bitcode needs typed
    /// pointers (see the file comment), so air-splice refuses them unless
    /// asked to build an unverified library.
    verified: bool = false,
};

/// Identical for every profile (see the file comment).
pub const datalayout =
    "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64" ++
    "-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128" ++
    "-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32";

pub const profiles = [_]Profile{
    .{ .name = .macos26, .macos_major = 26, .air_minor = 8, .lang_major = 4, .lang_minor = 0, .container_minor = 9, .triple = "air64_v28-apple-macosx26.0.0", .dynamic_header = true, .verified = true },
    .{ .name = .macos15, .macos_major = 15, .air_minor = 7, .lang_major = 3, .lang_minor = 2, .container_minor = 8, .triple = "air64_v27-apple-macosx15.0.0" },
    .{ .name = .macos14, .macos_major = 14, .air_minor = 6, .lang_major = 3, .lang_minor = 1, .container_minor = 7, .triple = "air64_v26-apple-macosx14.0.0" },
    .{ .name = .macos13, .macos_major = 13, .air_minor = 5, .lang_major = 3, .lang_minor = 0, .container_minor = 7, .triple = "air64_v25-apple-macosx13.0.0", .frame_pointer_flag = false },
};

/// The profile every build uses unless `-Dmetal-target` says otherwise.
pub const default = get(.macos26);

pub fn get(name: Name) Profile {
    for (profiles) |p| if (p.name == name) return p;
    unreachable;
}

/// `macos15` -> the macOS 15 profile; null for an unknown name.
pub fn fromName(text: []const u8) ?Profile {
    const name = std.meta.stringToEnum(Name, text) orelse return null;
    return get(name);
}

/// Why `p` cannot be trusted, or null when it is verified. air-splice prints
/// this and stops unless `--allow-unverified` is given.
pub fn unverifiedReason(p: Profile) ?[]const u8 {
    if (p.verified) return null;
    return "its container matches Apple's output, but this project writes opaque-pointer bitcode and the macOS 26 runtime " ++
        "refuses opaque pointers under a pre-2.8 AIR version (\"Failed to upgrade function bitcode\"); Apple's libraries " ++
        "for this target use typed pointers. Whether the target's own macOS accepts it is untested";
}

/// The profile a container header describes, from its u16 at offset 8 and the
/// macOS major (u16) at offset 12; null when no known profile matches. Apple
/// writes the same header for every Metal language version of one target, so
/// this identifies the target, not the library's `VERS`.
pub fn fromHeader(container_minor: u16, macos_major: u16) ?Profile {
    for (profiles) |p| {
        if (p.container_minor == container_minor and p.macos_major == macos_major) return p;
    }
    return null;
}

/// A macOS version, as the container header stores it (u16 major at offset 12,
/// u16 minor at 14) and as `kern.osproductversion` reports it ("26.3").
pub const Version = struct {
    major: u16,
    minor: u16 = 0,

    /// "26.3" -> 26.3, "15" -> 15.0, "26.3.1" -> 26.3; null unless the text
    /// starts with a number.
    pub fn parse(text: []const u8) ?Version {
        var parts = std.mem.splitScalar(u8, text, '.');
        const major = std.fmt.parseInt(u16, parts.next() orelse return null, 10) catch return null;
        const minor = if (parts.next()) |m| std.fmt.parseInt(u16, m, 10) catch return null else 0;
        return .{ .major = major, .minor = minor };
    }

    /// Whether Metal on macOS `host` accepts a library that targets `self`:
    /// only a newer major is refused (see the file comment for the
    /// measurement). Metal reads the bitcode triple, not the header, but the
    /// two carry the same version in every library Apple's compiler or this
    /// project writes.
    pub fn acceptedOn(self: Version, host: Version) bool {
        return self.major <= host.major;
    }
};

test "every triple follows air64_v<20 + air minor>-apple-macosx<major>.0.0" {
    var buf: [64]u8 = undefined;
    for (profiles) |p| {
        const want = try std.fmt.bufPrint(&buf, "air64_v{d}-apple-macosx{d}.0.0", .{ 20 + p.air_minor, p.macos_major });
        try std.testing.expectEqualStrings(want, p.triple);
    }
}

test "the default profile is the one the project was validated on" {
    try std.testing.expectEqual(Name.macos26, default.name);
    try std.testing.expectEqualStrings("air64_v28-apple-macosx26.0.0", default.triple);
    try std.testing.expectEqual(@as(u16, 9), default.container_minor);
    try std.testing.expect(default.dynamic_header);
    try std.testing.expect(default.frame_pointer_flag);
}

test "only the default profile is verified; the others explain why not" {
    for (profiles) |p| {
        try std.testing.expectEqual(p.name == .macos26, p.verified);
        try std.testing.expectEqual(p.verified, unverifiedReason(p) == null);
    }
}

test "structural differences follow Apple's output: HDYN only for macOS 26, no frame-pointer flag for macOS 13" {
    for (profiles) |p| {
        try std.testing.expectEqual(p.name == .macos26, p.dynamic_header);
        try std.testing.expectEqual(p.name != .macos13, p.frame_pointer_flag);
    }
}

test "every Name has exactly one profile" {
    const names = std.enums.values(Name);
    try std.testing.expectEqual(names.len, profiles.len);
    for (names) |name| {
        var count: usize = 0;
        for (profiles) |p| {
            if (p.name == name) count += 1;
        }
        try std.testing.expectEqual(@as(usize, 1), count);
        try std.testing.expectEqual(name, get(name).name);
    }
}

test "names, lookups and header decoding round-trip" {
    for (profiles) |p| {
        try std.testing.expectEqual(p, fromName(@tagName(p.name)).?);
        try std.testing.expectEqual(p, fromHeader(p.container_minor, p.macos_major).?);
    }
    try std.testing.expect(fromName("macos99") == null);
    try std.testing.expect(fromName("") == null);
    try std.testing.expect(fromHeader(9, 15) == null);
}

test "Version parses kern.osproductversion and orders deployment targets" {
    try std.testing.expectEqual(Version{ .major = 26, .minor = 3 }, Version.parse("26.3").?);
    try std.testing.expectEqual(Version{ .major = 15, .minor = 0 }, Version.parse("15").?);
    try std.testing.expectEqual(Version{ .major = 26, .minor = 3 }, Version.parse("26.3.1").?);
    try std.testing.expect(Version.parse("") == null);
    try std.testing.expect(Version.parse("macOS") == null);
    try std.testing.expect(Version.parse("26.x") == null);
    // Measured on macOS 26.3 (file comment): 26.0, 26.4 and 26.9 targets load,
    // a 27.0 target is rejected.
    const host = Version{ .major = 26, .minor = 3 };
    try std.testing.expect((Version{ .major = 26 }).acceptedOn(host));
    try std.testing.expect((Version{ .major = 26, .minor = 4 }).acceptedOn(host));
    try std.testing.expect((Version{ .major = 26, .minor = 9 }).acceptedOn(host));
    try std.testing.expect((Version{ .major = 15, .minor = 4 }).acceptedOn(host));
    try std.testing.expect(!(Version{ .major = 27 }).acceptedOn(host));
    try std.testing.expect(!(Version{ .major = 26 }).acceptedOn(.{ .major = 15, .minor = 6 }));
}
