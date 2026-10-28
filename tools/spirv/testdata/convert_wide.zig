//! Integer widenings into an i128 through Zig's SPIR-V backend, Vulkan
//! flavour: tools/spirv/lower.zig's producer-workaround tests and exec test
//! zig_convert_wide. Zig converts the source into the u64 limb 0 with
//! buildConvert, which picks OpUConvert for the unsigned limb type even for a
//! signed source (`OpUConvert %u64 %i16`, a zero extension by the
//! specification), then fills the upper limbs from limb 0's top bit
//! (`OpBitcast %i64`, `OpShiftRightArithmetic ... 63`): the workaround must
//! make that OpUConvert a sign extension, and leave the unsigned source and
//! the `@intCast` to u64 zero-extended. For each in[i], out[8i..8i+8] =
//!   0, 1: bits 64-95 and 96-127 of @as(i128, low 16 bits as i16)
//!   2:    bits 64-95 of @as(i128, low 16 bits as u16)
//!   3:    bits 0-31 of @as(i128, low 16 bits as i16)
//!   4, 7: bits 64-95 and 32-63 of @as(i128, in[i] as i32)
//!   5, 6: bits 0-31 and 32-63 of @as(u64, @intCast(low 15 bits as i16))
//!
//!   zig build-obj convert_wide.zig -target spirv64-vulkan -fno-llvm -O ReleaseFast -mcpu=generic+int8+int16 -femit-bin=convert-wide.spv

const In = extern struct { x: [4]u32 };
const Out = extern struct { y: [32]u32 };
const in = @extern(*addrspace(.storage_buffer) const In, .{ .name = "in", .decoration = .{ .descriptor = .{ .set = 0, .binding = 0 } } });
const out = @extern(*addrspace(.storage_buffer) Out, .{ .name = "out", .decoration = .{ .descriptor = .{ .set = 0, .binding = 1 } } });

export fn zig_convert_wide() callconv(.{ .spirv_kernel = .{ .x = 1, .y = 1, .z = 1 } }) void {
    for (0..4) |i| {
        const x = in.x[i];
        const s: i16 = @bitCast(@as(u16, @truncate(x)));
        const u: u16 = @truncate(x);
        const w: u128 = @bitCast(@as(i128, s));
        const v: u128 = @bitCast(@as(i128, u));
        const d: u128 = @bitCast(@as(i128, @as(i32, @bitCast(x))));
        const n: i16 = @bitCast(u & 0x7fff);
        const nu: u64 = @intCast(n);
        out.y[8 * i] = @truncate(w >> 64);
        out.y[8 * i + 1] = @truncate(w >> 96);
        out.y[8 * i + 2] = @truncate(v >> 64);
        out.y[8 * i + 3] = @truncate(w);
        out.y[8 * i + 4] = @truncate(d >> 64);
        out.y[8 * i + 5] = @truncate(nu);
        out.y[8 * i + 6] = @truncate(nu >> 32);
        out.y[8 * i + 7] = @truncate(d >> 32);
    }
}
