//! Integer widenings through Zig's SPIR-V backend, Vulkan flavour (signed and
//! unsigned OpTypeInt): tools/spirv/lower.zig's producer-workaround tests and
//! exec test zig_convert_vulkan. For each in[i]: out[3i] = the low 16 bits
//! zero-extended (`OpSConvert %i32 %u16`, Zig's bug: must be a zext),
//! out[3i+1] = the low 16 bits as i16 sign-extended (`OpSConvert %i32 %i16`),
//! out[3i+2] = the low byte zero-extended (`OpSConvert %i32 %u8`).
//!
//!   zig build-obj convert_vulkan.zig -target spirv64-vulkan -fno-llvm -O ReleaseFast -mcpu=generic+int8+int16 -femit-bin=convert-vulkan.spv

const In = extern struct { x: [4]u32 };
const Out = extern struct { y: [12]i32 };
const in = @extern(*addrspace(.storage_buffer) const In, .{ .name = "in", .decoration = .{ .descriptor = .{ .set = 0, .binding = 0 } } });
const out = @extern(*addrspace(.storage_buffer) Out, .{ .name = "out", .decoration = .{ .descriptor = .{ .set = 0, .binding = 1 } } });

export fn zig_convert_vulkan() callconv(.{ .spirv_kernel = .{ .x = 1, .y = 1, .z = 1 } }) void {
    for (0..4) |i| {
        const x = in.x[i];
        const lo: u16 = @truncate(x);
        const byte: u8 = @truncate(x);
        out.y[3 * i] = @as(i32, lo);
        out.y[3 * i + 1] = @as(i32, @as(i16, @bitCast(lo)));
        out.y[3 * i + 2] = @as(i32, byte);
    }
}
