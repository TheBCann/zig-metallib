//! Integer sign extensions through Zig's SPIR-V backend, OpenCL flavour
//! (every OpTypeInt unsigned): tools/spirv/lower.zig's producer-workaround
//! tests and exec test zig_convert_opencl. Here `OpSConvert %u32 %u16` IS a
//! sign extension (the source is i16), so the Zig workaround must not touch
//! it. For each in[i]: out[2i] = the low 16 bits as i16 sign-extended,
//! out[2i+1] = the low byte as i8 sign-extended.
//!
//!   zig build-obj convert_opencl.zig -target spirv64-opencl -fno-llvm -O ReleaseFast -mcpu=generic+int8+int16 -femit-bin=convert-opencl.spv

export fn zig_convert_opencl(in: *addrspace(.global) const [4]u32, out: *addrspace(.global) [8]i32) callconv(.kernel) void {
    for (0..4) |i| {
        const x = in[i];
        const lo: i16 = @bitCast(@as(u16, @truncate(x)));
        const byte: i8 = @bitCast(@as(u8, @truncate(x)));
        out[2 * i] = @as(i32, lo);
        out[2 * i + 1] = @as(i32, byte);
    }
}
