//! Whole-module execution tests: small SPIR-V compute modules with fixed
//! inputs and the exact words every output buffer must hold afterwards.
//! tools/spirv_probe_run.zig lowers each one, builds its pipeline on Metal,
//! binds the buffers at the indices the manifest gives and dispatches
//! `threads` threads; `zig build test` checks that each lowers, assembles
//! and binds (no GPU). They pin down lowering facts that a one-instruction
//! probe cannot show: interface forms (BufferBlock SSBOs, OpenCL
//! program-scope constant tables), memory layout (vec3 members of
//! std430/scalar structs, 12-byte vec3 arrays: a 16-byte store would
//! clobber the neighbour), control-flow edges (a phi under a switch with
//! two cases on one target), OpSelect of an aggregate loaded from memory
//! (the shape that crashes MTLCompilerService as one `select`), saturating
//! conversions at their edges, NonSemantic debug instructions, packed
//! structs inside naturally laid out ones (LLVM aligns a packed struct to
//! 1), OpenCL CPacked structs, a WorkgroupSize specialization constant
//! overriding LocalSize (dispatched with the manifest's size), and Zig's own
//! integer widenings in both flavours (the producer workaround must
//! zero-extend Zig's `OpSConvert %i32 %u16` and keep the OpenCL flavour's
//! real sign extensions, and sign-extend Zig's `OpUConvert %u64 %i16` into
//! an i128's sign-filled limb), an interface decorated entirely through
//! decoration groups (reader.zig's expansion: Bindings, DescriptorSet,
//! BufferBlock, ArrayStride, a member Offset), results the specifications
//! define at the edges (OpBitField* at Offset = width with Count 0 and
//! constant operands, which LLVM folds; OpenCL powr, normalize, length and
//! distance), a packed member's alignment carried by a pointer through
//! OpSelect, a helper parameter, a loop phi and a Function variable, and a
//! barrier in a loop whose uniform bound glslang routes through Function
//! variables (a helper argument passed by pointer, a local struct or
//! vector read through OpAccessChain: the uniformity analysis follows
//! them).

const std = @import("std");
const Allocator = std.mem.Allocator;

pub const Buffer = struct {
    /// `[[buffer(index)]]`; must match the manifest.
    index: u32,
    /// Initial contents.
    words: []const u32,
    /// The contents after the dispatch, when checked.
    expect: ?[]const u32 = null,
    /// The word `expect[0]` is compared with (the words before it, and
    /// after the expected ones, are not checked).
    expect_at: usize = 0,
    /// Not bound to the kernel (no manifest buffer has `index`, which only
    /// names it here): made resident with useResource and reached through
    /// an address another buffer holds (GL_EXT_buffer_reference).
    unbound: bool = false,
    /// Words that receive another buffer's gpuAddress before the dispatch
    /// (64 bits, low word first).
    addresses: []const Address = &.{},
};

pub const Address = struct {
    /// The first of the two words.
    word: usize,
    /// The `index` of the buffer whose address is written.
    of: u32,
};

pub const Test = struct {
    /// The module's (only) entry point.
    name: []const u8,
    text: []const u8,
    threads: u32,
    buffers: []const Buffer,
    /// Dispatch threadgroups of the manifest's `local_size` (else one group
    /// of `threads`): the test checks the size the manifest tells a host.
    group_from_manifest: bool = false,
    /// Checks a result a fast-math library may change (a signed zero), so
    /// it runs only without `--fast-math`.
    strict_only: bool = false,
};

fn f32Bits(x: f32) u32 {
    return @bitCast(x);
}

fn seq(a: Allocator, n: usize, comptime f: fn (usize) u32) ![]const u32 {
    const out = try a.alloc(u32, n);
    for (out, 0..) |*w, i| w.* = f(i);
    return out;
}

fn zeros(a: Allocator, n: usize) ![]const u32 {
    const out = try a.alloc(u32, n);
    @memset(out, 0);
    return out;
}

/// The vec3-layout test's data: `Rec { vec3 v @0; float s @12 }` (16 bytes)
/// with v = (i, i+1, i+2), s = 10i, and the expected `(2v, s + v.y)`.
fn recIn(i: usize) [4]u32 {
    const x: f32 = @floatFromInt(i);
    return .{ f32Bits(x), f32Bits(x + 1), f32Bits(x + 2), f32Bits(10 * x) };
}

fn recOut(i: usize) [4]u32 {
    const x: f32 = @floatFromInt(i);
    return .{ f32Bits(2 * x), f32Bits(2 * x + 2), f32Bits(2 * x + 4), f32Bits(10 * x + x + 1) };
}

pub fn all(a: Allocator) ![]const Test {
    var list: std.ArrayList(Test) = .empty;

    // A SPIR-V 1.0 SSBO: Uniform storage class, BufferBlock struct (glslang
    // for Vulkan 1.0). Device memory, written; the UBO beside it shares the
    // `Uniform` element pointer type and stays constant memory.
    try list.append(a, .{ .name = "bufferblock_ssbo", .threads = 16, .text =
        \\OpCapability Shader
        \\OpMemoryModel Logical GLSL450
        \\OpEntryPoint GLCompute %k "bufferblock_ssbo" %gid
        \\OpExecutionMode %k LocalSize 16 1 1
        \\OpDecorate %gid BuiltIn GlobalInvocationId
        \\OpDecorate %rt ArrayStride 4
        \\OpDecorate %Buf BufferBlock
        \\OpMemberDecorate %Buf 0 Offset 0
        \\OpDecorate %U Block
        \\OpMemberDecorate %U 0 Offset 0
        \\OpDecorate %in DescriptorSet 0
        \\OpDecorate %in Binding 0
        \\OpDecorate %out DescriptorSet 0
        \\OpDecorate %out Binding 1
        \\OpDecorate %ubo DescriptorSet 0
        \\OpDecorate %ubo Binding 2
        \\%void = OpTypeVoid
        \\%fn = OpTypeFunction %void
        \\%u32 = OpTypeInt 32 0
        \\%v3u32 = OpTypeVector %u32 3
        \\%c0 = OpConstant %u32 0
        \\%c5 = OpConstant %u32 5
        \\%rt = OpTypeRuntimeArray %u32
        \\%Buf = OpTypeStruct %rt
        \\%U = OpTypeStruct %u32
        \\%pBuf = OpTypePointer Uniform %Buf
        \\%pU = OpTypePointer Uniform %U
        \\%pu = OpTypePointer Uniform %u32
        \\%pin = OpTypePointer Input %v3u32
        \\%gid = OpVariable %pin Input
        \\%in = OpVariable %pBuf Uniform
        \\%out = OpVariable %pBuf Uniform
        \\%ubo = OpVariable %pU Uniform
        \\%k = OpFunction %void None %fn
        \\%e = OpLabel
        \\%gv = OpLoad %v3u32 %gid
        \\%g = OpCompositeExtract %u32 %gv 0
        \\%pi = OpAccessChain %pu %in %c0 %g
        \\%x = OpLoad %u32 %pi
        \\%pk = OpAccessChain %pu %ubo %c0
        \\%kk = OpLoad %u32 %pk
        \\%y0 = OpIAdd %u32 %x %c5
        \\%y = OpIAdd %u32 %y0 %kk
        \\%po = OpAccessChain %pu %out %c0 %g
        \\OpStore %po %y
        \\OpReturn
        \\OpFunctionEnd
    , .buffers = try a.dupe(Buffer, &.{
        .{ .index = 0, .words = try seq(a, 16, struct {
            fn f(i: usize) u32 {
                return @intCast(i + 1);
            }
        }.f) },
        .{ .index = 1, .words = try zeros(a, 16), .expect = try seq(a, 16, struct {
            fn f(i: usize) u32 {
                return @intCast(i + 1 + 5 + 100);
            }
        }.f) },
        .{ .index = 2, .words = &.{100} },
    }) });

    // OpenCL program-scope `__constant uint tbl[4] = {11, 22, 33, 44};`.
    try list.append(a, .{ .name = "constant_table", .threads = 8, .text =
        \\OpCapability Kernel
        \\OpCapability Addresses
        \\OpCapability Int64
        \\OpMemoryModel Physical64 OpenCL
        \\OpEntryPoint Kernel %k "constant_table" %gid
        \\OpDecorate %gid BuiltIn GlobalInvocationId
        \\OpDecorate %tbl Constant
        \\%void = OpTypeVoid
        \\%u32 = OpTypeInt 32 0
        \\%u64 = OpTypeInt 64 0
        \\%v3u64 = OpTypeVector %u64 3
        \\%c3 = OpConstant %u32 3
        \\%c4 = OpConstant %u32 4
        \\%c11 = OpConstant %u32 11
        \\%c22 = OpConstant %u32 22
        \\%c33 = OpConstant %u32 33
        \\%c44 = OpConstant %u32 44
        \\%arr = OpTypeArray %u32 %c4
        \\%init = OpConstantComposite %arr %c11 %c22 %c33 %c44
        \\%ptbl = OpTypePointer UniformConstant %arr
        \\%puc = OpTypePointer UniformConstant %u32
        \\%tbl = OpVariable %ptbl UniformConstant %init
        \\%pcw = OpTypePointer CrossWorkgroup %u32
        \\%pin = OpTypePointer Input %v3u64
        \\%gid = OpVariable %pin Input
        \\%fn = OpTypeFunction %void %pcw %pcw
        \\%k = OpFunction %void None %fn
        \\%dst = OpFunctionParameter %pcw
        \\%src = OpFunctionParameter %pcw
        \\%e = OpLabel
        \\%gv = OpLoad %v3u64 %gid
        \\%g = OpCompositeExtract %u64 %gv 0
        \\%ps = OpInBoundsPtrAccessChain %pcw %src %g
        \\%x = OpLoad %u32 %ps Aligned 4
        \\%i = OpBitwiseAnd %u32 %x %c3
        \\%pt = OpInBoundsAccessChain %puc %tbl %i
        \\%v = OpLoad %u32 %pt Aligned 4
        \\%pd = OpInBoundsPtrAccessChain %pcw %dst %g
        \\OpStore %pd %v Aligned 4
        \\OpReturn
        \\OpFunctionEnd
    , .buffers = try a.dupe(Buffer, &.{
        .{ .index = 0, .words = try zeros(a, 8), .expect = &.{ 11, 22, 33, 44, 11, 22, 33, 44 } },
        .{ .index = 1, .words = &.{ 0, 1, 2, 3, 4, 5, 6, 7 } },
    }) });

    // convert_uchar_sat / convert_char_sat / convert_int_sat(float) /
    // convert_int_rte(float) at their edges (OpenCL flavour).
    try list.append(a, .{
        .name = "saturated_conversions",
        .threads = 9,
        .text =
        \\OpCapability Kernel
        \\OpCapability Addresses
        \\OpCapability Int64
        \\OpCapability Int8
        \\OpMemoryModel Physical64 OpenCL
        \\OpEntryPoint Kernel %k "saturated_conversions" %gid
        \\OpDecorate %gid BuiltIn GlobalInvocationId
        \\OpDecorate %us SaturatedConversion
        \\OpDecorate %ss SaturatedConversion
        \\OpDecorate %fs SaturatedConversion
        \\OpDecorate %fr SaturatedConversion
        \\OpDecorate %fr FPRoundingMode RTE
        \\%void = OpTypeVoid
        \\%u8 = OpTypeInt 8 0
        \\%u32 = OpTypeInt 32 0
        \\%u64 = OpTypeInt 64 0
        \\%f32 = OpTypeFloat 32
        \\%v3u64 = OpTypeVector %u64 3
        \\%c4 = OpConstant %u64 4
        \\%c1 = OpConstant %u64 1
        \\%c2 = OpConstant %u64 2
        \\%c3 = OpConstant %u64 3
        \\%pcw = OpTypePointer CrossWorkgroup %u32
        \\%pin = OpTypePointer Input %v3u64
        \\%gid = OpVariable %pin Input
        \\%fn = OpTypeFunction %void %pcw %pcw
        \\%k = OpFunction %void None %fn
        \\%dst = OpFunctionParameter %pcw
        \\%src = OpFunctionParameter %pcw
        \\%e = OpLabel
        \\%gv = OpLoad %v3u64 %gid
        \\%g = OpCompositeExtract %u64 %gv 0
        \\%ps = OpInBoundsPtrAccessChain %pcw %src %g
        \\%x = OpLoad %u32 %ps Aligned 4
        \\%xf = OpBitcast %f32 %x
        \\%us = OpUConvert %u8 %x
        \\%usw = OpUConvert %u32 %us
        \\%ss = OpSConvert %u8 %x
        \\%ssw = OpSConvert %u32 %ss
        \\%fs = OpConvertFToS %u32 %xf
        \\%fr = OpConvertFToS %u32 %xf
        \\%base = OpIMul %u64 %g %c4
        \\%p0 = OpInBoundsPtrAccessChain %pcw %dst %base
        \\OpStore %p0 %usw Aligned 4
        \\%o1 = OpIAdd %u64 %base %c1
        \\%p1 = OpInBoundsPtrAccessChain %pcw %dst %o1
        \\OpStore %p1 %ssw Aligned 4
        \\%o2 = OpIAdd %u64 %base %c2
        \\%p2 = OpInBoundsPtrAccessChain %pcw %dst %o2
        \\OpStore %p2 %fs Aligned 4
        \\%o3 = OpIAdd %u64 %base %c3
        \\%p3 = OpInBoundsPtrAccessChain %pcw %dst %o3
        \\OpStore %p3 %fr Aligned 4
        \\OpReturn
        \\OpFunctionEnd
        ,
        .buffers = try a.dupe(Buffer, &.{
            // Per input: convert_uchar_sat(x), convert_char_sat(x) widened,
            // convert_int_sat(as_float(x)), convert_int_sat_rte(as_float(x)).
            // 300 and 7 are subnormal floats (0); 0xFFFFFF80 and 0x7FC00000
            // are NaNs (0); 3e9 / -3e9 clamp; 2.5 / 3.5 / -2.5 truncate or
            // round half to even.
            .{
                .index = 0,
                .words = try zeros(a, 36),
                .expect = &.{
                    255, 127,        0,          0,
                    255, 0xFFFFFF80, 0,          0,
                    255, 127,        0x7FFFFFFF, 0x7FFFFFFF,
                    255, 0xFFFFFF80, 0x80000000, 0x80000000,
                    255, 127,        0,          0,
                    255, 127,        2,          2,
                    255, 127,        3,          4,
                    255, 0xFFFFFF80, 0xFFFFFFFE, 0xFFFFFFFE,
                    7,   7,          0,          0,
                },
            },
            .{ .index = 1, .words = &.{ 300, 0xFFFFFF80, 0x4F32D05E, 0xCF32D05E, 0x7FC00000, 0x40200000, 0x40600000, 0xC0200000, 7 } },
        }),
    });

    // OpSelect of a struct loaded from a Function array (the shape whose
    // single aggregate `select` crashes MTLCompilerService).
    try list.append(a, .{ .name = "composite_select", .threads = 8, .text =
        \\OpCapability Shader
        \\OpMemoryModel Logical GLSL450
        \\OpEntryPoint GLCompute %k "composite_select" %gid
        \\OpExecutionMode %k LocalSize 8 1 1
        \\OpDecorate %gid BuiltIn GlobalInvocationId
        \\OpDecorate %rt ArrayStride 4
        \\OpDecorate %Buf Block
        \\OpMemberDecorate %Buf 0 Offset 0
        \\OpDecorate %in DescriptorSet 0
        \\OpDecorate %in Binding 0
        \\OpDecorate %in NonWritable
        \\OpDecorate %out DescriptorSet 0
        \\OpDecorate %out Binding 1
        \\%void = OpTypeVoid
        \\%fn = OpTypeFunction %void
        \\%bool = OpTypeBool
        \\%u32 = OpTypeInt 32 0
        \\%v3u32 = OpTypeVector %u32 3
        \\%v4u32 = OpTypeVector %u32 4
        \\%c0 = OpConstant %u32 0
        \\%c1 = OpConstant %u32 1
        \\%c2 = OpConstant %u32 2
        \\%c3 = OpConstant %u32 3
        \\%c4 = OpConstant %u32 4
        \\%c7 = OpConstant %u32 7
        \\%cv = OpConstantComposite %v4u32 %c1 %c2 %c3 %c4
        \\%S = OpTypeStruct %u32 %v4u32
        \\%arr2 = OpTypeArray %S %c2
        \\%cS = OpConstantComposite %S %c7 %cv
        \\%nS = OpConstantNull %S
        \\%init = OpConstantComposite %arr2 %cS %nS
        \\%pfarr = OpTypePointer Function %arr2
        \\%pfS = OpTypePointer Function %S
        \\%rt = OpTypeRuntimeArray %u32
        \\%Buf = OpTypeStruct %rt
        \\%pBuf = OpTypePointer StorageBuffer %Buf
        \\%pu = OpTypePointer StorageBuffer %u32
        \\%pin = OpTypePointer Input %v3u32
        \\%gid = OpVariable %pin Input
        \\%in = OpVariable %pBuf StorageBuffer
        \\%out = OpVariable %pBuf StorageBuffer
        \\%k = OpFunction %void None %fn
        \\%e = OpLabel
        \\%var = OpVariable %pfarr Function %init
        \\%gv = OpLoad %v3u32 %gid
        \\%g = OpCompositeExtract %u32 %gv 0
        \\%pi = OpAccessChain %pu %in %c0 %g
        \\%x = OpLoad %u32 %pi
        \\%sidx = OpBitwiseAnd %u32 %x %c1
        \\%cond = OpULessThan %bool %x %c4
        \\%sp = OpAccessChain %pfS %var %sidx
        \\%sv = OpLoad %S %sp
        \\%ssel = OpSelect %S %cond %sv %nS
        \\%r = OpCompositeExtract %u32 %ssel 1 0
        \\%r2 = OpCompositeExtract %u32 %ssel 0
        \\%sum = OpIAdd %u32 %r %r2
        \\%po = OpAccessChain %pu %out %c0 %g
        \\OpStore %po %sum
        \\OpReturn
        \\OpFunctionEnd
    , .buffers = try a.dupe(Buffer, &.{
        .{ .index = 0, .words = &.{ 0, 1, 2, 3, 4, 5, 6, 7 } },
        .{ .index = 1, .words = try zeros(a, 8), .expect = &.{ 8, 0, 8, 0, 0, 0, 0, 0 } },
    }) });

    // A phi whose parent reaches it over two switch cases, and one under a
    // conditional branch with equal targets (LLVM: one entry per edge).
    try list.append(a, .{ .name = "phi_edges", .threads = 8, .text =
        \\OpCapability Shader
        \\OpMemoryModel Logical GLSL450
        \\OpEntryPoint GLCompute %k "phi_edges" %gid
        \\OpExecutionMode %k LocalSize 8 1 1
        \\OpDecorate %gid BuiltIn GlobalInvocationId
        \\OpDecorate %rt ArrayStride 4
        \\OpDecorate %Buf Block
        \\OpMemberDecorate %Buf 0 Offset 0
        \\OpDecorate %in DescriptorSet 0
        \\OpDecorate %in Binding 0
        \\OpDecorate %out DescriptorSet 0
        \\OpDecorate %out Binding 1
        \\%void = OpTypeVoid
        \\%fn = OpTypeFunction %void
        \\%bool = OpTypeBool
        \\%u32 = OpTypeInt 32 0
        \\%v3u32 = OpTypeVector %u32 3
        \\%c0 = OpConstant %u32 0
        \\%c5 = OpConstant %u32 5
        \\%c100 = OpConstant %u32 100
        \\%c200 = OpConstant %u32 200
        \\%rt = OpTypeRuntimeArray %u32
        \\%Buf = OpTypeStruct %rt
        \\%pBuf = OpTypePointer StorageBuffer %Buf
        \\%pu = OpTypePointer StorageBuffer %u32
        \\%pin = OpTypePointer Input %v3u32
        \\%gid = OpVariable %pin Input
        \\%in = OpVariable %pBuf StorageBuffer
        \\%out = OpVariable %pBuf StorageBuffer
        \\%k = OpFunction %void None %fn
        \\%e = OpLabel
        \\%gv = OpLoad %v3u32 %gid
        \\%g = OpCompositeExtract %u32 %gv 0
        \\%pi = OpAccessChain %pu %in %c0 %g
        \\%x = OpLoad %u32 %pi
        \\OpSelectionMerge %m None
        \\OpSwitch %x %other 1 %m 2 %m
        \\%other = OpLabel
        \\OpBranch %m
        \\%m = OpLabel
        \\%r = OpPhi %u32 %c100 %e %c200 %other
        \\%t = OpULessThan %bool %x %c5
        \\OpSelectionMerge %m2 None
        \\OpBranchConditional %t %m2 %m2
        \\%m2 = OpLabel
        \\%r2 = OpPhi %u32 %r %m
        \\%po = OpAccessChain %pu %out %c0 %g
        \\OpStore %po %r2
        \\OpReturn
        \\OpFunctionEnd
    , .buffers = try a.dupe(Buffer, &.{
        .{ .index = 0, .words = &.{ 0, 1, 2, 3, 4, 5, 6, 7 } },
        .{ .index = 1, .words = try zeros(a, 8), .expect = &.{ 200, 100, 100, 200, 200, 200, 200, 200 } },
    }) });

    // std430 `struct Rec { vec3 v; float s; }` (s at Offset 12) read and
    // written whole and member by member, and a scalar-layout `vec3[]`
    // (ArrayStride 12) written by neighbouring threads.
    var rin: std.ArrayList(u32) = .empty;
    var rout: std.ArrayList(u32) = .empty;
    var v3in: std.ArrayList(u32) = .empty;
    var v3out: std.ArrayList(u32) = .empty;
    for (0..16) |i| {
        try rin.appendSlice(a, &recIn(i));
        try rout.appendSlice(a, &recOut(i));
        const x: f32 = @floatFromInt(3 * i);
        try v3in.appendSlice(a, &.{ f32Bits(x), f32Bits(x + 1), f32Bits(x + 2) });
        try v3out.appendSlice(a, &.{ f32Bits(2 * x), f32Bits(2 * x + 2), f32Bits(2 * x + 4) });
    }
    try list.append(a, .{ .name = "vec3_layouts", .threads = 16, .text =
        \\OpCapability Shader
        \\OpMemoryModel Logical GLSL450
        \\OpEntryPoint GLCompute %k "vec3_layouts" %gid
        \\OpExecutionMode %k LocalSize 16 1 1
        \\OpDecorate %gid BuiltIn GlobalInvocationId
        \\OpMemberDecorate %Rec 0 Offset 0
        \\OpMemberDecorate %Rec 1 Offset 12
        \\OpDecorate %rtR ArrayStride 16
        \\OpDecorate %BufR Block
        \\OpMemberDecorate %BufR 0 Offset 0
        \\OpDecorate %rt3 ArrayStride 12
        \\OpDecorate %Buf3 Block
        \\OpMemberDecorate %Buf3 0 Offset 0
        \\OpDecorate %in DescriptorSet 0
        \\OpDecorate %in Binding 0
        \\OpDecorate %out DescriptorSet 0
        \\OpDecorate %out Binding 1
        \\OpDecorate %out2 DescriptorSet 0
        \\OpDecorate %out2 Binding 2
        \\OpDecorate %in3 DescriptorSet 0
        \\OpDecorate %in3 Binding 3
        \\OpDecorate %out3 DescriptorSet 0
        \\OpDecorate %out3 Binding 4
        \\%void = OpTypeVoid
        \\%fn = OpTypeFunction %void
        \\%u32 = OpTypeInt 32 0
        \\%f32 = OpTypeFloat 32
        \\%v3u32 = OpTypeVector %u32 3
        \\%v3f = OpTypeVector %f32 3
        \\%c0 = OpConstant %u32 0
        \\%c1 = OpConstant %u32 1
        \\%two = OpConstant %f32 2
        \\%Rec = OpTypeStruct %v3f %f32
        \\%rtR = OpTypeRuntimeArray %Rec
        \\%BufR = OpTypeStruct %rtR
        \\%rt3 = OpTypeRuntimeArray %v3f
        \\%Buf3 = OpTypeStruct %rt3
        \\%pBufR = OpTypePointer StorageBuffer %BufR
        \\%pBuf3 = OpTypePointer StorageBuffer %Buf3
        \\%pRec = OpTypePointer StorageBuffer %Rec
        \\%pv3 = OpTypePointer StorageBuffer %v3f
        \\%pf = OpTypePointer StorageBuffer %f32
        \\%pin = OpTypePointer Input %v3u32
        \\%gid = OpVariable %pin Input
        \\%in = OpVariable %pBufR StorageBuffer
        \\%out = OpVariable %pBufR StorageBuffer
        \\%out2 = OpVariable %pBufR StorageBuffer
        \\%in3 = OpVariable %pBuf3 StorageBuffer
        \\%out3 = OpVariable %pBuf3 StorageBuffer
        \\%k = OpFunction %void None %fn
        \\%e = OpLabel
        \\%gv = OpLoad %v3u32 %gid
        \\%g = OpCompositeExtract %u32 %gv 0
        \\%pr = OpAccessChain %pRec %in %c0 %g
        \\%rec = OpLoad %Rec %pr
        \\%v = OpCompositeExtract %v3f %rec 0
        \\%s = OpCompositeExtract %f32 %rec 1
        \\%v2 = OpVectorTimesScalar %v3f %v %two
        \\%vy = OpCompositeExtract %f32 %rec 0 1
        \\%s1 = OpFAdd %f32 %s %vy
        \\%nr = OpCompositeConstruct %Rec %v2 %s1
        \\%po = OpAccessChain %pRec %out %c0 %g
        \\OpStore %po %nr
        \\%ps = OpAccessChain %pf %out2 %c0 %g %c1
        \\OpStore %ps %s1
        \\%pv = OpAccessChain %pv3 %out2 %c0 %g %c0
        \\OpStore %pv %v2
        \\%pe = OpAccessChain %pv3 %in3 %c0 %g
        \\%w = OpLoad %v3f %pe
        \\%w2 = OpVectorTimesScalar %v3f %w %two
        \\%pe2 = OpAccessChain %pv3 %out3 %c0 %g
        \\OpStore %pe2 %w2
        \\OpReturn
        \\OpFunctionEnd
    , .buffers = try a.dupe(Buffer, &.{
        .{ .index = 0, .words = rin.items },
        .{ .index = 1, .words = try zeros(a, 64), .expect = rout.items },
        .{ .index = 2, .words = try zeros(a, 64), .expect = rout.items },
        .{ .index = 3, .words = v3in.items },
        .{ .index = 4, .words = try zeros(a, 48), .expect = v3out.items },
    }) });

    // NonSemantic.Shader.DebugInfo.100 at module scope and in the body
    // (dropped), naming a buffer the kernel does not otherwise use (which
    // must not become a parameter).
    try list.append(a, .{ .name = "nonsemantic_debuginfo", .threads = 4, .text =
        \\OpCapability Shader
        \\OpExtension "SPV_KHR_non_semantic_info"
        \\%dbg = OpExtInstImport "NonSemantic.Shader.DebugInfo.100"
        \\OpMemoryModel Logical GLSL450
        \\OpEntryPoint GLCompute %k "nonsemantic_debuginfo" %gid
        \\OpExecutionMode %k LocalSize 4 1 1
        \\%file = OpString "k.comp"
        \\OpDecorate %gid BuiltIn GlobalInvocationId
        \\OpDecorate %rt ArrayStride 4
        \\OpDecorate %Buf Block
        \\OpMemberDecorate %Buf 0 Offset 0
        \\OpDecorate %out DescriptorSet 0
        \\OpDecorate %out Binding 0
        \\%void = OpTypeVoid
        \\%fn = OpTypeFunction %void
        \\%u32 = OpTypeInt 32 0
        \\%v3u32 = OpTypeVector %u32 3
        \\%c0 = OpConstant %u32 0
        \\%c9 = OpConstant %u32 9
        \\%rt = OpTypeRuntimeArray %u32
        \\%Buf = OpTypeStruct %rt
        \\%pBuf = OpTypePointer StorageBuffer %Buf
        \\%pu = OpTypePointer StorageBuffer %u32
        \\%pin = OpTypePointer Input %v3u32
        \\%gid = OpVariable %pin Input
        \\%out = OpVariable %pBuf StorageBuffer
        \\%unused = OpVariable %pBuf StorageBuffer
        \\%src = OpExtInst %void %dbg 35 %file
        \\%k = OpFunction %void None %fn
        \\%e = OpLabel
        \\%scope = OpExtInst %void %dbg 23 %src
        \\%decl = OpExtInst %void %dbg 28 %src %unused %c0
        \\%gv = OpLoad %v3u32 %gid
        \\%g = OpCompositeExtract %u32 %gv 0
        \\%y = OpIAdd %u32 %g %c9
        \\%line = OpExtInst %void %dbg 103 %src %c0 %c0 %c0 %c0
        \\%po = OpAccessChain %pu %out %c0 %g
        \\OpStore %po %y
        \\OpReturn
        \\OpFunctionEnd
    , .buffers = try a.dupe(Buffer, &.{
        .{ .index = 0, .words = try zeros(a, 4), .expect = &.{ 9, 10, 11, 12 } },
    }) });
    try appendLayoutTests(a, &list);
    try appendZigConvertTests(a, &list);
    try appendEdgeTests(a, &list);
    try appendBarrierTests(a, &list);
    try appendLoweringFixTests(a, &list);
    try appendRound2LoweringTests(a, &list);
    try appendUniformBoundTests(a, &list);

    // Every interface decoration applied through decoration groups
    // (SPIRV-LLVM-Translator and older glslang emit them; reader.zig expands
    // them into the decoration table): DescriptorSet shared by both buffers,
    // swapped Bindings (input at 3, output at 1, so a lost expansion binds
    // the wrong buffer or none), BufferBlock, ArrayStride and a member Offset
    // through OpGroupMemberDecorate.
    try list.append(a, .{ .name = "decoration_groups", .threads = 16, .text =
        \\OpCapability Shader
        \\OpMemoryModel Logical GLSL450
        \\OpEntryPoint GLCompute %k "decoration_groups" %gid
        \\OpExecutionMode %k LocalSize 16 1 1
        \\OpDecorate %gid BuiltIn GlobalInvocationId
        \\OpDecorate %gset DescriptorSet 0
        \\OpDecorate %gin Binding 3
        \\OpDecorate %gout Binding 1
        \\OpDecorate %gblock BufferBlock
        \\OpDecorate %gstride ArrayStride 4
        \\OpDecorate %goff Offset 0
        \\%gset = OpDecorationGroup
        \\%gin = OpDecorationGroup
        \\%gout = OpDecorationGroup
        \\%gblock = OpDecorationGroup
        \\%gstride = OpDecorationGroup
        \\%goff = OpDecorationGroup
        \\OpGroupDecorate %gset %in %out
        \\OpGroupDecorate %gin %in
        \\OpGroupDecorate %gout %out
        \\OpGroupDecorate %gblock %Buf
        \\OpGroupDecorate %gstride %rt
        \\OpGroupMemberDecorate %goff %Buf 0
        \\%void = OpTypeVoid
        \\%fn = OpTypeFunction %void
        \\%u32 = OpTypeInt 32 0
        \\%v3u32 = OpTypeVector %u32 3
        \\%c0 = OpConstant %u32 0
        \\%c3 = OpConstant %u32 3
        \\%c7 = OpConstant %u32 7
        \\%rt = OpTypeRuntimeArray %u32
        \\%Buf = OpTypeStruct %rt
        \\%pBuf = OpTypePointer Uniform %Buf
        \\%pu = OpTypePointer Uniform %u32
        \\%pin = OpTypePointer Input %v3u32
        \\%gid = OpVariable %pin Input
        \\%in = OpVariable %pBuf Uniform
        \\%out = OpVariable %pBuf Uniform
        \\%k = OpFunction %void None %fn
        \\%e = OpLabel
        \\%gv = OpLoad %v3u32 %gid
        \\%g = OpCompositeExtract %u32 %gv 0
        \\%pi = OpAccessChain %pu %in %c0 %g
        \\%x = OpLoad %u32 %pi
        \\%x3 = OpIMul %u32 %x %c3
        \\%y = OpIAdd %u32 %x3 %c7
        \\%po = OpAccessChain %pu %out %c0 %g
        \\OpStore %po %y
        \\OpReturn
        \\OpFunctionEnd
    , .buffers = try a.dupe(Buffer, &.{
        .{ .index = 3, .words = try seq(a, 16, struct {
            fn f(i: usize) u32 {
                return @intCast(i + 1);
            }
        }.f) },
        .{ .index = 1, .words = try zeros(a, 16), .expect = try seq(a, 16, struct {
            fn f(i: usize) u32 {
                return @intCast(3 * (i + 1) + 7);
            }
        }.f) },
    }) });
    return list.items;
}

/// Results the specifications define at the edges, where a plain mapping
/// gives poison or another function's value (review findings), and the
/// alignment of a packed member reaching a load through OpSelect, a phi, a
/// helper parameter and a Function variable holding the pointer.
fn appendEdgeTests(a: Allocator, list: *std.ArrayList(Test)) !void {
    // OpBitFieldUExtract / SExtract with Offset = 32 and Count = 0 (defined:
    // 0) and OpBitFieldInsert (defined: the base), with constant operands,
    // which LLVM folds: `lshr x, 32` is poison and `and poison, 0` too, so
    // an extract that masks instead of selecting the result was folded away
    // and Metal dropped the store (out[0] kept 0x11111111). The same with
    // runtime operands (in[1] = 32, in[2] = 0), a 2-lane vector, and two
    // ordinary extracts as controls.
    try list.append(a, .{ .name = "bitfield_full_offset", .threads = 1, .text =
        \\OpCapability Shader
        \\OpMemoryModel Logical GLSL450
        \\OpEntryPoint GLCompute %k "bitfield_full_offset"
        \\OpExecutionMode %k LocalSize 1 1 1
        \\OpDecorate %rt ArrayStride 4
        \\OpDecorate %O Block
        \\OpMemberDecorate %O 0 Offset 0
        \\OpDecorate %in DescriptorSet 0
        \\OpDecorate %in Binding 0
        \\OpDecorate %out DescriptorSet 0
        \\OpDecorate %out Binding 1
        \\%void = OpTypeVoid
        \\%fn = OpTypeFunction %void
        \\%u32 = OpTypeInt 32 0
        \\%i32 = OpTypeInt 32 1
        \\%v2u32 = OpTypeVector %u32 2
        \\%c0 = OpConstant %u32 0
        \\%c1 = OpConstant %u32 1
        \\%c2 = OpConstant %u32 2
        \\%c3 = OpConstant %u32 3
        \\%c4 = OpConstant %u32 4
        \\%c5 = OpConstant %u32 5
        \\%c6 = OpConstant %u32 6
        \\%c7 = OpConstant %u32 7
        \\%c8 = OpConstant %u32 8
        \\%c32 = OpConstant %u32 32
        \\%ca = OpConstant %u32 3735928559
        \\%cb = OpConstant %u32 305419896
        \\%ci = OpConstant %i32 -559038737
        \\%cv = OpConstantComposite %v2u32 %ca %cb
        \\%rt = OpTypeRuntimeArray %u32
        \\%O = OpTypeStruct %rt
        \\%pO = OpTypePointer StorageBuffer %O
        \\%pu = OpTypePointer StorageBuffer %u32
        \\%in = OpVariable %pO StorageBuffer
        \\%out = OpVariable %pO StorageBuffer
        \\%k = OpFunction %void None %fn
        \\%e = OpLabel
        \\%r0 = OpBitFieldUExtract %u32 %ca %c32 %c0
        \\%po0 = OpAccessChain %pu %out %c0 %c0
        \\OpStore %po0 %r0
        \\%r1s = OpBitFieldSExtract %i32 %ci %c32 %c0
        \\%r1 = OpBitcast %u32 %r1s
        \\%po1 = OpAccessChain %pu %out %c0 %c1
        \\OpStore %po1 %r1
        \\%r2 = OpBitFieldInsert %u32 %ca %cb %c32 %c0
        \\%po2 = OpAccessChain %pu %out %c0 %c2
        \\OpStore %po2 %r2
        \\%pa = OpAccessChain %pu %in %c0 %c0
        \\%a = OpLoad %u32 %pa
        \\%poff = OpAccessChain %pu %in %c0 %c1
        \\%off = OpLoad %u32 %poff
        \\%pcnt = OpAccessChain %pu %in %c0 %c2
        \\%cnt = OpLoad %u32 %pcnt
        \\%r3 = OpBitFieldUExtract %u32 %a %off %cnt
        \\%po3 = OpAccessChain %pu %out %c0 %c3
        \\OpStore %po3 %r3
        \\%r4v = OpBitFieldUExtract %v2u32 %cv %c32 %c0
        \\%r4 = OpCompositeExtract %u32 %r4v 0
        \\%r5 = OpCompositeExtract %u32 %r4v 1
        \\%po4 = OpAccessChain %pu %out %c0 %c4
        \\OpStore %po4 %r4
        \\%po5 = OpAccessChain %pu %out %c0 %c5
        \\OpStore %po5 %r5
        \\%r6 = OpBitFieldUExtract %u32 %ca %c4 %c8
        \\%po6 = OpAccessChain %pu %out %c0 %c6
        \\OpStore %po6 %r6
        \\%r7 = OpBitFieldUExtract %u32 %a %c0 %c32
        \\%po7 = OpAccessChain %pu %out %c0 %c7
        \\OpStore %po7 %r7
        \\OpReturn
        \\OpFunctionEnd
    , .buffers = try a.dupe(Buffer, &.{
        .{ .index = 0, .words = &.{ 0xdeadbeef, 32, 0 } },
        .{ .index = 1, .words = try filled(a, 8, 0x11111111), .expect = &.{ 0, 0, 0xdeadbeef, 0, 0, 0, 0xee, 0xdeadbeef } },
    }) });

    // OpenCL C's own edge cases (3.0, "Additional Requirements Beyond C99"
    // and the geometric functions), which pow and v/|v| miss: powr is NaN
    // for x < 0, (0, 0), (+inf, 0), (1, inf) and a NaN operand, and +0 /
    // +inf for x = -0; pow keeps C99's pow(-2, 2) = 4. normalize returns a
    // zero vector unchanged, treats infinite elements as ±1 and the rest as
    // 0, and like length and distance must not overflow or underflow in the
    // sum of squares (1e30 and 1e-30); a scalar is ±1, or 0 for 0. in: -2,
    // 2, 0, -0, 1, inf, 3, -3, 1e30, 1e-30, NaN. The words match with and
    // without fast math on the M3.
    try list.append(a, .{
        .name = "opencl_edge_math",
        .threads = 1,
        .text =
        \\OpCapability Kernel
        \\OpCapability Addresses
        \\OpCapability Int64
        \\%ocl = OpExtInstImport "OpenCL.std"
        \\OpMemoryModel Physical64 OpenCL
        \\OpEntryPoint Kernel %k "opencl_edge_math"
        \\%void = OpTypeVoid
        \\%u64 = OpTypeInt 64 0
        \\%f32 = OpTypeFloat 32
        \\%v2f32 = OpTypeVector %f32 2
        \\%pf = OpTypePointer CrossWorkgroup %f32
        \\%fn = OpTypeFunction %void %pf %pf
        \\%c0 = OpConstant %u64 0
        \\%c1 = OpConstant %u64 1
        \\%c2 = OpConstant %u64 2
        \\%c3 = OpConstant %u64 3
        \\%c4 = OpConstant %u64 4
        \\%c5 = OpConstant %u64 5
        \\%c6 = OpConstant %u64 6
        \\%c7 = OpConstant %u64 7
        \\%c8 = OpConstant %u64 8
        \\%c9 = OpConstant %u64 9
        \\%c10 = OpConstant %u64 10
        \\%c11 = OpConstant %u64 11
        \\%c12 = OpConstant %u64 12
        \\%c13 = OpConstant %u64 13
        \\%c14 = OpConstant %u64 14
        \\%c15 = OpConstant %u64 15
        \\%c16 = OpConstant %u64 16
        \\%c17 = OpConstant %u64 17
        \\%c18 = OpConstant %u64 18
        \\%c19 = OpConstant %u64 19
        \\%c20 = OpConstant %u64 20
        \\%c21 = OpConstant %u64 21
        \\%c22 = OpConstant %u64 22
        \\%c23 = OpConstant %u64 23
        \\%c24 = OpConstant %u64 24
        \\%c25 = OpConstant %u64 25
        \\%c26 = OpConstant %u64 26
        \\%c27 = OpConstant %u64 27
        \\%c28 = OpConstant %u64 28
        \\%c29 = OpConstant %u64 29
        \\%c30 = OpConstant %u64 30
        \\%c31 = OpConstant %u64 31
        \\%k = OpFunction %void None %fn
        \\%in = OpFunctionParameter %pf
        \\%out = OpFunctionParameter %pf
        \\%e = OpLabel
        \\%pi0 = OpInBoundsPtrAccessChain %pf %in %c0
        \\%x0 = OpLoad %f32 %pi0 Aligned 4
        \\%pi1 = OpInBoundsPtrAccessChain %pf %in %c1
        \\%x1 = OpLoad %f32 %pi1 Aligned 4
        \\%pi2 = OpInBoundsPtrAccessChain %pf %in %c2
        \\%x2 = OpLoad %f32 %pi2 Aligned 4
        \\%pi3 = OpInBoundsPtrAccessChain %pf %in %c3
        \\%x3 = OpLoad %f32 %pi3 Aligned 4
        \\%pi4 = OpInBoundsPtrAccessChain %pf %in %c4
        \\%x4 = OpLoad %f32 %pi4 Aligned 4
        \\%pi5 = OpInBoundsPtrAccessChain %pf %in %c5
        \\%x5 = OpLoad %f32 %pi5 Aligned 4
        \\%pi6 = OpInBoundsPtrAccessChain %pf %in %c6
        \\%x6 = OpLoad %f32 %pi6 Aligned 4
        \\%pi7 = OpInBoundsPtrAccessChain %pf %in %c7
        \\%x7 = OpLoad %f32 %pi7 Aligned 4
        \\%pi8 = OpInBoundsPtrAccessChain %pf %in %c8
        \\%x8 = OpLoad %f32 %pi8 Aligned 4
        \\%pi9 = OpInBoundsPtrAccessChain %pf %in %c9
        \\%x9 = OpLoad %f32 %pi9 Aligned 4
        \\%pi10 = OpInBoundsPtrAccessChain %pf %in %c10
        \\%x10 = OpLoad %f32 %pi10 Aligned 4
        \\%r0 = OpExtInst %f32 %ocl powr %x0 %x1
        \\%po0 = OpInBoundsPtrAccessChain %pf %out %c0
        \\OpStore %po0 %r0 Aligned 4
        \\%r1 = OpExtInst %f32 %ocl powr %x2 %x2
        \\%po1 = OpInBoundsPtrAccessChain %pf %out %c1
        \\OpStore %po1 %r1 Aligned 4
        \\%r2 = OpExtInst %f32 %ocl powr %x5 %x2
        \\%po2 = OpInBoundsPtrAccessChain %pf %out %c2
        \\OpStore %po2 %r2 Aligned 4
        \\%r3 = OpExtInst %f32 %ocl powr %x4 %x5
        \\%po3 = OpInBoundsPtrAccessChain %pf %out %c3
        \\OpStore %po3 %r3 Aligned 4
        \\%r4 = OpExtInst %f32 %ocl powr %x3 %x6
        \\%po4 = OpInBoundsPtrAccessChain %pf %out %c4
        \\OpStore %po4 %r4 Aligned 4
        \\%r5 = OpExtInst %f32 %ocl powr %x3 %x7
        \\%po5 = OpInBoundsPtrAccessChain %pf %out %c5
        \\OpStore %po5 %r5 Aligned 4
        \\%r6 = OpExtInst %f32 %ocl powr %x1 %x6
        \\%po6 = OpInBoundsPtrAccessChain %pf %out %c6
        \\OpStore %po6 %r6 Aligned 4
        \\%r7 = OpExtInst %f32 %ocl powr %x4 %x10
        \\%po7 = OpInBoundsPtrAccessChain %pf %out %c7
        \\OpStore %po7 %r7 Aligned 4
        \\%r8 = OpExtInst %f32 %ocl pow %x0 %x1
        \\%po8 = OpInBoundsPtrAccessChain %pf %out %c8
        \\OpStore %po8 %r8 Aligned 4
        \\%v0 = OpCompositeConstruct %v2f32 %x2 %x3
        \\%n0 = OpExtInst %v2f32 %ocl normalize %v0
        \\%n0_0 = OpCompositeExtract %f32 %n0 0
        \\%po9 = OpInBoundsPtrAccessChain %pf %out %c9
        \\OpStore %po9 %n0_0 Aligned 4
        \\%n0_1 = OpCompositeExtract %f32 %n0 1
        \\%po10 = OpInBoundsPtrAccessChain %pf %out %c10
        \\OpStore %po10 %n0_1 Aligned 4
        \\%v1 = OpCompositeConstruct %v2f32 %x5 %x6
        \\%n1 = OpExtInst %v2f32 %ocl normalize %v1
        \\%n1_0 = OpCompositeExtract %f32 %n1 0
        \\%po11 = OpInBoundsPtrAccessChain %pf %out %c11
        \\OpStore %po11 %n1_0 Aligned 4
        \\%n1_1 = OpCompositeExtract %f32 %n1 1
        \\%po12 = OpInBoundsPtrAccessChain %pf %out %c12
        \\OpStore %po12 %n1_1 Aligned 4
        \\%v2 = OpCompositeConstruct %v2f32 %x8 %x2
        \\%n2 = OpExtInst %v2f32 %ocl normalize %v2
        \\%n2_0 = OpCompositeExtract %f32 %n2 0
        \\%po13 = OpInBoundsPtrAccessChain %pf %out %c13
        \\OpStore %po13 %n2_0 Aligned 4
        \\%n2_1 = OpCompositeExtract %f32 %n2 1
        \\%po14 = OpInBoundsPtrAccessChain %pf %out %c14
        \\OpStore %po14 %n2_1 Aligned 4
        \\%v3 = OpCompositeConstruct %v2f32 %x9 %x2
        \\%n3 = OpExtInst %v2f32 %ocl normalize %v3
        \\%n3_0 = OpCompositeExtract %f32 %n3 0
        \\%po15 = OpInBoundsPtrAccessChain %pf %out %c15
        \\OpStore %po15 %n3_0 Aligned 4
        \\%n3_1 = OpCompositeExtract %f32 %n3 1
        \\%po16 = OpInBoundsPtrAccessChain %pf %out %c16
        \\OpStore %po16 %n3_1 Aligned 4
        \\%s0 = OpExtInst %f32 %ocl normalize %x2
        \\%po17 = OpInBoundsPtrAccessChain %pf %out %c17
        \\OpStore %po17 %s0 Aligned 4
        \\%s1 = OpExtInst %f32 %ocl normalize %x7
        \\%po18 = OpInBoundsPtrAccessChain %pf %out %c18
        \\OpStore %po18 %s1 Aligned 4
        \\%ninf = OpFNegate %f32 %x5
        \\%s2 = OpExtInst %f32 %ocl normalize %ninf
        \\%po19 = OpInBoundsPtrAccessChain %pf %out %c19
        \\OpStore %po19 %s2 Aligned 4
        \\%l0 = OpExtInst %f32 %ocl length %v2
        \\%po20 = OpInBoundsPtrAccessChain %pf %out %c20
        \\OpStore %po20 %l0 Aligned 4
        \\%v4 = OpCompositeConstruct %v2f32 %x2 %x2
        \\%l1 = OpExtInst %f32 %ocl distance %v2 %v4
        \\%po21 = OpInBoundsPtrAccessChain %pf %out %c21
        \\OpStore %po21 %l1 Aligned 4
        \\%v5 = OpCompositeConstruct %v2f32 %x6 %x2
        \\%l2 = OpExtInst %f32 %ocl length %v5
        \\%po22 = OpInBoundsPtrAccessChain %pf %out %c22
        \\OpStore %po22 %l2 Aligned 4
        \\%v6 = OpCompositeConstruct %v2f32 %x10 %x4
        \\%n4 = OpExtInst %v2f32 %ocl normalize %v6
        \\%n4_0 = OpCompositeExtract %f32 %n4 0
        \\%po23 = OpInBoundsPtrAccessChain %pf %out %c23
        \\OpStore %po23 %n4_0 Aligned 4
        \\%n4_1 = OpCompositeExtract %f32 %n4 1
        \\%po24 = OpInBoundsPtrAccessChain %pf %out %c24
        \\OpStore %po24 %n4_1 Aligned 4
        \\OpReturn
        \\OpFunctionEnd
        ,
        .buffers = try a.dupe(Buffer, &.{
            .{ .index = 0, .words = &.{ 0xc0000000, 0x40000000, 0, 0x80000000, 0x3f800000, 0x7f800000, 0x40400000, 0xc0400000, 0x7149f2ca, 0x0da24260, 0x7fc00000 } },
            .{
                .index = 1,
                .words = try zeros(a, 25),
                .expect = &.{
                    // powr: NaN NaN NaN NaN +0 +inf 8 NaN; pow(-2, 2) = 4
                    0x7fc00000, 0x7fc00000, 0x7fc00000, 0x7fc00000, 0,          0x7f800000, 0x41000000, 0x7fc00000,
                    0x40800000,
                    // normalize (0, -0), (inf, 3), (1e30, 0), (1e-30, 0)
                    0,          0x80000000, 0x3f800000, 0,          0x3f800000, 0,          0x3f800000,
                    0,
                    // normalize 0, -3, -inf; length (1e30, 0); distance; length (3, 0)
                             0,          0xbf800000, 0xbf800000, 0x7149f2ca, 0x7149f2ca, 0x40400000,
                    // normalize (NaN, 1)
                    0x7fc00000,
                    0x7fc00000,
                },
            },
        }),
    });

    // A scalar-layout `P { uint a @0; uvec4 v @4 }` array (ArrayStride
    // 20): &ps[g].v is only 4-aligned, which the access chain knows. The
    // pointer then reaches a load through OpSelect, a helper's parameter, a
    // loop phi (whose back-edge value is lowered after the load) and a
    // Function variable holding it (VariablePointers); each load must say
    // `align 4`, not uvec4's 16 (LLVM undefined behaviour). out[4g..]:
    // select(odd g, ps[g], ps[g+1]).v.x, h(&ps[g].v) = .y, the loop's sum of
    // ps[g].v.z + ps[g+1].v.z, and ps[g+1].v.w through the variable; in
    // holds ps[k] = { k, 100k+1, 100k+2, 100k+3, 100k+4 }.
    var ps: std.ArrayList(u32) = .empty;
    for (0..6) |k| {
        const c: u32 = @intCast(k);
        try ps.appendSlice(a, &.{ c, 100 * c + 1, 100 * c + 2, 100 * c + 3, 100 * c + 4 });
    }
    try list.append(a, .{ .name = "packed_pointer_flow", .threads = 4, .text =
        \\OpCapability Shader
        \\OpCapability VariablePointers
        \\OpExtension "SPV_KHR_variable_pointers"
        \\OpMemoryModel Logical GLSL450
        \\OpEntryPoint GLCompute %k "packed_pointer_flow" %gid
        \\OpExecutionMode %k LocalSize 4 1 1
        \\OpDecorate %gid BuiltIn GlobalInvocationId
        \\OpMemberDecorate %P 0 Offset 0
        \\OpMemberDecorate %P 1 Offset 4
        \\OpDecorate %rtP ArrayStride 20
        \\OpDecorate %I Block
        \\OpMemberDecorate %I 0 Offset 0
        \\OpDecorate %rtu ArrayStride 4
        \\OpDecorate %O Block
        \\OpMemberDecorate %O 0 Offset 0
        \\OpDecorate %in DescriptorSet 0
        \\OpDecorate %in Binding 0
        \\OpDecorate %in NonWritable
        \\OpDecorate %out DescriptorSet 0
        \\OpDecorate %out Binding 1
        \\%void = OpTypeVoid
        \\%fn = OpTypeFunction %void
        \\%bool = OpTypeBool
        \\%u32 = OpTypeInt 32 0
        \\%v3u32 = OpTypeVector %u32 3
        \\%v4u32 = OpTypeVector %u32 4
        \\%c0 = OpConstant %u32 0
        \\%c1 = OpConstant %u32 1
        \\%c2 = OpConstant %u32 2
        \\%c3 = OpConstant %u32 3
        \\%c4 = OpConstant %u32 4
        \\%P = OpTypeStruct %u32 %v4u32
        \\%rtP = OpTypeRuntimeArray %P
        \\%I = OpTypeStruct %rtP
        \\%rtu = OpTypeRuntimeArray %u32
        \\%O = OpTypeStruct %rtu
        \\%pI = OpTypePointer StorageBuffer %I
        \\%pO = OpTypePointer StorageBuffer %O
        \\%psu = OpTypePointer StorageBuffer %u32
        \\%psv = OpTypePointer StorageBuffer %v4u32
        \\%pfv = OpTypePointer Function %psv
        \\%pin = OpTypePointer Input %v3u32
        \\%fnh = OpTypeFunction %u32 %psv
        \\%gid = OpVariable %pin Input
        \\%in = OpVariable %pI StorageBuffer
        \\%out = OpVariable %pO StorageBuffer
        \\%h = OpFunction %u32 None %fnh
        \\%hp = OpFunctionParameter %psv
        \\%he = OpLabel
        \\%hv = OpLoad %v4u32 %hp
        \\%hy = OpCompositeExtract %u32 %hv 1
        \\OpReturnValue %hy
        \\OpFunctionEnd
        \\%k = OpFunction %void None %fn
        \\%e = OpLabel
        \\%var = OpVariable %pfv Function
        \\%gv = OpLoad %v3u32 %gid
        \\%g = OpCompositeExtract %u32 %gv 0
        \\%g1 = OpIAdd %u32 %g %c1
        \\%pa = OpAccessChain %psv %in %c0 %g %c1
        \\%pb = OpAccessChain %psv %in %c0 %g1 %c1
        \\%gm = OpBitwiseAnd %u32 %g %c1
        \\%odd = OpIEqual %bool %gm %c1
        \\%psel = OpSelect %psv %odd %pa %pb
        \\%vs = OpLoad %v4u32 %psel
        \\%r0 = OpCompositeExtract %u32 %vs 0
        \\%o = OpIMul %u32 %g %c4
        \\%po0 = OpAccessChain %psu %out %c0 %o
        \\OpStore %po0 %r0
        \\%r1 = OpFunctionCall %u32 %h %pa
        \\%o1 = OpIAdd %u32 %o %c1
        \\%po1 = OpAccessChain %psu %out %c0 %o1
        \\OpStore %po1 %r1
        \\OpBranch %loop
        \\%loop = OpLabel
        \\%i = OpPhi %u32 %c0 %e %inext %body
        \\%acc = OpPhi %u32 %c0 %e %acc2 %body
        \\%pl = OpPhi %psv %pa %e %pnext %body
        \\%more = OpULessThan %bool %i %c2
        \\OpLoopMerge %done %body None
        \\OpBranchConditional %more %body %done
        \\%body = OpLabel
        \\%lv = OpLoad %v4u32 %pl
        \\%lz = OpCompositeExtract %u32 %lv 2
        \\%acc2 = OpIAdd %u32 %acc %lz
        \\%inext = OpIAdd %u32 %i %c1
        \\%gi = OpIAdd %u32 %g %inext
        \\%pnext = OpAccessChain %psv %in %c0 %gi %c1
        \\OpBranch %loop
        \\%done = OpLabel
        \\%o2 = OpIAdd %u32 %o %c2
        \\%po2 = OpAccessChain %psu %out %c0 %o2
        \\OpStore %po2 %acc
        \\OpStore %var %pb
        \\%pv = OpLoad %psv %var
        \\%wv = OpLoad %v4u32 %pv
        \\%w = OpCompositeExtract %u32 %wv 3
        \\%o3 = OpIAdd %u32 %o %c3
        \\%po3 = OpAccessChain %psu %out %c0 %o3
        \\OpStore %po3 %w
        \\OpReturn
        \\OpFunctionEnd
    , .buffers = try a.dupe(Buffer, &.{
        .{ .index = 0, .words = ps.items },
        .{ .index = 1, .words = try zeros(a, 16), .expect = &.{
            101, 2,   106, 104,
            101, 102, 306, 204,
            301, 202, 506, 304,
            301, 302, 706, 404,
        } },
    }) });
}

/// Words of `Particle { uvec3 pos @0; uint mass @12; uvec4 color @16 }`
/// (std430, 32 bytes) number `i`: pos (i, 2i, 3i), mass 100 + i, color
/// (10i, 10i + 1, 10i + 2, 10i + 3).
fn particle(i: u32) [8]u32 {
    return .{ i, 2 * i, 3 * i, 100 + i, 10 * i, 10 * i + 1, 10 * i + 2, 10 * i + 3 };
}

fn appendLayoutTests(a: Allocator, list: *std.ArrayList(Test)) !void {
    // A packed struct (Offsets that AIR's natural layout misses: a vec3
    // followed by a scalar at 12) as a member of a struct whose own
    // Offsets look natural. LLVM gives a packed struct ABI alignment 1, so
    // `{ i32, [0 x <{ ... }>] }` would put the array at byte 4, not 16:
    // the parent must be packed too. std430 SSBO `{ uint count; Particle
    // ps[]; }` and std140 UBO `{ uint count; Light l; }` (the review's
    // t12 / t15), reading the scalar after the vec3 and a lane of the vec4.
    var ssbo: std.ArrayList(u32) = .empty;
    try ssbo.appendSlice(a, &.{ 7, 0, 0, 0 });
    for (0..4) |i| try ssbo.appendSlice(a, &particle(@intCast(i)));
    var ubo: std.ArrayList(u32) = .empty;
    try ubo.appendSlice(a, &.{ 9, 0, 0, 0 });
    try ubo.appendSlice(a, &particle(5));
    try list.append(a, .{
        .name = "packed_in_natural",
        .threads = 4,
        .text =
        \\OpCapability Shader
        \\OpMemoryModel Logical GLSL450
        \\OpEntryPoint GLCompute %k "packed_in_natural" %gid
        \\OpExecutionMode %k LocalSize 4 1 1
        \\OpDecorate %gid BuiltIn GlobalInvocationId
        \\OpMemberDecorate %P 0 Offset 0
        \\OpMemberDecorate %P 1 Offset 12
        \\OpMemberDecorate %P 2 Offset 16
        \\OpDecorate %rtP ArrayStride 32
        \\OpDecorate %B Block
        \\OpMemberDecorate %B 0 Offset 0
        \\OpMemberDecorate %B 1 Offset 16
        \\OpDecorate %U Block
        \\OpMemberDecorate %U 0 Offset 0
        \\OpMemberDecorate %U 1 Offset 16
        \\OpDecorate %rtu ArrayStride 4
        \\OpDecorate %O Block
        \\OpMemberDecorate %O 0 Offset 0
        \\OpDecorate %in DescriptorSet 0
        \\OpDecorate %in Binding 0
        \\OpDecorate %in NonWritable
        \\OpDecorate %out DescriptorSet 0
        \\OpDecorate %out Binding 1
        \\OpDecorate %ubo DescriptorSet 0
        \\OpDecorate %ubo Binding 2
        \\%void = OpTypeVoid
        \\%fn = OpTypeFunction %void
        \\%u32 = OpTypeInt 32 0
        \\%v3u32 = OpTypeVector %u32 3
        \\%v4u32 = OpTypeVector %u32 4
        \\%c0 = OpConstant %u32 0
        \\%c1 = OpConstant %u32 1
        \\%c2 = OpConstant %u32 2
        \\%c3 = OpConstant %u32 3
        \\%c4 = OpConstant %u32 4
        \\%c8 = OpConstant %u32 8
        \\%c9 = OpConstant %u32 9
        \\%c1000 = OpConstant %u32 1000
        \\%P = OpTypeStruct %v3u32 %u32 %v4u32
        \\%rtP = OpTypeRuntimeArray %P
        \\%B = OpTypeStruct %u32 %rtP
        \\%U = OpTypeStruct %u32 %P
        \\%rtu = OpTypeRuntimeArray %u32
        \\%O = OpTypeStruct %rtu
        \\%pB = OpTypePointer StorageBuffer %B
        \\%pU = OpTypePointer Uniform %U
        \\%pO = OpTypePointer StorageBuffer %O
        \\%psu = OpTypePointer StorageBuffer %u32
        \\%puu = OpTypePointer Uniform %u32
        \\%pin = OpTypePointer Input %v3u32
        \\%gid = OpVariable %pin Input
        \\%in = OpVariable %pB StorageBuffer
        \\%out = OpVariable %pO StorageBuffer
        \\%ubo = OpVariable %pU Uniform
        \\%k = OpFunction %void None %fn
        \\%e = OpLabel
        \\%gv = OpLoad %v3u32 %gid
        \\%g = OpCompositeExtract %u32 %gv 0
        \\%pcount = OpAccessChain %psu %in %c0
        \\%count = OpLoad %u32 %pcount
        \\%pmass = OpAccessChain %psu %in %c1 %g %c1
        \\%mass = OpLoad %u32 %pmass
        \\%cm = OpIMul %u32 %count %c1000
        \\%r0 = OpIAdd %u32 %cm %mass
        \\%po0 = OpAccessChain %psu %out %c0 %g
        \\OpStore %po0 %r0
        \\%pcy = OpAccessChain %psu %in %c1 %g %c2 %c1
        \\%cy = OpLoad %u32 %pcy
        \\%i1 = OpIAdd %u32 %g %c4
        \\%po1 = OpAccessChain %psu %out %c0 %i1
        \\OpStore %po1 %cy
        \\%pucount = OpAccessChain %puu %ubo %c0
        \\%ucount = OpLoad %u32 %pucount
        \\%pub = OpAccessChain %puu %ubo %c1 %c1
        \\%ub = OpLoad %u32 %pub
        \\%ucm = OpIMul %u32 %ucount %c1000
        \\%r2 = OpIAdd %u32 %ucm %ub
        \\%pucw = OpAccessChain %puu %ubo %c1 %c2 %c3
        \\%ucw = OpLoad %u32 %pucw
        \\%r3 = OpIAdd %u32 %r2 %ucw
        \\%i2 = OpIAdd %u32 %g %c8
        \\%po2 = OpAccessChain %psu %out %c0 %i2
        \\OpStore %po2 %r3
        \\OpReturn
        \\OpFunctionEnd
        ,
        .buffers = try a.dupe(Buffer, &.{
            .{ .index = 0, .words = ssbo.items },
            // count * 1000 + mass, color.y, then 9 * 1000 + 105 + 53 per thread.
            .{ .index = 1, .words = try zeros(a, 12), .expect = &.{ 7100, 7101, 7102, 7103, 1, 11, 21, 31, 9158, 9158, 9158, 9158 } },
            .{ .index = 2, .words = ubo.items },
        }),
    });

    // Scalar block layout: `P { uint a @0; uvec4 v @4 }` (20 bytes, packed)
    // in `B2 { uint count @0; P ps[] @16 }` (a natural parent's Offsets,
    // the review's t2) and in `B3 { uint n @0; P p @4 }`, where the natural
    // layout is right because the packed member's ABI alignment is 1. The
    // uvec4 loads are aligned 4, as their offsets are.
    var b2: std.ArrayList(u32) = .empty;
    try b2.appendSlice(a, &.{ 7, 0, 0, 0 });
    for (0..4) |i| {
        const x: u32 = @intCast(i);
        try b2.appendSlice(a, &.{ 100 + x, 10 * x, 10 * x + 1, 10 * x + 2, 10 * x + 3 });
    }
    try list.append(a, .{
        .name = "scalar_packed",
        .threads = 4,
        .text =
        \\OpCapability Shader
        \\OpMemoryModel Logical GLSL450
        \\OpEntryPoint GLCompute %k "scalar_packed" %gid
        \\OpExecutionMode %k LocalSize 4 1 1
        \\OpDecorate %gid BuiltIn GlobalInvocationId
        \\OpMemberDecorate %P 0 Offset 0
        \\OpMemberDecorate %P 1 Offset 4
        \\OpDecorate %rtP ArrayStride 20
        \\OpDecorate %B2 Block
        \\OpMemberDecorate %B2 0 Offset 0
        \\OpMemberDecorate %B2 1 Offset 16
        \\OpDecorate %B3 Block
        \\OpMemberDecorate %B3 0 Offset 0
        \\OpMemberDecorate %B3 1 Offset 4
        \\OpDecorate %rtu ArrayStride 4
        \\OpDecorate %O Block
        \\OpMemberDecorate %O 0 Offset 0
        \\OpDecorate %in DescriptorSet 0
        \\OpDecorate %in Binding 0
        \\OpDecorate %out DescriptorSet 0
        \\OpDecorate %out Binding 1
        \\OpDecorate %in3 DescriptorSet 0
        \\OpDecorate %in3 Binding 2
        \\%void = OpTypeVoid
        \\%fn = OpTypeFunction %void
        \\%u32 = OpTypeInt 32 0
        \\%v3u32 = OpTypeVector %u32 3
        \\%v4u32 = OpTypeVector %u32 4
        \\%c0 = OpConstant %u32 0
        \\%c1 = OpConstant %u32 1
        \\%c2 = OpConstant %u32 2
        \\%c4 = OpConstant %u32 4
        \\%c8 = OpConstant %u32 8
        \\%c1000 = OpConstant %u32 1000
        \\%P = OpTypeStruct %u32 %v4u32
        \\%rtP = OpTypeRuntimeArray %P
        \\%B2 = OpTypeStruct %u32 %rtP
        \\%B3 = OpTypeStruct %u32 %P
        \\%rtu = OpTypeRuntimeArray %u32
        \\%O = OpTypeStruct %rtu
        \\%pB2 = OpTypePointer StorageBuffer %B2
        \\%pB3 = OpTypePointer StorageBuffer %B3
        \\%pO = OpTypePointer StorageBuffer %O
        \\%psu = OpTypePointer StorageBuffer %u32
        \\%psv = OpTypePointer StorageBuffer %v4u32
        \\%pin = OpTypePointer Input %v3u32
        \\%gid = OpVariable %pin Input
        \\%in = OpVariable %pB2 StorageBuffer
        \\%out = OpVariable %pO StorageBuffer
        \\%in3 = OpVariable %pB3 StorageBuffer
        \\%k = OpFunction %void None %fn
        \\%e = OpLabel
        \\%gv = OpLoad %v3u32 %gid
        \\%g = OpCompositeExtract %u32 %gv 0
        \\%pcount = OpAccessChain %psu %in %c0
        \\%count = OpLoad %u32 %pcount
        \\%pa = OpAccessChain %psu %in %c1 %g %c0
        \\%av = OpLoad %u32 %pa
        \\%cm = OpIMul %u32 %count %c1000
        \\%r0 = OpIAdd %u32 %cm %av
        \\%po0 = OpAccessChain %psu %out %c0 %g
        \\OpStore %po0 %r0
        \\%pv = OpAccessChain %psv %in %c1 %g %c1
        \\%v = OpLoad %v4u32 %pv
        \\%vz = OpCompositeExtract %u32 %v 2
        \\%i1 = OpIAdd %u32 %g %c4
        \\%po1 = OpAccessChain %psu %out %c0 %i1
        \\OpStore %po1 %vz
        \\%pv3 = OpAccessChain %psv %in3 %c1 %c1
        \\%q3 = OpLoad %v4u32 %pv3
        \\%w = OpVectorExtractDynamic %u32 %q3 %g
        \\%pn = OpAccessChain %psu %in3 %c0
        \\%n = OpLoad %u32 %pn
        \\%r2 = OpIAdd %u32 %w %n
        \\%i2 = OpIAdd %u32 %g %c8
        \\%po2 = OpAccessChain %psu %out %c0 %i2
        \\OpStore %po2 %r2
        \\OpReturn
        \\OpFunctionEnd
        ,
        .buffers = try a.dupe(Buffer, &.{
            .{ .index = 0, .words = b2.items },
            // count * 1000 + a, v.z, then B3.p.v[g] + B3.n.
            .{ .index = 1, .words = try zeros(a, 12), .expect = &.{ 7100, 7101, 7102, 7103, 2, 12, 22, 32, 1005, 2005, 3005, 4005 } },
            .{ .index = 2, .words = &.{ 5, 77, 1000, 2000, 3000, 4000 } },
        }),
    });

    // OpenCL `struct __attribute__((packed)) S { uchar a; uint b; }`: no
    // padding, 5-byte elements, unaligned loads (the review's t9). Packed
    // bytes 11 aa bb cc dd | 22 01 02 03 04: S[0].b = 0xddccbbaa,
    // S[1].b = 0x04030201.
    try list.append(a, .{ .name = "cpacked_struct", .threads = 2, .text =
        \\OpCapability Kernel
        \\OpCapability Addresses
        \\OpCapability Int64
        \\OpCapability Int8
        \\OpMemoryModel Physical64 OpenCL
        \\OpEntryPoint Kernel %k "cpacked_struct" %gid
        \\OpDecorate %gid BuiltIn GlobalInvocationId
        \\OpDecorate %S CPacked
        \\%void = OpTypeVoid
        \\%u8 = OpTypeInt 8 0
        \\%u32 = OpTypeInt 32 0
        \\%u64 = OpTypeInt 64 0
        \\%v3u64 = OpTypeVector %u64 3
        \\%c1 = OpConstant %u32 1
        \\%S = OpTypeStruct %u8 %u32
        \\%pS = OpTypePointer CrossWorkgroup %S
        \\%pcw = OpTypePointer CrossWorkgroup %u32
        \\%pin = OpTypePointer Input %v3u64
        \\%gid = OpVariable %pin Input
        \\%fn = OpTypeFunction %void %pcw %pS
        \\%k = OpFunction %void None %fn
        \\%dst = OpFunctionParameter %pcw
        \\%src = OpFunctionParameter %pS
        \\%e = OpLabel
        \\%gv = OpLoad %v3u64 %gid
        \\%g = OpCompositeExtract %u64 %gv 0
        \\%pm = OpInBoundsPtrAccessChain %pcw %src %g %c1
        \\%x = OpLoad %u32 %pm
        \\%pd = OpInBoundsPtrAccessChain %pcw %dst %g
        \\OpStore %pd %x Aligned 4
        \\OpReturn
        \\OpFunctionEnd
    , .buffers = try a.dupe(Buffer, &.{
        .{ .index = 0, .words = try zeros(a, 2), .expect = &.{ 0xddccbbaa, 0x04030201 } },
        .{ .index = 1, .words = &.{ 0xccbbaa11, 0x020122dd, 0x00000403 } },
    }) });

    // glslang's `layout(local_size_x_id = 0) in;`: `LocalSize 1 1 1` plus
    // a specialization-constant composite decorated BuiltIn WorkgroupSize,
    // which takes precedence (the review's finding). The manifest must say
    // 64, and a dispatch with it fills words 0..127 with 64; with groups of
    // 1 only every 64th word would be written.
    try list.append(a, .{ .name = "workgroup_size_constant", .threads = 128, .group_from_manifest = true, .text =
        \\OpCapability Shader
        \\OpMemoryModel Logical GLSL450
        \\OpEntryPoint GLCompute %k "workgroup_size_constant" %wid %lid
        \\OpExecutionMode %k LocalSize 1 1 1
        \\OpDecorate %wid BuiltIn WorkgroupId
        \\OpDecorate %lid BuiltIn LocalInvocationId
        \\OpDecorate %sx SpecId 0
        \\OpDecorate %wsz BuiltIn WorkgroupSize
        \\OpDecorate %rt ArrayStride 4
        \\OpDecorate %O Block
        \\OpMemberDecorate %O 0 Offset 0
        \\OpDecorate %out DescriptorSet 0
        \\OpDecorate %out Binding 0
        \\%void = OpTypeVoid
        \\%fn = OpTypeFunction %void
        \\%u32 = OpTypeInt 32 0
        \\%v3u32 = OpTypeVector %u32 3
        \\%c0 = OpConstant %u32 0
        \\%c1 = OpConstant %u32 1
        \\%sx = OpSpecConstant %u32 64
        \\%wsz = OpSpecConstantComposite %v3u32 %sx %c1 %c1
        \\%rt = OpTypeRuntimeArray %u32
        \\%O = OpTypeStruct %rt
        \\%pO = OpTypePointer StorageBuffer %O
        \\%pu = OpTypePointer StorageBuffer %u32
        \\%pin = OpTypePointer Input %v3u32
        \\%wid = OpVariable %pin Input
        \\%lid = OpVariable %pin Input
        \\%out = OpVariable %pO StorageBuffer
        \\%k = OpFunction %void None %fn
        \\%e = OpLabel
        \\%wv = OpLoad %v3u32 %wid
        \\%w = OpCompositeExtract %u32 %wv 0
        \\%lv = OpLoad %v3u32 %lid
        \\%l = OpCompositeExtract %u32 %lv 0
        \\%sz = OpCompositeExtract %u32 %wsz 0
        \\%base = OpIMul %u32 %w %sz
        \\%i = OpIAdd %u32 %base %l
        \\%po = OpAccessChain %pu %out %c0 %i
        \\OpStore %po %sz
        \\OpReturn
        \\OpFunctionEnd
    , .buffers = try a.dupe(Buffer, &.{
        .{ .index = 0, .words = try zeros(a, 128), .expect = try filled(a, 128, 64) },
    }) });
}

/// Zig's own SPIR-V for integer widenings (tools/spirv/testdata/
/// convert_vulkan.zig and convert_opencl.zig, compiled by the commands in
/// their headers; the text is tools/spirv/dis.zig's disassembly, which
/// re-assembles to the same bytes). The Generator word names Zig, so the
/// lowering's producer workaround is on: the Vulkan module's `OpSConvert
/// %i32 %u16` / `%u8` (Zig's bug for `@as(i32, u16)`) must zero-extend and
/// its `OpSConvert %i32 %i16` sign-extend; in the OpenCL module every
/// integer type is unsigned and `OpSConvert %u32 %u16` / `%u8` is a real
/// `i16`/`i8` sign extension that the workaround must leave alone.
fn appendZigConvertTests(a: Allocator, list: *std.ArrayList(Test)) !void {
    const inputs: []const u32 = &.{ 1, 0x7fff, 0x8000, 0xffff80f1 };
    try list.append(a, .{
        .name = "zig_convert_vulkan",
        .threads = 1,
        .text =
        \\; SPIR-V
        \\; Version: 1.0
        \\; Generator: 0x00290900
        \\; Bound: 176
        \\; Schema: 0
        \\OpCapability Shader
        \\OpCapability Int64
        \\OpCapability PhysicalStorageBufferAddresses
        \\OpCapability Int16
        \\OpCapability Int8
        \\OpExtension "SPV_KHR_physical_storage_buffer"
        \\OpMemoryModel PhysicalStorageBuffer64 GLSL450
        \\OpEntryPoint GLCompute %convert_vulkan.zig_convert_vulkan "zig_convert_vulkan"
        \\OpExecutionMode %convert_vulkan.zig_convert_vulkan LocalSize 1 1 1
        \\%12 = OpString "convert_vulkan.zig"
        \\OpSourceExtension "zig_errors:"
        \\OpSource Zig 2304
        \\OpName %void "void"
        \\OpName %u64 "u64"
        \\OpName %u32 "u32"
        \\OpName %u16 "u16"
        \\OpName %u8 "u8"
        \\OpName %i32 "i32"
        \\OpName %i16 "i16"
        \\OpName %convert_vulkan.zig_convert_vulkan "convert_vulkan.zig_convert_vulkan"
        \\OpName %i "i"
        \\OpName %x "x"
        \\OpName %lo "lo"
        \\OpName %byte "byte"
        \\OpName %in "in"
        \\OpName %out "out"
        \\OpDecorate %33 ArrayStride 4
        \\OpDecorate %60 ArrayStride 4
        \\OpDecorate %163 Block
        \\OpMemberDecorate %163 0 Offset 0
        \\OpDecorate %168 ArrayStride 16
        \\OpDecorate %in NonWritable
        \\OpDecorate %in Binding 0
        \\OpDecorate %in DescriptorSet 0
        \\OpDecorate %170 Block
        \\OpMemberDecorate %170 0 Offset 0
        \\OpDecorate %175 ArrayStride 48
        \\OpDecorate %out Binding 1
        \\OpDecorate %out DescriptorSet 0
        \\%void = OpTypeVoid
        \\%u64 = OpTypeInt 64 0
        \\%6 = OpTypePointer Function %u64
        \\%10 = OpConstant %u64 0
        \\%23 = OpConstant %u64 4
        \\%25 = OpTypeBool
        \\%u32 = OpTypeInt 32 0
        \\%32 = OpConstant %u32 4
        \\%33 = OpTypeArray %u32 %32
        \\%34 = OpTypePointer StorageBuffer %33
        \\%37 = OpConstant %u32 0
        \\%40 = OpTypePointer StorageBuffer %u32
        \\%u16 = OpTypeInt 16 0
        \\%u8 = OpTypeInt 8 0
        \\%53 = OpConstant %u64 3
        \\%i32 = OpTypeInt 32 1
        \\%59 = OpConstant %u32 12
        \\%60 = OpTypeArray %i32 %59
        \\%61 = OpTypePointer StorageBuffer %60
        \\%67 = OpTypePointer StorageBuffer %i32
        \\%81 = OpConstant %u64 1
        \\%i16 = OpTypeInt 16 1
        \\%111 = OpConstant %u64 2
        \\%131 = OpConstant %u32 6
        \\%134 = OpConstant %u32 3
        \\%159 = OpTypeFunction %void
        \\%163 = OpTypeStruct %33
        \\%168 = OpTypePointer StorageBuffer %163
        \\%in = OpVariable %168 StorageBuffer
        \\%170 = OpTypeStruct %60
        \\%175 = OpTypePointer StorageBuffer %170
        \\%out = OpVariable %175 StorageBuffer
        \\%convert_vulkan.zig_convert_vulkan = OpFunction %void None %159
        \\%4 = OpLabel
        \\%7 = OpVariable %6 Function
        \\OpStore %7 %10 Aligned 8
        \\OpLine %12 16 10
        \\OpBranch %14
        \\%14 = OpLabel
        \\OpLoopMerge %15 %16 None
        \\OpBranch %13
        \\%13 = OpLabel
        \\%i = OpLoad %u64 %7 Aligned 8
        \\%20 = OpBitcast %u64 %i
        \\%24 = OpULessThan %25 %20 %23
        \\OpSelectionMerge %28 None
        \\OpBranchConditional %24 %26 %27
        \\%26 = OpLabel
        \\OpLine %12 17 23
        \\%38 = OpInBoundsAccessChain %34 %in %37
        \\%41 = OpInBoundsAccessChain %40 %38 %i
        \\%x = OpLoad %u32 %41 Aligned 4
        \\OpLine %12 18 25
        \\%lo = OpUConvert %u16 %x
        \\OpLine %12 19 26
        \\%byte = OpUConvert %u8 %x
        \\OpLine %12 20 17
        \\%54 = OpIMul %u64 %53 %i
        \\OpLine %12 20 14
        \\%65 = OpInBoundsAccessChain %61 %out %37
        \\%68 = OpInBoundsAccessChain %67 %65 %54
        \\%69 = OpSConvert %i32 %lo
        \\OpStore %68 %69 Aligned 4
        \\OpLine %12 21 17
        \\%76 = OpIMul %u64 %53 %i
        \\OpLine %12 21 21
        \\%82 = OpIAdd %u64 %76 %81
        \\OpLine %12 21 14
        \\%92 = OpInBoundsAccessChain %61 %out %37
        \\%95 = OpInBoundsAccessChain %67 %92 %82
        \\OpLine %12 21 46
        \\%98 = OpBitcast %i16 %lo
        \\%99 = OpSConvert %i32 %98
        \\OpStore %95 %99 Aligned 4
        \\OpLine %12 22 17
        \\%106 = OpIMul %u64 %53 %i
        \\OpLine %12 22 21
        \\%112 = OpIAdd %u64 %106 %111
        \\OpLine %12 22 14
        \\%122 = OpInBoundsAccessChain %61 %out %37
        \\%125 = OpInBoundsAccessChain %67 %122 %112
        \\%126 = OpSConvert %i32 %byte
        \\OpStore %125 %126 Aligned 4
        \\OpLine %12 23 5
        \\OpBranch %132
        \\%132 = OpLabel
        \\OpBranch %28
        \\%27 = OpLabel
        \\OpBranch %135
        \\%135 = OpLabel
        \\OpBranch %28
        \\%28 = OpLabel
        \\%136 = OpPhi %u32 %131 %132 %134 %135
        \\OpBranch %138
        \\%138 = OpLabel
        \\%141 = OpIEqual %25 %136 %131
        \\OpBranchConditional %141 %143 %15
        \\%143 = OpLabel
        \\%147 = OpIAdd %u64 %i %81
        \\OpStore %7 %147 Aligned 8
        \\OpBranch %16
        \\%15 = OpLabel
        \\%150 = OpPhi %u32 %136 %138
        \\OpBranch %152
        \\%16 = OpLabel
        \\OpBranch %14
        \\%152 = OpLabel
        \\%155 = OpIEqual %25 %150 %134
        \\OpSelectionMerge %157 None
        \\OpBranchConditional %155 %158 %157
        \\%158 = OpLabel
        \\OpReturn
        \\%157 = OpLabel
        \\OpUnreachable
        \\OpFunctionEnd
        ,
        .buffers = try a.dupe(Buffer, &.{
            .{ .index = 0, .words = inputs },
            // Per input: u16 zero-extended, i16 sign-extended, u8 zero-extended.
            .{ .index = 1, .words = try zeros(a, 12), .expect = &.{ 1, 1, 1, 0x7fff, 0x7fff, 0xff, 0x8000, 0xffff8000, 0, 0x80f1, 0xffff80f1, 0xf1 } },
        }),
    });
    try list.append(a, .{
        .name = "zig_convert_opencl",
        .threads = 1,
        .text =
        \\; SPIR-V
        \\; Version: 1.0
        \\; Generator: 0x00290900
        \\; Bound: 178
        \\; Schema: 0
        \\OpCapability Kernel
        \\OpCapability Addresses
        \\OpCapability Int64
        \\OpCapability Int16
        \\OpCapability Int8
        \\OpMemoryModel Physical64 OpenCL
        \\OpEntryPoint Kernel %convert_opencl.zig_convert_opencl "zig_convert_opencl"
        \\OpExecutionMode %convert_opencl.zig_convert_opencl LocalSize 1 1 1
        \\%62 = OpString "convert_opencl.zig"
        \\OpSourceExtension "zig_errors:"
        \\OpSource Zig 2304
        \\OpName %void "void"
        \\OpName %u32 "u32"
        \\OpName %u64 "u64"
        \\OpName %u16 "u16"
        \\OpName %u8 "u8"
        \\OpName %convert_opencl.zig_convert_opencl "convert_opencl.zig_convert_opencl"
        \\OpName %i "i"
        \\OpName %x "x"
        \\OpName %lo "lo"
        \\OpName %byte "byte"
        \\%void = OpTypeVoid
        \\%u32 = OpTypeInt 32 0
        \\%6 = OpConstant %u32 4
        \\%7 = OpTypeArray %u32 %6
        \\%8 = OpTypePointer CrossWorkgroup %7
        \\%12 = OpConstant %u32 8
        \\%13 = OpTypeArray %u32 %12
        \\%14 = OpTypePointer CrossWorkgroup %13
        \\%22 = OpTypePointer Function %8
        \\%41 = OpTypePointer Function %14
        \\%u64 = OpTypeInt 64 0
        \\%56 = OpTypePointer Function %u64
        \\%60 = OpConstant %u64 0
        \\%73 = OpConstant %u64 4
        \\%75 = OpTypeBool
        \\%87 = OpTypePointer CrossWorkgroup %u32
        \\%u16 = OpTypeInt 16 0
        \\%u8 = OpTypeInt 8 0
        \\%106 = OpConstant %u64 2
        \\%131 = OpConstant %u64 1
        \\%149 = OpConstant %u32 14
        \\%152 = OpConstant %u32 11
        \\%177 = OpTypeFunction %void %8 %14
        \\%convert_opencl.zig_convert_opencl = OpFunction %void None %177
        \\%9 = OpFunctionParameter %8
        \\%15 = OpFunctionParameter %14
        \\%16 = OpLabel
        \\%23 = OpVariable %22 Function
        \\%42 = OpVariable %41 Function
        \\%57 = OpVariable %56 Function
        \\OpStore %23 %9 Aligned 8
        \\%35 = OpBitcast %22 %23
        \\OpStore %42 %15 Aligned 8
        \\%54 = OpBitcast %41 %42
        \\OpStore %57 %60 Aligned 8
        \\OpLine %62 11 10
        \\OpBranch %64
        \\%64 = OpLabel
        \\OpLoopMerge %65 %66 None
        \\OpBranch %63
        \\%63 = OpLabel
        \\%i = OpLoad %u64 %57 Aligned 8
        \\%70 = OpBitcast %u64 %i
        \\%74 = OpULessThan %75 %70 %73
        \\OpSelectionMerge %78 None
        \\OpBranchConditional %74 %76 %77
        \\%76 = OpLabel
        \\OpLine %62 12 21
        \\%85 = OpLoad %8 %35 Aligned 8
        \\%88 = OpInBoundsAccessChain %87 %85 %i
        \\%x = OpLoad %u32 %88 Aligned 4
        \\OpLine %62 13 43
        \\%92 = OpUConvert %u16 %x
        \\OpLine %62 13 25
        \\%lo = OpBitcast %u16 %92
        \\OpLine %62 14 43
        \\%98 = OpUConvert %u8 %x
        \\OpLine %62 14 26
        \\%byte = OpBitcast %u8 %98
        \\OpLine %62 15 15
        \\%107 = OpIMul %u64 %106 %i
        \\OpLine %62 15 12
        \\%115 = OpLoad %14 %54 Aligned 8
        \\%118 = OpInBoundsAccessChain %87 %115 %107
        \\%119 = OpSConvert %u32 %lo
        \\OpStore %118 %119 Aligned 4
        \\OpLine %62 16 15
        \\%126 = OpIMul %u64 %106 %i
        \\OpLine %62 16 19
        \\%132 = OpIAdd %u64 %126 %131
        \\OpLine %62 16 12
        \\%140 = OpLoad %14 %54 Aligned 8
        \\%143 = OpInBoundsAccessChain %87 %140 %132
        \\%144 = OpSConvert %u32 %byte
        \\OpStore %143 %144 Aligned 4
        \\OpLine %62 17 5
        \\OpBranch %150
        \\%150 = OpLabel
        \\OpBranch %78
        \\%77 = OpLabel
        \\OpBranch %153
        \\%153 = OpLabel
        \\OpBranch %78
        \\%78 = OpLabel
        \\%154 = OpPhi %u32 %149 %150 %152 %153
        \\OpBranch %156
        \\%156 = OpLabel
        \\%159 = OpIEqual %75 %154 %149
        \\OpBranchConditional %159 %161 %65
        \\%161 = OpLabel
        \\%165 = OpIAdd %u64 %i %131
        \\OpStore %57 %165 Aligned 8
        \\OpBranch %66
        \\%65 = OpLabel
        \\%168 = OpPhi %u32 %154 %156
        \\OpBranch %170
        \\%66 = OpLabel
        \\OpBranch %64
        \\%170 = OpLabel
        \\%173 = OpIEqual %75 %168 %152
        \\OpSelectionMerge %175 None
        \\OpBranchConditional %173 %176 %175
        \\%176 = OpLabel
        \\OpReturn
        \\%175 = OpLabel
        \\OpUnreachable
        \\OpFunctionEnd
        ,
        .buffers = try a.dupe(Buffer, &.{
            .{ .index = 0, .words = inputs },
            // Per input: i16 sign-extended, i8 sign-extended.
            .{ .index = 1, .words = try zeros(a, 8), .expect = &.{ 1, 1, 0x7fff, 0xffffffff, 0xffff8000, 0, 0xffff80f1, 0xfffffff1 } },
        }),
    });
    // Zig's i128 widenings (testdata/convert_wide.zig): an i16 or i32 source
    // goes into the u64 limb 0 by `OpUConvert %u64 %i16` and the upper limbs
    // are filled from limb 0's top bit, so the producer workaround must
    // sign-extend it; the u16 source and the `@intCast` stay zero extensions.
    try list.append(a, .{
        .name = "zig_convert_wide",
        .threads = 1,
        .text =
        \\; SPIR-V
        \\; Version: 1.0
        \\; Generator: 0x00290900
        \\; Bound: 938
        \\; Schema: 0
        \\OpCapability Shader
        \\OpCapability Int64
        \\OpCapability PhysicalStorageBufferAddresses
        \\OpCapability Int16
        \\OpCapability Int8
        \\OpExtension "SPV_KHR_physical_storage_buffer"
        \\OpMemoryModel PhysicalStorageBuffer64 GLSL450
        \\OpEntryPoint GLCompute %convert_wide.zig_convert_wide "zig_convert_wide"
        \\OpExecutionMode %convert_wide.zig_convert_wide LocalSize 1 1 1
        \\%12 = OpString "convert_wide.zig"
        \\OpSourceExtension "zig_errors:"
        \\OpSource Zig 2304
        \\OpName %void "void"
        \\OpName %u64 "u64"
        \\OpName %u32 "u32"
        \\OpName %u16 "u16"
        \\OpName %i16 "i16"
        \\OpName %i64 "i64"
        \\OpName %i32 "i32"
        \\OpName %u8 "u8"
        \\OpName %convert_wide.zig_convert_wide "convert_wide.zig_convert_wide"
        \\OpName %i "i"
        \\OpName %x "x"
        \\OpName %s "s"
        \\OpName %u "u"
        \\OpName %w "w"
        \\OpName %v "v"
        \\OpName %d "d"
        \\OpName %n "n"
        \\OpName %nu "nu"
        \\OpName %in "in"
        \\OpName %out "out"
        \\OpDecorate %33 ArrayStride 4
        \\OpDecorate %133 ArrayStride 4
        \\OpDecorate %925 Block
        \\OpMemberDecorate %925 0 Offset 0
        \\OpDecorate %930 ArrayStride 16
        \\OpDecorate %in NonWritable
        \\OpDecorate %in Binding 0
        \\OpDecorate %in DescriptorSet 0
        \\OpDecorate %932 Block
        \\OpMemberDecorate %932 0 Offset 0
        \\OpDecorate %937 ArrayStride 128
        \\OpDecorate %out Binding 1
        \\OpDecorate %out DescriptorSet 0
        \\%void = OpTypeVoid
        \\%u64 = OpTypeInt 64 0
        \\%6 = OpTypePointer Function %u64
        \\%10 = OpConstant %u64 0
        \\%23 = OpConstant %u64 4
        \\%25 = OpTypeBool
        \\%u32 = OpTypeInt 32 0
        \\%32 = OpConstant %u32 4
        \\%33 = OpTypeArray %u32 %32
        \\%34 = OpTypePointer StorageBuffer %33
        \\%37 = OpConstant %u32 0
        \\%40 = OpTypePointer StorageBuffer %u32
        \\%u16 = OpTypeInt 16 0
        \\%i16 = OpTypeInt 16 1
        \\%i64 = OpTypeInt 64 1
        \\%60 = OpConstant %i64 63
        \\%65 = OpConstant %u32 2
        \\%66 = OpTypeArray %u64 %65
        \\%i32 = OpTypeInt 32 1
        \\%114 = OpConstant %u16 32767
        \\%126 = OpConstant %u64 8
        \\%132 = OpConstant %u32 32
        \\%133 = OpTypeArray %u32 %132
        \\%134 = OpTypePointer StorageBuffer %133
        \\%u8 = OpTypeInt 8 0
        \\%145 = OpConstant %u8 64
        \\%156 = OpConstant %u64 6
        \\%158 = OpConstant %u64 63
        \\%160 = OpConstant %u64 64
        \\%179 = OpConstant %u64 1
        \\%282 = OpConstant %u8 96
        \\%398 = OpConstant %u64 2
        \\%535 = OpConstant %u64 3
        \\%702 = OpConstant %u64 5
        \\%748 = OpConstant %u8 32
        \\%766 = OpConstant %u64 7
        \\%893 = OpConstant %u32 6
        \\%896 = OpConstant %u32 3
        \\%921 = OpTypeFunction %void
        \\%925 = OpTypeStruct %33
        \\%930 = OpTypePointer StorageBuffer %925
        \\%in = OpVariable %930 StorageBuffer
        \\%932 = OpTypeStruct %133
        \\%937 = OpTypePointer StorageBuffer %932
        \\%out = OpVariable %937 StorageBuffer
        \\%convert_wide.zig_convert_wide = OpFunction %void None %921
        \\%4 = OpLabel
        \\%7 = OpVariable %6 Function
        \\OpStore %7 %10 Aligned 8
        \\OpLine %12 24 10
        \\OpBranch %14
        \\%14 = OpLabel
        \\OpLoopMerge %15 %16 None
        \\OpBranch %13
        \\%13 = OpLabel
        \\%i = OpLoad %u64 %7 Aligned 8
        \\%20 = OpBitcast %u64 %i
        \\%24 = OpULessThan %25 %20 %23
        \\OpSelectionMerge %28 None
        \\OpBranchConditional %24 %26 %27
        \\%26 = OpLabel
        \\OpLine %12 25 23
        \\%38 = OpInBoundsAccessChain %34 %in %37
        \\%41 = OpInBoundsAccessChain %40 %38 %i
        \\%x = OpLoad %u32 %41 Aligned 4
        \\OpLine %12 26 42
        \\%45 = OpUConvert %u16 %x
        \\OpLine %12 26 24
        \\%s = OpBitcast %i16 %45
        \\OpLine %12 27 24
        \\%u = OpUConvert %u16 %x
        \\OpLine %12 28 9
        \\%55 = OpUConvert %u64 %s
        \\%58 = OpBitcast %i64 %55
        \\%61 = OpShiftRightArithmetic %i64 %58 %60
        \\%62 = OpBitcast %u64 %61
        \\%w = OpCompositeConstruct %66 %55 %62
        \\OpLine %12 28 25
        \\OpLine %12 29 9
        \\%75 = OpUConvert %u64 %u
        \\%v = OpCompositeConstruct %66 %75 %10
        \\OpLine %12 29 25
        \\OpLine %12 30 53
        \\%91 = OpBitcast %i32 %x
        \\%93 = OpUConvert %u64 %91
        \\%96 = OpBitcast %i64 %93
        \\%99 = OpShiftRightArithmetic %i64 %96 %60
        \\%100 = OpBitcast %u64 %99
        \\%d = OpCompositeConstruct %66 %93 %100
        \\OpLine %12 30 25
        \\OpLine %12 31 9
        \\%115 = OpBitwiseAnd %u16 %u %114
        \\OpLine %12 31 24
        \\%n = OpBitcast %i16 %115
        \\OpLine %12 32 25
        \\%nu = OpUConvert %u64 %n
        \\OpLine %12 33 17
        \\%127 = OpIMul %u64 %126 %i
        \\OpLine %12 33 14
        \\%138 = OpInBoundsAccessChain %134 %out %37
        \\%141 = OpInBoundsAccessChain %40 %138 %127
        \\OpLine %12 33 36
        \\%146 = OpUConvert %u64 %145
        \\%149 = OpCompositeExtract %u64 %w 0
        \\%150 = OpCompositeExtract %u64 %w 1
        \\%162 = OpShiftRightLogical %u64 %146 %156
        \\%164 = OpBitwiseAnd %u64 %146 %158
        \\%166 = OpISub %u64 %160 %164
        \\%167 = OpIEqual %25 %164 %10
        \\%173 = OpIAdd %u64 %10 %162
        \\%174 = OpIEqual %25 %10 %173
        \\%176 = OpShiftRightLogical %u64 %149 %164
        \\%177 = OpSelect %u64 %174 %176 %10
        \\%181 = OpIAdd %u64 %173 %179
        \\%182 = OpIEqual %25 %10 %181
        \\%184 = OpShiftLeftLogical %u64 %149 %166
        \\%185 = OpSelect %u64 %167 %10 %184
        \\%186 = OpSelect %u64 %182 %185 %10
        \\%190 = OpIAdd %u64 %10 %162
        \\%191 = OpIEqual %25 %179 %190
        \\%193 = OpShiftRightLogical %u64 %150 %164
        \\%194 = OpSelect %u64 %191 %193 %177
        \\%198 = OpIAdd %u64 %190 %179
        \\%199 = OpIEqual %25 %179 %198
        \\%201 = OpShiftLeftLogical %u64 %150 %166
        \\%202 = OpSelect %u64 %167 %10 %201
        \\%203 = OpSelect %u64 %199 %202 %186
        \\%205 = OpBitwiseOr %u64 %194 %203
        \\%211 = OpIAdd %u64 %179 %162
        \\%212 = OpIEqual %25 %10 %211
        \\%214 = OpShiftRightLogical %u64 %149 %164
        \\%215 = OpSelect %u64 %212 %214 %10
        \\%219 = OpIAdd %u64 %211 %179
        \\%220 = OpIEqual %25 %10 %219
        \\%222 = OpShiftLeftLogical %u64 %149 %166
        \\%223 = OpSelect %u64 %167 %10 %222
        \\%224 = OpSelect %u64 %220 %223 %10
        \\%228 = OpIAdd %u64 %179 %162
        \\%229 = OpIEqual %25 %179 %228
        \\%231 = OpShiftRightLogical %u64 %150 %164
        \\%232 = OpSelect %u64 %229 %231 %215
        \\%236 = OpIAdd %u64 %228 %179
        \\%237 = OpIEqual %25 %179 %236
        \\%239 = OpShiftLeftLogical %u64 %150 %166
        \\%240 = OpSelect %u64 %167 %10 %239
        \\%241 = OpSelect %u64 %237 %240 %224
        \\%243 = OpBitwiseOr %u64 %232 %241
        \\%248 = OpCompositeConstruct %66 %205 %243
        \\OpLine %12 33 24
        \\%251 = OpCompositeExtract %u64 %248 0
        \\%252 = OpUConvert %u32 %251
        \\OpStore %141 %252 Aligned 4
        \\OpLine %12 34 17
        \\%259 = OpIMul %u64 %126 %i
        \\OpLine %12 34 21
        \\%265 = OpIAdd %u64 %259 %179
        \\OpLine %12 34 14
        \\%275 = OpInBoundsAccessChain %134 %out %37
        \\%278 = OpInBoundsAccessChain %40 %275 %265
        \\OpLine %12 34 40
        \\%283 = OpUConvert %u64 %282
        \\%296 = OpShiftRightLogical %u64 %283 %156
        \\%298 = OpBitwiseAnd %u64 %283 %158
        \\%300 = OpISub %u64 %160 %298
        \\%301 = OpIEqual %25 %298 %10
        \\%307 = OpIAdd %u64 %10 %296
        \\%308 = OpIEqual %25 %10 %307
        \\%310 = OpShiftRightLogical %u64 %149 %298
        \\%311 = OpSelect %u64 %308 %310 %10
        \\%315 = OpIAdd %u64 %307 %179
        \\%316 = OpIEqual %25 %10 %315
        \\%318 = OpShiftLeftLogical %u64 %149 %300
        \\%319 = OpSelect %u64 %301 %10 %318
        \\%320 = OpSelect %u64 %316 %319 %10
        \\%324 = OpIAdd %u64 %10 %296
        \\%325 = OpIEqual %25 %179 %324
        \\%327 = OpShiftRightLogical %u64 %150 %298
        \\%328 = OpSelect %u64 %325 %327 %311
        \\%332 = OpIAdd %u64 %324 %179
        \\%333 = OpIEqual %25 %179 %332
        \\%335 = OpShiftLeftLogical %u64 %150 %300
        \\%336 = OpSelect %u64 %301 %10 %335
        \\%337 = OpSelect %u64 %333 %336 %320
        \\%339 = OpBitwiseOr %u64 %328 %337
        \\%345 = OpIAdd %u64 %179 %296
        \\%346 = OpIEqual %25 %10 %345
        \\%348 = OpShiftRightLogical %u64 %149 %298
        \\%349 = OpSelect %u64 %346 %348 %10
        \\%353 = OpIAdd %u64 %345 %179
        \\%354 = OpIEqual %25 %10 %353
        \\%356 = OpShiftLeftLogical %u64 %149 %300
        \\%357 = OpSelect %u64 %301 %10 %356
        \\%358 = OpSelect %u64 %354 %357 %10
        \\%362 = OpIAdd %u64 %179 %296
        \\%363 = OpIEqual %25 %179 %362
        \\%365 = OpShiftRightLogical %u64 %150 %298
        \\%366 = OpSelect %u64 %363 %365 %349
        \\%370 = OpIAdd %u64 %362 %179
        \\%371 = OpIEqual %25 %179 %370
        \\%373 = OpShiftLeftLogical %u64 %150 %300
        \\%374 = OpSelect %u64 %301 %10 %373
        \\%375 = OpSelect %u64 %371 %374 %358
        \\%377 = OpBitwiseOr %u64 %366 %375
        \\%382 = OpCompositeConstruct %66 %339 %377
        \\OpLine %12 34 28
        \\%385 = OpCompositeExtract %u64 %382 0
        \\%386 = OpUConvert %u32 %385
        \\OpStore %278 %386 Aligned 4
        \\OpLine %12 35 17
        \\%393 = OpIMul %u64 %126 %i
        \\OpLine %12 35 21
        \\%399 = OpIAdd %u64 %393 %398
        \\OpLine %12 35 14
        \\%409 = OpInBoundsAccessChain %134 %out %37
        \\%412 = OpInBoundsAccessChain %40 %409 %399
        \\OpLine %12 35 40
        \\%417 = OpUConvert %u64 %145
        \\%420 = OpCompositeExtract %u64 %v 0
        \\%421 = OpCompositeExtract %u64 %v 1
        \\%433 = OpShiftRightLogical %u64 %417 %156
        \\%435 = OpBitwiseAnd %u64 %417 %158
        \\%437 = OpISub %u64 %160 %435
        \\%438 = OpIEqual %25 %435 %10
        \\%444 = OpIAdd %u64 %10 %433
        \\%445 = OpIEqual %25 %10 %444
        \\%447 = OpShiftRightLogical %u64 %420 %435
        \\%448 = OpSelect %u64 %445 %447 %10
        \\%452 = OpIAdd %u64 %444 %179
        \\%453 = OpIEqual %25 %10 %452
        \\%455 = OpShiftLeftLogical %u64 %420 %437
        \\%456 = OpSelect %u64 %438 %10 %455
        \\%457 = OpSelect %u64 %453 %456 %10
        \\%461 = OpIAdd %u64 %10 %433
        \\%462 = OpIEqual %25 %179 %461
        \\%464 = OpShiftRightLogical %u64 %421 %435
        \\%465 = OpSelect %u64 %462 %464 %448
        \\%469 = OpIAdd %u64 %461 %179
        \\%470 = OpIEqual %25 %179 %469
        \\%472 = OpShiftLeftLogical %u64 %421 %437
        \\%473 = OpSelect %u64 %438 %10 %472
        \\%474 = OpSelect %u64 %470 %473 %457
        \\%476 = OpBitwiseOr %u64 %465 %474
        \\%482 = OpIAdd %u64 %179 %433
        \\%483 = OpIEqual %25 %10 %482
        \\%485 = OpShiftRightLogical %u64 %420 %435
        \\%486 = OpSelect %u64 %483 %485 %10
        \\%490 = OpIAdd %u64 %482 %179
        \\%491 = OpIEqual %25 %10 %490
        \\%493 = OpShiftLeftLogical %u64 %420 %437
        \\%494 = OpSelect %u64 %438 %10 %493
        \\%495 = OpSelect %u64 %491 %494 %10
        \\%499 = OpIAdd %u64 %179 %433
        \\%500 = OpIEqual %25 %179 %499
        \\%502 = OpShiftRightLogical %u64 %421 %435
        \\%503 = OpSelect %u64 %500 %502 %486
        \\%507 = OpIAdd %u64 %499 %179
        \\%508 = OpIEqual %25 %179 %507
        \\%510 = OpShiftLeftLogical %u64 %421 %437
        \\%511 = OpSelect %u64 %438 %10 %510
        \\%512 = OpSelect %u64 %508 %511 %495
        \\%514 = OpBitwiseOr %u64 %503 %512
        \\%519 = OpCompositeConstruct %66 %476 %514
        \\OpLine %12 35 28
        \\%522 = OpCompositeExtract %u64 %519 0
        \\%523 = OpUConvert %u32 %522
        \\OpStore %412 %523 Aligned 4
        \\OpLine %12 36 17
        \\%530 = OpIMul %u64 %126 %i
        \\OpLine %12 36 21
        \\%536 = OpIAdd %u64 %530 %535
        \\OpLine %12 36 14
        \\%546 = OpInBoundsAccessChain %134 %out %37
        \\%549 = OpInBoundsAccessChain %40 %546 %536
        \\OpLine %12 36 28
        \\%552 = OpCompositeExtract %u64 %w 0
        \\%553 = OpUConvert %u32 %552
        \\OpStore %549 %553 Aligned 4
        \\OpLine %12 37 17
        \\%560 = OpIMul %u64 %126 %i
        \\OpLine %12 37 21
        \\%566 = OpIAdd %u64 %560 %23
        \\OpLine %12 37 14
        \\%576 = OpInBoundsAccessChain %134 %out %37
        \\%579 = OpInBoundsAccessChain %40 %576 %566
        \\OpLine %12 37 40
        \\%584 = OpUConvert %u64 %145
        \\%587 = OpCompositeExtract %u64 %d 0
        \\%588 = OpCompositeExtract %u64 %d 1
        \\%600 = OpShiftRightLogical %u64 %584 %156
        \\%602 = OpBitwiseAnd %u64 %584 %158
        \\%604 = OpISub %u64 %160 %602
        \\%605 = OpIEqual %25 %602 %10
        \\%611 = OpIAdd %u64 %10 %600
        \\%612 = OpIEqual %25 %10 %611
        \\%614 = OpShiftRightLogical %u64 %587 %602
        \\%615 = OpSelect %u64 %612 %614 %10
        \\%619 = OpIAdd %u64 %611 %179
        \\%620 = OpIEqual %25 %10 %619
        \\%622 = OpShiftLeftLogical %u64 %587 %604
        \\%623 = OpSelect %u64 %605 %10 %622
        \\%624 = OpSelect %u64 %620 %623 %10
        \\%628 = OpIAdd %u64 %10 %600
        \\%629 = OpIEqual %25 %179 %628
        \\%631 = OpShiftRightLogical %u64 %588 %602
        \\%632 = OpSelect %u64 %629 %631 %615
        \\%636 = OpIAdd %u64 %628 %179
        \\%637 = OpIEqual %25 %179 %636
        \\%639 = OpShiftLeftLogical %u64 %588 %604
        \\%640 = OpSelect %u64 %605 %10 %639
        \\%641 = OpSelect %u64 %637 %640 %624
        \\%643 = OpBitwiseOr %u64 %632 %641
        \\%649 = OpIAdd %u64 %179 %600
        \\%650 = OpIEqual %25 %10 %649
        \\%652 = OpShiftRightLogical %u64 %587 %602
        \\%653 = OpSelect %u64 %650 %652 %10
        \\%657 = OpIAdd %u64 %649 %179
        \\%658 = OpIEqual %25 %10 %657
        \\%660 = OpShiftLeftLogical %u64 %587 %604
        \\%661 = OpSelect %u64 %605 %10 %660
        \\%662 = OpSelect %u64 %658 %661 %10
        \\%666 = OpIAdd %u64 %179 %600
        \\%667 = OpIEqual %25 %179 %666
        \\%669 = OpShiftRightLogical %u64 %588 %602
        \\%670 = OpSelect %u64 %667 %669 %653
        \\%674 = OpIAdd %u64 %666 %179
        \\%675 = OpIEqual %25 %179 %674
        \\%677 = OpShiftLeftLogical %u64 %588 %604
        \\%678 = OpSelect %u64 %605 %10 %677
        \\%679 = OpSelect %u64 %675 %678 %662
        \\%681 = OpBitwiseOr %u64 %670 %679
        \\%686 = OpCompositeConstruct %66 %643 %681
        \\OpLine %12 37 28
        \\%689 = OpCompositeExtract %u64 %686 0
        \\%690 = OpUConvert %u32 %689
        \\OpStore %579 %690 Aligned 4
        \\OpLine %12 38 17
        \\%697 = OpIMul %u64 %126 %i
        \\OpLine %12 38 21
        \\%703 = OpIAdd %u64 %697 %702
        \\OpLine %12 38 14
        \\%713 = OpInBoundsAccessChain %134 %out %37
        \\%716 = OpInBoundsAccessChain %40 %713 %703
        \\OpLine %12 38 28
        \\%718 = OpUConvert %u32 %nu
        \\OpStore %716 %718 Aligned 4
        \\OpLine %12 39 17
        \\%725 = OpIMul %u64 %126 %i
        \\OpLine %12 39 21
        \\%731 = OpIAdd %u64 %725 %156
        \\OpLine %12 39 14
        \\%741 = OpInBoundsAccessChain %134 %out %37
        \\%744 = OpInBoundsAccessChain %40 %741 %731
        \\OpLine %12 39 41
        \\%749 = OpUConvert %u64 %748
        \\%751 = OpShiftRightLogical %u64 %nu %749
        \\OpLine %12 39 28
        \\%754 = OpUConvert %u32 %751
        \\OpStore %744 %754 Aligned 4
        \\OpLine %12 40 17
        \\%761 = OpIMul %u64 %126 %i
        \\OpLine %12 40 21
        \\%767 = OpIAdd %u64 %761 %766
        \\OpLine %12 40 14
        \\%777 = OpInBoundsAccessChain %134 %out %37
        \\%780 = OpInBoundsAccessChain %40 %777 %767
        \\OpLine %12 40 40
        \\%785 = OpUConvert %u64 %748
        \\%798 = OpShiftRightLogical %u64 %785 %156
        \\%800 = OpBitwiseAnd %u64 %785 %158
        \\%802 = OpISub %u64 %160 %800
        \\%803 = OpIEqual %25 %800 %10
        \\%809 = OpIAdd %u64 %10 %798
        \\%810 = OpIEqual %25 %10 %809
        \\%812 = OpShiftRightLogical %u64 %587 %800
        \\%813 = OpSelect %u64 %810 %812 %10
        \\%817 = OpIAdd %u64 %809 %179
        \\%818 = OpIEqual %25 %10 %817
        \\%820 = OpShiftLeftLogical %u64 %587 %802
        \\%821 = OpSelect %u64 %803 %10 %820
        \\%822 = OpSelect %u64 %818 %821 %10
        \\%826 = OpIAdd %u64 %10 %798
        \\%827 = OpIEqual %25 %179 %826
        \\%829 = OpShiftRightLogical %u64 %588 %800
        \\%830 = OpSelect %u64 %827 %829 %813
        \\%834 = OpIAdd %u64 %826 %179
        \\%835 = OpIEqual %25 %179 %834
        \\%837 = OpShiftLeftLogical %u64 %588 %802
        \\%838 = OpSelect %u64 %803 %10 %837
        \\%839 = OpSelect %u64 %835 %838 %822
        \\%841 = OpBitwiseOr %u64 %830 %839
        \\%847 = OpIAdd %u64 %179 %798
        \\%848 = OpIEqual %25 %10 %847
        \\%850 = OpShiftRightLogical %u64 %587 %800
        \\%851 = OpSelect %u64 %848 %850 %10
        \\%855 = OpIAdd %u64 %847 %179
        \\%856 = OpIEqual %25 %10 %855
        \\%858 = OpShiftLeftLogical %u64 %587 %802
        \\%859 = OpSelect %u64 %803 %10 %858
        \\%860 = OpSelect %u64 %856 %859 %10
        \\%864 = OpIAdd %u64 %179 %798
        \\%865 = OpIEqual %25 %179 %864
        \\%867 = OpShiftRightLogical %u64 %588 %800
        \\%868 = OpSelect %u64 %865 %867 %851
        \\%872 = OpIAdd %u64 %864 %179
        \\%873 = OpIEqual %25 %179 %872
        \\%875 = OpShiftLeftLogical %u64 %588 %802
        \\%876 = OpSelect %u64 %803 %10 %875
        \\%877 = OpSelect %u64 %873 %876 %860
        \\%879 = OpBitwiseOr %u64 %868 %877
        \\%884 = OpCompositeConstruct %66 %841 %879
        \\OpLine %12 40 28
        \\%887 = OpCompositeExtract %u64 %884 0
        \\%888 = OpUConvert %u32 %887
        \\OpStore %780 %888 Aligned 4
        \\OpLine %12 41 5
        \\OpBranch %894
        \\%894 = OpLabel
        \\OpBranch %28
        \\%27 = OpLabel
        \\OpBranch %897
        \\%897 = OpLabel
        \\OpBranch %28
        \\%28 = OpLabel
        \\%898 = OpPhi %u32 %893 %894 %896 %897
        \\OpBranch %900
        \\%900 = OpLabel
        \\%903 = OpIEqual %25 %898 %893
        \\OpBranchConditional %903 %905 %15
        \\%905 = OpLabel
        \\%909 = OpIAdd %u64 %i %179
        \\OpStore %7 %909 Aligned 8
        \\OpBranch %16
        \\%15 = OpLabel
        \\%912 = OpPhi %u32 %898 %900
        \\OpBranch %914
        \\%16 = OpLabel
        \\OpBranch %14
        \\%914 = OpLabel
        \\%917 = OpIEqual %25 %912 %896
        \\OpSelectionMerge %919 None
        \\OpBranchConditional %917 %920 %919
        \\%920 = OpLabel
        \\OpReturn
        \\%919 = OpLabel
        \\OpUnreachable
        \\OpFunctionEnd
        ,
        .buffers = try a.dupe(Buffer, &.{
            .{ .index = 0, .words = inputs },
            // Per input: i128(i16) bits 64-95, 96-127; i128(u16) bits 64-95;
            // i128(i16) bits 0-31; i128(i32) bits 64-95; @intCast(i16 & 0x7fff)
            // to u64, bits 0-31 and 32-63; i128(i32) bits 32-63.
            .{ .index = 1, .words = try zeros(a, 32), .expect = &.{
                0,          0,          0, 1,          0,          1,      0, 0,
                0,          0,          0, 0x7fff,     0,          0x7fff, 0, 0,
                0xffffffff, 0xffffffff, 0, 0xffff8000, 0,          0,      0, 0,
                0xffffffff, 0xffffffff, 0, 0xffff80f1, 0xffffffff, 0xf1,   0, 0xffffffff,
            } },
        }),
    });
}

/// Barriers between two branches on the same per-thread condition, every
/// one in uniform control flow (glslang's workgroup-counter and shared-flag
/// shapes; review findings "barriers emitted without convergent"). The
/// SPIR-V path hands Metal unoptimized IR, so Metal's own optimizer sees
/// `if (lid == 0) ...; barrier; ...; barrier; if (lid == 0) ...` and, unless
/// the barrier is declared `convergent`, jump-threads the block between the
/// two branches into both arms: the lanes of one SIMD-group then run the
/// arms one after the other and reach different copies of the barriers.
/// Measured on an M3 before the assembler marked the barrier convergent:
/// barrier_workgroup_counter's 16 finals were 0x40000021 (33 of 64
/// increments counted), barrier_shared_flag's 256 lid < 16 words read 0.
fn appendBarrierTests(a: Allocator, list: *std.ArrayList(Test)) !void {
    // shared uint cell; if (lid == 0) atomicStore(cell, 0x40000000);
    // barrier(); out[gid] = atomicAdd(cell, 1); barrier();
    // if (lid == 0) out[2048 + wid] = atomicLoad(cell);
    // Every final must be 0x40000000 + 64.
    try list.append(a, .{ .name = "barrier_workgroup_counter", .threads = 1024, .group_from_manifest = true, .text =
        \\OpCapability Shader
        \\OpMemoryModel Logical GLSL450
        \\OpEntryPoint GLCompute %k "barrier_workgroup_counter" %gid %lid %wid
        \\OpExecutionMode %k LocalSize 64 1 1
        \\OpDecorate %gid BuiltIn GlobalInvocationId
        \\OpDecorate %lid BuiltIn LocalInvocationId
        \\OpDecorate %wid BuiltIn WorkgroupId
        \\OpDecorate %arr ArrayStride 4
        \\OpDecorate %Buf Block
        \\OpMemberDecorate %Buf 0 Offset 0
        \\OpDecorate %out DescriptorSet 0
        \\OpDecorate %out Binding 2
        \\%void = OpTypeVoid
        \\%bool = OpTypeBool
        \\%u32 = OpTypeInt 32 0
        \\%v3u32 = OpTypeVector %u32 3
        \\%fn = OpTypeFunction %void
        \\%c0 = OpConstant %u32 0
        \\%c2048 = OpConstant %u32 2048
        \\%c4096 = OpConstant %u32 4096
        \\%init = OpConstant %u32 1073741824
        \\%scope_wg = OpConstant %u32 2
        \\%sem_rlx = OpConstant %u32 0
        \\%barsem = OpConstant %u32 264
        \\%arr = OpTypeArray %u32 %c4096
        \\%Buf = OpTypeStruct %arr
        \\%pBuf = OpTypePointer StorageBuffer %Buf
        \\%pu = OpTypePointer StorageBuffer %u32
        \\%pwu = OpTypePointer Workgroup %u32
        \\%pin = OpTypePointer Input %v3u32
        \\%gid = OpVariable %pin Input
        \\%lid = OpVariable %pin Input
        \\%wid = OpVariable %pin Input
        \\%out = OpVariable %pBuf StorageBuffer
        \\%cell = OpVariable %pwu Workgroup
        \\%k = OpFunction %void None %fn
        \\%e = OpLabel
        \\%gv = OpLoad %v3u32 %gid
        \\%g = OpCompositeExtract %u32 %gv 0
        \\%lv = OpLoad %v3u32 %lid
        \\%l = OpCompositeExtract %u32 %lv 0
        \\%wv = OpLoad %v3u32 %wid
        \\%w = OpCompositeExtract %u32 %wv 0
        \\%is0 = OpIEqual %bool %l %c0
        \\OpSelectionMerge %j None
        \\OpBranchConditional %is0 %st %j
        \\%st = OpLabel
        \\OpAtomicStore %cell %scope_wg %sem_rlx %init
        \\OpBranch %j
        \\%j = OpLabel
        \\OpControlBarrier %scope_wg %scope_wg %barsem
        \\%old = OpAtomicIIncrement %u32 %cell %scope_wg %sem_rlx
        \\%po = OpAccessChain %pu %out %c0 %g
        \\OpStore %po %old
        \\OpControlBarrier %scope_wg %scope_wg %barsem
        \\OpSelectionMerge %j2 None
        \\OpBranchConditional %is0 %ld %j2
        \\%ld = OpLabel
        \\%f = OpAtomicLoad %u32 %cell %scope_wg %sem_rlx
        \\%wi = OpIAdd %u32 %w %c2048
        \\%pf = OpAccessChain %pu %out %c0 %wi
        \\OpStore %pf %f
        \\OpBranch %j2
        \\%j2 = OpLabel
        \\OpReturn
        \\OpFunctionEnd
    , .buffers = try a.dupe(Buffer, &.{
        .{ .index = 2, .words = try zeros(a, 4096), .expect = try filled(a, 16, 0x40000040), .expect_at = 2048 },
    }) });

    // shared uint flag[64], tile[64]; if (lid < 16) flag[lid] = 1;
    // barrier(); tile[lid] = in[gid]; barrier();
    // if (lid < 16) out[gid] = tile[lid + 16];
    try list.append(a, .{ .name = "barrier_shared_flag", .threads = 1024, .group_from_manifest = true, .text =
        \\OpCapability Shader
        \\OpMemoryModel Logical GLSL450
        \\OpEntryPoint GLCompute %main "barrier_shared_flag" %gid_var %lid_var
        \\OpExecutionMode %main LocalSize 64 1 1
        \\OpDecorate %gid_var BuiltIn GlobalInvocationId
        \\OpDecorate %lid_var BuiltIn LocalInvocationId
        \\OpDecorate %rta ArrayStride 4
        \\OpMemberDecorate %Out 0 Offset 0
        \\OpDecorate %Out Block
        \\OpDecorate %outb DescriptorSet 0
        \\OpDecorate %outb Binding 1
        \\OpMemberDecorate %In 0 Offset 0
        \\OpMemberDecorate %In 0 NonWritable
        \\OpDecorate %In Block
        \\OpDecorate %inb DescriptorSet 0
        \\OpDecorate %inb Binding 0
        \\%void = OpTypeVoid
        \\%fnty = OpTypeFunction %void
        \\%bool = OpTypeBool
        \\%uint = OpTypeInt 32 0
        \\%int = OpTypeInt 32 1
        \\%v3uint = OpTypeVector %uint 3
        \\%uint_0 = OpConstant %uint 0
        \\%uint_1 = OpConstant %uint 1
        \\%uint_2 = OpConstant %uint 2
        \\%int_0 = OpConstant %int 0
        \\%uint_16 = OpConstant %uint 16
        \\%uint_64 = OpConstant %uint 64
        \\%uint_264 = OpConstant %uint 264
        \\%_ptr_Input_v3uint = OpTypePointer Input %v3uint
        \\%_ptr_Input_uint = OpTypePointer Input %uint
        \\%gid_var = OpVariable %_ptr_Input_v3uint Input
        \\%lid_var = OpVariable %_ptr_Input_v3uint Input
        \\%rta = OpTypeRuntimeArray %uint
        \\%Out = OpTypeStruct %rta
        \\%In = OpTypeStruct %rta
        \\%_ptr_SB_Out = OpTypePointer StorageBuffer %Out
        \\%_ptr_SB_In = OpTypePointer StorageBuffer %In
        \\%_ptr_SB_uint = OpTypePointer StorageBuffer %uint
        \\%outb = OpVariable %_ptr_SB_Out StorageBuffer
        \\%inb = OpVariable %_ptr_SB_In StorageBuffer
        \\%arr = OpTypeArray %uint %uint_64
        \\%_ptr_WG_arr = OpTypePointer Workgroup %arr
        \\%_ptr_WG_uint = OpTypePointer Workgroup %uint
        \\%tile = OpVariable %_ptr_WG_arr Workgroup
        \\%flag = OpVariable %_ptr_WG_arr Workgroup
        \\%main = OpFunction %void None %fnty
        \\%entry = OpLabel
        \\%gidp = OpAccessChain %_ptr_Input_uint %gid_var %uint_0
        \\%gid = OpLoad %uint %gidp
        \\%inp = OpAccessChain %_ptr_SB_uint %inb %int_0 %gid
        \\%x = OpLoad %uint %inp
        \\%outp = OpAccessChain %_ptr_SB_uint %outb %int_0 %gid
        \\%lp = OpAccessChain %_ptr_Input_uint %lid_var %uint_0
        \\%lid = OpLoad %uint %lp
        \\%small = OpULessThan %bool %lid %uint_16
        \\OpSelectionMerge %j0 None
        \\OpBranchConditional %small %t0 %j0
        \\%t0 = OpLabel
        \\%fp = OpAccessChain %_ptr_WG_uint %flag %lid
        \\OpStore %fp %uint_1
        \\OpBranch %j0
        \\%j0 = OpLabel
        \\OpControlBarrier %uint_2 %uint_2 %uint_264
        \\%tp = OpAccessChain %_ptr_WG_uint %tile %lid
        \\OpStore %tp %x
        \\OpControlBarrier %uint_2 %uint_2 %uint_264
        \\%small2 = OpULessThan %bool %lid %uint_16
        \\OpSelectionMerge %j1 None
        \\OpBranchConditional %small2 %t1 %j1
        \\%t1 = OpLabel
        \\%li = OpIAdd %uint %lid %uint_16
        \\%rp = OpAccessChain %_ptr_WG_uint %tile %li
        \\%rv = OpLoad %uint %rp
        \\OpStore %outp %rv
        \\OpBranch %j1
        \\%j1 = OpLabel
        \\OpReturn
        \\OpFunctionEnd
    , .buffers = try a.dupe(Buffer, &.{
        .{ .index = 0, .words = try seq(a, 1024, struct {
            fn f(i: usize) u32 {
                return @intCast(i + 1);
            }
        }.f) },
        .{ .index = 1, .words = try zeros(a, 1024), .expect = try seq(a, 1024, struct {
            fn f(i: usize) u32 {
                return if (i % 64 < 16) @intCast(i + 17) else 0;
            }
        }.f) },
    }) });
}

/// The review's lowering defects (round 1), each run where it computed a
/// wrong answer or was refused: OpenCL atan/atan2 keep the sign of a zero
/// result, an access chain on a bitcast pointer GEPs from the cast address,
/// Workgroup Block variables alias, scalar-layout structs ending in a vec3
/// are stored without their AIR tail padding, a packed member's pointer
/// keeps its alignment through 70 helper calls, and GL_EXT_buffer_reference
/// pointers (members of a push-constant block, from a uint64, from a uvec2)
/// address buffers the host made resident.
fn appendLoweringFixTests(a: Allocator, list: *std.ArrayList(Test)) !void {
    // atan(±0) = ±0 and atan2(y, x) has y's sign (C99 Annex F, OpenCL C 3.0
    // 7.5); Metal's precise air.atan / air.atan2 return +0 here.
    try list.append(a, .{
        .name = "atan_signed_zero",
        .threads = 4,
        .text =
        \\OpCapability Kernel
        \\OpCapability Addresses
        \\OpCapability Int64
        \\%ocl = OpExtInstImport "OpenCL.std"
        \\OpMemoryModel Physical64 OpenCL
        \\OpEntryPoint Kernel %k "atan_signed_zero" %gid
        \\OpDecorate %gid BuiltIn GlobalInvocationId
        \\%void = OpTypeVoid
        \\%u64 = OpTypeInt 64 0
        \\%f32 = OpTypeFloat 32
        \\%v3u64 = OpTypeVector %u64 3
        \\%pin = OpTypePointer Input %v3u64
        \\%gid = OpVariable %pin Input
        \\%pf = OpTypePointer CrossWorkgroup %f32
        \\%fn = OpTypeFunction %void %pf %pf %pf %pf
        \\%c1 = OpConstant %u64 1
        \\%c2 = OpConstant %u64 2
        \\%k = OpFunction %void None %fn
        \\%x = OpFunctionParameter %pf
        \\%y = OpFunctionParameter %pf
        \\%xx = OpFunctionParameter %pf
        \\%out = OpFunctionParameter %pf
        \\%e = OpLabel
        \\%gv = OpLoad %v3u64 %gid
        \\%g = OpCompositeExtract %u64 %gv 0
        \\%px = OpInBoundsPtrAccessChain %pf %x %g
        \\%va = OpLoad %f32 %px Aligned 4
        \\%py = OpInBoundsPtrAccessChain %pf %y %g
        \\%vb = OpLoad %f32 %py Aligned 4
        \\%pxx = OpInBoundsPtrAccessChain %pf %xx %g
        \\%vc = OpLoad %f32 %pxx Aligned 4
        \\%r1 = OpExtInst %f32 %ocl atan %va
        \\%r2 = OpExtInst %f32 %ocl atan2 %vb %vc
        \\%o1 = OpIMul %u64 %g %c2
        \\%po1 = OpInBoundsPtrAccessChain %pf %out %o1
        \\OpStore %po1 %r1 Aligned 4
        \\%o2 = OpIAdd %u64 %o1 %c1
        \\%po2 = OpInBoundsPtrAccessChain %pf %out %o2
        \\OpStore %po2 %r2 Aligned 4
        \\OpReturn
        \\OpFunctionEnd
        ,
        .buffers = try a.dupe(Buffer, &.{
            .{ .index = 0, .words = try a.dupe(u32, &.{ 0x80000000, 0, 0x80000000, 0 }) },
            .{ .index = 1, .words = try a.dupe(u32, &.{ f32Bits(-1e-30), f32Bits(1e-30), f32Bits(-1e-30), 0x80000000 }) },
            .{ .index = 2, .words = try a.dupe(u32, &.{ f32Bits(1e30), f32Bits(1e30), f32Bits(3.4e38), f32Bits(1) }) },
            .{ .index = 3, .words = try zeros(a, 8), .expect = try a.dupe(u32, &.{ 0x80000000, 0x80000000, 0, 0, 0x80000000, 0x80000000, 0, 0x80000000 }) },
        }),
    });

    // `q = (uvec4[4]*)&in.data; out = q[1]` (Zig's `@ptrCast(&buf.data)`):
    // the chain on the bitcast pointer indexes [4 x <4 x i32>], not the
    // source's [16 x i32] (which read words 1-4).
    try list.append(a, .{ .name = "bitcast_access_chain", .threads = 1, .text =
        \\OpCapability Shader
        \\OpCapability Int64
        \\OpCapability PhysicalStorageBufferAddresses
        \\OpExtension "SPV_KHR_physical_storage_buffer"
        \\OpMemoryModel PhysicalStorageBuffer64 GLSL450
        \\OpEntryPoint GLCompute %k "bitcast_access_chain" %in %out
        \\OpExecutionMode %k LocalSize 1 1 1
        \\OpDecorate %arr16 ArrayStride 4
        \\OpDecorate %arr4v4 ArrayStride 16
        \\OpDecorate %Buf Block
        \\OpMemberDecorate %Buf 0 Offset 0
        \\OpDecorate %in DescriptorSet 0
        \\OpDecorate %in Binding 0
        \\OpDecorate %in NonWritable
        \\OpDecorate %out DescriptorSet 0
        \\OpDecorate %out Binding 1
        \\OpDecorate %OBuf Block
        \\OpMemberDecorate %OBuf 0 Offset 0
        \\%void = OpTypeVoid
        \\%fn = OpTypeFunction %void
        \\%u32 = OpTypeInt 32 0
        \\%u64 = OpTypeInt 64 0
        \\%v4 = OpTypeVector %u32 4
        \\%c0 = OpConstant %u32 0
        \\%c1 = OpConstant %u64 1
        \\%c4 = OpConstant %u32 4
        \\%c16 = OpConstant %u32 16
        \\%arr16 = OpTypeArray %u32 %c16
        \\%arr4v4 = OpTypeArray %v4 %c4
        \\%Buf = OpTypeStruct %arr16
        \\%OBuf = OpTypeStruct %v4
        \\%pBuf = OpTypePointer StorageBuffer %Buf
        \\%pOBuf = OpTypePointer StorageBuffer %OBuf
        \\%parr16 = OpTypePointer StorageBuffer %arr16
        \\%parr4v4 = OpTypePointer StorageBuffer %arr4v4
        \\%pv4 = OpTypePointer StorageBuffer %v4
        \\%in = OpVariable %pBuf StorageBuffer
        \\%out = OpVariable %pOBuf StorageBuffer
        \\%k = OpFunction %void None %fn
        \\%e = OpLabel
        \\%p0 = OpInBoundsAccessChain %parr16 %in %c0
        \\%pb = OpBitcast %parr4v4 %p0
        \\%p2 = OpInBoundsAccessChain %pv4 %pb %c1
        \\%v = OpLoad %v4 %p2 Aligned 16
        \\%po = OpInBoundsAccessChain %pv4 %out %c0
        \\OpStore %po %v Aligned 16
        \\OpReturn
        \\OpFunctionEnd
    , .buffers = try a.dupe(Buffer, &.{
        .{ .index = 0, .words = try seq(a, 16, struct {
            fn f(i: usize) u32 {
                return @intCast(i);
            }
        }.f) },
        .{ .index = 1, .words = try zeros(a, 4), .expect = try a.dupe(u32, &.{ 4, 5, 6, 7 }) },
    }) });

    // SPV_KHR_workgroup_memory_explicit_layout: `shared A { uint x[64]; }`
    // and `shared B { uint y[64]; uint z[64]; }` are one allocation (the
    // larger, 512 bytes): x[lid] = in[gid]; barrier; out[gid] = y[63 - lid].
    try list.append(a, .{ .name = "workgroup_block_alias", .threads = 256, .group_from_manifest = true, .text =
        \\OpCapability Shader
        \\OpCapability WorkgroupMemoryExplicitLayoutKHR
        \\OpExtension "SPV_KHR_workgroup_memory_explicit_layout"
        \\OpMemoryModel Logical GLSL450
        \\OpEntryPoint GLCompute %k "workgroup_block_alias" %gid %lid %in %out %wa %wb
        \\OpExecutionMode %k LocalSize 64 1 1
        \\OpDecorate %gid BuiltIn GlobalInvocationId
        \\OpDecorate %lid BuiltIn LocalInvocationId
        \\OpDecorate %rt ArrayStride 4
        \\OpDecorate %Buf Block
        \\OpMemberDecorate %Buf 0 Offset 0
        \\OpDecorate %in DescriptorSet 0
        \\OpDecorate %in Binding 0
        \\OpDecorate %out DescriptorSet 0
        \\OpDecorate %out Binding 1
        \\OpDecorate %a64 ArrayStride 4
        \\OpDecorate %A Block
        \\OpMemberDecorate %A 0 Offset 0
        \\OpDecorate %B Block
        \\OpMemberDecorate %B 0 Offset 0
        \\OpMemberDecorate %B 1 Offset 256
        \\%void = OpTypeVoid
        \\%fn = OpTypeFunction %void
        \\%u32 = OpTypeInt 32 0
        \\%v3 = OpTypeVector %u32 3
        \\%c0 = OpConstant %u32 0
        \\%c2 = OpConstant %u32 2
        \\%c63 = OpConstant %u32 63
        \\%c64 = OpConstant %u32 64
        \\%c264 = OpConstant %u32 264
        \\%rt = OpTypeRuntimeArray %u32
        \\%Buf = OpTypeStruct %rt
        \\%a64 = OpTypeArray %u32 %c64
        \\%A = OpTypeStruct %a64
        \\%B = OpTypeStruct %a64 %a64
        \\%pBuf = OpTypePointer StorageBuffer %Buf
        \\%pu = OpTypePointer StorageBuffer %u32
        \\%pA = OpTypePointer Workgroup %A
        \\%pB = OpTypePointer Workgroup %B
        \\%pwu = OpTypePointer Workgroup %u32
        \\%pin = OpTypePointer Input %v3
        \\%gid = OpVariable %pin Input
        \\%lid = OpVariable %pin Input
        \\%in = OpVariable %pBuf StorageBuffer
        \\%out = OpVariable %pBuf StorageBuffer
        \\%wa = OpVariable %pA Workgroup
        \\%wb = OpVariable %pB Workgroup
        \\%k = OpFunction %void None %fn
        \\%e = OpLabel
        \\%g3 = OpLoad %v3 %gid
        \\%g = OpCompositeExtract %u32 %g3 0
        \\%l3 = OpLoad %v3 %lid
        \\%l = OpCompositeExtract %u32 %l3 0
        \\%pi = OpAccessChain %pu %in %c0 %g
        \\%x = OpLoad %u32 %pi
        \\%px = OpAccessChain %pwu %wa %c0 %l
        \\OpStore %px %x
        \\OpControlBarrier %c2 %c2 %c264
        \\%r = OpISub %u32 %c63 %l
        \\%py = OpAccessChain %pwu %wb %c0 %r
        \\%y = OpLoad %u32 %py
        \\%po = OpAccessChain %pu %out %c0 %g
        \\OpStore %po %y
        \\OpReturn
        \\OpFunctionEnd
    , .buffers = try a.dupe(Buffer, &.{
        .{ .index = 0, .words = try seq(a, 256, struct {
            fn f(i: usize) u32 {
                return @intCast(7 * i + 1);
            }
        }.f) },
        .{ .index = 1, .words = try zeros(a, 256), .expect = try seq(a, 256, struct {
            fn f(i: usize) u32 {
                return @intCast(7 * ((i / 64) * 64 + 63 - i % 64) + 1);
            }
        }.f) },
    }) });

    // VK_EXT_scalar_block_layout: `struct P { vec3 v; }` in an array at
    // ArrayStride 12, loaded and stored whole by adjacent threads, and
    // `struct B { Inner i; float f @12; }` with `Inner { vec3 v; }`, whose
    // whole-Inner store must not write f (AIR would give Inner 16 bytes).
    try list.append(a, .{ .name = "scalar_tight_structs", .threads = 16, .text =
        \\OpCapability Shader
        \\OpMemoryModel Logical GLSL450
        \\OpEntryPoint GLCompute %k "scalar_tight_structs" %gid
        \\OpExecutionMode %k LocalSize 16 1 1
        \\OpDecorate %gid BuiltIn GlobalInvocationId
        \\OpMemberDecorate %P 0 Offset 0
        \\OpDecorate %rtP ArrayStride 12
        \\OpMemberDecorate %PBuf 0 Offset 0
        \\OpDecorate %PBuf Block
        \\OpMemberDecorate %Inner 0 Offset 0
        \\OpMemberDecorate %B 0 Offset 0
        \\OpMemberDecorate %B 1 Offset 12
        \\OpDecorate %rtB ArrayStride 16
        \\OpMemberDecorate %BBuf 0 Offset 0
        \\OpDecorate %BBuf Block
        \\OpDecorate %src DescriptorSet 0
        \\OpDecorate %src Binding 0
        \\OpDecorate %src NonWritable
        \\OpDecorate %dst DescriptorSet 0
        \\OpDecorate %dst Binding 1
        \\OpDecorate %bb DescriptorSet 0
        \\OpDecorate %bb Binding 2
        \\%void = OpTypeVoid
        \\%fn = OpTypeFunction %void
        \\%u32 = OpTypeInt 32 0
        \\%f32 = OpTypeFloat 32
        \\%v3f32 = OpTypeVector %f32 3
        \\%v3u32 = OpTypeVector %u32 3
        \\%c0 = OpConstant %u32 0
        \\%c1 = OpConstant %u32 1
        \\%f2 = OpConstant %f32 2
        \\%P = OpTypeStruct %v3f32
        \\%rtP = OpTypeRuntimeArray %P
        \\%PBuf = OpTypeStruct %rtP
        \\%Inner = OpTypeStruct %v3f32
        \\%B = OpTypeStruct %Inner %f32
        \\%rtB = OpTypeRuntimeArray %B
        \\%BBuf = OpTypeStruct %rtB
        \\%pPBuf = OpTypePointer StorageBuffer %PBuf
        \\%pBBuf = OpTypePointer StorageBuffer %BBuf
        \\%pP = OpTypePointer StorageBuffer %P
        \\%pInner = OpTypePointer StorageBuffer %Inner
        \\%pf = OpTypePointer StorageBuffer %f32
        \\%pin = OpTypePointer Input %v3u32
        \\%gid = OpVariable %pin Input
        \\%src = OpVariable %pPBuf StorageBuffer
        \\%dst = OpVariable %pPBuf StorageBuffer
        \\%bb = OpVariable %pBBuf StorageBuffer
        \\%k = OpFunction %void None %fn
        \\%e = OpLabel
        \\%gv = OpLoad %v3u32 %gid
        \\%g = OpCompositeExtract %u32 %gv 0
        \\%ps = OpAccessChain %pP %src %c0 %g
        \\%s = OpLoad %P %ps
        \\%sv = OpCompositeExtract %v3f32 %s 0
        \\%sv2 = OpVectorTimesScalar %v3f32 %sv %f2
        \\%np = OpCompositeConstruct %P %sv2
        \\%pd = OpAccessChain %pP %dst %c0 %g
        \\OpStore %pd %np
        \\%pi = OpAccessChain %pInner %bb %c0 %g %c0
        \\%in = OpLoad %Inner %pi
        \\%pfv = OpAccessChain %pf %bb %c0 %g %c1
        \\%fv = OpLoad %f32 %pfv
        \\%iv = OpCompositeExtract %v3f32 %in 0
        \\%fs = OpCompositeConstruct %v3f32 %fv %fv %fv
        \\%iv2 = OpFAdd %v3f32 %iv %fs
        \\%ni = OpCompositeConstruct %Inner %iv2
        \\OpStore %pi %ni
        \\OpReturn
        \\OpFunctionEnd
    , .buffers = try a.dupe(Buffer, &.{
        .{ .index = 0, .words = try seq(a, 48, struct {
            fn f(i: usize) u32 {
                return f32Bits(@floatFromInt(i / 3 + i % 3));
            }
        }.f) },
        .{ .index = 1, .words = try zeros(a, 48), .expect = try seq(a, 48, struct {
            fn f(i: usize) u32 {
                return f32Bits(@floatFromInt(2 * (i / 3 + i % 3)));
            }
        }.f) },
        .{ .index = 2, .words = try seq(a, 64, struct {
            fn f(i: usize) u32 {
                const g: f32 = @floatFromInt(i / 4);
                const lane: f32 = @floatFromInt(i % 4 + 1);
                return f32Bits(if (i % 4 == 3) 10 else g * lane);
            }
        }.f), .expect = try seq(a, 64, struct {
            fn f(i: usize) u32 {
                const g: f32 = @floatFromInt(i / 4);
                const lane: f32 = @floatFromInt(i % 4 + 1);
                return f32Bits(if (i % 4 == 3) 10 else g * lane + 10);
            }
        }.f) },
    }) });

    // A uvec4 at Offset 4 of a block reached through 70 helper calls: the
    // load is 4-aligned (an `align 16` there is LLVM undefined behaviour),
    // which took one lowering round per call before.
    try list.append(a, .{ .name = "packed_pointer_call_chain", .threads = 1, .text = try callChainModule(a, 70), .buffers = try a.dupe(Buffer, &.{
        .{ .index = 0, .words = try a.dupe(u32, &.{ 0, 11, 22, 33, 44, 0, 0, 0 }), .expect = try a.dupe(u32, &.{ 55, 11, 22, 33, 44 }) },
    }) });

    // GL_EXT_buffer_reference (glslang's shape, OpTypeForwardPointer
    // included): pointer members of the push-constant block, a uint64
    // address (OpConvertUToPtr), a uvec2 address (OpBitcast,
    // GL_EXT_buffer_reference_uvec2) and OpConvertPtrToU compared with the
    // address the host wrote. The four buffers are only resident.
    try list.append(a, .{ .name = "buffer_reference", .threads = 64, .text =
        \\OpCapability Shader
        \\OpCapability Int64
        \\OpCapability PhysicalStorageBufferAddresses
        \\OpExtension "SPV_KHR_physical_storage_buffer"
        \\OpMemoryModel PhysicalStorageBuffer64 GLSL450
        \\OpEntryPoint GLCompute %main "buffer_reference" %gid %pc
        \\OpExecutionMode %main LocalSize 64 1 1
        \\OpDecorate %gid BuiltIn GlobalInvocationId
        \\OpMemberDecorate %PC 0 Offset 0
        \\OpMemberDecorate %PC 1 Offset 8
        \\OpMemberDecorate %PC 2 Offset 16
        \\OpMemberDecorate %PC 3 Offset 24
        \\OpMemberDecorate %PC 4 Offset 32
        \\OpMemberDecorate %PC 5 Offset 40
        \\OpDecorate %PC Block
        \\OpDecorate %rt ArrayStride 4
        \\OpMemberDecorate %Data 0 Offset 0
        \\OpDecorate %Data Block
        \\%void = OpTypeVoid
        \\%fn = OpTypeFunction %void
        \\%bool = OpTypeBool
        \\%u32 = OpTypeInt 32 0
        \\%i32 = OpTypeInt 32 1
        \\%u64 = OpTypeInt 64 0
        \\%v2u32 = OpTypeVector %u32 2
        \\%v3u32 = OpTypeVector %u32 3
        \\%pin = OpTypePointer Input %v3u32
        \\%gid = OpVariable %pin Input
        \\%c0 = OpConstant %u32 0
        \\%c1 = OpConstant %u32 1
        \\%i0 = OpConstant %i32 0
        \\%i1 = OpConstant %i32 1
        \\%i2 = OpConstant %i32 2
        \\%i3 = OpConstant %i32 3
        \\%i4 = OpConstant %i32 4
        \\%i5 = OpConstant %i32 5
        \\OpTypeForwardPointer %pData PhysicalStorageBuffer
        \\%PC = OpTypeStruct %pData %pData %u64 %v2u32 %u32 %u64
        \\%rt = OpTypeRuntimeArray %u32
        \\%Data = OpTypeStruct %rt
        \\%pData = OpTypePointer PhysicalStorageBuffer %Data
        \\%pPC = OpTypePointer PushConstant %PC
        \\%pc = OpVariable %pPC PushConstant
        \\%pcp = OpTypePointer PushConstant %pData
        \\%pcu64 = OpTypePointer PushConstant %u64
        \\%pcv2 = OpTypePointer PushConstant %v2u32
        \\%pcu = OpTypePointer PushConstant %u32
        \\%pdu = OpTypePointer PhysicalStorageBuffer %u32
        \\%main = OpFunction %void None %fn
        \\%e = OpLabel
        \\%gv = OpLoad %v3u32 %gid
        \\%g = OpCompositeExtract %u32 %gv 0
        \\%psrc = OpAccessChain %pcp %pc %i0
        \\%src = OpLoad %pData %psrc
        \\%pdst = OpAccessChain %pcp %pc %i1
        \\%dst = OpLoad %pData %pdst
        \\%pk = OpAccessChain %pcu %pc %i4
        \\%kk = OpLoad %u32 %pk
        \\%ps = OpAccessChain %pdu %src %i0 %g
        \\%s = OpLoad %u32 %ps Aligned 4
        \\%m = OpIMul %u32 %s %kk
        \\%pd = OpAccessChain %pdu %dst %i0 %g
        \\OpStore %pd %m Aligned 4
        \\%paux = OpAccessChain %pcu64 %pc %i2
        \\%auxu = OpLoad %u64 %paux
        \\%aux = OpConvertUToPtr %pData %auxu
        \\%sum = OpIAdd %u32 %s %kk
        \\%pa = OpAccessChain %pdu %aux %i0 %g
        \\OpStore %pa %sum Aligned 4
        \\%paux2 = OpAccessChain %pcv2 %pc %i3
        \\%aux2v = OpLoad %v2u32 %paux2
        \\%aux2 = OpBitcast %pData %aux2v
        \\%psa = OpAccessChain %pcu64 %pc %i5
        \\%srca = OpLoad %u64 %psa
        \\%srcu = OpConvertPtrToU %u64 %src
        \\%same = OpIEqual %bool %srcu %srca
        \\%g1 = OpIAdd %u32 %g %c1
        \\%flag = OpSelect %u32 %same %g1 %c0
        \\%pa2 = OpAccessChain %pdu %aux2 %i0 %g
        \\OpStore %pa2 %flag Aligned 4
        \\OpReturn
        \\OpFunctionEnd
    , .buffers = try a.dupe(Buffer, &.{
        .{ .index = 30, .words = try a.dupe(u32, &.{ 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0 }), .addresses = try a.dupe(Address, &.{
            .{ .word = 0, .of = 100 },
            .{ .word = 2, .of = 101 },
            .{ .word = 4, .of = 102 },
            .{ .word = 6, .of = 103 },
            .{ .word = 10, .of = 100 },
        }) },
        .{ .index = 100, .unbound = true, .words = try seq(a, 64, struct {
            fn f(i: usize) u32 {
                return @intCast(7 * i + 1);
            }
        }.f) },
        .{ .index = 101, .unbound = true, .words = try zeros(a, 64), .expect = try seq(a, 64, struct {
            fn f(i: usize) u32 {
                return @intCast(3 * (7 * i + 1));
            }
        }.f) },
        .{ .index = 102, .unbound = true, .words = try zeros(a, 64), .expect = try seq(a, 64, struct {
            fn f(i: usize) u32 {
                return @intCast(7 * i + 1 + 3);
            }
        }.f) },
        .{ .index = 103, .unbound = true, .words = try zeros(a, 64), .expect = try seq(a, 64, struct {
            fn f(i: usize) u32 {
                return @intCast(i + 1);
            }
        }.f) },
    }) });
}

/// A kernel that passes `&buf.v` (a uvec4 at Offset 4, so 4-aligned)
/// through `depth` + 1 helpers, the last of which stores v.x + v.w into
/// word 0 (the review's gen_callchain.py shape, with a result).
pub fn callChainModule(a: Allocator, depth: usize) ![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    try out.appendSlice(a,
        \\OpCapability Shader
        \\OpMemoryModel Logical GLSL450
        \\OpEntryPoint GLCompute %k "packed_pointer_call_chain"
        \\OpExecutionMode %k LocalSize 1 1 1
        \\OpDecorate %buf Binding 0
        \\OpDecorate %buf DescriptorSet 0
        \\OpDecorate %blk Block
        \\OpMemberDecorate %blk 0 Offset 0
        \\OpMemberDecorate %blk 1 Offset 4
        \\%void = OpTypeVoid
        \\%fn = OpTypeFunction %void
        \\%u32 = OpTypeInt 32 0
        \\%v4 = OpTypeVector %u32 4
        \\%c0 = OpConstant %u32 0
        \\%c1 = OpConstant %u32 1
        \\%blk = OpTypeStruct %u32 %v4
        \\%pblk = OpTypePointer StorageBuffer %blk
        \\%pv4 = OpTypePointer StorageBuffer %v4
        \\%pu32 = OpTypePointer StorageBuffer %u32
        \\%hfn = OpTypeFunction %void %pv4
        \\%buf = OpVariable %pblk StorageBuffer
        \\%k = OpFunction %void None %fn
        \\%entry = OpLabel
        \\%q = OpAccessChain %pv4 %buf %c1
        \\%r = OpFunctionCall %void %h0 %q
        \\OpReturn
        \\OpFunctionEnd
        \\
    );
    for (0..depth + 1) |i| {
        try out.print(a, "%h{d} = OpFunction %void None %hfn\n%p{d} = OpFunctionParameter %pv4\n%e{d} = OpLabel\n", .{ i, i, i });
        if (i < depth) {
            try out.print(a, "%r{d} = OpFunctionCall %void %h{d} %p{d}\n", .{ i, i + 1, i });
        } else {
            try out.print(a, "%l = OpLoad %v4 %p{d}\n%x = OpCompositeExtract %u32 %l 0\n%w = OpCompositeExtract %u32 %l 3\n%s = OpIAdd %u32 %x %w\n%o = OpAccessChain %pu32 %buf %c0\nOpStore %o %s\n", .{i});
        }
        try out.appendSlice(a, "OpReturn\nOpFunctionEnd\n");
    }
    return out.items;
}

/// Review round 2 (lowering): Volatile accesses re-read memory (a
/// spin-wait on a flag another SIMD-group sets, through the memory operand
/// and through glslang's Volatile member decorations), half sin, tan and
/// asin keep the sign of a zero, OpenCL normalize and length of half
/// vectors with subnormal elements, access chains continued from a pointer
/// to a struct element stored in its tight form, and an entry point named
/// like a generated symbol.
fn appendRound2LoweringTests(a: Allocator, list: *std.ArrayList(Test)) !void {
    // Thread 32 (another SIMD-group of the threadgroup) stores 42 with a
    // Volatile memory operand; thread 0 polls with Volatile loads, at most
    // 20,000,000 times, and stores what it saw last. With plain loads
    // Metal hoisted the read out of the loop: 20,000,000 polls, seen 0.
    try list.append(a, .{ .name = "volatile_spin_wait", .threads = 64, .text =
        \\OpCapability Kernel
        \\OpCapability Addresses
        \\OpCapability Int64
        \\OpMemoryModel Physical64 OpenCL
        \\OpEntryPoint Kernel %k "volatile_spin_wait" %gid
        \\OpExecutionMode %k LocalSize 64 1 1
        \\OpDecorate %gid BuiltIn GlobalInvocationId
        \\%void = OpTypeVoid
        \\%bool = OpTypeBool
        \\%u32 = OpTypeInt 32 0
        \\%u64 = OpTypeInt 64 0
        \\%v3u64 = OpTypeVector %u64 3
        \\%p_u32 = OpTypePointer CrossWorkgroup %u32
        \\%fn = OpTypeFunction %void %p_u32 %p_u32
        \\%p_in = OpTypePointer Input %v3u64
        \\%gid = OpVariable %p_in Input
        \\%u64_0 = OpConstant %u64 0
        \\%u64_1 = OpConstant %u64 1
        \\%u64_32 = OpConstant %u64 32
        \\%c0 = OpConstant %u32 0
        \\%c1 = OpConstant %u32 1
        \\%c42 = OpConstant %u32 42
        \\%limit = OpConstant %u32 20000000
        \\%k = OpFunction %void None %fn
        \\%flag = OpFunctionParameter %p_u32
        \\%out = OpFunctionParameter %p_u32
        \\%entry = OpLabel
        \\%g = OpLoad %v3u64 %gid
        \\%i = OpCompositeExtract %u64 %g 0
        \\%is32 = OpIEqual %bool %i %u64_32
        \\OpSelectionMerge %end None
        \\OpBranchConditional %is32 %writer %not32
        \\%writer = OpLabel
        \\OpStore %flag %c42 Volatile
        \\OpBranch %end
        \\%not32 = OpLabel
        \\%is0 = OpIEqual %bool %i %u64_0
        \\OpSelectionMerge %end2 None
        \\OpBranchConditional %is0 %loop %end2
        \\%loop = OpLabel
        \\%cnt = OpPhi %u32 %c0 %not32 %cnt1 %loop
        \\%v = OpLoad %u32 %flag Volatile
        \\%cnt1 = OpIAdd %u32 %cnt %c1
        \\%zero = OpIEqual %bool %v %c0
        \\%small = OpULessThan %bool %cnt1 %limit
        \\%again = OpLogicalAnd %bool %zero %small
        \\OpLoopMerge %done %loop None
        \\OpBranchConditional %again %loop %done
        \\%done = OpLabel
        \\OpStore %out %cnt1
        \\%o1 = OpInBoundsPtrAccessChain %p_u32 %out %u64_1
        \\OpStore %o1 %v
        \\OpBranch %end2
        \\%end2 = OpLabel
        \\OpBranch %end
        \\%end = OpLabel
        \\OpReturn
        \\OpFunctionEnd
    , .buffers = try a.dupe(Buffer, &.{
        .{ .index = 0, .words = try zeros(a, 1), .expect = try a.dupe(u32, &.{42}) },
        .{ .index = 1, .words = try zeros(a, 2), .expect = try a.dupe(u32, &.{42}), .expect_at = 1 },
    }) });

    // glslang's `layout(std430, binding = 0) coherent volatile buffer F {
    // uint flag; uint n; uint seen; } f;` for Vulkan 1.0 (a Uniform
    // BufferBlock whose members are decorated Volatile and Coherent, plain
    // OpLoad / OpStore): thread 32 sets the flag, thread 0 polls it (at
    // most 4,000,000 times) and stores the count and what it saw.
    try list.append(a, .{ .name = "volatile_member_spin_wait", .threads = 64, .text =
        \\OpCapability Shader
        \\OpMemoryModel Logical GLSL450
        \\OpEntryPoint GLCompute %main "volatile_member_spin_wait" %lidv
        \\OpExecutionMode %main LocalSize 64 1 1
        \\OpDecorate %lidv BuiltIn LocalInvocationId
        \\OpMemberDecorate %F 0 Coherent
        \\OpMemberDecorate %F 0 Volatile
        \\OpMemberDecorate %F 0 Offset 0
        \\OpMemberDecorate %F 1 Coherent
        \\OpMemberDecorate %F 1 Volatile
        \\OpMemberDecorate %F 1 Offset 4
        \\OpMemberDecorate %F 2 Coherent
        \\OpMemberDecorate %F 2 Volatile
        \\OpMemberDecorate %F 2 Offset 8
        \\OpDecorate %F BufferBlock
        \\OpDecorate %f DescriptorSet 0
        \\OpDecorate %f Binding 0
        \\%void = OpTypeVoid
        \\%fnty = OpTypeFunction %void
        \\%uint = OpTypeInt 32 0
        \\%int = OpTypeInt 32 1
        \\%bool = OpTypeBool
        \\%v3uint = OpTypeVector %uint 3
        \\%pv3 = OpTypePointer Input %v3uint
        \\%pin = OpTypePointer Input %uint
        \\%lidv = OpVariable %pv3 Input
        \\%F = OpTypeStruct %uint %uint %uint
        \\%pF = OpTypePointer Uniform %F
        \\%f = OpVariable %pF Uniform
        \\%pu = OpTypePointer Uniform %uint
        \\%uint_0 = OpConstant %uint 0
        \\%uint_1 = OpConstant %uint 1
        \\%uint_32 = OpConstant %uint 32
        \\%uint_lim = OpConstant %uint 4000000
        \\%int_0 = OpConstant %int 0
        \\%int_1 = OpConstant %int 1
        \\%int_2 = OpConstant %int 2
        \\%main = OpFunction %void None %fnty
        \\%e = OpLabel
        \\%lp = OpAccessChain %pin %lidv %uint_0
        \\%lid = OpLoad %uint %lp
        \\%pflag = OpAccessChain %pu %f %int_0
        \\%is32 = OpIEqual %bool %lid %uint_32
        \\OpSelectionMerge %m1 None
        \\OpBranchConditional %is32 %w %m1
        \\%w = OpLabel
        \\OpStore %pflag %uint_1
        \\OpBranch %m1
        \\%m1 = OpLabel
        \\%is0 = OpIEqual %bool %lid %uint_0
        \\OpSelectionMerge %m2 None
        \\OpBranchConditional %is0 %spin %m2
        \\%spin = OpLabel
        \\OpBranch %h
        \\%h = OpLabel
        \\%n = OpPhi %uint %uint_0 %spin %n1 %c
        \\OpLoopMerge %done %c None
        \\OpBranch %t
        \\%t = OpLabel
        \\%fl = OpLoad %uint %pflag
        \\%z = OpIEqual %bool %fl %uint_0
        \\%lt = OpULessThan %bool %n %uint_lim
        \\%go = OpLogicalAnd %bool %z %lt
        \\OpBranchConditional %go %c %done
        \\%c = OpLabel
        \\%n1 = OpIAdd %uint %n %uint_1
        \\OpBranch %h
        \\%done = OpLabel
        \\%pn = OpAccessChain %pu %f %int_1
        \\OpStore %pn %n
        \\%fl2 = OpLoad %uint %pflag
        \\%ps = OpAccessChain %pu %f %int_2
        \\OpStore %ps %fl2
        \\OpBranch %m2
        \\%m2 = OpLabel
        \\OpReturn
        \\OpFunctionEnd
    , .buffers = try a.dupe(Buffer, &.{
        .{ .index = 0, .words = try a.dupe(u32, &.{ 0, 0xdead, 0xdead }), .expect = try a.dupe(u32, &.{1}), .expect_at = 2 },
    }) });

    // sin, tan and asin of -0.0 and +0.0 in half: ±0 (C99 Annex F, OpenCL
    // C 3.0 section 7.5). Metal's precise air.sin/tan/asin.f16 give +0 for
    // -0. Out: sin, tan, asin per thread, as halves.
    try list.append(a, .{ .name = "half_signed_zero", .threads = 2, .strict_only = true, .text =
        \\OpCapability Kernel
        \\OpCapability Addresses
        \\OpCapability Int64
        \\OpCapability Float16
        \\%ocl = OpExtInstImport "OpenCL.std"
        \\OpMemoryModel Physical64 OpenCL
        \\OpEntryPoint Kernel %k "half_signed_zero" %gid
        \\OpDecorate %gid BuiltIn GlobalInvocationId
        \\%void = OpTypeVoid
        \\%u64 = OpTypeInt 64 0
        \\%f16 = OpTypeFloat 16
        \\%v3u64 = OpTypeVector %u64 3
        \\%pin = OpTypePointer Input %v3u64
        \\%gid = OpVariable %pin Input
        \\%ph = OpTypePointer CrossWorkgroup %f16
        \\%fn = OpTypeFunction %void %ph %ph
        \\%c1 = OpConstant %u64 1
        \\%c2 = OpConstant %u64 2
        \\%c3 = OpConstant %u64 3
        \\%k = OpFunction %void None %fn
        \\%x = OpFunctionParameter %ph
        \\%out = OpFunctionParameter %ph
        \\%e = OpLabel
        \\%gv = OpLoad %v3u64 %gid
        \\%g = OpCompositeExtract %u64 %gv 0
        \\%px = OpInBoundsPtrAccessChain %ph %x %g
        \\%v = OpLoad %f16 %px Aligned 2
        \\%s = OpExtInst %f16 %ocl sin %v
        \\%t = OpExtInst %f16 %ocl tan %v
        \\%as = OpExtInst %f16 %ocl asin %v
        \\%o0 = OpIMul %u64 %g %c3
        \\%po0 = OpInBoundsPtrAccessChain %ph %out %o0
        \\OpStore %po0 %s Aligned 2
        \\%o1 = OpIAdd %u64 %o0 %c1
        \\%po1 = OpInBoundsPtrAccessChain %ph %out %o1
        \\OpStore %po1 %t Aligned 2
        \\%o2 = OpIAdd %u64 %o0 %c2
        \\%po2 = OpInBoundsPtrAccessChain %ph %out %o2
        \\OpStore %po2 %as Aligned 2
        \\OpReturn
        \\OpFunctionEnd
    , .buffers = try a.dupe(Buffer, &.{
        .{ .index = 0, .words = try a.dupe(u32, &.{0x00008000}) },
        .{ .index = 1, .words = try filled(a, 3, 0x12341234), .expect = try a.dupe(u32, &.{ 0x80008000, 0x00008000, 0 }) },
    }) });

    // OpenCL normalize and length of half2 vectors with subnormal elements
    // (and one whose sum of squares overflows half): computed in float, as
    // half's range is too narrow to scale both ways. Per thread: (2^-24, 0),
    // (3 * 2^-24, 0), (3 * 2^-24, 4 * 2^-24), (0, 0), (60000, 60000).
    // normalize: (1, 0), (1, 0), (0.6, 0.8), (0, 0), (0.7071, 0.7071); length:
    // 2^-24, 3 * 2^-24, 5 * 2^-24, 0, inf. The 2^10 half scale gave (inf,
    // NaN) and 0 for the first.
    try list.append(a, .{ .name = "half_geometric_subnormal", .threads = 5, .text =
        \\OpCapability Kernel
        \\OpCapability Addresses
        \\OpCapability Int64
        \\OpCapability Float16
        \\%ocl = OpExtInstImport "OpenCL.std"
        \\OpMemoryModel Physical64 OpenCL
        \\OpEntryPoint Kernel %k "half_geometric_subnormal" %gid
        \\OpDecorate %gid BuiltIn GlobalInvocationId
        \\%void = OpTypeVoid
        \\%u64 = OpTypeInt 64 0
        \\%f16 = OpTypeFloat 16
        \\%v2f16 = OpTypeVector %f16 2
        \\%v3u64 = OpTypeVector %u64 3
        \\%pin = OpTypePointer Input %v3u64
        \\%gid = OpVariable %pin Input
        \\%pv = OpTypePointer CrossWorkgroup %v2f16
        \\%ph = OpTypePointer CrossWorkgroup %f16
        \\%fn = OpTypeFunction %void %pv %pv %ph
        \\%k = OpFunction %void None %fn
        \\%x = OpFunctionParameter %pv
        \\%nout = OpFunctionParameter %pv
        \\%lout = OpFunctionParameter %ph
        \\%e = OpLabel
        \\%gv = OpLoad %v3u64 %gid
        \\%g = OpCompositeExtract %u64 %gv 0
        \\%px = OpInBoundsPtrAccessChain %pv %x %g
        \\%v = OpLoad %v2f16 %px Aligned 4
        \\%n = OpExtInst %v2f16 %ocl normalize %v
        \\%l = OpExtInst %f16 %ocl length %v
        \\%pn = OpInBoundsPtrAccessChain %pv %nout %g
        \\OpStore %pn %n Aligned 4
        \\%pl = OpInBoundsPtrAccessChain %ph %lout %g
        \\OpStore %pl %l Aligned 2
        \\OpReturn
        \\OpFunctionEnd
    , .buffers = try a.dupe(Buffer, &.{
        .{ .index = 0, .words = try a.dupe(u32, &.{ 0x00000001, 0x00000003, 0x00040003, 0, 0x7B537B53 }) },
        .{ .index = 1, .words = try filled(a, 5, 0x12341234), .expect = try a.dupe(u32, &.{ 0x00003C00, 0x00003C00, 0x3A6638CD, 0, 0x39A839A8 }) },
        .{ .index = 2, .words = try zeros(a, 3), .expect = try a.dupe(u32, &.{ 0x00030001, 0x00000005, 0x00007C00 }) },
    }) });

    // Scalar block layout (GL_EXT_scalar_block_layout, 16-bit storage):
    // access chains continued from a pointer to an array element held in
    // its tight form. T { uint16_t a; uint b @4; uint c; uint d; vec3 v @16 }
    // at ArrayStride 28: `p = &t[g]; p->b = 0x1000 + g; (&p->v)->y = 2.0`.
    // Inner { float a; vec3 v @16 } at stride 28: `q = &in[g].v; *q = ...`.
    // B { I i; float f @12 } (I { vec3 v; }) at stride 16: a whole-I store
    // through `&b[g].i` leaves f alone. The continued chains walked the
    // natural layout's member indices into the tight form: b landed on a,
    // v.y on v.x, the vec3 at words 1-3.
    try list.append(a, .{ .name = "tight_chain_continued", .threads = 4, .text =
        \\OpCapability Shader
        \\OpCapability Int16
        \\OpCapability StorageBuffer16BitAccess
        \\OpExtension "SPV_KHR_16bit_storage"
        \\OpMemoryModel Logical GLSL450
        \\OpEntryPoint GLCompute %k "tight_chain_continued" %gid
        \\OpExecutionMode %k LocalSize 4 1 1
        \\OpDecorate %gid BuiltIn GlobalInvocationId
        \\OpMemberDecorate %T 0 Offset 0
        \\OpMemberDecorate %T 1 Offset 4
        \\OpMemberDecorate %T 2 Offset 8
        \\OpMemberDecorate %T 3 Offset 12
        \\OpMemberDecorate %T 4 Offset 16
        \\OpDecorate %rtT ArrayStride 28
        \\OpDecorate %BufT Block
        \\OpMemberDecorate %BufT 0 Offset 0
        \\OpMemberDecorate %Inner 0 Offset 0
        \\OpMemberDecorate %Inner 1 Offset 16
        \\OpDecorate %rtInner ArrayStride 28
        \\OpDecorate %BufInner Block
        \\OpMemberDecorate %BufInner 0 Offset 0
        \\OpMemberDecorate %I 0 Offset 0
        \\OpMemberDecorate %B 0 Offset 0
        \\OpMemberDecorate %B 1 Offset 12
        \\OpDecorate %rtB ArrayStride 16
        \\OpDecorate %BufB Block
        \\OpMemberDecorate %BufB 0 Offset 0
        \\OpDecorate %bt DescriptorSet 0
        \\OpDecorate %bt Binding 0
        \\OpDecorate %bi DescriptorSet 0
        \\OpDecorate %bi Binding 1
        \\OpDecorate %bb DescriptorSet 0
        \\OpDecorate %bb Binding 2
        \\%void = OpTypeVoid
        \\%fn = OpTypeFunction %void
        \\%u16 = OpTypeInt 16 0
        \\%u32 = OpTypeInt 32 0
        \\%f32 = OpTypeFloat 32
        \\%v3u32 = OpTypeVector %u32 3
        \\%v3 = OpTypeVector %f32 3
        \\%c0 = OpConstant %u32 0
        \\%c1 = OpConstant %u32 1
        \\%c4 = OpConstant %u32 4
        \\%c4096 = OpConstant %u32 4096
        \\%f1 = OpConstant %f32 1
        \\%f2 = OpConstant %f32 2
        \\%f3 = OpConstant %f32 3
        \\%f100 = OpConstant %f32 100
        \\%f200 = OpConstant %f32 200
        \\%f300 = OpConstant %f32 300
        \\%T = OpTypeStruct %u16 %u32 %u32 %u32 %v3
        \\%rtT = OpTypeRuntimeArray %T
        \\%BufT = OpTypeStruct %rtT
        \\%Inner = OpTypeStruct %f32 %v3
        \\%rtInner = OpTypeRuntimeArray %Inner
        \\%BufInner = OpTypeStruct %rtInner
        \\%I = OpTypeStruct %v3
        \\%B = OpTypeStruct %I %f32
        \\%rtB = OpTypeRuntimeArray %B
        \\%BufB = OpTypeStruct %rtB
        \\%pBufT = OpTypePointer StorageBuffer %BufT
        \\%pBufInner = OpTypePointer StorageBuffer %BufInner
        \\%pBufB = OpTypePointer StorageBuffer %BufB
        \\%pT = OpTypePointer StorageBuffer %T
        \\%pInner = OpTypePointer StorageBuffer %Inner
        \\%pB = OpTypePointer StorageBuffer %B
        \\%pI = OpTypePointer StorageBuffer %I
        \\%pv3 = OpTypePointer StorageBuffer %v3
        \\%pu32 = OpTypePointer StorageBuffer %u32
        \\%pf = OpTypePointer StorageBuffer %f32
        \\%pin = OpTypePointer Input %v3u32
        \\%gid = OpVariable %pin Input
        \\%bt = OpVariable %pBufT StorageBuffer
        \\%bi = OpVariable %pBufInner StorageBuffer
        \\%bb = OpVariable %pBufB StorageBuffer
        \\%k = OpFunction %void None %fn
        \\%e = OpLabel
        \\%g3 = OpLoad %v3u32 %gid
        \\%g = OpCompositeExtract %u32 %g3 0
        \\%gf = OpConvertUToF %f32 %g
        \\%val = OpIAdd %u32 %g %c4096
        \\%p = OpAccessChain %pT %bt %c0 %g
        \\%q = OpAccessChain %pu32 %p %c1
        \\OpStore %q %val
        \\%r = OpAccessChain %pf %p %c4 %c1
        \\OpStore %r %f2
        \\%x = OpFAdd %f32 %gf %f100
        \\%y = OpFAdd %f32 %gf %f200
        \\%z = OpFAdd %f32 %gf %f300
        \\%xyz = OpCompositeConstruct %v3 %x %y %z
        \\%pi = OpAccessChain %pInner %bi %c0 %g
        \\%qi = OpAccessChain %pv3 %pi %c1
        \\OpStore %qi %xyz
        \\%a1 = OpFAdd %f32 %gf %f1
        \\%a2 = OpFAdd %f32 %gf %f2
        \\%a3 = OpFAdd %f32 %gf %f3
        \\%av = OpCompositeConstruct %v3 %a1 %a2 %a3
        \\%iv = OpCompositeConstruct %I %av
        \\%pb = OpAccessChain %pB %bb %c0 %g
        \\%qb = OpAccessChain %pI %pb %c0
        \\OpStore %qb %iv
        \\OpReturn
        \\OpFunctionEnd
    , .buffers = try a.dupe(Buffer, &.{
        .{ .index = 0, .words = try filled(a, 28, 0xEEEEEEEE), .expect = try seq(a, 28, struct {
            fn f(i: usize) u32 {
                return switch (i % 7) {
                    1 => 0x1000 + @as(u32, @intCast(i / 7)),
                    5 => f32Bits(2),
                    else => 0xEEEEEEEE,
                };
            }
        }.f) },
        .{ .index = 1, .words = try filled(a, 28, 0xEEEEEEEE), .expect = try seq(a, 28, struct {
            fn f(i: usize) u32 {
                const g: f32 = @floatFromInt(i / 7);
                return switch (i % 7) {
                    4 => f32Bits(g + 100),
                    5 => f32Bits(g + 200),
                    6 => f32Bits(g + 300),
                    else => 0xEEEEEEEE,
                };
            }
        }.f) },
        .{ .index = 2, .words = try filled(a, 16, 0xEEEEEEEE), .expect = try seq(a, 16, struct {
            fn f(i: usize) u32 {
                const g: f32 = @floatFromInt(i / 4);
                return if (i % 4 == 3) 0xEEEEEEEE else f32Bits(g + @as(f32, @floatFromInt(i % 4 + 1)));
            }
        }.f) },
    }) });

    // An entry point named like the generated symbol of a helper (`@f20`
    // for function %20, which takes an argument): the helper is `@f20.0`.
    // It used to be defined twice and the call refused ("argument types do
    // not match the callee's declaration").
    try list.append(a, .{ .name = "f20", .threads = 8, .text =
        \\OpCapability Shader
        \\OpMemoryModel Logical GLSL450
        \\OpEntryPoint GLCompute %k "f20" %gid
        \\OpExecutionMode %k LocalSize 8 1 1
        \\OpDecorate %gid BuiltIn GlobalInvocationId
        \\OpDecorate %arr ArrayStride 4
        \\OpDecorate %blk Block
        \\OpMemberDecorate %blk 0 Offset 0
        \\OpDecorate %buf DescriptorSet 0
        \\OpDecorate %buf Binding 0
        \\%void = OpTypeVoid
        \\%fn = OpTypeFunction %void
        \\%u32 = OpTypeInt 32 0
        \\%v3 = OpTypeVector %u32 3
        \\%pv3 = OpTypePointer Input %v3
        \\%gid = OpVariable %pv3 Input
        \\%c0 = OpConstant %u32 0
        \\%c7 = OpConstant %u32 7
        \\%arr = OpTypeRuntimeArray %u32
        \\%blk = OpTypeStruct %arr
        \\%pblk = OpTypePointer StorageBuffer %blk
        \\%pu32 = OpTypePointer StorageBuffer %u32
        \\%buf = OpVariable %pblk StorageBuffer
        \\%hfn = OpTypeFunction %void %u32
        \\%20 = OpFunction %void None %hfn
        \\%hx = OpFunctionParameter %u32
        \\%he = OpLabel
        \\%hp = OpAccessChain %pu32 %buf %c0 %hx
        \\%hv = OpIAdd %u32 %hx %c7
        \\OpStore %hp %hv
        \\OpReturn
        \\OpFunctionEnd
        \\%k = OpFunction %void None %fn
        \\%entry = OpLabel
        \\%g = OpLoad %v3 %gid
        \\%x = OpCompositeExtract %u32 %g 0
        \\%r = OpFunctionCall %void %20 %x
        \\OpReturn
        \\OpFunctionEnd
    , .buffers = try a.dupe(Buffer, &.{
        .{ .index = 0, .words = try zeros(a, 8), .expect = try seq(a, 8, struct {
            fn f(i: usize) u32 {
                return @as(u32, @intCast(i)) + 7;
            }
        }.f) },
    }) });
}

/// A workgroup-tiled loop with a barrier in its body, bounded by a uniform
/// value (`pc.n` rounded up to whole tiles of 64) that glslang -V (no
/// optimizer) routes through Function variables (review round 2, "glslang
/// uniform loop bounds through a helper call or a local struct/vector"):
/// every helper argument is copied into a `param` variable and passed by
/// pointer (`uint tiles = ceilDiv(pc.n, 64u);`), and a local struct or
/// vector is read member by member through OpAccessChain (`P p = P(pc.n);
/// ... p.n`, `uvec2 r = uvec2(pc.n, 64u); ... r.x / r.y`). The uniformity
/// analysis used to treat an alloca whose address reached a call or a
/// getelementptr as per-thread and refused the barrier ("convergent call
/// in divergent control flow"), although the module is valid and uniform;
/// tools/air/divergence.zig now follows the variable through
/// getelementptr and into the helper. `pc.n` = 200: four tiles, two
/// workgroups of 64; `out[gid] = sum over t < 4 of in[t * 64 + 63 - lid]`.
/// The same kernels with the bound taken from the thread id are refused
/// (the test below).
fn appendUniformBoundTests(a: Allocator, list: *std.ArrayList(Test)) !void {
    const input = try a.alloc(u32, 256);
    for (input, 0..) |*w, i| w.* = f32Bits(@floatFromInt((i * 7) % 13));
    const want = try a.alloc(u32, 128);
    for (want, 0..) |*w, g| {
        const lid = g % 64;
        var acc: f32 = 0;
        for (0..4) |t| acc += @floatFromInt(((t * 64 + 63 - lid) * 7) % 13);
        w.* = f32Bits(acc);
    }
    for (uniform_bound_variants) |v| {
        var text = try std.mem.replaceOwned(u8, a, uniform_bound_template, "NAME", v.name);
        text = try std.mem.replaceOwned(u8, a, text, "BOUND", v.bound);
        try list.append(a, .{ .name = v.name, .threads = 128, .group_from_manifest = true, .text = text, .buffers = try a.dupe(Buffer, &.{
            .{ .index = 0, .words = input },
            .{ .index = 1, .words = try zeros(a, 128), .expect = want },
            .{ .index = 30, .words = try a.dupe(u32, &.{200}) },
        }) });
    }
}

const uniform_bound_variants = [_]struct { name: []const u8, bound: []const u8 }{
    // glslang's calling convention: `ceilDiv(u1;u1;` takes Function pointers.
    .{ .name = "uniform_bound_helper_by_pointer", .bound =
    \\OpStore %param %17
    \\OpStore %param_0 %uint_64
    \\%18 = OpFunctionCall %uint %ceilDiv_u1_u1_ %param %param_0
    },
    .{ .name = "uniform_bound_local_struct", .bound =
    \\%lpv = OpCompositeConstruct %LP %17
    \\OpStore %p %lpv
    \\%pn = OpAccessChain %_ptr_Function_uint %p %int_0
    \\%nn = OpLoad %uint %pn
    \\%18a = OpIAdd %uint %nn %uint_63
    \\%18 = OpUDiv %uint %18a %uint_64
    },
    .{ .name = "uniform_bound_local_vector", .bound =
    \\%rv = OpCompositeConstruct %v2uint %17 %uint_64
    \\OpStore %r %rv
    \\%rx = OpAccessChain %_ptr_Function_uint %r %uint_0
    \\%nx = OpLoad %uint %rx
    \\%ry = OpAccessChain %_ptr_Function_uint %r %uint_1
    \\%ny = OpLoad %uint %ry
    \\%18a = OpIAdd %uint %nx %uint_63
    \\%18 = OpUDiv %uint %18a %ny
    },
};

/// `for (t = 0; t < tiles; t++) { tile[lid] = in[t * 64 + lid]; barrier();
/// acc += tile[63 - lid]; barrier(); } out[gid] = acc;` with `tiles`
/// computed by BOUND from `%17` (= pc.n).
const uniform_bound_template =
    \\OpCapability Shader
    \\%glsl = OpExtInstImport "GLSL.std.450"
    \\OpMemoryModel Logical GLSL450
    \\OpEntryPoint GLCompute %main "NAME" %gl_LocalInvocationID %gl_GlobalInvocationID
    \\OpExecutionMode %main LocalSize 64 1 1
    \\OpDecorate %gl_LocalInvocationID BuiltIn LocalInvocationId
    \\OpDecorate %gl_GlobalInvocationID BuiltIn GlobalInvocationId
    \\OpMemberDecorate %PC 0 Offset 0
    \\OpDecorate %PC Block
    \\OpDecorate %_runtimearr_float ArrayStride 4
    \\OpMemberDecorate %In 0 NonWritable
    \\OpMemberDecorate %In 0 Offset 0
    \\OpDecorate %In BufferBlock
    \\OpDecorate %ina DescriptorSet 0
    \\OpDecorate %ina Binding 0
    \\OpMemberDecorate %Out 0 Offset 0
    \\OpDecorate %Out BufferBlock
    \\OpDecorate %outb DescriptorSet 0
    \\OpDecorate %outb Binding 1
    \\%void = OpTypeVoid
    \\%3 = OpTypeFunction %void
    \\%float = OpTypeFloat 32
    \\%_ptr_Function_float = OpTypePointer Function %float
    \\%float_0 = OpConstant %float 0
    \\%uint = OpTypeInt 32 0
    \\%_ptr_Function_uint = OpTypePointer Function %uint
    \\%fnty_u_pu_pu = OpTypeFunction %uint %_ptr_Function_uint %_ptr_Function_uint
    \\%uint_0 = OpConstant %uint 0
    \\%PC = OpTypeStruct %uint
    \\%_ptr_PushConstant_PC = OpTypePointer PushConstant %PC
    \\%pc = OpVariable %_ptr_PushConstant_PC PushConstant
    \\%int = OpTypeInt 32 1
    \\%int_0 = OpConstant %int 0
    \\%int_1 = OpConstant %int 1
    \\%_ptr_PushConstant_uint = OpTypePointer PushConstant %uint
    \\%bool = OpTypeBool
    \\%uint_64 = OpConstant %uint 64
    \\%uint_63 = OpConstant %uint 63
    \\%_arr_float_uint_64 = OpTypeArray %float %uint_64
    \\%_ptr_Workgroup__arr_float_uint_64 = OpTypePointer Workgroup %_arr_float_uint_64
    \\%tile = OpVariable %_ptr_Workgroup__arr_float_uint_64 Workgroup
    \\%v3uint = OpTypeVector %uint 3
    \\%_ptr_Input_v3uint = OpTypePointer Input %v3uint
    \\%gl_LocalInvocationID = OpVariable %_ptr_Input_v3uint Input
    \\%gl_GlobalInvocationID = OpVariable %_ptr_Input_v3uint Input
    \\%_ptr_Input_uint = OpTypePointer Input %uint
    \\%_runtimearr_float = OpTypeRuntimeArray %float
    \\%In = OpTypeStruct %_runtimearr_float
    \\%_ptr_Uniform_In = OpTypePointer Uniform %In
    \\%ina = OpVariable %_ptr_Uniform_In Uniform
    \\%Out = OpTypeStruct %_runtimearr_float
    \\%_ptr_Uniform_Out = OpTypePointer Uniform %Out
    \\%outb = OpVariable %_ptr_Uniform_Out Uniform
    \\%_ptr_Uniform_float = OpTypePointer Uniform %float
    \\%_ptr_Workgroup_float = OpTypePointer Workgroup %float
    \\%uint_2 = OpConstant %uint 2
    \\%uint_264 = OpConstant %uint 264
    \\%uint_1 = OpConstant %uint 1
    \\%LP = OpTypeStruct %uint
    \\%_ptr_Function_LP = OpTypePointer Function %LP
    \\%v2uint = OpTypeVector %uint 2
    \\%_ptr_Function_v2uint = OpTypePointer Function %v2uint
    \\%main = OpFunction %void None %3
    \\%5 = OpLabel
    \\%tiles = OpVariable %_ptr_Function_uint Function
    \\%param = OpVariable %_ptr_Function_uint Function
    \\%param_0 = OpVariable %_ptr_Function_uint Function
    \\%acc = OpVariable %_ptr_Function_float Function
    \\%t = OpVariable %_ptr_Function_uint Function
    \\%p = OpVariable %_ptr_Function_LP Function
    \\%r = OpVariable %_ptr_Function_v2uint Function
    \\%16 = OpAccessChain %_ptr_PushConstant_uint %pc %int_0
    \\%17 = OpLoad %uint %16
    \\BOUND
    \\OpStore %tiles %18
    \\OpStore %acc %float_0
    \\OpStore %t %uint_0
    \\OpBranch %10
    \\%10 = OpLabel
    \\OpLoopMerge %12 %13 None
    \\OpBranch %14
    \\%14 = OpLabel
    \\%15 = OpLoad %uint %t
    \\%19 = OpLoad %uint %tiles
    \\%20 = OpULessThan %bool %15 %19
    \\OpBranchConditional %20 %11 %12
    \\%11 = OpLabel
    \\%21 = OpAccessChain %_ptr_Input_uint %gl_LocalInvocationID %uint_0
    \\%22 = OpLoad %uint %21
    \\%23 = OpLoad %uint %t
    \\%24 = OpIMul %uint %23 %uint_64
    \\%25 = OpIAdd %uint %24 %22
    \\%26 = OpAccessChain %_ptr_Uniform_float %ina %int_0 %25
    \\%27 = OpLoad %float %26
    \\%28 = OpAccessChain %_ptr_Workgroup_float %tile %22
    \\OpStore %28 %27
    \\OpControlBarrier %uint_2 %uint_2 %uint_264
    \\%29 = OpISub %uint %uint_63 %22
    \\%30 = OpAccessChain %_ptr_Workgroup_float %tile %29
    \\%31 = OpLoad %float %30
    \\%32 = OpLoad %float %acc
    \\%33 = OpFAdd %float %32 %31
    \\OpStore %acc %33
    \\OpControlBarrier %uint_2 %uint_2 %uint_264
    \\OpBranch %13
    \\%13 = OpLabel
    \\%40 = OpLoad %uint %t
    \\%41 = OpIAdd %uint %40 %uint_1
    \\OpStore %t %41
    \\OpBranch %10
    \\%12 = OpLabel
    \\%60 = OpAccessChain %_ptr_Input_uint %gl_GlobalInvocationID %uint_0
    \\%61 = OpLoad %uint %60
    \\%62 = OpLoad %float %acc
    \\%63 = OpAccessChain %_ptr_Uniform_float %outb %int_0 %61
    \\OpStore %63 %62
    \\OpReturn
    \\OpFunctionEnd
    \\%ceilDiv_u1_u1_ = OpFunction %uint None %fnty_u_pu_pu
    \\%a = OpFunctionParameter %_ptr_Function_uint
    \\%b = OpFunctionParameter %_ptr_Function_uint
    \\%70 = OpLabel
    \\%71 = OpLoad %uint %a
    \\%72 = OpLoad %uint %b
    \\%73 = OpIAdd %uint %71 %72
    \\%74 = OpISub %uint %73 %uint_1
    \\%75 = OpLoad %uint %b
    \\%76 = OpUDiv %uint %74 %75
    \\OpReturnValue %76
    \\OpFunctionEnd
;

fn filled(a: Allocator, n: usize, v: u32) ![]const u32 {
    const out = try a.alloc(u32, n);
    @memset(out, v);
    return out;
}

// ── Tests ───────────────────────────────────────────────────────────────────

const lower = @import("lower.zig");

test "the uniform-bound kernels are refused when the bound comes from the thread id" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var n: usize = 0;
    for (try all(a)) |t| {
        if (!std.mem.startsWith(u8, t.name, "uniform_bound_")) continue;
        n += 1;
        try std.testing.expectEqualStrings("", (try lower.lowerTextForTest(a, t.text, .{})).message);
        const lid = "%17x = OpLoad %uint %16\n%l0 = OpAccessChain %_ptr_Input_uint %gl_LocalInvocationID %uint_0\n%17 = OpLoad %uint %l0\n";
        const divergent = try std.mem.replaceOwned(u8, a, t.text, "%17 = OpLoad %uint %16\n", lid);
        try std.testing.expect(!std.mem.eql(u8, divergent, t.text));
        try std.testing.expectError(error.Unsupported, lower.lowerTextForTest(a, divergent, .{}));
    }
    try std.testing.expectEqual(uniform_bound_variants.len, n);
}

test "every execution test lowers, assembles and binds exactly the manifest's buffers" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for (try all(a)) |t| {
        const r = try lower.lowerTextForTest(a, t.text, .{});
        if (r.message.len > 0) {
            std.debug.print("{s} refused: {s}\n", .{ t.name, r.message });
            return error.TestUnexpectedResult;
        }
        try std.testing.expectEqual(1, r.kernels.len);
        try std.testing.expectEqualStrings(t.name, r.kernels[0].name);
        var n: usize = 0;
        for (r.kernels[0].params) |p| switch (p) {
            .buffer => |b| {
                n += 1;
                var found = false;
                for (t.buffers) |tb| found = found or (tb.index == b.index and !tb.unbound);
                if (!found) {
                    std.debug.print("{s}: the manifest has buffer {d}, the test does not\n", .{ t.name, b.index });
                    return error.TestUnexpectedResult;
                }
            },
            .builtin => {},
        };
        var bound: usize = 0;
        for (t.buffers) |tb| bound += @intFromBool(!tb.unbound);
        try std.testing.expectEqual(bound, n);
    }
}
