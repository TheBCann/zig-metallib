const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // ── 1. Shaders: Zig source -> LLVM IR ──────────────────────────────────
    // Zig only allows GPU address spaces on GPU targets, so the shader module
    // is compiled for NVPTX. Nothing NVPTX-specific survives air-splice.
    // `-Dshader-optimize=safe` (ReleaseSafe) keeps Zig's safety checks in the
    // shader (they compile to `llvm.trap` + `unreachable` through the
    // `no_panic` handler in my_shader.zig); the assembler's uniformity
    // analysis treats the trapping blocks as no-return, so the barriers
    // after a bounds check stay accepted. Used as a verification build.
    const shader_optimize = b.option(std.builtin.OptimizeMode, "shader-optimize", "Optimize mode of the GPU shader module: fast (default), safe, debug, small") orelse .fast;
    const gpu_target = b.resolveTargetQuery(.{
        .cpu_arch = .nvptx64,
        .os_tag = .cuda,
        .cpu_model = .{ .explicit = &std.Target.nvptx.cpu.sm_75 },
    });
    const shader_obj = b.addObject(.{
        .name = "shader",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/engine/my_shader.zig"),
            .target = gpu_target,
            .optimize = shader_optimize,
        }),
        .use_llvm = true,
    });
    shader_obj.bundle_ubsan_rt = false;
    const shader_ir = shader_obj.getEmittedLlvmIr();

    // ── 2. air-splice: LLVM IR -> .metallib ────────────────────────────────
    // The tool imports the shader file on the host to read its `functions`
    // manifest at comptime; the shader bodies are never analysed there.
    // It rewrites the IR, assembles it to bitcode with std.zig.llvm.Builder
    // and packs the MTLB container itself. No Apple tooling is involved.
    const shader_manifest = b.createModule(.{
        .root_source_file = b.path("src/engine/my_shader.zig"),
        .target = target,
        .optimize = optimize,
    });
    const splice_mod = b.createModule(.{
        .root_source_file = b.path("tools/air_splice.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{.{ .name = "shader", .module = shader_manifest }},
    });
    const splice = b.addExecutable(.{ .name = "air-splice", .root_module = splice_mod });
    const run_splice = b.addRunArtifact(splice);
    run_splice.addFileArg(shader_ir);
    const metallib = run_splice.addOutputFileArg("default.metallib");
    // Debug text output: the rewritten IR plus metadata, also accepted by
    // `xcrun metal -Xclang -opaque-pointers` for cross-checking.
    const air_ll = run_splice.addOutputFileArg("shader.air.ll");

    // ── 4. The app ─────────────────────────────────────────────────────────
    const objc_mod = b.addModule("objc", .{
        .root_source_file = b.path("src/objc.zig"),
        .target = target,
    });

    const exe = b.addExecutable(.{
        .name = "window",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{.{ .name = "objc", .module = objc_mod }},
        }),
    });
    // `@embedFile("default_metallib")` in main.zig resolves to the build output.
    exe.root_module.addAnonymousImport("default_metallib", .{ .root_source_file = metallib });

    exe.root_module.linkSystemLibrary("c", .{});
    exe.root_module.linkSystemLibrary("objc", .{});
    exe.root_module.linkFramework("Cocoa", .{});
    exe.root_module.linkFramework("Metal", .{});
    exe.root_module.linkFramework("QuartzCore", .{});
    exe.root_module.linkFramework("MetalKit", .{});

    b.installArtifact(exe);

    const run_step = b.step("run", "Run the app");
    const run_cmd = b.addRunArtifact(exe);
    run_step.dependOn(&run_cmd.step);
    run_cmd.step.dependOn(b.getInstallStep());
    run_cmd.addPassthruArgs();

    // `zig build metallib` drops the compiled shader library and the debug IR
    // in zig-out/bin for inspection with `xcrun metal-objdump -d`.
    const metallib_step = b.step("metallib", "Install default.metallib and shader.air.ll for inspection");
    metallib_step.dependOn(&b.addInstallBinFile(metallib, "default.metallib").step);
    metallib_step.dependOn(&b.addInstallBinFile(air_ll, "shader.air.ll").step);

    // ── 5. Tools ───────────────────────────────────────────────────────────
    // `zig build check -- some.metallib` loads a library into Metal and builds
    // a pipeline from the manifest's entry points. Validates packer output.
    const check = b.addExecutable(.{
        .name = "metallib-check",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tools/metallib_check.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "objc", .module = objc_mod },
                .{ .name = "shader", .module = shader_manifest },
            },
        }),
    });
    check.root_module.linkSystemLibrary("c", .{});
    check.root_module.linkSystemLibrary("objc", .{});
    check.root_module.linkFramework("Foundation", .{});
    check.root_module.linkFramework("Metal", .{});
    const run_check = b.addRunArtifact(check);
    run_check.addPassthruArgs();
    b.step("check", "Load a .metallib into Metal and build a pipeline from it").dependOn(&run_check.step);

    // ── 6. Tests ───────────────────────────────────────────────────────────
    const splice_tests = b.addTest(.{ .root_module = splice_mod });
    const test_step = b.step("test", "Run tests");
    test_step.dependOn(&b.addRunArtifact(splice_tests).step);
}
