const std = @import("std");
const builtin = @import("builtin");
const zon = @import("build.zig.zon");
const air_target = @import("tools/air/target.zig");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // The project pins one exact Zig version in build.zig.zon. The assembler
    // is built on std.zig.llvm.Builder, which changes between Zig versions, so
    // say up front when the running compiler differs rather than leaving the
    // mismatch to surface as an unrelated-looking compile error.
    if (!std.mem.eql(u8, builtin.zig_version_string, zon.minimum_zig_version)) {
        std.log.warn("this project is pinned to Zig {s} (build.zig.zon) but this is Zig {s}; " ++
            "std.zig.llvm.Builder changes between Zig versions, so build errors may come from the version mismatch", .{ zon.minimum_zig_version, builtin.zig_version_string });
    }

    // The oldest macOS the generated library must load on (tools/air/target.zig).
    // Metal refuses a library that targets a newer macOS major than the one running.
    const metal_target = b.option(air_target.Name, "metal-target", "macOS version the .metallib targets (default macos26)") orelse air_target.default.name;
    // Only macos26 is verified; the older targets need typed-pointer bitcode
    // (tools/air/target.zig). This builds one anyway, e.g. to run
    // `zig build check` on that macOS and learn whether it accepts it.
    const allow_unverified = b.option(bool, "allow-unverified-target", "Build a -Dmetal-target that is not verified to work") orelse false;

    // ── 1. Shaders: Zig source -> LLVM IR ──────────────────────────────────
    // Zig only allows GPU address spaces on GPU targets, so the shader module
    // is compiled for NVPTX. Nothing NVPTX-specific survives air-splice.
    // `-Dshader-optimize=safe` (ReleaseSafe) keeps Zig's safety checks in the
    // shader (they compile to `llvm.trap` + `unreachable` through the
    // `no_panic` handler in my_shader.zig); the assembler's uniformity
    // analysis treats the trapping blocks as no-return, so the barriers
    // after a bounds check stay accepted. Used as a verification build.
    const shader_optimize = b.option(std.lang.Optimize, "shader-optimize", "Optimize mode of the GPU shader module: fast (default), safe, debug, small") orelse .fast;
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
    run_splice.addArg(b.fmt("--target={t}", .{metal_target}));
    if (allow_unverified) run_splice.addArg("--allow-unverified");
    run_splice.addFileArg(shader_ir);
    const metallib = run_splice.addOutputFileArg2("default.metallib", .{});
    // Debug text output: the rewritten IR plus metadata, also accepted by
    // `xcrun metal -Xclang -opaque-pointers` for cross-checking.
    const air_ll = run_splice.addOutputFileArg2("shader.air.ll", .{});

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
    // Compile (not run) metallib-check too: nothing else builds it without a
    // GPU, so a compile error in it would otherwise pass `zig build test`.
    test_step.dependOn(&check.step);

    // ── 7. Differential test bench: `zig build crosscheck -Dseed=N` ─────────────
    // tools/crosscheck/gen.zig writes a random total i32 `body(a, b)` for the seed.
    // That ONE generated file is compiled twice: into the GPU kernel
    // (src/engine/crosscheck_harness.zig -> a second air-splice -> crosscheck.metallib)
    // and into the host driver, which runs the kernel on the Metal device and
    // `body` on the CPU over the same inputs and compares. A mismatch is a
    // codegen bug. None of this is built by the default steps.
    const crosscheck_seed = b.option(u64, "seed", "Seed for `zig build crosscheck` (default 1)") orelse 1;
    // Triage and reduction: run a hand-written or reduced body through the
    // same pipeline instead of a generated one. It must define `pub const
    // seed: u64` and `pub fn body(a: i32, b: i32) i32`.
    const crosscheck_body_file = b.option([]const u8, "crosscheck-body", "Use this body file instead of generating one from -Dseed");

    const crosscheck_gen = b.addExecutable(.{
        .name = "crosscheck-gen",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tools/crosscheck/gen.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    const crosscheck_body_src: std.Build.LazyPath = if (crosscheck_body_file) |path| .{ .cwd_relative = path } else blk: {
        const run_crosscheck_gen = b.addRunArtifact(crosscheck_gen);
        run_crosscheck_gen.addArg(b.fmt("--seed={d}", .{crosscheck_seed}));
        break :blk run_crosscheck_gen.addOutputFileArg2("crosscheck_body.zig", .{});
    };

    // GPU side: harness + generated body for nvptx, exactly like the shader.
    const crosscheck_body_gpu = b.createModule(.{ .root_source_file = crosscheck_body_src, .target = gpu_target, .optimize = shader_optimize });
    const crosscheck_shader_obj = b.addObject(.{
        .name = "crosscheck_shader",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/engine/crosscheck_harness.zig"),
            .target = gpu_target,
            .optimize = shader_optimize,
            .imports = &.{.{ .name = "crosscheck_body", .module = crosscheck_body_gpu }},
        }),
        .use_llvm = true,
    });
    crosscheck_shader_obj.bundle_ubsan_rt = false;

    // A second air-splice whose manifest is the harness.
    const crosscheck_body_host = b.createModule(.{ .root_source_file = crosscheck_body_src, .target = target, .optimize = optimize });
    const crosscheck_manifest = b.createModule(.{
        .root_source_file = b.path("src/engine/crosscheck_harness.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{.{ .name = "crosscheck_body", .module = crosscheck_body_host }},
    });
    const crosscheck_splice = b.addExecutable(.{
        .name = "crosscheck-air-splice",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tools/air_splice.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{.{ .name = "shader", .module = crosscheck_manifest }},
        }),
    });
    const run_crosscheck_splice = b.addRunArtifact(crosscheck_splice);
    run_crosscheck_splice.addFileArg(crosscheck_shader_obj.getEmittedLlvmIr());
    const crosscheck_metallib = run_crosscheck_splice.addOutputFileArg2("crosscheck.metallib", .{});
    // The rewritten IR as text, for the Apple-assembler cross-check below.
    const crosscheck_air_ll = run_crosscheck_splice.addOutputFileArg2("crosscheck.air.ll", .{});

    // Host side: the driver runs `body` on the CPU and the kernel on the GPU.
    const crosscheck_driver = b.addExecutable(.{
        .name = "crosscheck-driver",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tools/crosscheck/driver.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "objc", .module = objc_mod },
                .{ .name = "crosscheck_body", .module = crosscheck_body_host },
            },
        }),
    });
    crosscheck_driver.root_module.linkSystemLibrary("c", .{});
    crosscheck_driver.root_module.linkSystemLibrary("objc", .{});
    crosscheck_driver.root_module.linkFramework("Foundation", .{});
    crosscheck_driver.root_module.linkFramework("Metal", .{});
    const run_crosscheck = b.addRunArtifact(crosscheck_driver);
    run_crosscheck.addArg(b.fmt("--seed={d}", .{crosscheck_seed}));
    // So the driver's reproduce hint names the body file rather than a seed.
    if (crosscheck_body_file) |path| run_crosscheck.addArg(b.fmt("--body={s}", .{path}));
    run_crosscheck.addFileArg(crosscheck_metallib);
    b.step("crosscheck", "Differential test: one generated kernel on the GPU vs the CPU (-Dseed=N)").dependOn(&run_crosscheck.step);

    // Attribution: the same kernel, but the library is built from our printed
    // IR by Apple's own assembler (`xcrun metal`, as CI does for shader.air.ll)
    // instead of tools/air/assembler.zig. A finding that persists here is in
    // Metal's GPU compiler; one that disappears is in our assembler or
    // container. Needs the Metal toolchain (`xcrun metal --version`).
    const apple_assemble = b.addSystemCommand(&.{ "xcrun", "metal", "-Xclang", "-opaque-pointers" });
    apple_assemble.addFileArg(crosscheck_air_ll);
    apple_assemble.addArg("-o");
    const crosscheck_apple_metallib = apple_assemble.addOutputFileArg2("crosscheck.apple.metallib", .{});
    const run_crosscheck_apple = b.addRunArtifact(crosscheck_driver);
    run_crosscheck_apple.addArg(b.fmt("--seed={d}", .{crosscheck_seed}));
    if (crosscheck_body_file) |path| run_crosscheck_apple.addArg(b.fmt("--body={s}", .{path}));
    run_crosscheck_apple.addFileArg(crosscheck_apple_metallib);
    b.step("crosscheck-apple", "The crosscheck kernel with the library built by Apple's assembler from our IR (attribution)").dependOn(&run_crosscheck_apple.step);
}
