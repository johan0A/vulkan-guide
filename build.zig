const std = @import("std");

pub fn build(b: *std.Build) !void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const name = b.option([]const u8, "name", "set the name of the emitted binary");

    const root_module = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });

    const shader_module = shadersModule(b, b.path("./shaders"));
    root_module.addImport("shaders", shader_module);

    {
        const vulkan = b.dependency("vulkan", .{
            .registry = b.path("vendor/vk.xml"),
        });
        root_module.addImport("vulkan", vulkan.module("vulkan-zig"));

        const tracy = b.dependency("tracy", .{
            .enable_tracing = b.option(bool, "enable_tracing", "Enable Tracy profile markers") orelse false,
            .enable_fibers = b.option(bool, "enable_fibers", "Enable Tracy fiber support") orelse false,
            .on_demand = b.option(bool, "on_demand", "Build tracy with TRACY_ON_DEMAND") orelse false,
            .callstack_support = b.option(bool, "callstack_support", "Builds tracy with TRACY_USE_CALLSTACK") orelse false,
            .default_callstack_depth = b.option(u32, "default_callstack_depth", "sets TRACY_CALLSTACK to the depth provided") orelse 0,
        });
        root_module.addImport("tracy", tracy.module("tracy"));

        const zla = b.dependency("zla", .{
            .target = target,
            .optimize = optimize, // todo: change to ReleaseFast/ReleaseSafe?
        });
        root_module.addImport("zla", zla.module("zla"));

        const gltf = b.dependency("zgltf", .{
            .target = target,
            .optimize = optimize,
        });
        root_module.addImport("gltf", gltf.module("zgltf"));
    }

    {
        const sdl_dep = b.dependency("sdl", .{
            .target = target,
            .optimize = optimize,
            .preferred_link_mode = .static,
        });
        root_module.linkLibrary(sdl_dep.artifact("SDL3"));

        const vma = b.dependency("VulkanMemoryAllocator", .{
            .target = target,
            .optimize = optimize,
            .macro_static_vulkan_functions = false,
            .macro_dynamic_vulkan_functions = true,
            .@"install-vulkan-headers" = true,
        });
        root_module.linkLibrary(vma.artifact("VulkanMemoryAllocator"));

        const vulkan_headers_dep = b.dependency("vulkan_headers", .{
            .target = target,
            .optimize = optimize,
        });

        const cimgui_dep = b.dependency("dcimgui", .{
            .target = target,
            .optimize = optimize,
            .@"include-path-list" = @as([]const std.Build.LazyPath, &.{
                vulkan_headers_dep.artifact("vulkan-headers").getEmittedIncludeTree(),
                sdl_dep.artifact("SDL3").getEmittedIncludeTree(),
            }),
        });
        root_module.linkLibrary(cimgui_dep.artifact("cimgui_clib"));

        const c_translate = b.addTranslateC(.{
            .root_source_file = b.addWriteFiles().add("c.h",
                \\#include <SDL3/SDL.h>
                \\#include <SDL3/SDL_vulkan.h>
                \\#include <vk_mem_alloc.h>
                \\#include <dcimgui.h>
                \\#include <dcimgui_impl_sdl3.h>
                \\#include <dcimgui_impl_vulkan.h>
            ),
            .target = target,
            .optimize = optimize,
        });
        c_translate.addIncludePath(sdl_dep.artifact("SDL3").getEmittedIncludeTree());
        c_translate.addIncludePath(vma.artifact("VulkanMemoryAllocator").getEmittedIncludeTree());
        c_translate.addIncludePath(cimgui_dep.artifact("cimgui_clib").getEmittedIncludeTree());
        root_module.addImport("c", c_translate.createModule());
    }

    {
        const exe = b.addExecutable(.{ .name = name orelse "zig-exe-template", .root_module = root_module });
        exe.subsystem = if (b.option(
            bool,
            "no-console",
            "on Windows: disables console output and opening a console window with the app window",
        )) |option| (if (option) .Windows else null) else null;

        b.installArtifact(exe);
        const run_cmd = b.addRunArtifact(exe);
        run_cmd.cwd = b.path("zig-out/bin/");
        run_cmd.step.dependOn(b.getInstallStep());

        if (b.args) |args| run_cmd.addArgs(args);

        const run_step = b.step("run", "Run the app");
        run_step.dependOn(&run_cmd.step);
    }

    {
        const tests = b.addTest(.{ .name = name orelse "test", .root_module = root_module });

        const run_tests = b.addRunArtifact(tests);
        const test_step = b.step("test", "Run unit tests");
        test_step.dependOn(&run_tests.step);

        const debug_tests_artifact = b.addInstallArtifact(tests, .{});
        const debug_tests_step = b.step("build-test", "Create a test artifact that runs the tests");
        debug_tests_step.dependOn(&debug_tests_artifact.step);
    }

    {
        const exe_check = b.addExecutable(.{ .name = "check", .root_module = root_module });
        const tests_check = b.addTest(.{ .name = "check", .root_module = root_module });

        const check = b.step("check", "Check if exe and tests compile");
        check.dependOn(&exe_check.step);
        check.dependOn(&tests_check.step);
    }
}

fn shadersModule(
    b: *std.Build,
    path: std.Build.LazyPath,
) *std.Build.Module {
    const shaders_dir = std.fs.openDirAbsolute(path.getPath2(b, null), .{ .iterate = true }) catch @panic("failed to open shader directory");
    var shaders_files_it = shaders_dir.iterate();

    var generated_file: std.ArrayListUnmanaged(u8) = .empty;

    const module = b.createModule(.{});

    while (shaders_files_it.next() catch @panic("failed to iterate on shader file")) |shader_file| {
        if (shader_file.kind != .file) continue;
        if (!std.mem.eql(u8, std.fs.path.extension(shader_file.name), ".slang")) continue;

        const stem = std.fs.path.stem(shader_file.name);

        generated_file.appendSlice(
            b.allocator,
            b.fmt("pub const {s} = \"shaders/{s}.spv\";", .{ stem, stem }),
        ) catch @panic("OOM");

        const slang_path = b.dependency("zig_slang_binaries", .{}).namedLazyPath("binaries");
        const slang_exe_path = slang_path.join(b.allocator, "bin/slangc") catch @panic("OOM");

        const system_command = b.addSystemCommand(&.{slang_exe_path.getPath2(b, null)});
        system_command.addFileArg(path.join(b.allocator, shader_file.name) catch @panic("OOM"));
        system_command.addArg("-o");
        const out_path = system_command.addOutputFileArg(b.fmt("{s}.spv", .{stem}));

        const install = b.addInstallFile(out_path, b.fmt("bin/shaders/{s}.spv", .{stem}));
        b.getInstallStep().dependOn(&install.step);
    }

    module.root_source_file = b.addWriteFiles().add("shaders.zig", generated_file.items);

    return module;
}
