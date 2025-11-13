const std = @import("std");

pub fn build(b: *std.Build) !void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const is_release = b.option(bool, "release", "build a release") orelse false;

    const options = .{
        .assets_path = b.option([]const u8, "assets-path", "") orelse "assets",
        .shaders_path = b.option([]const u8, "shaders-path", "") orelse if (is_release) "shaders" else "zig-out/shaders",
        .enable_validation_layers = if (b.option(bool, "no-validation-layers", "")) |result| !result else !is_release,
    };

    if (is_release) {
        const install = b.addInstallDirectory(.{
            .install_subdir = "assets",
            .install_dir = .{ .custom = "release" },
            .source_dir = b.path("assets"),
        });
        b.getInstallStep().dependOn(&install.step);
    }

    const root_module = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });

    {
        var options_step = b.addOptions();
        inline for (std.meta.fields(@TypeOf(options))) |field| {
            options_step.addOption(field.type, field.name, @field(options, field.name));
        }
        root_module.addImport("options", options_step.createModule());
    }

    {
        const slang_dep = b.dependency("zig_slang_binaries", .{});
        const slang_path = slang_dep.namedLazyPath("binaries").path(b, "bin/slangc");
        const shaders_path = b.path("shaders");

        var shader_paths_src = b.addOptions();

        for ([_][]const u8{
            "circle.slang",
            "color.slang",
            "colored_triangle_frag.slang",
            "colored_triangle_mesh_vert.slang",
            "colored_triangle_vert.slang",
            "imgui_frag.slang",
            "mesh_frag.slang",
            "mesh_vert.slang",
            "tex_image_frag.slang",
        }) |name| {
            const command: *std.Build.Step.Run = .create(b, b.fmt("compile shader {s}", .{name}));
            command.addFileArg(slang_path);
            command.addFileArg(shaders_path.path(b, name));
            command.addArg("-o");
            const stem = std.fs.path.stem(name);
            const out_path = command.addOutputFileArg(b.fmt("{s}.spv", .{stem}));

            const install = b.addInstallFile(out_path, b.fmt("{s}/{s}.spv", .{ if (is_release) "release/shaders" else "shaders", stem }));
            b.getInstallStep().dependOn(&install.step);

            shader_paths_src.addOption([]const u8, stem, b.fmt("{s}/{s}.spv", .{ options.shaders_path, stem }));
        }

        const shaders_paths_mod = shader_paths_src.createModule();
        root_module.addImport("shaders", shaders_paths_mod);
    }

    const vulkan_headers_dep = b.dependency("vulkan_headers", .{});

    {
        const vulkan = b.dependency("vulkan", .{
            .registry = vulkan_headers_dep.path("registry/vk.xml"),
        });
        root_module.addImport("vulkan", vulkan.module("vulkan-zig"));

        const tracy = b.dependency("tracy", .{
            .enable_tracing = b.option(bool, "enable_tracing", "Enable Tracy profile markers") orelse false,
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

        const vma_dep = b.dependency("VulkanMemoryAllocator", .{
            .target = target,
            .optimize = optimize,
            .macro_static_vulkan_functions = false,
            .macro_dynamic_vulkan_functions = true,
            .@"install-vulkan-headers" = true,
        });
        root_module.linkLibrary(vma_dep.artifact("VulkanMemoryAllocator"));

        const ImguiBackend = @import("dcimgui").Backend;
        const dcimgui_dep = b.dependency("dcimgui", .{
            .target = target,
            .optimize = optimize,
            .docking = true,
            .backends = &[_]ImguiBackend{ .imgui_impl_sdl3, .imgui_impl_vulkan },
            .@"include-path-list" = &[_]std.Build.LazyPath{
                vulkan_headers_dep.path("include"),
                sdl_dep.artifact("SDL3").getEmittedIncludeTree(),
            },
            .imconfig = b.addWriteFiles().add("imconfig.h",
                \\ #pragma once
                \\ #define IMGUI_IMPL_VULKAN_NO_PROTOTYPES 
            ),
        });
        root_module.linkLibrary(dcimgui_dep.artifact("dcimgui"));

        const c_translate = b.addTranslateC(.{
            .root_source_file = b.addWriteFiles().add("stub.h",
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
        c_translate.addIncludePath(vma_dep.artifact("VulkanMemoryAllocator").getEmittedIncludeTree());
        c_translate.addIncludePath(dcimgui_dep.artifact("dcimgui").getEmittedIncludeTree());
        root_module.addImport("c", c_translate.createModule());
    }

    {
        const exe = b.addExecutable(.{ .name = "vulkan-tutorial", .root_module = root_module });
        exe.subsystem = if (is_release) .Windows else null;

        b.getInstallStep().dependOn(&b.addInstallArtifact(exe, .{
            .dest_dir = if (is_release) .{ .override = .{ .custom = "release" } } else .default,
        }).step);
        const run_cmd = b.addRunArtifact(exe);
        run_cmd.cwd = b.path("");
        run_cmd.step.dependOn(b.getInstallStep());

        if (b.args) |args| run_cmd.addArgs(args);

        const run_step = b.step("run", "Run the app");
        run_step.dependOn(&run_cmd.step);
    }

    {
        const tests = b.addTest(.{ .name = "test", .root_module = root_module });

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
