const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const name = b.option([]const u8, "name", "set the name of the emitted binary");

    const root_module = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });

    const vulkan = b.dependency("vulkan", .{
        .registry = b.path("vendor/vk.xml"),
    });
    root_module.addImport("vulkan", vulkan.module("vulkan-zig"));

    const sdl_dep = b.dependency("sdl", .{
        .target = target,
        .optimize = optimize,
        .preferred_link_mode = .static,
    });
    root_module.linkLibrary(sdl_dep.artifact("SDL3"));

    const c_translate = b.addTranslateC(.{
        .root_source_file = b.addWriteFiles().add("c.c",
            \\#include <SDL3/SDL.h>
            \\#include <SDL3/SDL_vulkan.h>
        ),
        .target = target,
        .optimize = optimize,
    });
    c_translate.addIncludePath(sdl_dep.path("include"));
    root_module.addImport("c", c_translate.createModule());

    {
        const exe = b.addExecutable(.{ .name = name orelse "zig-exe-template", .root_module = root_module });

        b.installArtifact(exe);
        const run_cmd = b.addRunArtifact(exe);
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
