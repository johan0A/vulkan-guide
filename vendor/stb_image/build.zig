const std = @import("std");

pub fn build(b: *std.Build) void {
    const optimize = b.standardOptimizeOption(.{});
    const target = b.standardTargetOptions(.{});

    const root_module = b.createModule(.{
        .optimize = optimize,
        .target = target,
        .link_libc = true,
    });
    root_module.addCSourceFile(.{ .file = b.addWriteFiles().add("stub.c",
        \\#define STB_IMAGE_IMPLEMENTATION
        \\#include "stb_image.h"
    ) });
    root_module.addIncludePath(b.path("."));

    const lib = b.addLibrary(.{ .root_module = root_module, .name = "stb_image" });
    b.installArtifact(lib);

    lib.installHeader(b.path("stb_image.h"), "stb_image.h");
}
