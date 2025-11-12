const std = @import("std");
const builtin = @import("builtin");

pub fn build(b: *std.Build) void {
    const host_os = b.option(std.Target.Os.Tag, "override host os", "") orelse b.graph.host.result.os.tag;
    const host_arch = b.option(std.Target.Cpu.Arch, "override host cpu arch", "") orelse b.graph.host.result.cpu.arch;

    const dep_name = b.fmt("{s}_{s}", .{
        switch (host_os) {
            .windows => "windows",
            .linux => "linux",
            .macos => "macos",
            else => @panic("host os not supported by zig-slang-binaries"),
        },
        switch (host_arch) {
            .x86_64 => "x86_64",
            .aarch64 => "aarch64",
            else => @panic("host cpu arch not supported by zig-slang-binaries"),
        },
    });

    if (b.lazyDependency(dep_name, .{})) |dep| {
        b.addNamedLazyPath("binaries", dep.path(""));
    }
}
