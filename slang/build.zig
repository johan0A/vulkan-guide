const std = @import("std");
const builtin = @import("builtin");

pub fn build(b: *std.Build) void {
    const host_os = b.option(std.Target.Os.Tag, "override host os", "") orelse builtin.os.tag;
    const host_arch = b.option(std.Target.Cpu.Arch, "override host cpu arch", "") orelse builtin.cpu.arch;

    var buff: [1024]u8 = undefined;
    if (b.lazyDependency(std.fmt.bufPrint(&buff, "{s}_{s}", .{
        switch (host_os) {
            .windows => "windows",
            .linux => "linux",
            .macos => "macos",
            else => @panic("host os not supported by zig-slang-bin"),
        },
        switch (host_arch) {
            .x86_64 => "x86_64",
            .aarch64 => "aarch64",
            else => @panic("host arch not supported by zig-slang-bin"),
        },
    }) catch @panic(""), .{})) |dep| {
        b.addNamedLazyPath("slang", dep.path(""));
    }
}
