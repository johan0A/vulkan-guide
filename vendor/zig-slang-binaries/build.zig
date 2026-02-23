const std = @import("std");
const builtin = @import("builtin");

pub fn build(b: *std.Build) void {
    const host_os = b.option(std.Target.Os.Tag, "host_os", "override host os") orelse b.graph.host.result.os.tag;
    const host_arch = b.option(std.Target.Cpu.Arch, "host_arch", "override host cpu arch") orelse b.graph.host.result.cpu.arch;

    const os = switch (host_os) {
        .windows => "windows",
        .linux => "linux",
        .macos => "macos",
        else => @panic("host os not supported by zig-slang-binaries"),
    };

    const arch = switch (host_arch) {
        .x86_64 => "x86_64",
        .aarch64 => "aarch64",
        else => @panic("host cpu arch not supported by zig-slang-binaries"),
    };

    const version = "2026.3.1";

    const download_url = b.fmt(
        "https://github.com/shader-slang/slang/releases/download/v{[version]s}/slang-{[version]s}-{[os]s}-{[arch]s}.tar.gz",
        .{
            .version = version,
            .os = os,
            .arch = arch,
        },
    );

    const fetch = b.addRunArtifact(b.addExecutable(.{
        .name = "fetch",
        .root_module = b.createModule(.{
            .root_source_file = b.path("fetch_tar_gz.zig"),
            .target = b.graph.host,
            .optimize = .Debug,
        }),
    }));

    fetch.setName("fetch slang binaries");

    fetch.addArg(download_url);
    const path = fetch.addOutputDirectoryArg("slang");

    b.addNamedLazyPath("binaries", path);
}
