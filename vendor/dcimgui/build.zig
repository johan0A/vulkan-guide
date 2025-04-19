// TODO: finnish this and make it its own package

// TODO: add warning for this:
// NOTE: unfortunately switching to the 'prefix-less' functions in
// zimgui.h isn't that easy because some Dear ImGui functions collide
// with Win32 function (Set/GetCursorPos and Set/GetWindowPos).
const std = @import("std");
const builtin = @import("builtin");

pub fn build(b: *std.Build) !void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const backends = b.option([]Backend, "backends", "") orelse &.{};
    _ = backends; // autofix

    const lib_cimgui = b.addLibrary(.{
        .linkage = b.option(std.builtin.LinkMode, "linkage", "default to static") orelse .static,
        .name = "cimgui_clib",
        .root_module = b.createModule(.{
            .target = target,
            .optimize = optimize,
        }),
    });
    lib_cimgui.linkLibCpp();
    lib_cimgui.linkLibC();

    {
        const flags: []const []const u8 = if (target.result.cpu.arch.isWasm()) &.{"-fno-sanitize=undefined"} else &.{};

        lib_cimgui.addCSourceFiles(.{
            .root = b.dependency("imgui", .{}).path(""),
            .files = &.{
                "imgui_demo.cpp",
                "imgui_draw.cpp",
                "imgui_tables.cpp",
                "imgui_widgets.cpp",
                "imgui.cpp",
            },
            .flags = flags,
        });

        lib_cimgui.addCSourceFile(.{ .file = b.path("./src/regular/dcimgui.cpp"), .flags = flags });

        lib_cimgui.addCSourceFiles(.{
            .root = b.dependency("imgui", .{}).path("backends/"),
            .files = &.{ "imgui_impl_vulkan.cpp", "imgui_impl_sdl3.cpp" },
            .flags = flags,
        });

        // for (backend) |backend| {
        //     @tagName(backend);
        //     lib_cimgui.addCSourceFile(.{ .file = b.path(.src/backends) })
        // }

        lib_cimgui.addCSourceFiles(.{
            .root = b.path("./src/backends"),
            .files = &.{ "dcimgui_impl_vulkan.cpp", "dcimgui_impl_sdl3.cpp" },
            .flags = flags,
        });

        lib_cimgui.root_module.addCMacro("IMGUI_IMPL_VULKAN_NO_PROTOTYPES", ""); // TODO: add option to define macros
    }

    lib_cimgui.installHeadersDirectory(b.path("src/regular/"), "", .{});
    lib_cimgui.installHeadersDirectory(b.path("src/backends/"), "", .{});
    lib_cimgui.installHeadersDirectory(b.dependency("imgui", .{}).path(""), "", .{});

    lib_cimgui.addIncludePath(b.path("src/regular/"));
    lib_cimgui.addIncludePath(b.path("src/backends/"));
    lib_cimgui.addIncludePath(b.dependency("imgui", .{}).path(""));
    lib_cimgui.addIncludePath(b.dependency("imgui", .{}).path("backends/"));

    const header_path_list = b.option([]std.Build.LazyPath, "include-path-list", "list of path to headers to be included for compiling the various backends that need it") orelse &.{};
    for (header_path_list) |headers_path| {
        lib_cimgui.addIncludePath(headers_path);
    }

    b.installArtifact(lib_cimgui);

    {
        const generator = b.addWriteFiles();
        _ = generator.addCopyDirectory(b.dependency("dear_bindings", .{}).path(""), "", .{});
        _ = generator.addCopyDirectory(b.dependency("ply", .{}).path("src"), "", .{});

        const generator_path = try generator.getDirectory().join(b.allocator, "dear_bindings.py");
        const python = try b.findProgram(&.{ "py", "python3", "python" }, &.{});

        const regenerate = b.step("gen", "");

        const delete_docking = ClearDir.create(b, b.path("./src/docking/"));
        const delete_regular = ClearDir.create(b, b.path("./src/regular/"));
        const delete_backends = ClearDir.create(b, b.path("./src/backends/"));

        regenerate.dependOn(&delete_docking.step);
        regenerate.dependOn(&delete_regular.step);
        regenerate.dependOn(&delete_backends.step);

        for (
            [_][]const u8{ "imgui", "imgui_docking" },
            [_][]const u8{ "src/regular/dcimgui", "src/docking/dcimgui" },
        ) |dep_name, src_path| {
            const run = b.addSystemCommand(&.{python});
            run.addFileArg(generator_path);

            run.addArg("-o");
            run.addFileArg(b.path(src_path));

            run.addFileArg(b.dependency(dep_name, .{}).path("imgui.h"));

            regenerate.dependOn(&run.step);
        }

        inline for (@typeInfo(Backend).@"enum".fields) |field| {
            const run = b.addSystemCommand(&.{python});
            run.addFileArg(generator_path);
            run.addArg("--backend");

            run.addArg("--include");
            run.addFileArg(b.dependency("imgui", .{}).path("imgui.h"));

            run.addArg("-o");
            run.addFileArg(b.path(b.fmt("src/backends/dc{s}", .{field.name})));

            run.addFileArg(b.dependency("imgui", .{}).path(b.fmt("backends/{s}.h", .{field.name})));

            regenerate.dependOn(&run.step);
        }
    }
}

const ClearDir = struct {
    step: std.Build.Step,
    dir: std.Build.LazyPath,

    fn create(b: *std.Build, dir: std.Build.LazyPath) *ClearDir {
        const clear_dir = b.allocator.create(ClearDir) catch @panic("OOM");
        clear_dir.* = .{ .step = .init(.{ .id = .custom, .name = "clearing dir", .owner = b, .makeFn = make }), .dir = dir };
        return clear_dir;
    }

    fn make(step: *std.Build.Step, _: std.Build.Step.MakeOptions) !void {
        const clear_dir: *ClearDir = @fieldParentPtr("step", step);
        const dir_path = clear_dir.dir.getPath2(step.owner, null);
        const dir = try std.fs.openDirAbsolute(dir_path, .{ .iterate = true });

        var it = dir.iterate();
        while (try it.next()) |entry| {
            switch (entry.kind) {
                .directory => try std.fs.Dir.deleteTree(dir, entry.name),
                else => try std.fs.Dir.deleteFile(dir, entry.name),
            }
        }
    }
};

const DeleteJson = struct {
    step: std.Build.Step,
    dir: std.Build.LazyPath,

    fn create(b: *std.Build, dir: std.Build.LazyPath) *DeleteJson {
        const delete_json = b.allocator.create(DeleteJson) catch @panic("OOM");
        delete_json.* = .{ .step = .init(.{ .id = .custom, .name = "clearing dir", .owner = b, .makeFn = make }), .dir = dir };
        return delete_json;
    }

    fn make(step: *std.Build.Step, _: std.Build.Step.MakeOptions) !void {
        const clear_dir: *DeleteJson = @fieldParentPtr("step", step);
        const dir_path = clear_dir.dir.getPath2(step.owner, null);
        const dir = try std.fs.openDirAbsolute(dir_path, .{ .iterate = true });

        var it = dir.iterate();
        while (try it.next()) |entry| {
            switch (entry.kind) {
                .file => if (std.mem.eql(u8, std.fs.path.extension(entry.name), ".json"))
                    dir.deleteFile(entry.name),
                else => {},
            }
        }
    }
};

const Backend = enum {
    imgui_impl_allegro5,
    imgui_impl_android,
    imgui_impl_dx10,
    imgui_impl_dx11,
    imgui_impl_dx12,
    imgui_impl_dx9,
    imgui_impl_glfw,
    imgui_impl_glut,
    imgui_impl_opengl2,
    imgui_impl_opengl3,
    imgui_impl_sdl2,
    imgui_impl_sdl3,
    imgui_impl_sdlrenderer2,
    imgui_impl_sdlrenderer3,
    imgui_impl_win32,
    imgui_impl_vulkan,
    // imgui_impl_metal, // unsupported
    // imgui_impl_osx, // unsupported
    // imgui_impl_sdlgpu3, // unsupported
};
