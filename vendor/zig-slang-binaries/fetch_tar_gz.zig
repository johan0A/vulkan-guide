const std = @import("std");
const Io = std.Io;

fn printUsageAndExit() noreturn {
    std.log.info("Usage: fetch <url> <output-path>", .{});
    std.process.exit(1);
}

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const arena = init.arena.allocator();
    const args = try init.minimal.args.toSlice(arena);

    if (args.len != 3) {
        std.log.err("missing arguments\n", .{});
        printUsageAndExit();
    }

    const url = args[1];
    const output_path = args[2];

    var client: std.http.Client = .{ .allocator = arena, .io = io };
    defer client.deinit();

    const cwd = Io.Dir.cwd();

    const dest = try cwd.createDirPathOpen(io, output_path, .{});
    defer dest.close(io);

    const uri = try std.Uri.parse(url);
    var req = try client.request(.GET, uri, .{});
    try req.sendBodiless();
    defer req.deinit();

    var redirect_buf: [1024 * 8]u8 = undefined;
    var response = try req.receiveHead(&redirect_buf);
    if (response.head.status != .ok) return error.HttpRequestFailed;

    var transfer_buf: [1024 * 16]u8 = undefined;
    const reader = response.reader(&transfer_buf);
    var window: [std.compress.flate.max_window_len]u8 = undefined;
    var decompress: std.compress.flate.Decompress = .init(reader, .gzip, &window);
    try std.tar.pipeToFileSystem(io, dest, &decompress.reader, .{});
}
