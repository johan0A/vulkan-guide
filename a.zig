const base64_table = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" ++
    "abcdefghijklmnopqrstuvwxyz" ++
    "0123456789+/";

fn encodeBase64(input: []const u8, buff: []u8) void {
    var buff_idx: usize = 0;
    for (0..(input.len / 3) + 1) |i| {
        const bytes: u24 = std.mem.readInt(u24, input[i * 3 ..][0..3], .big);

        inline for (0..4) |j| {
            const shift = (3 - j) * 6;
            const mask = @as(u24, 0b111111);
            buff[buff_idx] = base64_table[(bytes >> shift) & mask];
            buff_idx += 1;
        }
    }
}

pub fn main() !void {
    var buff: [8]u8 = undefined;
    encodeBase64("ManMa", &buff);

    std.debug.print("buff: {s}", .{buff});
}

const std = @import("std");
