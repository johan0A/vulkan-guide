pub const vec = @import("vector.zig");
pub const Mat = @import("matrix.zig").Mat;

pub fn toRadians(T: type, degrees: T) T {
    return degrees * (std.math.pi / 180.0);
}

pub fn toDegrees(T: type, radians: T) T {
    return radians * (1 / (std.math.pi / 180.0));
}

test {
    @import("std").testing.refAllDecls(@This());
}

const std = @import("std");
