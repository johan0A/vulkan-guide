pub const vec = @import("vector.zig");
pub const matrix = @import("matrix.zig");
pub const Mat = matrix.Mat;
pub const Quat = @import("quaternion.zig").Quat;

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
