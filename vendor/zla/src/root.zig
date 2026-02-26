pub const vec = @import("vector.zig");
pub const Mat = @import("matrix.zig").Mat;

pub fn toRadians(T: type, degrees: T) @TypeOf(degrees) {
    return degrees * (std.math.pi / 180.0);
}

pub fn toDegrees(radians: anytype) @TypeOf(radians) {
    return radians * (1 / (std.math.pi / 180));
}

test {
    @import("std").testing.refAllDeclsRecursive(@This());
}

const std = @import("std");
