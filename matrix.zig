/// column major generic matrix type
pub fn Mat(comptime T: type, comptime cols_: usize, comptime rows_: usize) type {
    return extern struct {
        const Self = @This();

        items: [cols][rows]T,

        pub const rows: comptime_int = rows_;
        pub const cols: comptime_int = cols_;
        pub const Type: type = T;

        pub const identity = blk: {
            if (rows != cols) @compileError("Identity matrix must be square");
            var result: Self = .zero;
            for (0..cols) |i| {
                result.items[i][i] = 1;
            }
            break :blk result;
        };

        pub const zero: Self = .fromColumnMajorArray(@splat(@splat(0)));

        pub inline fn fromColumnMajorArray(values: [cols][rows]T) Self {
            return .{ .items = values };
        }

        /// performs a transpose operation, usefull for more human readable matrix literals
        pub inline fn fromRowMajorArray(values: [rows][cols]T) Self {
            return Mat(T, rows, cols).fromColumnMajorArray(values).transpose();
        }

        // generated code seams fast but tbd
        pub fn transpose(self: Self) Mat(T, rows, cols) {
            var result: Mat(T, rows, cols) = .fromColumnMajorArray(undefined);
            for (0..cols) |c| {
                for (0..rows) |r| {
                    result.items[r][c] = self.items[c][r];
                }
            }
            return result;
        }

        /// Scalar multiplication
        pub fn scalarMul(self: Self, scalar: T) Self {
            const items: [rows * cols]T = @bitCast(self.items);
            var result_items: [rows * cols]T = undefined;
            for (&result_items, items) |*result_item, item| {
                result_item.* = item * scalar;
            }
            return .fromColumnMajorArray(@bitCast(result_items));
        }

        // `other` must be a matrix with the same number of rows as the numbers of columns of `self`
        pub fn mul(self: Self, other: anytype) Mat(T, @TypeOf(other).cols, Self.rows) {
            if (Self.cols != @TypeOf(other).rows) @compileError("number of columns of self must be equal to number of rows of other");
            if (Self.Type != @TypeOf(other).Type) @compileError("type of self must be equal to Type of other");
            const Wt = @Vector(rows, T);

            var result: Mat(T, @TypeOf(other).cols, Self.rows) = undefined;
            for (0..@TypeOf(result).cols) |i| {
                result.items[i] = @as(Wt, self.items[0]) * @as(Wt, @splat(other.items[i][0]));
                for (1..Self.cols) |j| {
                    result.items[i] = @as(Wt, result.items[i]) + @as(Wt, self.items[j]) * @as(Wt, @splat(other.items[i][j]));
                }
            }
            return result;
        }

        pub inline fn selfMul(self: *Self, other: anytype) void {
            self.* = self.mul(other);
        }

        pub fn add(self: Self, other: Self) Self {
            var result = Self{ .items = undefined };
            for (0..cols) |c| {
                for (0..rows) |r| {
                    result.items[c][r] = self.items[c][r] + other.items[c][r];
                }
            }
            return result;
        }

        pub inline fn selfAdd(self: *Self, other: Self) void {
            self.* = self.add(other);
        }

        pub fn sub(self: Self, other: Self) Self {
            var result: Self = .{ .items = undefined };
            for (0..cols) |c| {
                for (0..rows) |r| {
                    result.items[c][r] = self.items[c][r] - other.items[c][r];
                }
            }
            return result;
        }

        pub inline fn selfSub(self: *Self, other: Self) void {
            self.* = self.sub(other);
        }

        pub const ClipControl = enum {
            LH_ZO, // Left-handed, Z to [0,1]
            LH_NO, // Left-handed, Z to [-1,1]
            RH_ZO, // Right-handed, Z to [0,1] (Vulkan default)
            RH_NO, // Right-handed, Z to [-1,1] (OpenGL default)
        };

        /// create a perspective projection matrix
        pub fn perspective(comptime clip_control: ClipControl, fovy: T, aspect: T, near: T, far: T) Self {
            if (rows != 4 or cols != 4) @compileError("Perspective matrix must be 4x4");

            const tanHalfFovy = std.math.tan(fovy / 2);

            var result: Self = .zero;
            result.items[0][0] = 1.0 / (aspect * tanHalfFovy);
            result.items[1][1] = 1.0 / tanHalfFovy;

            switch (clip_control) {
                .RH_ZO => {
                    result.items[2][2] = far / (near - far);
                    result.items[2][3] = -1.0;
                    result.items[3][2] = -(far * near) / (far - near);
                },
                .RH_NO => {
                    result.items[2][2] = -(far + near) / (far - near);
                    result.items[2][3] = -1.0;
                    result.items[3][2] = -(2.0 * far * near) / (far - near);
                },
                .LH_ZO => {
                    result.items[2][2] = far / (far - near);
                    result.items[2][3] = 1.0;
                    result.items[3][2] = -(far * near) / (far - near);
                },
                .LH_NO => {
                    result.items[2][2] = (far + near) / (far - near);
                    result.items[2][3] = 1.0;
                    result.items[3][2] = -(2.0 * far * near) / (far - near);
                },
            }

            return result;
        }

        /// create a look-at view matrix
        pub fn lookAt(eye: @Vector(3, T), center: @Vector(3, T), up: @Vector(3, T)) Self {
            if (rows != 4 or cols != 4) @compileError("Look-at matrix must be 4x4");

            const f = vec.normalize(center - eye);
            const s = vec.normalize(vec.cross(f, up));
            const u = vec.cross(s, f);

            var result: Self = .identity;
            result.items[0][0] = s[0];
            result.items[1][0] = s[1];
            result.items[2][0] = s[2];

            result.items[0][1] = u[0];
            result.items[1][1] = u[1];
            result.items[2][1] = u[2];

            result.items[0][2] = -f[0];
            result.items[1][2] = -f[1];
            result.items[2][2] = -f[2];

            result.items[3][0] = -vec.dot(s, eye);
            result.items[3][1] = -vec.dot(u, eye);
            result.items[3][2] = vec.dot(f, eye);

            return result;
        }

        pub fn translate(self: Self, vector: @Vector(rows - 1, T)) Self {
            if (rows != cols) @compileError("Transform matrix must be square");
            var result = self;
            result.items[cols - 1][0 .. rows - 1].* = self.items[cols - 1][0 .. rows - 1].* + vector;
            return result;
        }

        pub inline fn selfTranslate(self: *Self, vector: @Vector(rows - 1, T)) void {
            self.* = self.translate(vector);
        }

        pub inline fn position(self: Self) @Vector(rows - 1, T) {
            if (rows != cols) @compileError("Transform matrix must be square");
            return self.items[cols - 1][0 .. rows - 1].*;
        }

        /// Scaling transform matrix
        pub fn scale(self: Self, factors: @Vector(rows - 1, T)) Self {
            if (rows != cols) @compileError("Transform matrix must be square");

            var result = self;
            for (0..rows - 1) |i| {
                result.items[i][0 .. rows - 1].* = self.items[i][0 .. rows - 1].* * @as(@Vector(rows - 1, T), @splat(factors[i]));
            }
            return result;
        }

        pub inline fn selfScale(self: *Self, vector: @Vector(rows - 1, T)) void {
            self.* = self.scale(vector);
        }

        pub fn rotate(self: Self, angle: T, axis: @Vector(3, T)) Self {
            if (rows != cols) @compileError("Transform matrix must be square");
            if (rows != 4 or cols != 4) @compileError("unsuported dimensions, only suports 4x4");

            const a = vec.normalize(axis);
            const c = std.math.cos(angle);
            const s = std.math.sin(angle);
            const t = 1.0 - c;

            const rot: Self = .{
                .items = .{
                    .{ t * a[0] * a[0] + c, t * a[0] * a[1] - s * a[2], t * a[0] * a[2] + s * a[1], 0 },
                    .{ t * a[0] * a[1] + s * a[2], t * a[1] * a[1] + c, t * a[1] * a[2] - s * a[0], 0 },
                    .{ t * a[0] * a[2] - s * a[1], t * a[1] * a[2] + s * a[0], t * a[2] * a[2] + c, 0 },
                    .{ 0, 0, 0, 1 },
                },
            };

            return self.mul(rot);
        }

        pub inline fn selfRotate(self: *Self, angle: T, axis: @Vector(3, T)) void {
            self.* = self.rotate(angle, axis);
        }

        pub fn format(self: @This(), writer: *std.io.Writer) std.io.Writer.Error!void {
            var max_widths: [cols]usize = [_]usize{0} ** cols;

            for (0..cols) |c| {
                for (0..rows) |r| {
                    const len = std.fmt.count("{d}", .{self.items[c][r]});
                    max_widths[c] = @max(max_widths[c], len);
                }
            }

            for (0..rows) |r| {
                try writer.writeAll("[");
                for (0..cols) |c| {
                    const len = std.fmt.count("{d}", .{self.items[c][r]});
                    for (0..max_widths[c] - len) |_| {
                        try writer.writeByte(' ');
                    }
                    try writer.print("{d}", .{self.items[c][r]});
                    if (c < cols - 1) try writer.writeAll(", ");
                }
                try writer.writeByte(']');
                if (r != rows - 1) try writer.writeByte('\n');
            }
        }

        pub fn eql(self: Self, other: Self) bool {
            for (0..cols) |c| {
                for (0..rows) |r| {
                    if (self.items[c][r] != other.items[c][r]) {
                        return false;
                    }
                }
            }
            return true;
        }
    };
}

test "format" {
    const c: Mat(f32, 3, 3) = .fromRowMajorArray(.{
        .{ 9, 12, 15 },
        .{ 19, 26, 33 },
        .{ 29, 40, 51 },
    });
    var buff: [128]u8 = undefined;
    const result = try std.fmt.bufPrint(&buff, "{f}", .{c});
    try std.testing.expectEqualStrings(
        \\[ 9, 12, 15]
        \\[19, 26, 33]
        \\[29, 40, 51]
    , result);
}

test "translate" {
    const c: Mat(f32, 4, 4) = .translate(.identity, .{ 1, 2, 3 });
    const excpected_c: Mat(f32, 4, 4) = .fromRowMajorArray(.{
        .{ 1, 0, 0, 1 },
        .{ 0, 1, 0, 2 },
        .{ 0, 0, 1, 3 },
        .{ 0, 0, 0, 1 },
    });
    try std.testing.expectEqual(excpected_c, c);
}

test "mul" {
    {
        const a: Mat(f32, 2, 2) = .fromRowMajorArray(.{
            .{ 1, 3 },
            .{ 2, 4 },
        });
        const b: Mat(f32, 2, 2) = .fromRowMajorArray(.{
            .{ 5, 7 },
            .{ 6, 8 },
        });
        const c = a.mul(b);

        const excpected_c: Mat(f32, 2, 2) = .fromRowMajorArray(.{
            .{ 23, 31 },
            .{ 34, 46 },
        });
        try std.testing.expectEqual(excpected_c, c);
    }

    {
        const a: Mat(f32, 2, 3) = .fromRowMajorArray(.{
            .{ 1, 4 },
            .{ 2, 5 },
            .{ 3, 6 },
        });
        const b: Mat(f32, 3, 2) = .fromRowMajorArray(.{
            .{ 1, 3, 5 },
            .{ 2, 4, 6 },
        });
        const c = a.mul(b);

        const excpected_c: Mat(f32, 3, 3) = .fromRowMajorArray(.{
            .{ 9, 19, 29 },
            .{ 12, 26, 40 },
            .{ 15, 33, 51 },
        });
        try std.testing.expectEqual(excpected_c, c);
    }

    {
        const a: Mat(f32, 4, 4) = .fromRowMajorArray(.{
            .{ 1, 5, 9, 13 },
            .{ 2, 6, 10, 14 },
            .{ 3, 7, 11, 15 },
            .{ 4, 8, 12, 16 },
        });
        const b: Mat(f32, 4, 4) = .fromRowMajorArray(.{
            .{ 17, 21, 25, 29 },
            .{ 18, 22, 26, 30 },
            .{ 19, 23, 27, 31 },
            .{ 20, 24, 28, 32 },
        });
        const c = a.mul(b);

        const excpected_c: Mat(f32, 4, 4) = .fromRowMajorArray(.{
            .{ 538, 650, 762, 874 },
            .{ 612, 740, 868, 996 },
            .{ 686, 830, 974, 1118 },
            .{ 760, 920, 1080, 1240 },
        });
        try std.testing.expectEqual(excpected_c, c);
    }
}

test "scale" {
    {
        const mat: Mat(f32, 4, 4) = .fromRowMajorArray(.{
            .{ 1, 0, 0, 5 },
            .{ 0, 1, 0, 6 },
            .{ 0, 0, 1, 7 },
            .{ 0, 0, 0, 1 },
        });
        const scaled = mat.scale(.{ 2, 3, 4 });

        const expected: Mat(f32, 4, 4) = .fromRowMajorArray(.{
            .{ 2, 0, 0, 5 },
            .{ 0, 3, 0, 6 },
            .{ 0, 0, 4, 7 },
            .{ 0, 0, 0, 1 },
        });
        try std.testing.expectEqual(expected, scaled);
    }

    {
        const mat: Mat(f32, 3, 3) = .fromRowMajorArray(.{
            .{ 0.707, -0.707, 0 },
            .{ 0.707, 0.707, 0 },
            .{ 0, 0, 1 },
        });

        const scaled = mat.scale(.{ 2, 3 });

        const expected: Mat(f32, 3, 3) = .fromRowMajorArray(.{
            .{ 1.414, -2.121, 0 },
            .{ 1.414, 2.121, 0 },
            .{ 0, 0, 1 },
        });

        for (0..3) |c| {
            for (0..3) |r| {
                try std.testing.expectApproxEqAbs(expected.items[c][r], scaled.items[c][r], 0.001);
            }
        }
    }
}

const std = @import("std");
const vec = @import("root.zig").vec;
