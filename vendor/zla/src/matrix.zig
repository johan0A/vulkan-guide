pub const ClipDepth = enum {
    /// OpenGL
    negative_one_to_one,
    /// Vulkan, DirectX, Metal
    zero_to_one,
};

pub const YDirection = enum {
    /// OpenGL, DirectX
    y_up,
    /// Vulkan
    y_down,
};

pub const Layout = enum {
    /// column major
    ///
    /// items[col][row] OpenGL/Vulkan/GLM
    cm,
    /// row major
    ///
    /// items[row][col] DirectX
    rm,
};

pub const Config = struct {
    clip_depth: ClipDepth = .zero_to_one,
    y_direction: YDirection = .y_down,
};

pub fn Mat(
    comptime layout: Layout,
    comptime T: type,
    comptime cols_: usize,
    comptime rows_: usize,
) type {
    const cm = layout == .cm;

    return extern struct {
        const Self = @This();

        pub const rows: comptime_int = rows_;
        pub const cols: comptime_int = cols_;
        pub const Type: type = T;

        pub const n_major: comptime_int = if (cm) cols else rows;
        pub const n_minor: comptime_int = if (cm) rows else cols;

        items: [n_major][n_minor]T,

        pub inline fn at(self: Self, r: usize, c: usize) T {
            return if (cm) self.items[c][r] else self.items[r][c];
        }

        pub inline fn put(self: *Self, r: usize, c: usize, val: T) void {
            if (cm) {
                self.items[c][r] = val;
            } else {
                self.items[r][c] = val;
            }
        }

        pub const identity: Self = blk: {
            if (rows != cols) @compileError("identity matrix must be square");
            var result: Self = .zero;
            for (0..rows) |i| {
                result.items[i][i] = 1;
            }
            break :blk result;
        };

        pub const zero: Self = .{ .items = @splat(@splat(0)) };

        /// Import from a column-major [cols][rows] array.
        pub inline fn fromColumnMajorArray(values: [cols][rows]T) Self {
            if (cm) {
                return .{ .items = values };
            } else {
                var result: Self = undefined;
                for (0..cols) |c| {
                    for (0..rows) |r| {
                        result.items[r][c] = values[c][r];
                    }
                }
                return result;
            }
        }

        /// Import from a row-major [rows][cols] array.
        pub inline fn fromRowMajorArray(values: [rows][cols]T) Self {
            if (cm) {
                var result: Self = undefined;
                for (0..rows) |r| {
                    for (0..cols) |c| {
                        result.items[c][r] = values[r][c];
                    }
                }
                return result;
            } else {
                return .{ .items = values };
            }
        }

        pub fn transpose(self: Self) Mat(layout, T, rows, cols) {
            var result: Mat(layout, T, rows, cols) = undefined;
            for (0..cols) |c| {
                for (0..rows) |r| {
                    result.put(c, r, self.at(r, c));
                }
            }
            return result;
        }

        pub fn scalarMul(self: Self, scalar: T) Self {
            var result: Self = undefined;
            for (0..n_major) |i| {
                const s: @Vector(n_minor, T) = self.items[i];
                result.items[i] = s * @as(@Vector(n_minor, T), @splat(scalar));
            }
            return result;
        }

        pub fn scalarDiv(self: Self, scalar: T) Self {
            var result: Self = undefined;
            for (0..n_major) |i| {
                const s: @Vector(n_minor, T) = self.items[i];
                result.items[i] = s / @as(@Vector(n_minor, T), @splat(scalar));
            }
            return result;
        }

        pub fn negate(self: Self) Self {
            var result: Self = undefined;
            for (0..n_major) |i| {
                const s: @Vector(n_minor, T) = self.items[i];
                result.items[i] = -s;
            }
            return result;
        }

        pub fn add(self: Self, other: Self) Self {
            var result: Self = undefined;
            for (0..n_major) |i| {
                const a: @Vector(n_minor, T) = self.items[i];
                const b: @Vector(n_minor, T) = other.items[i];
                result.items[i] = a + b;
            }
            return result;
        }

        pub fn sub(self: Self, other: Self) Self {
            var result: Self = undefined;
            for (0..n_major) |i| {
                const a: @Vector(n_minor, T) = self.items[i];
                const b: @Vector(n_minor, T) = other.items[i];
                result.items[i] = a - b;
            }
            return result;
        }

        pub inline fn selfAdd(self: *Self, other: Self) void {
            self.* = self.add(other);
        }

        pub inline fn selfSub(self: *Self, other: Self) void {
            self.* = self.sub(other);
        }

        pub fn mul(self: Self, other: anytype) Mat(layout, T, @TypeOf(other).cols, Self.rows) {
            const Other = @TypeOf(other);
            if (Self.cols != Other.rows) @compileError("column count of lhs must equal row count of rhs");
            if (Self.Type != Other.Type) @compileError("element types must match");

            const Result = Mat(layout, T, Other.cols, Self.rows);
            var result: Result = undefined;

            if (cm) {
                for (0..Result.cols) |i| {
                    var strip: @Vector(Self.rows, T) = @as(@Vector(Self.rows, T), self.items[0]) *
                        @as(@Vector(Self.rows, T), @splat(other.items[i][0]));
                    for (1..Self.cols) |j| {
                        strip += @as(@Vector(Self.rows, T), self.items[j]) *
                            @as(@Vector(Self.rows, T), @splat(other.items[i][j]));
                    }
                    result.items[i] = strip;
                }
            } else {
                for (0..Result.rows) |i| {
                    var strip: @Vector(Other.cols, T) = @as(@Vector(Other.cols, T), other.items[0]) *
                        @as(@Vector(Other.cols, T), @splat(self.items[i][0]));
                    for (1..Self.cols) |k| {
                        strip += @as(@Vector(Other.cols, T), other.items[k]) *
                            @as(@Vector(Other.cols, T), @splat(self.items[i][k]));
                    }
                    result.items[i] = strip;
                }
            }
            return result;
        }

        pub fn mulVec(self: Self, v: @Vector(cols, T)) @Vector(rows, T) {
            if (cm) {
                var result: @Vector(rows, T) = @as(@Vector(rows, T), self.items[0]) *
                    @as(@Vector(rows, T), @splat(v[0]));
                inline for (1..cols) |j| {
                    result += @as(@Vector(rows, T), self.items[j]) *
                        @as(@Vector(rows, T), @splat(v[j]));
                }
                return result;
            } else {
                var result: [rows]T = undefined;
                inline for (0..rows) |i| {
                    result[i] = @reduce(.Add, @as(@Vector(cols, T), self.items[i]) * v);
                }
                return result;
            }
        }

        pub inline fn selfMul(self: *Self, other: anytype) void {
            self.* = self.mul(other);
        }

        pub fn determinant(self: Self) T {
            if (rows != cols) @compileError("determinant requires a square matrix");
            return cofactorDet(rows, self.items);
        }

        pub fn inverse(self: Self) ?Self {
            if (rows != cols) @compileError("inverse requires a square matrix");

            const d = self.determinant();
            if (d == 0) return null;
            const inv_det = 1.0 / d;

            var result: Self = undefined;
            inline for (0..rows) |r| {
                inline for (0..cols) |c| {
                    const sign: T = if ((c + r) % 2 == 0) 1 else -1;
                    const minor = logicalSubmatrix(rows, self.items, c, r);
                    result.put(r, c, sign * cofactorDet(rows - 1, minor) * inv_det);
                }
            }
            return result;
        }

        fn cofactorDet(comptime n: usize, items: [n][n]T) T {
            if (n == 0) @compileError("matrix dimensions must be > 0");
            if (n == 1) return items[0][0];

            var result: T = 0;
            inline for (0..n) |col| {
                const sign: T = if (col % 2 == 0) 1 else -1;
                result += sign * items[col][0] * cofactorDet(n - 1, rawSubmatrix(n, items, col, 0));
            }
            return result;
        }

        fn rawSubmatrix(
            comptime n: usize,
            items: [n][n]T,
            comptime skip_outer: usize,
            comptime skip_inner: usize,
        ) [n - 1][n - 1]T {
            var result: [n - 1][n - 1]T = undefined;
            inline for (0..n - 1) |a| {
                const sa = if (a < skip_outer) a else a + 1;
                inline for (0..n - 1) |b| {
                    const sb = if (b < skip_inner) b else b + 1;
                    result[a][b] = items[sa][sb];
                }
            }
            return result;
        }

        fn logicalSubmatrix(
            comptime n: usize,
            items: [n][n]T,
            comptime skip_row: usize,
            comptime skip_col: usize,
        ) [n - 1][n - 1]T {
            return switch (cm) {
                true => rawSubmatrix(n, items, skip_col, skip_row),
                false => rawSubmatrix(n, items, skip_row, skip_col),
            };
        }

        pub fn perspective(fovy: T, aspect: T, near: T, far: T, config: Config) Self {
            if (rows != 4 or cols != 4) @compileError("perspective matrix must be 4x4");

            const f = 1.0 / std.math.tan(fovy / 2.0);
            const y_sign: T = if (config.y_direction == .y_down) -1.0 else 1.0;

            var result: Self = .zero;
            result.put(0, 0, f / aspect);
            result.put(1, 1, f * y_sign);
            result.put(3, 2, -1.0);

            switch (config.clip_depth) {
                .zero_to_one => {
                    result.put(2, 2, far / (near - far));
                    result.put(2, 3, -(far * near) / (far - near));
                },
                .negative_one_to_one => {
                    result.put(2, 2, -(far + near) / (far - near));
                    result.put(2, 3, -(2.0 * far * near) / (far - near));
                },
            }

            return result;
        }

        pub fn perspectiveReverseZ(fovy: T, aspect: T, near: T, config: Config) Self {
            if (rows != 4 or cols != 4) @compileError("perspective matrix must be 4x4");
            std.debug.assert(config.clip_depth == .zero_to_one); // reverse-Z requires zero_to_one clip depth

            const f = 1.0 / std.math.tan(fovy / 2.0);
            const y_sign: T = if (config.y_direction == .y_down) -1.0 else 1.0;

            var result: Self = .zero;
            result.put(0, 0, f / aspect);
            result.put(1, 1, f * y_sign);
            result.put(3, 2, -1.0);
            result.put(2, 3, near);
            return result;
        }

        pub fn orthographic(left: T, right: T, bottom: T, top: T, near: T, far: T, config: Config) Self {
            if (rows != 4 or cols != 4) @compileError("orthographic matrix must be 4x4");

            const y_sign: T = if (config.y_direction == .y_down) -1.0 else 1.0;
            const rl = right - left;
            const tb = top - bottom;
            const fn_ = far - near;

            var result: Self = .zero;
            result.put(0, 0, 2.0 / rl);
            result.put(1, 1, y_sign * 2.0 / tb);
            result.put(0, 3, -(right + left) / rl);
            result.put(1, 3, -(top + bottom) / tb);
            result.put(3, 3, 1.0);

            switch (config.clip_depth) {
                .zero_to_one => {
                    result.put(2, 2, -1.0 / fn_);
                    result.put(2, 3, -near / fn_);
                },
                .negative_one_to_one => {
                    result.put(2, 2, -2.0 / fn_);
                    result.put(2, 3, -(far + near) / fn_);
                },
            }

            return result;
        }

        pub fn lookAt(eye: @Vector(3, T), center: @Vector(3, T), up: @Vector(3, T)) Self {
            if (rows != 4 or cols != 4) @compileError("lookAt matrix must be 4x4");

            const f = vec.normalize(center - eye);
            const s = vec.normalize(vec.cross(f, up));
            const u = vec.cross(s, f);

            var result: Self = .identity;
            result.put(0, 0, s[0]);
            result.put(0, 1, s[1]);
            result.put(0, 2, s[2]);

            result.put(1, 0, u[0]);
            result.put(1, 1, u[1]);
            result.put(1, 2, u[2]);

            result.put(2, 0, -f[0]);
            result.put(2, 1, -f[1]);
            result.put(2, 2, -f[2]);

            result.put(0, 3, -vec.dot(s, eye));
            result.put(1, 3, -vec.dot(u, eye));
            result.put(2, 3, vec.dot(f, eye));

            return result;
        }

        pub fn translate(self: Self, vector: @Vector(rows - 1, T)) Self {
            if (rows != cols) @compileError("transform matrix must be square");

            var result = self;
            inline for (0..rows) |r| {
                var sum: T = self.at(r, cols - 1);
                inline for (0..rows - 1) |i| {
                    sum += self.at(r, i) * vector[i];
                }
                result.put(r, cols - 1, sum);
            }
            return result;
        }

        pub inline fn selfTranslate(self: *Self, vector: @Vector(rows - 1, T)) void {
            self.* = self.translate(vector);
        }

        pub inline fn position(self: Self) @Vector(rows - 1, T) {
            if (rows != cols) @compileError("transform matrix must be square");
            var result: [rows - 1]T = undefined;
            inline for (0..rows - 1) |r| {
                result[r] = self.at(r, cols - 1);
            }
            return result;
        }

        pub fn scale(self: Self, factors: @Vector(rows - 1, T)) Self {
            if (rows != cols) @compileError("transform matrix must be square");

            var result = self;
            inline for (0..rows - 1) |i| {
                inline for (0..rows) |r| {
                    result.put(r, i, self.at(r, i) * factors[i]);
                }
            }
            return result;
        }

        pub inline fn selfScale(self: *Self, factors: @Vector(rows - 1, T)) void {
            self.* = self.scale(factors);
        }

        pub fn rotate(self: Self, angle: T, axis: @Vector(3, T)) Self {
            if (rows != cols) @compileError("rotate requires a square matrix");
            if (rows < 3) @compileError("rotate requires at least 3x3");

            const a = vec.normalize(axis);
            const co = std.math.cos(angle);
            const s = std.math.sin(angle);
            const t = 1.0 - co;

            var rot: Self = .identity;
            rot.put(0, 0, t * a[0] * a[0] + co);
            rot.put(1, 0, t * a[0] * a[1] + s * a[2]);
            rot.put(2, 0, t * a[0] * a[2] - s * a[1]);
            rot.put(0, 1, t * a[0] * a[1] - s * a[2]);
            rot.put(1, 1, t * a[1] * a[1] + co);
            rot.put(2, 1, t * a[1] * a[2] + s * a[0]);
            rot.put(0, 2, t * a[0] * a[2] + s * a[1]);
            rot.put(1, 2, t * a[1] * a[2] - s * a[0]);
            rot.put(2, 2, t * a[2] * a[2] + co);

            return self.mul(rot);
        }

        pub inline fn selfRotate(self: *Self, angle: T, axis: @Vector(3, T)) void {
            self.* = self.rotate(angle, axis);
        }

        pub fn fromAxisAngle(axis: @Vector(3, T), angle: T) Self {
            if (rows != cols) @compileError("fromAxisAngle requires a square matrix");
            if (rows < 3) @compileError("fromAxisAngle requires at least 3x3");

            return Self.identity.rotate(angle, axis);
        }

        pub fn eql(self: Self, other: Self) bool {
            for (0..n_major) |i| {
                for (0..n_minor) |j| {
                    if (self.items[i][j] != other.items[i][j]) return false;
                }
            }
            return true;
        }

        pub fn approxEql(self: Self, other: Self, tolerance: T) bool {
            for (0..n_major) |i| {
                for (0..n_minor) |j| {
                    if (@abs(self.items[i][j] - other.items[i][j]) > tolerance) return false;
                }
            }
            return true;
        }

        pub fn format(self: @This(), writer: *std.Io.Writer) std.Io.Writer.Error!void {
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
    };
}

test "format" {
    const c: Mat(.cm, f32, 3, 3) = .fromRowMajorArray(.{
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
    const c: Mat(.cm, f32, 4, 4) = .translate(.identity, .{ 1, 2, 3 });
    const expected: Mat(.cm, f32, 4, 4) = .fromRowMajorArray(.{
        .{ 1, 0, 0, 1 },
        .{ 0, 1, 0, 2 },
        .{ 0, 0, 1, 3 },
        .{ 0, 0, 0, 1 },
    });
    try std.testing.expectEqual(expected, c);
}

test "translate with rotation" {
    const rot: Mat(.cm, f32, 4, 4) = .fromAxisAngle(.{ 0, 0, 1 }, std.math.pi / 2.0);
    const translated = rot.translate(.{ 1, 0, 0 });

    try std.testing.expectApproxEqAbs(@as(f32, 0), translated.position()[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1), translated.position()[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0), translated.position()[2], 1e-5);
}

test "mul" {
    {
        const a: Mat(.cm, f32, 2, 2) = .fromRowMajorArray(.{
            .{ 1, 3 },
            .{ 2, 4 },
        });
        const b: Mat(.cm, f32, 2, 2) = .fromRowMajorArray(.{
            .{ 5, 7 },
            .{ 6, 8 },
        });
        const expected: Mat(.cm, f32, 2, 2) = .fromRowMajorArray(.{
            .{ 23, 31 },
            .{ 34, 46 },
        });
        try std.testing.expectEqual(expected, a.mul(b));
    }

    {
        const a: Mat(.cm, f32, 2, 3) = .fromRowMajorArray(.{
            .{ 1, 4 },
            .{ 2, 5 },
            .{ 3, 6 },
        });
        const b: Mat(.cm, f32, 3, 2) = .fromRowMajorArray(.{
            .{ 1, 3, 5 },
            .{ 2, 4, 6 },
        });
        const expected: Mat(.cm, f32, 3, 3) = .fromRowMajorArray(.{
            .{ 9, 19, 29 },
            .{ 12, 26, 40 },
            .{ 15, 33, 51 },
        });
        try std.testing.expectEqual(expected, a.mul(b));
    }

    {
        const a: Mat(.cm, f32, 4, 4) = .fromRowMajorArray(.{
            .{ 1, 5, 9, 13 },
            .{ 2, 6, 10, 14 },
            .{ 3, 7, 11, 15 },
            .{ 4, 8, 12, 16 },
        });
        const b: Mat(.cm, f32, 4, 4) = .fromRowMajorArray(.{
            .{ 17, 21, 25, 29 },
            .{ 18, 22, 26, 30 },
            .{ 19, 23, 27, 31 },
            .{ 20, 24, 28, 32 },
        });
        const expected: Mat(.cm, f32, 4, 4) = .fromRowMajorArray(.{
            .{ 538, 650, 762, 874 },
            .{ 612, 740, 868, 996 },
            .{ 686, 830, 974, 1118 },
            .{ 760, 920, 1080, 1240 },
        });
        try std.testing.expectEqual(expected, a.mul(b));
    }
}

test "mulVec" {
    const m: Mat(.cm, f32, 4, 4) = .fromRowMajorArray(.{
        .{ 1, 0, 0, 5 },
        .{ 0, 1, 0, 6 },
        .{ 0, 0, 1, 7 },
        .{ 0, 0, 0, 1 },
    });
    const v = m.mulVec(.{ 1, 2, 3, 1 });
    try std.testing.expectEqual(@Vector(4, f32){ 6, 8, 10, 1 }, v);
}

test "scale" {
    {
        const mat: Mat(.cm, f32, 4, 4) = .fromRowMajorArray(.{
            .{ 1, 0, 0, 5 },
            .{ 0, 1, 0, 6 },
            .{ 0, 0, 1, 7 },
            .{ 0, 0, 0, 1 },
        });
        const expected: Mat(.cm, f32, 4, 4) = .fromRowMajorArray(.{
            .{ 2, 0, 0, 5 },
            .{ 0, 3, 0, 6 },
            .{ 0, 0, 4, 7 },
            .{ 0, 0, 0, 1 },
        });
        try std.testing.expectEqual(expected, mat.scale(.{ 2, 3, 4 }));
    }

    {
        const mat: Mat(.cm, f32, 3, 3) = .fromRowMajorArray(.{
            .{ 0.707, -0.707, 0 },
            .{ 0.707, 0.707, 0 },
            .{ 0, 0, 1 },
        });
        const expected: Mat(.cm, f32, 3, 3) = .fromRowMajorArray(.{
            .{ 1.414, -2.121, 0 },
            .{ 1.414, 2.121, 0 },
            .{ 0, 0, 1 },
        });

        const scaled = mat.scale(.{ 2, 3 });
        for (0..3) |c| {
            for (0..3) |r| {
                try std.testing.expectApproxEqAbs(expected.items[c][r], scaled.items[c][r], 0.001);
            }
        }
    }
}

test "determinant" {
    const m: Mat(.cm, f32, 3, 3) = .fromRowMajorArray(.{
        .{ 1, 2, 3 },
        .{ 0, 1, 4 },
        .{ 5, 6, 0 },
    });
    try std.testing.expectApproxEqAbs(@as(f32, 1), m.determinant(), 1e-5);
}

test "inverse 4x4" {
    const m: Mat(.cm, f32, 4, 4) = .fromRowMajorArray(.{
        .{ 1, 0, 0, 3 },
        .{ 0, 2, 0, 0 },
        .{ 0, 0, 1, -1 },
        .{ 0, 0, 0, 1 },
    });
    const inv = m.inverse() orelse return error.Singular;
    const product = m.mul(inv);

    for (0..4) |c| {
        for (0..4) |r| {
            const expected: f32 = if (r == c) 1.0 else 0.0;
            try std.testing.expectApproxEqAbs(expected, product.items[c][r], 1e-5);
        }
    }
}

const std = @import("std");
const vec = @import("root.zig").vec;
const matrix = @This();
