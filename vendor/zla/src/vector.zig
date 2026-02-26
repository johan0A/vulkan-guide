const std = @import("std");

const Float = std.meta.Float;

inline fn len(T: type) usize {
    return switch (@typeInfo(T)) {
        .vector => |vector| vector.len,
        .array => |array| array.len,
        else => @compileError("Expected vector or array type, got: " ++ @typeName(T)),
    };
}

fn innerVec(vec: anytype) @Vector(len(@TypeOf(vec)), std.meta.Child(@TypeOf(vec))) {
    const type_info = @typeInfo(@TypeOf(vec));
    if (type_info != .vector and type_info != .array) @compileError("Expected vector or array type, got: " ++ @typeName(@TypeOf(vec)));
    return vec;
}

/// Returns a new vector with the components swizzled.
///
/// example:
/// ```
/// const v: [3]f32 = .{ 1, 2 };
/// const v2 = swizzle("yx"); // <= here x and y are swapped
/// ```
/// v2 is equal to `[2]f32{ 2, 1 }`
///
/// this is also valid:
/// ```
/// const v: [4]f32 = .{ 1, 2, 3, 4 };
/// const v2 = swizzle("wxx");
/// ```
/// here v2 is equal to `[4]f32{ 4, 1, 1 }`
pub fn swizzle(vec: anytype, comptime components: []const u8) @Vector(components.len, std.meta.Child(@TypeOf(vec))) {
    const inner_vec = innerVec(vec);
    const T = std.meta.Child(@TypeOf(inner_vec));

    comptime var mask: [components.len]u8 = undefined;
    inline for (components, 0..) |c, i| {
        switch (c) {
            'x' => mask[i] = 0,
            'y' => mask[i] = 1,
            'z' => mask[i] = 2,
            'w' => mask[i] = 3,
            else => @compileError("swizzle: invalid component"),
        }
    }

    return @shuffle(T, inner_vec, @as(@Vector(1, T), undefined), mask);
}

/// Returns the norm of the vector as a Float.
///
/// the precsion parameter is the number of bits of the output.
/// the precision of the calculations will match the precision of the output type.
pub fn normAdv(vec: anytype, comptime precision: u8) Float(precision) {
    const inner_vec = innerVec(vec);
    const T = std.meta.Child(@TypeOf(inner_vec));

    if (@typeInfo(T) == .int) {
        return std.math.sqrt(
            @as(
                Float(precision),
                @floatFromInt(@reduce(
                    .Add,
                    inner_vec * inner_vec,
                )),
            ),
        );
    } else {
        const items: blk: {
            if (precision > @bitSizeOf(T)) {
                break :blk @Vector(len(@TypeOf(inner_vec)), Float(precision));
            } else {
                break :blk @TypeOf(inner_vec);
            }
        } = inner_vec;

        return @floatCast(std.math.sqrt(
            @reduce(
                .Add,
                items * items,
            ),
        ));
    }
}

/// Returns the norm of the vector as a Float with the default precision.
/// the precision of the output is the number of bits of the output.
/// see `normAdv`
pub inline fn norm(vec: anytype) Float(@bitSizeOf(std.meta.Child(@TypeOf(vec)))) {
    return normAdv(vec, @bitSizeOf(std.meta.Child(@TypeOf(vec))));
}

/// Returns a new vector with the same direction as the original vector, but with a norm closest to 1.
pub fn normalize(vec: anytype) @TypeOf(vec) {
    const inner_vec = innerVec(vec);
    const vec_norm = switch (@typeInfo(std.meta.Child(@TypeOf(inner_vec)))) {
        .float => norm(inner_vec),
        .int => @as(std.meta.Child(@TypeOf(inner_vec)), @intFromFloat(norm(inner_vec))),
        else => unreachable,
    };
    return inner_vec / @as(
        @TypeOf(inner_vec),
        @splat(vec_norm),
    );
}

/// dot product of two vectors.
pub fn dot(vec: anytype, other: anytype) std.meta.Child(@TypeOf(vec)) {
    const inner_vec = innerVec(vec);
    const inner_other: @TypeOf(inner_vec) = other;
    return @reduce(.Add, inner_vec * inner_other);
}

/// Returns the cross product of two vectors.
pub fn cross(vec: anytype, other: anytype) @TypeOf(vec) {
    const inner_vec = innerVec(vec);
    const inner_other: @TypeOf(inner_vec) = other;
    if (len(@TypeOf(inner_vec)) != 3) @compileError("vector must have three elements for cross() to be defined");
    const T = @typeInfo(@TypeOf(inner_vec)).vector.child;

    const vec1 = @shuffle(T, inner_vec, inner_vec, [3]u8{ 1, 2, 0 });
    const vec2 = @shuffle(T, inner_vec, inner_vec, [3]u8{ 2, 0, 1 });
    const other1 = @shuffle(T, inner_other, inner_other, [3]u8{ 1, 2, 0 });
    const other2 = @shuffle(T, inner_other, inner_other, [3]u8{ 2, 0, 1 });

    return vec1 * other2 - vec2 * other1;
}

/// Returns the distance between two vectors.
///
/// the precsion parameter is the number of bits of the Vector type T.
/// the precision of the calculations will match the precision of the output type.
pub fn distanceAdv(vec: anytype, other: anytype, comptime precision: u8) Float(precision) {
    const inner_vec = innerVec(vec);
    const inner_other: @TypeOf(inner_vec) = other;
    return normAdv(inner_vec - inner_other, precision);
}

/// Returns the distance between two vectors.
///
/// the precision of the output is the number of bits of T.
/// see `distanceAdv`
pub inline fn distance(vec: anytype, other: anytype) Float(@bitSizeOf(std.meta.Child(@TypeOf(vec)))) {
    return distanceAdv(vec, other, @bitSizeOf(std.meta.Child(@TypeOf(vec))));
}

/// Returns the angle between two vectors.
///
/// the precsion parameter is the number of bits of the Vector type T.
/// the precision of the calculations will match the precision of the output type.
pub fn angleAdv(vec: anytype, other: anytype, comptime precision: u8) Float(precision) {
    const inner_vec = innerVec(vec);
    const inner_other: @TypeOf(inner_vec) = other;
    return std.math.acos(dot(inner_vec, inner_other) / (normAdv(inner_vec, precision) * normAdv(inner_other, precision)));
}

/// Returns the angle between two vectors.
pub inline fn angle(vec: anytype, other: anytype) std.meta.Child(@TypeOf(vec)) {
    return angleAdv(vec, other, @bitSizeOf(std.meta.Child(@TypeOf(vec))));
}

/// Returns a new vector that is the reflection of the original vector on the given normal.
pub fn reflect(vec: anytype, normal: anytype) @TypeOf(vec) {
    const inner_vec = innerVec(vec);
    const inner_normal: @TypeOf(inner_vec) = normal;
    const dot_product = dot(inner_vec, normal);
    return inner_vec - (inner_normal *
        @as(@TypeOf(inner_normal), @splat(2)) *
        @as(@TypeOf(inner_normal), @splat(dot_product)));
}

/// Returns a new vector with a direction closest to the original vector, but with a magnitude scaled by the given value.
pub inline fn scale(vec: anytype, value: std.meta.Child(@TypeOf(vec))) @TypeOf(vec) {
    const inner_vec = innerVec(vec);
    return inner_vec * @as(@TypeOf(inner_vec), @splat(value));
}

test scale {
    const v: [2]f32 = .{ 1, 2 };
    try std.testing.expectEqual([2]f32{ 2, 4 }, scale(v, 2));
    try std.testing.expectEqual([2]f32{ 0.5, 1 }, scale(v, 0.5));
}

test swizzle {
    const v: [2]f32 = .{ 1, 2 };
    try std.testing.expectEqual(@as(f32, 2), swizzle(v, "yx")[0]);
    try std.testing.expectEqual(@as(f32, 1), swizzle(v, "yx")[1]);
    const v2: [3]f32 = .{ 1, 2, 3 };
    const v2_expected: [3]f32 = .{ 2, 3, 1 };
    try std.testing.expectEqual(v2_expected, swizzle(v2, "yzx"));
}

test angleAdv {
    const v1: [2]f32 = .{ 1, 0 };
    const v2: [2]f32 = .{ 0, 1 };
    const expected: f64 = @as(f64, std.math.pi) / @as(f64, 2);
    try std.testing.expectApproxEqAbs(expected, angleAdv(v1, v2, 32), 0.0000001);
    try std.testing.expectApproxEqAbs(expected, angleAdv(v1, v2, 64), 0.00000000001);
}

test normalize {
    // Test with floating point vector
    const v_f32: [3]f32 = .{ 3, 4, 0 };
    const normalized_f32 = normalize(v_f32);
    try std.testing.expectApproxEqAbs(@as(f32, 0.6), normalized_f32[0], 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.8), normalized_f32[1], 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 0), normalized_f32[2], 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 1), norm(normalized_f32), 0.0001);

    // Test with integer vector
    const v_i32: [2]i32 = .{ 3, 4 };
    const normalized_i32 = normalize(v_i32);
    try std.testing.expectEqual([2]i32{ 0, 0 }, normalized_i32); // Integer division limitations
}

test dot {
    const v1: [3]f32 = .{ 1, 2, 3 };
    const v2: [3]f32 = .{ 4, 5, 6 };
    try std.testing.expectEqual(@as(f32, 32), dot(v1, v2));

    const v3: [2]i32 = .{ 2, 3 };
    const v4: [2]i32 = .{ 4, 5 };
    try std.testing.expectEqual(@as(i32, 23), dot(v3, v4));
}

test cross {
    const v1: [3]f32 = .{ 1, 0, 0 };
    const v2: [3]f32 = .{ 0, 1, 0 };
    const expected: [3]f32 = .{ 0, 0, 1 };
    try std.testing.expectEqual(expected, cross(v1, v2));

    const v3: [3]i32 = .{ 2, 3, 4 };
    const v4: [3]i32 = .{ 5, 6, 7 };
    const expected_i: [3]i32 = .{ -3, 6, -3 };
    try std.testing.expectEqual(expected_i, cross(v3, v4));
}

test distance {
    const v1: [2]f32 = .{ 1, 1 };
    const v2: [2]f32 = .{ 4, 5 };
    try std.testing.expectApproxEqAbs(@as(f32, 5), distance(v1, v2), 0.0001);

    const v3: [3]f64 = .{ 1, 2, 3 };
    const v4: [3]f64 = .{ 4, 6, 8 };
    try std.testing.expectApproxEqAbs(@as(f64, 7.0710678118654755), distance(v3, v4), 0.0000001);
}

test angle {
    const v1: [2]f32 = .{ 1, 0 };
    const v2: [2]f32 = .{ 0, 1 };
    try std.testing.expectApproxEqAbs(@as(f32, std.math.pi / 2.0), angle(v1, v2), 0.0001);

    const v3: [2]f32 = .{ 1, 1 };
    const v4: [2]f32 = .{ -1, 1 };
    try std.testing.expectApproxEqAbs(@as(f32, std.math.pi / 2.0), angle(v3, v4), 0.0001);
}

test reflect {
    const v: [2]f32 = .{ 1, -1 };
    const normal: [2]f32 = .{ 0, 1 };
    const expected: [2]f32 = .{ 1, 1 };
    try std.testing.expectEqual(expected, reflect(v, normal));

    const v2: [3]f32 = .{ 1, -1, 0.5 };
    const normal2: [3]f32 = .{ 0, 1, 0 };
    const expected2: [3]f32 = .{ 1, 1, 0.5 };

    const result = reflect(v2, normal2);
    try std.testing.expectApproxEqAbs(expected2[0], result[0], 0.0001);
    try std.testing.expectApproxEqAbs(expected2[1], result[1], 0.0001);
    try std.testing.expectApproxEqAbs(expected2[2], result[2], 0.0001);
}

test normAdv {
    const v: [2]f64 = .{ 1, 0 };
    try std.testing.expectEqual(@as(f32, 1), normAdv(v, 32));
    const v2: [2]f16 = .{ 1, 2 };
    const result = normAdv(v2, 64);
    try std.testing.expect(@TypeOf(result) == f64);
    const expected = 2.2360679774997896964091736687;
    try std.testing.expectEqual(@as(f16, expected), normAdv(v2, 16));
    try std.testing.expectEqual(@as(f32, expected), normAdv(v2, 32));
    try std.testing.expectEqual(@as(f64, expected), normAdv(v2, 64));
    const v3: [2]i8 = .{ 1, 2 };
    try std.testing.expect(@TypeOf(normAdv(v3, 64)) == f64);
    try std.testing.expectEqual(@as(f16, expected), normAdv(v3, 16));
    try std.testing.expectEqual(@as(f32, expected), normAdv(v3, 32));
    try std.testing.expectEqual(@as(f64, expected), normAdv(v3, 64));
    const v4: [2]u8 = .{ 1, 2 };
    try std.testing.expect(@TypeOf(normAdv(v4, 64)) == f64);
    try std.testing.expectEqual(@as(f16, expected), normAdv(v4, 16));
    try std.testing.expectEqual(@as(f32, expected), normAdv(v4, 32));
    try std.testing.expectEqual(@as(f64, expected), normAdv(v4, 64));
}

test norm {
    const v1: [2]f32 = .{ 3, 4 };
    try std.testing.expectEqual(@as(f32, 5), norm(v1));

    const v2: [3]f64 = .{ 1, 2, 2 };
    try std.testing.expectApproxEqAbs(@as(f64, 3), norm(v2), 0.0001);

    const v3: [4]i32 = .{ 1, 2, 2, 0 };
    try std.testing.expectApproxEqAbs(@as(f32, 3), norm(v3), 0.0001);
}
