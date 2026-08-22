const std = @import("std");
const root = @import("root.zig");
const vec = root.vec;
const mat = root.matrix;

pub fn Quat(comptime T: type) type {
    return extern struct {
        const Self = @This();

        v: [4]T,

        pub const identity: Self = .{ .v = .{ 0, 0, 0, 1 } };

        pub inline fn init(xv: T, yv: T, zv: T, wv: T) Self {
            return .{ .v = .{ xv, yv, zv, wv } };
        }

        pub inline fn x(self: Self) T {
            return self.v[0];
        }

        pub inline fn y(self: Self) T {
            return self.v[1];
        }

        pub inline fn z(self: Self) T {
            return self.v[2];
        }

        pub inline fn w(self: Self) T {
            return self.v[3];
        }

        pub inline fn imaginary(self: Self) @Vector(3, T) {
            return .{ self.v[0], self.v[1], self.v[2] };
        }

        pub fn fromAxisAngle(axis: @Vector(3, T), angle: T) Self {
            const half = angle * 0.5;
            const s = std.math.sin(half);
            const a = vec.normalize(axis);
            return .{ .v = .{
                a[0] * s,
                a[1] * s,
                a[2] * s,
                std.math.cos(half),
            } };
        }

        pub fn toAxisAngle(self: Self) struct { axis: @Vector(3, T), angle: T } {
            const n = self.normalize();
            const w_clamped = std.math.clamp(n.w(), @as(T, -1.0), @as(T, 1.0));
            const angle = 2.0 * std.math.acos(w_clamped);
            const s2 = 1.0 - w_clamped * w_clamped;
            if (s2 < 1e-8) {
                return .{ .axis = .{ 1, 0, 0 }, .angle = angle };
            }
            const inv_s: T = 1.0 / std.math.sqrt(s2);
            return .{
                .axis = .{ n.x() * inv_s, n.y() * inv_s, n.z() * inv_s },
                .angle = angle,
            };
        }

        pub fn fromEuler(pitch: T, yaw: T, roll: T) Self {
            const hp = pitch * 0.5;
            const hy = yaw * 0.5;
            const hr = roll * 0.5;
            const sp = std.math.sin(hp);
            const cp = std.math.cos(hp);
            const sy = std.math.sin(hy);
            const cy = std.math.cos(hy);
            const sr = std.math.sin(hr);
            const cr = std.math.cos(hr);
            return .{ .v = .{
                cr * cy * sp - sr * sy * cp,
                cr * sy * cp + sr * cy * sp,
                sr * cy * cp - cr * sy * sp,
                cr * cy * cp + sr * sy * sp,
            } };
        }

        pub fn toEuler(self: Self) struct { pitch: T, yaw: T, roll: T } {
            const n = self.normalize();
            const sinp = 2.0 * (n.w() * n.y() - n.z() * n.x());

            if (@abs(sinp) >= 1.0 - 1e-6) {
                const sign: T = if (sinp >= 0) 1.0 else -1.0;
                return .{
                    .pitch = -sign * 2.0 * std.math.atan2(n.x(), n.w()),
                    .yaw = sign * std.math.pi / 2.0,
                    .roll = 0,
                };
            }

            return .{
                .pitch = std.math.atan2(
                    2.0 * (n.w() * n.x() + n.y() * n.z()),
                    1.0 - 2.0 * (n.x() * n.x() + n.y() * n.y()),
                ),
                .yaw = std.math.asin(std.math.clamp(sinp, @as(T, -1.0), @as(T, 1.0))),
                .roll = std.math.atan2(
                    2.0 * (n.w() * n.z() + n.x() * n.y()),
                    1.0 - 2.0 * (n.y() * n.y() + n.z() * n.z()),
                ),
            };
        }

        pub fn fromMat(m: anytype) Self {
            const M = @TypeOf(m);
            if (M.rows < 3 or M.cols < 3) @compileError("fromMat requires at least a 3x3 matrix");

            const m00 = m.at(0, 0);
            const m01 = m.at(0, 1);
            const m02 = m.at(0, 2);
            const m10 = m.at(1, 0);
            const m11 = m.at(1, 1);
            const m12 = m.at(1, 2);
            const m20 = m.at(2, 0);
            const m21 = m.at(2, 1);
            const m22 = m.at(2, 2);

            const trace = m00 + m11 + m22;

            if (trace > 0) {
                const r = std.math.sqrt(1.0 + trace);
                const s: T = 0.5 / r;
                return .{ .v = .{
                    (m21 - m12) * s,
                    (m02 - m20) * s,
                    (m10 - m01) * s,
                    0.5 * r,
                } };
            } else if (m00 > m11 and m00 > m22) {
                const r = std.math.sqrt(1.0 + m00 - m11 - m22);
                const s: T = 0.5 / r;
                return .{ .v = .{
                    0.5 * r,
                    (m01 + m10) * s,
                    (m02 + m20) * s,
                    (m21 - m12) * s,
                } };
            } else if (m11 > m22) {
                const r = std.math.sqrt(1.0 - m00 + m11 - m22);
                const s: T = 0.5 / r;
                return .{ .v = .{
                    (m01 + m10) * s,
                    0.5 * r,
                    (m12 + m21) * s,
                    (m02 - m20) * s,
                } };
            } else {
                const r = std.math.sqrt(1.0 - m00 - m11 + m22);
                const s: T = 0.5 / r;
                return .{ .v = .{
                    (m02 + m20) * s,
                    (m12 + m21) * s,
                    0.5 * r,
                    (m10 - m01) * s,
                } };
            }
        }

        pub fn toMat4(self: Self, comptime layout: mat.Layout) mat.Mat(layout, T, 4, 4) {
            const n = self.normalize();
            const M = mat.Mat(layout, T, 4, 4);

            const xx = n.x() * n.x();
            const yy = n.y() * n.y();
            const zz = n.z() * n.z();
            const xy = n.x() * n.y();
            const xz = n.x() * n.z();
            const yz = n.y() * n.z();
            const wx = n.w() * n.x();
            const wy = n.w() * n.y();
            const wz = n.w() * n.z();

            var result: M = .identity;
            result.put(0, 0, 1.0 - 2.0 * (yy + zz));
            result.put(1, 0, 2.0 * (xy + wz));
            result.put(2, 0, 2.0 * (xz - wy));

            result.put(0, 1, 2.0 * (xy - wz));
            result.put(1, 1, 1.0 - 2.0 * (xx + zz));
            result.put(2, 1, 2.0 * (yz + wx));

            result.put(0, 2, 2.0 * (xz + wy));
            result.put(1, 2, 2.0 * (yz - wx));
            result.put(2, 2, 1.0 - 2.0 * (xx + yy));

            return result;
        }

        pub fn toMat3(self: Self, comptime layout: mat.Layout) mat.Mat(layout, T, 3, 3) {
            const n = self.normalize();
            const M = mat.Mat(layout, T, 3, 3);

            const xx = n.x() * n.x();
            const yy = n.y() * n.y();
            const zz = n.z() * n.z();
            const xy = n.x() * n.y();
            const xz = n.x() * n.z();
            const yz = n.y() * n.z();
            const wx = n.w() * n.x();
            const wy = n.w() * n.y();
            const wz = n.w() * n.z();

            var result: M = .identity;
            result.put(0, 0, 1.0 - 2.0 * (yy + zz));
            result.put(1, 0, 2.0 * (xy + wz));
            result.put(2, 0, 2.0 * (xz - wy));

            result.put(0, 1, 2.0 * (xy - wz));
            result.put(1, 1, 1.0 - 2.0 * (xx + zz));
            result.put(2, 1, 2.0 * (yz + wx));

            result.put(0, 2, 2.0 * (xz + wy));
            result.put(1, 2, 2.0 * (yz - wx));
            result.put(2, 2, 1.0 - 2.0 * (xx + yy));

            return result;
        }

        pub fn mul(self: Self, other: Self) Self {
            return .{ .v = .{
                self.w() * other.x() + self.x() * other.w() + self.y() * other.z() - self.z() * other.y(),
                self.w() * other.y() - self.x() * other.z() + self.y() * other.w() + self.z() * other.x(),
                self.w() * other.z() + self.x() * other.y() - self.y() * other.x() + self.z() * other.w(),
                self.w() * other.w() - self.x() * other.x() - self.y() * other.y() - self.z() * other.z(),
            } };
        }

        pub inline fn selfMul(self: *Self, other: Self) void {
            self.* = self.mul(other);
        }

        pub fn add(self: Self, other: Self) Self {
            return .{ .v = self.v + other.v };
        }

        pub fn sub(self: Self, other: Self) Self {
            return .{ .v = self.v - other.v };
        }

        pub fn scalarMul(self: Self, s: T) Self {
            return .{ .v = self.v * @as(@Vector(4, T), @splat(s)) };
        }

        pub fn negate(self: Self) Self {
            return .{ .v = -@as(@Vector(4, T), self.v) };
        }

        pub fn conjugate(self: Self) Self {
            return .{ .v = .{ -self.x(), -self.y(), -self.z(), self.w() } };
        }

        pub fn norm2(self: Self) T {
            return @reduce(.Add, @as(@Vector(4, T), self.v) * @as(@Vector(4, T), self.v));
        }

        pub fn norm(self: Self) T {
            return std.math.sqrt(self.norm2());
        }

        pub fn normalize(self: Self) Self {
            const n = self.norm();
            if (n < 1e-15) return .identity;
            return .{ .v = self.v / @as(@Vector(4, T), @splat(n)) };
        }

        pub fn inverse(self: Self) Self {
            const n2 = self.norm2();
            if (n2 < 1e-15) return .identity;
            const conj = self.conjugate();
            return .{ .v = conj.v / @as(@Vector(4, T), @splat(n2)) };
        }

        pub fn dot(self: Self, other: Self) T {
            return @reduce(.Add, @as(@Vector(4, T), self.v) * @as(@Vector(4, T), other.v));
        }

        pub fn rotateVec(self: Self, point: @Vector(3, T)) @Vector(3, T) {
            const u = self.imaginary();
            const uv = vec.cross(u, point);
            const uuv = vec.cross(u, uv);
            return point + vec.splat(3, 2.0 * self.w()) * uv + vec.splat(3, @as(T, 2.0)) * uuv;
        }

        pub fn nlerp(self: Self, other: Self, t: T) Self {
            var b = other;
            if (self.dot(b) < 0) b.v = -@as(@Vector(4, T), b.v);
            const tv: @Vector(4, T) = @splat(t);
            const omtv: @Vector(4, T) = @splat(1.0 - t);
            return (Self{ .v = self.v * omtv + b.v * tv }).normalize();
        }

        pub fn slerp(self: Self, other: Self, t: T) Self {
            var b = other;
            var d = self.dot(b);
            if (d < 0) {
                b.v = -@as(@Vector(4, T), b.v);
                d = -d;
            }
            if (d > 1.0 - 1e-6) return self.nlerp(b, t);

            const theta = std.math.acos(std.math.clamp(d, @as(T, -1.0), @as(T, 1.0)));
            const sin_theta = std.math.sin(theta);
            const s0: T = std.math.sin((1.0 - t) * theta) / sin_theta;
            const s1: T = std.math.sin(t * theta) / sin_theta;
            return .{ .v = self.v * @as(@Vector(4, T), @splat(s0)) + b.v * @as(@Vector(4, T), @splat(s1)) };
        }

        pub fn eql(self: Self, other: Self) bool {
            return @reduce(.And, self.v == other.v);
        }

        pub fn approxEql(self: Self, other: Self, tolerance: T) bool {
            const diff = @abs(@as(@Vector(4, T), self.v) - @as(@Vector(4, T), other.v));
            const tol: @Vector(4, T) = @splat(tolerance);
            return @reduce(.And, diff <= tol);
        }

        pub fn rotationEql(self: Self, other: Self, tolerance: T) bool {
            return self.approxEql(other, tolerance) or self.negate().approxEql(other, tolerance);
        }

        pub fn format(self: Self, writer: *std.Io.Writer) std.Io.Writer.Error!void {
            try writer.print("Quat({d}, {d}, {d}, {d})", .{ self.x(), self.y(), self.z(), self.w() });
        }
    };
}
