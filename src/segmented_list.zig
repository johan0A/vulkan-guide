const std = @import("std");

pub fn SegmentedList(comptime T: type) type {
    const min_exp = 8;
    const shelve_count = 40 - min_exp;

    return struct {
        const Self = @This();
        const ShelfIndex = std.math.Log2Int(usize);

        pub const empty: Self = .{ .shelves = undefined, .len = 0, .shelves_allocated = 0 };

        shelves: [shelve_count][*]T,
        len: usize,
        shelves_allocated: ShelfIndex,

        pub fn deinit(self: *Self, gpa: std.mem.Allocator) void {
            for (0..self.shelves_allocated) |i| gpa.free(self.shelves[i][0..shelfSize(@intCast(i))]);
            self.* = undefined;
        }

        pub fn append(self: *Self, gpa: std.mem.Allocator, item: T) !void {
            (try self.addOne(gpa)).* = item;
        }

        pub fn pop(self: *Self) ?T {
            if (self.len == 0) return null;
            self.len -= 1;
            return self.uncheckedAt(self.len).*;
        }

        pub fn at(self: *Self, index: usize) *T {
            std.debug.assert(index < self.len);
            const si = shelfIndex(index);
            return &self.shelves[si][boxIndex(index, si)];
        }

        pub fn addOne(self: *Self, gpa: std.mem.Allocator) !*T {
            try self.ensureCapacity(gpa, self.len + 1);
            const ptr = self.uncheckedAt(self.len);
            self.len += 1;
            return ptr;
        }

        pub fn ensureCapacity(self: *Self, gpa: std.mem.Allocator, needed: usize) !void {
            const new = shelfCount(needed);
            for (self.shelves_allocated..new) |i| self.shelves[i] = (try gpa.alloc(T, shelfSize(@intCast(i)))).ptr;
            self.shelves_allocated = @max(self.shelves_allocated, new);
        }

        pub fn clearRetainingCapacity(self: *Self) void {
            self.len = 0;
        }

        fn shelfCount(count: usize) ShelfIndex {
            if (count == 0) return 0;
            return @intCast(std.math.log2_int_ceil(usize, count + (1 << min_exp)) - min_exp);
        }

        fn shelfSize(shelf: ShelfIndex) usize {
            return @as(usize, 1) << (shelf + min_exp);
        }

        fn shelfIndex(index: usize) ShelfIndex {
            return std.math.log2_int(usize, index + (1 << min_exp)) - min_exp;
        }

        fn boxIndex(index: usize, shelf: ShelfIndex) usize {
            return index + (1 << min_exp) - (@as(usize, 1) << (shelf + min_exp));
        }
    };
}
