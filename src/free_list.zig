pub fn FreeList(T: type) type {
    return struct {
        slots: std.ArrayList(Slot),
        next_free: usize,

        const nil = std.math.maxInt(usize);

        pub const empty: @This() = .{ .next_free = nil, .slots = .empty };

        const Slot = union { next_free: usize, item: T };

        pub fn new(free_list: *@This(), gpa: std.mem.Allocator, item: T) std.mem.Allocator.Error!usize {
            if (free_list.next_free == nil) {
                try free_list.slots.append(gpa, .{ .item = item });
                return free_list.slots.items.len - 1;
            }
            const next_free = free_list.next_free;
            free_list.next_free = free_list.slots.items[next_free].next_free;
            free_list.slots.items[next_free] = .{ .item = item };
            return next_free;
        }

        pub fn delete(free_list: *@This(), index: usize) void {
            _ = free_list.slots.items[index].item; // safe builds trap here on double-delete
            free_list.slots.items[index] = .{ .next_free = free_list.next_free };
            free_list.next_free = index;
        }

        pub fn get(free_list: @This(), index: usize) *T {
            return &free_list.slots.items[index].item;
        }

        pub fn deinit(free_list: *@This(), gpa: std.mem.Allocator) void {
            free_list.slots.deinit(gpa);
            free_list.* = undefined;
        }
    };
}

test FreeList {
    const gpa = std.testing.allocator;

    var list: FreeList(u16) = .empty;
    defer list.deinit(gpa);

    var prng = std.Random.DefaultPrng.init(std.testing.random_seed);
    const rand = prng.random();

    const Item = struct { index: usize, value: u16 };
    var items: std.ArrayList(Item) = .empty;
    defer items.deinit(gpa);

    for (0..512) |i| {
        try items.append(gpa, .{ .index = try list.new(gpa, @intCast(i)), .value = @intCast(i) });
    }

    rand.shuffle(Item, items.items);

    for (0..256) |_| {
        const item = items.pop().?;
        try std.testing.expectEqual(item.value, list.get(item.index).*);
        list.delete(item.index);
    }

    rand.shuffle(Item, items.items);

    for (0..512) |i| {
        try items.append(gpa, .{ .index = try list.new(gpa, @intCast(i)), .value = @intCast(i) });
    }

    rand.shuffle(Item, items.items);

    for (items.items) |item| {
        try std.testing.expectEqual(item.value, list.get(item.index).*);
        list.delete(item.index);
    }
}

const std = @import("std");
