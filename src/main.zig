const std = @import("std");
const VulkanEngine = @import("vk_engine.zig").VulkanEngine;

pub fn main() !void {
    var debug_allocator: std.heap.DebugAllocator(.{}) = .init;
    defer _ = debug_allocator.deinit();
    const allocator = debug_allocator.allocator();

    const engine = try VulkanEngine.init(allocator);
    defer engine.deinit(allocator);
}
