const std = @import("std");
const VulkanEngine = @import("vk_engine.zig").VulkanEngine;
const c = @import("c");

pub fn main() !void {
    var debug_allocator: std.heap.DebugAllocator(.{}) = .init;
    defer _ = debug_allocator.deinit();
    const allocator = debug_allocator.allocator();

    var engine = try VulkanEngine.init(allocator);
    defer engine.deinit(allocator);

    var stop_rendering: bool = false;
    var e: c.SDL_Event = undefined;
    var bQuit: bool = false;

    while (!bQuit) {
        while (c.SDL_PollEvent(&e) != false) {
            switch (e.type) {
                c.SDL_EVENT_QUIT => bQuit = true,
                c.SDL_EVENT_WINDOW_MINIMIZED => stop_rendering = true,
                c.SDL_EVENT_WINDOW_RESTORED => stop_rendering = false,
                else => {},
            }
        }

        if (stop_rendering) {
            std.time.sleep(std.time.ns_per_ms * 100);
        } else {
            try engine.draw();
        }
    }
}
