const std = @import("std");
const tracy = @import("tracy");
const VulkanEngine = @import("vk_engine.zig").Engine;
const c = @import("c");

pub fn main() !void {
    var debug_allocator: std.heap.DebugAllocator(.{}) = .init;
    defer _ = debug_allocator.deinit();
    var tracy_allocator = tracy.TracyAllocator.init(debug_allocator.allocator(), "main gpa");
    const allocator = tracy_allocator.allocator();

    var engine = try VulkanEngine.init(allocator);
    defer engine.deinit(allocator);

    var stop_rendering: bool = false;
    var event: c.SDL_Event = undefined;
    var quit: bool = false;

    while (!quit) {
        while (c.SDL_PollEvent(&event) != false) {
            switch (event.type) {
                c.SDL_EVENT_QUIT => quit = true,
                c.SDL_EVENT_WINDOW_MINIMIZED => stop_rendering = true,
                c.SDL_EVENT_WINDOW_RESTORED => stop_rendering = false,
                else => {},
            }
            _ = c.cImGui_ImplSDL3_ProcessEvent(&event);
        }

        if (stop_rendering) {
            std.Thread.sleep(std.time.ns_per_ms * 100);
            continue;
        }

        try engine.draw(allocator);

        if (engine.resize_requested) {
            try engine.resizeSwapchain(allocator);
        }
    }
}
