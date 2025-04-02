const std = @import("std");
const tracy = @import("tracy");
const VulkanEngine = @import("vk_engine.zig").VulkanEngine;
const c = @import("c");

pub fn main() !void {
    var debug_allocator: std.heap.DebugAllocator(.{}) = .init;
    defer _ = debug_allocator.deinit();
    var tracy_allocator = tracy.TracyAllocator.init(debug_allocator.allocator(), "main gpa");
    const allocator = tracy_allocator.allocator();

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
            _ = c.cImGui_ImplSDL3_ProcessEvent(&e);
        }

        if (stop_rendering) {
            std.time.sleep(std.time.ns_per_ms * 100);
            continue;
        }

        // imgui new frame
        c.cImGui_ImplVulkan_NewFrame();
        c.cImGui_ImplSDL3_NewFrame();
        c.ImGui_NewFrame();

        //some imgui UI to test
        c.ImGui_ShowDemoWindow(null);

        //make imgui calculate internal draw structures
        c.ImGui_Render();

        try engine.draw();
    }
}
