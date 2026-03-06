pub fn main(init: std.process.Init) !void {
    var tracy_allocator = tracy.TracyAllocator.init(init.gpa, "main gpa");
    const gpa = tracy_allocator.allocator();

    var scratch: Scratch = try .init(gpa);
    defer scratch.deinit();

    var engine = try VulkanEngine.init(gpa, &scratch, init.io);
    defer engine.deinit(gpa);

    var stop_rendering: bool = false;
    var event: c.SDL_Event = undefined;
    var quit: bool = false;

    var mouse_captured = false;

    while (!quit) {
        while (c.SDL_PollEvent(&event) != false) {
            switch (event.type) {
                c.SDL_EVENT_QUIT => quit = true,
                c.SDL_EVENT_WINDOW_MINIMIZED => stop_rendering = true,
                c.SDL_EVENT_WINDOW_RESTORED => stop_rendering = false,
                c.SDL_EVENT_KEY_DOWN => {
                    if (event.key.scancode == c.SDL_SCANCODE_ESCAPE) {
                        mouse_captured = !mouse_captured;
                        _ = c.SDL_SetWindowRelativeMouseMode(engine.window, mouse_captured);
                    }
                },
                else => {},
            }

            if (!mouse_captured) _ = c.cImGui_ImplSDL3_ProcessEvent(&event);
            if (mouse_captured) engine.main_camera.processSDLEvent(&event);
        }

        if (stop_rendering) {
            try init.io.sleep(.fromMilliseconds(100), .awake);
            continue;
        }

        try engine.draw(gpa, &scratch);

        if (engine.resize_requested) {
            try engine.resizeSwapchain(gpa, &scratch);
        }
    }
}

const std = @import("std");
const tracy = @import("tracy");
const VulkanEngine = @import("vk_engine.zig").Engine;
const Scratch = @import("scratch_allocator");
const c = @import("c");
