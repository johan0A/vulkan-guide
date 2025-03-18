const std = @import("std");
const vk = @import("vulkan");
const c = @import("c");

const enable_validation_layers = true;
const validation_layers = [_][:0]const u8{"VK_LAYER_KHRONOS_validation"};

pub const required_device_extensions = [_][:0]const u8{vk.extensions.khr_swapchain.name};

const AllocatedImage = struct {
    image: vk.Image,
    image_view: vk.ImageView,
    allocation: c.VmaAllocation,
    image_extent: vk.Extent3D,
    image_format: vk.Format,
};

const Allocator = std.mem.Allocator;

pub const apis: []const vk.ApiInfo = &.{
    vk.features.version_1_3,
    vk.features.version_1_2,
    vk.features.version_1_1,
    vk.features.version_1_0,
    vk.extensions.khr_surface,
    vk.extensions.khr_swapchain,
};

pub const QueueFamilyIndices = struct {
    graphics_family: ?u32,
    present_family: ?u32,
};

pub const Dispatch = struct {
    pub const Base = vk.BaseWrapper(apis);
    pub const Instance = vk.InstanceWrapper(apis);
    pub const Device = vk.DeviceWrapper(apis);
};

const DeletionQueue = struct {
    const DeinitContext = struct {
        device: DeviceProxy,
        vma_allocator: c.VmaAllocator,
    };

    const QueueItem = union(enum) {
        vma_allocator: c.VmaAllocator,
        image_view: vk.ImageView,
        vma_allocated_image: struct { image: vk.Image, allocation: c.VmaAllocation },

        fn deinit(self: QueueItem, context: DeinitContext) void {
            switch (self) {
                .vma_allocator => |item| c.vmaDestroyAllocator(item),
                .image_view => |item| context.device.destroyImageView(item, null),
                .vma_allocated_image => |item| c.vmaDestroyImage(context.vma_allocator, @ptrFromInt(@intFromEnum(item.image)), item.allocation),
            }
        }
    };

    queue: std.ArrayListUnmanaged(QueueItem),

    pub const init: DeletionQueue = .{
        .queue = .empty,
    };

    pub fn flush(self: *DeletionQueue, context: DeinitContext) void {
        for (0..self.queue.items.len) |i| {
            self.queue.items[self.queue.items.len - i - 1].deinit(context);
        }
        self.queue.clearRetainingCapacity();
    }

    pub fn deinit(self: *DeletionQueue, alloc: Allocator, context: DeinitContext) void {
        self.flush(context);
        self.queue.deinit(alloc);
    }

    pub fn append(self: *DeletionQueue, alloc: Allocator, item: QueueItem) !void {
        try self.queue.append(alloc, item);
    }
};

const InstanceProxy = vk.InstanceProxy(apis);
const DeviceProxy = vk.DeviceProxy(apis);

const SwapChain = struct {
    handle: vk.SwapchainKHR,
    image_format: vk.Format,
    image_color_space: vk.ColorSpaceKHR,
    images: []vk.Image,
    image_views: []vk.ImageView,
    extent: vk.Extent2D,

    fn deinit(self: SwapChain, alloc: Allocator, device: DeviceProxy) void {
        device.destroySwapchainKHR(self.handle, null);
        for (self.image_views) |image_view| {
            device.destroyImageView(image_view, null);
        }
        alloc.free(self.images);
        alloc.free(self.image_views);
    }
};

const FrameData = struct {
    const FRAME_OVERLAP = 2;

    command_pool: vk.CommandPool,
    main_command_buffer: vk.CommandBuffer,

    swapchain_semaphore: vk.Semaphore,
    render_semaphore: vk.Semaphore,
    render_fence: vk.Fence,

    deletion_queue: DeletionQueue,
};

pub const VulkanEngine = struct {
    init_arena: std.heap.ArenaAllocator,

    window: *c.SDL_Window,

    base_dispatch: Dispatch.Base,

    instance: InstanceProxy,
    device: DeviceProxy,

    debug_messenger: vk.DebugUtilsMessengerEXT, // Vulkan debug output handle
    chosen_gpu: vk.PhysicalDevice, // GPU chosen as the default device
    surface: vk.SurfaceKHR, // Vulkan window surface

    swapchain: SwapChain,

    graphics_queue: vk.Queue,
    graphics_queue_family: u32,

    frame_number: u64,
    frames: [FrameData.FRAME_OVERLAP]FrameData,

    main_deletion_queue: DeletionQueue,

    vma_allocator: c.VmaAllocator,

    //draw resources
    draw_image: AllocatedImage,
    draw_extent: vk.Extent2D,

    pub fn draw(self: *VulkanEngine) !void {
        const local = struct {
            fn drawBackground(engine: *VulkanEngine, cmd: vk.CommandBuffer) void {
                //make a clear-color from frame number. This will flash with a 120 frame period.
                const flash = @abs(std.math.sin(@as(f32, @floatFromInt(engine.frame_number)) / 120));
                var clear_value: vk.ClearColorValue = .{ .float_32 = .{ 0, 0, flash, 1 } };

                const clear_range: vk.ImageSubresourceRange = vk_init.imageSubresourceRange(.{ .color_bit = true });

                //clear image
                engine.device.cmdClearColorImage(cmd, engine.draw_image.image, .general, &clear_value, 1, (&clear_range)[0..1]);
            }
        };

        _ = try self.device.waitForFences(1, (&self.currentFrame().render_fence)[0..1], vk.TRUE, 1e9);
        _ = try self.device.resetFences(1, (&self.currentFrame().render_fence)[0..1]);

        const swapchain_image_index: u32 = (try self.device.acquireNextImageKHR(
            self.swapchain.handle,
            1e9,
            self.currentFrame().swapchain_semaphore,
            .null_handle,
        )).image_index;

        //naming it cmd for shorter writing
        const cmd: vk.CommandBuffer = self.currentFrame().main_command_buffer;

        self.draw_extent.width = self.draw_image.image_extent.width; // TODO: I dont like this, its duplication of state for no good reasons
        self.draw_extent.height = self.draw_image.image_extent.height;

        // now that we are sure that the commands finished executing, we can safely
        // reset the command buffer to begin recording again.
        try self.device.resetCommandBuffer(cmd, .{});

        // begin the command buffer recording. We will use this command buffer exactly once, so we want to let vulkan know that
        try self.device.beginCommandBuffer(cmd, &.{ .flags = .{ .one_time_submit_bit = true } });
        {
            // transition our main draw image into general layout so we can write into it
            // we will overwrite it all so we dont care about what was the older layout
            vk_image.transitionImage(self.device, cmd, self.draw_image.image, .undefined, .general);

            local.drawBackground(self, cmd);

            //transition the draw image and the swapchain image into their correct transfer layouts
            vk_image.transitionImage(self.device, cmd, self.draw_image.image, .general, .transfer_src_optimal);
            vk_image.transitionImage(self.device, cmd, self.swapchain.images[swapchain_image_index], .undefined, .transfer_dst_optimal);

            // execute a copy from the draw image into the swapchain
            vk_image.copy_image_to_image(self.device, cmd, self.draw_image.image, self.swapchain.images[swapchain_image_index], self.draw_extent, self.swapchain.extent);

            // set swapchain image layout to Present so we can show it on the screen
            vk_image.transitionImage(self.device, cmd, self.swapchain.images[swapchain_image_index], .transfer_dst_optimal, .present_src_khr);

            //finalize the command buffer (we can no longer add commands, but it can now be executed)
        }
        try self.device.endCommandBuffer(cmd);

        //prepare the submission to the queue.
        //we want to wait on the _presentSemaphore, as that semaphore is signaled when the swapchain is ready
        //we will signal the _renderSemaphore, to signal that rendering has finished
        {
            const cmd_info: vk.CommandBufferSubmitInfo = vk_init.commandBufferSubmitInfo(cmd);
            const wait_info: vk.SemaphoreSubmitInfo = vk_init.semaphoreSubmitInfo(.{ .color_attachment_output_bit = true }, self.currentFrame().swapchain_semaphore);
            const signal_info: vk.SemaphoreSubmitInfo = vk_init.semaphoreSubmitInfo(.{ .all_graphics_bit = true }, self.currentFrame().render_semaphore);

            const submit_info = vk_init.submitInfo(&cmd_info, &signal_info, &wait_info);

            //submit command buffer to the queue and execute it.
            // _render_fence will now block until the graphic commands finish execution
            try self.device.queueSubmit2(self.graphics_queue, 1, (&submit_info)[0..1], self.currentFrame().render_fence);
        }

        const present_info: vk.PresentInfoKHR = .{
            .p_swapchains = (&self.swapchain.handle)[0..1],
            .swapchain_count = 1,
            .p_wait_semaphores = (&self.currentFrame().render_semaphore)[0..1],
            .wait_semaphore_count = 1,
            .p_image_indices = (&swapchain_image_index)[0..1],
        };
        _ = try self.device.queuePresentKHR(self.graphics_queue, &present_info);

        self.frame_number += 1;
    }

    pub inline fn currentFrame(self: *VulkanEngine) *FrameData {
        return &self.frames[self.frame_number % FrameData.FRAME_OVERLAP];
    }

    pub fn init(allocator: Allocator) !VulkanEngine {
        if (!c.SDL_Init(c.SDL_INIT_VIDEO)) return error.engine_init_failure;

        var init_arena: std.heap.ArenaAllocator = .init(allocator);
        const init_alloc = init_arena.allocator();

        var temp_arena: std.heap.ArenaAllocator = .init(allocator);
        defer temp_arena.deinit();
        const temp_alloc = temp_arena.allocator();

        const window_width = 400;
        const window_height = 400;

        const window = c.SDL_CreateWindow("title", window_width, window_height, c.SDL_WINDOW_VULKAN) orelse return error.engine_init_failure;

        const base_dispatch = try Dispatch.Base.load(@as(vk.PfnGetInstanceProcAddr, @ptrCast(c.SDL_Vulkan_GetVkGetInstanceProcAddr())));

        const instance = try vk_init.createVkInstance(base_dispatch, temp_alloc);
        const instance_dispatch = try init_alloc.create(Dispatch.Instance);
        instance_dispatch.* = try Dispatch.Instance.load(instance, base_dispatch.dispatch.vkGetInstanceProcAddr);
        const instance_proxy = InstanceProxy.init(instance, instance_dispatch);

        var surface: vk.SurfaceKHR = undefined;
        if (!c.SDL_Vulkan_CreateSurface(window, @ptrFromInt(@intFromEnum(instance)), null, @ptrCast(&surface))) return error.engine_init_failure;

        const physical_device = try vk_init.pickPhysicalDevice(instance_proxy, surface, temp_alloc);
        const queue_family_indices = try vk_init.findQueueFamilies(physical_device, instance_dispatch.*, surface, temp_alloc);

        const device = try vk_init.createLogicalDevice(physical_device, instance_dispatch.*, queue_family_indices);
        const device_dispatch = try init_alloc.create(Dispatch.Device);
        device_dispatch.* = try Dispatch.Device.load(device, instance_dispatch.dispatch.vkGetDeviceProcAddr);
        const device_proxy = DeviceProxy.init(device, device_dispatch);

        var frames: [FrameData.FRAME_OVERLAP]FrameData = undefined;
        for (&frames) |*frame| {
            frame.* = try vk_init.initFrame(device_proxy, queue_family_indices.graphics_family.?);
        }

        var vma_allocator: c.VmaAllocator = undefined;
        if (c.vmaCreateAllocator(&.{
            .physicalDevice = @ptrFromInt(@intFromEnum(physical_device)),
            .device = @ptrFromInt(@intFromEnum(device)),
            .instance = @ptrFromInt(@intFromEnum(instance)),
            .flags = c.VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
            .pVulkanFunctions = &c.VmaVulkanFunctions{
                .vkGetDeviceProcAddr = @ptrCast(instance_dispatch.dispatch.vkGetDeviceProcAddr),
                .vkGetInstanceProcAddr = @ptrCast(base_dispatch.dispatch.vkGetInstanceProcAddr),
            },
        }, &vma_allocator) != 0) return error.vma_allocator_init_failed;

        var main_deletion_queue: DeletionQueue = .init;
        try main_deletion_queue.append(allocator, .{ .vma_allocator = vma_allocator });

        const draw_allocated_image = blk: {
            const draw_image_format: vk.Format = .r16g16b16a16_sfloat;

            const draw_image_usages: vk.ImageUsageFlags = .{
                .transfer_src_bit = true,
                .transfer_dst_bit = true,
                .storage_bit = true,
                .color_attachment_bit = true,
            };

            const draw_image_extent: vk.Extent3D = .{
                .width = window_width,
                .height = window_height,
                .depth = 1,
            };

            const rimg_info: vk.ImageCreateInfo = vk_init.imageCreateInfo(draw_image_format, draw_image_usages, draw_image_extent);

            //for the draw image, we want to allocate it from gpu local memory
            const rimg_allocinfo: c.VmaAllocationCreateInfo = .{
                .usage = c.VMA_MEMORY_USAGE_GPU_ONLY,
                .requiredFlags = @bitCast(vk.MemoryPropertyFlags{ .device_local_bit = true }),
            };

            var draw_image: vk.Image = undefined;
            var draw_image_allocation: c.VmaAllocation = undefined;

            //allocate and create the image
            const result: vk.Result = @enumFromInt(c.vmaCreateImage(
                vma_allocator,
                @ptrCast(&rimg_info),
                @ptrCast(&rimg_allocinfo),
                @ptrCast(&draw_image),
                &draw_image_allocation,
                null,
            ));
            _ = result; // TODO: handle failure

            //build a image-view for the draw image to use for rendering
            const rview_info: vk.ImageViewCreateInfo = vk_init.imageViewCreateInfo(draw_image_format, draw_image, .{ .color_bit = true });

            const draw_allocated_image: AllocatedImage = .{
                //hardcoding the draw format to 32 bit float
                .image_format = .r16g16b16a16_sfloat,
                //draw image size will match the window
                .image_extent = .{
                    .width = window_width,
                    .height = window_height,
                    .depth = 1,
                },
                .image = draw_image,
                .image_view = try device_proxy.createImageView(&rview_info, null),
                .allocation = draw_image_allocation,
            };

            //add to deletion queues
            // main_deletion_queue.append(
            //     vkDestroyImageView(_device, _drawImage.imageView, nullptr);
            //     vmaDestroyImage(_allocator, _drawImage.image, _drawImage.allocation);
            // );
            try main_deletion_queue.append(allocator, .{ .image_view = draw_allocated_image.image_view });
            try main_deletion_queue.append(allocator, .{ .vma_allocated_image = .{ .image = draw_allocated_image.image, .allocation = draw_allocated_image.allocation } });

            break :blk draw_allocated_image;
        };

        return .{
            .init_arena = init_arena,

            .window = window,
            .base_dispatch = base_dispatch,
            .instance = instance_proxy,
            .device = device_proxy,
            .swapchain = try vk_init.createSwapchain(
                init_alloc,
                temp_alloc,
                physical_device,
                device_proxy,
                surface,
                window_width,
                window_height,
                instance_dispatch.*,
            ),
            .debug_messenger = .null_handle,
            .chosen_gpu = physical_device,
            .surface = surface,

            .graphics_queue = device_proxy.getDeviceQueue(queue_family_indices.graphics_family.?, 0),
            .graphics_queue_family = queue_family_indices.graphics_family.?,

            .frame_number = 0,
            .frames = frames,

            .main_deletion_queue = main_deletion_queue,

            .vma_allocator = vma_allocator,

            .draw_image = draw_allocated_image,
            .draw_extent = .{ .width = 0, .height = 0 },
        };
    }

    pub fn deinit(self: *VulkanEngine, allocator: Allocator) void {
        self.device.deviceWaitIdle() catch @panic(""); // TODO

        for (0..self.frames.len) |i| {
            self.device.destroyCommandPool(self.frames[i].command_pool, null);

            self.device.destroyFence(self.frames[i].render_fence, null);
            self.device.destroySemaphore(self.frames[i].swapchain_semaphore, null);
            self.device.destroySemaphore(self.frames[i].render_semaphore, null);

            self.frames[i].deletion_queue.deinit(allocator, .{
                .device = self.device,
                .vma_allocator = self.vma_allocator,
            });
        }

        self.main_deletion_queue.deinit(allocator, .{
            .device = self.device,
            .vma_allocator = self.vma_allocator,
        });

        self.swapchain.deinit(self.init_arena.allocator(), self.device);

        self.device.destroyDevice(null);
        self.instance.destroySurfaceKHR(self.surface, null);
        self.instance.destroyInstance(null);

        self.init_arena.deinit();
    }

    fn init_vulkan() void {}
    fn init_swapchain() void {}
    fn init_sync_structures() void {}
};

const vk_image = struct {
    fn transitionImage(
        device: DeviceProxy,
        cmd: vk.CommandBuffer,
        image: vk.Image,
        currentLayout: vk.ImageLayout,
        newLayout: vk.ImageLayout,
    ) void {
        const imageBarrier: vk.ImageMemoryBarrier2 = .{
            .src_stage_mask = .{ .all_commands_bit = true },
            .src_access_mask = .{ .memory_write_bit = true },
            .dst_stage_mask = .{ .all_commands_bit = true },
            .dst_access_mask = .{
                .memory_write_bit = true,
                .memory_read_bit = true,
            },
            .old_layout = currentLayout,
            .new_layout = newLayout,
            .subresource_range = vk_init.imageSubresourceRange(
                if (newLayout == .depth_attachment_optimal) .{ .depth_bit = true } else .{ .color_bit = true },
            ),
            .image = image,
            .src_queue_family_index = 0,
            .dst_queue_family_index = 0,
        };

        const dep_info: vk.DependencyInfo = .{
            .image_memory_barrier_count = 1,
            .p_image_memory_barriers = (&imageBarrier)[0..1],
        };

        device.cmdPipelineBarrier2(cmd, &dep_info);
    }

    fn copy_image_to_image(
        device: DeviceProxy,
        cmd: vk.CommandBuffer,
        source: vk.Image,
        destination: vk.Image,
        srcSize: vk.Extent2D,
        dstSize: vk.Extent2D,
    ) void {
        const blitRegion: vk.ImageBlit2 = .{
            .src_offsets = .{ .{ .x = 0, .y = 0, .z = 0 }, .{
                .x = @intCast(srcSize.width),
                .y = @intCast(srcSize.height),
                .z = 1,
            } },
            .dst_offsets = .{ .{ .x = 0, .y = 0, .z = 0 }, .{
                .x = @intCast(dstSize.width),
                .y = @intCast(dstSize.height),
                .z = 1,
            } },
            .src_subresource = .{
                .aspect_mask = .{ .color_bit = true },
                .base_array_layer = 0,
                .layer_count = 1,
                .mip_level = 0,
            },
            .dst_subresource = .{
                .aspect_mask = .{ .color_bit = true },
                .base_array_layer = 0,
                .layer_count = 1,
                .mip_level = 0,
            },
        };

        const blitInfo: vk.BlitImageInfo2 = .{
            .dst_image = destination,
            .dst_image_layout = .transfer_dst_optimal,
            .src_image = source,
            .src_image_layout = .transfer_src_optimal,
            .filter = .linear,
            .region_count = 1,
            .p_regions = (&blitRegion)[0..1],
        };
        device.cmdBlitImage2(cmd, &blitInfo);
    }
};

const vk_init = struct {
    fn imageCreateInfo(format: vk.Format, usage_flags: vk.ImageUsageFlags, extent: vk.Extent3D) vk.ImageCreateInfo {
        return vk.ImageCreateInfo{
            .image_type = .@"2d",

            .format = format,
            .extent = extent,

            .mip_levels = 1,
            .array_layers = 1,

            //for MSAA. we will not be using it by default, so default it to 1 sample per pixel.
            .samples = .{ .@"1_bit" = true },

            //optimal tiling, which means the image is stored on the best gpu format
            .tiling = .optimal,
            .usage = usage_flags,

            .initial_layout = .undefined,
            .sharing_mode = .exclusive,
        };
    }

    /// build a image-view for the depth image to use for rendering
    fn imageViewCreateInfo(format: vk.Format, image: vk.Image, aspect_flags: vk.ImageAspectFlags) vk.ImageViewCreateInfo {
        const info: vk.ImageViewCreateInfo = .{
            .view_type = .@"2d",
            .image = image,
            .format = format,
            .subresource_range = .{
                .base_mip_level = 0,
                .level_count = 1,
                .base_array_layer = 0,
                .layer_count = 1,
                .aspect_mask = aspect_flags,
            },

            .components = std.mem.zeroInit(vk.ComponentMapping, .{}),
        };
        return info;
    }

    fn semaphoreSubmitInfo(stage_mask: vk.PipelineStageFlags2, semaphore: vk.Semaphore) vk.SemaphoreSubmitInfo {
        return vk.SemaphoreSubmitInfo{
            .semaphore = semaphore,
            .stage_mask = stage_mask,
            .device_index = 0,
            .value = 1,
        };
    }

    fn commandBufferSubmitInfo(cmd: vk.CommandBuffer) vk.CommandBufferSubmitInfo {
        return vk.CommandBufferSubmitInfo{
            .command_buffer = cmd,
            .device_mask = 0,
        };
    }

    fn submitInfo(
        cmd: *const vk.CommandBufferSubmitInfo,
        signal_semaphore_info: ?*const vk.SemaphoreSubmitInfo,
        wait_semaphore_info: ?*const vk.SemaphoreSubmitInfo,
    ) vk.SubmitInfo2 {
        return vk.SubmitInfo2{
            .wait_semaphore_info_count = if (wait_semaphore_info == null) 0 else 1,
            .p_wait_semaphore_infos = if (wait_semaphore_info) |info| info[0..1] else null,

            .signal_semaphore_info_count = if (signal_semaphore_info == null) 0 else 1,
            .p_signal_semaphore_infos = if (signal_semaphore_info) |info| info[0..1] else null,

            .command_buffer_info_count = 1,
            .p_command_buffer_infos = cmd[0..1],
        };
    }

    fn imageSubresourceRange(aspect_mask: vk.ImageAspectFlags) vk.ImageSubresourceRange {
        const subImage: vk.ImageSubresourceRange = .{
            .aspect_mask = aspect_mask,
            .base_mip_level = 0,
            .level_count = vk.REMAINING_MIP_LEVELS,
            .base_array_layer = 0,
            .layer_count = vk.REMAINING_ARRAY_LAYERS,
        };

        return subImage;
    }

    fn initFrame(device: DeviceProxy, graphics_queue_family_index: u32) !FrameData {
        const command_pool_info: vk.CommandPoolCreateInfo = .{
            .flags = .{ .reset_command_buffer_bit = true },
            .queue_family_index = graphics_queue_family_index,
        };

        const command_pool = try device.createCommandPool(&command_pool_info, null);

        var main_command_buffer: vk.CommandBuffer = undefined;
        const cmd_alloc_info: vk.CommandBufferAllocateInfo = .{
            .command_pool = command_pool,
            .command_buffer_count = 1,
            .level = .primary,
        };
        try device.allocateCommandBuffers(&cmd_alloc_info, (&main_command_buffer)[0..1]);

        return .{
            .command_pool = command_pool,
            .render_fence = try device.createFence(&.{ .flags = .{ .signaled_bit = true } }, null),
            .swapchain_semaphore = try device.createSemaphore(&.{}, null),
            .render_semaphore = try device.createSemaphore(&.{}, null),
            .main_command_buffer = main_command_buffer,
            .deletion_queue = .init,
        };
    }

    pub fn createVkInstance(base_dispatch: Dispatch.Base, temp_alloc: Allocator) !vk.Instance {
        const appinfo = vk.ApplicationInfo{
            .p_application_name = "Vulkan Tutorial",
            .application_version = vk.makeApiVersion(1, 0, 0, 0),
            .p_engine_name = "No Engine",
            .engine_version = vk.makeApiVersion(1, 0, 0, 0),
            .api_version = vk.makeApiVersion(1, 3, 0, 0),
        };

        var glfw_extensions_count: u32 = undefined;
        const glfw_extensions = c.SDL_Vulkan_GetInstanceExtensions(&glfw_extensions_count) orelse
            return error.GLFWGetRequiredInstanceExtensionsFailed;

        if (enable_validation_layers) {
            try checkValidationLayerSupport(temp_alloc, base_dispatch);
        }

        const create_info = vk.InstanceCreateInfo{
            .p_application_info = &appinfo,
            .enabled_extension_count = glfw_extensions_count,
            .pp_enabled_extension_names = @ptrCast(glfw_extensions),
            .pp_enabled_layer_names = if (enable_validation_layers) @ptrCast(&validation_layers) else null,
            .enabled_layer_count = if (enable_validation_layers) @intCast(validation_layers.len) else 0,
        };

        // // TODO: add checking for extensions
        // const glfwExtensions = try base_dispatch.enumerateInstanceLayerPropertiesAlloc(alloc);
        // _ = glfwExtensions; // autofix

        return try base_dispatch.createInstance(
            &create_info,
            null,
        );
    }

    pub fn checkValidationLayerSupport(temp_alloc: Allocator, base_dispatch: Dispatch.Base) !void {
        const available_layers = try base_dispatch.enumerateInstanceLayerPropertiesAlloc(temp_alloc);
        defer temp_alloc.free(available_layers);

        outer: for (validation_layers) |validation_layer| {
            for (available_layers) |available_layer| {
                if (std.mem.eql(
                    u8,
                    std.mem.span(@as([*:0]const u8, @ptrCast(&available_layer.layer_name))),
                    validation_layer,
                )) {
                    continue :outer;
                }
            }
            return error.NotAllValidationLayersSupported;
        }
    }

    pub fn pickPhysicalDevice(instance: InstanceProxy, surface: vk.SurfaceKHR, temp_alloc: Allocator) !vk.PhysicalDevice {
        const physical_devices = try instance.enumeratePhysicalDevicesAlloc(temp_alloc);
        defer temp_alloc.free(physical_devices);

        if (physical_devices.len == 0) return error.NoPhysicalDeviceFound;

        for (physical_devices) |physical_device| {
            if (try isPhysicalDeviceSuitable(physical_device, instance.wrapper.*, surface, temp_alloc)) {
                return physical_device;
            }
        }

        return error.NoSuitablePhysicalDeviceFound;
    }

    pub fn isPhysicalDeviceSuitable(
        physical_device: vk.PhysicalDevice,
        instance_dispatch: Dispatch.Instance,
        surface: vk.SurfaceKHR,
        temp_alloc: Allocator,
    ) !bool {
        const formats = try instance_dispatch.getPhysicalDeviceSurfaceFormatsAllocKHR(physical_device, surface, temp_alloc);
        const present_modes = try instance_dispatch.getPhysicalDeviceSurfacePresentModesAllocKHR(physical_device, surface, temp_alloc);
        return (try findQueueFamilies(physical_device, instance_dispatch, surface, temp_alloc)).graphics_family != null and
            try checkDeviceExtensionSupport(physical_device, instance_dispatch, temp_alloc) and
            formats.len > 0 and
            present_modes.len > 0;
    }

    pub fn findQueueFamilies(
        physical_device: vk.PhysicalDevice,
        instance_dispatch: Dispatch.Instance,
        surface: vk.SurfaceKHR,
        temp_alloc: Allocator,
    ) !QueueFamilyIndices {
        var indices: QueueFamilyIndices = .{
            .graphics_family = null,
            .present_family = null,
        };

        const queue_families = try instance_dispatch.getPhysicalDeviceQueueFamilyPropertiesAlloc(physical_device, temp_alloc);

        // TODO: prefer queue that supports both graphics and KHR

        for (queue_families, 0..) |queue_familie, i| {
            if (queue_familie.queue_flags.graphics_bit) {
                indices.graphics_family = @intCast(i);
                break;
            }
        }

        for (queue_families, 0..) |_, i| {
            if ((try instance_dispatch.getPhysicalDeviceSurfaceSupportKHR(physical_device, @intCast(i), surface) != 0)) {
                indices.present_family = @intCast(i);
                break;
            }
        }

        return indices;
    }

    pub fn checkDeviceExtensionSupport(
        physical_device: vk.PhysicalDevice,
        instance_dispatch: Dispatch.Instance,
        temp_alloc: Allocator,
    ) !bool {
        const available_extensions = try instance_dispatch.enumerateDeviceExtensionPropertiesAlloc(physical_device, null, temp_alloc);

        outer: for (required_device_extensions) |required_device_extension| {
            for (available_extensions) |available_extension| {
                if (std.mem.eql(
                    u8,
                    std.mem.span(@as([*:0]const u8, @ptrCast(&available_extension.extension_name))),
                    required_device_extension,
                )) {
                    continue :outer;
                }
            }
            return false;
        }

        return true;
    }

    pub fn createLogicalDevice(
        physical_device: vk.PhysicalDevice,
        instance_dispatch: Dispatch.Instance,
        queue_family_indices: QueueFamilyIndices,
    ) !vk.Device {
        const indices = [_]u32{
            queue_family_indices.graphics_family.?,
            queue_family_indices.present_family.?,
        };

        const queue_prioritie: f32 = 1;
        var queue_create_infos_buff: [indices.len]vk.DeviceQueueCreateInfo = undefined;
        var queue_create_infos = std.ArrayListUnmanaged(vk.DeviceQueueCreateInfo).initBuffer(&queue_create_infos_buff);
        outer: for (indices, 0..) |indice, i| {
            for (indices[0..i]) |previous_indice| {
                if (previous_indice == indice) continue :outer;
            }

            queue_create_infos.appendAssumeCapacity(.{
                .queue_family_index = indice,
                .queue_count = 1,
                .p_queue_priorities = (&queue_prioritie)[0..1],
            });
        }

        var device_features_vk13 = vk.PhysicalDeviceVulkan13Features{
            .dynamic_rendering = vk.TRUE,
            .synchronization_2 = vk.TRUE,
        };

        var device_features_vk12 = vk.PhysicalDeviceVulkan12Features{
            .p_next = (&device_features_vk13)[0..1],
            .buffer_device_address = vk.TRUE,
            .descriptor_indexing = vk.TRUE,
        };

        const create_info = vk.DeviceCreateInfo{
            .p_next = (&device_features_vk12)[0..1],
            .p_queue_create_infos = queue_create_infos.items.ptr,
            .queue_create_info_count = @intCast(queue_create_infos.items.len),
            .pp_enabled_extension_names = @ptrCast(&required_device_extensions),
            .enabled_extension_count = required_device_extensions.len,
        };

        return try instance_dispatch.createDevice(physical_device, &create_info, null);
    }

    fn createSwapchain(
        alloc: std.mem.Allocator,
        temp_alloc: std.mem.Allocator,
        physical_device: vk.PhysicalDevice,
        device: DeviceProxy,
        surface: vk.SurfaceKHR,
        window_width: u32,
        window_height: u32,
        instance_dispatch: Dispatch.Instance,
    ) !SwapChain {

        // Get surface formats
        const surface_formats = try instance_dispatch.getPhysicalDeviceSurfaceFormatsAllocKHR(physical_device, surface, temp_alloc);

        // Try to find preferred format
        var swapchain_image_format = vk.Format.r8g8b8a8_srgb;
        var swapchain_image_color_space: vk.ColorSpaceKHR = undefined;
        var swapchain_image_format_found = false;

        for (surface_formats) |format| {
            if (swapchain_image_format == format.format) {
                swapchain_image_format_found = true;
                swapchain_image_color_space = format.color_space;
                break;
            }
        }

        // Try fallback format if preferred not found
        if (!swapchain_image_format_found) {
            swapchain_image_format = .b8g8r8a8_srgb;
            for (surface_formats) |format| {
                if (swapchain_image_format == format.format) {
                    swapchain_image_format_found = true;
                    swapchain_image_color_space = format.color_space;
                    break;
                }
            }

            if (!swapchain_image_format_found) {
                std.log.err("Cannot find suitable swapchain image format", .{});
                return error.SwapchainCreationFailed;
            }
        }

        const surface_capabilities = try instance_dispatch.getPhysicalDeviceSurfaceCapabilitiesKHR(physical_device, surface);

        // Create swapchain
        const min_image_count = if (surface_capabilities.min_image_count > 3) surface_capabilities.min_image_count else 3;

        const swapchain_extent = vk.Extent2D{
            .width = window_width,
            .height = window_height,
        };

        const swapchain_create_info = vk.SwapchainCreateInfoKHR{
            .surface = surface,
            .min_image_count = min_image_count,
            .image_format = swapchain_image_format,
            .image_color_space = swapchain_image_color_space,
            .image_extent = swapchain_extent,
            .image_array_layers = 1,
            .image_usage = .{
                .transfer_src_bit = true,
                .color_attachment_bit = true,
                .transfer_dst_bit = true,
            },
            .image_sharing_mode = .exclusive,
            .queue_family_index_count = 0,
            .p_queue_family_indices = null,
            .pre_transform = .{ .identity_bit_khr = true },
            .composite_alpha = .{ .opaque_bit_khr = true },
            .present_mode = .fifo_khr,
            .clipped = vk.FALSE,
            .old_swapchain = .null_handle,
        };

        const swapchain_handle = try device.createSwapchainKHR(&swapchain_create_info, null);
        errdefer device.destroySwapchainKHR(swapchain_handle, null);

        // Get swapchain images
        const images = try device.getSwapchainImagesAllocKHR(swapchain_handle, alloc);

        // Allocate arrays for images and image views
        var image_views = try alloc.alloc(vk.ImageView, images.len);
        errdefer alloc.free(image_views);

        // Create image views for each swapchain image
        const subresource_range = vk.ImageSubresourceRange{
            .aspect_mask = .{ .color_bit = true },
            .base_mip_level = 0,
            .level_count = 1,
            .base_array_layer = 0,
            .layer_count = 1,
        };

        for (images, 0..) |image, i| {
            const image_view_create_info = vk.ImageViewCreateInfo{
                .image = image,
                .view_type = .@"2d",
                .format = swapchain_image_format,
                .components = .{
                    .r = .identity,
                    .g = .identity,
                    .b = .identity,
                    .a = .identity,
                },
                .subresource_range = subresource_range,
            };

            errdefer {
                for (image_views[0..i]) |view| {
                    device.destroyImageView(view, null);
                }
            }

            image_views[i] = try device.createImageView(&image_view_create_info, null);
        }

        return .{
            .handle = swapchain_handle,
            .image_format = swapchain_image_format,
            .image_color_space = swapchain_image_color_space,
            .images = images,
            .image_views = image_views,
            .extent = swapchain_extent,
        };
    }
};
