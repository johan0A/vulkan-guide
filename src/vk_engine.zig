const std = @import("std");
const vk = @import("vulkan");
const c = @import("c");

const enable_validation_layers = true;
const validation_layers = [_][:0]const u8{"VK_LAYER_KHRONOS_validation"};

pub const required_device_extensions = [_][:0]const u8{vk.extensions.khr_swapchain.name};

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

const InstanceProxy = vk.InstanceProxy(apis);
const DeviceProxy = vk.DeviceProxy(apis);

const SwapChain = struct {
    vk_handle: vk.SwapchainKHR,
    image_format: vk.Format,
    image_color_space: vk.ColorSpaceKHR,
    images: []vk.Image,
    image_views: []vk.ImageView,
    extent: vk.Extent2D,

    fn deinit(self: SwapChain, alloc: Allocator, device: DeviceProxy) void {
        device.destroySwapchainKHR(self.vk_handle, null);
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
};

pub const VulkanEngine = struct {
    window: *c.SDL_Window,

    base_dispatch: Dispatch.Base,

    instance_proxy: InstanceProxy,
    device_proxy: DeviceProxy,

    instance: vk.Instance, // Vulkan library handle
    debug_messenger: vk.DebugUtilsMessengerEXT, // Vulkan debug output handle
    chosen_gpu: vk.PhysicalDevice, // GPU chosen as the default device
    device: vk.Device, // Vulkan device for commands
    surface: vk.SurfaceKHR, // Vulkan window surface

    swapchain: SwapChain,

    graphics_queue: vk.Queue,
    graphics_queue_family: u32,

    frame_number: u64,
    frames: [FrameData.FRAME_OVERLAP]FrameData,

    pub fn draw(self: *VulkanEngine) !void {
        _ = try self.device_proxy.waitForFences(1, (&self.currentFrame().render_fence)[0..1], vk.TRUE, 1e9);
        _ = try self.device_proxy.resetFences(1, (&self.currentFrame().render_fence)[0..1]);

        const swapchain_image_index: u32 = (try self.device_proxy.acquireNextImageKHR(
            self.swapchain.vk_handle,
            1e9,
            self.currentFrame().swapchain_semaphore,
            .null_handle,
        )).image_index;

        //naming it cmd for shorter writing
        const cmd: vk.CommandBuffer = self.currentFrame().main_command_buffer;

        // now that we are sure that the commands finished executing, we can safely
        // reset the command buffer to begin recording again.
        try self.device_proxy.resetCommandBuffer(cmd, .{});

        // begin the command buffer recording. We will use this command buffer exactly once, so we want to let vulkan know that
        try self.device_proxy.beginCommandBuffer(cmd, &.{ .flags = .{ .one_time_submit_bit = true } });
        {
            //make the swapchain image into writeable mode before rendering
            vk_image.transition_image(self.device_proxy, cmd, self.swapchain.images[swapchain_image_index], .undefined, .general);

            const flash = @abs(std.math.sin(@as(f32, @floatFromInt(self.frame_number)) / 120));
            var clear_value: vk.ClearColorValue = .{ .float_32 = .{ 0, 0, flash, 1 } };

            const clear_range: vk.ImageSubresourceRange = vk_init.imageSubresourceRange(.{ .color_bit = true });

            self.device_proxy.cmdClearColorImage(cmd, self.swapchain.images[swapchain_image_index], .general, &clear_value, 1, (&clear_range)[0..1]);

            //make the swapchain image into presentable mode
            vk_image.transition_image(self.device_proxy, cmd, self.swapchain.images[swapchain_image_index], .general, .present_src_khr);
        }
        try self.device_proxy.endCommandBuffer(cmd);

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
            try self.device_proxy.queueSubmit2(self.graphics_queue, 1, (&submit_info)[0..1], self.currentFrame().render_fence);
        }

        const present_info: vk.PresentInfoKHR = .{
            .p_swapchains = (&self.swapchain.vk_handle)[0..1],
            .swapchain_count = 1,
            .p_wait_semaphores = (&self.currentFrame().render_semaphore)[0..1],
            .wait_semaphore_count = 1,
            .p_image_indices = (&swapchain_image_index)[0..1],
        };
        _ = try self.device_proxy.queuePresentKHR(self.graphics_queue, &present_info);

        self.frame_number += 1;
    }

    pub inline fn currentFrame(self: *VulkanEngine) *FrameData {
        return &self.frames[self.frame_number % FrameData.FRAME_OVERLAP];
    }

    pub fn init(allocator: Allocator) !VulkanEngine {
        if (!c.SDL_Init(c.SDL_INIT_VIDEO)) return error.engine_init_failure;

        const window_width = 400;
        const window_height = 400;

        const window = c.SDL_CreateWindow("title", window_width, window_height, c.SDL_WINDOW_VULKAN) orelse return error.engine_init_failure;

        const base_dispatch = try Dispatch.Base.load(@as(vk.PfnGetInstanceProcAddr, @ptrCast(c.SDL_Vulkan_GetVkGetInstanceProcAddr())));

        const instance = try vk_init.createVkInstance(base_dispatch, allocator);
        const instance_dispatch = try allocator.create(Dispatch.Instance);
        instance_dispatch.* = try Dispatch.Instance.load(instance, base_dispatch.dispatch.vkGetInstanceProcAddr);
        const instance_proxy = InstanceProxy.init(instance, instance_dispatch);

        var surface: vk.SurfaceKHR = undefined;
        if (!c.SDL_Vulkan_CreateSurface(window, @ptrFromInt(@intFromEnum(instance)), null, @ptrCast(&surface))) return error.engine_init_failure;

        const physical_device = try vk_init.pickPhysicalDevice(instance_proxy, surface, allocator);
        const queue_family_indices = try vk_init.findQueueFamilies(physical_device, instance_dispatch.*, surface, allocator);

        const device = try vk_init.createLogicalDevice(physical_device, instance_dispatch.*, queue_family_indices);
        const device_dispatch = try allocator.create(Dispatch.Device);
        device_dispatch.* = try Dispatch.Device.load(device, instance_dispatch.dispatch.vkGetDeviceProcAddr);
        const device_proxy = DeviceProxy.init(device, device_dispatch);

        var frames: [FrameData.FRAME_OVERLAP]FrameData = undefined;
        try vk_init.initFramesCommands(device_proxy, queue_family_indices.graphics_family.?, &frames);
        try vk_init.initFramesSyncStructures(device_proxy, &frames);

        return .{
            .window = window,
            .base_dispatch = base_dispatch,
            .instance_proxy = instance_proxy,
            .device_proxy = device_proxy,
            .swapchain = try vk_init.createSwapchain(
                allocator,
                physical_device,
                device_proxy,
                surface,
                window_width,
                window_height,
                instance_dispatch.*,
            ),
            .instance = instance,
            .debug_messenger = .null_handle,
            .chosen_gpu = physical_device,
            .device = device,
            .surface = surface,

            .graphics_queue = device_proxy.getDeviceQueue(queue_family_indices.graphics_family.?, 0),
            .graphics_queue_family = queue_family_indices.graphics_family.?,

            .frame_number = 0,
            .frames = frames,
        };
    }

    pub fn deinit(self: VulkanEngine, allocator: Allocator) void {
        self.device_proxy.deviceWaitIdle() catch @panic(""); // TODO

        for (0..self.frames.len) |i| {
            self.device_proxy.destroyCommandPool(self.frames[i].command_pool, null);
            self.device_proxy.destroyFence(self.frames[i].render_fence, null);
            self.device_proxy.destroySemaphore(self.frames[i].swapchain_semaphore, null);
            self.device_proxy.destroySemaphore(self.frames[i].render_semaphore, null);
        }

        self.swapchain.deinit(allocator, self.device_proxy);

        self.device_proxy.destroyDevice(null);
        self.instance_proxy.destroySurfaceKHR(self.surface, null);
        self.instance_proxy.destroyInstance(null);

        allocator.destroy(self.device_proxy.wrapper);
        allocator.destroy(self.instance_proxy.wrapper);
    }

    fn init_vulkan() void {}
    fn init_swapchain() void {}
    fn init_sync_structures() void {}
};

const vk_image = struct {
    fn transition_image(device: DeviceProxy, cmd: vk.CommandBuffer, image: vk.Image, currentLayout: vk.ImageLayout, newLayout: vk.ImageLayout) void {
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
};

const vk_init = struct {
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

    fn submitInfo(cmd: *const vk.CommandBufferSubmitInfo, signal_semaphore_info: ?*const vk.SemaphoreSubmitInfo, wait_semaphore_info: ?*const vk.SemaphoreSubmitInfo) vk.SubmitInfo2 {
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

    fn initFramesCommands(device: DeviceProxy, graphics_queue_family_index: u32, frames: *[FrameData.FRAME_OVERLAP]FrameData) !void {
        const command_pool_info: vk.CommandPoolCreateInfo = .{
            .flags = .{ .reset_command_buffer_bit = true },
            .queue_family_index = graphics_queue_family_index,
        };

        for (0..frames.len) |i| {
            frames[i].command_pool = try device.createCommandPool(&command_pool_info, null);

            const cmd_alloc_info: vk.CommandBufferAllocateInfo = .{
                .command_pool = frames[i].command_pool,
                .command_buffer_count = 1,
                .level = .primary,
            };
            try device.allocateCommandBuffers(&cmd_alloc_info, (&frames[i].main_command_buffer)[0..1]);
        }
    }

    fn initFramesSyncStructures(device: DeviceProxy, frames: *[FrameData.FRAME_OVERLAP]FrameData) !void {
        for (0..frames.len) |i| {
            frames[i].render_fence = try device.createFence(&.{ .flags = .{ .signaled_bit = true } }, null);
            frames[i].swapchain_semaphore = try device.createSemaphore(&.{}, null);
            frames[i].render_semaphore = try device.createSemaphore(&.{}, null);
        }
    }

    pub fn createVkInstance(base_dispatch: Dispatch.Base, alloc: Allocator) !vk.Instance {
        const appinfo = vk.ApplicationInfo{
            .p_application_name = "Vulkan Tutorial",
            .application_version = vk.makeApiVersion(1, 0, 0, 0),
            .p_engine_name = "No Engine",
            .engine_version = vk.makeApiVersion(1, 0, 0, 0),
            .api_version = vk.makeApiVersion(1, 3, 0, 0),
        };

        var glfw_extensions_count: u32 = undefined;
        const glfw_extensions = c.SDL_Vulkan_GetInstanceExtensions(&glfw_extensions_count) orelse return error.GLFWGetRequiredInstanceExtensionsFailed;

        if (enable_validation_layers) {
            try checkValidationLayerSupport(alloc, base_dispatch);
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

    pub fn checkValidationLayerSupport(alloc: Allocator, base_dispatch: Dispatch.Base) !void {
        const available_layers = try base_dispatch.enumerateInstanceLayerPropertiesAlloc(alloc);
        defer alloc.free(available_layers);

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

    pub fn pickPhysicalDevice(instance: InstanceProxy, surface: vk.SurfaceKHR, alloc: Allocator) !vk.PhysicalDevice {
        const physical_devices = try instance.enumeratePhysicalDevicesAlloc(alloc);
        defer alloc.free(physical_devices);

        if (physical_devices.len == 0) return error.NoPhysicalDeviceFound;

        for (physical_devices) |physical_device| {
            if (try isPhysicalDeviceSuitable(physical_device, instance.wrapper.*, surface, alloc)) {
                return physical_device;
            }
        }

        return error.NoSuitablePhysicalDeviceFound;
    }

    pub fn isPhysicalDeviceSuitable(
        physical_device: vk.PhysicalDevice,
        instance_dispatch: Dispatch.Instance,
        surface: vk.SurfaceKHR,
        alloc: Allocator,
    ) !bool {
        const formats = try instance_dispatch.getPhysicalDeviceSurfaceFormatsAllocKHR(physical_device, surface, alloc);
        defer alloc.free(formats);
        const present_modes = try instance_dispatch.getPhysicalDeviceSurfacePresentModesAllocKHR(physical_device, surface, alloc);
        defer alloc.free(present_modes);
        return (try findQueueFamilies(physical_device, instance_dispatch, surface, alloc)).graphics_family != null and
            try checkDeviceExtensionSupport(physical_device, instance_dispatch, alloc) and
            formats.len > 0 and
            present_modes.len > 0;
    }

    pub fn findQueueFamilies(
        physical_device: vk.PhysicalDevice,
        instance_dispatch: Dispatch.Instance,
        surface: vk.SurfaceKHR,
        alloc: Allocator,
    ) !QueueFamilyIndices {
        var indices: QueueFamilyIndices = .{
            .graphics_family = null,
            .present_family = null,
        };

        const queue_families = try instance_dispatch.getPhysicalDeviceQueueFamilyPropertiesAlloc(physical_device, alloc);
        defer alloc.free(queue_families);

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

    pub fn checkDeviceExtensionSupport(physical_device: vk.PhysicalDevice, instance_dispatch: Dispatch.Instance, alloc: Allocator) !bool {
        const available_extensions = try instance_dispatch.enumerateDeviceExtensionPropertiesAlloc(physical_device, null, alloc);
        defer alloc.free(available_extensions);

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

    pub fn createLogicalDevice(physical_device: vk.PhysicalDevice, instance_dispatch: Dispatch.Instance, queue_family_indices: QueueFamilyIndices) !vk.Device {
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
        allocator: std.mem.Allocator,
        physical_device: vk.PhysicalDevice,
        device: DeviceProxy,
        surface: vk.SurfaceKHR,
        window_width: u32,
        window_height: u32,
        instance_dispatch: Dispatch.Instance,
    ) !SwapChain {
        const surface_capabilities = try instance_dispatch.getPhysicalDeviceSurfaceCapabilitiesKHR(physical_device, surface);
        // Get surface formats
        var surface_format_count: u32 = 0;
        _ = try instance_dispatch.getPhysicalDeviceSurfaceFormatsKHR(physical_device, surface, &surface_format_count, null);

        var surface_formats = try std.ArrayList(vk.SurfaceFormatKHR).initCapacity(allocator, surface_format_count);
        defer surface_formats.deinit();

        surface_formats.appendNTimesAssumeCapacity(undefined, surface_format_count);

        _ = try instance_dispatch.getPhysicalDeviceSurfaceFormatsKHR(physical_device, surface, &surface_format_count, surface_formats.items.ptr);

        // Try to find preferred format
        var swapchain_image_format = vk.Format.r8g8b8a8_srgb;
        var swapchain_image_color_space: vk.ColorSpaceKHR = undefined;
        var swapchain_image_format_found = false;

        for (surface_formats.items) |format| {
            if (swapchain_image_format == format.format) {
                swapchain_image_format_found = true;
                swapchain_image_color_space = format.color_space;
                break;
            }
        }

        // Try fallback format if preferred not found
        if (!swapchain_image_format_found) {
            swapchain_image_format = .b8g8r8a8_srgb;
            for (surface_formats.items) |format| {
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

        const vk_handle = try device.createSwapchainKHR(&swapchain_create_info, null);
        errdefer device.destroySwapchainKHR(vk_handle, null);

        // Get swapchain images
        var swapchain_image_count: u32 = 0;
        _ = try device.getSwapchainImagesKHR(vk_handle, &swapchain_image_count, null);

        // Allocate arrays for images and image views
        const images = try allocator.alloc(vk.Image, swapchain_image_count);
        errdefer allocator.free(images);

        var image_views = try allocator.alloc(vk.ImageView, swapchain_image_count);
        errdefer allocator.free(image_views);

        // Get the swapchain images
        _ = try device.getSwapchainImagesKHR(vk_handle, &swapchain_image_count, images.ptr);

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
            .vk_handle = vk_handle,
            .image_format = swapchain_image_format,
            .image_color_space = swapchain_image_color_space,
            .images = images,
            .image_views = image_views,
            .extent = swapchain_extent,
        };
    }
};
