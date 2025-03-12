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
    base: Base,
    instance: Instance,
    device: Device,
};

const SwapChain = struct {
    vk_handle: vk.SwapchainKHR,
    image_format: vk.Format,
    image_color_space: vk.ColorSpaceKHR,
    images: []vk.Image,
    image_views: []vk.ImageView,
    extent: vk.Extent2D,

    fn deinit(self: SwapChain, alloc: Allocator, device: vk.Device, device_dispatch: Dispatch.Device) void {
        device_dispatch.destroySwapchainKHR(device, self.vk_handle, null);
        for (self.image_views) |image_view| {
            device_dispatch.destroyImageView(device, image_view, null);
        }
        alloc.free(self.images);
        alloc.free(self.image_views);
    }
};

pub const VulkanEngine = struct {
    window: *c.SDL_Window,

    dispatch: Dispatch,

    instance: vk.Instance, // Vulkan library handle
    debug_messenger: vk.DebugUtilsMessengerEXT, // Vulkan debug output handle
    chosen_gpu: vk.PhysicalDevice, // GPU chosen as the default device
    device: vk.Device, // Vulkan device for commands
    surface: vk.SurfaceKHR, // Vulkan window surface

    swapchain: SwapChain,

    pub fn init(allocator: Allocator) !VulkanEngine {
        if (!c.SDL_Init(c.SDL_INIT_VIDEO)) return error.engine_init_failure;

        const window_width = 400;
        const window_height = 400;

        const window = c.SDL_CreateWindow("title", window_width, window_height, c.SDL_WINDOW_VULKAN) orelse return error.engine_init_failure;

        const base_dispatch = try Dispatch.Base.load(@as(vk.PfnGetInstanceProcAddr, @ptrCast(c.SDL_Vulkan_GetVkGetInstanceProcAddr())));

        const instance = try vk_init.createVkInstance(base_dispatch, allocator);
        const instance_dispatch = try Dispatch.Instance.load(instance, base_dispatch.dispatch.vkGetInstanceProcAddr);

        var surface: vk.SurfaceKHR = undefined;
        if (!c.SDL_Vulkan_CreateSurface(window, @ptrFromInt(@intFromEnum(instance)), null, @ptrCast(&surface))) return error.engine_init_failure;

        const physical_device = try vk_init.pickPhysicalDevice(instance, instance_dispatch, surface, allocator);
        const queue_family_indices = try vk_init.findQueueFamilies(physical_device, instance_dispatch, surface, allocator);

        const device = try vk_init.createLogicalDevice(physical_device, queue_family_indices, instance_dispatch);
        const device_dispatch = try Dispatch.Device.load(device, instance_dispatch.dispatch.vkGetDeviceProcAddr);

        return .{
            .window = window,

            .dispatch = Dispatch{
                .base = base_dispatch,
                .device = device_dispatch,
                .instance = instance_dispatch,
            },

            .swapchain = try vk_init.createSwapchain(
                allocator,
                physical_device,
                device,
                surface,
                window_width,
                window_height,
                instance_dispatch,
                device_dispatch,
            ),
            .instance = instance,
            .debug_messenger = .null_handle,
            .chosen_gpu = physical_device,
            .device = device,
            .surface = surface,
        };
    }

    pub fn deinit(self: VulkanEngine, allocator: Allocator) void {
        self.swapchain.deinit(allocator, self.device, self.dispatch.device);

        self.dispatch.device.destroyDevice(self.device, null);
        self.dispatch.instance.destroySurfaceKHR(self.instance, self.surface, null);
        self.dispatch.instance.destroyInstance(self.instance, null);
    }

    fn init_vulkan() void {}
    fn init_swapchain() void {}
    fn init_commands() void {}
    fn init_sync_structures() void {}
};

const vk_init = struct {
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

    pub fn pickPhysicalDevice(instance: vk.Instance, instance_dispatch: Dispatch.Instance, surface: vk.SurfaceKHR, alloc: Allocator) !vk.PhysicalDevice {
        const devices = try instance_dispatch.enumeratePhysicalDevicesAlloc(instance, alloc);
        defer alloc.free(devices);

        if (devices.len == 0) return error.NoPhysicalDeviceFound;

        for (devices) |device| {
            if (try isPhysicalDeviceSuitable(device, instance_dispatch, surface, alloc)) {
                return device;
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

    pub fn createLogicalDevice(physical_device: vk.PhysicalDevice, queue_family_indices: QueueFamilyIndices, instance_dispatch: Dispatch.Instance) !vk.Device {
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
                .p_queue_priorities = @ptrCast(&queue_prioritie),
            });
        }

        var device_features_vk13 = vk.PhysicalDeviceVulkan13Features{
            .dynamic_rendering = vk.TRUE,
            .synchronization_2 = vk.TRUE,
        };

        var device_features_vk12 = vk.PhysicalDeviceVulkan12Features{
            .p_next = @ptrCast(&device_features_vk13),
            .buffer_device_address = vk.TRUE,
            .descriptor_indexing = vk.TRUE,
        };

        const create_info = vk.DeviceCreateInfo{
            .p_next = @ptrCast(&device_features_vk12),
            .p_queue_create_infos = queue_create_infos.items.ptr,
            .queue_create_info_count = @intCast(queue_create_infos.items.len),
            // TODO: apparently validation layers for device have been deprecated so should remove ?
            .pp_enabled_layer_names = if (enable_validation_layers) @ptrCast(&validation_layers) else null,
            .enabled_layer_count = if (enable_validation_layers) @intCast(validation_layers.len) else 0,
            .pp_enabled_extension_names = @ptrCast(&required_device_extensions),
            .enabled_extension_count = required_device_extensions.len,
        };

        return try instance_dispatch.createDevice(physical_device, &create_info, null);
    }

    fn createSwapchain(
        allocator: std.mem.Allocator,
        physical_device: vk.PhysicalDevice,
        device: vk.Device,
        surface: vk.SurfaceKHR,
        window_width: u32,
        window_height: u32,
        instance_dispatch: Dispatch.Instance,
        device_dispatch: Dispatch.Device,
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
            .image_usage = .{ .color_attachment_bit = true },
            .image_sharing_mode = .exclusive,
            .queue_family_index_count = 0,
            .p_queue_family_indices = null,
            .pre_transform = .{ .identity_bit_khr = true },
            .composite_alpha = .{ .opaque_bit_khr = true },
            .present_mode = .fifo_khr,
            .clipped = vk.FALSE,
            .old_swapchain = .null_handle,
        };

        const vk_handle = try device_dispatch.createSwapchainKHR(device, &swapchain_create_info, null);
        errdefer device_dispatch.destroySwapchainKHR(device, vk_handle, null);

        // Get swapchain images
        var swapchain_image_count: u32 = 0;
        _ = try device_dispatch.getSwapchainImagesKHR(device, vk_handle, &swapchain_image_count, null);

        // Allocate arrays for images and image views
        const images = try allocator.alloc(vk.Image, swapchain_image_count);
        errdefer allocator.free(images);

        var image_views = try allocator.alloc(vk.ImageView, swapchain_image_count);
        errdefer allocator.free(image_views);

        // Get the swapchain images
        _ = try device_dispatch.getSwapchainImagesKHR(device, vk_handle, &swapchain_image_count, images.ptr);

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
                    device_dispatch.destroyImageView(device, view, null);
                }
            }

            image_views[i] = try device_dispatch.createImageView(device, &image_view_create_info, null);
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
