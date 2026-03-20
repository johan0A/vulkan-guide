const validation_layers = [_][:0]const u8{"VK_LAYER_KHRONOS_validation"};
const required_device_extensions = [_][*:0]const u8{vk.extensions.khr_swapchain.name};

base_dispatch: vk.BaseWrapper,
instance: vk.InstanceProxy,
physical_device: vk.PhysicalDevice,
window_surface: vk.SurfaceKHR,
debug_messenger: vk.DebugUtilsMessengerEXT,

device: vk.DeviceProxy,
queues: Queues,
vma_allocator: c.VmaAllocator,

imm: ImmSubmit,

bindless_descriptors: BindlessDescriptors,
bindless_pipeline_layout: vk.PipelineLayout,

arena: std.heap.ArenaAllocator,

pub fn init(
    gpa: Allocator,
    scratch: *Scratch,
    window: *c.SDL_Window,
) !GraphicsCtx {
    const base_dispatch = vk.BaseWrapper.load(@as(vk.PfnGetInstanceProcAddr, @ptrCast(c.SDL_Vulkan_GetVkGetInstanceProcAddr())));

    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    const arena = arena_state.allocator();

    const instance_handle = try createVkInstance(scratch, base_dispatch, options.enable_validation_layers);
    const instance_dispatch = try arena.create(vk.InstanceWrapper);
    instance_dispatch.* = vk.InstanceWrapper.load(instance_handle, base_dispatch.dispatch.vkGetInstanceProcAddr.?);
    const instance: vk.InstanceProxy = .init(instance_handle, instance_dispatch);

    var window_surface: vk.SurfaceKHR = undefined;
    if (!c.SDL_Vulkan_CreateSurface(window, @ptrFromInt(@intFromEnum(instance_handle)), null, @ptrCast(&window_surface))) return error.engine_init_failure;

    const physical_device = try pickPhysicalDevice(scratch, instance, window_surface);

    const families = (try findQueueFamilies(scratch, physical_device, instance_dispatch.*, window_surface)).?;

    const device_handle = try createLogicalDevice(physical_device, instance_dispatch.*, families);
    const device_dispatch = try arena.create(vk.DeviceWrapper);
    device_dispatch.* = vk.DeviceWrapper.load(device_handle, instance_dispatch.dispatch.vkGetDeviceProcAddr.?);
    const device: vk.DeviceProxy = .init(device_handle, device_dispatch);

    const queues: Queues = .init(families, device);

    const debug_callback = struct {
        fn debugCallback(
            message_severity: vk.DebugUtilsMessageSeverityFlagsEXT,
            message_types: vk.DebugUtilsMessageTypeFlagsEXT,
            p_callback_data: ?*const vk.DebugUtilsMessengerCallbackDataEXT,
            p_user_data: ?*anyopaque,
        ) callconv(vk.vulkan_call_conv) vk.Bool32 {
            _ = .{ message_types, p_user_data };
            const callback_data = p_callback_data orelse @panic("");
            const message = std.mem.span(callback_data.p_message orelse "no message");
            if (message_severity.error_bit_ext) {
                std.log.err("Validation: {s}", .{message});
            } else if (message_severity.warning_bit_ext) {
                std.log.warn("Validation: {s}", .{message});
            } else {
                std.log.info("Validation: {s}", .{message});
            }
            std.debug.dumpCurrentStackTrace(.{});
            return .false;
        }
    };

    const debug_messenger_info: vk.DebugUtilsMessengerCreateInfoEXT = .{
        .message_severity = .{ .verbose_bit_ext = true, .warning_bit_ext = true, .error_bit_ext = true },
        .message_type = .{ .general_bit_ext = true, .validation_bit_ext = true, .performance_bit_ext = true },
        .pfn_user_callback = debug_callback.debugCallback,
    };
    const debug_messenger = try instance.createDebugUtilsMessengerEXT(&debug_messenger_info, null);

    var vma_allocator: c.VmaAllocator = undefined;
    if (c.vmaCreateAllocator(&.{
        .physicalDevice = @ptrFromInt(@intFromEnum(physical_device)),
        .device = @ptrFromInt(@intFromEnum(device_handle)),
        .instance = @ptrFromInt(@intFromEnum(instance_handle)),
        .flags = c.VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
        .pVulkanFunctions = &c.VmaVulkanFunctions{
            .vkGetDeviceProcAddr = @ptrCast(instance_dispatch.dispatch.vkGetDeviceProcAddr),
            .vkGetInstanceProcAddr = @ptrCast(base_dispatch.dispatch.vkGetInstanceProcAddr),
        },
    }, &vma_allocator) != 0) return error.vma_allocator_init_failed;

    const imm: ImmSubmit = try .init(device, queues);
    var bindless_descriptors: BindlessDescriptors = try .init(device);

    const push_range: vk.PushConstantRange = .{
        .offset = 0,
        .size = @sizeOf(vk_engine.GPUDrawPushConstants),
        .stage_flags = .{ .vertex_bit = true, .fragment_bit = true },
    };
    const bindless_pipeline_layout = try device.createPipelineLayout(&.{
        .set_layout_count = 1,
        .p_set_layouts = (&bindless_descriptors.layout)[0..1],
        .push_constant_range_count = 1,
        .p_push_constant_ranges = &.{push_range},
    }, null);

    return .{
        .base_dispatch = base_dispatch,
        .instance = instance,
        .physical_device = physical_device,
        .window_surface = window_surface,
        .debug_messenger = debug_messenger,

        .device = device,
        .queues = queues,
        .vma_allocator = vma_allocator,

        .imm = imm,

        .bindless_descriptors = bindless_descriptors,
        .bindless_pipeline_layout = bindless_pipeline_layout,

        .arena = arena_state,
    };
}

pub fn deinit(self: *GraphicsCtx, gpa: Allocator) void {
    self.device.deviceWaitIdle() catch {};
    self.device.destroyPipelineLayout(self.bindless_pipeline_layout, null);
    self.bindless_descriptors.deinit(gpa, self.device);
    self.imm.deinit(self.device);
    c.vmaDestroyAllocator(self.vma_allocator);
    self.device.destroyDevice(null);
    self.instance.destroySurfaceKHR(self.window_surface, null);
    self.instance.destroyDebugUtilsMessengerEXT(self.debug_messenger, null);
    self.instance.destroyInstance(null);
    self.arena.deinit();
}

pub const ImmSubmit = struct {
    fence: vk.Fence,
    cmd: vk.CommandBuffer,
    command_pool: vk.CommandPool,

    fn init(
        device: vk.DeviceProxy,
        queues: Queues,
    ) !ImmSubmit {
        const command_pool_info: vk.CommandPoolCreateInfo = .{
            .flags = .{ .reset_command_buffer_bit = true },
            .queue_family_index = queues.families.graphics,
        };

        const fence_create_info: vk.FenceCreateInfo = .{ .flags = .{ .signaled_bit = true } };

        const imm_command_pool = try device.createCommandPool(&command_pool_info, null);

        const imm_cmd_alloc_info: vk.CommandBufferAllocateInfo = .{
            .command_pool = imm_command_pool,
            .command_buffer_count = 1,
            .level = .primary,
        };

        var imm_command_buffer: vk.CommandBuffer = undefined;
        try device.allocateCommandBuffers(&imm_cmd_alloc_info, (&imm_command_buffer)[0..1]);

        const imm_fence = try device.createFence(&fence_create_info, null);

        return .{
            .cmd = imm_command_buffer,
            .command_pool = imm_command_pool,
            .fence = imm_fence,
        };
    }

    fn deinit(self: *ImmSubmit, device: vk.DeviceProxy) void {
        device.destroyFence(self.fence, null);
        device.destroyCommandPool(self.command_pool, null);
    }
};

pub const Queues = struct {
    families: Families,

    graphics: vk.Queue,
    present: vk.Queue,
    compute: vk.Queue,
    transfer: vk.Queue,

    pub const Families = struct {
        graphics: u32,
        present: u32,
        compute: u32,
        transfer: u32,

        pub fn unique(self: Families) struct { families: [4]u32, len: u32 } {
            var result: [4]u32 = undefined;
            var len: u32 = 0;
            const all = [_]u32{ self.graphics, self.present, self.compute, self.transfer };
            outer: for (all) |family| {
                for (result[0..len]) |existing| if (existing == family) continue :outer;
                result[len] = family;
                len += 1;
            }
            return .{ .families = result, .len = len };
        }
    };

    pub fn init(families: Families, device: vk.DeviceProxy) Queues {
        return .{
            .families = families,
            .graphics = device.getDeviceQueue(families.graphics, 0),
            .present = device.getDeviceQueue(families.present, 0),
            .compute = device.getDeviceQueue(families.compute, 0),
            .transfer = device.getDeviceQueue(families.transfer, 0),
        };
    }
};

pub const BindlessDescriptors = struct {
    pool: vk.DescriptorPool,
    layout: vk.DescriptorSetLayout,
    set: vk.DescriptorSet,

    free_texture_indices: std.ArrayList(u32),
    next_texture_index: u32,

    pub fn init(device: vk.DeviceProxy) !BindlessDescriptors {
        const max_textures = 16384;

        const pool_sizes = [_]vk.DescriptorPoolSize{
            .{ .type = .combined_image_sampler, .descriptor_count = max_textures },
        };

        const pool = try device.createDescriptorPool(&.{
            .flags = .{ .update_after_bind_bit = true },
            .max_sets = 1,
            .pool_size_count = pool_sizes.len,
            .p_pool_sizes = &pool_sizes,
        }, null);

        const bindings = [_]vk.DescriptorSetLayoutBinding{
            .{
                .binding = 0,
                .descriptor_type = .combined_image_sampler,
                .descriptor_count = max_textures,
                .stage_flags = .{ .fragment_bit = true },
                .p_immutable_samplers = null,
            },
        };

        const binding_flags = [_]vk.DescriptorBindingFlags{
            .{ .partially_bound_bit = true, .update_after_bind_bit = true },
        };
        const flags_info: vk.DescriptorSetLayoutBindingFlagsCreateInfo = .{
            .binding_count = binding_flags.len,
            .p_binding_flags = &binding_flags,
        };

        const layout = try device.createDescriptorSetLayout(&.{
            .p_next = &flags_info,
            .flags = .{ .update_after_bind_pool_bit = true },
            .binding_count = bindings.len,
            .p_bindings = &bindings,
        }, null);

        const alloc_info: vk.DescriptorSetAllocateInfo = .{
            .descriptor_pool = pool,
            .descriptor_set_count = 1,
            .p_set_layouts = &.{layout},
        };

        var set: vk.DescriptorSet = undefined;
        try device.allocateDescriptorSets(&alloc_info, (&set)[0..1]);

        return .{
            .pool = pool,
            .layout = layout,
            .set = set,
            .free_texture_indices = .empty,
            .next_texture_index = 0,
        };
    }

    pub fn deinit(self: *BindlessDescriptors, gpa: Allocator, device: vk.DeviceProxy) void {
        device.destroyDescriptorPool(self.pool, null);
        device.destroyDescriptorSetLayout(self.layout, null);
        self.free_texture_indices.deinit(gpa);
    }

    pub fn registerTexture(
        self: *BindlessDescriptors,
        gpa: Allocator,
        device: vk.DeviceProxy,
        view: vk.ImageView,
        sampler: vk.Sampler,
    ) !u32 {
        const index = try self.allocTextureIndex(gpa);
        device.updateDescriptorSets(&.{.{
            .dst_set = self.set,
            .dst_binding = 0,
            .dst_array_element = index,
            .descriptor_count = 1,
            .descriptor_type = .combined_image_sampler,
            .p_image_info = &.{.{
                .sampler = sampler,
                .image_view = view,
                .image_layout = .shader_read_only_optimal,
            }},
            .p_buffer_info = undefined,
            .p_texel_buffer_view = undefined,
        }}, null);
        return index;
    }

    pub fn allocTextureIndex(self: *BindlessDescriptors, gpa: Allocator) !u32 {
        if (self.free_texture_indices.pop()) |idx| {
            return idx;
        } else {
            const result = self.next_texture_index;
            self.next_texture_index += 1;
            try self.free_texture_indices.ensureTotalCapacity(gpa, self.next_texture_index);
            return result;
        }
    }

    pub fn releaseTexture(self: *BindlessDescriptors, idx: u32) void {
        self.free_texture_indices.appendAssumeCapacity(idx);
    }
};

pub fn createVkInstance(scratch: *Scratch, base_dispatch: vk.BaseWrapper, enable_validation_layers: bool) !vk.Instance {
    const checkpoint = scratch.checkpoint();
    defer scratch.restoreCheckpoint(checkpoint);

    const appinfo = vk.ApplicationInfo{
        .p_application_name = "Vulkan Tutorial",
        .application_version = @bitCast(vk.makeApiVersion(1, 0, 0, 0)),
        .p_engine_name = "No Engine",
        .engine_version = @bitCast(vk.makeApiVersion(1, 0, 0, 0)),
        .api_version = @bitCast(vk.makeApiVersion(1, 3, 0, 0)),
    };

    const sdl_required_extensions = blk: {
        var sdl_required_extensions_count: u32 = undefined;
        const sdl_required_extensions_ptr = c.SDL_Vulkan_GetInstanceExtensions(&sdl_required_extensions_count) orelse
            return error.SDL_Vulkan_GetInstanceExtensionsFailed;
        break :blk sdl_required_extensions_ptr[0..sdl_required_extensions_count];
    };

    const available_extensions = try base_dispatch.enumerateInstanceExtensionPropertiesAlloc(null, scratch.allocator());
    for (sdl_required_extensions) |required_ext| {
        for (available_extensions) |available_ext| {
            if (std.mem.eql(u8, std.mem.span(required_ext), std.mem.sliceTo(&available_ext.extension_name, 0))) break;
        } else {
            return error.extensionRequiredBySdlIsNotAvailable;
        }
    }

    if (enable_validation_layers) try checkValidationLayerSupport(scratch, base_dispatch);

    var extensions: std.ArrayList([*:0]const u8) = .empty;
    try extensions.appendSlice(scratch.allocator(), @ptrCast(sdl_required_extensions));
    try extensions.appendSlice(scratch.allocator(), &.{
        vk.extensions.ext_debug_utils.name.ptr,
    });

    const create_info = vk.InstanceCreateInfo{
        .p_application_info = &appinfo,
        .enabled_extension_count = @intCast(extensions.items.len),
        .pp_enabled_extension_names = @ptrCast(extensions.items),
        .pp_enabled_layer_names = if (enable_validation_layers) @ptrCast(&validation_layers) else null,
        .enabled_layer_count = if (enable_validation_layers) @intCast(validation_layers.len) else 0,
    };

    return try base_dispatch.createInstance(&create_info, null);
}

pub fn checkValidationLayerSupport(scratch: *Scratch, base_dispatch: vk.BaseWrapper) !void {
    const checkpoint = scratch.checkpoint();
    defer scratch.restoreCheckpoint(checkpoint);
    const available_layers = try base_dispatch.enumerateInstanceLayerPropertiesAlloc(scratch.allocator());
    for (validation_layers) |validation_layer| {
        for (available_layers) |available_layer| {
            if (std.mem.eql(u8, std.mem.sliceTo(&available_layer.layer_name, 0), validation_layer)) break;
        } else return error.NotAllValidationLayersSupported;
    }
}

pub fn createLogicalDevice(
    physical_device: vk.PhysicalDevice,
    instance_dispatch: vk.InstanceWrapper,
    families: Queues.Families,
) !vk.Device {
    const unique = families.unique();
    const queue_priorities: [1]f32 = .{1};

    var queue_create_infos: [4]vk.DeviceQueueCreateInfo = undefined;
    for (unique.families[0..unique.len], queue_create_infos[0..unique.len]) |family, *info| {
        info.* = .{
            .queue_family_index = family,
            .queue_count = queue_priorities.len,
            .p_queue_priorities = &queue_priorities,
        };
    }

    var device_features_vk13: vk.PhysicalDeviceVulkan13Features = .{
        .dynamic_rendering = .true,
        .synchronization_2 = .true,
    };
    var device_features_vk12: vk.PhysicalDeviceVulkan12Features = .{
        .p_next = &device_features_vk13,
        .buffer_device_address = .true,
        .runtime_descriptor_array = .true,
        .descriptor_binding_partially_bound = .true,
        .descriptor_binding_sampled_image_update_after_bind = .true,
        .descriptor_binding_storage_buffer_update_after_bind = .true,
        .scalar_block_layout = .true,
    };
    const device_features_vk11: vk.PhysicalDeviceVulkan11Features = .{
        .p_next = &device_features_vk12,
        .shader_draw_parameters = .true,
    };
    return try instance_dispatch.createDevice(physical_device, &.{
        .p_next = &device_features_vk11,
        .p_queue_create_infos = queue_create_infos[0..unique.len].ptr,
        .queue_create_info_count = unique.len,
        .pp_enabled_extension_names = &required_device_extensions,
        .enabled_extension_count = required_device_extensions.len,
        .p_enabled_features = &.{
            .shader_int_64 = .true,
            .sampler_anisotropy = .true,
            .multi_draw_indirect = .true,
            .robust_buffer_access = .true,
        },
    }, null);
}

pub fn pickPhysicalDevice(scratch: *Scratch, instance: vk.InstanceProxy, surface: vk.SurfaceKHR) !vk.PhysicalDevice {
    const checkpoint = scratch.checkpoint();
    defer scratch.restoreCheckpoint(checkpoint);

    const physical_devices = try instance.enumeratePhysicalDevicesAlloc(scratch.allocator());

    if (physical_devices.len == 0) return error.NoPhysicalDeviceFound;

    for (physical_devices) |physical_device| {
        const is_suitable = blk: {
            const formats = try instance.getPhysicalDeviceSurfaceFormatsAllocKHR(physical_device, surface, scratch.allocator());
            const present_modes = try instance.getPhysicalDeviceSurfacePresentModesAllocKHR(physical_device, surface, scratch.allocator());
            const has_families = (try findQueueFamilies(scratch, physical_device, instance.wrapper.*, surface)) != null;
            break :blk has_families and
                try checkDeviceExtensionSupport(physical_device, instance.wrapper.*, scratch) and
                formats.len > 0 and
                present_modes.len > 0;
        };

        if (is_suitable) return physical_device;
    }

    return error.NoSuitablePhysicalDeviceFound;
}

pub fn checkDeviceExtensionSupport(
    physical_device: vk.PhysicalDevice,
    instance_dispatch: vk.InstanceWrapper,
    scratch: *Scratch,
) !bool {
    const checkpoint = scratch.checkpoint();
    defer scratch.restoreCheckpoint(checkpoint);

    const available_extensions = try instance_dispatch.enumerateDeviceExtensionPropertiesAlloc(physical_device, null, scratch.allocator());

    for (required_device_extensions) |required_device_extension| {
        for (available_extensions) |available_extension| {
            if (std.mem.eql(
                u8,
                std.mem.sliceTo(&available_extension.extension_name, 0),
                std.mem.sliceTo(required_device_extension, 0),
            )) break;
        } else {
            return false;
        }
    }

    return true;
}

pub fn findQueueFamilies(
    scratch: *Scratch,
    physical_device: vk.PhysicalDevice,
    instance_dispatch: vk.InstanceWrapper,
    surface: vk.SurfaceKHR,
) !?Queues.Families {
    const checkpoint = scratch.checkpoint();
    defer scratch.restoreCheckpoint(checkpoint);

    const queue_families = try instance_dispatch.getPhysicalDeviceQueueFamilyPropertiesAlloc(physical_device, scratch.allocator());

    var graphics_family: ?u32 = null;
    for (queue_families, 0..) |family, i| {
        if (family.queue_flags.graphics_bit) {
            graphics_family = @intCast(i);
            break;
        }
    }
    const gfx = graphics_family orelse return null;

    var present_family: ?u32 = null;
    for (queue_families, 0..) |_, i| {
        const idx: u32 = @intCast(i);
        if (try instance_dispatch.getPhysicalDeviceSurfaceSupportKHR(physical_device, idx, surface) == .true) {
            present_family = idx;
            if (idx == gfx) break;
        }
    }
    const present = present_family orelse return null;

    var compute: u32 = gfx;
    for (queue_families, 0..) |family, i| {
        if (family.queue_flags.compute_bit and !family.queue_flags.graphics_bit) {
            compute = @intCast(i);
            break;
        }
    }

    var transfer: u32 = gfx;
    for (queue_families, 0..) |family, i| {
        if (family.queue_flags.transfer_bit and !family.queue_flags.graphics_bit and !family.queue_flags.compute_bit) {
            transfer = @intCast(i);
            break;
        }
    }

    return .{ .graphics = gfx, .present = present, .compute = compute, .transfer = transfer };
}

const GraphicsCtx = @This();
const vk = @import("vulkan");
const c = @import("c");
const std = @import("std");
const Scratch = @import("scratch_allocator");
const options = @import("options");

const vk_engine = @import("vk_engine.zig"); // TODO: remove

const Allocator = std.mem.Allocator;
