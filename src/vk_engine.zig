const std = @import("std");
const Allocator = std.mem.Allocator;

const vk = @import("vulkan");
const c = @import("c");
const tracy = @import("tracy");
const shaders = @import("shaders");

// TODO: make those not globals?
const enable_validation_layers = true;
const validation_layers = [_][:0]const u8{"VK_LAYER_KHRONOS_validation"};
const required_device_extensions = [_][:0]const u8{vk.extensions.khr_swapchain.name};

const ShaderData = struct {
    /// in bytes
    size: usize,
    ptr: [*]const u32,
};

const AllocatedImage = struct {
    image: vk.Image,
    image_view: vk.ImageView,
    allocation: c.VmaAllocation,
    image_extent: vk.Extent3D,
    image_format: vk.Format,
};

pub const QueueFamilyIndices = struct {
    graphics_family: ?u32,
    present_family: ?u32,
};

pub const PipelineBuilder = struct {
    shaderStages: std.ArrayListUnmanaged(vk.PipelineShaderStageCreateInfo) = .empty,

    inputAssembly: vk.PipelineInputAssemblyStateCreateInfo = std.mem.zeroInit(vk.PipelineInputAssemblyStateCreateInfo, .{}),
    rasterizer: vk.PipelineRasterizationStateCreateInfo = std.mem.zeroInit(vk.PipelineRasterizationStateCreateInfo, .{}),
    colorBlendAttachment: vk.PipelineColorBlendAttachmentState = std.mem.zeroInit(vk.PipelineColorBlendAttachmentState, .{}),
    multisampling: vk.PipelineMultisampleStateCreateInfo = std.mem.zeroInit(vk.PipelineMultisampleStateCreateInfo, .{}),
    pipelineLayout: vk.PipelineLayout = .null_handle,
    depthStencil: vk.PipelineDepthStencilStateCreateInfo = std.mem.zeroInit(vk.PipelineDepthStencilStateCreateInfo, .{}),
    renderInfo: vk.PipelineRenderingCreateInfo = std.mem.zeroInit(vk.PipelineRenderingCreateInfo, .{}),
    colorAttachmentformat: vk.Format = .undefined,

    // PipelineBuilder(){ clear(); }

    // void clear();

    pub fn buildPipeline(self: PipelineBuilder, device: vk.DeviceProxy) !vk.Pipeline {
        // make viewport state from our stored viewport and scissor.
        // at the moment we wont support multiple viewports or scissors
        const viewportState: vk.PipelineViewportStateCreateInfo = .{
            .viewport_count = 1,
            .scissor_count = 1,
        };

        // setup dummy color blending. We arent using transparent objects yet
        // the blending is just "no blend", but we do write to the color attachment
        const colorBlending: vk.PipelineColorBlendStateCreateInfo = .{
            .logic_op_enable = vk.FALSE,
            .logic_op = .copy,
            .attachment_count = 1,
            .p_attachments = (&self.colorBlendAttachment)[0..1],

            .blend_constants = .{ 0, 0, 0, 0 },
        };

        // completely clear VertexInputStateCreateInfo, as we have no need for it
        const vertexInputInfo = vk.PipelineVertexInputStateCreateInfo{};

        const states = [_]vk.DynamicState{ .viewport, .scissor };
        const dynamicInfo: vk.PipelineDynamicStateCreateInfo = .{
            .p_dynamic_states = &states,
            .dynamic_state_count = @intCast(states.len),
        };

        // build the actual pipeline
        // we now use all of the info structs we have been writing into into this one
        // to create the pipeline
        const pipelineInfo: vk.GraphicsPipelineCreateInfo = .{
            // connect the renderInfo to the pNext extension mechanism
            .p_next = &self.renderInfo,

            .stage_count = @intCast(self.shaderStages.items.len),
            .p_stages = self.shaderStages.items.ptr,
            .p_vertex_input_state = &vertexInputInfo,
            .p_input_assembly_state = &self.inputAssembly,
            .p_viewport_state = &viewportState,
            .p_rasterization_state = &self.rasterizer,
            .p_multisample_state = &self.multisampling,
            .p_color_blend_state = &colorBlending,
            .p_depth_stencil_state = &self.depthStencil,
            .layout = self.pipelineLayout,

            .p_dynamic_state = &dynamicInfo,

            .subpass = 0, // undefined
            .base_pipeline_index = 0, // undefined
        };

        // TODO: handle error ???
        // its easy to error out on create graphics pipeline, so we handle it a bit
        // better than the common VK_CHECK case
        var newPipeline: vk.Pipeline = undefined;
        _ = try device.createGraphicsPipelines(.null_handle, 1, (&pipelineInfo)[0..1], null, (&newPipeline)[0..1]);
        return newPipeline;
    }

    pub fn setShaders(
        self: *PipelineBuilder,
        vertexShader: vk.ShaderModule,
        fragmentShader: vk.ShaderModule,
        alloc: Allocator,
    ) !void {
        self.shaderStages.clearRetainingCapacity();

        try self.shaderStages.append(alloc, vk_init.pipelineShaderStageCreateInfo(.{ .vertex_bit = true }, vertexShader));
        try self.shaderStages.append(alloc, vk_init.pipelineShaderStageCreateInfo(.{ .fragment_bit = true }, fragmentShader));
    }

    pub fn setInputTopology(self: *PipelineBuilder, topology: vk.PrimitiveTopology) void {
        self.inputAssembly.topology = topology;
        // we are not going to use primitive restart on the entire tutorial so leave
        // it on false
        self.inputAssembly.primitive_restart_enable = vk.FALSE;
    }

    pub fn setPolygonMode(self: *PipelineBuilder, mode: vk.PolygonMode) void {
        self.rasterizer.polygon_mode = mode;
        self.rasterizer.line_width = 1;
    }

    pub fn setCullMode(self: *PipelineBuilder, cullMode: vk.CullModeFlags, frontFace: vk.FrontFace) void {
        self.rasterizer.cull_mode = cullMode;
        self.rasterizer.front_face = frontFace;
    }

    pub fn setMultisamplingNone(self: *PipelineBuilder) void {
        self.multisampling.sample_shading_enable = vk.FALSE;
        // multisampling defaulted to no multisampling (1 sample per pixel)
        self.multisampling.rasterization_samples = .{ .@"1_bit" = true };
        self.multisampling.min_sample_shading = 1;
        self.multisampling.p_sample_mask = null;
        // no alpha to coverage either
        self.multisampling.alpha_to_coverage_enable = vk.FALSE;
        self.multisampling.alpha_to_one_enable = vk.FALSE;
    }

    pub fn disableBlending(self: *PipelineBuilder) void {
        // default write mask
        self.colorBlendAttachment.color_write_mask = .{ .r_bit = true, .g_bit = true, .b_bit = true, .a_bit = true };
        // no blending
        self.colorBlendAttachment.blend_enable = vk.FALSE;
    }

    pub fn setColorAttachmentFormat(self: *PipelineBuilder, format: vk.Format) void {
        self.colorAttachmentformat = format;
        // connect the format to the renderInfo  structure
        self.renderInfo.color_attachment_count = 1;
        self.renderInfo.p_color_attachment_formats = (&self.colorAttachmentformat)[0..1];
    }

    pub fn setDepthFormat(self: *PipelineBuilder, format: vk.Format) void {
        self.renderInfo.depth_attachment_format = format;
    }

    pub fn disableDepthtest(self: *PipelineBuilder) void {
        self.depthStencil.depth_test_enable = vk.FALSE;
        self.depthStencil.depth_write_enable = vk.FALSE;
        self.depthStencil.depth_compare_op = .never;
        self.depthStencil.depth_bounds_test_enable = vk.FALSE;
        self.depthStencil.stencil_test_enable = vk.FALSE;
        // self.depthStencil.front = .{
        //     .fail_op = .keep, // undefined
        //     .pass_op = .keep, // undefined
        //     .depth_fail_op = .keep, // undefined
        // };
        // self.depthStencil.back = .{};
        self.depthStencil.min_depth_bounds = 0;
        self.depthStencil.max_depth_bounds = 1;
    }

    fn deinit(self: *PipelineBuilder, alloc: Allocator) void {
        self.shaderStages.deinit(alloc);
    }
};

const DeletionQueue = struct {
    const DeinitContext = struct {
        device: vk.DeviceProxy,
        vma_allocator: ?c.VmaAllocator,
    };

    const QueueItem = union(enum) {
        vma_allocator: c.VmaAllocator,
        image_view: vk.ImageView,
        vma_allocated_image: struct { image: vk.Image, allocation: c.VmaAllocation },
        descriptor_allocator: DescriptorAllocator,
        descriptor_set_layout: vk.DescriptorSetLayout,
        pipeline_layout: vk.PipelineLayout,
        pipeline: vk.Pipeline,
        command_pool: vk.CommandPool,
        fence: vk.Fence,
        descriptor_pool: vk.DescriptorPool,
        imgui_impl_vulkan: void,

        fn deinit(self: QueueItem, context: DeinitContext) void {
            switch (self) {
                .vma_allocator => |item| c.vmaDestroyAllocator(item),
                .image_view => |item| context.device.destroyImageView(item, null),
                .vma_allocated_image => |item| c.vmaDestroyImage(context.vma_allocator.?, @ptrFromInt(@intFromEnum(item.image)), item.allocation),
                .descriptor_allocator => |item| item.destroyPool(context.device),
                .descriptor_set_layout => |item| context.device.destroyDescriptorSetLayout(item, null),
                .pipeline_layout => |item| context.device.destroyPipelineLayout(item, null),
                .pipeline => |item| context.device.destroyPipeline(item, null),
                .command_pool => |item| context.device.destroyCommandPool(item, null),
                .fence => |item| context.device.destroyFence(item, null),
                .descriptor_pool => |item| context.device.destroyDescriptorPool(item, null),
                .imgui_impl_vulkan => c.cImGui_ImplVulkan_Shutdown(),
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

const DescriptorAllocator = struct {
    const PoolSizeRatio = struct {
        type: vk.DescriptorType,
        ratio: f32,
    };

    pool: vk.DescriptorPool,

    pub fn initPool(temp: Allocator, device: vk.DeviceProxy, max_sets: u32, pool_ratios: []const PoolSizeRatio) !DescriptorAllocator {
        const pool_sizes = try temp.alloc(vk.DescriptorPoolSize, pool_ratios.len);
        for (pool_sizes, pool_ratios) |*size, ratio| {
            size.* = vk.DescriptorPoolSize{
                .type = ratio.type,
                .descriptor_count = @intFromFloat(ratio.ratio * @as(f32, @floatFromInt(max_sets))),
            };
        }

        const pool_info: vk.DescriptorPoolCreateInfo = .{
            .max_sets = max_sets,
            .pool_size_count = @intCast(pool_sizes.len),
            .p_pool_sizes = pool_sizes.ptr,
        };

        return .{
            .pool = try device.createDescriptorPool(&pool_info, null),
        };
    }

    pub fn clearDescriptors(self: DescriptorAllocator, device: vk.DeviceProxy) void {
        device.resetDescriptorPool(self.pool, .{});
    }

    pub fn destroyPool(self: DescriptorAllocator, device: vk.DeviceProxy) void {
        device.destroyDescriptorPool(self.pool, null);
    }

    pub fn allocate(self: DescriptorAllocator, device: vk.DeviceProxy, layout: vk.DescriptorSetLayout) !vk.DescriptorSet {
        const alloc_info: vk.DescriptorSetAllocateInfo = .{
            .descriptor_pool = self.pool,
            .descriptor_set_count = 1,
            .p_set_layouts = (&layout)[0..1],
        };
        var ds: vk.DescriptorSet = undefined;
        try device.allocateDescriptorSets(&alloc_info, (&ds)[0..1]);
        return ds;
    }
};

const SwapChain = struct {
    handle: vk.SwapchainKHR,
    image_format: vk.Format,
    image_color_space: vk.ColorSpaceKHR,
    images: []vk.Image,
    image_views: []vk.ImageView,
    extent: vk.Extent2D,

    fn deinit(self: SwapChain, alloc: Allocator, device: vk.DeviceProxy) void {
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

    base_dispatch: vk.BaseWrapper,

    instance: vk.InstanceProxy,
    device: vk.DeviceProxy,

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

    draw_image_descriptors: vk.DescriptorSet,
    draw_image_descriptor_set_layout: vk.DescriptorSetLayout,

    globalDescriptorAllocator: DescriptorAllocator,

    background_effects: []ComputeEffect,
    active_background_effect: u32,

    // immediate submit structures
    imm_fence: vk.Fence,
    imm_command_buffer: vk.CommandBuffer,
    imm_command_pool: vk.CommandPool,

    triangle_pipeline_layout: vk.PipelineLayout,
    triangle_pipeline: vk.Pipeline,

    pub fn draw(self: *VulkanEngine) !void {
        const local = struct {
            fn drawBackground(engine: *VulkanEngine, cmd: vk.CommandBuffer) void {
                // bind the gradient drawing compute pipeline
                engine.device.cmdBindPipeline(cmd, .compute, engine.background_effects[engine.active_background_effect].pipeline);

                // bind the descriptor set containing the draw image for the compute pipeline
                engine.device.cmdBindDescriptorSets(
                    cmd,
                    .compute,
                    engine.background_effects[engine.active_background_effect].layout,
                    0,
                    1,
                    (&engine.draw_image_descriptors)[0..1],
                    0,
                    null,
                );

                std.debug.print("", .{});

                engine.device.cmdPushConstants(
                    cmd,
                    engine.background_effects[engine.active_background_effect].layout,
                    .{ .compute_bit = true },
                    0,
                    @intCast(engine.background_effects[engine.active_background_effect].data.size()),
                    @ptrCast(engine.background_effects[engine.active_background_effect].data.payloadPtr()),
                );

                // execute the compute pipeline dispatch. We are using 16x16 workgroup size so we need to divide by it
                engine.device.cmdDispatch(
                    cmd,
                    std.math.divCeil(u32, engine.draw_extent.width, 16) catch unreachable,
                    std.math.divCeil(u32, engine.draw_extent.height, 16) catch unreachable,
                    1,
                );
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

            vk_image.transitionImage(self.device, cmd, self.draw_image.image, .general, .color_attachment_optimal);

            self.draw_geometry(cmd);

            //transtion the draw image and the swapchain image into their correct transfer layouts
            vk_image.transitionImage(self.device, cmd, self.draw_image.image, .color_attachment_optimal, .transfer_src_optimal);

            vk_image.transitionImage(self.device, cmd, self.swapchain.images[swapchain_image_index], .undefined, .transfer_dst_optimal);

            // execute a copy from the draw image into the swapchain
            vk_image.copyImageToImage(self.device, cmd, self.draw_image.image, self.swapchain.images[swapchain_image_index], self.draw_extent, self.swapchain.extent);

            // set swapchain image layout to Attachment Optimal so we can draw it
            vk_image.transitionImage(self.device, cmd, self.swapchain.images[swapchain_image_index], .transfer_dst_optimal, .color_attachment_optimal);

            //draw imgui into the swapchain image
            self.drawImgui(cmd, self.swapchain.image_views[swapchain_image_index]);

            // set swapchain image layout to Present so we can show it on the screen
            vk_image.transitionImage(self.device, cmd, self.swapchain.images[swapchain_image_index], .color_attachment_optimal, .present_src_khr);
        }
        //finalize the command buffer (we can no longer add commands, but it can now be executed)
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

    fn immediateModeBegin(self: *VulkanEngine) vk.CommandBuffer {
        try self.device.resetFences(1, &self.imm_fence);
        try vk.ResetCommandBuffer(self.imm_command_buffer, 0);

        try self.device.beginCommandBuffer(self.imm_command_buffer, &.{ .flags = .{ .one_time_submit_bit = true } });
    }

    fn immediateModeEnd(self: *VulkanEngine) vk.CommandBuffer {
        try self.device.endCommandBuffer(self.imm_command_buffer);

        const cmdinfo: vk.CommandBufferSubmitInfo = vk_init.commandBufferSubmitInfo(self.imm_command_buffer);
        const submit: vk.SubmitInfo2 = vk_init.submitInfo(&cmdinfo, null, null);

        // submit command buffer to the queue and execute it.
        //  _renderFence will now block until the graphic commands finish execution
        try self.device.queueSubmit2(self.graphics_queue, 1, &submit, self.imm_fence);
        try self.device.waitForFences(1, &self.imm_fence, true, 9999999999);
    }

    pub inline fn currentFrame(self: *VulkanEngine) *FrameData {
        return &self.frames[self.frame_number % FrameData.FRAME_OVERLAP];
    }

    pub fn init(allocator: Allocator) !VulkanEngine {
        const tracy_init_engine = tracy.zoneEx(@src(), .{ .name = "init engine" });
        defer tracy_init_engine.end();

        const tracy_SDL_Init = tracy.zoneEx(@src(), .{ .name = "SDL_Init" });
        if (!c.SDL_Init(c.SDL_INIT_VIDEO)) return error.engine_init_failure;
        tracy_SDL_Init.end();

        var init_arena: std.heap.ArenaAllocator = .init(allocator);
        errdefer init_arena.deinit();
        const init_alloc = init_arena.allocator();

        var temp_arena: std.heap.ArenaAllocator = .init(allocator);
        defer temp_arena.deinit();
        const temp_alloc = temp_arena.allocator();

        const window_width = 1080;
        const window_height = 1080;

        const tracy_SDL_CreateWindow = tracy.zoneEx(@src(), .{ .name = "SDL_CreateWindow" });
        const window = c.SDL_CreateWindow("title", window_width, window_height, c.SDL_WINDOW_VULKAN) orelse return error.engine_init_failure;
        tracy_SDL_CreateWindow.end();

        const tracy_load_base_dispatch = tracy.zoneEx(@src(), .{ .name = "load base_dispatch" });
        const base_dispatch = vk.BaseWrapper.load(@as(vk.PfnGetInstanceProcAddr, @ptrCast(c.SDL_Vulkan_GetVkGetInstanceProcAddr())));
        tracy_load_base_dispatch.end();

        const instance = try vk_init.createVkInstance(base_dispatch, temp_alloc);
        const instance_dispatch = try init_alloc.create(vk.InstanceWrapper);
        {
            const tracy_load_instance_dispatch = tracy.zoneEx(@src(), .{ .name = "load instance_dispatch" });
            instance_dispatch.* = vk.InstanceWrapper.load(instance, base_dispatch.dispatch.vkGetInstanceProcAddr.?);
            tracy_load_instance_dispatch.end();
        }
        const instance_proxy: vk.InstanceProxy = .init(instance, instance_dispatch);

        const tracy_SDL_Vulkan_CreateSurface = tracy.zoneEx(@src(), .{ .name = "SDL_Vulkan_CreateSurface" });
        var surface: vk.SurfaceKHR = undefined;
        if (!c.SDL_Vulkan_CreateSurface(window, @ptrFromInt(@intFromEnum(instance)), null, @ptrCast(&surface))) return error.engine_init_failure;
        tracy_SDL_Vulkan_CreateSurface.end();

        const physical_device = try vk_init.pickPhysicalDevice(instance_proxy, surface, temp_alloc);
        const queue_family_indices = try vk_init.findQueueFamilies(physical_device, instance_dispatch.*, surface, temp_alloc);

        const device = try vk_init.createLogicalDevice(physical_device, instance_dispatch.*, queue_family_indices);
        const device_dispatch = try init_alloc.create(vk.DeviceWrapper);
        {
            const tracy_load_device_dispatch = tracy.zoneEx(@src(), .{ .name = "load device_dispatch" });
            device_dispatch.* = vk.DeviceWrapper.load(device, instance_dispatch.dispatch.vkGetDeviceProcAddr.?);
            tracy_load_device_dispatch.end();
        }
        const device_proxy: vk.DeviceProxy = .init(device, device_dispatch);

        // init_commands() {
        // init_sync_structures() {
        const command_pool_info: vk.CommandPoolCreateInfo = .{
            .flags = .{ .reset_command_buffer_bit = true },
            .queue_family_index = queue_family_indices.graphics_family.?,
        };

        const fence_create_info: vk.FenceCreateInfo = .{ .flags = .{ .signaled_bit = true } };

        var frames: [FrameData.FRAME_OVERLAP]FrameData = undefined;
        for (&frames) |*frame| {
            const command_pool = try device_proxy.createCommandPool(&command_pool_info, null);

            var main_command_buffer: vk.CommandBuffer = undefined;
            const cmd_alloc_info: vk.CommandBufferAllocateInfo = .{
                .command_pool = command_pool,
                .command_buffer_count = 1,
                .level = .primary,
            };
            try device_proxy.allocateCommandBuffers(&cmd_alloc_info, (&main_command_buffer)[0..1]);

            frame.* = .{
                .command_pool = command_pool,
                .render_fence = try device_proxy.createFence(&fence_create_info, null),
                .swapchain_semaphore = try device_proxy.createSemaphore(&.{}, null),
                .render_semaphore = try device_proxy.createSemaphore(&.{}, null),
                .main_command_buffer = main_command_buffer,
                .deletion_queue = .init,
            };
        }

        const imm_command_pool = try device_proxy.createCommandPool(&command_pool_info, null);

        const cmd_alloc_info: vk.CommandBufferAllocateInfo = .{
            .command_pool = imm_command_pool,
            .command_buffer_count = 1,
            .level = .primary,
        };

        var imm_command_buffer: vk.CommandBuffer = undefined;
        try device_proxy.allocateCommandBuffers(&cmd_alloc_info, (&imm_command_buffer)[0..1]);

        var main_deletion_queue: DeletionQueue = .init;
        try main_deletion_queue.append(allocator, .{ .command_pool = imm_command_pool });

        const imm_fence = try device_proxy.createFence(&fence_create_info, null);
        try main_deletion_queue.append(allocator, .{ .fence = imm_fence });
        // }}

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

        try main_deletion_queue.append(allocator, .{ .vma_allocator = vma_allocator });

        errdefer main_deletion_queue.deinit(allocator, .{
            .device = device_proxy,
            .vma_allocator = vma_allocator,
        }); // TODO: find a way to move this next to the main_deletion_queue's init

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
            const rview_info: vk.ImageViewCreateInfo = vk_init.imageViewCreateInfo(
                draw_image_format,
                draw_image,
                .{ .color_bit = true },
            );

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
            try main_deletion_queue.append(allocator, .{ .image_view = draw_allocated_image.image_view });
            try main_deletion_queue.append(allocator, .{ .vma_allocated_image = .{
                .image = draw_allocated_image.image,
                .allocation = draw_allocated_image.allocation,
            } });

            break :blk draw_allocated_image;
        };

        //create a descriptor pool that will hold 10 sets with 1 image each
        const sizes: []const DescriptorAllocator.PoolSizeRatio = &.{.{ .type = .storage_image, .ratio = 1 }};
        const global_descriptor_allocator: DescriptorAllocator = try .initPool(temp_alloc, device_proxy, 10, sizes);

        //make the descriptor set layout for our compute draw
        var bindings = [_]vk.DescriptorSetLayoutBinding{
            vk_descriptors.layout.createSetBinding(0, .storage_image),
        };
        const draw_image_descriptor_layout = try vk_descriptors.layout.createSet(&bindings, device_proxy, .{ .compute_bit = true }, null, .{});

        const draw_image_descriptors = try global_descriptor_allocator.allocate(device_proxy, draw_image_descriptor_layout);

        const img_info: vk.DescriptorImageInfo = .{
            .image_layout = .general,
            .image_view = draw_allocated_image.image_view,

            .sampler = .null_handle,
        };

        const draw_image_write: vk.WriteDescriptorSet = .{
            .dst_binding = 0,
            .dst_set = draw_image_descriptors,
            .descriptor_count = 1,
            .descriptor_type = .storage_image,
            .p_image_info = (&img_info)[0..1],
            .dst_array_element = 0,
            // image descriptor, not a buffer or texel buffer: (TODO: is undefined correct here?)
            .p_buffer_info = undefined,
            .p_texel_buffer_view = undefined,
        };

        device_proxy.updateDescriptorSets(1, (&draw_image_write)[0..1], 0, null);

        try main_deletion_queue.append(allocator, .{ .descriptor_set_layout = draw_image_descriptor_layout });
        try main_deletion_queue.append(allocator, .{ .descriptor_allocator = global_descriptor_allocator });

        // backround effects init ------------
        const effect_infos = [_]struct { path: []const u8, default_data: ComputeEffect.ComputeData, name: [:0]const u8 }{
            .{ .path = shaders.color, .default_data = .{ .color = .{ 1, 0, 0, 1 } }, .name = "color" },
            .{ .path = shaders.circle, .default_data = .{ .circle_effect = .{} }, .name = "circle" },
        };

        const background_effects = try init_alloc.alloc(ComputeEffect, effect_infos.len);

        for (effect_infos, background_effects) |effect_info, *background_effect| {
            const shader_data = try loadShader(effect_info.path, temp_alloc);

            const pushConstant: vk.PushConstantRange = .{
                .offset = 0,
                .size = @intCast(effect_info.default_data.size()),
                .stage_flags = .{ .compute_bit = true },
            };

            const computeLayout: vk.PipelineLayoutCreateInfo = .{
                .p_set_layouts = (&draw_image_descriptor_layout)[0..1],
                .set_layout_count = 1,
                .p_push_constant_ranges = (&pushConstant)[0..1],
                .push_constant_range_count = 1,
            };

            const gradient_pipeline_layout = try device_proxy.createPipelineLayout(&computeLayout, null);

            const computeDrawShader: vk.ShaderModule = try vk_init.loadShaderModule(shader_data, device_proxy);
            defer device_proxy.destroyShaderModule(computeDrawShader, null);

            const stageinfo: vk.PipelineShaderStageCreateInfo = .{
                .stage = .{ .compute_bit = true },
                .module = computeDrawShader,
                .p_name = "main",
            };

            const computePipelineCreateInfo: vk.ComputePipelineCreateInfo = .{
                .layout = gradient_pipeline_layout,
                .stage = stageinfo,

                .base_pipeline_index = 0,
            };

            var gradient_pipeline: vk.Pipeline = undefined;
            _ = try device_proxy.createComputePipelines(.null_handle, 1, (&computePipelineCreateInfo)[0..1], null, (&gradient_pipeline)[0..1]);

            try main_deletion_queue.append(allocator, .{ .pipeline_layout = gradient_pipeline_layout });
            try main_deletion_queue.append(allocator, .{ .pipeline = gradient_pipeline });

            background_effect.* = ComputeEffect{
                .name = effect_info.name,

                .pipeline = gradient_pipeline,
                .layout = gradient_pipeline_layout,

                .data = effect_info.default_data,
            };
        }

        const graphics_queue = device_proxy.getDeviceQueue(queue_family_indices.graphics_family.?, 0);

        const swapchain = try vk_init.createSwapchain(
            init_alloc,
            temp_alloc,
            physical_device,
            device_proxy,
            surface,
            window_width,
            window_height,
            instance_dispatch.*,
        );

        {
            // TODO: fix gamma corection of imgui:
            // https://tuket.github.io/posts/2022-11-24-imgui-gamma/
            // https://github.com/ocornut/imgui/issues/6611

            // 1: create descriptor pool for IMGUI
            //  the size of the pool is very oversize, but it's copied from imgui demo
            //  itself.
            const pool_sizes = [_]vk.DescriptorPoolSize{
                .{ .type = .sampler, .descriptor_count = 1000 },
                .{ .type = .combined_image_sampler, .descriptor_count = 1000 },
                .{ .type = .sampled_image, .descriptor_count = 1000 },
                .{ .type = .storage_image, .descriptor_count = 1000 },
                .{ .type = .uniform_texel_buffer, .descriptor_count = 1000 },
                .{ .type = .storage_texel_buffer, .descriptor_count = 1000 },
                .{ .type = .uniform_buffer, .descriptor_count = 1000 },
                .{ .type = .storage_buffer, .descriptor_count = 1000 },
                .{ .type = .uniform_buffer_dynamic, .descriptor_count = 1000 },
                .{ .type = .storage_buffer_dynamic, .descriptor_count = 1000 },
                .{ .type = .input_attachment, .descriptor_count = 1000 },
            };

            const pool_info: vk.DescriptorPoolCreateInfo = .{
                .flags = .{ .free_descriptor_set_bit = true },
                .max_sets = 1000,
                .pool_size_count = pool_sizes.len,
                .p_pool_sizes = &pool_sizes,
            };

            const imguiPool = try device_proxy.createDescriptorPool(&pool_info, null);

            const imgui_vk_loader = struct {
                var loader: vk.PfnGetInstanceProcAddr = undefined;
                var instance_: vk.Instance = undefined;

                fn f(name: [*c]const u8, _: ?*anyopaque) callconv(.c) c.PFN_vkVoidFunction {
                    return @ptrCast(loader(
                        instance_,
                        @ptrCast(name),
                    ));
                }
            };
            imgui_vk_loader.loader = base_dispatch.dispatch.vkGetInstanceProcAddr.?;
            imgui_vk_loader.instance_ = instance;

            _ = c.cImGui_ImplVulkan_LoadFunctions(
                @bitCast(vk.makeApiVersion(1, 3, 0, 0)), // TODO: set with global variable
                imgui_vk_loader.f,
            );

            // 2: initialize imgui library

            // this initializes the core structures of imgui
            _ = c.ImGui_CreateContext(null);
            const io: *c.ImGuiIO = c.ImGui_GetIO();
            io.ConfigFlags |= c.ImGuiConfigFlags_NavEnableKeyboard; // Enable Keyboard Controls
            io.ConfigFlags |= c.ImGuiConfigFlags_NavEnableGamepad; // Enable Gamepad Controls

            // this initializes imgui for SDL
            _ = c.cImGui_ImplSDL3_InitForVulkan(window);

            const frag_shader = try loadShader(shaders.imgui_frag, temp_alloc);

            // this initializes imgui for Vulkan
            var init_info: c.ImGui_ImplVulkan_InitInfo = .{
                .Instance = @ptrFromInt(@intFromEnum(instance)),
                .PhysicalDevice = @ptrFromInt(@intFromEnum(physical_device)),
                .Device = @ptrFromInt(@intFromEnum(device)),
                .Queue = @ptrFromInt(@intFromEnum(graphics_queue)),
                .DescriptorPool = @ptrFromInt(@intFromEnum(imguiPool)),
                .MinImageCount = 3,
                .ImageCount = 3,
                .UseDynamicRendering = true,
                //dynamic rendering parameters for imgui to use
                .PipelineRenderingCreateInfo = .{
                    .sType = c.VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
                    .colorAttachmentCount = 1,
                    .pColorAttachmentFormats = @ptrCast(&swapchain.image_format),
                },
                .MSAASamples = c.VK_SAMPLE_COUNT_1_BIT,
                .CustomFragShader = frag_shader.ptr,
                .CustomFragShaderSize = frag_shader.size,
            };

            _ = c.cImGui_ImplVulkan_Init(&init_info);
            _ = c.cImGui_ImplVulkan_CreateFontsTexture();

            try main_deletion_queue.append(allocator, .{ .descriptor_pool = imguiPool });
            try main_deletion_queue.append(allocator, .imgui_impl_vulkan);
        }

        const triangleFragShader = try vk_init.loadShaderModule(try loadShader(shaders.colored_triangle_frag, temp_alloc), device_proxy);
        defer device_proxy.destroyShaderModule(triangleFragShader, null);
        const triangleVertexShader = try vk_init.loadShaderModule(try loadShader(shaders.colored_triangle_vert, temp_alloc), device_proxy);
        defer device_proxy.destroyShaderModule(triangleVertexShader, null);

        //build the pipeline layout that controls the inputs/outputs of the shader
        //we are not using descriptor sets or other systems yet, so no need to use anything other than empty default
        const triangle_pipeline_layout = try device_proxy.createPipelineLayout(&vk_init.pipelineLayoutCreateInfo(), null);

        var pipelineBuilder: PipelineBuilder = .{};

        //use the triangle layout we created
        pipelineBuilder.pipelineLayout = triangle_pipeline_layout;
        //connecting the vertex and pixel shaders to the pipeline
        try pipelineBuilder.setShaders(triangleVertexShader, triangleFragShader, temp_alloc);
        //it will draw triangles
        pipelineBuilder.setInputTopology(.triangle_list);
        //filled triangles
        pipelineBuilder.setPolygonMode(.fill);
        //no backface culling
        pipelineBuilder.setCullMode(.{}, .clockwise);
        //no multisampling
        pipelineBuilder.setMultisamplingNone();
        //no blending
        pipelineBuilder.disableBlending();
        //no depth testing
        pipelineBuilder.disableDepthtest();

        //connect the image format we will draw into, from draw image
        pipelineBuilder.setColorAttachmentFormat(draw_allocated_image.image_format);
        pipelineBuilder.setDepthFormat(.undefined);

        //finally build the pipeline
        const triangle_pipeline = try pipelineBuilder.buildPipeline(device_proxy);

        try main_deletion_queue.append(allocator, .{ .pipeline_layout = triangle_pipeline_layout });
        try main_deletion_queue.append(allocator, .{ .pipeline = triangle_pipeline });

        //clean structures TODO
        // main_deletion_queue.append(allocator, [&]() {
        //     vkDestroyPipelineLayout(_device, _trianglePipelineLayout, nullptr);
        //     vkDestroyPipeline(_device, _trianglePipeline, nullptr);
        // });

        return .{
            .init_arena = init_arena,

            .window = window,
            .base_dispatch = base_dispatch,
            .instance = instance_proxy,
            .device = device_proxy,
            .swapchain = swapchain,
            .debug_messenger = .null_handle,
            .chosen_gpu = physical_device,
            .surface = surface,

            .graphics_queue = graphics_queue,
            .graphics_queue_family = queue_family_indices.graphics_family.?,

            .frame_number = 0,
            .frames = frames,

            .main_deletion_queue = main_deletion_queue,

            .vma_allocator = vma_allocator,

            .draw_image = draw_allocated_image,
            .draw_extent = .{ .width = 0, .height = 0 },

            .globalDescriptorAllocator = global_descriptor_allocator,

            .draw_image_descriptors = draw_image_descriptors,
            .draw_image_descriptor_set_layout = draw_image_descriptor_layout,

            .background_effects = background_effects,
            .active_background_effect = 0,

            .imm_command_buffer = imm_command_buffer,
            .imm_command_pool = imm_command_pool,
            .imm_fence = imm_fence,

            .triangle_pipeline_layout = triangle_pipeline_layout,
            .triangle_pipeline = triangle_pipeline,
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

        c.SDL_DestroyWindow(self.window);
        c.SDL_Quit();

        self.init_arena.deinit();
    }

    fn drawImgui(self: *VulkanEngine, cmd: vk.CommandBuffer, targetImageView: vk.ImageView) void {
        const colorAttachment = vk_init.attachmentInfo(targetImageView, null, .attachment_optimal);
        const renderInfo = vk_init.renderingInfo(self.swapchain.extent, &colorAttachment, null);

        self.device.cmdBeginRendering(cmd, &renderInfo);

        c.cImGui_ImplVulkan_NewFrame();
        c.cImGui_ImplSDL3_NewFrame();
        c.ImGui_NewFrame();

        c.ImGui_SetNextWindowSize(.{ .x = 300, .y = 200 }, c.ImGuiCond_Once);
        if (c.ImGui_Begin("out", null, 0)) {
            if (self.background_effects.len > 1) {
                if (c.ImGui_BeginCombo("Select an option", self.background_effects[self.active_background_effect].name, 0)) {
                    for (self.background_effects, 0..) |effect, i| {
                        if (c.ImGui_Selectable(effect.name)) self.active_background_effect = @intCast(i);
                        if (self.active_background_effect == i) c.ImGui_SetItemDefaultFocus();
                    }
                    c.ImGui_EndCombo();
                }
            }
            self.background_effects[self.active_background_effect].data.imGuiMenu();
        }
        c.ImGui_End();

        c.ImGui_Render();
        c.cImGui_ImplVulkan_RenderDrawData(c.ImGui_GetDrawData(), @ptrFromInt(@intFromEnum(cmd)));

        self.device.cmdEndRendering(cmd);
    }

    pub fn draw_geometry(self: VulkanEngine, cmd: vk.CommandBuffer) void {
        //begin a render pass  connected to our draw image
        const colorAttachment: vk.RenderingAttachmentInfo = vk_init.attachmentInfo(self.draw_image.image_view, null, .attachment_optimal);

        const renderInfo = vk_init.renderingInfo(self.draw_extent, &colorAttachment, null);
        self.device.cmdBeginRendering(cmd, &renderInfo);

        self.device.cmdBindPipeline(cmd, .graphics, self.triangle_pipeline);

        //set dynamic viewport and scissor
        const viewport: vk.Viewport = .{
            .x = 0,
            .y = 0,
            .width = @floatFromInt(self.draw_extent.width),
            .height = @floatFromInt(self.draw_extent.height),
            .min_depth = 0,
            .max_depth = 1,
        };

        self.device.cmdSetViewport(cmd, 0, 1, (&viewport)[0..1]);

        const scissor: vk.Rect2D = .{
            .offset = .{
                .x = 0,
                .y = 0,
            },
            .extent = .{
                .width = self.draw_extent.width,
                .height = self.draw_extent.height,
            },
        };
        self.device.cmdSetScissor(cmd, 0, 1, (&scissor)[0..1]);

        //launch a draw command to draw 3 vertices
        self.device.cmdDraw(cmd, 3, 1, 0, 0);
        self.device.cmdEndRendering(cmd);
    }
};

const vk_descriptors = struct {
    const layout = struct {
        fn createSetBinding(
            binding: u32,
            descriptor_type: vk.DescriptorType,
        ) vk.DescriptorSetLayoutBinding {
            return .{
                .binding = binding,
                .descriptor_count = 1,
                .descriptor_type = descriptor_type,

                .stage_flags = .{},
            };
        }

        pub fn createSet(
            bindings: []vk.DescriptorSetLayoutBinding,
            device: vk.DeviceProxy,
            shader_stages: vk.ShaderStageFlags,
            p_next: ?*anyopaque,
            flags: vk.DescriptorSetLayoutCreateFlags,
        ) !vk.DescriptorSetLayout {
            for (bindings) |*binding| {
                @as(*u32, @ptrCast(&binding.stage_flags)).* |= @bitCast(shader_stages);
            }

            const info: vk.DescriptorSetLayoutCreateInfo = .{
                .p_next = p_next,

                .p_bindings = bindings.ptr,
                .binding_count = @intCast(bindings.len),
                .flags = flags,
            };

            const set = try device.createDescriptorSetLayout(&info, null);

            return set;
        }
    };
};

const vk_image = struct {
    fn transitionImage(
        device: vk.DeviceProxy,
        cmd: vk.CommandBuffer,
        image: vk.Image,
        current_layout: vk.ImageLayout,
        new_layout: vk.ImageLayout,
    ) void {
        const image_barrier: vk.ImageMemoryBarrier2 = .{
            .src_stage_mask = .{ .all_commands_bit = true },
            .src_access_mask = .{ .memory_write_bit = true },
            .dst_stage_mask = .{ .all_commands_bit = true },
            .dst_access_mask = .{
                .memory_write_bit = true,
                .memory_read_bit = true,
            },
            .old_layout = current_layout,
            .new_layout = new_layout,
            .subresource_range = vk_init.imageSubresourceRange(
                if (new_layout == .depth_attachment_optimal) .{ .depth_bit = true } else .{ .color_bit = true },
            ),
            .image = image,
            .src_queue_family_index = 0,
            .dst_queue_family_index = 0,
        };

        const dep_info: vk.DependencyInfo = .{
            .image_memory_barrier_count = 1,
            .p_image_memory_barriers = (&image_barrier)[0..1],
        };

        device.cmdPipelineBarrier2(cmd, &dep_info);
    }

    fn copyImageToImage(
        device: vk.DeviceProxy,
        cmd: vk.CommandBuffer,
        source: vk.Image,
        destination: vk.Image,
        src_size: vk.Extent2D,
        dst_size: vk.Extent2D,
    ) void {
        const blitRegion: vk.ImageBlit2 = .{
            .src_offsets = .{ .{ .x = 0, .y = 0, .z = 0 }, .{
                .x = @intCast(src_size.width),
                .y = @intCast(src_size.height),
                .z = 1,
            } },
            .dst_offsets = .{ .{ .x = 0, .y = 0, .z = 0 }, .{
                .x = @intCast(dst_size.width),
                .y = @intCast(dst_size.height),
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
    pub fn pipelineLayoutCreateInfo() vk.PipelineLayoutCreateInfo {
        return .{
            //empty defaults
            .flags = .{},
            .set_layout_count = 0,
            .p_set_layouts = null,
            .push_constant_range_count = 0,
            .p_push_constant_ranges = null,
        };
    }

    pub fn pipelineShaderStageCreateInfo(stage: vk.ShaderStageFlags, shaderModule: vk.ShaderModule) vk.PipelineShaderStageCreateInfo {
        return .{
            //shader stage
            .stage = stage,
            //module containing the code for this shader stage
            .module = shaderModule,
            //the entry point of the shader
            .p_name = "main",
        };
    }

    fn renderingInfo(renderExtent: vk.Extent2D, colorAttachment: *const vk.RenderingAttachmentInfo, depthAttachment: ?*const vk.RenderingAttachmentInfo) vk.RenderingInfo {
        const renderInfo: vk.RenderingInfo = .{
            .render_area = vk.Rect2D{ .offset = vk.Offset2D{ .x = 0, .y = 0 }, .extent = renderExtent },
            .layer_count = 1,
            .color_attachment_count = 1,
            .p_color_attachments = colorAttachment[0..1],
            .p_depth_attachment = depthAttachment,
            .p_stencil_attachment = null,

            .view_mask = 0,
        };
        return renderInfo;
    }

    // VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
    fn attachmentInfo(view: vk.ImageView, clear: ?*vk.ClearValue, layout: vk.ImageLayout) vk.RenderingAttachmentInfo {
        const colorAttachment: vk.RenderingAttachmentInfo = .{
            .image_view = view,
            .image_layout = layout,
            .load_op = if (clear) |_| .clear else .load,
            .store_op = .store,
            .clear_value = if (clear) |item| item.* else std.mem.zeroes(vk.ClearValue),

            .resolve_mode = .{},
            .resolve_image_layout = .undefined,
        };

        return colorAttachment;
    }

    pub fn loadShaderModule(shader_data: ShaderData, device: vk.DeviceProxy) !vk.ShaderModule {
        return try device.createShaderModule(&.{
            .code_size = shader_data.size,
            .p_code = shader_data.ptr,
        }, null);
    }

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

    pub fn createVkInstance(base_dispatch: vk.BaseWrapper, temp_alloc: Allocator) !vk.Instance {
        const zone = tracy.zone(@src());
        defer zone.end();

        const appinfo = vk.ApplicationInfo{
            .p_application_name = "Vulkan Tutorial",
            .application_version = @bitCast(vk.makeApiVersion(1, 0, 0, 0)),
            .p_engine_name = "No Engine",
            .engine_version = @bitCast(vk.makeApiVersion(1, 0, 0, 0)),
            .api_version = @bitCast(vk.makeApiVersion(1, 3, 0, 0)),
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

    pub fn checkValidationLayerSupport(temp_alloc: Allocator, base_dispatch: vk.BaseWrapper) !void {
        const zone = tracy.zone(@src());
        defer zone.end();

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

    pub fn pickPhysicalDevice(instance: vk.InstanceProxy, surface: vk.SurfaceKHR, temp_alloc: Allocator) !vk.PhysicalDevice {
        const zone = tracy.zone(@src());
        defer zone.end();

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
        instance_dispatch: vk.InstanceWrapper,
        surface: vk.SurfaceKHR,
        temp_alloc: Allocator,
    ) !bool {
        const zone = tracy.zone(@src());
        defer zone.end();

        const formats = try instance_dispatch.getPhysicalDeviceSurfaceFormatsAllocKHR(physical_device, surface, temp_alloc);
        const present_modes = try instance_dispatch.getPhysicalDeviceSurfacePresentModesAllocKHR(physical_device, surface, temp_alloc);
        return (try findQueueFamilies(physical_device, instance_dispatch, surface, temp_alloc)).graphics_family != null and
            try checkDeviceExtensionSupport(physical_device, instance_dispatch, temp_alloc) and
            formats.len > 0 and
            present_modes.len > 0;
    }

    pub fn findQueueFamilies(
        physical_device: vk.PhysicalDevice,
        instance_dispatch: vk.InstanceWrapper,
        surface: vk.SurfaceKHR,
        temp_alloc: Allocator,
    ) !QueueFamilyIndices {
        const zone = tracy.zone(@src());
        defer zone.end();

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
        instance_dispatch: vk.InstanceWrapper,
        temp_alloc: Allocator,
    ) !bool {
        const zone = tracy.zone(@src());
        defer zone.end();
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
        instance_dispatch: vk.InstanceWrapper,
        queue_family_indices: QueueFamilyIndices,
    ) !vk.Device {
        const zone = tracy.zone(@src());
        defer zone.end();

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

    // TODO: review this function, not convinced that this is done correctly
    fn createSwapchain(
        alloc: std.mem.Allocator,
        temp_alloc: std.mem.Allocator,
        physical_device: vk.PhysicalDevice,
        device: vk.DeviceProxy,
        surface: vk.SurfaceKHR,
        window_width: u32,
        window_height: u32,
        instance_dispatch: vk.InstanceWrapper,
    ) !SwapChain {
        const zone = tracy.zone(@src());
        defer zone.end();

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
            swapchain_image_format = .r8g8b8a8_srgb;
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

fn loadShader(relative_path: []const u8, allocator: Allocator) !ShaderData {
    const file = try std.fs.cwd().openFile(relative_path, .{});
    const data = try file.readToEndAllocOptions(allocator, 1e6, null, @alignOf(u32), null);

    return ShaderData{
        .ptr = @ptrCast(data.ptr),
        .size = data.len,
    };
}

const ComputeEffect = struct {
    name: [:0]const u8,

    pipeline: vk.Pipeline,
    layout: vk.PipelineLayout,

    data: ComputeData,

    const ComputeData = union(enum) {
        color: @Vector(4, f32),
        circle_effect: extern struct {
            background_color: @Vector(4, f32) = .{ 0, 0, 0, 1 },
            time: f32 = 2,
        },

        fn size(self: ComputeData) usize {
            return switch (self) {
                inline else => |data| @sizeOf(@TypeOf(data)),
            };
        }

        fn payloadPtr(self: *ComputeData) *anyopaque {
            return switch (self.*) {
                inline else => |_, tag| return &@field(self.*, @tagName(tag)),
            };
        }

        fn imGuiMenu(self: *ComputeData) void {
            switch (self.*) {
                .color => |*color| {
                    _ = c.ImGui_ColorPicker4("color", @ptrCast(color), 0, @as([*]const f32, &.{ 1, 0, 0, 1 }));
                },
                .circle_effect => |*circle_effect| {
                    _ = c.ImGui_ColorPicker4("background_color", @ptrCast(&circle_effect.background_color), 0, @as([*]const f32, &.{ 0, 0, 0, 1 }));
                    _ = c.ImGui_DragFloatEx("time", &circle_effect.time, 0.01, 2, std.math.floatMax(f32), null, 0);
                },
            }
        }
    };
};
