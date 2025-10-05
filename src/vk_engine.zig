const std = @import("std");
const Allocator = std.mem.Allocator;

const vk = @import("vulkan");
const c = @import("c");
const tracy = @import("tracy");
const shaders = @import("shaders");
const loader = @import("loader.zig");
const zla = @import("zla");
const vec = zla.vec;
const Mat4 = zla.Mat(f32, 4, 4);
const options = @import("options");

// TODO: make those not globals?
const validation_layers = [_][:0]const u8{"VK_LAYER_KHRONOS_validation"};
const required_device_extensions = [_][:0]const u8{vk.extensions.khr_swapchain.name};

const ShaderData = struct {
    ptr: [*]const u32,
    /// in bytes
    size: usize,
};

const AllocatedBuffer = struct {
    buffer: vk.Buffer,
    allocation: c.VmaAllocation,
    info: c.VmaAllocationInfo,

    pub fn create(allocator: c.VmaAllocator, size: usize, usage: vk.BufferUsageFlags, memory_usage: c.VmaMemoryUsage) !AllocatedBuffer {
        const bufferInfo: vk.BufferCreateInfo = .{
            .usage = usage,
            .size = size,

            .sharing_mode = .exclusive,
        };

        const vma_alloc_info: c.VmaAllocationCreateInfo = .{
            .usage = memory_usage,
            .flags = c.VMA_ALLOCATION_CREATE_MAPPED_BIT,
        };

        var newBuffer: AllocatedBuffer = undefined;
        const result: vk.Result = @enumFromInt(c.vmaCreateBuffer(allocator, @ptrCast(&bufferInfo), &vma_alloc_info, @ptrCast(&newBuffer.buffer), &newBuffer.allocation, &newBuffer.info));
        if (result != .success) {
            std.log.err("vma allocation: error {s}\n", .{@tagName(result)});
            return error.vma_allocation_failed;
        }
        return newBuffer;
    }

    pub fn destroy(buffer: AllocatedBuffer, allocator: c.VmaAllocator) void {
        c.vmaDestroyBuffer(allocator, @ptrFromInt(@intFromEnum(buffer.buffer)), buffer.allocation);
    }
};

pub const Vertex = extern struct {
    position: [3]f32,
    uv_x: f32,
    normal: [3]f32,
    uv_y: f32,
    color: [4]f32,
};

pub const GPUSceneData = extern struct {
    view: Mat4,
    proj: Mat4,
    viewproj: Mat4,
    ambientColor: [4]f32,
    sunlightDirection: [4]f32, // w for sun power
    sunlightColor: [4]f32,
};

// holds the resources needed for a mesh
pub const GPUMeshBuffers = struct {
    indexBuffer: AllocatedBuffer,
    vertexBuffer: AllocatedBuffer,
    vertexBufferAddress: vk.DeviceAddress,
};

// push constants for our mesh object draws
pub const GPUDrawPushConstants = struct {
    worldMatrix: Mat4,
    vertexBuffer: vk.DeviceAddress,
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
    shader_stages: std.ArrayListUnmanaged(vk.PipelineShaderStageCreateInfo) = .empty,

    input_assembly: vk.PipelineInputAssemblyStateCreateInfo = std.mem.zeroInit(vk.PipelineInputAssemblyStateCreateInfo, .{}),
    rasterizer: vk.PipelineRasterizationStateCreateInfo = std.mem.zeroInit(vk.PipelineRasterizationStateCreateInfo, .{}),
    color_blend_attachment: vk.PipelineColorBlendAttachmentState = std.mem.zeroInit(vk.PipelineColorBlendAttachmentState, .{}),
    multisampling: vk.PipelineMultisampleStateCreateInfo = std.mem.zeroInit(vk.PipelineMultisampleStateCreateInfo, .{}),
    pipeline_layout: vk.PipelineLayout = .null_handle,
    depth_stencil: vk.PipelineDepthStencilStateCreateInfo = std.mem.zeroInit(vk.PipelineDepthStencilStateCreateInfo, .{}),
    render_info: vk.PipelineRenderingCreateInfo = std.mem.zeroInit(vk.PipelineRenderingCreateInfo, .{}),
    color_attachmentformat: vk.Format = .undefined,

    pub fn buildPipeline(self: PipelineBuilder, device: vk.DeviceProxy) !vk.Pipeline {
        // make viewport state from our stored viewport and scissor.
        // at the moment we wont support multiple viewports or scissors
        const viewport_state: vk.PipelineViewportStateCreateInfo = .{
            .viewport_count = 1,
            .scissor_count = 1,
        };

        // setup dummy color blending. We arent using transparent objects yet
        // the blending is just "no blend", but we do write to the color attachment
        const color_blending: vk.PipelineColorBlendStateCreateInfo = .{
            .logic_op_enable = .false,
            .logic_op = .copy,
            .attachment_count = 1,
            .p_attachments = (&self.color_blend_attachment)[0..1],

            .blend_constants = .{ 0, 0, 0, 0 },
        };

        // completely clear VertexInputStateCreateInfo, as we have no need for it
        const vertex_input_info = vk.PipelineVertexInputStateCreateInfo{};

        const states = [_]vk.DynamicState{ .viewport, .scissor };
        const dynamic_info: vk.PipelineDynamicStateCreateInfo = .{
            .p_dynamic_states = &states,
            .dynamic_state_count = @intCast(states.len),
        };

        // build the actual pipeline
        // we now use all of the info structs we have been writing into into this one
        // to create the pipeline
        const pipeline_info: vk.GraphicsPipelineCreateInfo = .{
            // connect the renderInfo to the pNext extension mechanism
            .p_next = &self.render_info,

            .stage_count = @intCast(self.shader_stages.items.len),
            .p_stages = self.shader_stages.items.ptr,
            .p_vertex_input_state = &vertex_input_info,
            .p_input_assembly_state = &self.input_assembly,
            .p_viewport_state = &viewport_state,
            .p_rasterization_state = &self.rasterizer,
            .p_multisample_state = &self.multisampling,
            .p_color_blend_state = &color_blending,
            .p_depth_stencil_state = &self.depth_stencil,
            .layout = self.pipeline_layout,

            .p_dynamic_state = &dynamic_info,

            .subpass = 0, // undefined
            .base_pipeline_index = 0, // undefined
        };

        // TODO: handle error ???
        // // its easy to error out on create graphics pipeline, so we handle it a bit
        // // better than the common VK_CHECK case
        var new_pipeline: vk.Pipeline = undefined;
        _ = try device.createGraphicsPipelines(.null_handle, 1, (&pipeline_info)[0..1], null, (&new_pipeline)[0..1]);
        return new_pipeline;
    }

    pub fn setShaders(
        self: *PipelineBuilder,
        vertex_shader: vk.ShaderModule,
        fragment_shader: vk.ShaderModule,
        alloc: Allocator,
    ) !void {
        self.shader_stages.clearRetainingCapacity();

        try self.shader_stages.append(alloc, vk_init.pipelineShaderStageCreateInfo(.{ .vertex_bit = true }, vertex_shader));
        try self.shader_stages.append(alloc, vk_init.pipelineShaderStageCreateInfo(.{ .fragment_bit = true }, fragment_shader));
    }

    pub fn setInputTopology(self: *PipelineBuilder, topology: vk.PrimitiveTopology) void {
        self.input_assembly.topology = topology;
        // we are not going to use primitive restart on the entire tutorial so leave
        // it on false
        self.input_assembly.primitive_restart_enable = .false;
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
        self.multisampling.sample_shading_enable = .false;
        // multisampling defaulted to no multisampling (1 sample per pixel)
        self.multisampling.rasterization_samples = .{ .@"1_bit" = true };
        self.multisampling.min_sample_shading = 1;
        self.multisampling.p_sample_mask = null;
        // no alpha to coverage either
        self.multisampling.alpha_to_coverage_enable = .false;
        self.multisampling.alpha_to_one_enable = .false;
    }

    pub fn disableBlending(self: *PipelineBuilder) void {
        // default write mask
        self.color_blend_attachment.color_write_mask = .{ .r_bit = true, .g_bit = true, .b_bit = true, .a_bit = true };
        // no blending
        self.color_blend_attachment.blend_enable = .false;
    }

    pub fn setColorAttachmentFormat(self: *PipelineBuilder, format: vk.Format) void {
        self.color_attachmentformat = format;
        // connect the format to the renderInfo  structure
        self.render_info.color_attachment_count = 1;
        self.render_info.p_color_attachment_formats = (&self.color_attachmentformat)[0..1];
    }

    pub fn setDepthFormat(self: *PipelineBuilder, format: vk.Format) void {
        self.render_info.depth_attachment_format = format;
    }

    pub fn disableDepthtest(self: *PipelineBuilder) void {
        self.depth_stencil.depth_test_enable = .false;
        self.depth_stencil.depth_write_enable = .false;
        self.depth_stencil.depth_compare_op = .never;
        self.depth_stencil.depth_bounds_test_enable = .false;
        self.depth_stencil.stencil_test_enable = .false;
        // self.depthStencil.front = .{};
        // self.depthStencil.back = .{};
        self.depth_stencil.min_depth_bounds = 0;
        self.depth_stencil.max_depth_bounds = 1;
    }

    pub fn enableDepthtest(self: *PipelineBuilder, depthWriteEnable: bool, op: vk.CompareOp) void {
        self.depth_stencil.depth_test_enable = .true;
        self.depth_stencil.depth_write_enable = if (depthWriteEnable) .true else .false;
        self.depth_stencil.depth_compare_op = op;
        self.depth_stencil.depth_bounds_test_enable = .false;
        self.depth_stencil.stencil_test_enable = .false;
        // self.depth_stencil.front = {};
        // self.depth_stencil.back = {};
        self.depth_stencil.min_depth_bounds = 0;
        self.depth_stencil.max_depth_bounds = 1;
    }

    pub fn enableBlendingAdditive(self: *PipelineBuilder) void {
        // TODO: maybe use the same pattern of init all at once for other builder methods?
        self.color_blend_attachment = .{
            .color_write_mask = .{ .r_bit = true, .g_bit = true, .b_bit = true, .a_bit = true },
            .blend_enable = .true,
            .src_color_blend_factor = .src_alpha,
            .dst_color_blend_factor = .one,
            .color_blend_op = .add,
            .src_alpha_blend_factor = .one,
            .dst_alpha_blend_factor = .zero,
            .alpha_blend_op = .add,
        };
    }

    pub fn enableBlendingAlphablend(self: *PipelineBuilder) void {
        self.color_blend_attachment = .{
            .color_write_mask = .{ .r_bit = true, .g_bit = true, .b_bit = true, .a_bit = true },
            .blend_enable = .true,
            .src_color_blend_factor = .src_alpha,
            .dst_color_blend_factor = .one_minus_src_alpha,
            .color_blend_op = .add,
            .src_alpha_blend_factor = .one,
            .dst_alpha_blend_factor = .zero,
            .alpha_blend_op = .add,
        };
    }

    fn deinit(self: *PipelineBuilder, alloc: Allocator) void {
        self.shader_stages.deinit(alloc);
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
        allocated_buffer: AllocatedBuffer,

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
                .allocated_buffer => |item| item.destroy(context.vma_allocator.?),
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

    // TODO: move sapchain creation here

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

    swapchain_semaphore: vk.Semaphore,
    render_semaphore: vk.Semaphore,
    render_fence: vk.Fence,

    command_pool: vk.CommandPool,
    main_command_buffer: vk.CommandBuffer,

    deletion_queue: DeletionQueue,
    frame_descriptors: descriptors.DescriptorAllocatorGrowable,
};

pub const Engine = struct {
    pub const VkContext = struct {
        base_dispatch: vk.BaseWrapper,
        instance: vk.InstanceProxy,

        chosen_gpu: vk.PhysicalDevice, // GPU chosen as the default device
        surface: vk.SurfaceKHR, // Vulkan window surface
        debug_messenger: vk.DebugUtilsMessengerEXT, // Vulkan debug output handle
    };

    pub const DeviceContext = struct {
        device: vk.DeviceProxy,
        graphics_queue: vk.Queue,
        graphics_queue_family: u32,
        vma_allocator: c.VmaAllocator,
    };

    // immediate submit structures
    pub const ImmSubmit = struct {
        fence: vk.Fence,
        command_buffer: vk.CommandBuffer,
        command_pool: vk.CommandPool,
    };

    init_arena: std.heap.ArenaAllocator,

    window: *c.SDL_Window,

    vk_ctx: VkContext,
    device_ctx: DeviceContext,

    swapchain: SwapChain,
    resize_requested: bool,

    frame_number: u64,
    frames: [FrameData.FRAME_OVERLAP]FrameData,

    main_deletion_queue: DeletionQueue,

    //draw resources
    draw_image: AllocatedImage,
    depth_image: AllocatedImage,
    draw_extent: vk.Extent2D,

    draw_image_descriptors: vk.DescriptorSet,
    draw_image_descriptor_set_layout: vk.DescriptorSetLayout,

    scene_data: GPUSceneData,
    gpu_scene_data_descriptor_layout: vk.DescriptorSetLayout,

    global_descriptor_allocator: DescriptorAllocator,

    background_effects: []ComputeEffect,
    active_background_effect: u32,

    imm: ImmSubmit,

    triangle_pipeline_layout: vk.PipelineLayout,
    triangle_pipeline: vk.Pipeline,

    mesh_pipeline_layout: vk.PipelineLayout,
    mesh_pipeline: vk.Pipeline,

    rectangle: GPUMeshBuffers,

    test_meshes: std.ArrayListUnmanaged(loader.MeshAsset),

    pub fn draw(self: *Engine, allocator: Allocator) !void {
        const local = struct {
            fn drawBackground(engine: *Engine, cmd: vk.CommandBuffer) void {
                // bind the gradient drawing compute pipeline
                const device = engine.device_ctx.device;
                device.cmdBindPipeline(cmd, .compute, engine.background_effects[engine.active_background_effect].pipeline);

                // bind the descriptor set containing the draw image for the compute pipeline
                device.cmdBindDescriptorSets(
                    cmd,
                    .compute,
                    engine.background_effects[engine.active_background_effect].layout,
                    0,
                    1,
                    (&engine.draw_image_descriptors)[0..1],
                    0,
                    null,
                );

                device.cmdPushConstants(
                    cmd,
                    engine.background_effects[engine.active_background_effect].layout,
                    .{ .compute_bit = true },
                    0,
                    @intCast(engine.background_effects[engine.active_background_effect].data.size()),
                    @ptrCast(engine.background_effects[engine.active_background_effect].data.payloadPtr()),
                );

                // execute the compute pipeline dispatch. We are using 16x16 workgroup size so we need to divide by it
                device.cmdDispatch(
                    cmd,
                    std.math.divCeil(u32, engine.draw_extent.width, 16) catch unreachable,
                    std.math.divCeil(u32, engine.draw_extent.height, 16) catch unreachable,
                    1,
                );
            }
        };

        var arena = std.heap.ArenaAllocator.init(allocator);
        defer arena.deinit();
        const temp_alloc = arena.allocator();

        const device = self.device_ctx.device;

        _ = try device.waitForFences(1, (&self.currentFrame().render_fence)[0..1], .true, 1e9);

        self.currentFrame().deletion_queue.flush(.{
            .device = device,
            .vma_allocator = self.device_ctx.vma_allocator,
        });

        try self.currentFrame().frame_descriptors.clearPools(allocator, device);

        _ = try device.resetFences(1, (&self.currentFrame().render_fence)[0..1]);

        const acquire_next_image_result = device.acquireNextImageKHR(
            self.swapchain.handle,
            1e9,
            self.currentFrame().swapchain_semaphore,
            .null_handle,
        ) catch |err| switch (err) {
            error.OutOfDateKHR => {
                self.resize_requested = true;
                return;
            },
            else => |e| return e,
        };
        const swapchain_image_index: u32 = acquire_next_image_result.image_index;

        //naming it cmd for shorter writing
        const cmd: vk.CommandBuffer = self.currentFrame().main_command_buffer;

        self.draw_extent.width = self.draw_image.image_extent.width; // TODO: I dont like this, its duplication of state for no good reasons
        self.draw_extent.height = self.draw_image.image_extent.height;

        // now that we are sure that the commands finished executing, we can safely
        // reset the command buffer to begin recording again.
        try device.resetCommandBuffer(cmd, .{});

        // begin the command buffer recording. We will use this command buffer exactly once, so we want to let vulkan know that
        try device.beginCommandBuffer(cmd, &.{ .flags = .{ .one_time_submit_bit = true } });
        {
            // transition our main draw image into general layout so we can write into it
            // we will overwrite it all so we dont care about what was the older layout
            vk_image.transitionImage(device, cmd, self.draw_image.image, .undefined, .general);

            local.drawBackground(self, cmd);

            vk_image.transitionImage(device, cmd, self.draw_image.image, .general, .color_attachment_optimal);
            vk_image.transitionImage(device, cmd, self.depth_image.image, .undefined, .depth_attachment_optimal);
            try self.drawGeometry(allocator, temp_alloc, cmd);

            //transtion the draw image and the swapchain image into their correct transfer layouts
            vk_image.transitionImage(device, cmd, self.draw_image.image, .color_attachment_optimal, .transfer_src_optimal);

            vk_image.transitionImage(device, cmd, self.swapchain.images[swapchain_image_index], .undefined, .transfer_dst_optimal);

            // execute a copy from the draw image into the swapchain
            vk_image.copyImageToImage(device, cmd, self.draw_image.image, self.swapchain.images[swapchain_image_index], self.draw_extent, self.swapchain.extent);

            // set swapchain image layout to Attachment Optimal so we can draw it
            vk_image.transitionImage(device, cmd, self.swapchain.images[swapchain_image_index], .transfer_dst_optimal, .color_attachment_optimal);

            //draw imgui into the swapchain image
            self.drawImgui(cmd, self.swapchain.image_views[swapchain_image_index]);

            // set swapchain image layout to Present so we can show it on the screen
            vk_image.transitionImage(device, cmd, self.swapchain.images[swapchain_image_index], .color_attachment_optimal, .present_src_khr);
        }
        //finalize the command buffer (we can no longer add commands, but it can now be executed)
        try device.endCommandBuffer(cmd);

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
            try device.queueSubmit2(self.device_ctx.graphics_queue, 1, (&submit_info)[0..1], self.currentFrame().render_fence);
        }

        const present_info: vk.PresentInfoKHR = .{
            .p_swapchains = (&self.swapchain.handle)[0..1],
            .swapchain_count = 1,
            .p_wait_semaphores = (&self.currentFrame().render_semaphore)[0..1],
            .wait_semaphore_count = 1,
            .p_image_indices = (&swapchain_image_index)[0..1],
        };
        _ = device.queuePresentKHR(self.device_ctx.graphics_queue, &present_info) catch |err| switch (err) {
            error.OutOfDateKHR => {
                self.resize_requested = true;
            },
            else => |e| return e,
        };

        self.frame_number += 1;
    }

    fn immediateModeBegin(device: vk.DeviceProxy, imm_fence: vk.Fence, imm_command_buffer: vk.CommandBuffer) !void {
        try device.resetFences(1, (&imm_fence)[0..1]);
        try device.resetCommandBuffer(imm_command_buffer, .{});

        try device.beginCommandBuffer(imm_command_buffer, &.{ .flags = .{ .one_time_submit_bit = true } });
    }

    fn immediateModeEnd(device: vk.DeviceProxy, imm_fence: vk.Fence, imm_command_buffer: vk.CommandBuffer, graphics_queue: vk.Queue) !void {
        try device.endCommandBuffer(imm_command_buffer);

        const cmdinfo: vk.CommandBufferSubmitInfo = vk_init.commandBufferSubmitInfo(imm_command_buffer);
        const submit: vk.SubmitInfo2 = vk_init.submitInfo(&cmdinfo, null, null);

        // submit command buffer to the queue and execute it.
        //  _renderFence will now block until the graphic commands finish execution
        try device.queueSubmit2(graphics_queue, 1, (&submit)[0..1], imm_fence);
        _ = try device.waitForFences(1, (&imm_fence)[0..1], .true, 9999999999);
    }

    pub inline fn currentFrame(self: *Engine) *FrameData {
        return &self.frames[self.frame_number % FrameData.FRAME_OVERLAP];
    }

    pub fn init(allocator: Allocator) !Engine {
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
        const window = c.SDL_CreateWindow("title", window_width, window_height, c.SDL_WINDOW_VULKAN | c.SDL_WINDOW_RESIZABLE) orelse return error.engine_init_failure;
        tracy_SDL_CreateWindow.end();

        const tracy_load_base_dispatch = tracy.zoneEx(@src(), .{ .name = "load base_dispatch" });
        const base_dispatch = vk.BaseWrapper.load(@as(vk.PfnGetInstanceProcAddr, @ptrCast(c.SDL_Vulkan_GetVkGetInstanceProcAddr())));
        tracy_load_base_dispatch.end();

        const instance = try vk_init.createVkInstance(base_dispatch, temp_alloc, options.enable_validation_layers);
        const instance_dispatch = try init_alloc.create(vk.InstanceWrapper);
        instance_dispatch.* = vk.InstanceWrapper.load(instance, base_dispatch.dispatch.vkGetInstanceProcAddr.?);
        const instance_proxy: vk.InstanceProxy = .init(instance, instance_dispatch);

        var sdl_window_surface: vk.SurfaceKHR = undefined;
        if (!c.SDL_Vulkan_CreateSurface(window, @ptrFromInt(@intFromEnum(instance)), null, @ptrCast(&sdl_window_surface))) return error.engine_init_failure;

        const physical_device = try vk_init.pickPhysicalDevice(instance_proxy, sdl_window_surface, temp_alloc);
        const queue_family_indices = try vk_init.findQueueFamilies(physical_device, instance_dispatch.*, sdl_window_surface, temp_alloc);

        const device = try vk_init.createLogicalDevice(physical_device, instance_dispatch.*, queue_family_indices);
        const device_dispatch = try init_alloc.create(vk.DeviceWrapper);
        device_dispatch.* = vk.DeviceWrapper.load(device, instance_dispatch.dispatch.vkGetDeviceProcAddr.?);
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

                .frame_descriptors = undefined,
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

        //allocate images {
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
            .image_format = draw_image_format,
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
        try main_deletion_queue.append(allocator, .{ .vma_allocated_image = .{ .image = draw_allocated_image.image, .allocation = draw_allocated_image.allocation } });

        const depthImageUsages: vk.ImageUsageFlags = .{ .depth_stencil_attachment_bit = true };

        const depth_image_format: vk.Format = .d32_sfloat;
        const dimg_info: vk.ImageCreateInfo = vk_init.imageCreateInfo(depth_image_format, depthImageUsages, draw_image_extent);

        var depth_image_allocation: c.VmaAllocation = undefined;
        //allocate and create the image
        var depth_image: vk.Image = undefined;
        _ = c.vmaCreateImage(vma_allocator, @ptrCast(&dimg_info), &rimg_allocinfo, @ptrCast(&depth_image), &depth_image_allocation, null); // TODO: handle error?

        //build a image-view for the draw image to use for rendering
        const dview_info: vk.ImageViewCreateInfo = vk_init.imageViewCreateInfo(depth_image_format, depth_image, .{ .depth_bit = true });

        const depth_allocated_image: AllocatedImage = .{
            .image_format = depth_image_format,
            .image_extent = draw_image_extent,
            .image_view = try device_proxy.createImageView(&dview_info, null),
            .image = depth_image,
            .allocation = depth_image_allocation,
        };

        try main_deletion_queue.append(allocator, .{ .image_view = depth_allocated_image.image_view });
        try main_deletion_queue.append(allocator, .{ .vma_allocated_image = .{ .image = depth_allocated_image.image, .allocation = depth_allocated_image.allocation } });
        //}

        //create a descriptor pool that will hold 10 sets with 1 image each
        const sizes: []const DescriptorAllocator.PoolSizeRatio = &.{.{ .type = .storage_image, .ratio = 1 }};
        const global_descriptor_allocator: DescriptorAllocator = try .initPool(temp_alloc, device_proxy, 10, sizes);

        //make the descriptor set layout for our compute draw
        var image_bindings = [_]vk.DescriptorSetLayoutBinding{
            descriptors.layout.createSetBinding(0, .storage_image),
        };
        const draw_image_descriptor_layout = try descriptors.layout.createSet(&image_bindings, device_proxy, .{ .compute_bit = true }, null, .{});

        const draw_image_descriptors = try global_descriptor_allocator.allocate(device_proxy, draw_image_descriptor_layout);

        // {
        // const img_info: vk.DescriptorImageInfo = .{
        //     .image_layout = .general,
        //     .image_view = draw_allocated_image.image_view,

        //     .sampler = .null_handle,
        // };

        // const draw_image_write: vk.WriteDescriptorSet = .{
        //     .dst_binding = 0,
        //     .dst_set = draw_image_descriptors,
        //     .descriptor_count = 1,
        //     .descriptor_type = .storage_image,
        //     .p_image_info = (&img_info)[0..1],
        //     .dst_array_element = 0,
        //     // image descriptor, not a buffer or texel buffer: (TODO: is undefined correct here?)
        //     .p_buffer_info = undefined,
        //     .p_texel_buffer_view = undefined,
        // };

        // device_proxy.updateDescriptorSets(1, (&draw_image_write)[0..1], 0, null);
        // TODO: above replaced with below, is abstraction good here?
        var writer: descriptors.DescriptorWriter = .{};
        defer writer.deinit(allocator);
        try writer.writeImage(allocator, 0, draw_allocated_image.image_view, .null_handle, .general, .storage_image);

        writer.updateSet(device_proxy, draw_image_descriptors);

        for (&frames) |*frame| {
            const frame_sizes: []const descriptors.DescriptorAllocatorGrowable.PoolSizeRatio = &.{
                .{ .type = .storage_image, .ratio = 3 },
                .{ .type = .storage_buffer, .ratio = 3 },
                .{ .type = .uniform_buffer, .ratio = 3 },
                .{ .type = .combined_image_sampler, .ratio = 4 },
            };

            frame.frame_descriptors = .empty;
            try frame.frame_descriptors.init(allocator, temp_alloc, device_proxy, 1000, frame_sizes);

            // _mainDeletionQueue.push_function([&, i]() {
            //     _frames[i]._frameDescriptors.destroy_pools(_device);
            // });

            // TODO: add to deletion queue
            // main_deletion_queue.append(allocator, .{ .descriptor_pool =  });
        }

        var scene_data_bindings = [_]vk.DescriptorSetLayoutBinding{
            descriptors.layout.createSetBinding(0, .uniform_buffer),
        };
        const gpu_scene_data_descriptor_layout = try descriptors.layout.createSet(&scene_data_bindings, device_proxy, .{ .vertex_bit = true, .fragment_bit = true }, null, .{});

        try main_deletion_queue.append(allocator, .{ .descriptor_set_layout = gpu_scene_data_descriptor_layout });

        try main_deletion_queue.append(allocator, .{ .descriptor_set_layout = draw_image_descriptor_layout });
        try main_deletion_queue.append(allocator, .{ .descriptor_allocator = global_descriptor_allocator });
        // }

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
        const swapchain = try vk_init.createSwapchain(allocator, temp_alloc, physical_device, device_proxy, sdl_window_surface, window_width, window_height, instance_dispatch.*);

        {
            // 1: create descriptor pool for IMGUI
            // the size of the pool is very oversized, but it's copied from imgui demo itself.
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

            const imgui_pool = try device_proxy.createDescriptorPool(&pool_info, null);

            const ImguiVkLoader = struct {
                var instance_proc_addr: vk.PfnGetInstanceProcAddr = undefined;
                var instance_: vk.Instance = undefined;

                fn f(name: [*c]const u8, _: ?*anyopaque) callconv(.c) c.PFN_vkVoidFunction {
                    return @ptrCast(instance_proc_addr(
                        instance_,
                        @ptrCast(name),
                    ));
                }
            };
            ImguiVkLoader.instance_proc_addr = base_dispatch.dispatch.vkGetInstanceProcAddr.?;
            ImguiVkLoader.instance_ = instance;

            _ = c.cImGui_ImplVulkan_LoadFunctions(
                @bitCast(vk.makeApiVersion(1, 3, 0, 0)), // TODO: set with global variable
                ImguiVkLoader.f,
            );

            // 2: initialize imgui library
            _ = c.ImGui_CreateContext(null);
            const io: *c.ImGuiIO = c.ImGui_GetIO();
            io.ConfigFlags |= c.ImGuiConfigFlags_NavEnableKeyboard; // Enable Keyboard Controls
            io.ConfigFlags |= c.ImGuiConfigFlags_NavEnableGamepad; // Enable Gamepad Controls

            _ = c.cImGui_ImplSDL3_InitForVulkan(window);

            const frag_shader = try loadShader(shaders.imgui_frag, temp_alloc);

            var init_info: c.ImGui_ImplVulkan_InitInfo = .{
                .Instance = @ptrFromInt(@intFromEnum(instance)),
                .PhysicalDevice = @ptrFromInt(@intFromEnum(physical_device)),
                .Device = @ptrFromInt(@intFromEnum(device)),
                .Queue = @ptrFromInt(@intFromEnum(graphics_queue)),
                .DescriptorPool = @ptrFromInt(@intFromEnum(imgui_pool)),
                .MinImageCount = 3,
                .ImageCount = 3,
                .UseDynamicRendering = true,
                .PipelineInfoMain = .{
                    //dynamic rendering parameters for imgui to use
                    .PipelineRenderingCreateInfo = .{
                        .sType = c.VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
                        .colorAttachmentCount = 1,
                        .pColorAttachmentFormats = @ptrCast(&swapchain.image_format),
                    },
                    .MSAASamples = c.VK_SAMPLE_COUNT_1_BIT,
                },
                .CustomShaderFragCreateInfo = .{
                    .sType = c.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                    .pCode = frag_shader.ptr,
                    .codeSize = frag_shader.size,
                },
            };

            _ = c.cImGui_ImplVulkan_Init(&init_info);

            try main_deletion_queue.append(allocator, .{ .descriptor_pool = imgui_pool });
            try main_deletion_queue.append(allocator, .imgui_impl_vulkan);
        }

        // init_triangle_pipeline {
        const triangleFragShader = try vk_init.loadShaderModule(try loadShader(shaders.colored_triangle_frag, temp_alloc), device_proxy);
        defer device_proxy.destroyShaderModule(triangleFragShader, null);
        const triangleVertexShader = try vk_init.loadShaderModule(try loadShader(shaders.colored_triangle_vert, temp_alloc), device_proxy);
        defer device_proxy.destroyShaderModule(triangleVertexShader, null);

        //build the pipeline layout that controls the inputs/outputs of the shader
        //we are not using descriptor sets or other systems yet, so no need to use anything other than empty default
        const triangle_pipeline_layout = try device_proxy.createPipelineLayout(&vk_init.pipelineLayoutCreateInfo(), null);

        const triangle_pipeline = blk: {
            var pipelineBuilder: PipelineBuilder = .{};
            pipelineBuilder.pipeline_layout = triangle_pipeline_layout;
            try pipelineBuilder.setShaders(triangleVertexShader, triangleFragShader, temp_alloc);
            pipelineBuilder.setInputTopology(.triangle_list);
            pipelineBuilder.setPolygonMode(.fill);
            pipelineBuilder.setCullMode(.{}, .clockwise);
            pipelineBuilder.setMultisamplingNone();
            pipelineBuilder.disableBlending();
            pipelineBuilder.enableDepthtest(true, .greater_or_equal);
            pipelineBuilder.setColorAttachmentFormat(draw_allocated_image.image_format);
            pipelineBuilder.setDepthFormat(depth_allocated_image.image_format);
            break :blk try pipelineBuilder.buildPipeline(device_proxy);
        };

        try main_deletion_queue.append(allocator, .{ .pipeline_layout = triangle_pipeline_layout });
        try main_deletion_queue.append(allocator, .{ .pipeline = triangle_pipeline });
        // }

        // init_mesh_pipeline {
        const meshFragShader = try vk_init.loadShaderModule(try loadShader(shaders.colored_triangle_frag, temp_alloc), device_proxy);
        defer device_proxy.destroyShaderModule(meshFragShader, null);
        const meshVertexShader = try vk_init.loadShaderModule(try loadShader(shaders.colored_triangle_mesh_vert, temp_alloc), device_proxy);
        defer device_proxy.destroyShaderModule(meshVertexShader, null);

        const bufferRange: vk.PushConstantRange = .{
            .offset = 0,
            .size = @sizeOf(GPUDrawPushConstants),
            .stage_flags = .{ .vertex_bit = true },
        };

        var mesh_pipeline_layout_info: vk.PipelineLayoutCreateInfo = vk_init.pipelineLayoutCreateInfo();
        mesh_pipeline_layout_info.p_push_constant_ranges = (&bufferRange)[0..1];
        mesh_pipeline_layout_info.push_constant_range_count = 1;

        //build the pipeline layout that controls the inputs/outputs of the shader
        //we are not using descriptor sets or other systems yet, so no need to use anything other than empty default
        const mesh_pipeline_layout = try device_proxy.createPipelineLayout(&mesh_pipeline_layout_info, null);

        //finally build the pipeline
        const mesh_pipeline = blk: {
            var pipelineBuilder: PipelineBuilder = .{};

            //use the triangle layout we created
            pipelineBuilder.pipeline_layout = mesh_pipeline_layout;
            //connecting the vertex and pixel shaders to the pipeline
            try pipelineBuilder.setShaders(meshVertexShader, meshFragShader, temp_alloc);
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
            // pipelineBuilder.disableDepthtest();
            pipelineBuilder.enableDepthtest(true, .greater_or_equal);

            //connect the image format we will draw into, from draw image
            pipelineBuilder.setColorAttachmentFormat(draw_allocated_image.image_format);
            pipelineBuilder.setDepthFormat(depth_allocated_image.image_format);
            break :blk try pipelineBuilder.buildPipeline(device_proxy);
        };
        try main_deletion_queue.append(allocator, .{ .pipeline_layout = mesh_pipeline_layout });
        try main_deletion_queue.append(allocator, .{ .pipeline = mesh_pipeline });
        // }

        // init_default_data {
        const rect_vertices: [4]Vertex = .{
            .{ .position = .{ 0.5, -0.5, 0 }, .color = .{ 0, 0, 0, 1 }, .uv_x = 0, .uv_y = 0, .normal = @splat(0) },
            .{ .position = .{ 0.5, 0.5, 0 }, .color = .{ 0.5, 0.5, 0.5, 1 }, .uv_x = 0, .uv_y = 0, .normal = @splat(0) },
            .{ .position = .{ -0.5, -0.5, 0 }, .color = .{ 1, 0, 0, 1 }, .uv_x = 0, .uv_y = 0, .normal = @splat(0) },
            .{ .position = .{ -0.5, 0.5, 0 }, .color = .{ 0, 1, 0, 1 }, .uv_x = 0, .uv_y = 0, .normal = @splat(0) },
        };
        const rect_indices: [6]u32 = .{ 0, 1, 2, 2, 1, 3 };
        const device_ctx: DeviceContext = .{
            .device = device_proxy,
            .graphics_queue = graphics_queue,
            .graphics_queue_family = queue_family_indices.graphics_family.?,
            .vma_allocator = vma_allocator,
        };
        const imm: ImmSubmit = .{
            .command_buffer = imm_command_buffer,
            .command_pool = imm_command_pool,
            .fence = imm_fence,
        };

        const rectangle = try uploadMesh(device_ctx, imm, &rect_indices, &rect_vertices);

        try main_deletion_queue.append(allocator, .{ .allocated_buffer = rectangle.indexBuffer });
        try main_deletion_queue.append(allocator, .{ .allocated_buffer = rectangle.vertexBuffer });

        const testMeshes = try loader.loadGltfMeshes(init_alloc, temp_alloc, device_ctx, imm, options.assets_path ++ "/basicmesh.glb");
        // }

        return .{
            .vk_ctx = .{
                .base_dispatch = base_dispatch,
                .instance = instance_proxy,
                .surface = sdl_window_surface,
                .chosen_gpu = physical_device,
                .debug_messenger = .null_handle,
            },

            .device_ctx = device_ctx,

            .init_arena = init_arena,

            .window = window,

            .swapchain = swapchain,
            .resize_requested = false,

            .frame_number = 0,
            .frames = frames,

            .main_deletion_queue = main_deletion_queue,

            .draw_image = draw_allocated_image,
            .depth_image = depth_allocated_image,
            .draw_extent = .{ .width = 0, .height = 0 },

            .draw_image_descriptors = draw_image_descriptors,
            .draw_image_descriptor_set_layout = draw_image_descriptor_layout,

            .scene_data = .{
                .view = .identity,
                .proj = .identity,
                .viewproj = .identity,
                .ambientColor = @splat(0),
                .sunlightDirection = @splat(0), // w for sun power
                .sunlightColor = @splat(0),
            },
            .gpu_scene_data_descriptor_layout = gpu_scene_data_descriptor_layout,

            .global_descriptor_allocator = global_descriptor_allocator,

            .background_effects = background_effects,
            .active_background_effect = 0,

            .imm = imm,

            .triangle_pipeline_layout = triangle_pipeline_layout,
            .triangle_pipeline = triangle_pipeline,

            .mesh_pipeline_layout = mesh_pipeline_layout,
            .mesh_pipeline = mesh_pipeline,

            .rectangle = rectangle,

            .test_meshes = testMeshes,
        };
    }

    pub fn resizeSwapchain(self: *Engine, alloc: Allocator) !void {
        const device = self.device_ctx.device;
        try device.deviceWaitIdle();

        self.swapchain.deinit(alloc, device);

        var w: i32 = undefined;
        var h: i32 = undefined;
        _ = c.SDL_GetWindowSize(self.window, &w, &h);
        self.swapchain.extent.width = @intCast(w);
        self.swapchain.extent.height = @intCast(h);

        var temp_arena = std.heap.ArenaAllocator.init(alloc);
        defer temp_arena.deinit();

        self.swapchain = try vk_init.createSwapchain(
            alloc,
            temp_arena.allocator(),
            self.vk_ctx.chosen_gpu,
            device,
            self.vk_ctx.surface,
            self.swapchain.extent.width,
            self.swapchain.extent.height,
            self.vk_ctx.instance.wrapper.*,
        );

        self.resize_requested = false;
    }

    pub fn deinit(self: *Engine, allocator: Allocator) void {
        const device = self.device_ctx.device;
        device.deviceWaitIdle() catch @panic(""); // TODO

        for (0..self.frames.len) |i| {
            device.destroyCommandPool(self.frames[i].command_pool, null);

            device.destroyFence(self.frames[i].render_fence, null);
            device.destroySemaphore(self.frames[i].swapchain_semaphore, null);
            device.destroySemaphore(self.frames[i].render_semaphore, null);

            self.frames[i].deletion_queue.deinit(allocator, .{
                .device = device,
                .vma_allocator = self.device_ctx.vma_allocator,
            });

            self.frames[i].frame_descriptors.deinit(allocator, device);
        }

        for (self.test_meshes.items) |mesh| {
            mesh.mesh_buffers.indexBuffer.destroy(self.device_ctx.vma_allocator);
            mesh.mesh_buffers.vertexBuffer.destroy(self.device_ctx.vma_allocator);
        }

        self.main_deletion_queue.deinit(allocator, .{
            .device = device,
            .vma_allocator = self.device_ctx.vma_allocator,
        });

        self.swapchain.deinit(allocator, device);

        device.destroyDevice(null);
        self.vk_ctx.instance.destroySurfaceKHR(self.vk_ctx.surface, null);
        self.vk_ctx.instance.destroyInstance(null);

        c.SDL_DestroyWindow(self.window);
        c.SDL_Quit();

        self.init_arena.deinit();
    }

    fn drawImgui(self: *Engine, cmd: vk.CommandBuffer, target_image_view: vk.ImageView) void {
        const device = self.device_ctx.device;
        const color_attachment = vk_init.attachmentInfo(target_image_view, null, .attachment_optimal);
        const render_info = vk_init.renderingInfo(self.swapchain.extent, &color_attachment, null);

        device.cmdBeginRendering(cmd, &render_info);

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

        device.cmdEndRendering(cmd);
    }

    pub fn drawGeometry(self: *Engine, allocator: Allocator, temp_alloc: Allocator, cmd: vk.CommandBuffer) !void {
        //begin a render pass  connected to our draw image
        const color_attachment: vk.RenderingAttachmentInfo = vk_init.attachmentInfo(self.draw_image.image_view, null, .attachment_optimal);
        const depthAttachment: vk.RenderingAttachmentInfo = vk_init.depthAttachmentInfo(self.depth_image.image_view, .depth_attachment_optimal);

        const render_info = vk_init.renderingInfo(self.draw_extent, &color_attachment, &depthAttachment);
        const device = self.device_ctx.device;

        device.cmdBeginRendering(cmd, &render_info);
        device.cmdBindPipeline(cmd, .graphics, self.triangle_pipeline);

        //set dynamic viewport and scissor
        const viewport: vk.Viewport = .{
            .x = 0,
            .y = 0,
            .width = @floatFromInt(self.draw_extent.width),
            .height = @floatFromInt(self.draw_extent.height),
            .min_depth = 0,
            .max_depth = 1,
        };

        device.cmdSetViewport(cmd, 0, 1, (&viewport)[0..1]);

        const scissor: vk.Rect2D = .{
            .offset = .{ .x = 0, .y = 0 },
            .extent = .{ .width = self.draw_extent.width, .height = self.draw_extent.height },
        };
        device.cmdSetScissor(cmd, 0, 1, (&scissor)[0..1]);

        //launch a draw command to draw 3 vertices
        device.cmdDraw(cmd, 3, 1, 0, 0);

        {
            device.cmdBindPipeline(cmd, .graphics, self.mesh_pipeline);

            var push_constants: GPUDrawPushConstants = .{
                .worldMatrix = .identity,
                .vertexBuffer = self.rectangle.vertexBufferAddress,
            };

            // device.cmdPushConstants(cmd, self.mesh_pipeline_layout, .{ .vertex_bit = true }, 0, @sizeOf(GPUDrawPushConstants), &push_constants);
            // device.cmdBindIndexBuffer(cmd, self.rectangle.indexBuffer.buffer, 0, .uint32);

            // device.cmdDrawIndexed(cmd, 6, 1, 0, 0, 0);

            // draw monkey
            const view = Mat4.identity.translate(.{ 0, 0, -5 });
            var projection: Mat4 = .perspective(70, @as(f32, @floatFromInt(self.draw_extent.width)) / @as(f32, @floatFromInt(self.draw_extent.height)), 10000, 0.1);
            // invert the Y direction on projection matrix so that we are more similar
            // to opengl and gltf axis
            projection.items[1][1] *= -1;
            push_constants.worldMatrix = projection.mul(view);
            push_constants.vertexBuffer = self.test_meshes.items[2].mesh_buffers.vertexBufferAddress;

            device.cmdPushConstants(cmd, self.mesh_pipeline_layout, .{ .vertex_bit = true }, 0, @sizeOf(GPUDrawPushConstants), &push_constants);
            device.cmdBindIndexBuffer(cmd, self.test_meshes.items[2].mesh_buffers.indexBuffer.buffer, 0, .uint32);

            device.cmdDrawIndexed(cmd, self.test_meshes.items[2].surfaces.items[0].count, 1, self.test_meshes.items[2].surfaces.items[0].start_index, 0, 0);
        }

        {
            //allocate a new uniform buffer for the scene data
            const gpuSceneDataBuffer: AllocatedBuffer = try .create(self.device_ctx.vma_allocator, @sizeOf(GPUSceneData), .{ .uniform_buffer_bit = true }, c.VMA_MEMORY_USAGE_CPU_TO_GPU);

            //add it to the deletion queue of this frame so it gets deleted once its been used
            try self.currentFrame().deletion_queue.append(allocator, .{ .allocated_buffer = gpuSceneDataBuffer });

            // //write the buffer
            // GPUSceneData* sceneUniformData = (GPUSceneData*)gpuSceneDataBuffer.allocation->GetMappedData();
            // *sceneUniformData = sceneData;

            var data: *GPUSceneData = undefined;
            _ = c.vmaMapMemory(self.device_ctx.vma_allocator, gpuSceneDataBuffer.allocation, @ptrCast(&data)); // TODO: handle error?
            defer c.vmaUnmapMemory(self.device_ctx.vma_allocator, gpuSceneDataBuffer.allocation);
            data.* = self.scene_data;

            // //create a descriptor set that binds that buffer and update it
            // VkDescriptorSet globalDescriptor = get_current_frame()._frameDescriptors.allocate(_device, _gpuSceneDataDescriptorLayout);

            const globalDescriptor: vk.DescriptorSet = try self.currentFrame().frame_descriptors.allocate(allocator, temp_alloc, device, self.gpu_scene_data_descriptor_layout, null);

            var writer: descriptors.DescriptorWriter = .{};
            try writer.writeBuffer(temp_alloc, 0, gpuSceneDataBuffer.buffer, @sizeOf(GPUSceneData), 0, .uniform_buffer);
            writer.updateSet(device, globalDescriptor);
        }

        device.cmdEndRendering(cmd);
    }

    pub fn uploadMesh(
        device_ctx: Engine.DeviceContext,
        imm: Engine.ImmSubmit,
        indices: []const u32,
        vertices: []const Vertex,
    ) !GPUMeshBuffers {
        const device = device_ctx.device;
        const vma_allocator = device_ctx.vma_allocator;

        const vertexBufferSize: usize = vertices.len * @sizeOf(Vertex);
        const indexBufferSize: usize = indices.len * @sizeOf(u32);

        //create vertex buffer
        const vertexBuffer: AllocatedBuffer = try .create(
            vma_allocator,
            vertexBufferSize,
            .{ .storage_buffer_bit = true, .transfer_dst_bit = true, .shader_device_address_bit = true },
            c.VMA_MEMORY_USAGE_GPU_ONLY,
        );

        //find the adress of the vertex buffer
        const deviceAdressInfo: vk.BufferDeviceAddressInfo = .{ .buffer = vertexBuffer.buffer };
        const vertexBufferAddress = device.getBufferDeviceAddress(&deviceAdressInfo);

        //create index buffer
        const indexBuffer: AllocatedBuffer = try .create(
            vma_allocator,
            indexBufferSize,
            .{ .storage_buffer_bit = true, .transfer_dst_bit = true, .index_buffer_bit = true },
            c.VMA_MEMORY_USAGE_GPU_ONLY,
        );
        const newSurface: GPUMeshBuffers = .{
            .vertexBuffer = vertexBuffer,
            .indexBuffer = indexBuffer,
            .vertexBufferAddress = vertexBufferAddress,
        };

        const staging: AllocatedBuffer = try .create(
            vma_allocator,
            vertexBufferSize + indexBufferSize,
            .{ .transfer_src_bit = true },
            c.VMA_MEMORY_USAGE_CPU_ONLY,
        );
        defer staging.destroy(vma_allocator);

        var data: [*]u8 = undefined;
        _ = c.vmaMapMemory(vma_allocator, staging.allocation, @ptrCast(&data)); // TODO: handle error?
        defer c.vmaUnmapMemory(vma_allocator, staging.allocation);

        @memcpy(@as([*]Vertex, @ptrCast(@alignCast(data))), vertices); // copy vertex buffer
        @memcpy(@as([*]u32, @ptrCast(@alignCast(data[vertexBufferSize..]))), indices); // copy index buffer

        {
            try immediateModeBegin(device, imm.fence, imm.command_buffer);

            const vertexCopy: vk.BufferCopy = .{
                .dst_offset = 0,
                .src_offset = 0,
                .size = vertexBufferSize,
            };
            device.cmdCopyBuffer(imm.command_buffer, staging.buffer, newSurface.vertexBuffer.buffer, 1, (&vertexCopy)[0..1]);

            const indexCopy: vk.BufferCopy = .{
                .dst_offset = 0,
                .src_offset = vertexBufferSize,
                .size = indexBufferSize,
            };
            device.cmdCopyBuffer(imm.command_buffer, staging.buffer, newSurface.indexBuffer.buffer, 1, (&indexCopy)[0..1]);

            try immediateModeEnd(device, imm.fence, imm.command_buffer, device_ctx.graphics_queue);
        }

        return newSurface;
    }
};

const descriptors = struct {
    const DescriptorAllocatorGrowable = struct {
        // TODO: the design of this is pretty bad
        const PoolSizeRatio = struct {
            type: vk.DescriptorType,
            ratio: f32,
        };

        ratios: std.ArrayListUnmanaged(PoolSizeRatio),
        fullPools: std.ArrayListUnmanaged(vk.DescriptorPool),
        readyPools: std.ArrayListUnmanaged(vk.DescriptorPool),
        setsPerPool: u32,

        pub const empty: DescriptorAllocatorGrowable = .{
            .ratios = .empty,
            .fullPools = .empty,
            .readyPools = .empty,
            .setsPerPool = 0,
        };

        pub fn init(
            self: *DescriptorAllocatorGrowable,
            allocator: Allocator,
            temp_alloc: Allocator,
            device: vk.DeviceProxy,
            max_sets: u32,
            pool_ratios: []const PoolSizeRatio,
        ) !void {
            self.ratios.clearRetainingCapacity();
            try self.ratios.appendSlice(allocator, pool_ratios);

            self.setsPerPool = @intFromFloat(@as(f64, @floatFromInt(max_sets)) * 1.5); //grow it next allocation

            const new_pool = try createPool(temp_alloc, device, max_sets, pool_ratios);
            try self.readyPools.append(allocator, new_pool);
        }

        fn deinit(
            self: *DescriptorAllocatorGrowable,
            allocator: Allocator,
            device: vk.DeviceProxy,
        ) void {
            self.destroyPools(device);
            self.ratios.deinit(allocator);
            self.fullPools.deinit(allocator);
            self.readyPools.deinit(allocator);
        }

        pub fn clearPools(self: *DescriptorAllocatorGrowable, allocator: Allocator, device: vk.DeviceProxy) !void {
            for (self.readyPools.items) |pool| {
                try device.resetDescriptorPool(pool, .{});
            }

            for (self.fullPools.items) |pool| {
                try device.resetDescriptorPool(pool, .{});
                try self.readyPools.append(allocator, pool);
            }
            self.fullPools.clearRetainingCapacity();
        }

        pub fn destroyPools(self: *DescriptorAllocatorGrowable, device: vk.DeviceProxy) void {
            for (self.readyPools.items) |pool| {
                device.destroyDescriptorPool(pool, null);
            }
            self.readyPools.clearRetainingCapacity();

            for (self.fullPools.items) |pool| {
                device.destroyDescriptorPool(pool, null);
            }
            self.fullPools.clearRetainingCapacity();
        }

        pub fn allocate(self: *DescriptorAllocatorGrowable, allocator: Allocator, temp_alloc: Allocator, device: vk.DeviceProxy, layout_: vk.DescriptorSetLayout, pNext: ?*anyopaque) !vk.DescriptorSet {
            //get or create a pool to allocate from
            var poolToUse = try self.getPool(temp_alloc, device);

            var allocInfo: vk.DescriptorSetAllocateInfo = .{
                .p_next = pNext,
                .descriptor_pool = poolToUse,
                .descriptor_set_count = 1,
                .p_set_layouts = (&layout_)[0..1],
            };

            var result: vk.DescriptorSet = undefined;
            device.allocateDescriptorSets(&allocInfo, (&result)[0..1]) catch |err| switch (err) {
                error.OutOfPoolMemory, error.FragmentedPool => { //allocation failed. Try again
                    try self.fullPools.append(allocator, poolToUse);

                    poolToUse = try self.getPool(temp_alloc, device);
                    allocInfo.descriptor_pool = poolToUse;

                    try device.allocateDescriptorSets(&allocInfo, (&result)[0..1]);
                },
                error.OutOfHostMemory, error.OutOfDeviceMemory, error.Unknown => |e| return e,
            };

            try self.readyPools.append(allocator, poolToUse);
            return result;
        }

        pub fn getPool(self: *DescriptorAllocatorGrowable, temp_alloc: Allocator, device: vk.DeviceProxy) !vk.DescriptorPool {
            if (self.readyPools.items.len != 0) {
                return self.readyPools.pop().?;
            } else {
                self.setsPerPool = @intFromFloat(@as(f64, @floatFromInt(self.setsPerPool)) * 1.5);
                if (self.setsPerPool > 4092) {
                    self.setsPerPool = 4092;
                }

                return try createPool(temp_alloc, device, self.setsPerPool, self.ratios.items);
            }
        }

        pub fn createPool(temp_alloc: Allocator, device: vk.DeviceProxy, setCount: u32, poolRatios: []const PoolSizeRatio) !vk.DescriptorPool {
            var poolSizes: std.ArrayListUnmanaged(vk.DescriptorPoolSize) = .empty;
            for (poolRatios) |ratio| {
                try poolSizes.append(temp_alloc, .{ .type = ratio.type, .descriptor_count = @as(u32, @intFromFloat(ratio.ratio)) * setCount });
            }

            return try device.createDescriptorPool(&.{
                .flags = .{},
                .max_sets = setCount,
                .pool_size_count = @intCast(poolSizes.items.len),
                .p_pool_sizes = poolSizes.items.ptr,
            }, null);
        }
    };

    const layout = struct {
        pub fn createSetBinding(
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

            return try device.createDescriptorSetLayout(&.{
                .p_next = p_next,

                .p_bindings = bindings.ptr,
                .binding_count = @intCast(bindings.len),
                .flags = flags,
            }, null);
        }
    };

    const DescriptorWriter = struct {
        imageInfos: std.SegmentedList(vk.DescriptorImageInfo, 128) = .{},
        bufferInfos: std.SegmentedList(vk.DescriptorBufferInfo, 128) = .{},
        writes: std.ArrayListUnmanaged(vk.WriteDescriptorSet) = .empty,

        pub fn writeImage(self: *DescriptorWriter, allocator: Allocator, binding: u32, image: vk.ImageView, sampler: vk.Sampler, layout_: vk.ImageLayout, @"type": vk.DescriptorType) !void {
            const info: *vk.DescriptorImageInfo = try self.imageInfos.addOne(allocator);
            info.* = .{ .sampler = sampler, .image_view = image, .image_layout = layout_ };

            try self.writes.append(allocator, .{
                .dst_binding = binding,
                .dst_set = .null_handle, // left empty for now until we need to write it
                .descriptor_count = 1,
                .descriptor_type = @"type",
                .p_image_info = info[0..1],

                .dst_array_element = 0,
                .p_buffer_info = undefined,
                .p_texel_buffer_view = undefined,
            });
        }

        pub fn writeBuffer(self: *DescriptorWriter, allocator: Allocator, binding: u32, buffer: vk.Buffer, size: usize, offset: usize, @"type": vk.DescriptorType) !void {
            const info: *vk.DescriptorBufferInfo = try self.bufferInfos.addOne(allocator);
            info.* = .{ .buffer = buffer, .offset = offset, .range = size };

            try self.writes.append(allocator, .{
                .dst_binding = binding,
                .dst_set = .null_handle, // left empty for now until we need to write it
                .descriptor_count = 1,
                .descriptor_type = @"type",
                .p_buffer_info = info[0..1],

                .dst_array_element = 0,
                .p_image_info = undefined,
                .p_texel_buffer_view = undefined,
            });
        }

        pub fn clear(self: *DescriptorWriter) void {
            self.imageInfos.clearRetainingCapacity();
            self.writes.clearRetainingCapacity();
            self.bufferInfos.clearRetainingCapacity();
        }

        pub fn deinit(self: *DescriptorWriter, allocator: Allocator) void {
            self.imageInfos.deinit(allocator);
            self.writes.deinit(allocator);
            self.bufferInfos.deinit(allocator);
        }

        pub fn updateSet(self: *DescriptorWriter, device: vk.DeviceProxy, set: vk.DescriptorSet) void {
            for (self.writes.items) |*write| write.dst_set = set;
            device.updateDescriptorSets(@intCast(self.writes.items.len), self.writes.items.ptr, 0, null);
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
            .dst_access_mask = .{ .memory_write_bit = true, .memory_read_bit = true },
            .old_layout = current_layout,
            .new_layout = new_layout,
            .subresource_range = vk_init.imageSubresourceRange(
                if (new_layout == .depth_attachment_optimal) .{ .depth_bit = true } else .{ .color_bit = true },
            ),
            .image = image,
            .src_queue_family_index = 0,
            .dst_queue_family_index = 0,
        };
        device.cmdPipelineBarrier2(cmd, &.{
            .image_memory_barrier_count = 1,
            .p_image_memory_barriers = (&image_barrier)[0..1],
        });
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
            .src_offsets = .{
                .{ .x = 0, .y = 0, .z = 0 },
                .{ .x = @intCast(src_size.width), .y = @intCast(src_size.height), .z = 1 },
            },
            .dst_offsets = .{
                .{ .x = 0, .y = 0, .z = 0 },
                .{ .x = @intCast(dst_size.width), .y = @intCast(dst_size.height), .z = 1 },
            },
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

        device.cmdBlitImage2(cmd, &.{
            .dst_image = destination,
            .dst_image_layout = .transfer_dst_optimal,
            .src_image = source,
            .src_image_layout = .transfer_src_optimal,
            .filter = .linear,
            .region_count = 1,
            .p_regions = (&blitRegion)[0..1],
        });
    }
};

const vk_init = struct {
    pub fn pipelineLayoutCreateInfo() vk.PipelineLayoutCreateInfo {
        return .{ //empty defaults
            .flags = .{},
            .set_layout_count = 0,
            .p_set_layouts = null,
            .push_constant_range_count = 0,
            .p_push_constant_ranges = null,
        };
    }

    pub fn pipelineShaderStageCreateInfo(stage: vk.ShaderStageFlags, shader_module: vk.ShaderModule) vk.PipelineShaderStageCreateInfo {
        return .{
            .stage = stage,
            .module = shader_module,
            .p_name = "main",
        };
    }

    fn renderingInfo(render_extent: vk.Extent2D, color_attachment: *const vk.RenderingAttachmentInfo, depth_attachment: ?*const vk.RenderingAttachmentInfo) vk.RenderingInfo {
        return .{
            .render_area = vk.Rect2D{ .offset = vk.Offset2D{ .x = 0, .y = 0 }, .extent = render_extent },
            .layer_count = 1,
            .color_attachment_count = 1,
            .p_color_attachments = color_attachment[0..1],
            .p_depth_attachment = depth_attachment,
            .p_stencil_attachment = null,

            .view_mask = 0,
        };
    }

    fn attachmentInfo(view: vk.ImageView, clear: ?*vk.ClearValue, layout: vk.ImageLayout) vk.RenderingAttachmentInfo {
        return .{
            .image_view = view,
            .image_layout = layout,
            .load_op = if (clear) |_| .clear else .load,
            .store_op = .store,
            .clear_value = if (clear) |item| item.* else std.mem.zeroes(vk.ClearValue),

            .resolve_mode = .{},
            .resolve_image_layout = .undefined,
        };
    }

    pub fn depthAttachmentInfo(
        view: vk.ImageView,
        layout: ?vk.ImageLayout,
    ) vk.RenderingAttachmentInfo {
        return .{
            .image_view = view,
            .image_layout = layout orelse .color_attachment_optimal,
            .load_op = .clear,
            .store_op = .store,
            .clear_value = .{ .depth_stencil = .{ .depth = 0, .stencil = 0 } },
            .resolve_mode = .{},
            .resolve_image_layout = .undefined,
        };
    }

    pub fn loadShaderModule(shader_data: ShaderData, device: vk.DeviceProxy) !vk.ShaderModule {
        return try device.createShaderModule(&.{ .code_size = shader_data.size, .p_code = shader_data.ptr }, null);
    }

    fn imageCreateInfo(format: vk.Format, usage_flags: vk.ImageUsageFlags, extent: vk.Extent3D) vk.ImageCreateInfo {
        return .{
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
        return .{
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
    }

    fn semaphoreSubmitInfo(stage_mask: vk.PipelineStageFlags2, semaphore: vk.Semaphore) vk.SemaphoreSubmitInfo {
        return .{ .semaphore = semaphore, .stage_mask = stage_mask, .device_index = 0, .value = 1 };
    }

    fn commandBufferSubmitInfo(cmd: vk.CommandBuffer) vk.CommandBufferSubmitInfo {
        return .{ .command_buffer = cmd, .device_mask = 0 };
    }

    fn submitInfo(
        cmd: *const vk.CommandBufferSubmitInfo,
        signal_semaphore_info: ?*const vk.SemaphoreSubmitInfo,
        wait_semaphore_info: ?*const vk.SemaphoreSubmitInfo,
    ) vk.SubmitInfo2 {
        return .{
            .wait_semaphore_info_count = if (wait_semaphore_info == null) 0 else 1,
            .p_wait_semaphore_infos = if (wait_semaphore_info) |info| info[0..1] else null,

            .signal_semaphore_info_count = if (signal_semaphore_info == null) 0 else 1,
            .p_signal_semaphore_infos = if (signal_semaphore_info) |info| info[0..1] else null,

            .command_buffer_info_count = 1,
            .p_command_buffer_infos = cmd[0..1],
        };
    }

    fn imageSubresourceRange(aspect_mask: vk.ImageAspectFlags) vk.ImageSubresourceRange {
        return .{
            .aspect_mask = aspect_mask,
            .base_mip_level = 0,
            .level_count = vk.REMAINING_MIP_LEVELS,
            .base_array_layer = 0,
            .layer_count = vk.REMAINING_ARRAY_LAYERS,
        };
    }

    pub fn createVkInstance(base_dispatch: vk.BaseWrapper, temp_alloc: Allocator, enable_validation_layers: bool) !vk.Instance {
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

        const available_extensions = try base_dispatch.enumerateInstanceExtensionPropertiesAlloc(null, temp_alloc);
        for (sdl_required_extensions) |required_ext| {
            for (available_extensions) |available_ext| {
                if (std.mem.eql(
                    u8,
                    std.mem.span(required_ext),
                    std.mem.span(@as([*:0]const u8, @ptrCast(&available_ext.extension_name))),
                )) break;
            } else {
                return error.extensionRequiredBySdlIsNotAvailable;
            }
        }

        if (enable_validation_layers) try checkValidationLayerSupport(temp_alloc, base_dispatch);

        const create_info = vk.InstanceCreateInfo{
            .p_application_info = &appinfo,
            .enabled_extension_count = @intCast(sdl_required_extensions.len),
            .pp_enabled_extension_names = @ptrCast(sdl_required_extensions.ptr),
            .pp_enabled_layer_names = if (enable_validation_layers) @ptrCast(&validation_layers) else null,
            .enabled_layer_count = if (enable_validation_layers) @intCast(validation_layers.len) else 0,
        };

        return try base_dispatch.createInstance(&create_info, null);
    }

    pub fn checkValidationLayerSupport(temp_alloc: Allocator, base_dispatch: vk.BaseWrapper) !void {
        const zone = tracy.zone(@src());
        defer zone.end();

        const available_layers = try base_dispatch.enumerateInstanceLayerPropertiesAlloc(temp_alloc);
        defer temp_alloc.free(available_layers);

        for (validation_layers) |validation_layer| {
            for (available_layers) |available_layer| {
                if (std.mem.eql(
                    u8,
                    std.mem.span(@as([*:0]const u8, @ptrCast(&available_layer.layer_name))),
                    validation_layer,
                )) break;
            } else {
                return error.NotAllValidationLayersSupported;
            }
        }
    }

    pub fn pickPhysicalDevice(instance: vk.InstanceProxy, surface: vk.SurfaceKHR, temp_alloc: Allocator) !vk.PhysicalDevice {
        const physical_devices = try instance.enumeratePhysicalDevicesAlloc(temp_alloc);

        if (physical_devices.len == 0) return error.NoPhysicalDeviceFound;

        for (physical_devices) |physical_device| {
            const is_suitable = blk: {
                const formats = try instance.getPhysicalDeviceSurfaceFormatsAllocKHR(physical_device, surface, temp_alloc);
                const present_modes = try instance.getPhysicalDeviceSurfacePresentModesAllocKHR(physical_device, surface, temp_alloc);
                break :blk (try findQueueFamilies(physical_device, instance.wrapper.*, surface, temp_alloc)).graphics_family != null and
                    try checkDeviceExtensionSupport(physical_device, instance.wrapper.*, temp_alloc) and
                    formats.len > 0 and
                    present_modes.len > 0;
            };

            if (is_suitable) return physical_device;
        }

        return error.NoSuitablePhysicalDeviceFound;
    }

    pub fn findQueueFamilies(
        physical_device: vk.PhysicalDevice,
        instance_dispatch: vk.InstanceWrapper,
        surface: vk.SurfaceKHR,
        temp_alloc: Allocator,
    ) !QueueFamilyIndices {
        var indices: QueueFamilyIndices = .{ .graphics_family = null, .present_family = null };
        const queue_families = try instance_dispatch.getPhysicalDeviceQueueFamilyPropertiesAlloc(physical_device, temp_alloc);

        // TODO: prefer queue that supports both graphics and KHR
        for (queue_families, 0..) |queue_familie, i| {
            if (queue_familie.queue_flags.graphics_bit) {
                indices.graphics_family = @intCast(i);
                break;
            }
        }

        for (queue_families, 0..) |_, i| {
            if ((try instance_dispatch.getPhysicalDeviceSurfaceSupportKHR(physical_device, @intCast(i), surface) == .true)) {
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

        for (required_device_extensions) |required_device_extension| {
            for (available_extensions) |available_extension| {
                if (std.mem.eql(
                    u8,
                    std.mem.span(@as([*:0]const u8, @ptrCast(&available_extension.extension_name))),
                    required_device_extension,
                )) break;
            } else {
                return false;
            }
        }

        return true;
    }

    pub fn createLogicalDevice(
        physical_device: vk.PhysicalDevice,
        instance_dispatch: vk.InstanceWrapper,
        queue_family_indices: QueueFamilyIndices,
    ) !vk.Device {
        const indices = [_]u32{
            queue_family_indices.graphics_family.?,
            queue_family_indices.present_family.?,
        };

        const queue_priorities: [1]f32 = .{1};

        var queue_create_infos_buff: [indices.len]vk.DeviceQueueCreateInfo = undefined;
        var queue_create_infos: std.ArrayListUnmanaged(vk.DeviceQueueCreateInfo) = .initBuffer(&queue_create_infos_buff);
        outer: for (indices, 0..) |indice, i| {
            for (indices[0..i]) |previous_indice| if (previous_indice == indice) continue :outer;
            queue_create_infos.appendAssumeCapacity(.{
                .queue_family_index = indice,
                .queue_count = queue_priorities.len,
                .p_queue_priorities = &queue_priorities,
            });
        }

        var device_features_vk13: vk.PhysicalDeviceVulkan13Features = .{ .dynamic_rendering = .true, .synchronization_2 = .true };
        return try instance_dispatch.createDevice(physical_device, &.{
            .p_next = (&vk.PhysicalDeviceVulkan12Features{
                .p_next = (&device_features_vk13)[0..1],
                .buffer_device_address = .true,
                .descriptor_indexing = .true,
            })[0..1],
            .p_queue_create_infos = queue_create_infos.items.ptr,
            .queue_create_info_count = @intCast(queue_create_infos.items.len),
            .pp_enabled_extension_names = @ptrCast(&required_device_extensions),
            .enabled_extension_count = required_device_extensions.len,
        }, null);
    }

    // TODO: review this function, not convinced that this is done best
    fn createSwapchain(
        alloc: std.mem.Allocator,
        temp_alloc: std.mem.Allocator,
        physical_device: vk.PhysicalDevice,
        device: vk.DeviceProxy,
        window_surface: vk.SurfaceKHR,
        window_width: u32,
        window_height: u32,
        instance_dispatch: vk.InstanceWrapper,
    ) !SwapChain {
        const surface_formats = try instance_dispatch.getPhysicalDeviceSurfaceFormatsAllocKHR(physical_device, window_surface, temp_alloc);

        const swapchain_image_format = blk: {
            const preferred_format: vk.SurfaceFormatKHR = .{ .format = .b8g8r8a8_srgb, .color_space = .srgb_nonlinear_khr };
            for (surface_formats) |format| if (std.meta.eql(preferred_format, format)) break :blk preferred_format;
            for (surface_formats) |format| if (preferred_format.format == format.format) break :blk format;
            return error.SwapchainCreationFailed;
        };

        const surface_capabilities = try instance_dispatch.getPhysicalDeviceSurfaceCapabilitiesKHR(physical_device, window_surface);
        const min_image_count = @min(surface_capabilities.min_image_count, 3);

        const swapchain_extent: vk.Extent2D = .{ .width = window_width, .height = window_height };

        const swapchain_create_info = vk.SwapchainCreateInfoKHR{
            .surface = window_surface,
            .min_image_count = min_image_count,
            .image_format = swapchain_image_format.format,
            .image_color_space = swapchain_image_format.color_space,
            .image_extent = swapchain_extent,
            .image_array_layers = 1,
            .image_usage = .{ .transfer_src_bit = true, .color_attachment_bit = true, .transfer_dst_bit = true },
            .image_sharing_mode = .exclusive,
            .queue_family_index_count = 0,
            .p_queue_family_indices = null,
            .pre_transform = .{ .identity_bit_khr = true },
            .composite_alpha = .{ .opaque_bit_khr = true },
            .present_mode = .fifo_khr,
            .clipped = .false,
            .old_swapchain = .null_handle,
        };

        const swapchain_handle = try device.createSwapchainKHR(&swapchain_create_info, null);
        errdefer device.destroySwapchainKHR(swapchain_handle, null);

        // Get swapchain images
        const images = try device.getSwapchainImagesAllocKHR(swapchain_handle, alloc);

        // Allocate arrays for images and image views
        var image_views = try alloc.alloc(vk.ImageView, images.len);
        errdefer for (image_views) |view| if (view != .null_handle) device.destroyImageView(view, null);
        errdefer alloc.free(image_views);

        // Create image views for each swapchain image
        for (images, 0..) |image, i| {
            image_views[i] = try device.createImageView(&.{
                .image = image,
                .view_type = .@"2d",
                .format = swapchain_image_format.format,
                .components = .{ .r = .identity, .g = .identity, .b = .identity, .a = .identity },
                .subresource_range = .{
                    .aspect_mask = .{ .color_bit = true },
                    .base_mip_level = 0,
                    .level_count = 1,
                    .base_array_layer = 0,
                    .layer_count = 1,
                },
            }, null);
        }

        return .{
            .handle = swapchain_handle,
            .image_format = swapchain_image_format.format,
            .image_color_space = swapchain_image_format.color_space,
            .images = images,
            .image_views = image_views,
            .extent = swapchain_extent,
        };
    }
};

fn loadShader(relative_path: []const u8, allocator: Allocator) !ShaderData {
    const data = try std.fs.cwd().readFileAllocOptions(allocator, relative_path, 1e6, null, .of(u32), null);
    return .{ .ptr = @ptrCast(data.ptr), .size = data.len };
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
