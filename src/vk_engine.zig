// TODO: make those not globals?
const validation_layers = [_][:0]const u8{"VK_LAYER_KHRONOS_validation"};
const required_device_extensions = [_][*:0]const u8{vk.extensions.khr_swapchain.name};

const Camera = struct {
    velocity: @Vector(3, f32) = @splat(0),
    position: @Vector(3, f32) = @splat(0),
    // vertical rotation
    pitch: f32 = 0,
    // horizontal rotation
    yaw: f32 = 0,

    fn getViewMatrix(self: Camera) Mat4 {
        const camera_translation: Mat4 = .translate(.identity, self.position);
        const camera_rotation: Mat4 = self.getRotationMatrix();
        return camera_translation.mul(camera_rotation).inverse().?;
    }

    fn getRotationMatrix(self: Camera) Mat4 {
        const pitch_rotation: Mat4 = .fromAxisAngle(.{ 1, 0, 0 }, self.pitch);
        const yaw_rotation: Mat4 = .fromAxisAngle(.{ 0, -1, 0 }, self.yaw);
        return yaw_rotation.mul(pitch_rotation);
    }

    pub fn processSDLEvent(self: *Camera, e: *const c.SDL_Event) void {
        if (e.type == c.SDL_EVENT_KEY_DOWN) {
            if (e.key.key == c.SDLK_W) self.velocity[2] = -1;
            if (e.key.key == c.SDLK_S) self.velocity[2] = 1;
            if (e.key.key == c.SDLK_A) self.velocity[0] = -1;
            if (e.key.key == c.SDLK_D) self.velocity[0] = 1;
        }

        if (e.type == c.SDL_EVENT_KEY_UP) {
            if (e.key.key == c.SDLK_W) self.velocity[2] = 0;
            if (e.key.key == c.SDLK_S) self.velocity[2] = 0;
            if (e.key.key == c.SDLK_A) self.velocity[0] = 0;
            if (e.key.key == c.SDLK_D) self.velocity[0] = 0;
        }

        if (e.type == c.SDL_EVENT_MOUSE_MOTION) {
            self.yaw += e.motion.xrel / 200;
            self.pitch -= e.motion.yrel / 200;
            self.pitch = std.math.clamp(self.pitch, -std.math.pi / 2.0, std.math.pi / 2.0);
        }
    }

    fn update(self: *Camera) void {
        const camera_rotation = self.getRotationMatrix();
        var velocity = std.simd.join(self.velocity, @Vector(1, f32){0});
        velocity *= @splat(0.1);
        self.position += std.simd.extract(camera_rotation.mulVec(velocity), 0, 3);
    }
};

pub const scene = struct {
    const RenderObject = struct {
        index_count: u32,
        first_index: u32,
        index_buffer: vk.Buffer,

        material: ?*MaterialInstance,

        transform: Mat4,
        vertex_buffer_address: vk.DeviceAddress,
    };

    const DrawContext = struct {
        opaque_surfaces: std.ArrayList(RenderObject),
        transparent_surfaces: std.ArrayList(RenderObject),
    };

    pub const LoadedGltf = struct {
        // storage for all the data on a given glTF file
        meshes: std.StringHashMapUnmanaged(*loader.MeshAsset),
        nodes: std.StringHashMapUnmanaged(*Node),
        images: std.StringHashMapUnmanaged(AllocatedImage),
        materials: std.StringHashMapUnmanaged(*loader.GltfMaterial),

        // nodes that dont have a parent, for iterating through the file in tree order
        top_nodes: std.ArrayListUnmanaged(*Node),

        samplers: std.ArrayListUnmanaged(vk.Sampler),

        material_data_buffer: VmaGpuBuffer,
    };

    pub const Node = struct {
        parent: ?*Node = null,
        children: std.ArrayList(*Node) = .empty,

        local_transform: Mat4 = .identity,
        world_transform: Mat4 = .identity,

        mesh: ?loader.MeshAsset = null,

        gltf: ?*LoadedGltf = null,

        pub fn refreshTransform(self: *Node, parent_matrix: Mat4) void {
            self.world_transform = parent_matrix.mul(self.local_transform);
            for (self.children.items) |child| child.refreshTransform(self.world_transform);
        }

        pub fn draw(self: Node, gpa: Allocator, top_matrix: Mat4, ctx: *DrawContext) !void {
            if (self.mesh) |mesh| {
                const node_matrix = top_matrix.mul(self.world_transform);

                for (mesh.surfaces.items) |surface| {
                    const def: RenderObject = .{
                        .index_count = surface.count,
                        .first_index = surface.start_index,
                        .index_buffer = mesh.mesh_buffers.index_buffer.buffer,
                        .material = if (surface.material) |material| &material.data else null,

                        .transform = node_matrix,
                        .vertex_buffer_address = mesh.mesh_buffers.vertex_buffer_address,
                    };
                    try ctx.opaque_surfaces.append(gpa, def);
                }
            }

            if (self.gltf) |gltf| {
                for (gltf.top_nodes.items) |node| {
                    try node.draw(gpa, top_matrix, ctx);
                }
            }

            for (self.children.items) |child| try child.draw(gpa, top_matrix, ctx);
        }

        pub fn clearAll(self: *Node, gpa: Allocator, engine: *const Engine) void {
            const device = engine.device_ctx.device;

            for (self.children.items) |child| {
                child.clearAll(gpa, engine);
            }
            self.children.deinit(gpa);

            if (self.gltf) |gltf| {
                gltf.material_data_buffer.destroy(engine.device_ctx.vma_allocator);

                var mesh_it = gltf.meshes.valueIterator();
                while (mesh_it.next()) |mesh| {
                    mesh.*.mesh_buffers.index_buffer.destroy(engine.device_ctx.vma_allocator);
                    mesh.*.mesh_buffers.vertex_buffer.destroy(engine.device_ctx.vma_allocator);
                    mesh.*.surfaces.deinit(gpa);
                    gpa.destroy(mesh.*);
                }

                var image_it = gltf.images.valueIterator();
                while (image_it.next()) |image| {
                    if (image.image == engine.error_checkerboard_image.image) {
                        continue;
                    }

                    engine.destroyImage(image);
                }

                var material_it = gltf.materials.valueIterator();
                while (material_it.next()) |material| {
                    gpa.destroy(material.*);
                }

                for (gltf.samplers.items) |sampler| {
                    device.destroySampler(sampler, null);
                }

                for (gltf.top_nodes.items) |top_node| {
                    top_node.clearAll(gpa, engine);
                }
                gltf.top_nodes.deinit(gpa);

                gltf.images.deinit(gpa);
                gltf.materials.deinit(gpa);
                gltf.meshes.deinit(gpa);
                gltf.samplers.deinit(gpa);
                gltf.nodes.deinit(gpa);

                gpa.destroy(gltf);
            }

            gpa.destroy(self);
        }
    };
};

pub const MaterialPass = enum(u8) {
    main_color,
    transparent,
};

pub const MaterialInstance = struct {
    pipeline: vk.Pipeline,
    bindless_index: u32,
    pass_type: MaterialPass,
};

pub const GltfMetallicRoughness = struct {
    opaque_pipeline: vk.Pipeline,
    transparent_pipeline: vk.Pipeline,

    /// Packed material data that lives in the global materials SSBO.
    /// Shader reads this by materialIndex.
    pub const GPUMaterialData = extern struct {
        color_factors: [4]f32,
        metal_rough_factors: [4]f32,
        color_texture: u32, // index into bindless textures[]
        metal_rough_texture: u32, // index into bindless textures[]
        _padding: [2]u32 = .{ 0, 0 },
    };

    pub const MaterialsBuffer = struct {
        gpu_buffer: VmaGpuBuffer,
        capacity: u32,
        len: u32,

        fn init(max_len: u32, allocator: c.VmaAllocator) !MaterialsBuffer {
            return .{
                .gpu_buffer = try .create(
                    allocator,
                    max_len * @sizeOf(GPUMaterialData),
                    .{ .storage_buffer_bit = true },
                    .auto,
                    .sequential_write,
                ),
                .capacity = max_len,
                .len = 0,
            };
        }

        fn deinit(self: MaterialsBuffer, allocator: c.VmaAllocator) void {
            self.gpu_buffer.destroy(allocator);
        }

        fn append(self: *MaterialsBuffer, item: GPUMaterialData) u32 {
            std.debug.assert(self.len < self.capacity);
            self.gpu_buffer.getMappedSlice(GPUMaterialData)[self.len] = item;
            defer self.len += 1;
            return self.len;
        }
    };

    pub const MaterialResources = struct {
        color_image: AllocatedImage,
        color_sampler: vk.Sampler,
        metal_rough_image: AllocatedImage,
        metal_rough_sampler: vk.Sampler,
        color_factors: [4]f32,
        metal_rough_factors: [4]f32,
    };

    pub fn init(
        scratch: *Scratch,
        io: std.Io,
        bindless_pipeline_layout: vk.PipelineLayout,
        draw_image: AllocatedImage,
        depth_image: AllocatedImage,
        device: vk.DeviceProxy,
    ) !GltfMetallicRoughness {
        const checkpoint = scratch.checkpoint();
        defer scratch.restoreCheckpoint(checkpoint);

        const mesh_frag_shader_data = try loadShader(scratch.allocator(), io, shaders.mesh_frag);
        const mesh_frag_shader = try vk_init.loadShaderModule(mesh_frag_shader_data, device);
        defer device.destroyShaderModule(mesh_frag_shader, null);

        const mesh_vertex_shader_data = try loadShader(scratch.allocator(), io, shaders.mesh_vert);
        const mesh_vertex_shader = try vk_init.loadShaderModule(mesh_vertex_shader_data, device);
        defer device.destroyShaderModule(mesh_vertex_shader, null);

        var pipeline_config: PipelineConfig = .{
            .shaders = .{ mesh_vertex_shader, mesh_frag_shader },
            .depth_test = .{ .write = true, .compare = .greater_or_equal },
            .color_format = draw_image.image_format,
            .depth_format = depth_image.image_format,
        };

        const opaque_pipeline = try createPipeline(device, bindless_pipeline_layout, pipeline_config);

        pipeline_config.blending = .additive;
        pipeline_config.depth_test = .{ .write = false, .compare = .greater_or_equal };

        const transparent_pipeline = try createPipeline(device, bindless_pipeline_layout, pipeline_config);

        return .{
            .opaque_pipeline = opaque_pipeline,
            .transparent_pipeline = transparent_pipeline,
        };
    }

    pub fn deinit(self: *GltfMetallicRoughness, device: vk.DeviceProxy) void {
        device.destroyPipeline(self.transparent_pipeline, null);
        device.destroyPipeline(self.opaque_pipeline, null);
    }

    pub fn writeMaterial(
        self: *GltfMetallicRoughness,
        gpa: Allocator,
        device: vk.DeviceProxy,
        pass: MaterialPass,
        resources: *const MaterialResources,
        bindless: *BindlessDescriptors,
        materials_buffer: *MaterialsBuffer,
    ) !MaterialInstance {
        const color_tex_index = try bindless.registerTexture(
            gpa,
            device,
            resources.color_image.image_view,
            resources.color_sampler,
        );
        const metal_rough_tex_index = try bindless.registerTexture(
            gpa,
            device,
            resources.metal_rough_image.image_view,
            resources.metal_rough_sampler,
        );

        const bindless_index = materials_buffer.append(.{
            .color_factors = resources.color_factors,
            .metal_rough_factors = resources.metal_rough_factors,
            .color_texture = color_tex_index,
            .metal_rough_texture = metal_rough_tex_index,
        });

        return .{
            .pass_type = pass,
            .pipeline = if (pass == .transparent) self.transparent_pipeline else self.opaque_pipeline,
            .bindless_index = bindless_index,
        };
    }
};

const ShaderData = struct {
    ptr: [*]const u32,
    /// in bytes
    size: usize,
};

pub const VmaGpuBuffer = struct {
    buffer: vk.Buffer,
    size: usize,
    allocation: c.VmaAllocation,
    mapped: ?[*]u8,

    pub const MemoryUsage = enum(c_uint) {
        unknown = 0,
        gpu_only = 1,
        cpu_only = 2,
        cpu_to_gpu = 3,
        gpu_to_cpu = 4,
        cpu_copy = 5,
        gpu_lazily_allocated = 6,
        auto = 7,
        auto_prefer_device = 8,
        auto_prefer_host = 9,
    };

    pub const HostAccess = enum(c_uint) {
        /// GPU-only, no host access needed
        none = 0,
        /// CPU writes, GPU reads (uniforms, dynamic vertex data, staging)
        sequential_write = c.VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
        /// CPU readback or random access writes
        random = c.VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT,
    };

    pub fn create(
        allocator: c.VmaAllocator,
        size: usize,
        usage: vk.BufferUsageFlags,
        memory_usage: MemoryUsage,
        host_access: HostAccess,
    ) !VmaGpuBuffer {
        const buffer_info: vk.BufferCreateInfo = .{
            .usage = usage,
            .size = size,
            .sharing_mode = .exclusive,
        };

        const host_flags: c_uint = @intFromEnum(host_access);
        const mapped_flag: c_uint = if (host_access != .none) c.VMA_ALLOCATION_CREATE_MAPPED_BIT else 0;

        const vma_alloc_info: c.VmaAllocationCreateInfo = .{
            .usage = @intFromEnum(memory_usage),
            .flags = host_flags | mapped_flag,
        };

        var info: c.VmaAllocationInfo = undefined;
        var new_buffer: VmaGpuBuffer = undefined;
        new_buffer.size = size;

        const result: vk.Result = @enumFromInt(c.vmaCreateBuffer(
            allocator,
            @ptrCast(&buffer_info),
            &vma_alloc_info,
            @ptrCast(&new_buffer.buffer),
            &new_buffer.allocation,
            &info,
        ));

        if (result != .success) {
            std.log.err("vma allocation: error {s}\n", .{@tagName(result)});
            return error.vma_allocation_failed;
        }

        new_buffer.mapped = @ptrCast(info.pMappedData);
        return new_buffer;
    }

    pub fn destroy(self: VmaGpuBuffer, allocator: c.VmaAllocator) void {
        c.vmaDestroyBuffer(allocator, @ptrFromInt(@intFromEnum(self.buffer)), self.allocation);
    }

    /// Returns the persistently mapped memory as a typed slice.
    pub fn getMappedSlice(self: VmaGpuBuffer, comptime T: type) []T {
        switch (@typeInfo(T)) {
            inline .@"struct", .@"union" => |info| switch (info.layout) {
                .@"extern", .@"packed" => {},
                .auto => @compileError("T must have a well-defined memory layout (extern or packed)"),
            },
            .int, .float => {},
            .pointer => |info| switch (info.size) {
                .one, .many, .c => {},
                .slice => @compileError("T must have a well-defined memory layout"),
            },
            else => @compileError("unsupported type for buffer mapping"),
        }
        std.debug.assert(self.size % @sizeOf(T) == 0);
        const ptr: [*]T = @ptrCast(@alignCast(self.mapped orelse std.debug.panic("getMappedSlice called on unmappable buffer", .{})));
        return ptr[0 .. self.size / @sizeOf(T)];
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
    index_buffer: VmaGpuBuffer,
    vertex_buffer: VmaGpuBuffer,
    vertex_buffer_address: vk.DeviceAddress,
};

// push constants for our mesh object draws
pub const GPUDrawPushConstants = extern struct {
    world_matrix: Mat4,
    vertex_buffer: vk.DeviceAddress,
    material_index: u32,
    scene_data_index: u32,
};

pub const AllocatedImage = struct {
    image: vk.Image,
    image_view: vk.ImageView,
    allocation: c.VmaAllocation,
    image_extent: vk.Extent3D,
    image_format: vk.Format,
};

pub const QueueFamilyIndices = struct {
    graphics_family: u32,
    present_family: u32,
};

const PipelineConfig = struct {
    shaders: struct { vk.ShaderModule, vk.ShaderModule },
    topology: vk.PrimitiveTopology = .triangle_list,
    polygon_mode: vk.PolygonMode = .fill,
    cull_mode: vk.CullModeFlags = .{},
    front_face: vk.FrontFace = .clockwise,
    depth_test: ?struct { write: bool = true, compare: vk.CompareOp = .greater_or_equal } = null,
    color_format: vk.Format,
    depth_format: vk.Format,
    blending: enum { none, additive, alpha } = .none,
};

fn createPipeline(device: vk.DeviceProxy, layout: vk.PipelineLayout, cfg: PipelineConfig) !vk.Pipeline {
    const stages = [_]vk.PipelineShaderStageCreateInfo{
        .{ .stage = .{ .vertex_bit = true }, .module = cfg.shaders[0], .p_name = "main" },
        .{ .stage = .{ .fragment_bit = true }, .module = cfg.shaders[1], .p_name = "main" },
    };

    const color_blend_attachment: vk.PipelineColorBlendAttachmentState = switch (cfg.blending) {
        .none => std.mem.zeroInit(vk.PipelineColorBlendAttachmentState, .{
            .color_write_mask = .{ .r_bit = true, .g_bit = true, .b_bit = true, .a_bit = true },
            .blend_enable = .false,
        }),
        .additive => .{
            .color_write_mask = .{ .r_bit = true, .g_bit = true, .b_bit = true, .a_bit = true },
            .blend_enable = .true,
            .src_color_blend_factor = .src_alpha,
            .dst_color_blend_factor = .one,
            .color_blend_op = .add,
            .src_alpha_blend_factor = .one,
            .dst_alpha_blend_factor = .zero,
            .alpha_blend_op = .add,
        },
        .alpha => .{
            .color_write_mask = .{ .r_bit = true, .g_bit = true, .b_bit = true, .a_bit = true },
            .blend_enable = .true,
            .src_color_blend_factor = .src_alpha,
            .dst_color_blend_factor = .one_minus_src_alpha,
            .color_blend_op = .add,
            .src_alpha_blend_factor = .one,
            .dst_alpha_blend_factor = .zero,
            .alpha_blend_op = .add,
        },
    };

    const dynamic_states = [_]vk.DynamicState{ .viewport, .scissor };

    var new_pipeline: vk.Pipeline = undefined;
    _ = try device.createGraphicsPipelines(
        .null_handle,
        &.{.{
            .p_next = &vk.PipelineRenderingCreateInfo{
                .color_attachment_count = 1,
                .p_color_attachment_formats = &.{cfg.color_format},
                .depth_attachment_format = cfg.depth_format,
                .view_mask = 0,
                .stencil_attachment_format = .undefined,
            },
            .stage_count = stages.len,
            .p_stages = &stages,
            .p_vertex_input_state = &.{},
            .p_input_assembly_state = &.{ .topology = cfg.topology, .primitive_restart_enable = .false },
            .p_viewport_state = &.{ .viewport_count = 1, .scissor_count = 1 },
            .p_rasterization_state = &.{
                .polygon_mode = cfg.polygon_mode,
                .line_width = 1,
                .cull_mode = cfg.cull_mode,
                .front_face = cfg.front_face,
                .depth_clamp_enable = .false,
                .rasterizer_discard_enable = .false,
                .depth_bias_enable = .false,
                .depth_bias_constant_factor = 0,
                .depth_bias_clamp = 0,
                .depth_bias_slope_factor = 0,
            },
            .p_multisample_state = &.{
                .rasterization_samples = .{ .@"1_bit" = true },
                .min_sample_shading = 1,
                .sample_shading_enable = .false,
                .alpha_to_coverage_enable = .false,
                .alpha_to_one_enable = .false,
            },
            .p_color_blend_state = &.{
                .logic_op = .copy,
                .attachment_count = 1,
                .p_attachments = &.{color_blend_attachment},
                .logic_op_enable = .false,
                .blend_constants = @splat(0),
            },
            .p_depth_stencil_state = if (cfg.depth_test) |dt| &.{
                .depth_test_enable = .true,
                .depth_write_enable = if (dt.write) .true else .false,
                .depth_compare_op = dt.compare,
                .max_depth_bounds = 1,
                .depth_bounds_test_enable = .false,
                .stencil_test_enable = .false,
                .front = std.mem.zeroes(vk.StencilOpState),
                .back = std.mem.zeroes(vk.StencilOpState),
                .min_depth_bounds = 0,
            } else &.{
                .depth_compare_op = .never,
                .max_depth_bounds = 1,
                .depth_test_enable = .false,
                .depth_write_enable = .false,
                .depth_bounds_test_enable = .false,
                .stencil_test_enable = .false,
                .front = std.mem.zeroes(vk.StencilOpState),
                .back = std.mem.zeroes(vk.StencilOpState),
                .min_depth_bounds = 0,
            },
            .layout = layout,
            .p_dynamic_state = &.{ .p_dynamic_states = &dynamic_states, .dynamic_state_count = dynamic_states.len },
            .subpass = 0,
            .base_pipeline_index = 0,
        }},
        null,
        (&new_pipeline)[0..1],
    );
    return new_pipeline;
}

const DeletionQueue = struct {
    const DeinitContext = struct {
        device: vk.DeviceProxy,
        vma_allocator: ?c.VmaAllocator,
    };

    const QueueItem = union(enum) {
        vma_allocator: c.VmaAllocator,
        image_view: vk.ImageView,
        vma_allocated_image: AllocatedImage,
        descriptor_allocator: DescriptorAllocator,
        descriptor_set_layout: vk.DescriptorSetLayout,
        pipeline_layout: vk.PipelineLayout,
        pipeline: vk.Pipeline,
        command_pool: vk.CommandPool,
        fence: vk.Fence,
        descriptor_pool: vk.DescriptorPool,
        imgui_impl_vulkan: void,
        allocated_buffer: VmaGpuBuffer,
        sampler: vk.Sampler,

        fn deinit(self: QueueItem, context: DeinitContext) void {
            switch (self) {
                .vma_allocator => |item| c.vmaDestroyAllocator(item),
                .image_view => |item| context.device.destroyImageView(item, null),
                .vma_allocated_image => |item| {
                    c.vmaDestroyImage(context.vma_allocator.?, @ptrFromInt(@intFromEnum(item.image)), item.allocation);
                    context.device.destroyImageView(item.image_view, null);
                },
                .descriptor_allocator => |item| item.destroyPool(context.device),
                .descriptor_set_layout => |item| context.device.destroyDescriptorSetLayout(item, null),
                .pipeline_layout => |item| context.device.destroyPipelineLayout(item, null),
                .pipeline => |item| context.device.destroyPipeline(item, null),
                .command_pool => |item| context.device.destroyCommandPool(item, null),
                .fence => |item| context.device.destroyFence(item, null),
                .descriptor_pool => |item| context.device.destroyDescriptorPool(item, null),
                .imgui_impl_vulkan => c.cImGui_ImplVulkan_Shutdown(),
                .allocated_buffer => |item| item.destroy(context.vma_allocator.?),
                .sampler => |item| context.device.destroySampler(item, null),
            }
        }
    };

    queue: std.ArrayListUnmanaged(QueueItem),

    pub const init: DeletionQueue = .{ .queue = .empty };

    pub fn flush(self: *DeletionQueue, context: DeinitContext) void {
        for (0..self.queue.items.len) |i| self.queue.items[self.queue.items.len - i - 1].deinit(context);
        self.queue.clearRetainingCapacity();
    }

    pub fn deinit(self: *DeletionQueue, gpa: Allocator, context: DeinitContext) void {
        self.flush(context);
        self.queue.deinit(gpa);
    }

    pub fn append(self: *DeletionQueue, gpa: Allocator, item: QueueItem) !void {
        try self.queue.append(gpa, item);
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
            size.* = .{
                .type = ratio.type,
                .descriptor_count = @intFromFloat(ratio.ratio * @as(f32, @floatFromInt(max_sets))),
            };
        }

        const pool_info: vk.DescriptorPoolCreateInfo = .{
            .max_sets = max_sets,
            .pool_size_count = @intCast(pool_sizes.len),
            .p_pool_sizes = pool_sizes.ptr,
        };
        return .{ .pool = try device.createDescriptorPool(&pool_info, null) };
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
    pub const SwapImage = struct {
        handle: vk.Image,
        view: vk.ImageView,
        render_semaphore: vk.Semaphore,
    };

    handle: vk.SwapchainKHR,
    image_format: vk.Format,
    image_color_space: vk.ColorSpaceKHR,
    images: []SwapImage,
    extent: vk.Extent2D,

    // TODO: review this function, not convinced that this is done best
    fn init(
        gpa: std.mem.Allocator,
        scratch: *Scratch,
        physical_device: vk.PhysicalDevice,
        device: vk.DeviceProxy,
        window_surface: vk.SurfaceKHR,
        window_width: u32,
        window_height: u32,
        instance_dispatch: vk.InstanceWrapper,
    ) !SwapChain {
        const checkpoint = scratch.checkpoint();
        defer scratch.restoreCheckpoint(checkpoint);

        const surface_formats = try instance_dispatch.getPhysicalDeviceSurfaceFormatsAllocKHR(physical_device, window_surface, scratch.allocator());

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

        const images = try device.getSwapchainImagesAllocKHR(swapchain_handle, scratch.allocator());

        const swap_images = try gpa.alloc(SwapImage, images.len);
        errdefer {
            gpa.free(swap_images);
            for (swap_images) |image| {
                if (image.view != .null_handle) device.destroyImageView(image.view, null);
            }
        }

        for (images, swap_images) |image, *swapchain_image| {
            swapchain_image.* = .{
                .handle = image,
                .render_semaphore = try device.createSemaphore(&.{}, null),
                .view = try device.createImageView(&.{
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
                }, null),
            };
        }

        return .{
            .handle = swapchain_handle,
            .image_format = swapchain_image_format.format,
            .image_color_space = swapchain_image_format.color_space,
            .images = swap_images,
            .extent = swapchain_extent,
        };
    }

    fn deinit(self: SwapChain, gpa: Allocator, device: vk.DeviceProxy) void {
        device.destroySwapchainKHR(self.handle, null);
        for (self.images) |image| {
            device.destroyImageView(image.view, null);
            device.destroySemaphore(image.render_semaphore, null);
        }
        gpa.free(self.images);
    }
};

const FrameData = struct {
    const frame_overlap = 2;

    swapchain_semaphore: vk.Semaphore,
    render_fence: vk.Fence,

    command_pool: vk.CommandPool,
    main_command_buffer: vk.CommandBuffer,

    deletion_queue: DeletionQueue,
};

pub const Engine = struct {
    pub const VkContext = struct {
        base_dispatch: vk.BaseWrapper,
        instance: vk.InstanceProxy,

        chosen_gpu: vk.PhysicalDevice,
        window_surface: vk.SurfaceKHR,
        debug_messenger: vk.DebugUtilsMessengerEXT,
    };

    pub const DeviceContext = struct {
        device: vk.DeviceProxy,
        graphics_queue: vk.Queue,
        graphics_queue_family: u32,
        vma_allocator: c.VmaAllocator,
    };

    pub const ImmSubmit = struct {
        fence: vk.Fence,
        cmd: vk.CommandBuffer,
        command_pool: vk.CommandPool,
    };

    init_arena: std.heap.ArenaAllocator,

    window: *c.SDL_Window,

    vk_ctx: VkContext,
    device_ctx: DeviceContext,

    swapchain: SwapChain,
    resize_requested: bool,

    frame_number: u64,
    frames: [FrameData.frame_overlap]FrameData,

    main_deletion_queue: DeletionQueue,

    //draw resources
    depth_image: AllocatedImage,
    draw_image: AllocatedImage,

    scene_data: GPUSceneData,

    imm: ImmSubmit,

    bindless_descriptors: BindlessDescriptors,
    bindless_pipeline_layout: vk.PipelineLayout,

    materials_buffer: GltfMetallicRoughness.MaterialsBuffer,
    scene_data_buffer: VmaGpuBuffer,

    // default images
    white_image: AllocatedImage,
    black_image: AllocatedImage,
    grey_image: AllocatedImage,
    error_checkerboard_image: AllocatedImage,

    default_sampler_linear: vk.Sampler,
    default_sampler_nearest: vk.Sampler,

    // default material
    default_data: MaterialInstance,
    metal_rough_material: GltfMetallicRoughness,

    main_camera: Camera,

    main_draw_context: scene.DrawContext,
    loaded_scenes: std.StringHashMapUnmanaged(*scene.Node),

    pub fn draw(self: *Engine, gpa: Allocator, scratch: *Scratch) !void {
        const zone = tracy.zone(@src());
        defer zone.end();

        try self.updateScene(gpa);

        const checkpoint = scratch.checkpoint();
        defer scratch.restoreCheckpoint(checkpoint);

        const device = self.device_ctx.device;

        const wait_fence = tracy.zoneEx(@src(), .{ .name = "wait_fence" });
        _ = try device.waitForFences(&.{self.currentFrame().render_fence}, .true, 1e9);
        wait_fence.end();

        self.currentFrame().deletion_queue.flush(.{ .device = device, .vma_allocator = self.device_ctx.vma_allocator });

        _ = try device.resetFences(&.{self.currentFrame().render_fence});

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
        const swapchain_image_index = acquire_next_image_result.image_index;
        const current_swap_image = self.swapchain.images[swapchain_image_index];

        const cmd: vk.CommandBuffer = self.currentFrame().main_command_buffer;

        // now that we are sure that the commands finished executing, we can safely
        // reset the command buffer to begin recording again.
        try device.resetCommandBuffer(cmd, .{});

        // we will use this command buffer exactly once, so we want to let vulkan know that
        try device.beginCommandBuffer(cmd, &.{ .flags = .{ .one_time_submit_bit = true } });
        {
            // device.cmdClearAttachments(cmd, p_attachments: []const ClearAttachment, p_rects: []const ClearRect)

            vk_image.transitionImage(device, cmd, self.draw_image.image, .undefined, .color_attachment_optimal);
            vk_image.transitionImage(device, cmd, self.depth_image.image, .undefined, .depth_attachment_optimal);
            try self.drawGeometry(scratch, cmd);

            // copy draw image into the swapchain
            vk_image.transitionImage(device, cmd, self.draw_image.image, .color_attachment_optimal, .transfer_src_optimal);
            vk_image.transitionImage(device, cmd, current_swap_image.handle, .undefined, .transfer_dst_optimal);
            vk_image.copyImageToImage(device, cmd, self.draw_image.image, current_swap_image.handle, self.drawExtent(), self.swapchain.extent);

            //draw imgui into the swapchain image
            vk_image.transitionImage(device, cmd, current_swap_image.handle, .transfer_dst_optimal, .color_attachment_optimal);
            self.drawImgui(cmd, current_swap_image.view);
            vk_image.transitionImage(device, cmd, current_swap_image.handle, .color_attachment_optimal, .present_src_khr);
        }
        //finalize the command buffer (we can no longer add commands, but it can now be executed)
        try device.endCommandBuffer(cmd);

        //prepare the submission to the queue.
        //we want to wait on the _presentSemaphore, as that semaphore is signaled when the swapchain is ready
        //we will signal the _renderSemaphore, to signal that rendering has finished
        {
            const cmd_info: vk.CommandBufferSubmitInfo = vk_init.commandBufferSubmitInfo(cmd);
            const wait_info: vk.SemaphoreSubmitInfo = vk_init.semaphoreSubmitInfo(.{ .color_attachment_output_bit = true }, self.currentFrame().swapchain_semaphore);
            const signal_info: vk.SemaphoreSubmitInfo = vk_init.semaphoreSubmitInfo(.{ .all_graphics_bit = true }, current_swap_image.render_semaphore);

            const submit_info = vk_init.submitInfo(&cmd_info, &signal_info, &wait_info);

            //submit command buffer to the queue and execute it.
            // _render_fence will now block until the graphic commands finish execution
            try device.queueSubmit2(self.device_ctx.graphics_queue, &.{submit_info}, self.currentFrame().render_fence);
        }

        const present_info: vk.PresentInfoKHR = .{
            .p_swapchains = (&self.swapchain.handle)[0..1],
            .swapchain_count = 1,
            .p_wait_semaphores = (&current_swap_image.render_semaphore)[0..1],
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

    // TODO: use device ctx and return imm_command_buffer?
    fn immediateModeBegin(device: vk.DeviceProxy, imm_fence: vk.Fence, imm_command_buffer: vk.CommandBuffer) !void {
        try device.resetFences(&.{imm_fence});
        try device.resetCommandBuffer(imm_command_buffer, .{});

        try device.beginCommandBuffer(imm_command_buffer, &.{ .flags = .{ .one_time_submit_bit = true } });
    }

    fn immediateModeEnd(device: vk.DeviceProxy, imm_fence: vk.Fence, imm_command_buffer: vk.CommandBuffer, graphics_queue: vk.Queue) !void {
        try device.endCommandBuffer(imm_command_buffer);

        const cmdinfo: vk.CommandBufferSubmitInfo = vk_init.commandBufferSubmitInfo(imm_command_buffer);
        const submit: vk.SubmitInfo2 = vk_init.submitInfo(&cmdinfo, null, null);

        // submit command buffer to the queue and execute it.
        //  _renderFence will now block until the graphic commands finish execution
        try device.queueSubmit2(graphics_queue, &.{submit}, imm_fence);
        _ = try device.waitForFences(&.{imm_fence}, .true, 9999999999);
    }

    pub inline fn currentFrameIndex(self: *const Engine) usize {
        return self.frame_number % FrameData.frame_overlap;
    }

    pub inline fn currentFrame(self: *Engine) *FrameData {
        return &self.frames[self.currentFrameIndex()];
    }

    pub fn init(gpa: Allocator, scratch: *Scratch, io: std.Io) !Engine {
        const tracy_init_engine = tracy.zoneEx(@src(), .{ .name = "init engine" });
        defer tracy_init_engine.end();

        const tracy_SDL_Init = tracy.zoneEx(@src(), .{ .name = "SDL_Init" });
        if (!c.SDL_Init(c.SDL_INIT_VIDEO)) return error.engine_init_failure;
        tracy_SDL_Init.end();

        var init_arena: std.heap.ArenaAllocator = .init(gpa);
        errdefer init_arena.deinit();
        const init_alloc = init_arena.allocator();

        const window_width = 1080;
        const window_height = 1080;

        const tracy_SDL_CreateWindow = tracy.zoneEx(@src(), .{ .name = "SDL_CreateWindow" });
        const window = c.SDL_CreateWindow("title", window_width, window_height, c.SDL_WINDOW_VULKAN | c.SDL_WINDOW_RESIZABLE) orelse return error.engine_init_failure;
        tracy_SDL_CreateWindow.end();

        const tracy_load_base_dispatch = tracy.zoneEx(@src(), .{ .name = "load base_dispatch" });
        const base_dispatch = vk.BaseWrapper.load(@as(vk.PfnGetInstanceProcAddr, @ptrCast(c.SDL_Vulkan_GetVkGetInstanceProcAddr())));
        tracy_load_base_dispatch.end();

        const instance = try vk_init.createVkInstance(scratch, base_dispatch, options.enable_validation_layers);
        const instance_dispatch = try init_alloc.create(vk.InstanceWrapper);
        instance_dispatch.* = vk.InstanceWrapper.load(instance, base_dispatch.dispatch.vkGetInstanceProcAddr.?);
        const instance_proxy: vk.InstanceProxy = .init(instance, instance_dispatch);

        const debug_callback = struct {
            fn debugCallback(
                message_severity: vk.DebugUtilsMessageSeverityFlagsEXT,
                message_types: vk.DebugUtilsMessageTypeFlagsEXT,
                p_callback_data: ?*const vk.DebugUtilsMessengerCallbackDataEXT,
                p_user_data: ?*anyopaque,
            ) callconv(vk.vulkan_call_conv) vk.Bool32 {
                _ = message_types;
                _ = p_user_data;
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
        const debug_messenger = try instance_proxy.createDebugUtilsMessengerEXT(&debug_messenger_info, null);

        var sdl_window_surface: vk.SurfaceKHR = undefined;
        if (!c.SDL_Vulkan_CreateSurface(window, @ptrFromInt(@intFromEnum(instance)), null, @ptrCast(&sdl_window_surface))) return error.engine_init_failure;

        const physical_device = try vk_init.pickPhysicalDevice(scratch, instance_proxy, sdl_window_surface);
        const queue_family_indices = (try vk_init.findQueueFamilies(scratch, physical_device, instance_dispatch.*, sdl_window_surface)).?;

        const device = try vk_init.createLogicalDevice(physical_device, instance_dispatch.*, queue_family_indices);
        const device_dispatch = try init_alloc.create(vk.DeviceWrapper);
        device_dispatch.* = vk.DeviceWrapper.load(device, instance_dispatch.dispatch.vkGetDeviceProcAddr.?);
        const device_proxy: vk.DeviceProxy = .init(device, device_dispatch);

        // init_commands() {
        // init_sync_structures() {
        const command_pool_info: vk.CommandPoolCreateInfo = .{
            .flags = .{ .reset_command_buffer_bit = true },
            .queue_family_index = queue_family_indices.graphics_family,
        };

        const fence_create_info: vk.FenceCreateInfo = .{ .flags = .{ .signaled_bit = true } };

        var frames: [FrameData.frame_overlap]FrameData = undefined;
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
        try main_deletion_queue.append(gpa, .{ .command_pool = imm_command_pool });

        const imm_fence = try device_proxy.createFence(&fence_create_info, null);
        try main_deletion_queue.append(gpa, .{ .fence = imm_fence });
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

        try main_deletion_queue.append(gpa, .{ .vma_allocator = vma_allocator });

        errdefer main_deletion_queue.deinit(gpa, .{
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
        try main_deletion_queue.append(gpa, .{ .vma_allocated_image = draw_allocated_image });

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

        try main_deletion_queue.append(gpa, .{ .vma_allocated_image = depth_allocated_image });
        //}

        const graphics_queue = device_proxy.getDeviceQueue(queue_family_indices.graphics_family, 0);
        const swapchain: SwapChain = try .init(
            gpa,
            scratch,
            physical_device,
            device_proxy,
            sdl_window_surface,
            window_width,
            window_height,
            instance_dispatch.*,
        );

        {
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
            const imgui_io: *c.ImGuiIO = c.ImGui_GetIO();
            imgui_io.ConfigFlags |= c.ImGuiConfigFlags_NavEnableKeyboard;
            imgui_io.ConfigFlags |= c.ImGuiConfigFlags_NavEnableGamepad;

            _ = c.cImGui_ImplSDL3_InitForVulkan(window);

            const frag_shader = try loadShader(scratch.allocator(), io, shaders.imgui_frag);

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

            try main_deletion_queue.append(gpa, .{ .descriptor_pool = imgui_pool });
            try main_deletion_queue.append(gpa, .imgui_impl_vulkan);
        }

        var bindless_descriptors: BindlessDescriptors = try .init(device_proxy);

        const push_range: vk.PushConstantRange = .{
            .offset = 0,
            .size = @sizeOf(GPUDrawPushConstants),
            .stage_flags = .{ .vertex_bit = true, .fragment_bit = true },
        };

        const bindless_pipeline_layout = try device_proxy.createPipelineLayout(&.{
            .set_layout_count = 1,
            .p_set_layouts = (&bindless_descriptors.layout)[0..1],
            .push_constant_range_count = 1,
            .p_push_constant_ranges = &.{push_range},
        }, null);

        var metal_rough_material: GltfMetallicRoughness = try .init(scratch, io, bindless_pipeline_layout, draw_allocated_image, depth_allocated_image, device_proxy);

        // init_default_data {
        const rect_vertices: [4]Vertex = .{
            .{ .position = .{ 0.5, -0.5, 0.5 }, .color = .{ 0, 0, 0, 1 }, .uv_x = 0, .uv_y = 0, .normal = @splat(0) },
            .{ .position = .{ 0.5, 0.5, 0.5 }, .color = .{ 0.5, 0.5, 0.5, 1 }, .uv_x = 0, .uv_y = 0, .normal = @splat(0) },
            .{ .position = .{ -0.5, -0.5, 0.5 }, .color = .{ 1, 0, 0, 1 }, .uv_x = 0, .uv_y = 0, .normal = @splat(0) },
            .{ .position = .{ -0.5, 0.5, 0.5 }, .color = .{ 0, 1, 0, 1 }, .uv_x = 0, .uv_y = 0, .normal = @splat(0) },
        };
        const rect_indices: [6]u32 = .{ 0, 1, 2, 2, 1, 3 };
        const device_ctx: DeviceContext = .{
            .device = device_proxy,
            .graphics_queue = graphics_queue,
            .graphics_queue_family = queue_family_indices.graphics_family,
            .vma_allocator = vma_allocator,
        };
        const imm: ImmSubmit = .{
            .cmd = imm_command_buffer,
            .command_pool = imm_command_pool,
            .fence = imm_fence,
        };

        const rectangle = try uploadMesh(device_ctx, imm, &rect_indices, &rect_vertices);

        try main_deletion_queue.append(gpa, .{ .allocated_buffer = rectangle.index_buffer });
        try main_deletion_queue.append(gpa, .{ .allocated_buffer = rectangle.vertex_buffer });

        //{ default images
        const Color = packed struct(u32) { r: u8, g: u8, b: u8, a: u8 };

        const white: Color = .{ .r = 255, .g = 255, .b = 255, .a = 255 };
        const white_image = try createAndUploadImage(device_ctx, imm, @ptrCast(&white), .{ .width = 1, .height = 1, .depth = 1 }, .r8g8b8a8_unorm, .{ .sampled_bit = true }, false);
        try main_deletion_queue.append(gpa, .{ .vma_allocated_image = white_image });

        const grey: Color = .{ .r = 168, .g = 168, .b = 168, .a = 255 };
        const grey_image = try createAndUploadImage(device_ctx, imm, @ptrCast(&grey), .{ .width = 1, .height = 1, .depth = 1 }, .r8g8b8a8_unorm, .{ .sampled_bit = true }, false);
        try main_deletion_queue.append(gpa, .{ .vma_allocated_image = grey_image });

        const black: Color = .{ .r = 0, .g = 0, .b = 0, .a = 255 };
        const black_image = try createAndUploadImage(device_ctx, imm, @ptrCast(&black), .{ .width = 1, .height = 1, .depth = 1 }, .r8g8b8a8_unorm, .{ .sampled_bit = true }, false);
        try main_deletion_queue.append(gpa, .{ .vma_allocated_image = black_image });

        const error_checkerboard_image = blk: {
            const magenta: Color = .{ .r = 255, .g = 0, .b = 255, .a = 255 };
            var pixels: [16][16]Color = undefined;
            for (0..pixels.len) |x| {
                for (0..pixels[0].len) |y| {
                    pixels[x][y] = if ((x % 2) ^ (y % 2) != 0) magenta else black;
                }
            }
            break :blk try createAndUploadImage(device_ctx, imm, @ptrCast(&pixels), .{ .width = 16, .height = 16, .depth = 1 }, .r8g8b8a8_unorm, .{ .sampled_bit = true }, false);
        };
        try main_deletion_queue.append(gpa, .{ .vma_allocated_image = error_checkerboard_image });

        var sampler_create_info: vk.SamplerCreateInfo = .{
            .mag_filter = .nearest,
            .min_filter = .nearest,
            .mipmap_mode = .nearest,
            .address_mode_u = .repeat,
            .address_mode_v = .repeat,
            .address_mode_w = .repeat,
            .mip_lod_bias = 0,
            .anisotropy_enable = .false,
            .max_anisotropy = 0,
            .compare_enable = .false,
            .compare_op = .never,
            .min_lod = 0,
            .max_lod = 0,
            .border_color = .float_transparent_black,
            .unnormalized_coordinates = .false,
        };

        const default_sampler_nearest = try device_ctx.device.createSampler(&sampler_create_info, null);
        try main_deletion_queue.append(gpa, .{ .sampler = default_sampler_nearest });

        sampler_create_info.mag_filter = .linear;
        sampler_create_info.min_filter = .linear;
        const default_sampler_linear = try device_ctx.device.createSampler(&sampler_create_info, null);
        try main_deletion_queue.append(gpa, .{ .sampler = default_sampler_linear });

        var materials_buffer: GltfMetallicRoughness.MaterialsBuffer = try .init(1024, vma_allocator);
        bindless_descriptors.registerBuffer(device_proxy, 1, materials_buffer.gpu_buffer.buffer, materials_buffer.gpu_buffer.size);

        const scene_data_buffer: VmaGpuBuffer = try .create(
            vma_allocator,
            FrameData.frame_overlap * @sizeOf(GPUSceneData),
            .{ .storage_buffer_bit = true },
            .auto,
            .sequential_write,
        );
        bindless_descriptors.registerBuffer(device_proxy, 2, scene_data_buffer.buffer, scene_data_buffer.size);

        var material_resources: GltfMetallicRoughness.MaterialResources = .{
            .color_image = white_image,
            .color_sampler = default_sampler_linear,
            .metal_rough_image = white_image,
            .metal_rough_sampler = default_sampler_linear,
            .color_factors = @splat(1),
            .metal_rough_factors = .{ 1, 0.5, 0, 0 },
        };

        const default_data = try metal_rough_material.writeMaterial(
            gpa,
            device_proxy,
            .main_color,
            &material_resources,
            &bindless_descriptors,
            &materials_buffer,
        );

        //}

        const structure_path = options.assets_path ++ "/structure.glb";
        const structure_file = try loader.loadGltf(
            gpa,
            scratch,
            io,
            &metal_rough_material,
            default_sampler_linear,
            white_image,
            error_checkerboard_image,
            device_ctx,
            imm,
            structure_path,
            &bindless_descriptors,
            &materials_buffer,
        );

        var loaded_scenes: std.StringHashMapUnmanaged(*scene.Node) = .empty;

        const loaded_scene_node = try gpa.create(scene.Node);
        loaded_scene_node.* = .{
            .gltf = structure_file,
        };

        try loaded_scenes.put(gpa, "structure", loaded_scene_node);

        return .{
            .vk_ctx = .{
                .base_dispatch = base_dispatch,
                .instance = instance_proxy,
                .window_surface = sdl_window_surface,
                .chosen_gpu = physical_device,
                .debug_messenger = debug_messenger,
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

            .scene_data = .{
                .view = .identity,
                .proj = .identity,
                .viewproj = .identity,
                .ambientColor = @splat(0),
                .sunlightDirection = @splat(0), // w for sun power
                .sunlightColor = @splat(0),
            },

            .imm = imm,

            .bindless_descriptors = bindless_descriptors,
            .bindless_pipeline_layout = bindless_pipeline_layout,

            .materials_buffer = materials_buffer,
            .scene_data_buffer = scene_data_buffer,

            .white_image = white_image,
            .grey_image = grey_image,
            .black_image = black_image,
            .error_checkerboard_image = error_checkerboard_image,

            .default_sampler_nearest = default_sampler_nearest,
            .default_sampler_linear = default_sampler_linear,

            .metal_rough_material = metal_rough_material,
            .default_data = default_data,

            .main_camera = .{
                .position = .{ 0, 0, 5 },
            },

            .main_draw_context = .{
                .opaque_surfaces = .empty,
                .transparent_surfaces = .empty,
            },
            .loaded_scenes = loaded_scenes,
        };
    }

    fn drawExtent(self: *const Engine) vk.Extent2D {
        return .{ .width = self.draw_image.image_extent.width, .height = self.draw_image.image_extent.height };
    }

    pub fn createImage(device_ctx: DeviceContext, size: vk.Extent3D, format: vk.Format, usage: vk.ImageUsageFlags, mipmapped: bool) !AllocatedImage {
        var img_info: vk.ImageCreateInfo = vk_init.imageCreateInfo(format, usage, size);
        if (mipmapped) {
            img_info.mip_levels = std.math.log2(@max(size.width, size.height)) + 1;
        }

        // always allocate images on dedicated GPU memory
        const alloc_info: c.VmaAllocationCreateInfo = .{
            .usage = c.VMA_MEMORY_USAGE_GPU_ONLY,
            .requiredFlags = c.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        };

        var image: vk.Image = undefined;
        var image_allocation: c.VmaAllocation = undefined;
        _ = c.vmaCreateImage(device_ctx.vma_allocator, @ptrCast(&img_info), &alloc_info, @ptrCast(&image), &image_allocation, null);

        // if the format is a depth format, we will need to have it use the correct aspect flag
        const aspect_flag: vk.ImageAspectFlags = switch (format == .d32_sfloat) {
            true => .{ .depth_bit = true },
            false => .{ .color_bit = true },
        };

        // build a image-view for the image
        var view_info: vk.ImageViewCreateInfo = vk_init.imageViewCreateInfo(format, image, aspect_flag);
        view_info.subresource_range.level_count = img_info.mip_levels;

        return .{
            .image_format = format,
            .image_extent = size,
            .image = image,
            .image_view = try device_ctx.device.createImageView(&view_info, null),
            .allocation = image_allocation,
        };
    }

    pub fn createAndUploadImage(device_ctx: DeviceContext, imm: ImmSubmit, data: *const anyopaque, size: vk.Extent3D, format: vk.Format, usage: vk.ImageUsageFlags, mipmapped: bool) !AllocatedImage {
        const data_size: usize = size.depth * size.width * size.height * 4;
        const upload_buffer: VmaGpuBuffer = try .create(
            device_ctx.vma_allocator,
            data_size,
            .{ .transfer_src_bit = true },
            .cpu_to_gpu,
            .sequential_write,
        );

        const map = upload_buffer.getMappedSlice(u8);
        @memcpy(map, @as([*]const u8, @ptrCast(data)));

        var new_usage = usage;
        new_usage.transfer_dst_bit = true;
        new_usage.transfer_src_bit = true;
        const new_image = try createImage(device_ctx, size, format, new_usage, mipmapped);

        {
            try immediateModeBegin(device_ctx.device, imm.fence, imm.cmd);

            vk_image.transitionImage(device_ctx.device, imm.cmd, new_image.image, .undefined, .transfer_dst_optimal);

            const copy_region: vk.BufferImageCopy = .{
                .buffer_offset = 0,
                .buffer_row_length = 0,
                .buffer_image_height = 0,

                .image_subresource = .{
                    .aspect_mask = .{ .color_bit = true },
                    .mip_level = 0,
                    .base_array_layer = 0,
                    .layer_count = 1,
                },
                .image_extent = size,
                .image_offset = .{ .x = 0, .y = 0, .z = 0 },
            };
            device_ctx.device.cmdCopyBufferToImage(imm.cmd, upload_buffer.buffer, new_image.image, .transfer_dst_optimal, &.{copy_region});

            if (mipmapped) {
                vk_image.generateMipmaps(device_ctx.device, imm.cmd, new_image.image, .{ .width = new_image.image_extent.width, .height = new_image.image_extent.height });
            } else {
                vk_image.transitionImage(device_ctx.device, imm.cmd, new_image.image, .transfer_dst_optimal, .read_only_optimal);
            }

            try immediateModeEnd(device_ctx.device, imm.fence, imm.cmd, device_ctx.graphics_queue);
        }

        upload_buffer.destroy(device_ctx.vma_allocator);
        return new_image;
    }

    pub fn destroyImage(self: Engine, img: *AllocatedImage) void {
        self.device_ctx.device.destroyImageView(img.image_view, null);
        c.vmaDestroyImage(self.device_ctx.vma_allocator, @ptrFromInt(@intFromEnum(img.image)), img.allocation);
    }

    pub fn resizeSwapchain(self: *Engine, gpa: Allocator, scratch: *Scratch) !void {
        const device = self.device_ctx.device;
        try device.deviceWaitIdle();

        self.swapchain.deinit(gpa, device);

        var w: i32 = undefined;
        var h: i32 = undefined;
        _ = c.SDL_GetWindowSize(self.window, &w, &h);
        const new_width: u32 = @intCast(w);
        const new_height: u32 = @intCast(h);

        self.swapchain = try .init(
            gpa,
            scratch,
            self.vk_ctx.chosen_gpu,
            device,
            self.vk_ctx.window_surface,
            new_width,
            new_height,
            self.vk_ctx.instance.wrapper.*,
        );

        self.destroyImage(&self.draw_image);
        self.destroyImage(&self.depth_image);

        const new_extent: vk.Extent3D = .{ .width = new_width, .height = new_height, .depth = 1 };

        self.draw_image = try createImage(
            self.device_ctx,
            new_extent,
            .r16g16b16a16_sfloat,
            .{ .transfer_src_bit = true, .transfer_dst_bit = true, .storage_bit = true, .color_attachment_bit = true },
            false,
        );

        self.depth_image = try createImage(
            self.device_ctx,
            new_extent,
            .d32_sfloat,
            .{ .depth_stencil_attachment_bit = true },
            false,
        );

        self.resize_requested = false;
    }

    pub fn deinit(self: *Engine, gpa: Allocator) void {
        const device = self.device_ctx.device;
        device.deviceWaitIdle() catch @panic(""); // TODO

        for (0..self.frames.len) |i| {
            device.destroyCommandPool(self.frames[i].command_pool, null);

            device.destroyFence(self.frames[i].render_fence, null);
            device.destroySemaphore(self.frames[i].swapchain_semaphore, null);

            self.frames[i].deletion_queue.deinit(gpa, .{
                .device = device,
                .vma_allocator = self.device_ctx.vma_allocator,
            });
        }

        var it = self.loaded_scenes.valueIterator();
        while (it.next()) |s| {
            s.*.clearAll(gpa, self);
        }
        self.loaded_scenes.deinit(gpa);

        self.metal_rough_material.deinit(device);

        self.main_deletion_queue.deinit(gpa, .{
            .device = device,
            .vma_allocator = self.device_ctx.vma_allocator,
        });

        self.main_draw_context.opaque_surfaces.deinit(gpa);

        self.bindless_descriptors.deinit(gpa, device);

        self.swapchain.deinit(gpa, device);

        device.destroyDevice(null);
        self.vk_ctx.instance.destroySurfaceKHR(self.vk_ctx.window_surface, null);
        self.vk_ctx.instance.destroyDebugUtilsMessengerEXT(self.vk_ctx.debug_messenger, null);
        self.vk_ctx.instance.destroyInstance(null);

        c.SDL_DestroyWindow(self.window);
        c.SDL_Quit();

        self.init_arena.deinit();
    }

    fn drawImgui(self: *Engine, cmd: vk.CommandBuffer, target_image_view: vk.ImageView) void {
        const zone = tracy.zone(@src());
        defer zone.end();

        const device = self.device_ctx.device;
        const color_attachment = vk_init.attachmentInfo(target_image_view, null, .attachment_optimal);
        const render_info = vk_init.renderingInfo(self.swapchain.extent, &color_attachment, null);

        device.cmdBeginRendering(cmd, &render_info);

        c.cImGui_ImplVulkan_NewFrame();
        c.cImGui_ImplSDL3_NewFrame();
        c.ImGui_NewFrame();

        c.ImGui_SetNextWindowSize(.{ .x = 300, .y = 200 }, c.ImGuiCond_Once);
        if (c.ImGui_Begin("out", null, 0)) {}
        c.ImGui_End();

        c.ImGui_Render();
        c.cImGui_ImplVulkan_RenderDrawData(c.ImGui_GetDrawData(), @ptrFromInt(@intFromEnum(cmd)));

        device.cmdEndRendering(cmd);
    }

    pub fn updateScene(self: *Engine, gpa: Allocator) !void {
        const zone = tracy.zone(@src());
        defer zone.end();

        self.main_draw_context.transparent_surfaces.clearRetainingCapacity();
        self.main_draw_context.opaque_surfaces.clearRetainingCapacity();

        try self.loaded_scenes.get("structure").?.draw(gpa, .identity, &self.main_draw_context);

        self.main_camera.update();

        const view = self.main_camera.getViewMatrix();

        // camera projection
        const projection: Mat4 = .perspectiveReverseZ(
            zla.toRadians(f32, 70),
            @as(f32, @floatFromInt(self.swapchain.extent.width)) / @as(f32, @floatFromInt(self.swapchain.extent.height)),
            0.1,
            .{},
        );

        self.scene_data.view = view;
        self.scene_data.proj = projection;
        self.scene_data.viewproj = self.scene_data.proj.mul(self.scene_data.view);

        //some default lighting parameters
        self.scene_data.ambientColor = @splat(0.1);
        self.scene_data.sunlightColor = @splat(1);
        self.scene_data.sunlightDirection = .{ 0, 1, 0.5, 1 };
    }

    pub fn drawGeometry(self: *Engine, scratch: *Scratch, cmd: vk.CommandBuffer) !void {
        const checkpoint = scratch.checkpoint();
        defer scratch.restoreCheckpoint(checkpoint);
        const zone = tracy.zone(@src());
        defer zone.end();

        //begin a render pass connected to our draw image
        const color_attachment: vk.RenderingAttachmentInfo = vk_init.attachmentInfo(self.draw_image.image_view, .{ .color = .{ .float_32 = .{ 0.0, 0.0, 0.0, 1.0 } } }, .attachment_optimal);
        const depthAttachment: vk.RenderingAttachmentInfo = vk_init.depthAttachmentInfo(self.depth_image.image_view, .depth_attachment_optimal);

        const render_info = vk_init.renderingInfo(self.drawExtent(), &color_attachment, &depthAttachment);
        const device = self.device_ctx.device;

        device.cmdBeginRendering(cmd, &render_info);
        {
            const frame_index = self.currentFrameIndex();

            self.scene_data_buffer.getMappedSlice(GPUSceneData)[frame_index] = self.scene_data;

            device.cmdBindDescriptorSets(
                cmd,
                .graphics,
                self.bindless_pipeline_layout,
                0,
                (&self.bindless_descriptors.set)[0..1],
                null,
            );

            device.cmdSetViewport(cmd, 0, (&vk.Viewport{
                .x = 0,
                .y = 0,
                .width = @floatFromInt(self.drawExtent().width),
                .height = @floatFromInt(self.drawExtent().height),
                .min_depth = 0,
                .max_depth = 1,
            })[0..1]);

            device.cmdSetScissor(cmd, 0, (&vk.Rect2D{
                .offset = .{ .x = 0, .y = 0 },
                .extent = self.drawExtent(),
            })[0..1]);

            var draw_calls: usize = 0;
            var triangle_count: usize = 0;

            var last_pipeline: vk.Pipeline = .null_handle;
            var last_index_buffer: vk.Buffer = .null_handle;

            for ([_][]scene.RenderObject{
                self.main_draw_context.opaque_surfaces.items,
                self.main_draw_context.transparent_surfaces.items,
            }) |surfaces| {
                for (surfaces) |surface| {
                    if (surface.material.?.pipeline != last_pipeline) {
                        last_pipeline = surface.material.?.pipeline;
                        device.cmdBindPipeline(cmd, .graphics, last_pipeline);
                    }

                    if (surface.index_buffer != last_index_buffer) {
                        last_index_buffer = surface.index_buffer;
                        device.cmdBindIndexBuffer(cmd, surface.index_buffer, 0, .uint32);
                    }

                    const push_constants: GPUDrawPushConstants = .{
                        .world_matrix = surface.transform,
                        .vertex_buffer = surface.vertex_buffer_address,
                        .material_index = surface.material.?.bindless_index,
                        .scene_data_index = @intCast(frame_index),
                    };
                    device.cmdPushConstants(
                        cmd,
                        self.bindless_pipeline_layout,
                        .{ .vertex_bit = true, .fragment_bit = true },
                        0,
                        @sizeOf(GPUDrawPushConstants),
                        std.mem.asBytes(&push_constants),
                    );

                    device.cmdDrawIndexed(cmd, surface.index_count, 1, surface.first_index, 0, 0);

                    draw_calls += 1;
                    triangle_count += surface.index_count / 3;
                }
            }

            tracy.plot("draw calls", @floatFromInt(draw_calls));
            tracy.plot("triangle count", @floatFromInt(triangle_count));
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
        const vertexBuffer: VmaGpuBuffer = try .create(
            vma_allocator,
            vertexBufferSize,
            .{ .storage_buffer_bit = true, .transfer_dst_bit = true, .shader_device_address_bit = true },
            .gpu_only,
            .none,
        );

        //find the adress of the vertex buffer
        const deviceAdressInfo: vk.BufferDeviceAddressInfo = .{ .buffer = vertexBuffer.buffer };
        const vertexBufferAddress = device.getBufferDeviceAddress(&deviceAdressInfo);

        //create index buffer
        const indexBuffer: VmaGpuBuffer = try .create(
            vma_allocator,
            indexBufferSize,
            .{ .storage_buffer_bit = true, .transfer_dst_bit = true, .index_buffer_bit = true },
            .gpu_only,
            .none,
        );
        const newSurface: GPUMeshBuffers = .{
            .vertex_buffer = vertexBuffer,
            .index_buffer = indexBuffer,
            .vertex_buffer_address = vertexBufferAddress,
        };

        const staging: VmaGpuBuffer = try .create(
            vma_allocator,
            vertexBufferSize + indexBufferSize,
            .{ .transfer_src_bit = true },
            .cpu_only,
            .sequential_write,
        );
        defer staging.destroy(vma_allocator);

        var staging_map = staging.getMappedSlice(u8);

        @memcpy(@as([*]Vertex, @ptrCast(@alignCast(staging_map.ptr))), vertices); // copy vertex buffer
        @memcpy(@as([*]u32, @ptrCast(@alignCast(staging_map[vertexBufferSize..]))), indices); // copy index buffer

        {
            try immediateModeBegin(device, imm.fence, imm.cmd);

            const vertexCopy: vk.BufferCopy = .{
                .dst_offset = 0,
                .src_offset = 0,
                .size = vertexBufferSize,
            };
            device.cmdCopyBuffer(imm.cmd, staging.buffer, newSurface.vertex_buffer.buffer, &.{vertexCopy});

            const indexCopy: vk.BufferCopy = .{
                .dst_offset = 0,
                .src_offset = vertexBufferSize,
                .size = indexBufferSize,
            };
            device.cmdCopyBuffer(imm.cmd, staging.buffer, newSurface.index_buffer.buffer, &.{indexCopy});

            try immediateModeEnd(device, imm.fence, imm.cmd, device_ctx.graphics_queue);
        }

        return newSurface;
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
            .{ .type = .storage_buffer, .descriptor_count = 2 },
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
            .{
                .binding = 1,
                .descriptor_type = .storage_buffer,
                .descriptor_count = 1,
                .stage_flags = .{ .vertex_bit = true, .fragment_bit = true },
                .p_immutable_samplers = null,
            },
            .{
                .binding = 2,
                .descriptor_type = .storage_buffer,
                .descriptor_count = 1,
                .stage_flags = .{ .vertex_bit = true, .fragment_bit = true },
                .p_immutable_samplers = null,
            },
        };

        const binding_flags = [_]vk.DescriptorBindingFlags{
            .{ .partially_bound_bit = true, .update_after_bind_bit = true },
            .{ .update_after_bind_bit = true },
            .{ .update_after_bind_bit = true },
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

    pub fn registerBuffer(self: *BindlessDescriptors, device: vk.DeviceProxy, binding: u32, buffer: vk.Buffer, size: usize) void {
        device.updateDescriptorSets(&.{.{
            .dst_set = self.set,
            .dst_binding = binding,
            .dst_array_element = 0,
            .descriptor_count = 1,
            .descriptor_type = .storage_buffer,
            .p_image_info = undefined,
            .p_buffer_info = &.{.{
                .buffer = buffer,
                .offset = 0,
                .range = size,
            }},
            .p_texel_buffer_view = undefined,
        }}, null);
    }
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
        const subresource: vk.ImageSubresourceLayers = .{
            .aspect_mask = .{ .color_bit = true },
            .base_array_layer = 0,
            .layer_count = 1,
            .mip_level = 0,
        };
        const offset_base: vk.Offset3D = .{ .x = 0, .y = 0, .z = 0 };
        const blitRegion: vk.ImageBlit2 = .{
            .src_offsets = .{ offset_base, .{ .x = @intCast(src_size.width), .y = @intCast(src_size.height), .z = 1 } },
            .dst_offsets = .{ offset_base, .{ .x = @intCast(dst_size.width), .y = @intCast(dst_size.height), .z = 1 } },
            .src_subresource = subresource,
            .dst_subresource = subresource,
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

    pub fn generateMipmaps(
        device: vk.DeviceProxy,
        cmd: vk.CommandBuffer,
        image: vk.Image,
        image_size: vk.Extent2D,
    ) void {
        const mip_levels: u32 = std.math.log2_int(u32, @max(image_size.width, image_size.height)) + 1;
        var previous_image_size = image_size;
        for (0..mip_levels) |mip| {
            const half_size: vk.Extent2D = .{
                .width = @max(previous_image_size.width / 2, 1),
                .height = @max(previous_image_size.height / 2, 1),
            };

            const aspect_mask: vk.ImageAspectFlags = .{ .color_bit = true };
            var subresource_range = vk_init.imageSubresourceRange(aspect_mask);
            subresource_range.level_count = 1;
            subresource_range.base_mip_level = @intCast(mip);

            const image_barrier: vk.ImageMemoryBarrier2 = .{
                .src_stage_mask = .{ .all_commands_bit = true },
                .src_access_mask = .{ .memory_write_bit = true },
                .dst_stage_mask = .{ .all_commands_bit = true },
                .dst_access_mask = .{ .memory_write_bit = true, .memory_read_bit = true },

                .old_layout = .transfer_dst_optimal,
                .new_layout = .transfer_src_optimal,

                .subresource_range = subresource_range,
                .image = image,

                .src_queue_family_index = 0,
                .dst_queue_family_index = 0,
            };

            const dep_info: vk.DependencyInfo = .{
                .image_memory_barrier_count = 1,
                .p_image_memory_barriers = &.{image_barrier},
            };
            device.cmdPipelineBarrier2(cmd, &dep_info);

            if (mip < mip_levels - 1) {
                const blit_region: vk.ImageBlit2 = .{
                    .src_offsets = .{
                        .{ .x = 0, .y = 0, .z = 0 },
                        .{ .x = @intCast(previous_image_size.width), .y = @intCast(previous_image_size.height), .z = 1 },
                    },
                    .dst_offsets = .{
                        .{ .x = 0, .y = 0, .z = 0 },
                        .{ .x = @intCast(half_size.width), .y = @intCast(half_size.height), .z = 1 },
                    },
                    .src_subresource = .{
                        .aspect_mask = .{ .color_bit = true },
                        .base_array_layer = 0,
                        .layer_count = 1,
                        .mip_level = @intCast(mip),
                    },
                    .dst_subresource = .{
                        .aspect_mask = .{ .color_bit = true },
                        .base_array_layer = 0,
                        .layer_count = 1,
                        .mip_level = @intCast(mip + 1),
                    },
                };
                const blit_info: vk.BlitImageInfo2 = .{
                    .dst_image = image,
                    .dst_image_layout = .transfer_dst_optimal,
                    .src_image = image,
                    .src_image_layout = .transfer_src_optimal,
                    .filter = .linear,
                    .region_count = 1,
                    .p_regions = &.{blit_region},
                };
                device.cmdBlitImage2(cmd, &blit_info);

                previous_image_size = half_size;
            }
        }
        vk_image.transitionImage(device, cmd, image, .transfer_src_optimal, .read_only_optimal);
    }
};

const vk_init = struct {
    pub fn pipelineShaderStageCreateInfo(stage: vk.ShaderStageFlags, shader_module: vk.ShaderModule) vk.PipelineShaderStageCreateInfo {
        return .{ .stage = stage, .module = shader_module, .p_name = "main" };
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

    fn attachmentInfo(view: vk.ImageView, clear: ?vk.ClearValue, layout: vk.ImageLayout) vk.RenderingAttachmentInfo {
        return .{
            .image_view = view,
            .image_layout = layout,
            .load_op = if (clear) |_| .clear else .load,
            .store_op = .store,
            .clear_value = if (clear) |item| item else std.mem.zeroes(vk.ClearValue),
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

    pub fn findQueueFamilies(
        scratch: *Scratch,
        physical_device: vk.PhysicalDevice,
        instance_dispatch: vk.InstanceWrapper,
        surface: vk.SurfaceKHR,
    ) !?QueueFamilyIndices {
        const checkpoint = scratch.checkpoint();
        defer scratch.restoreCheckpoint(checkpoint);

        const queue_families = try instance_dispatch.getPhysicalDeviceQueueFamilyPropertiesAlloc(physical_device, scratch.allocator());

        var graphics_family: u32 = undefined;
        // TODO: prefer queue that supports both graphics and KHR
        for (queue_families, 0..) |queue_familie, i| {
            if (queue_familie.queue_flags.graphics_bit) {
                graphics_family = @intCast(i);
                break;
            }
        } else return null;

        var present_family: u32 = undefined;
        for (queue_families, 0..) |_, i| {
            if (try instance_dispatch.getPhysicalDeviceSurfaceSupportKHR(physical_device, @intCast(i), surface) == .true) {
                present_family = @intCast(i);
                break;
            }
        } else return null;

        return .{ .graphics_family = graphics_family, .present_family = present_family };
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

    pub fn createLogicalDevice(
        physical_device: vk.PhysicalDevice,
        instance_dispatch: vk.InstanceWrapper,
        queue_family_indices: QueueFamilyIndices,
    ) !vk.Device {
        const indices = [_]u32{
            queue_family_indices.graphics_family,
            queue_family_indices.present_family,
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

        var device_features_vk13: vk.PhysicalDeviceVulkan13Features = .{
            .dynamic_rendering = .true,
            .synchronization_2 = .true,
        };
        const device_features_vk12: vk.PhysicalDeviceVulkan12Features = .{
            .p_next = &device_features_vk13,
            .runtime_descriptor_array = .true,
            .shader_sampled_image_array_non_uniform_indexing = .true,
            .descriptor_binding_partially_bound = .true,
            .descriptor_binding_sampled_image_update_after_bind = .true,
            .descriptor_binding_storage_buffer_update_after_bind = .true,
            .buffer_device_address = .true,
            .descriptor_indexing = .true,
        };
        return try instance_dispatch.createDevice(physical_device, &.{
            .p_next = &device_features_vk12,
            .p_queue_create_infos = queue_create_infos.items.ptr,
            .queue_create_info_count = @intCast(queue_create_infos.items.len),
            .pp_enabled_extension_names = &required_device_extensions,
            .enabled_extension_count = required_device_extensions.len,
            .p_enabled_features = &.{
                .shader_int_64 = .true,
                .sampler_anisotropy = .true,
            },
        }, null);
    }
};

fn loadShader(gpa: Allocator, io: std.Io, file_path: []const u8) !ShaderData {
    std.log.info("loading {s} shader", .{std.fs.path.basename(file_path)});
    const data = try std.Io.Dir.cwd().readFileAllocOptions(io, file_path, gpa, .unlimited, .of(u32), null);
    return .{ .ptr = @ptrCast(data.ptr), .size = data.len };
}

const std = @import("std");
const Allocator = std.mem.Allocator;

const vk = @import("vulkan");
const c = @import("c");
const tracy = @import("tracy");
const shaders = @import("shaders");
const loader = @import("loader.zig");
const zla = @import("zla");
const vec = zla.vec;
pub const Mat4 = zla.Mat(.cm, f32, 4, 4);
const options = @import("options");
const Scratch = @import("scratch_allocator");
const SegmentedList = @import("segmented_list.zig").SegmentedList;
