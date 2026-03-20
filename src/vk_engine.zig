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

pub const MeshBuffers = struct {
    vertex_buffer: VmaGpuBuffer,
    index_buffer: VmaGpuBuffer,

    next_vertex: u32,
    next_index: u32,

    pub const MeshEntry = struct {
        index_count: u32,
        index_offset: u32,
        vertex_offset: u32,
    };

    pub fn init(allocator: c.VmaAllocator, len: usize) !MeshBuffers {
        const vertex_buffer: VmaGpuBuffer = try .create(
            allocator,
            len * @sizeOf(Vertex),
            .{ .storage_buffer_bit = true, .shader_device_address_bit = true },
            .auto,
            .sequential_write,
        );

        return .{
            .vertex_buffer = vertex_buffer,
            .index_buffer = try .create(
                allocator,
                len * @sizeOf(u32),
                .{ .storage_buffer_bit = true, .index_buffer_bit = true },
                .auto,
                .sequential_write,
            ),
            .next_vertex = 0,
            .next_index = 0,
        };
    }

    pub fn upload(self: *MeshBuffers, vertices: []const Vertex, indices: []const u32) MeshEntry {
        const entry: MeshEntry = .{
            .index_count = @intCast(indices.len),
            .index_offset = self.next_index,
            .vertex_offset = @intCast(self.next_vertex),
        };
        @memcpy(self.vertex_buffer.getMappedSlice(Vertex)[self.next_vertex..][0..vertices.len], vertices);
        @memcpy(self.index_buffer.getMappedSlice(u32)[self.next_index..][0..indices.len], indices);
        self.next_vertex += @intCast(vertices.len);
        self.next_index += @intCast(indices.len);
        return entry;
    }
};

pub const scene = struct {
    const RenderObject = struct {
        mesh_entry: MeshBuffers.MeshEntry,

        material: ?*MaterialInstance,
        transform: Mat4,
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
                        .mesh_entry = .{
                            .index_count = surface.count,
                            .index_offset = mesh.mesh.index_offset + surface.start_index,
                            .vertex_offset = mesh.mesh.vertex_offset,
                        },
                        .material = if (surface.material) |material| &material.data else null,
                        .transform = node_matrix,
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
            const device = engine.graphics_ctx.device;

            for (self.children.items) |child| {
                child.clearAll(gpa, engine);
            }
            self.children.deinit(gpa);

            if (self.gltf) |gltf| {
                gltf.material_data_buffer.destroy(engine.graphics_ctx.vma_allocator);

                var mesh_it = gltf.meshes.valueIterator();
                while (mesh_it.next()) |mesh| {
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
        pad: [2]u32 = @splat(0),
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
                    .{ .storage_buffer_bit = true, .shader_device_address_bit = true },
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

    pub fn getDeviceAddress(self: VmaGpuBuffer, device: vk.DeviceProxy) vk.DeviceAddress {
        return device.getBufferDeviceAddress(&.{ .buffer = self.buffer });
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

    materials: vk.DeviceAddress,
    vertices: vk.DeviceAddress,
    draw_data: vk.DeviceAddress,
};

pub const GPUDrawPushConstants = extern struct {
    scene_data: vk.DeviceAddress,
    scene_data_index: u32,
};

pub const GPUDrawData = extern struct {
    world_matrix: Mat4,
    material_index: u32,
    pad: [3]u32 = @splat(0),
};

pub const AllocatedImage = struct {
    image: vk.Image,
    image_view: vk.ImageView,
    allocation: c.VmaAllocation,
    image_extent: vk.Extent3D,
    image_format: vk.Format,
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
        image_view: vk.ImageView,
        vma_allocated_image: AllocatedImage,
        descriptor_set_layout: vk.DescriptorSetLayout,
        pipeline_layout: vk.PipelineLayout,
        pipeline: vk.Pipeline,
        command_pool: vk.CommandPool,
        fence: vk.Fence,
        descriptor_pool: vk.DescriptorPool,
        allocated_buffer: VmaGpuBuffer,
        sampler: vk.Sampler,

        fn deinit(self: QueueItem, context: DeinitContext) void {
            switch (self) {
                .image_view => |item| context.device.destroyImageView(item, null),
                .vma_allocated_image => |item| {
                    c.vmaDestroyImage(context.vma_allocator.?, @ptrFromInt(@intFromEnum(item.image)), item.allocation);
                    context.device.destroyImageView(item.image_view, null);
                },
                .descriptor_set_layout => |item| context.device.destroyDescriptorSetLayout(item, null),
                .pipeline_layout => |item| context.device.destroyPipelineLayout(item, null),
                .pipeline => |item| context.device.destroyPipeline(item, null),
                .command_pool => |item| context.device.destroyCommandPool(item, null),
                .fence => |item| context.device.destroyFence(item, null),
                .descriptor_pool => |item| context.device.destroyDescriptorPool(item, null),
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
        queues: Queues,
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

        const concurrent = queues.families.graphics != queues.families.present;
        const family_indices = [_]u32{ queues.families.graphics, queues.families.present };
        const swapchain_create_info = vk.SwapchainCreateInfoKHR{
            .surface = window_surface,
            .min_image_count = min_image_count,
            .image_format = swapchain_image_format.format,
            .image_color_space = swapchain_image_format.color_space,
            .image_extent = swapchain_extent,
            .image_array_layers = 1,
            .image_usage = .{ .transfer_src_bit = true, .color_attachment_bit = true, .transfer_dst_bit = true },
            .image_sharing_mode = if (concurrent) .concurrent else .exclusive,
            .queue_family_index_count = if (concurrent) 2 else 0,
            .p_queue_family_indices = if (concurrent) &family_indices else null,
            .pre_transform = .{ .identity_bit_khr = true },
            .composite_alpha = .{ .opaque_bit_khr = true },
            .present_mode = .immediate_khr,
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
    const max_draws = 65536;

    swapchain_semaphore: vk.Semaphore,
    render_fence: vk.Fence,

    command_pool: vk.CommandPool,
    main_command_buffer: vk.CommandBuffer,

    indirect_buffer: VmaGpuBuffer,

    deletion_queue: DeletionQueue,
};

pub const Engine = struct {
    graphics_ctx: GraphicsCtx,

    window: *c.SDL_Window,

    swapchain: SwapChain,
    resize_requested: bool,

    frame_number: u64,
    frames: [FrameData.frame_overlap]FrameData,

    main_deletion_queue: DeletionQueue,

    //draw resources
    depth_image: AllocatedImage,
    draw_image: AllocatedImage,

    scene_data: GPUSceneData,

    materials_buffer: GltfMetallicRoughness.MaterialsBuffer,
    mesh_buffers: MeshBuffers,
    draw_data_buffer: VmaGpuBuffer,

    scene_data_buffer: VmaGpuBuffer,
    scene_data_adress: vk.DeviceAddress,

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

        const device = self.graphics_ctx.device;

        const wait_fence = tracy.zoneEx(@src(), .{ .name = "wait_fence" });
        const wait_result = try device.waitForFences(&.{self.currentFrame().render_fence}, .true, 1e9);
        if (wait_result == .timeout) return error.FenceTimeout;
        wait_fence.end();

        self.currentFrame().deletion_queue.flush(.{ .device = device, .vma_allocator = self.graphics_ctx.vma_allocator });

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

        _ = try device.resetFences(&.{self.currentFrame().render_fence});

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
            try device.queueSubmit2(self.graphics_ctx.queues.graphics, &.{submit_info}, self.currentFrame().render_fence);
        }

        const present_info: vk.PresentInfoKHR = .{
            .p_swapchains = (&self.swapchain.handle)[0..1],
            .swapchain_count = 1,
            .p_wait_semaphores = (&current_swap_image.render_semaphore)[0..1],
            .wait_semaphore_count = 1,
            .p_image_indices = (&swapchain_image_index)[0..1],
        };
        _ = device.queuePresentKHR(self.graphics_ctx.queues.present, &present_info) catch |err| switch (err) {
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

    fn immediateModeEnd(device: vk.DeviceProxy, imm_fence: vk.Fence, imm_command_buffer: vk.CommandBuffer, queue: vk.Queue) !void {
        try device.endCommandBuffer(imm_command_buffer);

        const cmdinfo: vk.CommandBufferSubmitInfo = vk_init.commandBufferSubmitInfo(imm_command_buffer);
        const submit: vk.SubmitInfo2 = vk_init.submitInfo(&cmdinfo, null, null);

        // submit command buffer to the queue and execute it.
        //  _renderFence will now block until the graphic commands finish execution
        try device.queueSubmit2(queue, &.{submit}, imm_fence);
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

        const window_width = 1080;
        const window_height = 1080;

        const tracy_SDL_CreateWindow = tracy.zoneEx(@src(), .{ .name = "SDL_CreateWindow" });
        const window = c.SDL_CreateWindow("title", window_width, window_height, c.SDL_WINDOW_VULKAN | c.SDL_WINDOW_RESIZABLE) orelse return error.engine_init_failure;
        tracy_SDL_CreateWindow.end();

        const tracy_load_base_dispatch = tracy.zoneEx(@src(), .{ .name = "load base_dispatch" });
        const base_dispatch = vk.BaseWrapper.load(@as(vk.PfnGetInstanceProcAddr, @ptrCast(c.SDL_Vulkan_GetVkGetInstanceProcAddr())));
        tracy_load_base_dispatch.end();

        var graphics_ctx: GraphicsCtx = try .init(gpa, scratch, window);

        const queues = graphics_ctx.queues;
        const device = graphics_ctx.device;
        const vma_allocator = graphics_ctx.vma_allocator;
        const physical_device = graphics_ctx.physical_device;
        const sdl_window_surface = graphics_ctx.window_surface;

        const instance_dispatch = graphics_ctx.instance.wrapper;
        const instance_handle = graphics_ctx.instance.handle;
        const device_handle = graphics_ctx.device.handle;
        const bindless_descriptors = &graphics_ctx.bindless_descriptors;
        const bindless_pipeline_layout = graphics_ctx.bindless_pipeline_layout;

        // init_commands() {
        // init_sync_structures() {
        const command_pool_info: vk.CommandPoolCreateInfo = .{
            .flags = .{ .reset_command_buffer_bit = true },
            .queue_family_index = queues.families.graphics,
        };

        const fence_create_info: vk.FenceCreateInfo = .{ .flags = .{ .signaled_bit = true } };
        var main_deletion_queue: DeletionQueue = .init;
        // }}

        var frames: [FrameData.frame_overlap]FrameData = undefined;
        for (&frames) |*frame| {
            const command_pool = try device.createCommandPool(&command_pool_info, null);

            var main_command_buffer: vk.CommandBuffer = undefined;
            const cmd_alloc_info: vk.CommandBufferAllocateInfo = .{
                .command_pool = command_pool,
                .command_buffer_count = 1,
                .level = .primary,
            };
            try device.allocateCommandBuffers(&cmd_alloc_info, (&main_command_buffer)[0..1]);

            frame.* = .{
                .command_pool = command_pool,
                .render_fence = try device.createFence(&fence_create_info, null),
                .swapchain_semaphore = try device.createSemaphore(&.{}, null),
                .main_command_buffer = main_command_buffer,
                .deletion_queue = .init,
                .indirect_buffer = try .create(
                    vma_allocator,
                    FrameData.max_draws * @sizeOf(vk.DrawIndexedIndirectCommand),
                    .{ .indirect_buffer_bit = true, .storage_buffer_bit = true },
                    .cpu_to_gpu,
                    .sequential_write,
                ),
            };
        }

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
            .image_view = try device.createImageView(&rview_info, null),
            .allocation = draw_image_allocation,
        };

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
            .image_view = try device.createImageView(&dview_info, null),
            .image = depth_image,
            .allocation = depth_image_allocation,
        };
        //}

        const swapchain: SwapChain = try .init(
            gpa,
            scratch,
            physical_device,
            device,
            sdl_window_surface,
            window_width,
            window_height,
            instance_dispatch.*,
            queues,
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

            const imgui_pool = try device.createDescriptorPool(&pool_info, null);

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
            ImguiVkLoader.instance_ = instance_handle;

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
                .Instance = @ptrFromInt(@intFromEnum(instance_handle)),
                .PhysicalDevice = @ptrFromInt(@intFromEnum(physical_device)),
                .Device = @ptrFromInt(@intFromEnum(device_handle)),
                .Queue = @ptrFromInt(@intFromEnum(queues.graphics)),
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
        }

        var metal_rough_material: GltfMetallicRoughness = try .init(scratch, io, bindless_pipeline_layout, draw_allocated_image, depth_allocated_image, device);

        // init_default_data {

        //{ default images
        const Color = packed struct(u32) { r: u8, g: u8, b: u8, a: u8 };

        const white: Color = .{ .r = 255, .g = 255, .b = 255, .a = 255 };
        const white_image = try createAndUploadImage(graphics_ctx, @ptrCast(&white), .{ .width = 1, .height = 1, .depth = 1 }, .r8g8b8a8_unorm, .{ .sampled_bit = true }, false);
        try main_deletion_queue.append(gpa, .{ .vma_allocated_image = white_image });

        const grey: Color = .{ .r = 168, .g = 168, .b = 168, .a = 255 };
        const grey_image = try createAndUploadImage(graphics_ctx, @ptrCast(&grey), .{ .width = 1, .height = 1, .depth = 1 }, .r8g8b8a8_unorm, .{ .sampled_bit = true }, false);
        try main_deletion_queue.append(gpa, .{ .vma_allocated_image = grey_image });

        const black: Color = .{ .r = 0, .g = 0, .b = 0, .a = 255 };
        const black_image = try createAndUploadImage(graphics_ctx, @ptrCast(&black), .{ .width = 1, .height = 1, .depth = 1 }, .r8g8b8a8_unorm, .{ .sampled_bit = true }, false);
        try main_deletion_queue.append(gpa, .{ .vma_allocated_image = black_image });

        const error_checkerboard_image = blk: {
            const magenta: Color = .{ .r = 255, .g = 0, .b = 255, .a = 255 };
            var pixels: [16][16]Color = undefined;
            for (0..pixels.len) |x| {
                for (0..pixels[0].len) |y| {
                    pixels[x][y] = if ((x % 2) ^ (y % 2) != 0) magenta else black;
                }
            }
            break :blk try createAndUploadImage(graphics_ctx, @ptrCast(&pixels), .{ .width = 16, .height = 16, .depth = 1 }, .r8g8b8a8_unorm, .{ .sampled_bit = true }, false);
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

        const default_sampler_nearest = try graphics_ctx.device.createSampler(&sampler_create_info, null);
        try main_deletion_queue.append(gpa, .{ .sampler = default_sampler_nearest });

        sampler_create_info.mag_filter = .linear;
        sampler_create_info.min_filter = .linear;
        const default_sampler_linear = try graphics_ctx.device.createSampler(&sampler_create_info, null);
        try main_deletion_queue.append(gpa, .{ .sampler = default_sampler_linear });

        var materials_buffer: GltfMetallicRoughness.MaterialsBuffer = try .init(1024, vma_allocator);

        const scene_data_buffer: VmaGpuBuffer = try .create(
            vma_allocator,
            FrameData.frame_overlap * @sizeOf(GPUSceneData),
            .{ .storage_buffer_bit = true, .shader_device_address_bit = true },
            .auto,
            .sequential_write,
        );

        const draw_data_buffer: VmaGpuBuffer = try .create(
            vma_allocator,
            FrameData.max_draws * FrameData.frame_overlap * @sizeOf(GPUDrawData),
            .{ .storage_buffer_bit = true, .shader_device_address_bit = true },
            .cpu_to_gpu,
            .sequential_write,
        );

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
            device,
            .main_color,
            &material_resources,
            bindless_descriptors,
            &materials_buffer,
        );

        //}

        var mesh_buffers: MeshBuffers = try .init(vma_allocator, 64 * 1024 * 1024);

        const structure_path = options.assets_path ++ "/structure.glb";
        const structure_file = try loader.loadGltf(
            gpa,
            scratch,
            io,
            &metal_rough_material,
            default_sampler_linear,
            white_image,
            error_checkerboard_image,
            graphics_ctx,
            structure_path,
            bindless_descriptors,
            &materials_buffer,
            &mesh_buffers,
        );

        var loaded_scenes: std.StringHashMapUnmanaged(*scene.Node) = .empty;

        const loaded_scene_node = try gpa.create(scene.Node);
        loaded_scene_node.* = .{
            .gltf = structure_file,
        };

        try loaded_scenes.put(gpa, "structure", loaded_scene_node);

        return .{
            .graphics_ctx = graphics_ctx,

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
                .draw_data = draw_data_buffer.getDeviceAddress(device),
                .materials = materials_buffer.gpu_buffer.getDeviceAddress(device),
                .vertices = mesh_buffers.vertex_buffer.getDeviceAddress(device),
            },

            .materials_buffer = materials_buffer,
            .mesh_buffers = mesh_buffers,
            .draw_data_buffer = draw_data_buffer,

            .scene_data_buffer = scene_data_buffer,
            .scene_data_adress = scene_data_buffer.getDeviceAddress(device),

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

    pub fn createImage(graphics_ctx: GraphicsCtx, size: vk.Extent3D, format: vk.Format, usage: vk.ImageUsageFlags, mipmapped: bool) !AllocatedImage {
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
        _ = c.vmaCreateImage(graphics_ctx.vma_allocator, @ptrCast(&img_info), &alloc_info, @ptrCast(&image), &image_allocation, null);

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
            .image_view = try graphics_ctx.device.createImageView(&view_info, null),
            .allocation = image_allocation,
        };
    }

    pub fn createAndUploadImage(graphics_ctx: GraphicsCtx, data: *const anyopaque, size: vk.Extent3D, format: vk.Format, usage: vk.ImageUsageFlags, mipmapped: bool) !AllocatedImage {
        const data_size: usize = size.depth * size.width * size.height * 4;
        const upload_buffer: VmaGpuBuffer = try .create(
            graphics_ctx.vma_allocator,
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
        const new_image = try createImage(graphics_ctx, size, format, new_usage, mipmapped);

        {
            try immediateModeBegin(graphics_ctx.device, graphics_ctx.imm.fence, graphics_ctx.imm.cmd);

            vk_image.transitionImage(graphics_ctx.device, graphics_ctx.imm.cmd, new_image.image, .undefined, .transfer_dst_optimal);

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
            graphics_ctx.device.cmdCopyBufferToImage(graphics_ctx.imm.cmd, upload_buffer.buffer, new_image.image, .transfer_dst_optimal, &.{copy_region});

            if (mipmapped) {
                vk_image.generateMipmaps(graphics_ctx.device, graphics_ctx.imm.cmd, new_image.image, .{ .width = new_image.image_extent.width, .height = new_image.image_extent.height });
            } else {
                vk_image.transitionImage(graphics_ctx.device, graphics_ctx.imm.cmd, new_image.image, .transfer_dst_optimal, .read_only_optimal);
            }

            try immediateModeEnd(graphics_ctx.device, graphics_ctx.imm.fence, graphics_ctx.imm.cmd, graphics_ctx.queues.graphics);
        }

        upload_buffer.destroy(graphics_ctx.vma_allocator);
        return new_image;
    }

    pub fn destroyImage(self: Engine, img: *AllocatedImage) void {
        self.graphics_ctx.device.destroyImageView(img.image_view, null);
        c.vmaDestroyImage(self.graphics_ctx.vma_allocator, @ptrFromInt(@intFromEnum(img.image)), img.allocation);
    }

    pub fn resizeSwapchain(self: *Engine, gpa: Allocator, scratch: *Scratch) !void {
        const device = self.graphics_ctx.device;
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
            self.graphics_ctx.physical_device,
            device,
            self.graphics_ctx.window_surface,
            new_width,
            new_height,
            self.graphics_ctx.instance.wrapper.*,
            self.graphics_ctx.queues,
        );

        self.destroyImage(&self.draw_image);
        self.destroyImage(&self.depth_image);

        const new_extent: vk.Extent3D = .{ .width = new_width, .height = new_height, .depth = 1 };

        self.draw_image = try createImage(
            self.graphics_ctx,
            new_extent,
            .r16g16b16a16_sfloat,
            .{ .transfer_src_bit = true, .transfer_dst_bit = true, .storage_bit = true, .color_attachment_bit = true },
            false,
        );

        self.depth_image = try createImage(
            self.graphics_ctx,
            new_extent,
            .d32_sfloat,
            .{ .depth_stencil_attachment_bit = true },
            false,
        );

        self.resize_requested = false;
    }

    pub fn deinit(self: *Engine, gpa: Allocator) void {
        const device = self.graphics_ctx.device;
        device.deviceWaitIdle() catch {};

        for (0..self.frames.len) |i| {
            device.destroyCommandPool(self.frames[i].command_pool, null);

            device.destroyFence(self.frames[i].render_fence, null);
            device.destroySemaphore(self.frames[i].swapchain_semaphore, null);

            self.frames[i].indirect_buffer.destroy(self.graphics_ctx.vma_allocator);

            self.frames[i].deletion_queue.deinit(gpa, .{
                .device = device,
                .vma_allocator = self.graphics_ctx.vma_allocator,
            });
        }

        var it = self.loaded_scenes.valueIterator();
        while (it.next()) |s| {
            s.*.clearAll(gpa, self);
        }
        self.loaded_scenes.deinit(gpa);

        self.metal_rough_material.deinit(device);

        self.mesh_buffers.vertex_buffer.destroy(self.graphics_ctx.vma_allocator);
        self.mesh_buffers.index_buffer.destroy(self.graphics_ctx.vma_allocator);
        self.materials_buffer.deinit(self.graphics_ctx.vma_allocator);
        self.scene_data_buffer.destroy(self.graphics_ctx.vma_allocator);
        self.draw_data_buffer.destroy(self.graphics_ctx.vma_allocator);

        self.destroyImage(&self.draw_image);
        self.destroyImage(&self.depth_image);

        c.cImGui_ImplVulkan_Shutdown();

        self.main_deletion_queue.deinit(gpa, .{
            .device = device,
            .vma_allocator = self.graphics_ctx.vma_allocator,
        });

        self.main_draw_context.opaque_surfaces.deinit(gpa);

        self.swapchain.deinit(gpa, device);

        self.graphics_ctx.deinit(gpa);

        c.SDL_DestroyWindow(self.window);
        c.SDL_Quit();
    }

    fn drawImgui(self: *Engine, cmd: vk.CommandBuffer, target_image_view: vk.ImageView) void {
        const zone = tracy.zone(@src());
        defer zone.end();

        const device = self.graphics_ctx.device;
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

        const clear_color: vk.ClearValue = .{ .color = .{ .float_32 = .{ 0, 0, 0, 1 } } };
        const color_attachment = vk_init.attachmentInfo(self.draw_image.image_view, clear_color, .attachment_optimal);
        const depth_attachment = vk_init.depthAttachmentInfo(self.depth_image.image_view, .depth_attachment_optimal);
        const render_info = vk_init.renderingInfo(self.drawExtent(), &color_attachment, &depth_attachment);
        const device = self.graphics_ctx.device;

        device.cmdBeginRendering(cmd, &render_info);
        defer device.cmdEndRendering(cmd);

        const frame_index = self.currentFrameIndex();
        const frame = &self.frames[frame_index];

        self.scene_data_buffer.getMappedSlice(GPUSceneData)[frame_index] = self.scene_data;

        const opaque_surfaces = self.main_draw_context.opaque_surfaces.items;
        const transparent_surfaces = self.main_draw_context.transparent_surfaces.items;
        const total_draws = opaque_surfaces.len + transparent_surfaces.len;
        if (total_draws == 0) return;

        const all_surfaces = try scratch.allocator().alloc(scene.RenderObject, total_draws);
        @memcpy(all_surfaces[0..opaque_surfaces.len], opaque_surfaces);
        @memcpy(all_surfaces[opaque_surfaces.len..], transparent_surfaces);

        std.sort.pdq(scene.RenderObject, all_surfaces, {}, struct {
            fn lessThan(_: void, a: scene.RenderObject, b: scene.RenderObject) bool {
                return @intFromEnum(a.material.?.pipeline) < @intFromEnum(b.material.?.pipeline);
            }
        }.lessThan);

        const commands = frame.indirect_buffer.getMappedSlice(vk.DrawIndexedIndirectCommand);

        const base_offset = frame_index * FrameData.max_draws;
        const draw_data = self.draw_data_buffer.getMappedSlice(GPUDrawData);

        for (all_surfaces, 0..) |surface, i| {
            commands[i] = .{
                .index_count = surface.mesh_entry.index_count,
                .instance_count = 1,
                .first_index = surface.mesh_entry.index_offset,
                .vertex_offset = @intCast(surface.mesh_entry.vertex_offset),
                .first_instance = @intCast(base_offset + i),
            };
            draw_data[base_offset + i] = .{
                .world_matrix = surface.transform,
                .material_index = surface.material.?.bindless_index,
            };
        }

        const Batch = struct { pipeline: vk.Pipeline, offset: u32, count: u32 };
        var batch_buff: [16]Batch = undefined;
        var batches: std.ArrayList(Batch) = .initBuffer(&batch_buff);

        var start: usize = 0;
        while (start < total_draws) {
            const pipeline = all_surfaces[start].material.?.pipeline;
            var end = start + 1;
            while (end < total_draws and all_surfaces[end].material.?.pipeline == pipeline) end += 1;
            batches.appendAssumeCapacity(.{
                .pipeline = pipeline,
                .offset = @intCast(start),
                .count = @intCast(end - start),
            });
            start = end;
        }

        device.cmdBindDescriptorSets(cmd, .graphics, self.graphics_ctx.bindless_pipeline_layout, 0, (&self.graphics_ctx.bindless_descriptors.set)[0..1], null);

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

        device.cmdBindIndexBuffer(cmd, self.mesh_buffers.index_buffer.buffer, 0, .uint32);

        device.cmdPushConstants(
            cmd,
            self.graphics_ctx.bindless_pipeline_layout,
            .{ .vertex_bit = true, .fragment_bit = true },
            0,
            @sizeOf(GPUDrawPushConstants),
            std.mem.asBytes(&GPUDrawPushConstants{
                .scene_data = self.scene_data_adress,
                .scene_data_index = @intCast(frame_index),
            }),
        );

        for (batches.items) |batch| {
            device.cmdBindPipeline(cmd, .graphics, batch.pipeline);
            device.cmdDrawIndexedIndirect(
                cmd,
                frame.indirect_buffer.buffer,
                batch.offset * @sizeOf(vk.DrawIndexedIndirectCommand),
                batch.count,
                @sizeOf(vk.DrawIndexedIndirectCommand),
            );
        }
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
            .src_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
            .dst_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
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
        var previous_size = image_size;
        for (0..mip_levels) |mip| {
            const half_size: vk.Extent2D = .{
                .width = @max(previous_size.width / 2, 1),
                .height = @max(previous_size.height / 2, 1),
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
                        .{ .x = @intCast(previous_size.width), .y = @intCast(previous_size.height), .z = 1 },
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

                previous_size = half_size;
            }
        }
        vk_image.transitionImage(device, cmd, image, .transfer_src_optimal, .read_only_optimal);
    }
};

const vk_init = struct {
    fn renderingInfo(render_extent: vk.Extent2D, color_attachment: *const vk.RenderingAttachmentInfo, depth_attachment: ?*const vk.RenderingAttachmentInfo) vk.RenderingInfo {
        return .{
            .render_area = .{ .offset = .{ .x = 0, .y = 0 }, .extent = render_extent },
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
            .components = std.mem.zeroes(vk.ComponentMapping),
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
};

fn loadShader(gpa: Allocator, io: std.Io, file_path: []const u8) !ShaderData {
    std.log.info("loading {s} shader", .{file_path});
    const data = try std.Io.Dir.cwd().readFileAllocOptions(io, file_path, gpa, .unlimited, .of(u32), null);
    return .{ .ptr = @ptrCast(data.ptr), .size = data.len };
}

const std = @import("std");

const vk = @import("vulkan");
const c = @import("c");
const tracy = @import("tracy");
const shaders = @import("shaders");
const loader = @import("loader.zig");
const zla = @import("zla");
pub const Mat4 = zla.Mat(.cm, f32, 4, 4);
const options = @import("options");
const Scratch = @import("scratch_allocator");
const GraphicsCtx = @import("GraphicsCtx.zig");

const Allocator = std.mem.Allocator;
const Queues = GraphicsCtx.Queues;
pub const BindlessDescriptors = GraphicsCtx.BindlessDescriptors;
