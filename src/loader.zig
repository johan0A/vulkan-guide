pub const GltfMaterial = struct {
    data: vk_engine.MaterialInstance,
};

pub const GeoSurface = struct {
    start_index: u32,
    count: u32,

    material: ?*GltfMaterial,
};

pub const MeshAsset = struct {
    name: []const u8,

    surfaces: std.ArrayListUnmanaged(GeoSurface),
    mesh_buffers: vk_engine.GPUMeshBuffers,
};

pub fn loadGltf(
    gpa: std.mem.Allocator,
    scratch: *Scratch,
    io: std.Io,
    // engine: vk_engine.Engine,
    metal_rough_material: *vk_engine.GltfMetallicRoughness,
    default_sampler_linear: vk.Sampler,
    white_image: vk_engine.AllocatedImage,
    error_checkerboard_image: vk_engine.AllocatedImage,
    device_ctx: vk_engine.Engine.DeviceContext,
    imm: vk_engine.Engine.ImmSubmit,
    filePath: []const u8,
) !*LoadedGltf {
    std.log.info("Loading GLTF: {s}", .{filePath});
    const checkpoint = scratch.checkpoint();
    defer scratch.restoreCheckpoint(checkpoint);

    const scene = try gpa.create(LoadedGltf); // TODO: rename
    scene.* = .{
        .images = .empty,
        .descriptor_pool = .empty,
        .samplers = .empty,
        .materials = .empty,
        .meshes = .empty,
        .nodes = .empty,
        .top_nodes = .empty,
        .material_data_buffer = undefined,
    };

    const file_ = try std.Io.Dir.cwd().openFile(io, filePath, .{});
    var file_reader = file_.reader(io, &.{});

    var gltf: Gltf = .init(scratch.allocator());
    defer gltf.deinit();
    try gltf.parse(try file_reader.interface.allocRemainingAlignedSentinel(scratch.allocator(), .unlimited, .@"4", null));

    // const gltf_options = ::DontRequireValidAssetMember | ::AllowDouble | ::LoadGLBBuffers | ::LoadExternalBuffers;
    // ::LoadExternalImages;

    // std::filesystem::path path = filePath;

    const sizes = [_]vk_engine.descriptors.DescriptorAllocatorGrowable.PoolSizeRatio{
        .{ .type = .combined_image_sampler, .ratio = 3 },
        .{ .type = .uniform_buffer, .ratio = 3 },
        .{ .type = .storage_buffer, .ratio = 1 },
    };

    scene.descriptor_pool = .empty;
    try scene.descriptor_pool.init(gpa, scratch, device_ctx.device, 10, &sizes);
    errdefer scene.descriptor_pool.deinit(gpa, device_ctx.device);

    // load samplers
    for (gltf.data.samplers) |sampler| {
        const sampl: vk.SamplerCreateInfo = .{
            .max_lod = vk.LOD_CLAMP_NONE,
            .min_lod = 0,
            .mag_filter = if (sampler.mag_filter) |f| if (f == .nearest) .nearest else .linear else .nearest,
            .min_filter = extractFilter(sampler.min_filter orelse .nearest),
            .mipmap_mode = extractMipmapMode(sampler.min_filter orelse .nearest),

            .address_mode_v = .repeat,
            .address_mode_w = .repeat,
            .address_mode_u = .repeat,
            .mip_lod_bias = 0,
            .anisotropy_enable = .false,
            .max_anisotropy = 0,
            .compare_enable = .false,
            .compare_op = .never,
            .border_color = .float_transparent_black,
            .unnormalized_coordinates = .false,
        };

        const new_sampler: vk.Sampler = try device_ctx.device.createSampler(&sampl, null);
        try scene.samplers.append(gpa, new_sampler);
    }

    // temporal arrays for all the objects to use while creating the GLTF data
    var meshes: std.ArrayList(*MeshAsset) = .empty;
    var nodes: std.ArrayList(*vk_engine.scene.Node) = .empty;
    var images: std.ArrayList(vk_engine.AllocatedImage) = .empty;
    var materials: std.ArrayList(*GltfMaterial) = .empty;

    // load all textures
    for (gltf.data.images) |_| {
        try images.append(gpa, error_checkerboard_image);
    }

    // create buffer to hold the material data
    scene.material_data_buffer = try .create(
        device_ctx.vma_allocator,
        @sizeOf(vk_engine.GltfMetallicRoughness.MaterialConstants) * gltf.data.materials.len,
        .{ .uniform_buffer_bit = true },
        .cpu_to_gpu,
    );

    {
        const scene_material_constants = try scene.material_data_buffer.map(device_ctx.vma_allocator, vk_engine.GltfMetallicRoughness.MaterialConstants);
        defer scene.material_data_buffer.unMap(device_ctx.vma_allocator);

        for (gltf.data.materials, 0..) |material, i| {
            const new_mat = try gpa.create(GltfMaterial);

            try materials.append(gpa, new_mat);
            try scene.materials.put(gpa, material.name.?, new_mat); // TODO: handle null

            const constants: vk_engine.GltfMetallicRoughness.MaterialConstants = .{
                .color_factors = material.metallic_roughness.base_color_factor,
                .metal_rough_factors = .{
                    material.metallic_roughness.metallic_factor,
                    material.metallic_roughness.roughness_factor,
                    0,
                    0,
                },
            };

            // write material parameters to buffer
            scene_material_constants[i] = constants;

            const pass_type: vk_engine.MaterialPass = switch (material.alpha_mode) {
                .blend => .transparent,
                else => .main_color,
            };

            // default the material textures
            var material_resources: vk_engine.GltfMetallicRoughness.MaterialResources = .{
                .color_image = white_image,
                .color_sampler = default_sampler_linear,
                .metal_rough_image = white_image,
                .metal_rough_sampler = default_sampler_linear,
                .data_buffer = .null_handle,
                .data_buffer_offset = 0,
            };

            // set the uniform buffer for the material data
            material_resources.data_buffer = scene.material_data_buffer.buffer;
            material_resources.data_buffer_offset = @intCast(i * @sizeOf(vk_engine.GltfMetallicRoughness.MaterialConstants));
            // grab textures from gltf file
            if (material.metallic_roughness.base_color_texture != null) {
                const image = gltf.data.textures[material.metallic_roughness.base_color_texture.?.index].source.?;
                const sampler = gltf.data.textures[material.metallic_roughness.base_color_texture.?.index].sampler.?;

                material_resources.color_image = images.items[image];
                material_resources.color_sampler = scene.samplers.items[sampler];
            }
            // build material
            new_mat.data = try metal_rough_material.writeMaterial(gpa, scratch, device_ctx.device, pass_type, &material_resources, &scene.descriptor_pool);
        }
    }

    // use the same vectors for all meshes so that the memory doesnt reallocate as
    // // often
    var indices: std.ArrayList(u32) = .empty;
    var vertices: std.ArrayList(vk_engine.Vertex) = .empty;

    for (gltf.data.meshes) |mesh| {
        defer indices.clearRetainingCapacity();
        defer vertices.clearRetainingCapacity();

        var new_mesh = try gpa.create(MeshAsset);
        new_mesh.* = .{
            .name = "",
            .surfaces = .empty,
            .mesh_buffers = undefined,
        };
        try meshes.append(gpa, new_mesh);

        try scene.meshes.put(gpa, mesh.name.?, new_mesh);
        new_mesh.name = mesh.name.?;

        for (mesh.primitives) |primitive| {
            const vertex_indices = gltf.data.accessors[primitive.indices.?];

            var new_surface: GeoSurface = .{
                .start_index = @intCast(indices.items.len),
                .count = @intCast(vertex_indices.count),
                .material = null,
            };

            const initial_vertex: u32 = @intCast(vertices.items.len);

            var vert_idx_it = vertex_indices.iterator(u16, &gltf, gltf.glb_binary.?);
            while (vert_idx_it.next()) |idx| {
                assert(idx.len == 1);
                try indices.append(gpa, initial_vertex + idx[0]);
            }

            for (primitive.attributes) |attribute| {
                switch (attribute) {
                    .position => |idx| { // TODO: this must be the first to be loaded, should I check for that, should I enforce it by iterating the attributes and finding it? not in example here: https://github.com/kooparse/zgltf
                        const accessor = gltf.data.accessors[idx];
                        var it = accessor.iterator(f32, &gltf, gltf.glb_binary.?);
                        while (it.next()) |v| {
                            assert(v.len == 3);
                            try vertices.append(gpa, .{
                                .position = v[0..3].*,
                                .normal = .{ 1, 0, 0 },
                                .color = @splat(1),
                                .uv_x = 0,
                                .uv_y = 0,
                            });
                        }
                    },
                    .normal => |idx| {
                        const accessor = gltf.data.accessors[idx];
                        var it = accessor.iterator(f32, &gltf, gltf.glb_binary.?);
                        var i: u32 = 0;
                        while (it.next()) |n| : (i += 1) {
                            assert(n.len == 3);
                            vertices.items[initial_vertex + i].normal = n[0..3].*;
                        }
                    },
                    .texcoord => |idx| {
                        const accessor = gltf.data.accessors[idx];
                        var it = accessor.iterator(f32, &gltf, gltf.glb_binary.?);
                        var i: u32 = 0;
                        while (it.next()) |tc| : (i += 1) {
                            assert(tc.len == 2);
                            vertices.items[initial_vertex + i].uv_x = tc[0];
                            vertices.items[initial_vertex + i].uv_y = tc[1];
                        }
                    },
                    .color => |idx| {
                        const accessor = gltf.data.accessors[idx];
                        var it = accessor.iterator(f32, &gltf, gltf.glb_binary.?);
                        var i: u32 = 0;
                        while (it.next()) |c| : (i += 1) {
                            assert(c.len == 4);
                            vertices.items[initial_vertex + i].color = c[0..4].*;
                        }
                    },
                    else => {
                        std.log.warn("attribute not handled when loading mesh from gltf file: {s}", .{@tagName(attribute)});
                    },
                }
            }

            if (primitive.material) |idx| {
                new_surface.material = materials.items[idx];
            } else {
                new_surface.material = materials.items[0];
            }

            try new_mesh.surfaces.append(gpa, new_surface);
        }

        new_mesh.mesh_buffers = try vk_engine.Engine.uploadMesh(device_ctx, imm, indices.items, vertices.items);
    }

    // load all nodes and their meshes
    for (gltf.data.nodes) |node| {
        const new_node = try gpa.create(vk_engine.scene.Node);

        new_node.* = .{
            .local_transform = .identity,
            .world_transform = .identity,
        };

        // find if the node has a mesh, and if it does hook it to the mesh pointer and allocate it with the meshnode class
        if (node.mesh) |m| {
            new_node.mesh = meshes.items[m].*;
        } else {
            new_node.mesh = null;
        }

        try nodes.append(gpa, new_node);
        try scene.nodes.put(gpa, node.name.?, new_node);

        if (node.matrix) |mat| {
            new_node.local_transform.items = @bitCast(mat);
        } else {
            var transform: vk_engine.Mat4 = .identity;
            transform.selfTranslate(node.translation);
            const rotation: zla.Quat(f32) = .{ .v = node.rotation };
            transform.selfMul(rotation.toMat4(.cm));
            transform.selfScale(node.scale);
            new_node.local_transform = transform;
        }
    }

    // run loop again to setup transform hierarchy
    for (gltf.data.nodes, 0..) |node, i| {
        const scene_node = nodes.items[i];

        for (node.children) |child| {
            try scene_node.children.append(gpa, nodes.items[child]);
            nodes.items[child].parent = scene_node;
        }
    }

    // find the top nodes, with no parents
    for (nodes.items) |node| {
        if (node.parent == null) {
            try scene.top_nodes.append(gpa, node);
            node.refreshTransform(.identity);
        }
    }

    return scene;
}

fn extractFilter(filter: Gltf.MinFilter) vk.Filter {
    return switch (filter) {
        .nearest, .nearest_mipmap_nearest, .nearest_mipmap_linear => .nearest,
        .linear, .linear_mipmap_nearest, .linear_mipmap_linear => .linear,
    };
}

fn extractMipmapMode(filter: Gltf.MinFilter) vk.SamplerMipmapMode {
    return switch (filter) {
        .nearest_mipmap_nearest, .linear_mipmap_nearest => return .nearest,
        .nearest_mipmap_linear, .linear_mipmap_linear => return .linear,
        else => .linear,
    };
}

const std = @import("std");
const zla = @import("zla");
const vk = @import("vulkan");
const Scratch = @import("scratch_allocator");
const Gltf = @import("gltf").Gltf;
const vk_engine = @import("vk_engine.zig");
const LoadedGltf = vk_engine.scene.LoadedGltf;
const assert = std.debug.assert;
