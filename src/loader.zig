const std = @import("std");
const Gltf = @import("gltf").Gltf;
const vk_engine = @import("vk_engine.zig");
const assert = std.debug.assert;

pub const GeoSurface = struct {
    start_index: u32,
    count: u32,
};

pub const MeshAsset = struct {
    name: []const u8,

    surfaces: std.ArrayListUnmanaged(GeoSurface),
    mesh_buffers: vk_engine.GPUMeshBuffers,
};

pub fn loadGltfMeshes(
    arena: std.mem.Allocator,
    temp: std.mem.Allocator,
    device_ctx: vk_engine.Engine.DeviceContext,
    imm: vk_engine.Engine.ImmSubmit,
    filePath: []const u8,
) !std.ArrayListUnmanaged(MeshAsset) {
    std.log.info("Loading GLTF {s}", .{filePath});

    const file = try std.fs.cwd().openFile(filePath, .{});

    var gltf: Gltf = .init(temp);
    defer gltf.deinit();

    try gltf.parse(try file.readToEndAllocOptions(temp, 1e9, null, .@"4", null));

    var meshes: std.ArrayListUnmanaged(MeshAsset) = .empty;

    var indices: std.ArrayListUnmanaged(u32) = .empty;
    var vertices: std.ArrayListUnmanaged(vk_engine.Vertex) = .empty;

    for (gltf.data.meshes) |mesh| {
        defer indices.clearRetainingCapacity();
        defer vertices.clearRetainingCapacity();

        var new_mesh: MeshAsset = .{
            .name = try arena.dupe(u8, mesh.name.?),
            .surfaces = .empty,
            .mesh_buffers = undefined,
        };

        for (mesh.primitives) |primitive_| {
            const primitive: Gltf.Primitive = @as(Gltf.Primitive, primitive_); // FIXME: this is to fix bug with zls, should report it probably

            const new_surface: GeoSurface = .{
                .start_index = @intCast(indices.items.len),
                .count = @intCast(gltf.data.accessors[primitive.indices.?].count),
            };
            try new_mesh.surfaces.append(arena, new_surface);

            const initial_vertex: u32 = @intCast(vertices.items.len);

            // load indexes
            {
                var it = gltf.data.accessors[primitive.indices.?].iterator(u16, &gltf, gltf.glb_binary.?);
                while (it.next()) |idx| {
                    assert(idx.len == 1);
                    try indices.append(arena, initial_vertex + idx[0]);
                }
            }

            // load attributes
            for (primitive.attributes) |attribute| {
                switch (attribute) {
                    .position => |idx| { // TODO: this must be the first to be loaded, should I check for that, should I enforce it by iterating the attributes and finding it? not in example here: https://github.com/kooparse/zgltf
                        const accessor = gltf.data.accessors[idx];
                        var it = accessor.iterator(f32, &gltf, gltf.glb_binary.?);
                        while (it.next()) |v| {
                            assert(v.len == 3);
                            try vertices.append(arena, .{
                                .position = v[0..3].*,
                                .normal = .{ 1, 0, 0 },
                                .color = .{ 1, 1, 1, 1 },
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
                        std.log.info("attribute not handled when loading mesh from gltf file: {s}", .{@tagName(attribute)});
                    },
                }
            }
        }

        // display the vertex normals
        const OverrideColors = true;
        if (OverrideColors) {
            for (vertices.items) |*vtx| {
                vtx.color = .{ vtx.normal[0], vtx.normal[1], vtx.normal[2], 1 };
            }
        }

        new_mesh.mesh_buffers = try vk_engine.Engine.uploadMesh(device_ctx, imm, indices.items, vertices.items);

        try meshes.append(arena, new_mesh);
    }

    return meshes;
}
