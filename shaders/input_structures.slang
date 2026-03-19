struct SceneData {
    float4x4 view;
    float4x4 proj;
    float4x4 viewproj;
    float4 ambient_color;
    float4 sunlight_direction;
    float4 sunlight_color;
};

struct GLTFMaterialData {
    float4 color_factors;
    float4 metal_rough_factors;
    uint color_texture;
    uint metal_rough_texture;
    uint2 _padding;
};

struct Vertex {
    float3 position;
    float uv_x;
    float3 normal;
    float uv_y;
    float4 color;
};

struct DrawData {
    float4x4 world_matrix;
    uint material_index;
    uint3 _padding;
};

[[vk::binding(0, 0)]]
Sampler2D textures[];

[[vk::binding(1, 0)]]
StructuredBuffer<GLTFMaterialData> materials;

[[vk::binding(2, 0)]]
StructuredBuffer<SceneData> scenes;

[[vk::binding(3, 0)]]
StructuredBuffer<Vertex> vertices;

[[vk::binding(4, 0)]] 
StructuredBuffer<DrawData> draw_data;

struct Constants {
    uint scene_data_index;
};