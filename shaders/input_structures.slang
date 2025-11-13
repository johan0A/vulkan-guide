struct SceneData {
    float4x4 view;
    float4x4 proj;
    float4x4 viewproj;
    float4 ambient_color;
    float4 sunlight_direction; // w for sun power
    float4 sunlight_color;
};

struct GLTFMaterialData {
    float4 color_factors;
    float4 metal_rough_factors;
};

[[vk::binding(0, 0)]]
ConstantBuffer<SceneData> scene_data;

[[vk::binding(0, 1)]]
ConstantBuffer<GLTFMaterialData> material_data;

[[vk::binding(1, 1)]]
Sampler2D color_tex;

[[vk::binding(2, 1)]]
Sampler2D metal_rough_tex;