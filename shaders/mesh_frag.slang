#include "input_structures.slang"

[shader("pixel")]
float4 main(
    uniform Constants push_constants,
    float3 normal : NORMAL,
    float3 color : COLOR,
    float2 uv : TEXCOORD,
    nointerpolation uint material_index : TEXCOORD1,
) {
    SceneData scene = scenes[push_constants.scene_data_index];
    GLTFMaterialData mat = materials[material_index];

    float light_value = max(dot(normal, scene.sunlight_direction.xyz), 0.1f);

    float3 albedo = color * textures[NonUniformResourceIndex(mat.color_texture)].Sample(uv).xyz;
    float3 ambient = albedo * scene.ambient_color.xyz;

    return float4(albedo * light_value * scene.sunlight_color.w + ambient, 1.0f);
}