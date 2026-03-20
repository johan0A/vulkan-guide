#include "input_structures.slang"

struct VSOutput {
    float4 position : SV_Position;
    float3 normal : NORMAL;
    float3 color : COLOR;
    float2 uv : TEXCOORD;
    nointerpolation uint material_index : TEXCOORD1;
};

[shader("vertex")]
VSOutput main(
    uniform Constants push_constants,
    uint vertex_id: SV_VulkanVertexID,
    uint draw_id: SV_VulkanInstanceID,
) {
    SceneData scene = push_constants.scenes[push_constants.scene_data_index];
    DrawData draw = scene.draw_data[draw_id];
    Vertex v = scene.vertices[vertex_id];
    GLTFMaterialData mat = scene.materials[draw.material_index];

    float4 position = float4(v.position, 1.0f);

    VSOutput output;
    output.position = mul(scene.viewproj, mul(draw.world_matrix, position));
    output.normal = mul(draw.world_matrix, float4(v.normal, 0.f)).xyz;
    output.color = v.color.xyz * mat.color_factors.xyz;
    output.uv = float2(v.uv_x, v.uv_y);
    output.material_index = draw.material_index;
    return output;
}