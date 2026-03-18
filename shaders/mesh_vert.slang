#include "input_structures.slang"

struct VSOutput {
    float4 position : SV_Position;
    float3 normal : NORMAL;
    float3 color : COLOR;
    float2 uv : TEXCOORD;
};

[shader("vertex")]
VSOutput main(
    uniform Constants push_constants,
    uint vertex_id: SV_VulkanVertexID,
) {
    Vertex v = vertices[vertex_id];
    SceneData scene = scenes[push_constants.scene_data_index];
    GLTFMaterialData mat = materials[push_constants.material_index];

    float4 position = float4(v.position, 1.0f);

    VSOutput output;

    output.position = mul(scene.viewproj, mul(push_constants.render_matrix, position));
    output.normal = mul(push_constants.render_matrix, float4(v.normal, 0.f)).xyz;
    output.color = v.color.xyz * mat.color_factors.xyz;
    output.uv = float2(v.uv_x, v.uv_y);

    return output;
}