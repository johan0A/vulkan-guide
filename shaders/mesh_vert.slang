#include "input_structures.slang"

struct VSOutput {
    float4 position : SV_Position;
    float3 normal : NORMAL;
    float3 color : COLOR;
    float2 uv : TEXCOORD;
};

struct Vertex {
    float3 position;
    float uv_x;
    float3 normal;
    float uv_y;
    float4 color;
};

struct Constants {
    float4x4 render_matrix;
    Vertex *vertices;
}

[shader("vertex")]
VSOutput main(
    uniform Constants push_constants,
    uint vertex_id: SV_VulkanVertexID,
) {
    Vertex v = push_constants.vertices[vertex_id];

    float4 position = float4(v.position, 1.0f);

    VSOutput output;

    output.position = mul(scene_data.viewproj, mul(push_constants.render_matrix, position));

    output.normal = mul(push_constants.render_matrix, float4(v.normal, 0.f)).xyz;
    output.color = v.color.xyz * material_data.color_factors.xyz;
    output.uv = float2(v.uv_x, v.uv_y);

    return output;
}
