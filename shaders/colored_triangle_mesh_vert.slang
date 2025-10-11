struct VSOutput {
    float4 position : SV_Position;
    float3 color : COLOR0;
    float2 uv : TEXCOORD0;
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
    uniform Constants PushConstants,
    uint vertexID: SV_VulkanVertexID
) {
    Vertex v = PushConstants.vertices[vertexID];

    VSOutput output;
    output.position = mul(PushConstants.render_matrix, float4(v.position, 1.0f));
    output.color = v.color.xyz;
    output.uv = float2(v.uv_x, v.uv_y);

    return output;
}
