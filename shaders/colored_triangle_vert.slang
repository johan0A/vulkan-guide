struct VSOutput {
    float4 position : SV_POSITION;
    float3 color : COLOR;
};

[shader("vertex")]
VSOutput main(
    uint vertexID: SV_VertexID,
) {
    float scale = 0.3;
    var offset = float3(0.0, -0.05, 0.0);
    
    // const array of positions for the triangle
    const float3 positions[3] = {
        float3(1.0, 1.0, 0.0) * scale + offset,
        float3(-1.0, 1.0, 0.0) * scale + offset,
        float3(0.0, -1.0, 0.0) * scale + offset,
    };

    // const array of colors for the triangle
    const float3 colors[3] = {
        float3(1.0, 0.0, 0.0), // red
        float3(0.0, 1.0, 0.0), // green
        float3(0.0, 0.0, 1.0)  // blue
    };

    VSOutput output;
    output.position = float4(positions[vertexID], 1.0);
    output.color = colors[vertexID];
    return output;
}