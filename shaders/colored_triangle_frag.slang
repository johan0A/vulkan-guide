[shader("pixel")]
float4 fragmentMain(
    float3 color,
) : SV_Target {
    return float4(color, 1.0);
}
