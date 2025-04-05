[[vk::binding(0, 0)]]
Sampler2D gSampler;

struct VertexOutput {
    float4 Color;
    float2 UV;
};

float3 linear_from_srgb(float3 srgb) {
    bool3 cutoff = srgb <= 0.04045f;
    float3 lower = srgb / 12.92f;
    float3 higher = pow((srgb + 0.055f) / 1.055f, 2.4f);
    return lerp(higher, lower, (float3)cutoff);
}

[shader("pixel")]
float4 main(VertexOutput In : COLOR0) : SV_Target0 {
    float4 texColor = gSampler.Sample(In.UV);
    float4 color = In.Color * texColor;
    return float4(linear_from_srgb(color.rgb), color.a);
}