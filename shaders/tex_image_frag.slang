[shader("pixel")]
float4 fragmentMain(
    Sampler2D displayTexture,
    float3 color : COLOR,
    float2 uv : TEXCOORD,
) : SV_Target {
    return displayTexture.Sample(uv);
}
