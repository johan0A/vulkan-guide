[shader("compute")]
[numthreads(16, 16, 1)]
void main(
    uniform float4 background_color,
    RWTexture2D<float4> gImage,
    uint3 globalID: SV_DispatchThreadID
) {
    int2 texelCoord = int2(globalID.xy);
    gImage[texelCoord] = background_color;
}