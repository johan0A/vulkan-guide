[[vk::binding(0, 0)]]
RWTexture2D<float4> gImage;
[shader("compute")]
[numthreads(16, 16, 1)]
void main(
    uint3 globalID : SV_DispatchThreadID,
    uint3 localID : SV_GroupThreadID,
) {
    int2 texelCoord = int2(globalID.xy);
    int2 size;
    gImage.GetDimensions(size.x, size.y);
    if(texelCoord.x < size.x && texelCoord.y < size.y) {
        float4 color = float4(0.0, 0.0, 0.0, 1.0);
       
        if(localID.x != 0 && localID.y != 0) {
            color.x = float(texelCoord.x) / (size.x);
            color.y = float(texelCoord.y) / (size.y);
        }
        
        float2 center = (size.x / 2.0, size.y / 2.0);
        float radius = min(size.x, size.y) * 0.3;
        float2 diff = texelCoord - center;
        float dist = length(diff);
        
        if(dist < radius) {
            color = float4(1.0, 1.0, 1.0, 1.0);
        }
       
        gImage[texelCoord] = color;
    }
}