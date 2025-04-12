// Shader ported from https://www.shadertoy.com/view/XdlSDs.
// Author: dynamite
// License: CC BY-NC-SA 3.0 https://creativecommons.org/licenses/by-nc-sa/3.0/deed.en

[shader("compute")]
[numthreads(16, 16, 1)]
void main(
    uniform float4 background_color,
    uniform float time,
    RWTexture2D<float4> gImage,
    uint3 globalID: SV_DispatchThreadID
) {
    int2 texelCoord = int2(globalID.xy);

    int2 screenSize;
    gImage.GetDimensions(screenSize.x, screenSize.y);

    if(texelCoord.x >= screenSize.x || texelCoord.y >= screenSize.y) {
        return;
    }

    float2 p = (float2(globalID.xy) * 2.0f - float2(screenSize)) / float(screenSize.y);
    float tau = 3.14159265359f * 2.0f;
    float a = atan2(p.x, p.y);
    float r = length(p) * 0.75f;
    float2 uv = float2(a / tau, r);

    float xCol = (uv.x - time / 3.0f) * 3.0f;
    xCol = fmod(abs(xCol), 3.0f);
    float3 horColour = float3(0.25f, 0.25f, 0.25f);
    
    if (xCol < 1.0f) {
        horColour.r += 1.0f - xCol;
        horColour.g += xCol;
    }
    else if (xCol < 2.0f) {
        xCol -= 1.0f;
        horColour.g += 1.0f - xCol;
        horColour.b += xCol;
    }
    else {
        xCol -= 2.0f;
        horColour.b += 1.0f - xCol;
        horColour.r += xCol;
    }

    uv = (2.0f * uv) - 1.0f;
    float beamWidth = (0.7f + 
        0.5f * cos(uv.x * 10.0f * tau * 0.15f * clamp(floor(5.0f + 10.0f * cos(time)), 0.0f, 10.0f)))
        * abs(1.0f / (30.0f * uv.y));
    float3 horBeam = float3(beamWidth, beamWidth, beamWidth);

    float4 circle_color = float4(linear_from_srgb(horBeam * horColour), 1.0f);

    gImage[texelCoord] = 1 - (1 - background_color) * (1 - circle_color) ;
}

float3 linear_from_srgb(float3 srgb) {
    bool3 cutoff = srgb <= 0.04045f;
    float3 lower = srgb / 12.92f;
    float3 higher = pow((srgb + 0.055f) / 1.055f, 2.4f);
    return lerp(higher, lower, (float3)cutoff);
}