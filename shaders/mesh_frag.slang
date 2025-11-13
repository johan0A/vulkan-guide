#include "input_structures.slang"

[shader("pixel")]
float4 main(
    float3 normal: NORMAL,
    float3 color: COLOR,
    float2 uv: TEXCOORD,
) {
	float light_value = max(dot(normal, scene_data.sunlight_direction.xyz), 0.1f);

    float3 color = color * color_tex.Sample(uv).xyz;
    float3 ambient = color * scene_data.ambient_color.xyz;

    return float4(color * light_value * scene_data.sunlight_color.w + ambient, 1.0f);
}