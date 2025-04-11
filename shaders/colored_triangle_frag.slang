[shader("pixel")]
float4 fragmentMain(
    float3 color: COLOR,
) : SV_Target {
  return float4(color, 1.0);
}