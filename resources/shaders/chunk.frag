#version 330 core

in vec2  fragUV;
in float fragTexIndex;
in float fragShade;
in float fragDayFactor;

uniform sampler2DArray uTexArray;

out vec4 outColor;

void main() {
    vec4 tex = texture(uTexArray, vec3(fragUV, fragTexIndex));
    vec3 color = tex.rgb;

    if (fragTexIndex < 0.5)
        color *= vec3(0.47, 0.75, 0.23);

    float ambient = 0.3 + 0.7 * fragDayFactor;
    float nightDim = (fragDayFactor < 0.15) ? 0.05 : 1.0;
    outColor = vec4(color * fragShade * ambient * nightDim, tex.a);
}
