#version 330 core

layout(location = 0) in vec3  aPos;
layout(location = 1) in vec2  aUV;
layout(location = 2) in float aTexIndex;
layout(location = 3) in float aShade;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform float uDayFactor = 1.0;

out vec2  fragUV;
out float fragTexIndex;
out float fragShade;
out float fragDayFactor;

void main() {
    gl_Position  = projection * view * model * vec4(aPos, 1.0);
    fragUV       = aUV;
    fragTexIndex = aTexIndex;
    fragShade    = aShade;
    fragDayFactor = uDayFactor;
}
