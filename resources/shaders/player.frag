#version 330 core

in vec3 fragColor;
in float fragShade;

uniform vec3 uPartColor;

out vec4 outColor;

void main() {
    outColor = vec4(uPartColor * fragShade, 1.0);
}
