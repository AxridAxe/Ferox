#pragma once
#include <glad/glad.h>
#include <glm/glm.hpp>
#include <vector>
#include "Shader.h"
#include "Camera.h"

// Simple Steve-like box model for third-person rendering.
// All dimensions in blocks (1 block = 1 unit).
// Minecraft Steve proportions (pixels → blocks, 1 pixel = 1/16 block):
//   Head:  8x8x8   = 0.5 x 0.5 x 0.5
//   Body:  8x12x4  = 0.5 x 0.75 x 0.25
//   Arm:   4x12x4  = 0.25 x 0.75 x 0.25
//   Leg:   4x12x4  = 0.25 x 0.75 x 0.25

class PlayerModel {
public:
    PlayerModel();
    ~PlayerModel();

    // Draw the model at the given feet position, facing yaw (degrees).
    // walkPhase is used for limb animation (radians, 0 = standing).
    void draw(const Camera& camera, int screenW, int screenH,
              glm::vec3 feetPos, float yaw, float walkPhase, bool isSneaking);

private:
    Shader m_shader;
    GLuint m_vao = 0, m_vbo = 0, m_ebo = 0;
    unsigned m_indexCount = 0;

    struct BoxDef {
        glm::vec3 size;     // half-extents
        glm::vec3 offset;   // pivot offset from feet
        glm::vec3 color;
    };

    // Build a colored box mesh into the vertex/index buffers
    static void appendBox(const glm::vec3& halfSize, const glm::vec3& color,
                          std::vector<float>& verts, std::vector<unsigned>& indices);

    void drawBox(const glm::mat4& vp, const glm::mat4& modelMat);
};
