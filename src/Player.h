#pragma once
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include "Camera.h"
#include "Block.h"

class World;

class Player {
public:
    Player(glm::vec3 spawnPos);

    void update(float dt, GLFWwindow* window, Camera& camera, World& world);

    glm::vec3 position;    // feet position (blocks)
    glm::vec3 vel{0.0f};   // velocity in blocks/tick
    bool      onGround = false;
    bool      sneaking  = false;
    float     walkPhase = 0.0f;  // limb animation phase (radians)

private:
    static constexpr float TPS         = 20.0f;
    static constexpr float PLAYER_W    = 0.6f;
    static constexpr float PLAYER_H    = 1.8f;
    static constexpr float EYE_OFFSET  = 1.62f;
    static constexpr float SNEAK_EYE   = 1.27f;
    static constexpr float STEP_HEIGHT = 0.6f;

    float m_tickAccum = 0.0f;

    // For rendering interpolation
    glm::vec3 m_prevPos{0.0f};
    glm::vec3 m_tickPos{0.0f};

    void tick(glm::vec3 wish, bool sprinting, bool sneaking, bool spaceHeld, World& world);
    void resolveX(World& world);
    void resolveY(World& world);
    void resolveZ(World& world);
    bool hasGroundBelow(World& world);

    static bool isSolid(BlockType b);
};
