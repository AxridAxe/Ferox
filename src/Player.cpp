#include "Player.h"
#include "World.h"
#include <cmath>
#include <algorithm>

Player::Player(glm::vec3 spawnPos)
    : position(spawnPos), m_prevPos(spawnPos), m_tickPos(spawnPos) {}

bool Player::isSolid(BlockType b) {
    return b != BlockType::Air && b != BlockType::Water;
}

// ── Collision resolution ──

void Player::resolveX(World& world) {
    float half = PLAYER_W * 0.5f;
    int minY = (int)std::floor(position.y);
    int maxY = (int)std::floor(position.y + PLAYER_H);
    int minZ = (int)std::floor(position.z - half);
    int maxZ = (int)std::floor(position.z + half);

    if (vel.x > 0.0f) {
        int bx = (int)std::floor(position.x + half);
        for (int y = minY; y <= maxY; ++y)
            for (int z = minZ; z <= maxZ; ++z)
                if (isSolid(world.getBlock(bx, y, z))) {
                    position.x = bx - half - 0.001f;
                    vel.x = 0.0f;
                    return;
                }
    } else if (vel.x < 0.0f) {
        int bx = (int)std::floor(position.x - half);
        for (int y = minY; y <= maxY; ++y)
            for (int z = minZ; z <= maxZ; ++z)
                if (isSolid(world.getBlock(bx, y, z))) {
                    position.x = bx + 1.0f + half + 0.001f;
                    vel.x = 0.0f;
                    return;
                }
    }
}

void Player::resolveY(World& world) {
    float half = PLAYER_W * 0.5f;
    int minX = (int)std::floor(position.x - half);
    int maxX = (int)std::floor(position.x + half);
    int minZ = (int)std::floor(position.z - half);
    int maxZ = (int)std::floor(position.z + half);

    if (vel.y < 0.0f) {
        int by = (int)std::floor(position.y);
        for (int x = minX; x <= maxX; ++x)
            for (int z = minZ; z <= maxZ; ++z)
                if (isSolid(world.getBlock(x, by, z))) {
                    position.y = by + 1.0f;
                    vel.y = 0.0f;
                    onGround = true;
                    return;
                }
        onGround = false;
    } else if (vel.y > 0.0f) {
        int by = (int)std::floor(position.y + PLAYER_H);
        for (int x = minX; x <= maxX; ++x)
            for (int z = minZ; z <= maxZ; ++z)
                if (isSolid(world.getBlock(x, by, z))) {
                    position.y = by - PLAYER_H - 0.001f;
                    vel.y = 0.0f;
                    return;
                }
    }
}

void Player::resolveZ(World& world) {
    float half = PLAYER_W * 0.5f;
    int minY = (int)std::floor(position.y);
    int maxY = (int)std::floor(position.y + PLAYER_H);
    int minX = (int)std::floor(position.x - half);
    int maxX = (int)std::floor(position.x + half);

    if (vel.z > 0.0f) {
        int bz = (int)std::floor(position.z + half);
        for (int y = minY; y <= maxY; ++y)
            for (int x = minX; x <= maxX; ++x)
                if (isSolid(world.getBlock(x, y, bz))) {
                    position.z = bz - half - 0.001f;
                    vel.z = 0.0f;
                    return;
                }
    } else if (vel.z < 0.0f) {
        int bz = (int)std::floor(position.z - half);
        for (int y = minY; y <= maxY; ++y)
            for (int x = minX; x <= maxX; ++x)
                if (isSolid(world.getBlock(x, y, bz))) {
                    position.z = bz + 1.0f + half + 0.001f;
                    vel.z = 0.0f;
                    return;
                }
    }
}

bool Player::hasGroundBelow(World& world) {
    float half = PLAYER_W * 0.5f;
    int minX = (int)std::floor(position.x - half);
    int maxX = (int)std::floor(position.x + half);
    int minZ = (int)std::floor(position.z - half);
    int maxZ = (int)std::floor(position.z + half);
    int by   = (int)std::floor(position.y) - 1;

    for (int x = minX; x <= maxX; ++x)
        for (int z = minZ; z <= maxZ; ++z)
            if (isSolid(world.getBlock(x, by, z)))
                return true;
    return false;
}

// ── One MC tick — exact MC Java order of operations ──
// All velocity in blocks/tick.
//
// MC order (from LivingEntity.travel):
//   1. Jump check (before movement)
//   2. Calculate acceleration from input + moveSpeed + friction
//   3. Apply acceleration to velocity
//   4. Move entity (position += velocity, collisions)
//   5. Apply horizontal friction: vel.xz *= blockSlip * 0.91
//   6. Apply vertical drag: vel.y *= 0.98
//   7. Apply gravity: vel.y -= 0.08

void Player::tick(glm::vec3 wish, bool sprinting, bool isSneaking, bool spaceHeld, World& world) {
    sneaking = isSneaking;

    // 1. Jump
    if (spaceHeld && onGround) {
        if (sneaking) {
            // Crouch-jump: half-block height (~0.5 blocks)
            vel.y = 0.30f;
            onGround = false;
        } else {
            vel.y = 0.42f;
            onGround = false;

            // Sprint-jump boost: +0.2 in facing direction
            if (sprinting && glm::length(wish) > 0.001f) {
                vel.x += wish.x * 0.2f;
                vel.z += wish.z * 0.2f;
            }
        }
    }

    // 2-3. Horizontal acceleration
    // Sneak input multiplier: 0.3
    float inputScale = sneaking ? 0.3f : 1.0f;

    // MC ground accel formula:
    //   accel = moveSpeed * (0.21600002 / (friction^3))
    //   where friction = blockSlip * 0.91, moveSpeed = attribute (0.1 walk, 0.13 sprint)
    // MC air accel: flat 0.02, or 0.026 if sprinting
    float blockSlip = 0.6f;
    float friction = blockSlip * 0.91f;  // 0.546

    float accel;
    if (onGround) {
        float moveSpeed = sprinting ? 0.16f : 0.1f;
        float f3 = friction * friction * friction;
        accel = moveSpeed * (0.21600002f / f3);
    } else {
        accel = sprinting ? 0.026f : 0.02f;
    }

    if (glm::length(wish) > 0.001f) {
        vel.x += wish.x * inputScale * accel;
        vel.z += wish.z * inputScale * accel;
    }

    // 4. Move (position += velocity, resolve collisions per axis)
    // Sneak edge-clamping: prevent walking off edges while sneaking
    glm::vec3 preMove = position;

    position.x += vel.x;
    resolveX(world);

    if (sneaking && onGround) {
        if (!hasGroundBelow(world)) {
            position.x = preMove.x;
            vel.x = 0.0f;
        }
    }

    position.y += vel.y;
    resolveY(world);

    position.z += vel.z;
    resolveZ(world);

    if (sneaking && onGround) {
        if (!hasGroundBelow(world)) {
            position.z = preMove.z;
            vel.z = 0.0f;
        }
    }

    // 5. Horizontal friction
    float hDrag = onGround ? friction : 0.91f;
    vel.x *= hDrag;
    vel.z *= hDrag;

    // 6-7. Vertical: drag THEN gravity (MC order)
    vel.y *= 0.98f;
    vel.y -= 0.08f;

    // Walk animation phase
    float hSpeed = std::sqrt(vel.x * vel.x + vel.z * vel.z);
    if (onGround && hSpeed > 0.01f) {
        walkPhase += hSpeed * 8.0f; // scale for visual speed
    } else {
        walkPhase *= 0.7f; // decay to rest
    }
}

void Player::update(float dt, GLFWwindow* window, Camera& camera, World& world) {
    glm::vec3 forward = glm::normalize(glm::vec3(camera.front.x, 0.0f, camera.front.z));
    glm::vec3 right   = glm::normalize(glm::cross(forward, glm::vec3(0.0f, 1.0f, 0.0f)));

    glm::vec3 wish(0.0f);
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) wish += forward;
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) wish -= forward;
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) wish -= right;
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) wish += right;

    if (glm::length(wish) > 0.001f) wish = glm::normalize(wish);

    bool isSneaking = glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS;
    bool sprinting = glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS
                     && glm::length(wish) > 0.001f && !isSneaking;
    bool spaceHeld = glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS;

    // Fixed 20 TPS tick loop with interpolation
    m_tickAccum += dt;
    if (m_tickAccum > 0.25f) m_tickAccum = 0.25f;

    constexpr float TICK_DT = 1.0f / 20.0f;
    while (m_tickAccum >= TICK_DT) {
        m_tickAccum -= TICK_DT;
        m_prevPos = position;
        tick(wish, sprinting, isSneaking, spaceHeld, world);
        m_tickPos = position;
    }

    // Interpolate position between ticks for smooth rendering
    float alpha = m_tickAccum / TICK_DT;
    glm::vec3 renderPos = glm::mix(m_prevPos, m_tickPos, alpha);

    float eye = sneaking ? SNEAK_EYE : EYE_OFFSET;
    camera.position = renderPos + glm::vec3(0.0f, eye, 0.0f);
}
