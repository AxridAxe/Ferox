#pragma once
#include <unordered_map>
#include <memory>
#include <glm/glm.hpp>
#include "Chunk.h"
#include "Noise.h"

struct RaycastHit {
    bool hit = false;
    int x = 0, y = 0, z = 0;
    int faceX = 0, faceY = 0, faceZ = 0;
};

class World {
public:
    World(uint32_t seed = 1337);

    void update(glm::vec3 playerPos, int renderDist);
    BlockType getBlock(int wx, int wy, int wz) const;
    int getSurfaceHeight(int wx, int wz) const;
    void setBlock(int wx, int wy, int wz, BlockType type);
    RaycastHit raycast(glm::vec3 origin, glm::vec3 direction, float maxDist = 6.0f);

    std::unordered_map<uint64_t, std::unique_ptr<Chunk>>& chunks() { return m_chunks; }
    const std::unordered_map<uint64_t, std::unique_ptr<Chunk>>& chunks() const { return m_chunks; }
    uint32_t seed() const { return m_seed; }

    static uint64_t key(int cx, int cz) {
        return ((uint64_t)(uint32_t)cx << 32) | (uint32_t)cz;
    }

private:
    std::unordered_map<uint64_t, std::unique_ptr<Chunk>> m_chunks;
    Noise m_noise;
    uint32_t m_seed = 1337;

    void generateChunk(int cx, int cz);
};
