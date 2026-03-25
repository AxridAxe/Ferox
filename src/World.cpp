#include "World.h"
#include <cmath>
#include <algorithm>
#include <limits>

World::World(uint32_t seed) : m_noise(seed), m_seed(seed) {}

void World::update(glm::vec3 playerPos, int renderDist) {
    int pcx = (int)std::floor(playerPos.x / CHUNK_W);
    int pcz = (int)std::floor(playerPos.z / CHUNK_D);

    for (int dx = -renderDist; dx <= renderDist; ++dx) {
        for (int dz = -renderDist; dz <= renderDist; ++dz) {
            int cx = pcx + dx, cz = pcz + dz;
            uint64_t k = key(cx, cz);
            
            if (!m_chunks.count(k)) {
                generateChunk(cx, cz);
            }
        }
    }

    int unload = renderDist + 2;
    for (auto it = m_chunks.begin(); it != m_chunks.end(); ) {
        int cx = (int)(int32_t)(it->first >> 32);
        int cz = (int)(int32_t)(it->first & 0xFFFFFFFF);
        if (std::abs(cx - pcx) > unload || std::abs(cz - pcz) > unload) {
            it = m_chunks.erase(it);
        } else {
            ++it;
        }
    }
}

BlockType World::getBlock(int wx, int wy, int wz) const {
    if (wy < 0 || wy >= CHUNK_H) return BlockType::Air;
    int cx = (int)std::floor((float)wx / CHUNK_W);
    int cz = (int)std::floor((float)wz / CHUNK_D);
    auto it = m_chunks.find(key(cx, cz));
    if (it == m_chunks.end()) return BlockType::Air;
    return it->second->getBlock(wx - cx * CHUNK_W, wy, wz - cz * CHUNK_D);
}

int World::getSurfaceHeight(int wx, int wz) const {
    for (int y = CHUNK_H - 1; y >= 0; --y) {
        BlockType b = getBlock(wx, y, wz);
        if (b != BlockType::Air && b != BlockType::Water)
            return y;
    }
    return 0;
}

void World::setBlock(int wx, int wy, int wz, BlockType type) {
    if (wy < 0 || wy >= CHUNK_H) return;
    int cx = (int)std::floor((float)wx / CHUNK_W);
    int cz = (int)std::floor((float)wz / CHUNK_D);
    auto it = m_chunks.find(key(cx, cz));
    if (it == m_chunks.end()) return;
    it->second->setBlock(wx - cx * CHUNK_W, wy, wz - cz * CHUNK_D, type);
}

RaycastHit World::raycast(glm::vec3 origin, glm::vec3 direction, float maxDist) {
    RaycastHit result;

    glm::vec3 dir = glm::normalize(direction);

    float t = 0.0f;
    glm::ivec3 currentBlock = glm::ivec3((int)std::floor(origin.x),
                                         (int)std::floor(origin.y),
                                         (int)std::floor(origin.z));

    glm::ivec3 step(
        dir.x > 0 ? 1 : (dir.x < 0 ? -1 : 0),
        dir.y > 0 ? 1 : (dir.y < 0 ? -1 : 0),
        dir.z > 0 ? 1 : (dir.z < 0 ? -1 : 0)
    );

    glm::vec3 tDelta(
        step.x != 0 ? std::abs(1.0f / dir.x) : std::numeric_limits<float>::max(),
        step.y != 0 ? std::abs(1.0f / dir.y) : std::numeric_limits<float>::max(),
        step.z != 0 ? std::abs(1.0f / dir.z) : std::numeric_limits<float>::max()
    );

    glm::vec3 originF(origin.x, origin.y, origin.z);
    glm::vec3 tMax(
        step.x > 0 ? (currentBlock.x + 1 - originF.x) / dir.x : (originF.x - currentBlock.x) / -dir.x,
        step.y > 0 ? (currentBlock.y + 1 - originF.y) / dir.y : (originF.y - currentBlock.y) / -dir.y,
        step.z > 0 ? (currentBlock.z + 1 - originF.z) / dir.z : (originF.z - currentBlock.z) / -dir.z
    );
    if (step.x > 0) tMax.x = std::abs(tMax.x);
    if (step.y > 0) tMax.y = std::abs(tMax.y);
    if (step.z > 0) tMax.z = std::abs(tMax.z);

    while (t < maxDist) {
        BlockType block = getBlock(currentBlock.x, currentBlock.y, currentBlock.z);
        if (block != BlockType::Air && block != BlockType::Water) {
            result.hit = true;
            result.x = currentBlock.x;
            result.y = currentBlock.y;
            result.z = currentBlock.z;

            glm::vec3 hitPoint = originF + dir * t;
            glm::vec3 diff = hitPoint - (glm::vec3(currentBlock) + 0.5f);
            const float eps = 0.001f;

            if (std::abs(diff.x) > std::abs(diff.y) && std::abs(diff.x) > std::abs(diff.z)) {
                result.faceX = -step.x;
            } else if (std::abs(diff.y) > std::abs(diff.z)) {
                result.faceY = -step.y;
            } else {
                result.faceZ = -step.z;
            }

            return result;
        }

        if (tMax.x < tMax.y && tMax.x < tMax.z) {
            t = tMax.x;
            currentBlock.x += step.x;
            tMax.x += std::abs(1.0f / dir.x);
        } else if (tMax.y < tMax.z) {
            t = tMax.y;
            currentBlock.y += step.y;
            tMax.y += std::abs(1.0f / dir.y);
        } else {
            t = tMax.z;
            currentBlock.z += step.z;
            tMax.z += std::abs(1.0f / dir.z);
        }
    }

    return result;
}

void World::generateChunk(int cx, int cz) {
    auto chunk = std::make_unique<Chunk>(cx, cz);

    for (int x = 0; x < CHUNK_W; ++x) {
        for (int z = 0; z < CHUNK_D; ++z) {
            float wx = (float)(cx * CHUNK_W + x);
            float wz = (float)(cz * CHUNK_D + z);

            float n = m_noise.fbm(wx * 0.005f, wz * 0.005f, 6, 0.5f, 2.0f);
            int height = (int)(40.0f + n * 30.0f);
            height = std::clamp(height, 1, CHUNK_H - 2);

            for (int y = 0; y < CHUNK_H; ++y) {
                BlockType type = BlockType::Air;

                if (y == 0) {
                    type = BlockType::Stone;
                } else if (y < height - 4) {
                    type = BlockType::Stone;
                } else if (y < height) {
                    type = BlockType::Dirt;
                } else if (y == height) {
                    type = (height <= 34) ? BlockType::Sand : BlockType::Grass;
                } else if (y <= 32 && height <= 32) {
                    type = BlockType::Water;
                }

                chunk->setBlock(x, y, z, type);
            }
        }
    }

    chunk->buildMesh();
    m_chunks[key(cx, cz)] = std::move(chunk);
}
