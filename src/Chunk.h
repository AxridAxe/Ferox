#pragma once
#include <glad/glad.h>
#include <glm/glm.hpp>
#include <vector>
#include <cstdint>
#include "Block.h"

// A chunk is a fixed-size box of blocks.
// The world is made up of a grid of chunks.
static constexpr int CHUNK_W = 16; // width  (X axis)
static constexpr int CHUNK_H = 64; // height (Y axis)
static constexpr int CHUNK_D = 16; // depth  (Z axis)

class Chunk {
public:
    Chunk(int cx, int cz);
    ~Chunk();

    void     setBlock(int x, int y, int z, BlockType type);
    BlockType getBlock(int x, int y, int z) const;

    void buildMesh();
    void draw() const;

    glm::vec3 worldPos() const { return { (float)m_cx * CHUNK_W, 0.0f, (float)m_cz * CHUNK_D }; }

    bool dirty = true;
    int indexCount() const { return m_indexCount; }

    bool inBounds(int x, int y, int z) const;

    void addFace(std::vector<float>& verts, std::vector<uint32_t>& idx,
                int x, int y, int z, int faceDir, BlockType type);

private:
    int m_cx, m_cz;

    BlockType m_blocks[CHUNK_W][CHUNK_H][CHUNK_D]{};

    GLuint m_vao = 0;
    GLuint m_vbo = 0;
    GLuint m_ebo = 0;
    int    m_indexCount = 0;
};