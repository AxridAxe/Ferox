#include "Chunk.h"
#include <iostream>

static const float FACE_VERTS[6][4][3] = {
    {{ 0,1,0 }, { 0,1,1 }, { 1,1,1 }, { 1,1,0 }},
    {{ 0,0,1 }, { 0,0,0 }, { 1,0,0 }, { 1,0,1 }},
    {{ 1,0,1 }, { 1,1,1 }, { 0,1,1 }, { 0,0,1 }},
    {{ 0,0,0 }, { 0,1,0 }, { 1,1,0 }, { 1,0,0 }},
    {{ 1,0,0 }, { 1,1,0 }, { 1,1,1 }, { 1,0,1 }},
    {{ 0,0,1 }, { 0,1,1 }, { 0,1,0 }, { 0,0,0 }},
};

static const float FACE_UVS[4][2] = {
    { 0,0 }, { 0,1 }, { 1,1 }, { 1,0 }
};

static const int FACE_NEIGHBOR[6][3] = {
    { 0, 1, 0 }, { 0,-1, 0 },
    { 0, 0, 1 }, { 0, 0,-1 },
    { 1, 0, 0 }, {-1, 0, 0 },
};

Chunk::Chunk(int cx, int cz) : m_cx(cx), m_cz(cz), m_indexCount(0), m_vao(0), m_vbo(0), m_ebo(0) {
    glGenVertexArrays(1, &m_vao);
    glGenBuffers(1, &m_vbo);
    glGenBuffers(1, &m_ebo);
}

Chunk::~Chunk() {
    glDeleteVertexArrays(1, &m_vao);
    glDeleteBuffers(1, &m_vbo);
    glDeleteBuffers(1, &m_ebo);
}

bool Chunk::inBounds(int x, int y, int z) const {
    return x >= 0 && x < CHUNK_W && y >= 0 && y < CHUNK_H && z >= 0 && z < CHUNK_D;
}

void Chunk::setBlock(int x, int y, int z, BlockType type) {
    if (inBounds(x, y, z)) { m_blocks[x][y][z] = type; dirty = true; }
}

BlockType Chunk::getBlock(int x, int y, int z) const {
    if (!inBounds(x, y, z)) return BlockType::Air;
    return m_blocks[x][y][z];
}

void Chunk::addFace(std::vector<float>& verts, std::vector<uint32_t>& idx,
                    int x, int y, int z, int faceDir, BlockType type)
{
    float texIdx = static_cast<float>(blockTexture(type, faceDir));
    float shade  = blockShade(faceDir);

    // Each vertex: x y z  u v  texIndex  shade  = 7 floats
    uint32_t base = static_cast<uint32_t>(verts.size() / 7);

    for (int v = 0; v < 4; v++) {
        verts.push_back(x + FACE_VERTS[faceDir][v][0]);
        verts.push_back(y + FACE_VERTS[faceDir][v][1]);
        verts.push_back(z + FACE_VERTS[faceDir][v][2]);
        verts.push_back(FACE_UVS[v][0]);
        verts.push_back(FACE_UVS[v][1]);
        verts.push_back(texIdx);
        verts.push_back(shade);
    }

    idx.push_back(base + 0); idx.push_back(base + 1); idx.push_back(base + 2);
    idx.push_back(base + 0); idx.push_back(base + 2); idx.push_back(base + 3);
}

void Chunk::buildMesh() {
    std::vector<float>    verts;
    std::vector<uint32_t> idx;

    verts.reserve(CHUNK_W * CHUNK_H * CHUNK_D * 6 * 4 * 7);
    idx.reserve(CHUNK_W * CHUNK_H * CHUNK_D * 6 * 6);

    for (int x = 0; x < CHUNK_W; x++)
    for (int y = 0; y < CHUNK_H; y++)
    for (int z = 0; z < CHUNK_D; z++) {
        BlockType type = m_blocks[x][y][z];
        if (type == BlockType::Air) continue;

        for (int face = 0; face < 6; face++) {
            int nx = x + FACE_NEIGHBOR[face][0];
            int ny = y + FACE_NEIGHBOR[face][1];
            int nz = z + FACE_NEIGHBOR[face][2];
            BlockType neighbor = getBlock(nx, ny, nz);
            if (neighbor == BlockType::Air || (!isBlockOpaque(neighbor) && type != neighbor))
                addFace(verts, idx, x, y, z, face, type);
        }
    }

    m_indexCount = static_cast<int>(idx.size());
    if (m_indexCount == 0) {
        glBindVertexArray(m_vao);
        glBindVertexArray(0);
        return;
    }

    glBindVertexArray(m_vao);

    glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
    glBufferData(GL_ARRAY_BUFFER, verts.size() * sizeof(float), verts.data(), GL_DYNAMIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, idx.size() * sizeof(uint32_t), idx.data(), GL_DYNAMIC_DRAW);

    int stride = 7 * sizeof(float);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, stride, (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, stride, (void*)(5 * sizeof(float)));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, stride, (void*)(6 * sizeof(float)));
    glEnableVertexAttribArray(3);

    glBindVertexArray(0);
    dirty = false;
}

void Chunk::draw() const {
    if (m_indexCount == 0) return;
    glBindVertexArray(m_vao);
    glDrawElements(GL_TRIANGLES, m_indexCount, GL_UNSIGNED_INT, nullptr);
    glBindVertexArray(0);
}
