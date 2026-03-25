#define GL_CHECK(ctx) \
    do { \
        GLenum err = glGetError(); \
        if (err != GL_NO_ERROR) { \
            std::cerr << "[GL ERROR] " << err << " at " << __FILE__ << ":" << __LINE__ << " (" << ctx << ")" << std::endl; \
        } \
    } while(0)

#include "Renderer.h"
#include "World.h"
#include "Player.h"
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <iostream>

Renderer::Renderer()
    : m_shader("resources/shaders/chunk.vert", "resources/shaders/chunk.frag")
    , m_overlayShader("resources/shaders/overlay.vert", "resources/shaders/overlay.frag")
    , m_atlas({
        "resources/textures/grass_block_top.png",
        "resources/textures/grass_block_side.png",
        "resources/textures/dirt.png",
        "resources/textures/stone.png",
        "resources/textures/cobblestone.png",
        "resources/textures/oak_planks.png",
        "resources/textures/sand.png",
        "resources/textures/water_still.png",
        "resources/textures/gravel.png",
      })
{
    float verts[] = {
        -1.0f,  1.0f,
        -1.0f, -1.0f,
         1.0f, -1.0f,
        -1.0f,  1.0f,
         1.0f, -1.0f,
         1.0f,  1.0f,
    };
    glGenVertexArrays(1, &m_overlayVAO);
    glGenBuffers(1, &m_overlayVBO);
    glBindVertexArray(m_overlayVAO);
    glBindBuffer(GL_ARRAY_BUFFER, m_overlayVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), nullptr);
    glBindVertexArray(0);

    glGenVertexArrays(1, &m_crosshairVAO);
    glGenBuffers(1, &m_crosshairVBO);
    glBindVertexArray(m_crosshairVAO);
    glBindBuffer(GL_ARRAY_BUFFER, m_crosshairVBO);
    glBufferData(GL_ARRAY_BUFFER, 8 * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), nullptr);
    glBindVertexArray(0);

    glGenVertexArrays(1, &m_hotbarVAO);
    glGenBuffers(1, &m_hotbarVBO);
    glBindVertexArray(m_hotbarVAO);
    glBindBuffer(GL_ARRAY_BUFFER, m_hotbarVBO);
    glBufferData(GL_ARRAY_BUFFER, 24 * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), nullptr);
    glBindVertexArray(0);
}

Renderer::~Renderer() {
    glDeleteBuffers(1, &m_overlayVBO);
    glDeleteVertexArrays(1, &m_overlayVAO);
    glDeleteBuffers(1, &m_crosshairVBO);
    glDeleteVertexArrays(1, &m_crosshairVAO);
    glDeleteBuffers(1, &m_hotbarVBO);
    glDeleteVertexArrays(1, &m_hotbarVAO);
}

void Renderer::draw(const Camera& camera, int screenW, int screenH, World& world, float dayFactor) {
    glm::mat4 view = camera.getView();
    glm::mat4 proj = camera.getProjection(static_cast<float>(screenW) / screenH);
    glm::mat4 viewProj = proj * view;

    m_atlas.bind(0);
    m_shader.use();
    m_shader.setInt("uTexArray", 0);
    m_shader.setMat4("view",       view);
    m_shader.setMat4("projection", proj);
    m_shader.setFloat("uDayFactor", dayFactor);

    GL_CHECK("Renderer::draw - shader setup");

    int renderDist = 16;
    int pcx = (int)std::floor(camera.position.x / CHUNK_W);
    int pcz = (int)std::floor(camera.position.z / CHUNK_D);

    for (auto& [key, chunk] : world.chunks()) {
        int cx = (int)(int32_t)(key >> 32);
        int cz = (int)(int32_t)(key & 0xFFFFFFFF);
        if (std::abs(cx - pcx) > renderDist || std::abs(cz - pcz) > renderDist) continue;

        glm::vec3 chunkPos = chunk->worldPos();
        glm::vec3 chunkMin = chunkPos;
        glm::vec3 chunkMax = chunkPos + glm::vec3(CHUNK_W, CHUNK_H, CHUNK_D);

        if (!isChunkInFrustum(chunkMin, chunkMax, viewProj)) continue;
        
        if (chunk->indexCount() == 0) continue;
        
        glm::mat4 model = glm::translate(glm::mat4(1.0f), chunkPos);
        m_shader.setMat4("model", model);
        chunk->draw();
        GL_CHECK("Renderer::draw - chunk draw");
    }
}

bool Renderer::isChunkInFrustum(const glm::vec3& chunkMin, const glm::vec3& chunkMax, const glm::mat4& viewProj) {
    glm::vec4 corners[8] = {
        glm::vec4(chunkMin.x, chunkMin.y, chunkMin.z, 1.0f),
        glm::vec4(chunkMax.x, chunkMin.y, chunkMin.z, 1.0f),
        glm::vec4(chunkMin.x, chunkMax.y, chunkMin.z, 1.0f),
        glm::vec4(chunkMax.x, chunkMax.y, chunkMin.z, 1.0f),
        glm::vec4(chunkMin.x, chunkMin.y, chunkMax.z, 1.0f),
        glm::vec4(chunkMax.x, chunkMin.y, chunkMax.z, 1.0f),
        glm::vec4(chunkMin.x, chunkMax.y, chunkMax.z, 1.0f),
        glm::vec4(chunkMax.x, chunkMax.y, chunkMax.z, 1.0f),
    };

    for (int i = 0; i < 8; ++i) {
        glm::vec4 clip = viewProj * corners[i];
        if (clip.w > 0 && 
            clip.x >= -clip.w && clip.x <= clip.w &&
            clip.y >= -clip.w && clip.y <= clip.w &&
            clip.z >= -clip.w && clip.z <= clip.w) {
            return true;
        }
    }
    return false;
}

void Renderer::drawCrosshair(int screenW, int screenH) {
    float size = 12.0f;
    float sx = size / screenW;
    float sy = size / screenH;

    float lines[] = {
        -sx,  0.0f,
         sx,  0.0f,
         0.0f, -sy,
         0.0f,  sy,
    };

    glDisable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glLineWidth(2.0f);

    m_overlayShader.use();
    m_overlayShader.setVec4("uColor", glm::vec4(1.0f, 1.0f, 1.0f, 0.85f));

    glBindVertexArray(m_crosshairVAO);
    glBindBuffer(GL_ARRAY_BUFFER, m_crosshairVBO);
    glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(lines), lines);
    glDrawArrays(GL_LINES, 0, 4);
    glBindVertexArray(0);

    glDisable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);
}

void Renderer::drawPauseOverlay() {
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDisable(GL_DEPTH_TEST);

    m_overlayShader.use();
    m_overlayShader.setVec4("uColor", glm::vec4(0.0f, 0.0f, 0.0f, 0.5f));
    glBindVertexArray(m_overlayVAO);
    glDrawArrays(GL_TRIANGLES, 0, 6);
    glBindVertexArray(0);

    glDisable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);
}

void Renderer::drawHotbar(int screenW, int screenH, const Inventory& inventory) {
    int slotCount = Inventory::HOTBAR_SIZE;
    float slotSize = 50.0f;
    float padding = 4.0f;
    float totalWidth = slotCount * slotSize + (slotCount - 1) * padding;
    float startX = (screenW - totalWidth) * 0.5f;
    float startY = (float)screenH - slotSize - 20.0f;

    glDisable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    m_overlayShader.use();

    for (int i = 0; i < slotCount; ++i) {
        float x = startX + i * (slotSize + padding);
        float y = startY;

        bool selected = (i == inventory.selectedSlot());

        float x1 = x / screenW * 2.0f - 1.0f;
        float y1 = 1.0f - y / screenH * 2.0f;
        float x2 = (x + slotSize) / screenW * 2.0f - 1.0f;
        float y2 = 1.0f - (y + slotSize) / screenH * 2.0f;

        float verts[] = {
            x1, y1, x2, y1, x2, y2,
            x1, y1, x2, y2, x1, y2,
        };

        if (selected) {
            m_overlayShader.setVec4("uColor", glm::vec4(1.0f, 1.0f, 1.0f, 0.3f));
        } else {
            m_overlayShader.setVec4("uColor", glm::vec4(0.2f, 0.2f, 0.2f, 0.7f));
        }

        glBindVertexArray(m_hotbarVAO);
        glBindBuffer(GL_ARRAY_BUFFER, m_hotbarVBO);
        glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(verts), verts);
        glDrawArrays(GL_TRIANGLES, 0, 6);
    }

    glBindVertexArray(0);
    glDisable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);
}

void Renderer::drawDebug(int screenW, int screenH, World& world, Player& player, const Camera& camera, int fps, float gameTime) {
    int lines = 12;
    float lineHeight = 14.0f;
    float startY = 5.0f;
    float startX = 5.0f;

    std::ostringstream ss;
    ss << std::fixed << std::setprecision(2);
    ss << "Ferox 1.0\n";
    ss << "FPS: " << fps << "\n";
    ss << "XYZ: " << player.position.x << " / " << player.position.y << " / " << player.position.z << "\n";
    ss << "Chunk: " << (int)(player.position.x / CHUNK_W) << " " << (int)(player.position.z / CHUNK_D) << "\n";
    ss << "Facing: " << camera.yaw() << " " << camera.pitch() << "\n";
    ss << "Chunks loaded: " << world.chunks().size() << "\n";
    ss << "Day time: " << (int)(gameTime * 1000.0f) % 24000 << " / 24000\n";
    ss << "Height: " << world.getSurfaceHeight((int)player.position.x, (int)player.position.z) << "\n";
    ss << "Velocity: " << player.vel.x << " " << player.vel.y << " " << player.vel.z << "\n";
    ss << "On ground: " << (player.onGround ? "Yes" : "No") << "\n";
    ss << "Keys: F3=Debug V=Camera K=FreeCam\n";
    ss << "Mouse: LMB=Break RMB=Place\n";

    std::string text = ss.str();

    glDisable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    m_overlayShader.use();
    m_overlayShader.setVec4("uColor", glm::vec4(0.0f, 0.0f, 0.0f, 0.5f));

    float bgWidth = 180.0f;
    float bgHeight = lines * lineHeight + 8.0f;
    float bgVerts[] = {
        (startX - 2) / screenW * 2.0f - 1.0f, 1.0f - (startY - 2) / screenH * 2.0f,
        (startX + bgWidth) / screenW * 2.0f - 1.0f, 1.0f - (startY - 2) / screenH * 2.0f,
        (startX + bgWidth) / screenW * 2.0f - 1.0f, 1.0f - (startY + bgHeight) / screenH * 2.0f,
        (startX - 2) / screenW * 2.0f - 1.0f, 1.0f - (startY - 2) / screenH * 2.0f,
        (startX + bgWidth) / screenW * 2.0f - 1.0f, 1.0f - (startY + bgHeight) / screenH * 2.0f,
        (startX - 2) / screenW * 2.0f - 1.0f, 1.0f - (startY + bgHeight) / screenH * 2.0f,
    };

    glBindVertexArray(m_hotbarVAO);
    glBindBuffer(GL_ARRAY_BUFFER, m_hotbarVBO);
    glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(bgVerts), bgVerts);
    glDrawArrays(GL_TRIANGLES, 0, 6);

    glDisable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);
}
