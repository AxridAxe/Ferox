#pragma once
#include "Shader.h"
#include "Camera.h"
#include "Chunk.h"
#include "TextureAtlas.h"
#include "Inventory.h"
#include <glm/glm.hpp>
#include <vector>
#include <memory>

class World;
class Player;

class Renderer {
public:
    Renderer();
    ~Renderer();

    void draw(const Camera& camera, int screenW, int screenH, World& world, float dayFactor = 1.0f);
    void drawCrosshair(int screenW, int screenH);
    void drawPauseOverlay();
    void drawHotbar(int screenW, int screenH, const Inventory& inventory);
    void drawDebug(int screenW, int screenH, World& world, Player& player, const Camera& camera, int fps, float gameTime);

private:
    Shader       m_shader;
    Shader       m_overlayShader;
    TextureAtlas m_atlas;
    GLuint m_overlayVAO = 0;
    GLuint m_overlayVBO = 0;
    GLuint m_crosshairVAO = 0;
    GLuint m_crosshairVBO = 0;
    GLuint m_hotbarVAO = 0;
    GLuint m_hotbarVBO = 0;

    bool isChunkInFrustum(const glm::vec3& chunkMin, const glm::vec3& chunkMax, const glm::mat4& viewProj);
};
