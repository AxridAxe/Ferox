#include "PlayerModel.h"
#include <glm/gtc/matrix_transform.hpp>
#include <vector>
#include <cmath>

// Face shading (same as chunks)
static constexpr float SHADE_TOP    = 1.00f;
static constexpr float SHADE_BOT    = 0.50f;
static constexpr float SHADE_NS     = 0.80f;
static constexpr float SHADE_EW     = 0.65f;

// Append a colored axis-aligned box centered at origin with given half-extents.
// Vertex format: pos(3) + color(3) + shade(1) = 7 floats
void PlayerModel::appendBox(const glm::vec3& h, const glm::vec3& c,
                            std::vector<float>& v, std::vector<unsigned>& idx)
{
    unsigned base = (unsigned)(v.size() / 7);

    // 8 unique positions
    glm::vec3 corners[8] = {
        {-h.x, -h.y, -h.z}, { h.x, -h.y, -h.z},
        { h.x,  h.y, -h.z}, {-h.x,  h.y, -h.z},
        {-h.x, -h.y,  h.z}, { h.x, -h.y,  h.z},
        { h.x,  h.y,  h.z}, {-h.x,  h.y,  h.z},
    };

    // 6 faces: indices into corners[], shade
    struct Face { int v[4]; float shade; };
    Face faces[6] = {
        {{3,2,6,7}, SHADE_TOP},   // +Y top
        {{0,1,5,4}, SHADE_BOT},   // -Y bottom
        {{4,5,6,7}, SHADE_NS},    // +Z north
        {{1,0,3,2}, SHADE_NS},    // -Z south
        {{5,1,2,6}, SHADE_EW},    // +X east
        {{0,4,7,3}, SHADE_EW},    // -X west
    };

    for (auto& f : faces) {
        unsigned fb = base;
        for (int i = 0; i < 4; ++i) {
            auto& p = corners[f.v[i]];
            v.push_back(p.x); v.push_back(p.y); v.push_back(p.z);
            v.push_back(c.x); v.push_back(c.y); v.push_back(c.z);
            v.push_back(f.shade);
        }
        // Two triangles
        idx.push_back(fb);     idx.push_back(fb + 1); idx.push_back(fb + 2);
        idx.push_back(fb);     idx.push_back(fb + 2); idx.push_back(fb + 3);
        base += 4;
    }
}

PlayerModel::PlayerModel()
    : m_shader("resources/shaders/player.vert", "resources/shaders/player.frag")
{
    // Build a single unit box (half-extent 0.5) — we'll scale per body part
    std::vector<float> verts;
    std::vector<unsigned> indices;
    appendBox(glm::vec3(0.5f), glm::vec3(1.0f), verts, indices);
    m_indexCount = (unsigned)indices.size();

    glGenVertexArrays(1, &m_vao);
    glGenBuffers(1, &m_vbo);
    glGenBuffers(1, &m_ebo);

    glBindVertexArray(m_vao);

    glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
    glBufferData(GL_ARRAY_BUFFER, verts.size() * sizeof(float), verts.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned), indices.data(), GL_STATIC_DRAW);

    // pos
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 7 * sizeof(float), (void*)0);
    // color
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 7 * sizeof(float), (void*)(3 * sizeof(float)));
    // shade
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, 7 * sizeof(float), (void*)(6 * sizeof(float)));

    glBindVertexArray(0);
}

PlayerModel::~PlayerModel() {
    glDeleteBuffers(1, &m_ebo);
    glDeleteBuffers(1, &m_vbo);
    glDeleteVertexArrays(1, &m_vao);
}

void PlayerModel::drawBox(const glm::mat4& vp, const glm::mat4& modelMat) {
    m_shader.setMat4("model", modelMat);
    glDrawElements(GL_TRIANGLES, m_indexCount, GL_UNSIGNED_INT, nullptr);
}

void PlayerModel::draw(const Camera& camera, int screenW, int screenH,
                       glm::vec3 feetPos, float yaw, float walkPhase, bool isSneaking)
{
    glm::mat4 view = camera.getView();
    glm::mat4 proj = camera.getProjection((float)screenW / screenH);
    glm::mat4 vp = proj * view;

    m_shader.use();
    m_shader.setMat4("view", view);
    m_shader.setMat4("projection", proj);

    glBindVertexArray(m_vao);

    float yawRad = glm::radians(-yaw - 90.0f);

    // Limb swing
    float armSwing = std::sin(walkPhase) * 0.8f;  // radians
    float legSwing = std::sin(walkPhase) * 0.6f;

    // Sneak: lean forward
    float sneakLean = isSneaking ? 0.35f : 0.0f; // radians forward tilt
    float sneakDrop = isSneaking ? -0.25f : 0.0f; // lower body slightly

    // ── Body base transform (translate to feet, rotate by yaw) ──
    glm::mat4 base = glm::translate(glm::mat4(1.0f), feetPos + glm::vec3(0.0f, sneakDrop, 0.0f));
    base = glm::rotate(base, yawRad, glm::vec3(0, 1, 0));

    // ── Legs (bottom of body) ──
    // Each leg: 0.25 x 0.75 x 0.25 blocks, pivot at top (hip)
    // Hips at y = 0.75 from feet
    {
        // Right leg
        glm::mat4 m = base;
        m = glm::translate(m, glm::vec3(-0.125f, 0.75f, 0.0f)); // hip position
        m = glm::rotate(m, -legSwing, glm::vec3(1, 0, 0));
        m = glm::translate(m, glm::vec3(0.0f, -0.375f, 0.0f));  // center of leg
        m = glm::scale(m, glm::vec3(0.25f, 0.75f, 0.25f));
        m_shader.setVec3("uPartColor", glm::vec3(0.23f, 0.30f, 0.65f)); // blue pants
        drawBox(vp, m);
    }
    {
        // Left leg
        glm::mat4 m = base;
        m = glm::translate(m, glm::vec3(0.125f, 0.75f, 0.0f));
        m = glm::rotate(m, legSwing, glm::vec3(1, 0, 0));
        m = glm::translate(m, glm::vec3(0.0f, -0.375f, 0.0f));
        m = glm::scale(m, glm::vec3(0.25f, 0.75f, 0.25f));
        m_shader.setVec3("uPartColor", glm::vec3(0.23f, 0.30f, 0.65f));
        drawBox(vp, m);
    }

    // ── Body (torso) ──
    // 0.5 x 0.75 x 0.25 blocks, bottom at y=0.75
    {
        glm::mat4 m = base;
        m = glm::translate(m, glm::vec3(0.0f, 0.75f, 0.0f)); // hip pivot
        m = glm::rotate(m, sneakLean, glm::vec3(1, 0, 0));    // lean forward when sneaking
        glm::mat4 bodyBase = m; // save for head/arms

        m = glm::translate(m, glm::vec3(0.0f, 0.375f, 0.0f)); // center of torso
        m = glm::scale(m, glm::vec3(0.5f, 0.75f, 0.25f));
        // Teal shirt color
        m_shader.setVec3("uPartColor", glm::vec3(0.25f, 0.60f, 0.60f));
        drawBox(vp, m);

        // ── Head ──
        // 0.5 x 0.5 x 0.5, sits on top of body at y = 0.75 + 0.75 = 1.5 from feet
        {
            glm::mat4 hm = bodyBase;
            hm = glm::translate(hm, glm::vec3(0.0f, 0.75f + 0.25f, 0.0f));
            hm = glm::scale(hm, glm::vec3(0.5f, 0.5f, 0.5f));
            // Skin color
            m_shader.setVec3("uPartColor", glm::vec3(0.76f, 0.60f, 0.42f));
            drawBox(vp, hm);
        }

        // ── Arms ──
        // 0.25 x 0.75 x 0.25, pivot at shoulder (top of arm)
        {
            // Right arm
            glm::mat4 am = bodyBase;
            am = glm::translate(am, glm::vec3(-0.375f, 0.75f, 0.0f)); // shoulder
            am = glm::rotate(am, armSwing, glm::vec3(1, 0, 0));
            am = glm::translate(am, glm::vec3(0.0f, -0.375f, 0.0f));
            am = glm::scale(am, glm::vec3(0.25f, 0.75f, 0.25f));
            m_shader.setVec3("uPartColor", glm::vec3(0.25f, 0.60f, 0.60f));
            drawBox(vp, am);
        }
        {
            // Left arm
            glm::mat4 am = bodyBase;
            am = glm::translate(am, glm::vec3(0.375f, 0.75f, 0.0f));
            am = glm::rotate(am, -armSwing, glm::vec3(1, 0, 0));
            am = glm::translate(am, glm::vec3(0.0f, -0.375f, 0.0f));
            am = glm::scale(am, glm::vec3(0.25f, 0.75f, 0.25f));
            drawBox(vp, am);
        }
    }

    glBindVertexArray(0);
}
