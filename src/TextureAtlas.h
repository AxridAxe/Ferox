#pragma once
#include <glad/glad.h>
#include <string>
#include <vector>

class TextureAtlas {
public:
    // Loads each path as one layer of a 2D texture array.
    // All images must be the same size (Minecraft textures are 16x16).
    TextureAtlas(const std::vector<std::string>& paths);
    ~TextureAtlas();

    void bind(int unit = 0) const;

private:
    GLuint m_id = 0;
};
