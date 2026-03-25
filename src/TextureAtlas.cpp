#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#include "TextureAtlas.h"
#include <stdexcept>
#include <vector>

TextureAtlas::TextureAtlas(const std::vector<std::string>& paths) {
    if (paths.empty()) return;

    int targetW = 16, targetH = 16;

    glGenTextures(1, &m_id);
    glBindTexture(GL_TEXTURE_2D_ARRAY, m_id);

    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    stbi_set_flip_vertically_on_load(true);

    glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_RGBA8, targetW, targetH,
                 static_cast<int>(paths.size()), 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

    for (int i = 0; i < (int)paths.size(); i++) {
        int w, h, ch;
        unsigned char* data = stbi_load(paths[i].c_str(), &w, &h, &ch, 4);
        if (!data) throw std::runtime_error("Failed to load texture: " + paths[i]);

        // If the image is taller than wide (animated strip), just use the first frame
        if (h > w) {
            // Upload only the bottom 16x16 (stb flips vertically, so bottom = first frame)
            glTexSubImage3D(GL_TEXTURE_2D_ARRAY, 0, 0, 0, i, targetW, targetH, 1,
                            GL_RGBA, GL_UNSIGNED_BYTE, data);
        } else {
            glTexSubImage3D(GL_TEXTURE_2D_ARRAY, 0, 0, 0, i, targetW, targetH, 1,
                            GL_RGBA, GL_UNSIGNED_BYTE, data);
        }
        stbi_image_free(data);
    }

    glBindTexture(GL_TEXTURE_2D_ARRAY, 0);
}

TextureAtlas::~TextureAtlas() {
    glDeleteTextures(1, &m_id);
}

void TextureAtlas::bind(int unit) const {
    glActiveTexture(GL_TEXTURE0 + unit);
    glBindTexture(GL_TEXTURE_2D_ARRAY, m_id);
}
