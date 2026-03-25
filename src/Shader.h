#pragma once
#include <glad/glad.h>
#include <glm/glm.hpp>
#include <string>

class Shader {
public:
    Shader(const std::string& vertPath, const std::string& fragPath);
    ~Shader();

    void use() const;

    void setInt  (const std::string& name, int val)              const;
    void setFloat(const std::string& name, float val)              const;
    void setMat4 (const std::string& name, const glm::mat4& val)   const;
    void setVec3 (const std::string& name, const glm::vec3& val)   const;
    void setVec4 (const std::string& name, const glm::vec4& val)   const;

private:
    GLuint m_id = 0;

    static GLuint      compileShader(GLenum type, const std::string& src);
    static std::string loadFile(const std::string& path);
};
