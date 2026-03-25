#include "Shader.h"
#include <glm/gtc/type_ptr.hpp>
#include <fstream>
#include <sstream>
#include <stdexcept>

Shader::Shader(const std::string& vertPath, const std::string& fragPath) {
    GLuint vert = compileShader(GL_VERTEX_SHADER,   loadFile(vertPath));
    GLuint frag = compileShader(GL_FRAGMENT_SHADER, loadFile(fragPath));

    m_id = glCreateProgram();
    glAttachShader(m_id, vert);
    glAttachShader(m_id, frag);
    glLinkProgram(m_id);

    GLint ok;
    glGetProgramiv(m_id, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[512];
        glGetProgramInfoLog(m_id, 512, nullptr, log);
        throw std::runtime_error(std::string("Shader link error:\n") + log);
    }

    // Once linked into a program the individual shaders aren't needed
    glDeleteShader(vert);
    glDeleteShader(frag);
}

Shader::~Shader() { glDeleteProgram(m_id); }

void Shader::use() const { glUseProgram(m_id); }

void Shader::setInt(const std::string& name, int val) const {
    glUniform1i(glGetUniformLocation(m_id, name.c_str()), val);
}

void Shader::setFloat(const std::string& name, float val) const {
    glUniform1f(glGetUniformLocation(m_id, name.c_str()), val);
}

void Shader::setMat4(const std::string& name, const glm::mat4& val) const {
    glUniformMatrix4fv(glGetUniformLocation(m_id, name.c_str()), 1, GL_FALSE, glm::value_ptr(val));
}

void Shader::setVec3(const std::string& name, const glm::vec3& val) const {
    glUniform3fv(glGetUniformLocation(m_id, name.c_str()), 1, glm::value_ptr(val));
}

void Shader::setVec4(const std::string& name, const glm::vec4& val) const {
    glUniform4fv(glGetUniformLocation(m_id, name.c_str()), 1, glm::value_ptr(val));
}

GLuint Shader::compileShader(GLenum type, const std::string& src) {
    GLuint shader = glCreateShader(type);
    const char* c = src.c_str();
    glShaderSource(shader, 1, &c, nullptr);
    glCompileShader(shader);

    GLint ok;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[512];
        glGetShaderInfoLog(shader, 512, nullptr, log);
        throw std::runtime_error(std::string("Shader compile error:\n") + log);
    }
    return shader;
}

std::string Shader::loadFile(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open())
        throw std::runtime_error("Cannot open shader file: " + path);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}
