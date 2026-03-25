#pragma once
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <string>

// Wraps a GLFW window and initialises OpenGL via GLAD.
class Window {
public:
    Window(int width, int height, const std::string& title);
    ~Window();

    bool shouldClose() const;
    void swapBuffers();
    void pollEvents();

    GLFWwindow* handle() const { return m_window; }
    int width()  const { return m_width; }
    int height() const { return m_height; }

private:
    GLFWwindow* m_window = nullptr;
    int m_width, m_height;

    static void framebufferCallback(GLFWwindow* win, int w, int h);
};
