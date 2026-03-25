#include "Window.h"
#include <stdexcept>

Window::Window(int width, int height, const std::string& title)
    : m_width(width), m_height(height)
{
    if (!glfwInit())
        throw std::runtime_error("Failed to initialise GLFW");

    // Tell GLFW we want OpenGL 3.3 Core Profile
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    m_window = glfwCreateWindow(width, height, title.c_str(), nullptr, nullptr);
    if (!m_window) {
        glfwTerminate();
        throw std::runtime_error("Failed to create window");
    }

    glfwMakeContextCurrent(m_window);
    glfwSetWindowUserPointer(m_window, this);
    glfwSetFramebufferSizeCallback(m_window, framebufferCallback);

    // Load all OpenGL function pointers through GLAD
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
        throw std::runtime_error("Failed to initialise GLAD");

    glfwSwapInterval(0); // uncapped framerate

    glEnable(GL_DEPTH_TEST);  // draw closer faces on top of farther ones
    glEnable(GL_CULL_FACE);   // skip faces that point away from the camera
    glCullFace(GL_BACK);
}

Window::~Window() {
    glfwDestroyWindow(m_window);
    glfwTerminate();
}

bool Window::shouldClose() const { return glfwWindowShouldClose(m_window); }
void Window::swapBuffers()       { glfwSwapBuffers(m_window); }
void Window::pollEvents()        { glfwPollEvents(); }

void Window::framebufferCallback(GLFWwindow* win, int w, int h) {
    glViewport(0, 0, w, h);
    auto* self = static_cast<Window*>(glfwGetWindowUserPointer(win));
    self->m_width  = w;
    self->m_height = h;
}
