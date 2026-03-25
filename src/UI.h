#pragma once
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <GLFW/glfw3.h>
#include "GameState.h"

extern int g_renderDistance;

class UI {
public:
    UI(GLFWwindow* window);
    ~UI();

    void beginFrame();
    void endFrame();

    GameState renderMainMenu(GameState current);
    GameState renderPauseMenu(GameState current);
    GameState renderDevMenu(GameState current, size_t chunksLoaded, uint32_t worldSeed);

private:
    GLFWwindow* m_window;
    void applyStyle();
};
