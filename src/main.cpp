#include <memory>
#include <vector>
#include "Window.h"
#include "UI.h"
#include "Camera.h"
#include "Chunk.h"
#include "Renderer.h"
#include "PlayerModel.h"
#include "GameState.h"

extern int g_renderDistance;
#include "World.h"
#include "Player.h"
#include "Inventory.h"

static Camera*   g_camera = nullptr;
static GameState g_state  = GameState::MainMenu;
static Inventory* g_inventory = nullptr;
static int g_scrollAccum = 0;

static void onMouseMove(GLFWwindow*, double xpos, double ypos) {
    if (g_camera && g_state == GameState::Playing)
        g_camera->processMouseMovement(xpos, ypos);
}

static void onScroll(GLFWwindow*, double xoffset, double yoffset) {
    if (g_inventory && g_state == GameState::Playing) {
        g_scrollAccum += (int)yoffset;
    }
}

int main() {
    Window window(854, 480, "Ferox");
    Camera camera(glm::vec3(0.0f, 0.0f, 0.0f));
    g_camera = &camera;

    glfwSetCursorPosCallback(window.handle(), onMouseMove);
    glfwSetScrollCallback(window.handle(), onScroll);

    std::unique_ptr<World>     world;
    std::unique_ptr<Player>    player;
    std::unique_ptr<Inventory> inventory;

    Renderer    renderer;
    PlayerModel playerModel;
    UI          ui(window.handle());

    float lastTime    = 0.0f;
    bool  escWasDown  = false;
    bool  f11WasDown  = false;
    bool  kWasDown    = false;
    bool  freeCam     = false;
    bool  isFullscreen = false;
    int   savedX = 0, savedY = 0, savedW = 854, savedH = 480;
    float freeCamSpeed = 20.0f;
    bool  vWasDown    = false;
    bool  f3WasDown   = false;
    bool  showDebug   = false;
    bool  leftWasDown = false;
    bool  rightWasDown = false;
    bool  scrollWasDown = false;
    bool numWasDown[10] = {false};
    bool zeroWasDown = false;

    GameState pendingState = GameState::MainMenu;
    float gameTime = 0.0f;
    bool firstFrame = true;

    while (!window.shouldClose()) {
        float now = static_cast<float>(glfwGetTime());
        float dt  = now - lastTime;
        lastTime  = now;

        window.pollEvents();

        if (pendingState != g_state) {
            GameState oldState = g_state;
            g_state = pendingState;

            if (g_state == GameState::Playing) {
                glfwSetInputMode(window.handle(), GLFW_CURSOR, GLFW_CURSOR_DISABLED);
                camera.resetMouse();
            } else {
                glfwSetInputMode(window.handle(), GLFW_CURSOR, GLFW_CURSOR_NORMAL);
            }

            if (g_state == GameState::Playing && oldState == GameState::MainMenu) {
                world     = std::make_unique<World>();
                player    = std::make_unique<Player>(glm::vec3(32.0f, 50.0f, 32.0f));
                inventory = std::make_unique<Inventory>();
                g_inventory = inventory.get();
                world->update(player->position, 8);
                int spawnHeight = world->getSurfaceHeight(32, 32) + 2;
                player->position.y = (float)spawnHeight;
                camera.position = player->position + glm::vec3(0.0f, 1.62f, 0.0f);
                camera.perspective = CameraPerspective::FirstPerson;
                camera.resetMouse();
                gameTime = 0.0f;
                firstFrame = true;
            }

            if (g_state == GameState::MainMenu) {
                world.reset();
                player.reset();
                inventory.reset();
                g_inventory = nullptr;
                freeCam = false;
            }
        }

        if (g_state == GameState::Playing && world && player && inventory) {
            gameTime += dt * 0.1f;
        }

        bool f11Down = glfwGetKey(window.handle(), GLFW_KEY_F11) == GLFW_PRESS;
        if (f11Down && !f11WasDown) {
            GLFWmonitor* monitor = glfwGetPrimaryMonitor();
            const GLFWvidmode* mode = glfwGetVideoMode(monitor);
            if (!isFullscreen) {
                glfwGetWindowPos(window.handle(), &savedX, &savedY);
                savedW = window.width();
                savedH = window.height();
                glfwSetWindowMonitor(window.handle(), monitor, 0, 0, mode->width, mode->height, mode->refreshRate);
            } else {
                glfwSetWindowMonitor(window.handle(), nullptr, savedX, savedY, savedW, savedH, 0);
            }
            isFullscreen = !isFullscreen;
        }
        f11WasDown = f11Down;

        bool vDown = glfwGetKey(window.handle(), GLFW_KEY_V) == GLFW_PRESS;
        if (vDown && !vWasDown && g_state == GameState::Playing)
            camera.cyclePerspective();
        vWasDown = vDown;

        if (g_state == GameState::Playing)
            camera.setZoom(glfwGetKey(window.handle(), GLFW_KEY_C) == GLFW_PRESS);

        bool kDown = glfwGetKey(window.handle(), GLFW_KEY_K) == GLFW_PRESS;
        if (kDown && !kWasDown && g_state == GameState::Playing)
            freeCam = !freeCam;
        kWasDown = kDown;

        bool escDown = glfwGetKey(window.handle(), GLFW_KEY_ESCAPE) == GLFW_PRESS;
        if (escDown && !escWasDown) {
            if (g_state == GameState::Playing)
                pendingState = GameState::Paused;
            else if (g_state == GameState::Paused)
                pendingState = GameState::Playing;
            else if (g_state == GameState::DevMenu)
                pendingState = GameState::Playing;
        }
        escWasDown = escDown;

        bool f3Down = glfwGetKey(window.handle(), GLFW_KEY_F3) == GLFW_PRESS;
        if (f3Down && !f3WasDown && g_state == GameState::Playing)
            showDebug = !showDebug;
        f3WasDown = f3Down;

        bool zeroDown = glfwGetKey(window.handle(), GLFW_KEY_0) == GLFW_PRESS;
        // Dev menu disabled for stability testing
        // if (zeroDown && !zeroWasDown && g_state == GameState::Playing) {
        //     pendingState = GameState::DevMenu;
        // }
        zeroWasDown = zeroDown;

            if ((g_state == GameState::Playing || g_state == GameState::DevMenu) && world && player && inventory) {
                world->update(player->position, g_renderDistance);

                for (int i = 0; i < 9; ++i) {
                    bool numDown = glfwGetKey(window.handle(), GLFW_KEY_1 + i) == GLFW_PRESS;
                    if (numDown && !numWasDown[i]) {
                        inventory->selectSlot(i);
                    }
                    numWasDown[i] = numDown;
                }

                if (freeCam) {
                glm::vec3 move(0.0f);
                if (glfwGetKey(window.handle(), GLFW_KEY_W) == GLFW_PRESS) move += camera.front;
                if (glfwGetKey(window.handle(), GLFW_KEY_S) == GLFW_PRESS) move -= camera.front;
                glm::vec3 right = glm::normalize(glm::cross(camera.front, glm::vec3(0,1,0)));
                if (glfwGetKey(window.handle(), GLFW_KEY_A) == GLFW_PRESS) move -= right;
                if (glfwGetKey(window.handle(), GLFW_KEY_D) == GLFW_PRESS) move += right;
                if (glfwGetKey(window.handle(), GLFW_KEY_SPACE) == GLFW_PRESS) move.y += 1.0f;
                if (glfwGetKey(window.handle(), GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) move.y -= 1.0f;
                if (glm::length(move) > 0.0f)
                    camera.position += glm::normalize(move) * freeCamSpeed * dt;
            } else {
                player->update(dt, window.handle(), camera, *world);
            }

            if (firstFrame) {
                firstFrame = false;
                camera.position = player->position + glm::vec3(0.0f, 1.62f, 0.0f);
            }
        }

        float sunAngle = gameTime * 0.05f;
        float dayFactor = (std::sin(sunAngle) + 1.0f) * 0.5f;
        float skyR = 0.5f * dayFactor + 0.1f * (1.0f - dayFactor);
        float skyG = 0.72f * dayFactor + 0.05f * (1.0f - dayFactor);
        float skyB = 1.0f * dayFactor + 0.02f * (1.0f - dayFactor);
        if (dayFactor < 0.15f) {
            skyR = 0.02f;
            skyG = 0.02f;
            skyB = 0.05f;
        }
        glClearColor(skyR, skyG, skyB, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        if (world)
            renderer.draw(camera, window.width(), window.height(), *world, dayFactor);

        if (world && player && camera.perspective != CameraPerspective::FirstPerson) {
            playerModel.draw(camera, window.width(), window.height(),
                             player->position, camera.yaw(),
                             player->walkPhase, player->sneaking);
        }

        if (g_state == GameState::Playing || g_state == GameState::Paused)
            renderer.drawCrosshair(window.width(), window.height());

        if (g_state == GameState::Playing && inventory) {
            renderer.drawHotbar(window.width(), window.height(), *inventory);
        }

        if ((g_state == GameState::Playing || g_state == GameState::DevMenu) && showDebug && world && player) {
            renderer.drawDebug(window.width(), window.height(), *world, *player, camera, (int)(1.0f/dt), gameTime);
        }

        if (g_state == GameState::Paused)
            renderer.drawPauseOverlay();

        ui.beginFrame();
        GameState uiNext = g_state;
        if (g_state == GameState::MainMenu)
            uiNext = ui.renderMainMenu(g_state);
        else if (g_state == GameState::Paused)
            uiNext = ui.renderPauseMenu(g_state);
        // Dev menu disabled for stability
        // else if (g_state == GameState::DevMenu)
        //     uiNext = ui.renderDevMenu(g_state, world ? world->chunks().size() : 0, world ? world->seed() : 0);
        ui.endFrame();

        if (uiNext != g_state)
            pendingState = uiNext;

        window.swapBuffers();
    }

    return 0;
}
