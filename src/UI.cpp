#include "UI.h"

int g_renderDistance = 4;

UI::UI(GLFWwindow* window) : m_window(window) {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();

    ImGuiIO& io = ImGui::GetIO();
    io.IniFilename = nullptr; // don't write imgui.ini

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330 core");

    applyStyle();
}

UI::~UI() {
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}

void UI::beginFrame() {
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}

void UI::endFrame() {
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void UI::applyStyle() {
    ImGuiStyle& style = ImGui::GetStyle();

    style.WindowRounding    = 0.0f;
    style.FrameRounding     = 0.0f;
    style.GrabRounding      = 0.0f;
    style.WindowBorderSize  = 2.0f;
    style.FrameBorderSize   = 1.0f;
    style.WindowPadding     = ImVec2(16.0f, 16.0f);
    style.FramePadding      = ImVec2(10.0f, 8.0f);
    style.ItemSpacing       = ImVec2(8.0f, 10.0f);

    ImVec4* c = style.Colors;
    c[ImGuiCol_WindowBg]      = ImVec4(0.10f, 0.10f, 0.10f, 0.92f);
    c[ImGuiCol_Border]        = ImVec4(0.25f, 0.25f, 0.25f, 1.00f);
    c[ImGuiCol_TitleBg]       = ImVec4(0.15f, 0.15f, 0.15f, 1.00f);
    c[ImGuiCol_TitleBgActive] = ImVec4(0.20f, 0.20f, 0.20f, 1.00f);
    c[ImGuiCol_Button]        = ImVec4(0.20f, 0.20f, 0.20f, 1.00f);
    c[ImGuiCol_ButtonHovered] = ImVec4(0.30f, 0.30f, 0.30f, 1.00f);
    c[ImGuiCol_ButtonActive]  = ImVec4(0.15f, 0.15f, 0.15f, 1.00f);
    c[ImGuiCol_Text]          = ImVec4(1.00f, 1.00f, 1.00f, 1.00f);
    c[ImGuiCol_TextDisabled]  = ImVec4(0.50f, 0.50f, 0.50f, 1.00f);
}

GameState UI::renderMainMenu(GameState current) {
    GameState next = current;

    ImGuiIO& io = ImGui::GetIO();
    float W = io.DisplaySize.x;
    float H = io.DisplaySize.y;

    ImGui::SetNextWindowPos(ImVec2(0, 0));
    ImGui::SetNextWindowSize(ImVec2(W, H));
    ImGui::SetNextWindowBgAlpha(0.75f);

    ImGuiWindowFlags flags =
        ImGuiWindowFlags_NoTitleBar   |
        ImGuiWindowFlags_NoResize     |
        ImGuiWindowFlags_NoMove       |
        ImGuiWindowFlags_NoScrollbar  |
        ImGuiWindowFlags_NoSavedSettings;

    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    ImGui::Begin("##mainmenu", nullptr, flags);
    ImGui::PopStyleVar();

    float btnW = 260.0f;
    float btnH = 40.0f;
    float centerX = (W - btnW) * 0.5f;

    // Title
    ImGui::SetCursorPosY(H * 0.25f);
    ImGui::SetWindowFontScale(3.0f);
    const char* title = "FEROX";
    float titleW = ImGui::CalcTextSize(title).x;
    ImGui::SetCursorPosX((W - titleW) * 0.5f);
    ImGui::TextColored(ImVec4(1.0f, 0.84f, 0.0f, 1.0f), "%s", title);
    ImGui::SetWindowFontScale(1.0f);

    // Subtitle
    const char* sub = "Alpha 0.0.2";
    float subW = ImGui::CalcTextSize(sub).x;
    ImGui::SetCursorPosX((W - subW) * 0.5f);
    ImGui::TextColored(ImVec4(0.65f, 0.65f, 0.65f, 1.0f), "%s", sub);

    ImGui::SetCursorPosY(H * 0.50f);

    // New World button
    ImGui::SetCursorPosX(centerX);
    ImGui::PushStyleColor(ImGuiCol_Button,        ImVec4(0.18f, 0.45f, 0.12f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.25f, 0.60f, 0.18f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive,  ImVec4(0.12f, 0.35f, 0.08f, 1.0f));
    if (ImGui::Button("New World", ImVec2(btnW, btnH)))
        next = GameState::Playing;
    ImGui::PopStyleColor(3);

    ImGui::End();
    return next;
}

GameState UI::renderPauseMenu(GameState current) {
    GameState next = current;

    ImGuiIO& io = ImGui::GetIO();
    float W = io.DisplaySize.x;
    float H = io.DisplaySize.y;

    float panelW = 280.0f;
    float panelH = 190.0f;
    ImGui::SetNextWindowPos(ImVec2((W - panelW) * 0.5f, (H - panelH) * 0.5f));
    ImGui::SetNextWindowSize(ImVec2(panelW, panelH));

    ImGuiWindowFlags flags =
        ImGuiWindowFlags_NoResize        |
        ImGuiWindowFlags_NoMove          |
        ImGuiWindowFlags_NoScrollbar     |
        ImGuiWindowFlags_NoSavedSettings |
        ImGuiWindowFlags_NoCollapse;

    ImGui::Begin("Game Paused", nullptr, flags);

    float btnW = panelW - 32.0f;
    float btnH = 38.0f;

    ImGui::PushStyleColor(ImGuiCol_Button,        ImVec4(0.18f, 0.45f, 0.12f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.25f, 0.60f, 0.18f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive,  ImVec4(0.12f, 0.35f, 0.08f, 1.0f));
    if (ImGui::Button("Resume", ImVec2(btnW, btnH)))
        next = GameState::Playing;
    ImGui::PopStyleColor(3);

    ImGui::Spacing();

    if (ImGui::Button("Quit to Menu", ImVec2(btnW, btnH)))
        next = GameState::MainMenu;

    ImGui::End();
    return next;
}

GameState UI::renderDevMenu(GameState current, size_t chunksLoaded, uint32_t worldSeed) {
    GameState next = current;

    ImGuiIO& io = ImGui::GetIO();
    float W = io.DisplaySize.x;
    float H = io.DisplaySize.y;

    float panelW = 320.0f;
    float panelH = 200.0f;
    ImGui::SetNextWindowPos(ImVec2(W - panelW - 20.0f, 20.0f));
    ImGui::SetNextWindowSize(ImVec2(panelW, panelH));

    ImGuiWindowFlags flags =
        ImGuiWindowFlags_NoResize        |
        ImGuiWindowFlags_NoMove          |
        ImGuiWindowFlags_NoScrollbar     |
        ImGuiWindowFlags_NoSavedSettings |
        ImGuiWindowFlags_NoCollapse;

    ImGui::Begin("Developer Menu", nullptr, flags);

    ImGui::Text("Render Distance: %d", g_renderDistance);
    ImGui::SliderInt("##rendist", &g_renderDistance, 4, 64, "%d chunks");

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    ImGui::Text("Chunks loaded: %zu", chunksLoaded);
    ImGui::Text("World seed: %u", worldSeed);

    ImGui::Spacing();
    ImGui::Text("Press ESC or 0 to close");

    ImGui::End();
    return next;
}
