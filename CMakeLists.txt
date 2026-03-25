cmake_minimum_required(VERSION 3.20)
project(Ferox VERSION 0.1.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

# GLFW - window creation and input
FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG        3.3.10
)
set(GLFW_BUILD_DOCS     OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(glfw)

# GLAD - loads OpenGL function pointers at runtime
# Using proper tag format with 'v' prefix
FetchContent_Declare(
    glad
    GIT_REPOSITORY https://github.com/Dav1dde/glad.git
    GIT_TAG        v0.1.36
)
set(GLAD_PROFILE   "core" CACHE STRING "" FORCE)
set(GLAD_API       "gl=3.3" CACHE STRING "" FORCE)
set(GLAD_GENERATOR "c"    CACHE STRING "" FORCE)
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
set(CMAKE_POLICY_VERSION_MAXIMUM 3.24)
FetchContent_MakeAvailable(glad)

# GLM - math library (vectors, matrices)
FetchContent_Declare(
    glm
    GIT_REPOSITORY https://github.com/g-truc/glm.git
    GIT_TAG        0.9.9.8
)
FetchContent_MakeAvailable(glm)

# stb_image - single-header image loader (local copy in src/stb/)
add_library(stb_image INTERFACE)
target_include_directories(stb_image INTERFACE ${CMAKE_SOURCE_DIR}/src/stb)

# Dear ImGui - immediate mode GUI
FetchContent_Declare(
    imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG        v1.91.1
)
FetchContent_MakeAvailable(imgui)

add_library(imgui STATIC
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
)
target_include_directories(imgui PUBLIC
    ${imgui_SOURCE_DIR}
    ${imgui_SOURCE_DIR}/backends
)
target_compile_definitions(imgui PUBLIC IMGUI_IMPL_OPENGL_LOADER_GLAD)
target_link_libraries(imgui PUBLIC glfw glad)

# Collect all .cpp files under src/
file(GLOB_RECURSE SOURCES "src/*.cpp")

# Explicitly include Inventory.cpp to ensure it's picked up
if(NOT EXISTS "${CMAKE_SOURCE_DIR}/src/Inventory.cpp")
    message(FATAL_ERROR "Inventory.cpp not found")
endif()

add_executable(Ferox ${SOURCES})
target_link_options(Ferox PRIVATE "/SUBSYSTEM:WINDOWS" "/ENTRY:mainCRTStartup")
target_include_directories(Ferox PRIVATE src)
target_link_libraries(Ferox PRIVATE glfw glad glm imgui stb_image)

# Copy the resources folder next to the built executable
add_custom_command(TARGET Ferox POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_SOURCE_DIR}/resources"
        "$<TARGET_FILE_DIR:Ferox>/resources"
    VERBATIM
)
