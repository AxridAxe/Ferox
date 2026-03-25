#pragma once
#include <glm/glm.hpp>

enum class CameraPerspective {
    FirstPerson,   // default
    ThirdBack,     // behind player
    ThirdFront,    // facing player
};

class Camera {
public:
    Camera(glm::vec3 startPos);

    void processMouseMovement(double xpos, double ypos);
    void resetMouse();

    // Get the actual rendering view matrix (accounts for perspective mode)
    glm::mat4 getView()                  const;
    glm::mat4 getProjection(float aspect) const;

    // Cycle through perspectives (F5)
    void cyclePerspective();

    // Zoom controls
    void setZoom(bool active);
    bool isZooming() const { return m_zooming; }

    glm::vec3 position;   // eye position (set by Player::update or third-person logic)
    glm::vec3 front{ 0.0f, 0.0f, -1.0f };

    CameraPerspective perspective = CameraPerspective::FirstPerson;

    // Yaw in degrees — needed by PlayerModel for orientation
    float yaw()   const { return m_yaw; }
    float pitch() const { return m_pitch; }

private:
    glm::vec3 m_up{ 0.0f, 1.0f, 0.0f };
    float m_yaw         = -90.0f;
    float m_pitch       =   0.0f;
    float m_speed       =  10.0f;
    float m_sensitivity =   0.1f;
    float m_fov         =  70.0f;
    float m_zoomFov     =  20.0f; // spyglass-like zoom FOV
    bool  m_zooming     = false;
    bool  m_firstMouse  = true;
    double m_lastX = 0.0, m_lastY = 0.0;

    void updateVectors();
};
