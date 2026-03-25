#include "Camera.h"
#include <glm/gtc/matrix_transform.hpp>
#include <algorithm>

static constexpr float THIRD_PERSON_DIST = 4.0f;

Camera::Camera(glm::vec3 startPos) : position(startPos) {
    updateVectors();
}

glm::mat4 Camera::getView() const {
    switch (perspective) {
        case CameraPerspective::ThirdBack: {
            glm::vec3 camPos = position - front * THIRD_PERSON_DIST;
            return glm::lookAt(camPos, position, m_up);
        }
        case CameraPerspective::ThirdFront: {
            glm::vec3 camPos = position + front * THIRD_PERSON_DIST;
            return glm::lookAt(camPos, position, m_up);
        }
        default: // FirstPerson
            return glm::lookAt(position, position + front, m_up);
    }
}

glm::mat4 Camera::getProjection(float aspect) const {
    float fov = m_zooming ? m_zoomFov : m_fov;
    return glm::perspective(glm::radians(fov), aspect, 0.1f, 500.0f);
}

void Camera::resetMouse() {
    m_firstMouse = true;
}

void Camera::cyclePerspective() {
    switch (perspective) {
        case CameraPerspective::FirstPerson:
            perspective = CameraPerspective::ThirdBack;
            break;
        case CameraPerspective::ThirdBack:
            perspective = CameraPerspective::ThirdFront;
            break;
        case CameraPerspective::ThirdFront:
            perspective = CameraPerspective::FirstPerson;
            break;
    }
}

void Camera::setZoom(bool active) {
    m_zooming = active;
}

void Camera::processMouseMovement(double xpos, double ypos) {
    if (m_firstMouse) {
        m_lastX = xpos;
        m_lastY = ypos;
        m_firstMouse = false;
        return;
    }

    double dx =  (xpos - m_lastX) * m_sensitivity;
    double dy = -(ypos - m_lastY) * m_sensitivity;
    m_lastX = xpos;
    m_lastY = ypos;

    // Reduce sensitivity when zooming
    if (m_zooming) {
        dx *= 0.3;
        dy *= 0.3;
    }

    m_yaw   += static_cast<float>(dx);
    m_pitch += static_cast<float>(dy);
    m_pitch  = std::clamp(m_pitch, -89.0f, 89.0f);

    updateVectors();
}

void Camera::updateVectors() {
    glm::vec3 f;
    f.x = cos(glm::radians(m_yaw)) * cos(glm::radians(m_pitch));
    f.y = sin(glm::radians(m_pitch));
    f.z = sin(glm::radians(m_yaw)) * cos(glm::radians(m_pitch));
    front = glm::normalize(f);
}
