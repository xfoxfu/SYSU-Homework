#include "vertex.hpp"

Vertex::Vertex() : Vertex(glm::vec3(0, 0, 0)) {}
Vertex::Vertex(glm::vec3 position)
    : Vertex(position, glm::vec3(0, 0, 0), glm::vec2(0, 0)) {}
Vertex::Vertex(glm::vec3 position, glm::vec3 color)
    : Vertex(position, color, glm::vec2(0, 0)) {}
Vertex::Vertex(glm::vec3 position, glm::vec2 texture_position)
    : Vertex(position, glm::vec3(0, 0, 0), texture_position) {}

Vertex::Vertex(glm::vec3 position, glm::vec3 color, glm::vec2 texture_position)
    : position(position), color(color), texture_position(texture_position) {}
