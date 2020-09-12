#pragma once

#include "vbo.hpp"
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>

class Vertex {
public:
  glm::vec3 position;
  glm::vec3 color;
  glm::vec2 texture_position;

  Vertex();
  Vertex(glm::vec3 position);
  Vertex(glm::vec3 position, glm::vec3 color);
  Vertex(glm::vec3 position, glm::vec2 texture_position);
  Vertex(glm::vec3 position, glm::vec3 color, glm::vec2 texture_position);

private:
};
