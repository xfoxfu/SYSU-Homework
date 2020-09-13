#include <glad/glad.h>

#include "ebo.hpp"

EBO::EBO() { glGenBuffers(1, &_id); }

EBO::~EBO() { glDeleteBuffers(1, &_id); }

void EBO::bind() const noexcept { glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _id); }

void EBO::buffer_data(void *indices, size_t count) const noexcept {
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, count, indices, GL_STATIC_DRAW);
}
