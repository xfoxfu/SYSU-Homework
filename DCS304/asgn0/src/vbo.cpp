#include <glad/glad.h>

#include "vbo.hpp"

VBO::VBO() { glGenBuffers(1, &ID); }

VBO::~VBO() { glDeleteBuffers(1, &ID); }

void VBO::buffer_data(const void *data, size_t len) {
  glBindBuffer(GL_ARRAY_BUFFER, ID);
  glBufferData(GL_ARRAY_BUFFER, len, data, GL_STATIC_DRAW);
}

void VBO::set_pointer(size_t index, size_t size, GLenum type, bool normalized,
                      size_t stride, const void *pointer) {
  glVertexAttribPointer(index, size, type, normalized, stride, pointer);
  glEnableVertexAttribArray(index);
}
