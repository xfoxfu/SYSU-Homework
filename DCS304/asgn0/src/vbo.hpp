#pragma once

#include <cstddef>
#include <glad/glad.h>

class VBO {
public:
  VBO();
  ~VBO();
  void buffer_data(const void *data, size_t len);
  void set_attrib(size_t index, size_t size, GLenum type, bool normalized,
                  size_t stride, size_t pointer);
  void set_attrib_vertex();

private:
  GLuint ID;
};
