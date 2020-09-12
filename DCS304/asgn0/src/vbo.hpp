#pragma once

#include <cstddef>

class VBO {
public:
  VBO();
  ~VBO();
  void buffer_data(const void *data, size_t len);
  void set_pointer(size_t index, size_t size, GLenum type, bool normalized,
                   size_t stride, const void *pointer);

private:
  GLuint ID;
};
