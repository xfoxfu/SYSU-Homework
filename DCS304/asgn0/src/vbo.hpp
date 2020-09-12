#pragma once

#include <cstddef>

class VBO {
public:
  VBO();
  ~VBO();
  void buffer_data(const void *data, size_t len);
  void set_attrib(size_t index, size_t size, GLenum type, bool normalized,
                  size_t stride, size_t pointer);

private:
  GLuint ID;
};
