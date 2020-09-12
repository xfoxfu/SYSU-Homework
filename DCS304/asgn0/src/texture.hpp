#pragma once

#include <cstdint>

class Texture {
public:
  Texture(const char *filename);
  ~Texture();
  void bind();
  void bind(GLenum unit);

private:
  const char *_filename;
  uint32_t _id;
  int32_t _width;
  int32_t _height;
  int32_t _nr_channels;
};
