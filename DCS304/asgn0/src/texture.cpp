#include <glad/glad.h>

#include "texture.hpp"
#include <iostream>
#include <stb_image.h>

Texture::Texture(const char *filename) {
  glGenTextures(1, &_id);
  _filename = filename;
  glBindTexture(GL_TEXTURE_2D, _id);
  // set the texture wrapping parameters
  glTexParameteri(
      GL_TEXTURE_2D, GL_TEXTURE_WRAP_S,
      GL_REPEAT); // set texture wrapping to GL_REPEAT (default wrapping method)
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  // set texture filtering parameters
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  // load image, create texture and generate mipmaps
  stbi_set_flip_vertically_on_load(true);
  unsigned char *data =
      stbi_load(_filename, &_width, &_height, &_nr_channels, 0);
  if (data == nullptr) {
    std::cout << "[TEXTURE]" << stbi_failure_reason() << std::endl;
  }

  auto type = GL_RGB;
  if (_nr_channels == 4) {
    type = GL_RGBA;
  }

  glTexImage2D(GL_TEXTURE_2D, 0, type, _width, _height, 0, type,
               GL_UNSIGNED_BYTE, data);
  glGenerateMipmap(GL_TEXTURE_2D);
  stbi_image_free(data);
}

Texture::~Texture() { glDeleteTextures(1, &_id); }

void Texture::bind() { bind(GL_TEXTURE0); }

void Texture::bind(GLenum unit) {
  glActiveTexture(unit);
  glBindTexture(GL_TEXTURE_2D, _id);
}
