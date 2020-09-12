#pragma once

class VAO {
public:
  VAO();
  ~VAO();
  void bind();
  void unbind();

private:
  GLuint ID;
};
