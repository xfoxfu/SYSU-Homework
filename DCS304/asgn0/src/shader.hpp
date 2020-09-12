#pragma once

#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <string>

class Shader {
public:
  unsigned int ID;
  Shader(const char *vertexPath, const char *fragmentPath);
  ~Shader();
  void use();
  void setBool(const std::string &name, bool value) const;
  void setInt(const std::string &name, int value) const;
  void setFloat(const std::string &name, float value) const;
  void set(const char *name, glm::vec3 value) const;
  void set(const char *name, glm::vec4 value) const;
  void bind_current_texture_to(const char *name) const;

private:
  void checkCompileErrors(unsigned int shader, std::string type);
};
