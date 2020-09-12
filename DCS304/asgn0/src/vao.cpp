#include <glad/glad.h>

#include "vao.hpp"

VAO::VAO() { glGenVertexArrays(/* count */ 1, &ID); }

VAO::~VAO() { glDeleteVertexArrays(/* count */ 1, &ID); }

void VAO::bind() { glBindVertexArray(ID); }

void VAO::unbind() { glBindVertexArray(0); }
