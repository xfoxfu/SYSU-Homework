#include <glad/glad.h>

#include "vbo.hpp"
#include "vertex.hpp"

VBO::VBO() { glGenBuffers(1, &ID); }

VBO::~VBO() { glDeleteBuffers(1, &ID); }

void VBO::buffer_data(const void *data, size_t len) {
  glBindBuffer(GL_ARRAY_BUFFER, ID);
  glBufferData(GL_ARRAY_BUFFER, len, data, GL_STATIC_DRAW);
}

/**
 * @brief  设置顶点对应的数组信息应当被如何解释
 * @note 对应 `glVertexAttribPointer` 和 `glEnableVertexAttribArray`
 * @param  index:      位置，对应 shader 中的 location 参数
 * @param  size:       元素数量
 * @param  type:       元素的数据类型
 * @param  normalized: 是否被标准化到 1.0
 * @param  stride:     元素间隔步长
 * @param  pointer:    起始位置偏移
 * @retval None
 */
void VBO::set_attrib(size_t index, size_t size, GLenum type, bool normalized,
                     size_t stride, size_t pointer) {
  glVertexAttribPointer(index, size, type, normalized, stride,
                        (const void *)pointer);
  glEnableVertexAttribArray(index);
}

void VBO::set_attrib_vertex() {
  set_attrib(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex),
             offsetof(Vertex, position));
  set_attrib(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), offsetof(Vertex, color));
  set_attrib(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex),
             offsetof(Vertex, texture_position));
}
