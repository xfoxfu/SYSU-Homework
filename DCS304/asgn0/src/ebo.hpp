#pragma once

#include <cstddef>
#include <cstdint>

class EBO {
public:
  EBO();
  ~EBO();
  void bind() const noexcept;
  void buffer_data(void *indices, size_t count) const noexcept;

private:
  uint32_t _id;
};
