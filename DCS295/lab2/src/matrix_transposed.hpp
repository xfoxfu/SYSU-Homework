#pragma once

#include "matrix.hpp"
#include <cstddef>

class TransposedMatrix : public Matrix {
public:
  TransposedMatrix(size_t, size_t);
  TransposedMatrix(size_t, size_t, bool);
  TransposedMatrix(data_t *, size_t, size_t);
  virtual ~TransposedMatrix();

  virtual size_t m() const;
  virtual size_t n() const;

  virtual data_t &operator()(size_t, size_t);
  virtual const data_t &operator()(size_t, size_t) const;

protected:
};
