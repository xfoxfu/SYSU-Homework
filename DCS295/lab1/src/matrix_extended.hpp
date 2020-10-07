#pragma once

#include "matrix.hpp"
#include <cstddef>

class ExtendedMatrix : public Matrix {
public:
  ExtendedMatrix(Matrix &, size_t, size_t);
  virtual ~ExtendedMatrix();

  virtual size_t m() const;
  virtual size_t n() const;

  virtual data_t &operator()(size_t, size_t);
  virtual const data_t &operator()(size_t, size_t) const;

protected:
  constexpr static const data_t _value = 0.0;
  size_t _mx;
  size_t _nx;
};
