#pragma once

#include "matrix.hpp"
#include <cstddef>

class TruncatedMatrix : public Matrix
{
public:
  TruncatedMatrix(Matrix &, size_t, size_t, size_t, size_t);
  virtual ~TruncatedMatrix();

  virtual size_t m() const;
  virtual size_t n() const;

  virtual data_t &operator()(size_t, size_t);
  virtual const data_t &operator()(size_t, size_t) const;

protected:
  size_t _i0;
  size_t _j0;
  size_t _mx;
  size_t _nx;
};
