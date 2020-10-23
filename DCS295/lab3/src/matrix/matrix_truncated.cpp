#include "matrix_truncated.hpp"

TruncatedMatrix::TruncatedMatrix(Matrix &ref, size_t i0, size_t j0, size_t m,
                                 size_t n)
    : Matrix(ref._data, ref.m(), ref.n())
{
  _i0 = i0;
  _j0 = j0;
  _mx = m;
  _nx = n;
}
TruncatedMatrix::~TruncatedMatrix() {}

size_t TruncatedMatrix::m() const { return _mx; }
size_t TruncatedMatrix::n() const { return _nx; }

TruncatedMatrix::data_t &TruncatedMatrix::operator()(size_t i, size_t j)
{
  if (i >= _mx || j >= _nx)
  {
    throw std::logic_error("out of bound B");
  }
  return Matrix::operator()(i + _i0, j + _j0);
}

const TruncatedMatrix::data_t &TruncatedMatrix::operator()(size_t i,
                                                           size_t j) const
{
  if (i >= _mx || j >= _nx)
  {
    throw std::logic_error("out of bound B");
  }
  return Matrix::operator()(i + _i0, j + _j0);
}
