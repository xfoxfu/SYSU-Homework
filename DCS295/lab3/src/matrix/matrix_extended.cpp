#include "matrix_extended.hpp"

ExtendedMatrix::ExtendedMatrix(Matrix &ref, size_t m, size_t n)
    : Matrix(ref._data, ref.m(), ref.n())
{
  _mx = m;
  _nx = n;
}
ExtendedMatrix::~ExtendedMatrix() {}

size_t ExtendedMatrix::m() const { return _mx; }
size_t ExtendedMatrix::n() const { return _nx; }

ExtendedMatrix::data_t &ExtendedMatrix::operator()(size_t i, size_t j)
{
  if (i >= _mx || j >= _nx)
  {
    throw std::logic_error("out of bound B");
  }
  else if (i >= _m || j >= _n)
  {
    return const_cast<data_t &>(_value);
  }
  return Matrix::operator()(i, j);
}

const ExtendedMatrix::data_t &ExtendedMatrix::operator()(size_t i,
                                                         size_t j) const
{
  if (i >= _mx || j >= _nx)
  {
    throw std::logic_error("out of bound B");
  }
  else if (i >= _m || j >= _n)
  {
    return _value;
  }
  return Matrix::operator()(i, j);
}
