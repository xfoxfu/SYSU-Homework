#include "matrix_transposed.hpp"

TransposedMatrix::TransposedMatrix(size_t m, size_t n) : Matrix(n, m) {}
TransposedMatrix::TransposedMatrix(size_t m, size_t n, bool d)
    : Matrix(n, m, d) {}
TransposedMatrix::TransposedMatrix(TransposedMatrix::data_t *data, size_t m,
                                   size_t n)
    : Matrix(data, n, m) {}
TransposedMatrix::~TransposedMatrix() {}

size_t TransposedMatrix::m() const { return Matrix::n(); }
size_t TransposedMatrix::n() const { return Matrix::m(); }

TransposedMatrix::data_t &TransposedMatrix::operator()(size_t i, size_t j)
{
  return Matrix::operator()(j, i);
}

const TransposedMatrix::data_t &TransposedMatrix::operator()(size_t i,
                                                             size_t j) const
{
  return Matrix::operator()(j, i);
}
