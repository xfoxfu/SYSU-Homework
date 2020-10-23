#include "matrix_extended.hpp"
#include "matrix_truncated.hpp"
#include "product.hpp"
#include <cmath>
#include <iostream>

Matrix sprod(Matrix &lhs, Matrix &rhs)
{
  // let mut out = SimpleMatrix::new(n, n);
  if (lhs.m() != lhs.n() || rhs.m() != rhs.n() || lhs.n() != rhs.m())
  {
    throw std::logic_error(
        "unable to product with strassen, size inconsistent");
  }
  auto n = lhs.m();
  if (n <= 64)
  {
    return product_standard(lhs, rhs);
  }
  if (n % 2 != 0)
  {
    throw std::logic_error(
        "unable to product with strassen, size not power of 2");
  }
  Matrix ret(n);
  // let block_size = n / 2;
  auto block_size = n / 2;
  TruncatedMatrix A11(lhs, 0, 0, block_size, block_size);
  TruncatedMatrix A12(lhs, 0, block_size, block_size, block_size);
  TruncatedMatrix A21(lhs, block_size, 0, block_size, block_size);
  TruncatedMatrix A22(lhs, block_size, block_size, block_size, block_size);
  TruncatedMatrix B11(rhs, 0, 0, block_size, block_size);
  TruncatedMatrix B12(rhs, 0, block_size, block_size, block_size);
  TruncatedMatrix B21(rhs, block_size, 0, block_size, block_size);
  TruncatedMatrix B22(rhs, block_size, block_size, block_size, block_size);

  Matrix S1 = B12 - B22;
  Matrix S2 = A11 + A12;
  Matrix S3 = A21 + A22;
  Matrix S4 = B21 - B11;
  Matrix S5 = A11 + A22;
  Matrix S6 = B11 + B22;
  Matrix S7 = A12 - A22;
  Matrix S8 = B21 + B22;
  Matrix S9 = A11 - A21;
  Matrix S10 = B11 + B12;

  Matrix P1 = product_strassen(A11, S1);
  Matrix P2 = product_strassen(S2, B22);
  Matrix P3 = product_strassen(S3, B11);
  Matrix P4 = product_strassen(A22, S4);
  Matrix P5 = product_strassen(S5, S6);
  Matrix P6 = product_strassen(S7, S8);
  Matrix P7 = product_strassen(S9, S10);

  for (size_t i = 0; i < block_size; i++)
  {
    for (size_t j = 0; j < block_size; j++)
    {
      ret(i, j) = P5(i, j) + P4(i, j) - P2(i, j) + P6(i, j);
    }
  }
  for (size_t i = 0; i < block_size; i++)
  {
    for (size_t j = 0; j < block_size; j++)
    {
      ret(i, j + block_size) = P1(i, j) + P2(i, j);
    }
  }
  for (size_t i = 0; i < block_size; i++)
  {
    for (size_t j = 0; j < block_size; j++)
    {
      ret(i + block_size, j) = P3(i, j) + P4(i, j);
    }
  }
  for (size_t i = 0; i < block_size; i++)
  {
    for (size_t j = 0; j < block_size; j++)
    {
      ret(i + block_size, j + block_size) =
          P5(i, j) + P1(i, j) - P3(i, j) - P7(i, j);
    }
  }

  return ret;
}

Matrix product_strassen(Matrix &lhs, Matrix &rhs)
{
  size_t max_size =
      std::max(std::max(lhs.m(), lhs.n()), std::max(rhs.m(), rhs.n()));
  size_t n = std::pow(2, std::ceil(std::log2(max_size)));
  if (n == lhs.m() && n == lhs.n() && n == rhs.m() && n == rhs.n())
  {
    return sprod(lhs, rhs);
  }
  else
  {
    ExtendedMatrix elhs(lhs, n, n);
    ExtendedMatrix erhs(rhs, n, n);
    return sprod(elhs, erhs);
  }
}
