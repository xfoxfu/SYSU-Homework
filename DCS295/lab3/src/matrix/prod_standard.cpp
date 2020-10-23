#include "matrix.hpp"

Matrix product_standard(const Matrix &L, const Matrix &R)
{
  L.ensure_consistent_product(R);

  Matrix Y(L.m(), R.n());
  for (size_t i = 0; i < L.m(); i++)
  {
    for (size_t j = 0; j < R.n(); j++)
    {
      Y(i, j) = 0;
      for (size_t k = 0; k < L.n(); k++)
      {
        Y(i, j) += L(i, k) * R(k, j);
      }
    }
  }
  return Y;
}
