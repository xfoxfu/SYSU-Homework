#include <algorithm>
#include <chrono>
#include <iostream>
#include <pthread.h>
#include <cstring>

#include "errors.hpp"
#include "matrix_transposed.hpp"
#include "matrix.hpp"
#include "prod_cuda.cuh"

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

int main(int argc, char **argv)
{
#ifndef NDEBUG
  std::cout << "WARNING: running in Debug mode." << std::endl;
#endif
  // read input paramaters
  if (argc <= 3)
  {
    std::cout << "Usage: " << argv[0] << " <M> <N> <K> [--output]" << std::endl;
    return MATRIX_INVALID_ARGUMENTS;
  }
  size_t m = atoi(argv[1]);
  size_t n = atoi(argv[2]);
  size_t k = atoi(argv[3]);
  bool output = false;
  if (argc > 4 && strcmp(argv[4], "--output") == 0)
  {
    output = true;
  }

  std::cout << "M=" << m << ",N=" << n << ",K=" << k << std::endl;

  // generate matrix
  Matrix L(m, n, true);
  if (output)
  {
    std::cout << L;
  }
  TransposedMatrix R(n, k, true);
  if (output)
  {
    std::cout << R;
  }

  // perform matrix production
  // record start time
  auto start = std::chrono::high_resolution_clock::now();
  // do some work
  Matrix Z = product_cublas(L, R);
  // record end time
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> diff = end - start;
  std::cout << "time: " << diff.count() << " ms\n";
  if (output)
  {
    std::cout << Z;
  }

#ifndef NDEBUG // debug assertion
  // record start time
  auto start2 = std::chrono::high_resolution_clock::now();
  // do some work
  Matrix W = product_standard(L, R);
  // record end time
  auto end2 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> diff2 = end2 - start2;
  std::cout << "normal: " << diff2.count() << "ms\n";

  for (size_t i = 0; i < m; i++)
  {
    for (size_t j = 0; j < k; j++)
    {
      if (abs(W(i, j) - Z(i, j)) > std::numeric_limits<double>().epsilon())
      {
        std::cout << "different at " << i << ", " << j << " = "
                  << (W(i, j) - Z(i, j)) << std::endl;
      }
    }
  }
#endif

  return MATRIX_OK;
}
