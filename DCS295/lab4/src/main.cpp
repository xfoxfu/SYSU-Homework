#include <algorithm>
#include <chrono>
#include <iostream>
#include <pthread.h>
#include <cstring>
#include <functional>

#include "errors.hpp"
#include "matrix_transposed.hpp"
#include "matrix.hpp"
#include "product.hpp"

typedef double value_t;

// template <typename T>
Matrix run_timed(const char *name, std::function<Matrix()> fn)
{
  // perform matrix production
  // record start time
  auto start = std::chrono::high_resolution_clock::now();
  // do some work
  auto ret = fn();
  // record end time
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> diff = end - start;
  std::cout << name << ": " << diff.count() << " ms\n";

  return ret;
}

int main(int argc, char **argv)
{
  // read input paramaters
  if (argc < 4)
  {
    std::cout << "Usage: " << argv[0] << " <M> <N> <K> [--no-output]" << std::endl;
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

  Matrix Zn = run_timed("OMP Normal", [&L, &R]() { return product_omp(L, R); });
  if (output)
    std::cout << Zn;
  Matrix Zs = run_timed("OMP Static", [&L, &R]() { return product_omp_static(L, R); });
  if (output)
    std::cout << Zs;
  Matrix Zd = run_timed("OMP Dynamic", [&L, &R]() { return product_omp_dynamic(L, R); });
  if (output)
    std::cout << Zd;
  Matrix Zp = run_timed("Parallel For", [&L, &R]() { return product_pfor(L, R); });
  if (output)
    std::cout << Zp;

#ifndef NDEBUG // debug assertion
  Matrix W = run_timed("Normal", [&L, &R]() { return product_standard(L, R); });

  for (size_t i = 0; i < m; i++)
  {
    for (size_t j = 0; j < k; j++)
    {
      if (abs(W(i, j) - Zn(i, j)) > std::numeric_limits<double>().epsilon())
      {
        std::cout << "different at " << i << ", " << j << " = "
                  << (W(i, j) - Zn(i, j)) << std::endl;
      }
      if (abs(W(i, j) - Zs(i, j)) > std::numeric_limits<double>().epsilon())
      {
        std::cout << "different at " << i << ", " << j << " = "
                  << (W(i, j) - Zs(i, j)) << std::endl;
      }
      if (abs(W(i, j) - Zd(i, j)) > std::numeric_limits<double>().epsilon())
      {
        std::cout << "different at " << i << ", " << j << " = "
                  << (W(i, j) - Zd(i, j)) << std::endl;
      }
      if (abs(W(i, j) - Zp(i, j)) > std::numeric_limits<double>().epsilon())
      {
        std::cout << "different at " << i << ", " << j << " = "
                  << (W(i, j) - Zp(i, j)) << std::endl;
      }
    }
  }
#endif

  return MATRIX_OK;
}
