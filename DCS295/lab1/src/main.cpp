#include "matrix.hpp"
#include "product.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>

int main(int argc, char **argv) {
  if (argc < 4) {
    std::cout << "Usage: " << argv[0] << " <M> <N> <K>" << std::endl;
    return 1;
  }

  size_t m = atoi(argv[1]);
  size_t n = atoi(argv[2]);
  size_t k = atoi(argv[3]);
  bool output = true;
  if (argc > 4 && std::strcmp(argv[4], "--no-output") == 0) {
    output = false;
  }

  std::cout << "M=" << m << ",N=" << n << ",K=" << k << std::endl;

  Matrix L(m, n, true);
  if (output) {
    std::cout << L;
  }
  Matrix R(n, k, true);
  if (output) {
    std::cout << R;
  }

  // record start time
  auto start = std::chrono::high_resolution_clock::now();
  // do some work
  Matrix Y = product_standard(L, R);
  // record end time
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> diff = end - start;
  std::cout << "Time to compute matrix production : " << diff.count()
            << " ms\n";
  // record start time
  auto start2 = std::chrono::high_resolution_clock::now();
  // do some work
  Matrix Z = product_strassen(L, R);
  // record end time
  auto end2 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> diff2 = end2 - start2;
  std::cout << "Time to compute matrix production : " << diff2.count()
            << " ms\n";
  if (output) {
    std::cout << Y;
  }

  return 0;
}
