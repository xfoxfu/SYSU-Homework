#include "matrix.hpp"
#include "matrix_transposed.hpp"
#include "product.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <mpi.h>

#define MPI_Exit(code)                                                         \
  MPI_Finalize();                                                              \
  return code;

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int mpi_size; // 进程数量
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  int mpi_rank; // 当前进程编号
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

  if (mpi_size > 8) {
    std::cout << "MPI process more than 8 is not supported" << std::endl;
    MPI_Exit(1);
  }

  if (argc < 4) {
    std::cout << "Usage: " << argv[0] << " <M> <N> <K>" << std::endl;
    MPI_Exit(1);
  }

  size_t m = atoi(argv[1]);
  size_t n = atoi(argv[2]);
  size_t k = atoi(argv[3]);
  bool output = false;
  if (argc > 4 && std::strcmp(argv[4], "--output") == 0) {
    output = true;
  }

  if (mpi_rank == 0) {
    std::cout << "M=" << m << ",N=" << n << ",K=" << k << std::endl;
  }

  Matrix L(m, n, true);
  if (output) {
    std::cout << L;
  }
  TransposedMatrix R(n, k, true);
  if (output) {
    std::cout << R;
  }

  if (mpi_rank != 0) {
    product_mpi(mpi_size, mpi_rank, nullptr, nullptr);
  } else {
    // record start time
    auto start = std::chrono::high_resolution_clock::now();
    // do some work
    Matrix Z = product_mpi(mpi_size, mpi_rank, &L, &R);
    // record end time
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> diff = end - start;
    std::cout << "mpi: " << diff.count() << " ms\n";
    if (output) {
      std::cout << Z;
    }

    // record start time
    auto start2 = std::chrono::high_resolution_clock::now();
    // do some work
    Matrix W = product_standard(L, R);
    // record end time
    auto end2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> diff2 = end2 - start2;
    std::cout << "normal: " << diff2.count() << "ms\n";

    for (size_t i = 0; i < m; i++) {
      for (size_t j = 0; j < k; j++) {
        if (abs(W(i, j) - Z(i, j)) > std::numeric_limits<double>().epsilon()) {
          std::cout << "different at " << i << ", " << j << " = "
                    << (W(i, j) - Z(i, j)) << std::endl;
        }
      }
    }
  }

  MPI_Exit(0);
}
