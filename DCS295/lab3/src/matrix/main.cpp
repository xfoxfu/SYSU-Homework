#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <thread>
#include <vector>
#include <pthread.h>

#include "errors.hpp"
#include "matrix_transposed.hpp"
#include "matrix.hpp"

#define THREAD_LIMIT 8
typedef double value_t;

struct worker_params
{
  size_t id;
  size_t threads;
  const Matrix *lhs;
  const Matrix *rhs;
  Matrix *out;
};

Matrix product_standard(const Matrix &L, const Matrix &R);
Matrix product(const Matrix &lhs, const Matrix &rhs, size_t p);
void *worker_wrapper(void *args);
void worker(size_t id, size_t threads, const Matrix &lhs, const Matrix &rhs, Matrix &out);

int main(int argc, char **argv)
{
  // read input paramaters
  if (argc < 5)
  {
    std::cout << "Usage: " << argv[0] << " <M> <N> <K> p [--no-output]" << std::endl;
    return MATRIX_INVALID_ARGUMENTS;
  }
  size_t m = atoi(argv[1]);
  size_t n = atoi(argv[2]);
  size_t k = atoi(argv[3]);
  size_t p = atoi(argv[4]);
  if (p > THREAD_LIMIT)
  {
    std::cout << "Thread count " << p << " exceeds limit " << THREAD_LIMIT << std::endl;
    return MATRIX_THREAD_LIMIT_EXCEED;
  }
  bool output = false;
  if (argc > 5 && std::strcmp(argv[5], "--output") == 0)
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
  Matrix Z = product(L, R, p);
  // record end time
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> diff = end - start;
  std::cout << "time: " << diff.count() << " ms\n";
  if (output)
  {
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

  return MATRIX_OK;
}

Matrix product(const Matrix &lhs, const Matrix &rhs, size_t p)
{
  Matrix out(lhs.m(), rhs.n());

  // spawn workers
  pthread_t threads[THREAD_LIMIT];
  worker_params *params = new worker_params[THREAD_LIMIT];
  for (size_t id = 0; id < p; id++)
  {
    params[id].id = id;
    params[id].threads = p;
    params[id].lhs = &lhs;
    params[id].rhs = &rhs;
    params[id].out = &out;
    pthread_create(&threads[id], NULL, worker_wrapper, reinterpret_cast<void *>(&params[id]));
  }

  // join workers
  for (size_t id = 0; id < p; id++)
  {
    pthread_join(threads[id], nullptr);
  }

  delete[] params;
  return out;
}

void *worker_wrapper(void *args_)
{
  worker_params *args = reinterpret_cast<worker_params *>(args_);
  worker(args->id, args->threads, *args->lhs, *args->rhs, *args->out);
  return nullptr;
}

std::mutex lock;

void worker(size_t id, size_t threads, const Matrix &lhs, const Matrix &rhs, Matrix &out)
{
  // $\delta = \lceil lhs.m() / threads \rceil$
  // compute rows in $[id\times \delta, (id + 1)\times \delta]$
  // $\delta$ is at least 1
  size_t delta = std::max(lhs.m() / threads, 1UL);
  size_t row_start = id * delta;
  size_t row_end = std::min(lhs.m(), (id + 1) * delta);
  // if too much threads
  if (row_start >= lhs.m())
  {
    return;
  }

  for (size_t i = row_start; i < row_end; i++)
  {
    for (size_t j = 0; j < rhs.n(); j++)
    {
      // for O[i, j]
      Matrix::data_t cell_out = 0;
      // accumulate L[i, k] * R[k, j]
      for (size_t k = 0; k < lhs.n() /* == rhs.m() */; k++)
      {
        cell_out += lhs(i, k) * rhs(k, j);
      }
      // set value
      out(i, j) = cell_out;
    }
  }
}
