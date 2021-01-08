#include "matrix.hpp"

Matrix product_cuda(const Matrix &L, const Matrix &R, size_t bs_x, size_t bs_y);
Matrix product_cuda_omp(const Matrix &L, const Matrix &R, size_t bs_x, size_t bs_y, size_t p);
Matrix product_cublas(const Matrix &L, const Matrix &R);
