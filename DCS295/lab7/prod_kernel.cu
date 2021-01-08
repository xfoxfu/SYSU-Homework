#include "prod_cuda.cuh"
#include "matrix.hpp"

__global__ void matrix_mult(Matrix::data_t *lhs, Matrix::data_t *rhs, Matrix::data_t *out, size_t M, size_t N, size_t K)
{
    const size_t i = threadIdx.x + blockDim.x * blockIdx.x;
    const size_t j = threadIdx.y + blockDim.y * blockIdx.y;
    Matrix::data_t sum = 0;
    if (i < M && j < K)
    {
        for (size_t k = 0; k < N; k++)
        {
            sum += lhs[i * N + k]  // lhs(i, k)
                 * rhs[j * K + k]; // rhs(k, j) => rhs'(j, k)
        }
        out[i * N + j] = sum; // out(i, j)
    }
}
