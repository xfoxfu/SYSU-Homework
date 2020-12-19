#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <cublas_v2.h>

#include "prod_cuda.cuh"
#include "prod_kernel.cuh"

#define CUDA_CHECKED_RUN(E)                              \
    {                                                    \
        auto _status = E;                                \
        if (_status != cudaSuccess)                      \
        {                                                \
            fprintf(stderr, "Error: %s at %s(%i): %s\n", \
                    #E, __FILE__, __LINE__,              \
                    cudaGetErrorString(_status));        \
            exit(EXIT_FAILURE);                          \
        }                                                \
    }

Matrix product_cuda(const Matrix &L, const Matrix &R, size_t bs_x, size_t bs_y)
{
    Matrix O(L.m(), R.n());
    // set device
    CUDA_CHECKED_RUN(cudaSetDevice(3));
    // allocate device memory
    Matrix::data_t *dev_l, *dev_r, *dev_o;
    CUDA_CHECKED_RUN(cudaMalloc(&dev_l, L.data_size()));
    CUDA_CHECKED_RUN(cudaMalloc(&dev_r, R.data_size()));
    CUDA_CHECKED_RUN(cudaMalloc(&dev_o, O.data_size()));
    // copy matrix to device
    cudaMemcpy(dev_l, L._data, L.data_size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_r, R._data, R.data_size(), cudaMemcpyHostToDevice);
    // perform product
    std::cout << "Computing: GRID(" << (O.m() / bs_x) << "," << (O.n() / bs_y) << ")"
              << ", BLOCK(" << bs_x << "," << bs_y << ")" << std::endl;
    dim3 grid(O.m() / bs_x, O.n() / bs_y);
    dim3 block(bs_x, bs_y);
    matrix_mult<<<grid, block>>>(dev_l, dev_r, dev_o, L.m(), L.n(), R.n());
    // copy result back
    CUDA_CHECKED_RUN(cudaMemcpy(O._data, dev_o, O.data_size(), cudaMemcpyDeviceToHost));
    // free memory
    CUDA_CHECKED_RUN(cudaFree(dev_l));
    CUDA_CHECKED_RUN(cudaFree(dev_r));
    CUDA_CHECKED_RUN(cudaFree(dev_o));

    return O;
}

Matrix product_cuda_omp(const Matrix &L, const Matrix &R, size_t bs_x, size_t bs_y, size_t p)
{
    Matrix O(L.m(), R.n());
    // set device
    CUDA_CHECKED_RUN(cudaSetDevice(3));
    // allocate device memory
    Matrix::data_t *dev_l, *dev_r, *dev_o;
    CUDA_CHECKED_RUN(cudaMalloc(&dev_l, L.data_size()));
    CUDA_CHECKED_RUN(cudaMalloc(&dev_r, R.data_size()));
    CUDA_CHECKED_RUN(cudaMalloc(&dev_o, O.data_size()));
    // copy matrix to device
    cudaMemcpy(dev_l, L._data, L.data_size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_r, R._data, R.data_size(), cudaMemcpyHostToDevice);
    // perform product
    std::cout << "Computing: GRID(" << (O.m() / bs_x / p) << "," << (O.n() / bs_y) << ")"
              << ", BLOCK(" << bs_x << "," << bs_y << ")"
              << " with OpenMP = " << p << std::endl;
    dim3 grid(O.m() / bs_x / p, O.n() / bs_y);
    dim3 block(bs_x, bs_y);
#pragma omp parallel for num_threads(p)
    for(size_t tid = 0; tid < p; tid++)
    {
      size_t offset = grid.x * bs_x * tid * L.n();
      matrix_mult<<<grid, block>>>(dev_l + offset, dev_r, dev_o + grid.x * bs_x * tid * L.n(), L.m() / p, L.n(), R.n());
    }
    // copy result back
    CUDA_CHECKED_RUN(cudaMemcpy(O._data, dev_o, O.data_size(), cudaMemcpyDeviceToHost));
    // free memory
    CUDA_CHECKED_RUN(cudaFree(dev_l));
    CUDA_CHECKED_RUN(cudaFree(dev_r));
    CUDA_CHECKED_RUN(cudaFree(dev_o));

    return O;
}

Matrix product_cublas(const Matrix &L, const Matrix &R)
{
    Matrix O(L.m(), R.n());
    // set device
    CUDA_CHECKED_RUN(cudaSetDevice(3));
    // allocate device memory
    Matrix::data_t *dev_l, *dev_r, *dev_o;
    CUDA_CHECKED_RUN(cudaMalloc(&dev_l, L.data_size()));
    CUDA_CHECKED_RUN(cudaMalloc(&dev_r, R.data_size()));
    CUDA_CHECKED_RUN(cudaMalloc(&dev_o, O.data_size()));
    // perform product
    cublasHandle_t handle;
    double alpha = 1.0;
    double beta = 1.0;
    cublasCreate(&handle);
    cublasSetMatrix(L.m(), L.n(), sizeof(Matrix::data_t), L._data, L.n(), dev_l, L.n());
    cublasSetMatrix(R.m(), R.n(), sizeof(Matrix::data_t), R._data, R.m(), dev_r, R.m());
    cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, L.m(), L.n(), R.n(), &alpha, dev_r, L.m(), dev_l, R.n(), &beta, dev_o, L.m());
    // copy result back
    cublasGetMatrix(O.m(), O.n(), sizeof(Matrix::data_t), dev_o, L.n(), O._data, L.n());
    // free memory
    CUDA_CHECKED_RUN(cudaFree(dev_l));
    CUDA_CHECKED_RUN(cudaFree(dev_r));
    CUDA_CHECKED_RUN(cudaFree(dev_o));

    return O;
}
