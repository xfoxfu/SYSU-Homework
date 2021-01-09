#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "errors.hpp"
#include "matrix.hpp"
#include <chrono>
#include <cudnn.h>
#include <fmt/color.h>
#include <fmt/core.h>
#include <fmt/ostream.h>
#include <vector>

#define CUDA_GUARD(E)                                \
    {                                                \
        auto _status = E;                            \
        if (_status != cudaSuccess)                  \
        {                                            \
            fmt::print(stderr,                       \
                       fg(fmt::color::red),          \
                       "Error: {}:{} ({}) {}",       \
                       __FILE__, __LINE__, #E,       \
                       cudaGetErrorString(_status)); \
            exit(EXIT_FAILURE);                      \
        }                                            \
    }

#define POS(m, n, p, i, j, k) \
    ((i) * (n) * (p) + (j) * (p) + (k))

__global__ void dev_conv2d(Matrix::data_t *in, Matrix::data_t *ker, Matrix::data_t *out,
                           uint32_t in_m, uint32_t in_n, uint32_t in_p,
                           uint32_t ker_m, uint32_t ker_n, uint32_t ker_p,
                           uint32_t out_m, uint32_t out_n, uint32_t out_p,
                           uint32_t stride)
{
    const int x = threadIdx.x + blockDim.x * blockIdx.x;
    const int y = threadIdx.y + blockDim.y * blockIdx.y;
    Matrix::data_t sum = 0;
    if (x < out_m && y < out_n)
    {
        for (uint32_t i = 0; i < ker_m; i++)
        {
            for (uint32_t j = 0; j < ker_n; j++)
            {
                uint32_t bi = x * stride;
                uint32_t bj = y * stride;
                int32_t di = i - ker_m / 2;
                int32_t dj = j - ker_n / 2;
                if ((int32_t)bi + di >= 0 && (int32_t)bi + di < in_m &&
                    (int32_t)bj + dj >= 0 && (int32_t)bj + dj < in_n)
                {
                    for (uint32_t k = 0; k < ker_p; k++)
                    {
                        sum +=
                            in[POS(in_m, in_n, in_p, bi + di, bj + dj, k)] *
                            ker[POS(ker_m, ker_n, ker_p, i, j, k)];
                    }
                }
            }
        }
        out[POS(out_m, out_n, out_p, x, y, 0)] = sum;
    }
}

Matrix conv_2d(const Matrix &input, const Matrix &kernel, size_t stride, size_t bs_x, size_t bs_y)
{
    size_t out_n = (input.n() + ((kernel.n() - 1) / 2) * 2 - kernel.n()) / stride + 1;
    size_t out_m = (input.m() + ((kernel.m() - 1) / 2) * 2 - kernel.m()) / stride + 1;
    Matrix out(out_m, out_n, 1);

    // allocate device memory
    Matrix::data_t *dev_in;
    CUDA_GUARD(cudaMalloc(&dev_in, input.data_size()));
    Matrix::data_t *dev_kernel;
    CUDA_GUARD(cudaMalloc(&dev_kernel, kernel.data_size()));
    Matrix::data_t *dev_out;
    CUDA_GUARD(cudaMalloc(&dev_out, out.data_size()));

    // copy input
    CUDA_GUARD(cudaMemcpy(dev_in, input._data, input.data_size(), cudaMemcpyHostToDevice));
    CUDA_GUARD(cudaMemcpy(dev_kernel, kernel._data, kernel.data_size(), cudaMemcpyHostToDevice));

    // compute
    dim3 grid(out.m() / bs_x, out.n() / bs_y);
    dim3 block(bs_x, bs_y);
    dev_conv2d<<<grid, block>>>(dev_in, dev_kernel, dev_out,
                                input.m(), input.n(), input.p(),
                                kernel.m(), kernel.n(), kernel.p(),
                                out.m(), out.n(), out.p(),
                                stride);

    // copy result
    CUDA_GUARD(cudaMemcpy(out._data, dev_out, out.data_size(), cudaMemcpyDeviceToHost));

    // free device memory
    CUDA_GUARD(cudaFree(dev_in));
    CUDA_GUARD(cudaFree(dev_kernel));
    CUDA_GUARD(cudaFree(dev_out));

    // sync
    CUDA_GUARD(cudaDeviceSynchronize());
    return out;
}

int main(int argc, char *argv[])
{
    if (argc <= 6)
    {
        fmt::print(stderr, fg(fmt::color::red),
                   "usage: {} <height> <width> <depth> <stride> <thread.x> <thread.y> [--output]\n", argv[0]);
        return CNN_INVALID_ARGUMENTS;
    }
    size_t height = std::stoull(argv[1]);
    size_t width = std::stoull(argv[2]);
    constexpr size_t depth = 3;
    constexpr size_t filter_size = 3;
    constexpr size_t filter_count = 3;
    size_t stride = std::stoull(argv[4]);
    size_t thread_x = std::stoull(argv[5]);
    size_t thread_y = std::stoull(argv[6]);
    bool has_output = false;
    if (argc > 7 && std::strcmp(argv[7], "--output") == 0)
    {
        has_output = true;
    }

    fmt::print(fg(fmt::color::blue), "generating input\n");
    Matrix input = Matrix(height, width, depth, true);
    if (has_output)
    {
        fmt::print("{}\n", input);
    }
    fmt::print(fg(fmt::color::blue), "generating kernel\n");
    Matrix kernels[filter_count];
    for (size_t i = 0; i < filter_count; i++)
    {
        kernels[i] = Matrix(filter_size, filter_size, filter_size, true);
        if (has_output)
        {
            fmt::print("{}\n", kernels[i]);
        }
    }

    // perform convolution
    auto start = std::chrono::high_resolution_clock::now();

    fmt::print(fg(fmt::color::blue), "compute conv_2d x{}\n", filter_count);
    Matrix R[filter_count];
    for (size_t i = 0; i < filter_count; i++)
    {
        R[i] = conv_2d(input, kernels[i], stride, thread_x, thread_y);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> diff = end - start;
    fmt::print(fg(fmt::color::orange), "time: {} ms\n", diff.count());
    if (has_output)
    {
        for (size_t i = 0; i < filter_count; i++)
        {
            fmt::print("{}\n", R[i]);
        }
    }

    return CNN_OK;
}
