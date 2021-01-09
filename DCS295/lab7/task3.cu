#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "errors.hpp"
#include "matrix.hpp"
#include <cassert>
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
                       "Error: {}:{} ({}) {}\n",     \
                       __FILE__, __LINE__, #E,       \
                       cudaGetErrorString(_status)); \
            exit(EXIT_FAILURE);                      \
        }                                            \
    }
#define CUDNN_GUARD(E)                                \
    {                                                 \
        auto _status = E;                             \
        if (_status != CUDNN_STATUS_SUCCESS)          \
        {                                             \
            fmt::print(stderr,                        \
                       fg(fmt::color::red),           \
                       "Error: {}:{} ({}) {}\n",      \
                       __FILE__, __LINE__, #E,        \
                       cudnnGetErrorString(_status)); \
            exit(EXIT_FAILURE);                       \
        }                                             \
    }

#define POS(m, n, p, i, j, k) \
    ((i) * (n) * (p) + (j) * (p) + (k))

Matrix conv_2d(const Matrix &in, const Matrix &ker, size_t stride, size_t bs_x, size_t bs_y)
{
    // copy device memory
    Matrix::data_t *dev_in;
    Matrix::data_t *dev_ker;
    CUDA_GUARD(cudaMalloc(&dev_in, in.data_size()));
    CUDA_GUARD(cudaMalloc(&dev_ker, ker.data_size()));
    CUDA_GUARD(cudaMemcpy(dev_in, in._data, in.data_size(), cudaMemcpyHostToDevice));
    CUDA_GUARD(cudaMemcpy(dev_ker, ker._data, ker.data_size(), cudaMemcpyHostToDevice));

    assert(ker.m() == ker.n());
    size_t padding = ((ker.n() - 1) / 2) * 2;

    cudnnHandle_t cudnn;
    CUDNN_GUARD(cudnnCreate(&cudnn));

    cudnnTensorDescriptor_t in_desc;
    CUDNN_GUARD(cudnnCreateTensorDescriptor(&in_desc));
    CUDNN_GUARD(cudnnSetTensor4dDescriptor(in_desc, CUDNN_TENSOR_NHWC, CUDNN_DATA_DOUBLE, 1, in.p(), in.m(), in.n()));

    cudnnFilterDescriptor_t ker_desc;
    CUDNN_GUARD(cudnnCreateFilterDescriptor(&ker_desc));
    CUDNN_GUARD(cudnnSetFilter4dDescriptor(ker_desc, CUDNN_DATA_DOUBLE, CUDNN_TENSOR_NHWC, 1, ker.p(), ker.m(), ker.n()));

    size_t pad_h = padding;
    size_t pad_w = padding;
    size_t str_h = stride;
    size_t str_w = stride;
    size_t dil_h = 1;
    size_t dil_w = 1;

    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_GUARD(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_GUARD(cudnnSetConvolution2dDescriptor(conv_desc, pad_h, pad_w, str_h, str_w, dil_h, dil_w, CUDNN_CONVOLUTION, CUDNN_DATA_DOUBLE));

    int out_c;
    int out_m;
    int out_n;
    int out_p;

    CUDNN_GUARD(cudnnGetConvolution2dForwardOutputDim(conv_desc, in_desc, ker_desc, &out_c, &out_p, &out_m, &out_n));

    cudnnTensorDescriptor_t out_desc;
    CUDNN_GUARD(cudnnCreateTensorDescriptor(&out_desc));
    CUDNN_GUARD(cudnnSetTensor4dDescriptor(out_desc, CUDNN_TENSOR_NHWC, CUDNN_DATA_DOUBLE, out_c, out_p, out_m, out_n));

    Matrix out(out_m, out_n, out_p);
    Matrix::data_t *dev_out;
    CUDA_GUARD(cudaMalloc(&dev_out, out.data_size()));

    cudnnConvolutionFwdAlgoPerf_t perf;
    int perf_count;
    CUDNN_GUARD(cudnnGetConvolutionForwardAlgorithm_v7(
        cudnn,
        in_desc, ker_desc, conv_desc, out_desc,
        1, &perf_count, &perf));

    size_t ws_len;
    CUDNN_GUARD(cudnnGetConvolutionForwardWorkspaceSize(
        cudnn,
        in_desc, ker_desc, conv_desc, out_desc,
        perf.algo, &ws_len));

    Matrix::data_t *ker_ws;
    CUDA_GUARD(cudaMalloc(&ker_ws, ws_len));

    Matrix::data_t alpha = 1.f;
    Matrix::data_t beta = 0.f;
    CUDNN_GUARD(cudnnConvolutionForward(
        cudnn, &alpha,
        in_desc, dev_in, ker_desc, dev_ker, conv_desc,
        perf.algo, ker_ws, ws_len,
        &beta, out_desc, dev_out));
    CUDA_GUARD(cudaMemcpy(out._data, dev_out, out.data_size(), cudaMemcpyDeviceToHost));

    CUDA_GUARD(cudaFree(dev_in));
    CUDA_GUARD(cudaFree(dev_ker));
    CUDA_GUARD(cudaFree(dev_out));
    CUDA_GUARD(cudaFree(ker_ws));
    CUDNN_GUARD(cudnnDestroy(cudnn));
    CUDNN_GUARD(cudnnDestroyConvolutionDescriptor(conv_desc));
    CUDNN_GUARD(cudnnDestroyFilterDescriptor(ker_desc));
    CUDNN_GUARD(cudnnDestroyTensorDescriptor(in_desc));
    CUDNN_GUARD(cudnnDestroyTensorDescriptor(out_desc));

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
