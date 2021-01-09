#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include"matrix.h"
#include <cudnn.h>
#include <vector>
#include <chrono>
#define IDX2C(i,j,ld) (((i)*(ld))+(j))
#define CUDA_CALL(f) { \
  cudaError_t err = (f); \
  if (err != cudaSuccess) { \
    std::cout \
        << __LINE__ <<":Error occurred: " << err<< std::endl; \
    std::exit(1); \
  } \
}

#define CUDNN_CALL(f) { \
  cudnnStatus_t err = (f); \
  if (err != CUDNN_STATUS_SUCCESS) { \
    std::cout \
        << "    Error occurred: " << err << std::endl; \
    std::exit(1); \
  } \
}
template<typename T>
void padding(vector<T*>& mat, int& col, int& row, int pad_size, int channels) {
    col += 2 * pad_size;
    row += 2 * pad_size;
    vector<T*>input(channels);
#pragma omp parallel for
    for (int i = 0; i < channels; i++)
    {
        input[i] = new T[col * row];
        memset(&input[i][0], 0, col * row * sizeof(T));
        for (int k = pad_size; k < col - pad_size; k++) {
            for (int j = pad_size; j < row - pad_size; j++) {
                input[i][IDX2C(k, j, row)] = mat[i][IDX2C(k - pad_size, j - pad_size, row - 2 * pad_size)];
            }
        }
        delete[]mat[i];
    }
    mat = input;
}

template<typename T>
__global__ void conv2d(T* mat, int* kernel, T* res, int col, int row, int stride, int kernel_size, int res_col, int res_row) {
    const int i = threadIdx.x + blockDim.x * blockIdx.x;
    const int j = threadIdx.y + blockDim.y * blockIdx.y;
    T sum = 0;
    if (i < res_col && j < res_row) {
        for (int x = 0; x < kernel_size; x++)
        {
            for (int y = 0; y < kernel_size; y++)
            {
                sum += mat[IDX2C(i * stride + x, j * stride + y, row)] * kernel[IDX2C(x, y, kernel_size)];
            }
        }
        res[IDX2C(i, j, res_row)] = sum;
    }
}

int main(int argc, char* argv[]) {
    if (argc != 7) {
        fprintf(stderr, "usage: TARGET [in_channels] [height] [width] [stride] [thread.x] [thread.y]\n");
        return -1;
    }
    int col = atoi(argv[2]),
        row = atoi(argv[3]),
        kernel_size = 3,
        stride = atoi(argv[4]),
        n_channels = atoi(argv[1]),
        thread_x = atoi(argv[5]), thread_y = atoi(argv[6]);

    int pad_size = 0;
    int res_col;
    int res_row;
    dim3 numBlocks;
    //pad_size = (kernel_size - 1) / 2;
    auto threadsPerBlock = dim3(thread_x, thread_y);
    auto input = getMat<float>(r, col, row, n_channels);
    vector<float*> d_input(n_channels, nullptr);
    vector<float*> output(n_channels, nullptr), d_output(n_channels, nullptr);
    vector<int*> d_kernel(n_channels, nullptr);
    vector<vector<int>>kernel(n_channels, vector<int>(kernel_size * kernel_size));
    padding(input, col, row, pad_size, n_channels);
    res_col = (col - kernel_size) / stride + 1;
    res_row = (row - kernel_size) / stride + 1;
    for (int i = 0; i < n_channels; i++)
    {

        //print(std::cout, input[i], col, row );

        numBlocks = dim3(res_col, res_row);

        output[i] = new float[res_col * res_row];
        kernel[i] = { 0,1,0,1,-4,1,0,1,0 };
        CUDA_CALL(cudaMalloc(&d_input[i], col * row * sizeof(d_input[0][0])));
        CUDA_CALL(cudaMalloc(&d_output[i], res_col * res_row * sizeof(d_output[0][0])));
        CUDA_CALL(cudaMalloc(&d_kernel[i], kernel_size * kernel_size * sizeof(kernel[0][0])));
        CUDA_CALL(cudaMemcpy(d_input[i], input[i], sizeof(float) * col * row, cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy(d_kernel[i], &kernel[i][0], sizeof(int) * kernel_size * kernel_size, cudaMemcpyHostToDevice));
    }

    auto timeStart = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < n_channels; i++)
    {
        conv2d << <numBlocks, threadsPerBlock >> > (d_input[i], d_kernel[i], d_output[i], col, row, stride, kernel_size, res_col, res_row);
    }
    auto timeEnd = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < n_channels; i++) {
        CUDA_CALL(cudaMemcpy(output[i], d_output[i], sizeof(float) * res_col * res_row, cudaMemcpyDeviceToHost));
        //print(std::cout, output[i], res_col, res_row);
    }

    auto passedTime = std::chrono::duration<float, std::milli>(timeEnd - timeStart).count();
    fprintf(stdout, "Conv2d Done: %.5f (ms)\n", passedTime);

    for (int i = 0; i < 3; i++) {
        cudaFree(&d_input[i]);
        cudaFree(&d_output[i]);
        cudaFree(&d_kernel[i]);
    }
    //save_img("2.png", output, res_col, res_row);
}