#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include"matrix.h"
#include <cudnn.h>
#include <vector>
#include <chrono>
#include <initializer_list>


template<typename T>
std::ostream& print(std::ostream& os, T* mat, int n, int c, int h, int w) {
    std::vector<T> buffer(n * c * h * w);
    cudaMemcpy(buffer.data(), mat, n * c * h * w * sizeof(T), cudaMemcpyDeviceToHost);
    int a = 0;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < c; ++j) {
            os << "n = " << i << ", c = " << j << ":" << std::endl;
            print(os, &buffer[a], h, w);
            a += h * w;
        }
    }
}

template<typename T>
T* getKernelGptr(std::initializer_list<T> filter, int filt_c, int filt_h, int filt_w) {
    vector<T>kernel(filter);
    T* filt_data;
    cudaMalloc(&filt_data, filt_c * filt_h * filt_w * sizeof(T));
    for (int i = 0; i < filt_c; i++)
        cudaMemcpy(filt_data+ filt_h * filt_w * i, &kernel[0], sizeof(kernel[0]) * kernel.size(), cudaMemcpyHostToDevice);
    return filt_data;
}
template<typename T>
T* getInputGptr(vector<T*> &input, int in_c, int in_h, int in_w) {
    T* in_data;
    cudaMalloc(&in_data, in_c * in_h * in_w * sizeof(T));
    for (int i = 0; i < in_c; i++)
        cudaMemcpy(in_data + in_h * in_w * i, &input[i][0], in_w * in_h * sizeof(T), cudaMemcpyHostToDevice);
    return in_data;
}

int main(int argc, char* argv[]){
    if (argc != 6) {
        fprintf(stderr, "usage: TARGET [in_channels] [height] [width] [stride] [padding]\n");
        return -1;
    }
    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);

    int stride = atoi(argv[4]), padding = atoi(argv[5]);
    int in_n = 1,in_c = atoi(argv[1]),in_h = atoi(argv[2]),in_w = atoi(argv[3]);

    int filt_k = in_n,filt_c = in_c,filt_h = 3,filt_w = 3;

    auto filt_data = getKernelGptr<float>({ 0, 1, 0, 1, -4, 1, 0, 1, 0 }, filt_c, filt_h, filt_w);
    auto in_data = getInputGptr(getMat<float>(r, in_h, in_w, in_c), in_c, in_h, in_w);

    cudnnTensorDescriptor_t in_desc;
    cudnnCreateTensorDescriptor(&in_desc);
    cudnnSetTensor4dDescriptor(in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, in_n, in_c, in_h, in_w);


    cudnnFilterDescriptor_t filt_desc;
    cudnnCreateFilterDescriptor(&filt_desc);
    cudnnSetFilter4dDescriptor(filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, filt_k, filt_c, filt_h, filt_w);

    const int pad_h = padding;
    const int pad_w = padding;
    const int str_h = stride;
    const int str_w = stride;
    const int dil_h = 1;
    const int dil_w = 1;

    cudnnConvolutionDescriptor_t conv_desc;
    cudnnCreateConvolutionDescriptor(&conv_desc);
    cudnnSetConvolution2dDescriptor( conv_desc, pad_h, pad_w, str_h, str_w, dil_h, dil_w, CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT);

    int out_n;
    int out_c;
    int out_h;
    int out_w;

    cudnnGetConvolution2dForwardOutputDim(conv_desc, in_desc, filt_desc,&out_n, &out_c, &out_h, &out_w);

    std::cout << "out_n: " << out_n << std::endl;
    std::cout << "out_c: " << out_c << std::endl;
    std::cout << "out_h: " << out_h << std::endl;
    std::cout << "out_w: " << out_w << std::endl;
    std::cout << std::endl;

    cudnnTensorDescriptor_t out_desc;
    cudnnCreateTensorDescriptor(&out_desc);
    cudnnSetTensor4dDescriptor(out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, out_n, out_c, out_h, out_w);

    float* out_data;
    cudaMalloc(&out_data, out_n * out_c * out_h * out_w * sizeof(float));

    cudnnConvolutionFwdAlgo_t algo;
    cudnnGetConvolutionForwardAlgorithm(cudnn, in_desc, filt_desc, conv_desc, out_desc, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo);

    std::cout << "Convolution algorithm: " << algo << std::endl;
    std::cout << std::endl;

    size_t ws_size;
    cudnnGetConvolutionForwardWorkspaceSize(cudnn, in_desc, filt_desc, conv_desc, out_desc, algo, &ws_size);

    float* ws_data;
    cudaMalloc(&ws_data, ws_size);

    std::cout << "Workspace size: " << ws_size << std::endl;
    std::cout << std::endl;

    float alpha = 1.f;
    float beta = 0.f;
    auto timeStart = std::chrono::high_resolution_clock::now();
    cudnnConvolutionForward(cudnn, &alpha, in_desc, in_data, filt_desc, filt_data, conv_desc, algo, ws_data, ws_size, &beta, out_desc, out_data);
    auto timeEnd = std::chrono::high_resolution_clock::now();

    auto passedTime = std::chrono::duration<double, std::milli>(timeEnd - timeStart).count();
    fprintf(stdout, "Cuda Done: %.5f (ms)\n", passedTime);



    cudaFree(ws_data);
    cudaFree(out_data);
    cudnnDestroyTensorDescriptor(out_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudaFree(filt_data);
    cudnnDestroyFilterDescriptor(filt_desc);
    cudaFree(in_data);
    cudnnDestroyTensorDescriptor(in_desc);
    cudnnDestroy(cudnn);
    return 0;
}