#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include"matrix.h"
#include <cudnn.h>
#include <omp.h>
#include <vector>
#include <chrono>
#define IDX2C(i,j,ld) (((i)*(ld))+(j))
template<typename T>
__global__ void matrix_mult(T* a, T* b, T* c, int Acol, int Arow, int Brow)
{
	const int x = threadIdx.x + blockDim.x * blockIdx.x;
	const int y = threadIdx.y + blockDim.y * blockIdx.y;
	const auto index = x * Brow + y;
	T sum = 0;
	if (x < Acol && y < Brow) {
		for (int k = 0; k < Arow; k++)
		{
			sum += a[x * Arow + k] * b[k * Brow + y];
		}
		c[index] = sum;
	}
}

template<typename T>
T* gemm(T* a, T* b, int Acol, int Arow, int Brow, int THREAD_X = 128, int THREAD_Y = 1)
{

	const int testsize = Acol * Brow;
	auto* c = new T[testsize];
	T* d_a, * d_b, * d_c;

	cudaMalloc(&d_a, sizeof(T) * Acol * Arow);
	cudaMalloc(&d_b, sizeof(T) * Arow * Brow);
	cudaMalloc(&d_c, sizeof(T) * testsize);
	cudaMemcpy(d_a, a, sizeof(T) * Acol * Arow, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, sizeof(T) * Arow * Brow, cudaMemcpyHostToDevice);
	dim3 griddim(Acol / THREAD_X, Brow / THREAD_Y);
	dim3 threadsPerBlock(THREAD_X, THREAD_Y);
	matrix_mult << <griddim, threadsPerBlock >> > (d_a, d_b, d_c, Acol, Arow, Brow);
	cudaMemcpy(c, d_c, sizeof(T) * testsize, cudaMemcpyDeviceToHost);
	//string filename = "x_" + to_string(THREAD_X) + "_y_" + to_string(THREAD_Y) + "_N_" + to_string(Acol) + ".txt";
	//ofstream outfileC(filename.c_str());
	//print(outfileC, c, Acol, Brow);
	//outfileC.close();

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	return c;
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
T* im2col(vector<T*>& mat, int col, int row, int kernel_size, int pad_size, int channels, int stride, int& res_col, int& res_row) {
	padding(mat, col, row, pad_size, channels);
	int height = (col - kernel_size) / stride + 1;
	int width = (row - kernel_size) / stride + 1;
	int elems = channels * kernel_size * kernel_size;
	res_col = elems; res_row = height * width;
	T* res = new T[res_col * res_row];
#pragma omp parallel for
	for (int c = 0; c < elems; c++)
	{
		int w_offset = c % kernel_size;
		int h_offset = (c / kernel_size) % kernel_size;
		int c_offset = c / kernel_size / kernel_size;
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				int im_col = h_offset + i * stride;
				int im_row = w_offset + j * stride;
				res[IDX2C((height * c + i), j, width)] = mat[c_offset][IDX2C(im_col, im_row, row)];
			}
		}
	}
	return res;
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


	int pad_size;
	dim3 numBlocks;
	pad_size = 0;
	//pad_size = (kernel_size-1) / 2;
	auto mat = getMat<float>(txt, col, row, n_channels);
	int Acol, Arow;
	vector<float> kernelVec = { 0, 1, 0, 1, -4, 1, 0, 1, 0 };
	float* kernel = new float[kernel_size * kernel_size * n_channels];
	for (int i = 0; i < n_channels; i++)
	{
		memcpy(kernel + kernel_size * kernel_size * i, &kernelVec[0], sizeof(kernelVec[0]) * kernelVec.size());
		//print(std::cout, mat[i], col, row);
	}
	auto timeStart = std::chrono::high_resolution_clock::now();

	auto a = im2col(mat, col, row, kernel_size, pad_size, n_channels, stride, Acol, Arow);
	//print(std::cout, a, Acol, Arow);
	//print(std::cout, kernel, 1, Acol);

	auto res = gemm(kernel, a, 1, Acol, Arow, thread_x, thread_y);
	auto timeEnd = std::chrono::high_resolution_clock::now();
	auto passedTime = std::chrono::duration<double, std::milli>(timeEnd - timeStart).count();
	fprintf(stdout, "im2col Done: %.5f (ms)\n", passedTime);

	int res_col = (col - kernel_size) / stride + 1;
	int res_row = (row - kernel_size) / stride + 1;
	//print(std::cout, res, res_col, res_row);
}