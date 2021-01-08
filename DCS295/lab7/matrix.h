#include <iomanip>
#include <iostream>
#include <fstream>
#include <random>
#include <string.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "stb_image.h"
using std::vector;
std::random_device rd;
// std::mt19937 gen(0);
std::mt19937 gen(0);
#define PATH "1.txt"
//#define PATH "test.png"

enum Mat_src
{
    r, img, txt
};


template<typename T>
T &get(T *mat, int col, int row, int i, int j)
{
    if (i >= col || j >= row)
        throw std::out_of_range("line 17:index out of range");
    return mat[i * row + j];
}
template<typename T>
void random_fill(T *&mat, int col, int row,int upperbound=255)
{
    for (int i = 0; i < row * col; i++)
    {
        mat[i] = gen() % upperbound;
    }
}
template<typename T>
std::ostream &print(std::ostream &os, T *mat, int col, int row, const char *end = "\n")
{
    std::ios_base::fmtflags oldFlags = os.flags();
    int width = 3;   // total with o          f the displayed number
    os.precision(5); // control the number of displayed decimals
    os.setf(std::ios_base::fixed);
    os << "[";
    for (int i = 0; i < col; i++)
    {
        for (int j = 0; j < row; j++)
        {
            if (i == j && j == 0)
                width = 2;
            else
                width = 3;
            os << std::setw(width) << get(mat, col, row, i, j) << " ";
        }
        if (i != col - 1)
            os << std::endl;
    }
    os << "]" << end;
    os.flags(oldFlags);
    return os;
}

template<typename T>
void transpose(T *&mat, int col, int row)
{
    auto *tmp = new T[col * row];
    for (int i = 0; i < col; i++)
    {
        for (int j = 0; j < row; j++)
        {
            get(tmp, row, col, i, j) = get(mat, col, row, j, i);
        }
    }
    delete[]mat;
    mat = tmp;
}
template<typename T>
T* multiply(T* matA, T* matB, int Acol, int Arow, int Bcol, int Brow) {
    T* res = new T[Brow * Acol];

    for (int i = 0; i < Acol; i++)
    {
        for (int j = 0; j < Brow; j++)
        {
            get(res, Acol, Brow, i, j) = 0;
            for (int k = 0; k < Arow; k++)
                get(res, Acol, Brow, i, j) += get(matA, Acol, Arow, i, k) * get(matB, Bcol, Brow, k, j);
        }
    }
    return res;
}

template<typename T>
vector<T*> random_mat(int col,int row,int n_channels) {
    vector<T*>mat(n_channels);
    for (int i = 0; i < n_channels; i++)
    {
        mat[i] = new T[col * row];
        random_fill(mat[i], col, row,10);
    }
    return mat;
}

template<typename T>
vector<T*> txt_mat(const char* filename,int col,int row,int n_channels) {
    vector<T*>mat(n_channels, nullptr);
    std::ifstream infile(filename);
    for (int i = 0; i < n_channels; i++)
    {
        mat[i] = new T[col * row];
        for (int j = 0; j < col*row; j++)
        {
            infile >> mat[i][j];
        }
    }
    return mat;
}

template<typename T>
vector<T*> img_mat(const char*filename,int &width,int &height, int &components_per_pixel) {
    unsigned char* data = stbi_load(
        filename, &width, &height, &components_per_pixel, components_per_pixel);

    if (!data) {
        std::cerr << "ERROR: Could not load image file '" << filename << "'.\n";
        width = height = 0;
    }
    vector<T*>vec(components_per_pixel,nullptr);
    for (int j = 0; j < components_per_pixel; j++)
    {
        vec[j] = new T[width * height];
    }
    for (int i = 0; i < height*width; i++)
    {
        for (int j = 0; j < components_per_pixel; j++)
        {
            vec[j][i] = data[i*components_per_pixel + j];
        }
    }

    return vec;
}

template<typename T>
void save_img(char* filename, vector<T*>img_mat,int height,int width) {
    auto * data = new unsigned char[height * width*4];
    for (int i = 0; i < height* width; i++)
    {
        data[i * 4 + 0]= (unsigned char)img_mat[0][i];
        data[i * 4 + 1]= (unsigned char)img_mat[1][i];
        data[i * 4 + 2]= (unsigned char)img_mat[2][i];
        data[i * 4 + 3]= (unsigned char)255;
    }
    stbi_write_png(filename, height, width, 4, static_cast<void*>(data), height * 4);
}

template<typename T>
vector<T*> getMat(Mat_src c, int& col, int& row,int &n_channels) {
    vector<T*>ret(n_channels, nullptr);
    switch (c)
    {
    case r:
        return random_mat<T>(col, row,n_channels);
    case img:
        return img_mat<T>(PATH, col, row,n_channels);
    case txt:
        return txt_mat<T>(PATH,col, row,n_channels);
    default:
        break;
    }
}
