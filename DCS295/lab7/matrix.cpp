#include "matrix.hpp"
#include <algorithm>
#include <random>
#include <stdexcept>

Matrix::Matrix() : Matrix(0) {}
Matrix::Matrix(size_t n) : Matrix(n, n, n) {}
Matrix::Matrix(size_t m, size_t n, size_t p)
{
  _m = m;
  _n = n;
  _p = p;
  _data = new Matrix::data_t[m * n * p];
}
Matrix::Matrix(size_t m, size_t n, size_t p, bool r) : Matrix(m, n, p)
{
  if (!r)
  {
    throw std::logic_error(
        "random constructor must pass third parameter as true");
  }
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<Matrix::data_t> dis(1, 2);

  for (size_t i = 0; i < m; i++)
    for (size_t j = 0; j < n; j++)
      for (size_t k = 0; k < p; k++)
        operator()(i, j, k) = dis(gen);
}
Matrix::Matrix(Matrix &&m)
{
  std::swap(m._data, _data);
  std::swap(m._m, _m);
  std::swap(m._n, _n);
  std::swap(m._p, _p);
}

Matrix &Matrix::operator=(Matrix &&m)
{
  std::swap(m._data, _data);
  std::swap(m._m, _m);
  std::swap(m._n, _n);
  std::swap(m._p, _p);
  return *this;
}

Matrix::~Matrix()
{
  if (_data != nullptr)
  {
    delete[] _data;
  }
}

size_t Matrix::m() const { return _m; }
size_t Matrix::n() const { return _n; }
size_t Matrix::p() const { return _p; }

void Matrix::resize(size_t m, size_t n, size_t p)
{
  if (m * n * p != _m * _n * _p)
    throw std::logic_error("inconsistent size");

  _m = m;
  _n = n;
  _p = p;
}

Matrix::data_t &Matrix::operator()(size_t i, size_t j, size_t k)
{
  if (i >= _m || j >= _n || k >= _p)
    throw std::logic_error("matrix visit out of bound");

  return _data[Matrix::access(_m, _n, _p, i, j, k)];
}
const Matrix::data_t &Matrix::operator()(size_t i, size_t j, size_t k) const
{
  if (i >= _m || j >= _n || k >= _p)
    throw std::logic_error("matrix visit out of bound");

  return _data[Matrix::access(_m, _n, _p, i, j, k)];
}

std::ostream &operator<<(std::ostream &os, const Matrix &m)
{
  os << m._m << " " << m._n << std::endl;
  for (size_t i = 0; i < m._m; i++)
  {
    for (size_t j = 0; j < m._n; j++)
    {
      for (size_t k = 0; k < m._p; k++)
      {
        os << m(i, j, k) << ",";
      }
      os << " ";
    }
    os << std::endl;
  }
  return os;
}

size_t Matrix::data_size() const
{
  return sizeof(data_t) * _m * _n * _p;
}
