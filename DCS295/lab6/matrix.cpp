#include "matrix.hpp"
#include <algorithm>
#include <random>
#include <stdexcept>

Matrix::Matrix() : Matrix(0) { _need_free = false; }
Matrix::Matrix(size_t n) : Matrix(n, n) {}
Matrix::Matrix(size_t m, size_t n)
{
  _m = m;
  _n = n;
  _data = new Matrix::data_t[m * n];
  _need_free = true;
}
Matrix::Matrix(size_t m, size_t n, bool r) : Matrix(m, n)
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
  {
    for (size_t j = 0; j < n; j++)
    {
      operator()(i, j) = dis(gen);
    }
  }
}
Matrix::Matrix(Matrix &&m)
{
  std::swap(m._data, _data);
  std::swap(m._m, _m);
  std::swap(m._n, _n);
  m._need_free = false;
}
Matrix::Matrix(data_t *data, size_t m, size_t n)
{
  _data = data;
  _m = m;
  _n = n;
  _need_free = false;
}

Matrix &Matrix::operator=(Matrix &&m)
{
  std::swap(m._data, _data);
  std::swap(m._m, _m);
  std::swap(m._n, _n);
  m._need_free = false;
  return *this;
}

Matrix::~Matrix()
{
  if (_need_free)
  {
    delete[] _data;
  }
}

size_t Matrix::m() const { return _m; }
size_t Matrix::n() const { return _n; }

Matrix::data_t &Matrix::operator()(size_t i, size_t j)
{
  if (i >= _m || j >= _n)
  {
    throw std::logic_error("matrix visit out of bound");
  }
  // std::cout << "M(" << i << ", " << j << ") = " << _data[i * _n + j]
  // << std::endl;
  return _data[i * _n + j];
}
const Matrix::data_t &Matrix::operator()(size_t i, size_t j) const
{
  if (i >= _m || j >= _n)
  {
    throw std::logic_error("matrix visit out of bound");
  }
  // std::cout << "M(" << i << ", " << j << ") = " << _data[i * _n + j]
  // << std::endl;
  return _data[i * _n + j];
}
Matrix Matrix::operator+(const Matrix &r) const
{
  if (n() != r.n() || m() != r.m())
  {
    throw std::logic_error("inconsistent matrix size A");
  }
  Matrix o(m(), n());
  for (size_t i = 0; i < m(); i++)
  {
    for (size_t j = 0; j < n(); j++)
    {
      o(i, j) = operator()(i, j) + r(i, j);
    }
  }
  return o;
}
Matrix Matrix::operator-(const Matrix &r) const
{
  if (n() != r.n() || m() != r.m())
  {
    throw std::logic_error("inconsistent matrix size S");
  }
  Matrix o(m(), n());
  for (size_t i = 0; i < m(); i++)
  {
    for (size_t j = 0; j < n(); j++)
    {
      o(i, j) = operator()(i, j) - r(i, j);
    }
  }
  return o;
}
std::ostream &operator<<(std::ostream &os, const Matrix &m)
{
  os << m._m << " " << m._n << std::endl;
  for (size_t i = 0; i < m._m; i++)
  {
    for (size_t j = 0; j < m._n; j++)
    {
      os << (j > 0 ? " " : "") << m(i, j);
    }
    os << std::endl;
  }
  return os;
}

bool Matrix::is_consistent_product(const Matrix &r) const
{
  return n() == r.m();
}
void Matrix::ensure_consistent_product(const Matrix &r) const
{
  if (!is_consistent_product(r))
  {
    throw std::logic_error("inconsistent matrix size P");
  }
}

Matrix Matrix::clone() const
{
  Matrix r(_m, _n);
  std::copy(_data, &_data[_m * _n], r._data);
  return r;
}

size_t Matrix::data_size() const
{
  return sizeof(data_t) * _m * _n;
}
