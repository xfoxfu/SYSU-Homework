#include "matrix.hpp"
#include <algorithm>
#include <random>
#include <stdexcept>

Matrix::Matrix() : Matrix(0) {}
Matrix::Matrix(size_t n) : Matrix(n, n) {}
Matrix::Matrix(size_t m, size_t n) {
  _m = m;
  _n = n;
  _data = new Matrix::data_t[m * n];
}
Matrix::Matrix(size_t m, size_t n, bool r) : Matrix(m, n) {
  if (!r) {
    throw std::logic_error(
        "random constructor must pass third parameter as true");
  }
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<Matrix::data_t> dis(1.0, 100.0);
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < n; j++) {
      operator()(i, j) = dis(gen);
    }
  }
}
Matrix::Matrix(Matrix &&m) {
  std::swap(m._n, _n);
  std::swap(m._m, _m);
  std::swap(m._data, _data);
}

Matrix &Matrix::operator=(Matrix &&m) {
  std::swap(m._n, _n);
  std::swap(m._m, _m);
  std::swap(m._data, _data);
  return *this;
}

Matrix::~Matrix() { delete[] _data; }

size_t Matrix::m() const { return _m; }
size_t Matrix::n() const { return _n; }

Matrix::data_t &Matrix::operator()(size_t i, size_t j) {
  return _data[i * _n + j];
}
const Matrix::data_t &Matrix::operator()(size_t i, size_t j) const {
  return _data[i * _n + j];
}
Matrix Matrix::operator+(const Matrix &r) const {
  if (_n != r._n || _m != r._m) {
    throw std::logic_error("inconsistent matrix size");
  }
  Matrix m(_m, _n);
  for (size_t i = 0; i < _m; i++) {
    for (size_t j = 0; j < _n; j++) {
      m(i, j) = operator()(i, j) + r(i, j);
    }
  }
  return m;
}
Matrix &Matrix::operator+=(const Matrix &r) {
  if (_n != r._n || _m != r._m) {
    throw std::logic_error("inconsistent matrix size");
  }
  for (size_t i = 0; i < _m; i++) {
    for (size_t j = 0; j < _n; j++) {
      operator()(i, j) += r(i, j);
    }
  }
  return *this;
}
Matrix Matrix::operator-(const Matrix &r) const {
  if (_n != r._n || _m != r._m) {
    throw std::logic_error("inconsistent matrix size");
  }
  Matrix m(_m, _n);
  for (size_t i = 0; i < _m; i++) {
    for (size_t j = 0; j < _n; j++) {
      m(i, j) = operator()(i, j) - r(i, j);
    }
  }
  return m;
}
Matrix &Matrix::operator-=(const Matrix &r) {
  if (_n != r._n || _m != r._m) {
    throw std::logic_error("inconsistent matrix size");
  }
  for (size_t i = 0; i < _m; i++) {
    for (size_t j = 0; j < _n; j++) {
      operator()(i, j) -= r(i, j);
    }
  }
  return *this;
}
std::ostream &operator<<(std::ostream &os, const Matrix &m) {
  os << m._m << " " << m._n << std::endl;
  for (size_t i = 0; i < m._m; i++) {
    for (size_t j = 0; j < m._n; j++) {
      os << " "[!j] << m(i, j);
    }
    os << std::endl;
  }
  return os;
}

bool Matrix::is_consistent_product(const Matrix &r) const { return _n == r._m; }
void Matrix::ensure_consistent_product(const Matrix &r) const {
  if (!is_consistent_product(r)) {
    throw std::logic_error("inconsistent matrix size");
  }
}

Matrix Matrix::clone() const {
  Matrix r(_m, _n);
  std::copy(_data, &_data[_m * _n], r._data);
  return r;
}
