#pragma once

#include <cstddef>
#include <cstdint>
#include <iostream>

class Matrix
{
public:
  typedef double data_t;

  Matrix();
  explicit Matrix(size_t n);
  Matrix(size_t m, size_t n, size_t p);
  Matrix(size_t m, size_t n, size_t p, bool);
  Matrix(Matrix &) = delete;
  Matrix(Matrix &&);

  Matrix &operator=(const Matrix &) = delete;
  Matrix &operator=(Matrix &&);

  ~Matrix();

  virtual size_t m() const;
  virtual size_t n() const;
  virtual size_t p() const;

  void resize(size_t m, size_t n, size_t p);

  static inline size_t access(size_t m, size_t n, size_t p, size_t i, size_t j, size_t k)
  {
    return i * n * p + j * p + k;
  }
  virtual data_t &operator()(size_t, size_t, size_t);
  virtual const data_t &operator()(size_t, size_t, size_t) const;

  friend std::ostream &operator<<(std::ostream &, const Matrix &);

  size_t data_size() const;

  // protected:
  size_t _p;
  size_t _n;
  size_t _m;
  data_t *_data;
};
