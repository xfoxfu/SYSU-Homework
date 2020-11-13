#pragma once

#include <cstddef>
#include <cstdint>
#include <iostream>

class Matrix
{
public:
  typedef long double data_t;

  Matrix();
  explicit Matrix(size_t n);
  Matrix(size_t, size_t);
  Matrix(size_t, size_t, bool);
  Matrix(Matrix &) = delete;
  Matrix(Matrix &&);
  Matrix(data_t *, size_t, size_t);

  Matrix &operator=(const Matrix &) = delete;
  Matrix &operator=(Matrix &&);

  virtual ~Matrix();

  virtual size_t m() const;
  virtual size_t n() const;

  virtual data_t &operator()(size_t, size_t);
  virtual const data_t &operator()(size_t, size_t) const;
  Matrix operator+(const Matrix &) const;
  // Matrix &operator+=(const Matrix &);
  Matrix operator-(const Matrix &) const;
  // Matrix &operator-=(const Matrix &);
  friend std::ostream &operator<<(std::ostream &, const Matrix &);

  bool is_consistent_product(const Matrix &) const;
  void ensure_consistent_product(const Matrix &r) const;

  Matrix clone() const;

  // protected:
  size_t _n;
  size_t _m;
  data_t *_data;
  bool _need_free;
};
