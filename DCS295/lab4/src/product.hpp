#pragma once

#include "matrix.hpp"

Matrix product_standard(const Matrix &L, const Matrix &R);
Matrix product_omp(const Matrix &lhs, const Matrix &rhs);
Matrix product_omp_static(const Matrix &lhs, const Matrix &rhs);
Matrix product_omp_dynamic(const Matrix &lhs, const Matrix &rhs);
Matrix product_pfor(const Matrix &L, const Matrix &R);
