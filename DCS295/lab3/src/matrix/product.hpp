#pragma once

#include "matrix.hpp"

Matrix product_standard(const Matrix &, const Matrix &);
Matrix product_strassen(Matrix &, Matrix &);
Matrix product_mpi(int mpi_size, int mpi_rank, const Matrix *lhs,
                   const Matrix *rhs);
