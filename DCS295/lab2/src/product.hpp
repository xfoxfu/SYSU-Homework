#pragma once

#include "matrix.hpp"

Matrix product_standard(const Matrix &, const Matrix &);
Matrix product_strassen(Matrix &, Matrix &);
Matrix product_mpi_master(int mpi_size, int mpi_rank, const Matrix &,
                          const Matrix &);
void product_mpi_worker(int mpi_size, int mpi_rank);
