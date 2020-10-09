#pragma once

#include "matrix.hpp"

Matrix product_standard(const Matrix &, const Matrix &);
Matrix product_strassen(Matrix &, Matrix &);
