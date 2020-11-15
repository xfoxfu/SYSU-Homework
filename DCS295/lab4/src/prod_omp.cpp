#include "product.hpp"

#define PRODUCT_FOR(LHS, RHS, OUT)                                \
    for (size_t i = 0; i < LHS.m(); i++)                          \
    {                                                             \
        for (size_t j = 0; j < RHS.n(); j++)                      \
        {                                                         \
            /* for O[i, j] */                                     \
            Matrix::data_t cell_out = 0;                          \
            /* accumulate L[i, k] * R[k, j] */                    \
            for (size_t k = 0; k < LHS.n() /* == rhs.m() */; k++) \
            {                                                     \
                cell_out += LHS(i, k) * RHS(k, j);                \
            }                                                     \
            /* set value */                                       \
            OUT(i, j) = cell_out;                                 \
        }                                                         \
    }

Matrix product_omp(const Matrix &lhs, const Matrix &rhs)
{
    Matrix out(lhs.m(), rhs.n());
#pragma omp parallel for
    PRODUCT_FOR(lhs, rhs, out)
    return out;
}

Matrix product_omp_static(const Matrix &lhs, const Matrix &rhs)
{
    Matrix out(lhs.m(), rhs.n());
#pragma omp parallel for schedule(static, 1)
    PRODUCT_FOR(lhs, rhs, out)
    return out;
}

Matrix product_omp_dynamic(const Matrix &lhs, const Matrix &rhs)
{
    Matrix out(lhs.m(), rhs.n());
#pragma omp parallel for schedule(dynamic, 1)
    PRODUCT_FOR(lhs, rhs, out)
    return out;
}
