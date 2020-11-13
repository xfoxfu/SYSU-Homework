#include "parallel_for.hpp"
#include "product.hpp"
#include <utility>

struct functor_args
{
    const Matrix *lhs;
    const Matrix *rhs;
    Matrix *out;
};

void *functor(void *_args);

Matrix product_pfor(const Matrix &L, const Matrix &R)
{
    Matrix out(L.m(), R.n());
    functor_args *args = new functor_args;
    args->lhs = &L;
    args->rhs = &R;
    args->out = &out;
    parallel_for(0, L.m(), 1, functor, args, 8);
    return out;
}

void *functor(void *_args)
{
    struct parallel_args &args = *reinterpret_cast<struct parallel_args *>(_args);
    functor_args &ops = *reinterpret_cast<functor_args *>(args.arg);
    const Matrix &lhs = *reinterpret_cast<const Matrix *>(ops.lhs);
    const Matrix &rhs = *reinterpret_cast<const Matrix *>(ops.rhs);
    Matrix &out = *reinterpret_cast<Matrix *>(ops.out);

    for (size_t i = args.start; i < args.end; i = i + args.increment)
    {
        for (size_t j = 0; j < rhs.n(); j++)
        {
            /* for O[i, j] */
            Matrix::data_t cell_out = 0; /* accumulate L[i, k] * R[k, j] */
            for (size_t k = 0; k < lhs.n() /* == rhs.m() */; k++)
            {
                cell_out += lhs(i, k) * rhs(k, j);
            }
            /* set value */
            out(i, j) = cell_out;
        }
    }

    return nullptr;
}
