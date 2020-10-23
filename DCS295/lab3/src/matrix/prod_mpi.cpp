#include "matrix_transposed.hpp"
#include "product.hpp"
#include <cassert>
#include <mpi.h>

struct MPI_Tags
{
  MPI_Tags(int rank) : rank(rank) {}
  int rank;

  int ctx() { return rank * 4 + 0; }
  int lhs() { return rank * 4 + 1; }
  int rhs() { return rank * 4 + 2; }
  int ret() { return rank * 4 + 3; }
};

struct SharedContext
{
  size_t m;
  size_t n;
  size_t k;

  SharedContext() : SharedContext(0, 0, 0) {}
  SharedContext(size_t m, size_t n, size_t k) : m(m), n(n), k(k) {}
};

Matrix::data_t dot_product(const Matrix::data_t *lhs, const Matrix::data_t *rhs,
                           size_t len)
{
  Matrix::data_t ret = 0;
  for (size_t i = 0; i < len; i++)
  {
    ret += lhs[i] * rhs[i];
  }
  return ret;
}

void mpi_compute(int mpi_size, int mpi_rank, SharedContext &ctx,
                 const TransposedMatrix &mat, const Matrix &vec)
{
  MPI_Tags tags(mpi_rank);

  size_t recv_count =
      (ctx.m / mpi_size) + ((size_t)mpi_rank < (ctx.m % mpi_size));

  Matrix ret(recv_count, ctx.k);
  MPI_Request *mpi_req_ret = new MPI_Request[recv_count];
  for (size_t i = 0; i < recv_count; i++)
  {
    for (size_t j = 0; j < ctx.n; j++)
    {
      for (size_t k = 0; k < ctx.k; k++)
      {
        ret(i, k) += vec(i, j) * mat(j, k);
      }
    }
  }

  delete[] mpi_req_ret;
}

Matrix product_mpi(int mpi_size, int mpi_rank, const Matrix *lhs,
                   const Matrix *rhs)
{
  assert(mpi_rank != 0 || (lhs != nullptr && rhs != nullptr));
  assert(mpi_rank == 0 || (mpi_rank != 0 && lhs == nullptr && rhs == nullptr));
  SharedContext ctx;
  Matrix result; // only used by root
  if (mpi_rank == 0)
  {
    lhs->ensure_consistent_product(*rhs);
    size_t M = lhs->m(), N = rhs->m(), K = rhs->n();
    ctx = SharedContext(M, N, K);
    result = Matrix(M, K);
  }

  // copy rhs to every worker
  MPI_Bcast(&ctx, sizeof(ctx), MPI_CHAR, 0, MPI_COMM_WORLD);
  if (mpi_rank != 0)
  {
    lhs = new TransposedMatrix(ctx.n, ctx.k);
    rhs = new TransposedMatrix(ctx.n, ctx.k);
  }
  MPI_Bcast(rhs->_data, ctx.n * ctx.k, MPI_LONG_DOUBLE, 0, MPI_COMM_WORLD);

  // distribute lhs by row
  int *sendcounts = new int[mpi_size];
  int *displs = new int[mpi_size];
  int sum = 0;
  for (int i = 0; i < mpi_size; i++)
  {
    sendcounts[i] =
        ctx.n * ((ctx.m / mpi_size) + ((size_t)mpi_rank < (ctx.m % mpi_size)));
    displs[i] = sum;
    sum += sendcounts[i];
    std::cout << "(" << sendcounts[i] << "," << displs[i] << ") ";
  }
  std::cout << std::endl;

  size_t recv_count = sendcounts[mpi_rank] / ctx.n;
  Matrix vec(recv_count, ctx.k);
  Matrix ret(recv_count, ctx.k);

  MPI_Scatterv(lhs->_data, sendcounts, displs, MPI_LONG_DOUBLE, vec._data,
               sendcounts[mpi_rank], MPI_LONG_DOUBLE, 0, MPI_COMM_WORLD);

  for (size_t i = 0; i < recv_count; i++)
  {
    for (size_t j = 0; j < ctx.n; j++)
    {
      for (size_t k = 0; k < ctx.k; k++)
      {
        ret(i, k) += vec(i, j) * (*rhs)(j, k);
      }
    }
  }

  MPI_Gatherv(ret._data, sendcounts[mpi_rank], MPI_LONG_DOUBLE, result._data,
              sendcounts, displs, MPI_LONG_DOUBLE, 0, MPI_COMM_WORLD);

  if (mpi_rank != 0)
  {
    delete lhs;
    delete rhs;
  }

  return result;
}
