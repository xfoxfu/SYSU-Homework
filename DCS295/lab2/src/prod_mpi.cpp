#include "matrix_transposed.hpp"
#include "product.hpp"
#include <cassert>
#include <mpi.h>

struct MPI_Tags {
  MPI_Tags(int rank) : rank(rank) {}
  int rank;

  int ctx() { return rank * 4 + 0; }
  int lhs() { return rank * 4 + 1; }
  int rhs() { return rank * 4 + 2; }
  int ret() { return rank * 4 + 3; }
};

struct SharedContext {
  size_t m;
  size_t n;
  size_t k;

  SharedContext() : SharedContext(0, 0, 0) {}
  SharedContext(size_t m, size_t n, size_t k) : m(m), n(n), k(k) {}
};

Matrix::data_t dot_product(const Matrix::data_t *lhs, const Matrix::data_t *rhs,
                           size_t len) {
  Matrix::data_t ret = 0;
  for (size_t i = 0; i < len; i++) {
    ret += lhs[i] * rhs[i];
  }
  return ret;
}

Matrix product_mpi_master(int mpi_size, int mpi_rank, const Matrix &lhs,
                          const Matrix &rhs) {
  assert(mpi_rank == 0);

  lhs.ensure_consistent_product(rhs);
  size_t M = lhs.m(), N = rhs.m(), K = rhs.n();
  SharedContext ctx(M, N, K);
  Matrix result = Matrix(M, K);

  // copy rhs to every worker
  MPI_Request *mpi_req_send_rhs = new MPI_Request[mpi_size];
  for (int i = 0; i < mpi_size; i++) {
    // tell context
    MPI_Send(&ctx, sizeof(ctx), MPI_CHAR, i, MPI_Tags(i).ctx(), MPI_COMM_WORLD);
    // tell matrix
    MPI_Isend(rhs._data, N * K, MPI_LONG_DOUBLE, i, MPI_Tags(i).rhs(),
              MPI_COMM_WORLD, mpi_req_send_rhs + i);
  }

  // distribute lhs by row
  MPI_Request *mpi_req_send_lhs = new MPI_Request[M];
  MPI_Request *mpi_req_ret = new MPI_Request[M];
  for (size_t i = 0; i < M; i++) {
    int to = i % mpi_size;
    MPI_Isend(lhs._data + i * N, N, MPI_LONG_DOUBLE, to, MPI_Tags(to).lhs(),
              MPI_COMM_WORLD, mpi_req_send_lhs + i);
    MPI_Irecv(result._data + i * N, N, MPI_LONG_DOUBLE, to, MPI_Tags(to).ret(),
              MPI_COMM_WORLD, mpi_req_ret + i);
  }

  product_mpi_worker(mpi_size, mpi_rank);

  MPI_Waitall(mpi_size, mpi_req_send_rhs, MPI_STATUSES_IGNORE);
  MPI_Waitall(M, mpi_req_send_lhs, MPI_STATUSES_IGNORE);
  MPI_Waitall(M, mpi_req_ret, MPI_STATUSES_IGNORE);

  delete[] mpi_req_send_rhs;
  delete[] mpi_req_send_lhs;
  delete[] mpi_req_ret;

  return result;
}

void product_mpi_worker(int mpi_size, int mpi_rank) {
  (void)(mpi_size);
  (void)(mpi_rank);

  // receive context
  MPI_Tags tags(mpi_rank);
  SharedContext ctx;
  MPI_Recv(&ctx, sizeof(ctx), MPI_CHAR, 0, tags.ctx(), MPI_COMM_WORLD,
           MPI_STATUS_IGNORE);

  // receive rhs matrix
  Matrix::data_t *data = new Matrix::data_t[ctx.n * ctx.k];
  TransposedMatrix mat(data, ctx.n, ctx.k);
  MPI_Recv(data, ctx.n * ctx.k, MPI_LONG_DOUBLE, 0, tags.rhs(), MPI_COMM_WORLD,
           MPI_STATUS_IGNORE);

  size_t recv_count =
      (ctx.m / mpi_size) + ((size_t)mpi_rank < (ctx.m % mpi_size));

  Matrix vec(recv_count, ctx.k);
  Matrix ret(recv_count, ctx.k);
  MPI_Request *mpi_req_ret = new MPI_Request[recv_count];
  for (size_t i = 0; i < recv_count; i++) {
    // receive
    MPI_Recv(&vec(i, 0), ctx.n, MPI_LONG_DOUBLE, 0, tags.lhs(), MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);

    // compute
    for (size_t j = 0; j < ctx.n; j++) {
      for (size_t k = 0; k < ctx.k; k++) {
        ret(i, k) += vec(i, j) * mat(j, k);
      }
    }

    // send
    MPI_Isend(&ret(i, 0), ctx.n, MPI_LONG_DOUBLE, 0, tags.ret(), MPI_COMM_WORLD,
              mpi_req_ret + i);
  }

  // wait for send buffer to complete
  MPI_Waitall(recv_count, mpi_req_ret, MPI_STATUSES_IGNORE);

  // recycle memory
  delete[] data;
  delete[] mpi_req_ret;
}
