#include "matrix_transposed.hpp"
#include "product.hpp"
#include <mpi.h>

struct MPI_Tags {
  MPI_Tags(int rank) : rank(rank) {}
  int rank;

  int len() { return rank * 4 + 0; }
  int lhs() { return rank * 4 + 1; }
  int rhs() { return rank * 4 + 2; }
  int ret() { return rank * 4 + 3; }
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
  Matrix result = Matrix(M, K);

  MPI_Request *mpi_send_req = new MPI_Request[M * K * 3];
  MPI_Request *mpi_recv_req = new MPI_Request[M * K];
  for (size_t i = 0; i < M; i++) {
    for (size_t j = 0; j < K; j++) {
      size_t pos = i * K + j;
      int id = pos % mpi_size;
      MPI_Isend(&N, 1, MPI_UNSIGNED_LONG, id, MPI_Tags(id).len(),
                MPI_COMM_WORLD, &mpi_send_req[pos * 3 + 0]);
      // product A_i* B_*j
      //   std::cout << "sending (" << i << "," << j << ") to " << id <<
      //   std::endl;
      MPI_Isend(lhs._data + i * N, N, MPI_LONG_DOUBLE, id, MPI_Tags(id).lhs(),
                MPI_COMM_WORLD, &mpi_send_req[pos * 3 + 1]);
      assert(dynamic_cast<const TransposedMatrix *>(&rhs) != nullptr);
      MPI_Isend(rhs._data + j * N, N, MPI_LONG_DOUBLE, id, MPI_Tags(id).rhs(),
                MPI_COMM_WORLD, &mpi_send_req[pos * 3 + 2]);
      MPI_Irecv(&result._data[pos], 1, MPI_LONG_DOUBLE, id, MPI_Tags(id).ret(),
                MPI_COMM_WORLD, &mpi_recv_req[pos]);
    }
  }
  for (int id = 0; id < mpi_size; id++) {
    size_t term = 0;
    MPI_Send(&term, 1, MPI_UNSIGNED_LONG, id, MPI_Tags(id).len(),
             MPI_COMM_WORLD);
  }
  //   std::cout << "Send of 0 finsihed" << std::endl;

  product_mpi_worker(mpi_size, mpi_rank);

  MPI_Waitall(M * K * 3, mpi_send_req, MPI_STATUSES_IGNORE);
  MPI_Waitall(M * K, mpi_recv_req, MPI_STATUSES_IGNORE);
  //   std::cout << "returned" << std::endl;
  delete[] mpi_send_req;
  delete[] mpi_recv_req;

  return result;
}

void product_mpi_worker(int mpi_size, int mpi_rank) {
  (void)(mpi_size);
  // receive length
  size_t len;
  MPI_Status status;
  while (MPI_Recv(&len, 1, MPI_UNSIGNED_LONG, 0, MPI_Tags(mpi_rank).len(),
                  MPI_COMM_WORLD, &status),
         len != 0) {
    // receive matrix
    // std::cout << mpi_rank << ": recv = " << len << std::endl;
    Matrix::data_t *lhs = new Matrix::data_t[len];
    MPI_Recv(lhs, len, MPI_LONG_DOUBLE, 0, MPI_Tags(mpi_rank).lhs(),
             MPI_COMM_WORLD, &status);
    Matrix::data_t *rhs = new Matrix::data_t[len];
    MPI_Recv(rhs, len, MPI_LONG_DOUBLE, 0, MPI_Tags(mpi_rank).rhs(),
             MPI_COMM_WORLD, &status);

    // compute
    auto v = dot_product(lhs, rhs, len);
    MPI_Send(&v, 1, MPI_LONG_DOUBLE, 0, MPI_Tags(mpi_rank).ret(),
             MPI_COMM_WORLD);

    delete[] lhs;
    delete[] rhs;
  }

  //   std::cout << "compute of " << mpi_rank << " finished" << std::endl;
}
