/**
 * 进程0向进程1发送一个整数.
 */
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <mpi.h>
#include <vector>

using namespace std::chrono;

#define N 100000

struct Exchange {
  high_resolution_clock::time_point time;
  char values[N];
};

int main(int argc, char *argv[]) {
  Exchange value;

  // initialize mpi
  MPI_Init(&argc, &argv);

  // get process id (rank)
  int current_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &current_rank);

  if (current_rank == 0) { // send process
    for (size_t i = sizeof(high_resolution_clock::time_point);
         i <= sizeof(Exchange); i += 1000) {
      for (int r = 0; r < 1000; r++) {
        value.time = high_resolution_clock::now();
        MPI_Send(&value, i, MPI_CHAR, 1, 99, MPI_COMM_WORLD);
      }
    }
  } else { // receive process
    for (size_t i = sizeof(high_resolution_clock::time_point);
         i <= sizeof(Exchange); i += 1000) {
      for (int r = 0; r < 1000; r++) {
        MPI_Status status;
        MPI_Recv(&value, i, MPI_INT, 0, 99, MPI_COMM_WORLD, &status);
        auto end = high_resolution_clock::now();
        auto dur = duration_cast<nanoseconds>(end - value.time).count();
        printf("size=%zu, recv at=%lldns\n", i, dur);
      }
    }
  }

  MPI_Finalize();
  return 0;
}
