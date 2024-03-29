#include <atomic>
#include <iostream>
#include <math.h>
#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include "parallel_for_closure.hpp"

#ifndef LAB5_SIZE
#define LAB5_SIZE 500
#endif

void atomic_update_max(std::atomic<double> &maximum_value, double const &value) noexcept
{
    double prev_value = maximum_value;
    while (prev_value < value &&
           !maximum_value.compare_exchange_weak(prev_value, value))
    {
    }
}

std::pair<size_t, size_t> mpi_bound(size_t start, size_t end, size_t rank, size_t npes)
{
    size_t step = (end - start) / npes + !!((end - start) % npes);
    return std::make_pair(start + rank * step, std::min(end, start + (rank + 1) * step));
}

struct Matrix
{
    size_t m;
    size_t n;
    double *data;

    Matrix(size_t m, size_t n) : m(m), n(n)
    {
        data = new double[m * n];
    }
    ~Matrix()
    {
        delete[] data;
    }
    double &operator()(size_t i, size_t j)
    {
        return data[i * n + j];
    }
    const double &operator()(size_t i, size_t j) const
    {
        return data[i * n + j];
    }
};

int main(int argc, char *argv[]);

/******************************************************************************/

int main(int argc, char *argv[])

/******************************************************************************/
/*
  Purpose:

    MAIN is the main program for HEATED_PLATE_OPENMP.

  Discussion:

    This code solves the steady state heat equation on a rectangular region.

    The sequential version of this program needs approximately
    18/epsilon iterations to complete. 


    The physical region, and the boundary conditions, are suggested
    by this diagram;

                   W = 0
             +------------------+
             |                  |
    W = 100  |                  | W = 100
             |                  |
             +------------------+
                   W = 100

    The region is covered with a grid of M by N nodes, and an N by N
    array W is used to record the temperature.  The correspondence between
    array indices and locations in the region is suggested by giving the
    indices of the four corners:

                  I = 0
          [0][0]-------------[0][N-1]
             |                  |
      J = 0  |                  |  J = N-1
             |                  |
        [M-1][0]-----------[M-1][N-1]
                  I = M-1

    The steady state solution to the discrete heat equation satisfies the
    following condition at an interior grid point:

      W[Central] = (1/4) * ( W[North] + W[South] + W[East] + W[West] )

    where "Central" is the index of the grid point, "North" is the index
    of its immediate neighbor to the "north", and so on.
   
    Given an approximate solution of the steady state heat equation, a
    "better" solution is given by replacing each interior point by the
    average of its 4 neighbors - in other words, by using the condition
    as an ASSIGNMENT statement:

      W[Central]  <=  (1/4) * ( W[North] + W[South] + W[East] + W[West] )

    If this process is repeated often enough, the difference between successive 
    estimates of the solution will go to zero.

    This program carries out such an iteration, using a tolerance specified by
    the user, and writes the final estimate of the solution to a file that can
    be used for graphic processing.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    18 October 2011

  Author:

    Original C version by Michael Quinn.
    This C version by John Burkardt.

  Reference:

    Michael Quinn,
    Parallel Programming in C with MPI and OpenMP,
    McGraw-Hill, 2004,
    ISBN13: 978-0071232654,
    LC: QA76.73.C15.Q55.

  Local parameters:

    Local, double DIFF, the norm of the change in the solution from one iteration
    to the next.

    Local, double MEAN, the average of the boundary values, used to initialize
    the values of the solution in the interior.

    Local, double U(M, N), the solution at the previous iteration.

    Local, double W(M, N), the solution computed at the latest iteration.
*/
{
    int mpi_npes, mpi_rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_npes);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    size_t size = 500;
    if (argc > 1)
        size = std::stoul(std::string(argv[1]));

    size_t M = size;
    size_t N = size;

    double diff;
    double epsilon = 0.001;
    int iterations;
    int iterations_print;
    double mean;
    Matrix u(M, N);
    Matrix w(M, N);
    double wtime;

    if (argc > 2)
        omp_set_num_threads(std::stoul(std::string(argv[2])));

    if (mpi_rank == 0)
    {
        printf("\n");
        printf("HEATED_PLATE_MPI\n");
        printf("  C++/MPI version\n");
        printf("  A program to solve for the steady state temperature distribution\n");
        printf("  over a rectangular plate.\n");
        printf("\n");
        printf("  Spatial grid of %zu by %zu points.\n", M, N);
        printf("  The iteration will be repeated until the change is <= %e\n", epsilon);
        printf("  Number of processors available = %d\n", omp_get_num_procs());
        printf("  Number of threads              = %d\n", omp_get_max_threads());
        printf("  Number of processes            = %d\n", mpi_npes);
    }
    /*
  Set the boundary values, which don't change. 
*/
    mean = 0.0;

    parallel_for_closure(1, M - 1, 1, [&w, M, N](size_t start, size_t end, size_t incr) {
        for (size_t i = start; i < end; i += incr)
            w(i, 0) = 100.0;
    });
    parallel_for_closure(1, M - 1, 1, [&w, M, N](size_t start, size_t end, size_t incr) {
        for (size_t i = start; i < end; i += incr)
            w(i, N - 1) = 100.0;
    });
    parallel_for_closure(0, N, 1, [&w, M, N](size_t start, size_t end, size_t incr) {
        for (size_t j = start; j < end; j += incr)
            w(M - 1, j) = 100.0;
    });
    parallel_for_closure(0, N, 1, [&w, M, N](size_t start, size_t end, size_t incr) {
        for (size_t j = start; j < end; j += incr)
            w(0, j) = 0.0;
    });
    /*
     * Average the boundary values, to come up with a reasonable
     * initial value for the interior.
     */
    mean += parallel_for_reduce(
        1, M - 1, 1,
        [&w, M, N](size_t start, size_t end, size_t incr) {
            double local_mean = 0.0;
            for (size_t i = start; i < end; i += incr)
                local_mean += w(i, 0) + w(i, N - 1);
            return local_mean;
        },
        [](double lhs, double rhs) { return lhs + rhs; });
    mean += parallel_for_reduce(
        1, N, 1,
        [&w, M, N](size_t start, size_t end, size_t incr) {
            double local_mean = 0.0;
            for (size_t j = start; j < end; j += incr)
                local_mean += w(M - 1, j) + w(0, j);
            return local_mean;
        },
        [](double lhs, double rhs) { return lhs + rhs; });

    /*
     * OpenMP note:
     * You cannot normalize MEAN inside the parallel region.  It
     * only gets its correct value once you leave the parallel region.
     * So we interrupt the parallel region, set MEAN, and go back in.
     */
    mean = mean / (double)(2 * M + 2 * N - 4);
    if (mpi_rank == 0)
    {
        printf("\n");
        printf("  MEAN = %f\n", mean);
    }
    MPI_Bcast(&mean, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    /* 
     * Initialize the interior solution to the mean value.
     */
    parallel_for_closure(1, M - 1, 1, [&w, &mean, M, N](size_t start, size_t end, size_t incr) {
        for (size_t i = start; i < end; i += incr)
        {
            for (size_t j = 1; j < N - 1; j++)
            {
                w(i, j) = mean;
            }
        }
    });
    /*
     * iterate until the  new solution W differs from the old solution U
     * by no more than EPSILON.
     */
    iterations = 0;
    iterations_print = 1;
    if (mpi_rank == 0)
    {
        printf("\n");
        printf(" Iteration  Change\n");
        printf("\n");
    }
    wtime = MPI_Wtime();

    diff = epsilon;

    while (epsilon <= diff)
    {
        const auto [start, end] = mpi_bound(1, M - 1, mpi_rank, mpi_npes);
        /*
         * Save the old solution in U.
         */
        for (size_t i = start; i < end; i++)
        {
            for (size_t j = 0; j < N; j++)
            {
                u(i, j) = w(i, j);
            }
        }

        constexpr int MPI_TAG_FORWARD = 1;
        constexpr int MPI_TAG_BACKWARD = 2;

        if (start > 1)
        {
            // assert(mpi_rank - 1 >= 0);
            MPI_Sendrecv(
                // send [start] backwards as [end]
                &u(start, 0), u.n, MPI_DOUBLE, mpi_rank - 1, MPI_TAG_BACKWARD,
                // receives [start-1] forwards from [end]
                &u(start - 1, 0), u.n, MPI_DOUBLE, mpi_rank - 1, MPI_TAG_FORWARD,
                MPI_COMM_WORLD, nullptr);
        }
        if (end + 1 < u.m)
        {
            // assert(mpi_rank + 1 < mpi_npes);
            MPI_Sendrecv(
                // send [end-1] forwards as [start-1]
                &u(end - 1, 0), u.n, MPI_DOUBLE, mpi_rank + 1, MPI_TAG_FORWARD,
                // receives [end] backwards from [start]
                &u(end, 0), u.n, MPI_DOUBLE, mpi_rank + 1, MPI_TAG_BACKWARD,
                MPI_COMM_WORLD, nullptr);
        }
        /*
         * Determine the new estimate of the solution at the interior points.
         * The new solution W is the average of north, south, east and west neighbors.
         */
        for (size_t i = start; i < end; i++)
        {
            for (size_t j = 1; j < N - 1; j++)
            {
                w(i, j) = (u(i - 1, j) + u(i + 1, j) + u(i, j - 1) + u(i, j + 1)) / 4.0;
            }
        }
        /*
         * C and C++ cannot compute a maximum as a reduction operation.
         * 
         * Therefore, we define a private variable local_diff for each thread.
         * Once they have all computed their values, we use a CRITICAL section
         * to update DIFF.
         */
        double local_diff = 0.0;

        for (size_t i = start; i < end; i++)
        {
            for (size_t j = 1; j < N - 1; j++)
            {
                local_diff = std::max(local_diff, fabs(w(i, j) - u(i, j)));
            }
        }

        MPI_Allreduce(&local_diff, &diff, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);

        iterations++;
        if (mpi_rank == 0 && iterations == iterations_print)
        {
            printf("  %8d  %f\n", iterations, diff);
            iterations_print = 2 * iterations_print;
        }
    }
    wtime = MPI_Wtime() - wtime;

    if (mpi_rank == 0)
    {
        printf("\n");
        printf("  %8d  %f\n", iterations, diff);
        printf("\n");
        printf("  Error tolerance achieved.\n");
        printf("  Wallclock time = %f\n", wtime);
        /*
         * Terminate.
         */
        printf("\n");
        printf("HEATED_PLATE_OPENMP:\n");
        printf("  Normal end of execution.\n");
    }

    MPI_Finalize();
    return 0;
}
