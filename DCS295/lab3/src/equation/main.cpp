#include <pthread.h>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <cmath>
#include <utility>
#include "semaphore.hpp"

#define EQ_OK 0
#define EQ_INVALID_ARGUMENTS 1
#define EQ_THREAD_LIMIT_EXCEED 2

std::pair<double, double> solve_parallel(double a, double b, double c);
std::pair<double, double> solve_plain(double a, double b, double c);

int main(int argc, char **argv)
{
    // read input paramaters
    if (argc < 4)
    {
        std::cout << "Usage: " << argv[0] << " a b c" << std::endl;
        return EQ_INVALID_ARGUMENTS;
    }
    double a = atof(argv[1]);
    double b = atof(argv[2]);
    double c = atof(argv[3]);

    {
        // perform matrix production
        // record start time
        auto start = std::chrono::high_resolution_clock::now();
        // do some work
        auto [x1, x2] = solve_parallel(a, b, c);
        // record end time
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> diff = end - start;
        std::cout << "(parallel) x1=" << x1 << " ,x2=" << x2 << std::endl
                  << "(parallel) time: " << diff.count() << " ms" << std::endl;
    }
    {
        // perform matrix production
        // record start time
        auto start = std::chrono::high_resolution_clock::now();
        // do some work
        auto [x1, x2] = solve_plain(a, b, c);
        // record end time
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> diff = end - start;
        std::cout << "(trivial) x1=" << x1 << " ,x2=" << x2 << std::endl
                  << "(trivial) time: " << diff.count() << " ms" << std::endl;
    }

    return EQ_OK;
}

struct worker_param
{
    double a;
    double b;
    double c;
    double *neg_b;
    double *sqrt_delta;
    double *two_a;
    double *x1;
    double *x2;
    semaphore *sem_neg_b;
    semaphore *sem_sqrt_delta;
    semaphore *sem_two_a;
};

void *worker_neg_b(worker_param *param);
void *worker_delta(worker_param *param);
void *worker_two_a(worker_param *param);
void *worker_x1(worker_param *param);
void *worker_x2(worker_param *param);

std::pair<double, double> solve_parallel(double a, double b, double c)
{
    pthread_t threads[5];
    worker_param *params = new worker_param[5];
    double *neg_b = new double;
    double *sqrt_delta = new double;
    double *two_a = new double;
    double *x1p = new double;
    double *x2p = new double;
    semaphore *sem_neg_b = new semaphore();
    semaphore *sem_sqrt_delta = new semaphore();
    semaphore *sem_two_a = new semaphore();
    for (size_t i = 0; i < 5; i++)
    {
        params[i].a = a;
        params[i].b = b;
        params[i].c = c;
        params[i].neg_b = neg_b;
        params[i].sqrt_delta = sqrt_delta;
        params[i].two_a = two_a;
        params[i].x1 = x1p;
        params[i].x2 = x2p;
        params[i].sem_neg_b = sem_neg_b;
        params[i].sem_sqrt_delta = sem_sqrt_delta;
        params[i].sem_two_a = sem_two_a;
    }
    pthread_create(&threads[0], nullptr, reinterpret_cast<void *(*)(void *)>(worker_neg_b),
                   reinterpret_cast<void *>(&params[0]));
    pthread_create(&threads[1], nullptr, reinterpret_cast<void *(*)(void *)>(worker_delta),
                   reinterpret_cast<void *>(&params[1]));
    pthread_create(&threads[2], nullptr, reinterpret_cast<void *(*)(void *)>(worker_two_a),
                   reinterpret_cast<void *>(&params[2]));
    pthread_create(&threads[3], nullptr, reinterpret_cast<void *(*)(void *)>(worker_x1),
                   reinterpret_cast<void *>(&params[3]));
    pthread_create(&threads[4], nullptr, reinterpret_cast<void *(*)(void *)>(worker_x2),
                   reinterpret_cast<void *>(&params[4]));

    for (size_t i = 0; i < 5; i++)
    {
        pthread_join(threads[i], nullptr);
    }

    delete[] params;
    delete sem_neg_b;
    delete sem_sqrt_delta;
    delete sem_two_a;
    double x1 = *x1p;
    double x2 = *x2p;
    delete x1p;
    delete x2p;
    return std::make_pair(x1, x2);
}

void *worker_neg_b(worker_param *param)
{
    *param->neg_b = -param->b;
    param->sem_neg_b->up();
    param->sem_neg_b->up();
    return nullptr;
}

void *worker_delta(worker_param *param)
{
    *param->sqrt_delta = std::sqrt(param->b * 2 - 4 * param->a * param->c);
    param->sem_sqrt_delta->up();
    param->sem_sqrt_delta->up();
    return nullptr;
}

void *worker_two_a(worker_param *param)
{
    *param->two_a = param->a * 2;
    param->sem_two_a->up();
    param->sem_two_a->up();
    return nullptr;
}

void *worker_x1(worker_param *param)
{
    param->sem_neg_b->down();
    param->sem_sqrt_delta->down();
    param->sem_two_a->down();
    *param->x1 = (*param->neg_b + *param->sqrt_delta) / *param->two_a;
    return nullptr;
}

void *worker_x2(worker_param *param)
{
    param->sem_neg_b->down();
    param->sem_sqrt_delta->down();
    param->sem_two_a->down();
    *param->x2 = (*param->neg_b - *param->sqrt_delta) / *param->two_a;
    return nullptr;
}

std::pair<double, double> solve_plain(double a, double b, double c)
{
    double delta = std::sqrt(b * b - 4 * a * c);
    return std::make_pair((-b + delta) / (2 * a), (-b - delta) / (2 * a));
}
