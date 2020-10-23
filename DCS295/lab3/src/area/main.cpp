#include <pthread.h>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include <atomic>

#define THREAD_LIMIT 8

#define AREA_OK 0
#define AREA_INVALID_ARGUMENTS 1
#define AREA_THREAD_LIMIT_EXCEED 2

using std::atomic;
using std::cout;
using std::endl;

struct worker_params
{
    size_t local_iter;
    atomic<size_t> *sum;
};

double calc_area(size_t iter, size_t thread_count);
void *worker_wrapper(void *args);
void worker(size_t local_iter, atomic<size_t> &sum);

int main(int argc, char **argv)
{
    // read input paramaters
    if (argc < 3)
    {
        cout << "Usage: " << argv[0] << " <N> <P>" << endl;
        return AREA_INVALID_ARGUMENTS;
    }
    size_t n = atoi(argv[1]);
    size_t p = atoi(argv[2]);
    cout << "iter=" << n << ",threads=" << p << endl;
    if (p > THREAD_LIMIT)
    {
        cout << "Thread count " << p << " exceeds limit " << THREAD_LIMIT << endl;
        return AREA_THREAD_LIMIT_EXCEED;
    }
    if (n % p != 0)
    {
        cout << "Iteration " << n << " is not divisible by threads " << p << endl;
        return AREA_INVALID_ARGUMENTS;
    }

    // perform matrix production
    // record start time
    auto start = std::chrono::high_resolution_clock::now();
    // do some work
    double area = calc_area(n, p);
    // record end time
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> diff = end - start;
    cout << "area: " << area << endl
         << "time: " << diff.count() << " ms" << endl;

    return AREA_OK;
}

double calc_area(size_t iter, size_t thread_count)
{
    atomic<size_t> *count = new atomic<size_t>(0);
    // spawn workers
    pthread_t threads[THREAD_LIMIT];
    worker_params *params = new worker_params[THREAD_LIMIT];
    for (size_t id = 0; id < thread_count; id++)
    {
        params[id].local_iter = iter / thread_count;
        params[id].sum = count;
        pthread_create(&threads[id], NULL, worker_wrapper, reinterpret_cast<void *>(&params[id]));
    }

    // join workers
    for (size_t id = 0; id < thread_count; id++)
    {
        pthread_join(threads[id], nullptr);
    }

    double cnt = static_cast<double>(count->load());
    delete[] params;
    delete count;
    return cnt / static_cast<double>(iter);
}

void *worker_wrapper(void *args_)
{
    worker_params *args = reinterpret_cast<worker_params *>(args_);
    worker(args->local_iter, *args->sum);
    return nullptr;
}

void worker(size_t local_iter, atomic<size_t> &sum)
{
    size_t local_sum = 0;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(0, 1);

    for (size_t i = 0; i < local_iter; i++)
    {
        double x = dis(gen);
        double y = dis(gen);
        double y_real = x * x;
        if (y_real > y)
        {
            local_sum += 1;
        }
    }

    sum.fetch_add(local_sum);
}
