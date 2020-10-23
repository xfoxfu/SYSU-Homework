#include <pthread.h>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>

#ifdef USE_ATOMIC
#include <atomic>
#endif

#define ARRAY_SIZE 1000
#define THREAD_LIMIT 8
typedef size_t value_t;

#define SUM_OK 0
#define SUM_INVALID_ARGUMENTS 1
#define SUM_THREAD_LIMIT_EXCEED 2

template <typename T>
struct safe_value
{
#ifndef USE_ATOMIC
    pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
    T value;
    T add(T incr)
    {
        pthread_mutex_lock(&mutex);
        T prev = value;
        value += incr;
        pthread_mutex_unlock(&mutex);
        return prev;
    }
#else
    std::atomic<value_t> value;
    T add(T incr)
    {
        return value.fetch_add(incr);
    }
#endif

    safe_value(T value) : value(value)
    {
    }
};

struct worker_params
{
    const value_t *arr;
    size_t len;
    size_t group_size;
    safe_value<size_t> *global_index;
    safe_value<value_t> *sum;
};

value_t sum_threads(const value_t *arr, size_t len, size_t g, size_t p);
#ifndef NDEBUG
value_t sum_plain(const value_t *arr, size_t len);
#endif
void *worker_wrapper(void *args);
void worker(const value_t *arr, size_t len, size_t group_size,
            safe_value<size_t> &global_index, safe_value<value_t> &sum);

int main(int argc, char **argv)
{
    // read input paramaters
    if (argc < 3)
    {
        std::cout << "Usage: " << argv[0] << " <G> <P>" << std::endl;
        return SUM_INVALID_ARGUMENTS;
    }
    size_t g = atoi(argv[1]);
    size_t p = atoi(argv[2]);
    std::cout << "group size=" << g << ",len=" << ARRAY_SIZE << ",threads=" << p << std::endl;
    if (p > THREAD_LIMIT)
    {
        std::cout << "Thread count " << p << " exceeds limit " << THREAD_LIMIT << std::endl;
        return SUM_THREAD_LIMIT_EXCEED;
    }

    size_t len = ARRAY_SIZE;

    // generate matrix
    value_t *arr = new value_t[len];
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<value_t> dis(1, 10);
    for (size_t i = 0; i < len; i++)
    {
        arr[i] = dis(gen);
    }

    // perform matrix production
    // record start time
    auto start = std::chrono::high_resolution_clock::now();
    // do some work
    value_t sum = sum_threads(arr, len, g, p);
    // record end time
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> diff = end - start;
    std::cout << "time: " << diff.count() << " ms\n";

#ifndef NDEBUG // debug assertion
    // record start time
    auto start2 = std::chrono::high_resolution_clock::now();
    // do some work
    value_t sum2 = sum_plain(arr, len);
    // record end time
    auto end2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> diff2 = end2 - start2;
    std::cout << "normal: " << diff2.count() << "ms\n";

    if (sum != sum2)
    {
        std::cout << "different = " << (sum - sum2) << std::endl;
    }
#endif

    return 0;
}

value_t sum_threads(const value_t *arr, size_t len, size_t g, size_t p)
{
    safe_value<size_t> *global_index = new safe_value<size_t>(0);
    safe_value<value_t> *sum = new safe_value<value_t>(0);

    // spawn workers
    pthread_t threads[THREAD_LIMIT];
    worker_params *params = new worker_params[THREAD_LIMIT];
    for (size_t id = 0; id < p; id++)
    {
        params[id].arr = arr;
        params[id].len = len;
        params[id].global_index = global_index;
        params[id].sum = sum;
        params[id].group_size = g;
        pthread_create(&threads[id], NULL, worker_wrapper, reinterpret_cast<void *>(&params[id]));
    }

    // join workers
    for (size_t id = 0; id < p; id++)
    {
        pthread_join(threads[id], nullptr);
    }

    value_t value = sum->value;
    delete[] params;
    delete global_index;
    delete sum;
    return value;
}

#ifndef NDEBUG
value_t sum_plain(const value_t *arr, size_t len)
{
    value_t sum = 0;
    for (size_t i = 0; i < len; i++)
    {
        sum += arr[i];
    }
    return sum;
}
#endif

void *worker_wrapper(void *args_)
{
    worker_params *args = reinterpret_cast<worker_params *>(args_);
    worker(args->arr, args->len, args->group_size,
           *args->global_index, *args->sum);
    return nullptr;
}

void worker(const value_t *arr, size_t len, size_t group_size,
            safe_value<size_t> &global_index, safe_value<value_t> &sum)
{
    size_t start;
    // compute bounds, and skip if all element computed
    while ((start = global_index.add(group_size)) < len)
    {
        size_t end = std::min(start + group_size, len);

        // compute sum
        value_t local_sum = 0;
        for (size_t i = start; i < end; i++)
        {
            local_sum += arr[i];
        }

        // write back
        sum.add(local_sum);
    }
}
