#include <algorithm>
#include <cstdlib>
#include <iostream>
#include "parallel_for.hpp"

#define THREAD_LIMIT 8

void parallel_for(size_t start, size_t end, size_t increment, void *(*functor)(void *), void *arg, size_t num_threads)
{
    if (num_threads > THREAD_LIMIT)
    {
        std::cout << "Thread num " << num_threads << " exceeds limit " << THREAD_LIMIT << std::endl;
        abort();
    }

    pthread_t threads[THREAD_LIMIT];
    parallel_args *params = new parallel_args[THREAD_LIMIT];
    size_t step = (end - start) / num_threads + !!((end - start) % num_threads);
    for (size_t id = 0; id < num_threads; id++)
    {
        params[id].start = start + id * step;
        params[id].end = std::min(end, start + (id + 1) * step);
        params[id].increment = increment;
        params[id].arg = arg;
        pthread_create(&threads[id], NULL, functor, reinterpret_cast<void *>(&params[id]));
    }

    // join workers
    for (size_t id = 0; id < num_threads; id++)
    {
        pthread_join(threads[id], nullptr);
    }

    delete[] params;
}
