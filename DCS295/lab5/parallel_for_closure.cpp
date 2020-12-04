#include <parallel_for.hpp>
#include "parallel_for_closure.hpp"

void *parallel_for_closure_functor(void *_args)
{
    parallel_args &args = *reinterpret_cast<parallel_args *>(_args);
    for (size_t i = args.start; i < args.end; i += args.increment)
    {
        (*reinterpret_cast<parallel_for_closure_args *>(args.arg)->iterate)(i);
    }
    return nullptr;
}

void parallel_for_closure(size_t start, size_t end, size_t increment, std::function<void(size_t)> iterate)
{
    parallel_for_closure_args *args = new parallel_for_closure_args;
    args->iterate = &iterate;
    parallel_for(start, end, increment, parallel_for_closure_functor, args, NUM_THREADS);
}

double parallel_for_reduce(size_t start, size_t end, size_t increment,
                           std::function<double(size_t)> iterate,
                           std::function<double(double, double)> reducer)
{
    double *result = new double[NUM_THREADS];
    for (size_t i = 0; i < NUM_THREADS; i++)
    {
        result[i] = 0.0;
    }
    size_t step = (end - start) / NUM_THREADS + !!((end - start) % NUM_THREADS);
    parallel_for_closure(start, end, increment, [&](size_t i) {
        result[(i - start) / step] = reducer(result[(i - start) / step], iterate(i));
    });
    double ret = 0;
    for (size_t i = 0; i < NUM_THREADS; i++)
    {
        ret = reducer(ret, result[i]);
    }
    return ret;
}
