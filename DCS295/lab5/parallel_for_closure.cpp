#include <parallel_for.hpp>
#include <omp.h>
#include "parallel_for_closure.hpp"

void *parallel_for_closure_functor(void *_args)
{
    parallel_args &args = *reinterpret_cast<parallel_args *>(_args);
    (*reinterpret_cast<parallel_for_closure_args *>(args.arg)->iterate)(args.start, args.end, args.increment);
    return nullptr;
}

void parallel_for_closure(size_t start, size_t end, size_t increment,
                          std::function<void(size_t, size_t, size_t)> iterate)
{
    int thread_count = omp_get_max_threads();
    parallel_for_closure_args *args = new parallel_for_closure_args;
    args->iterate = &iterate;
    parallel_for(start, end, increment, parallel_for_closure_functor, args, thread_count);
}

double parallel_for_reduce(size_t start, size_t end, size_t increment,
                           std::function<double(size_t, size_t, size_t)> iterate,
                           std::function<double(double, double)> reducer)
{
    int thread_count = omp_get_max_threads();
    double *result = new double[thread_count];
    for (size_t i = 0; i < thread_count; i++)
    {
        result[i] = 0.0;
    }
    size_t step = (end - start) / thread_count + !!((end - start) % thread_count);
    parallel_for_closure(start, end, increment, [&](size_t lstart, size_t lend, size_t lincr) {
        result[(lstart - start) / step] = iterate(lstart, lend, lincr);
    });
    double ret = 0;
    for (size_t i = 0; i < thread_count; i++)
    {
        ret = reducer(ret, result[i]);
    }
    return ret;
}
