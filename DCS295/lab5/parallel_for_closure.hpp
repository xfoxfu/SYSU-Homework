#pragma once
#include <cstddef>
#include <functional>

constexpr size_t NUM_THREADS = 8;

struct parallel_for_closure_args
{
    std::function<void(size_t)> *iterate;
};

void *parallel_for_closure_functor(void *_args);

void parallel_for_closure(size_t start, size_t end, size_t increment, std::function<void(size_t)> iterate);

double parallel_for_reduce(size_t start, size_t end, size_t increment,
                           std::function<double(size_t)> iterate,
                           std::function<double(double, double)> reducer);
