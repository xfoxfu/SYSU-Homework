#ifndef H_PARALLEL_FOR
#define H_PARALLEL_FOR

#include <stddef.h>

struct parallel_args
{
    size_t start;
    size_t end;
    size_t increment;
    void *arg;
};

void parallel_for(size_t start, size_t end, size_t increment, void *(*functor)(void *), void *arg, size_t num_threads);

#endif
