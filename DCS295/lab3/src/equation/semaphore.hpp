#pragma once

#include <stdio.h>
#include <pthread.h>

struct semaphore
{
    int cnt;
    pthread_mutex_t mutex;
    pthread_cond_t cv;

    semaphore();
    void up();
    void down();
};
