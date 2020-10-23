#include "semaphore.hpp"

semaphore::semaphore()
{
    cnt = 0;
}

void semaphore::up()
{
    pthread_mutex_lock(&mutex);
    cnt += 1;
    pthread_cond_signal(&cv);
    pthread_mutex_unlock(&mutex);
}

void semaphore::down()
{
    pthread_mutex_lock(&mutex);
    while (cnt == 0)
    {
        pthread_cond_wait(&cv, &mutex);
    }
    cnt -= 1;
    pthread_mutex_unlock(&mutex);
}
