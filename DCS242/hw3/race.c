#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

int Global;

void *Thread1(void *x) {
  Global = 42;
  return x;
}

int main() {
  pthread_t t;
  pthread_create(&t, NULL, Thread1, NULL);
  Global = 43;
  pthread_join(t, NULL);
  Thread1(NULL);
  return Global;
}
