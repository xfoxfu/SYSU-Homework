#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

int Global;

void *Thread1(void *x) {
  static int ssss = 5;
  ssss += 1;
  printf("Hello world!");
  Global = 42;
  x = malloc(sizeof(int));
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
