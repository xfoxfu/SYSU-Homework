#include <chrono>
#include <iostream>
#include <omp.h>
#include <optional>
#include <queue>
#include <utility>
#include <vector>

using std::cout;
using std::endl;
using std::optional;
using std::queue;

void pub(queue<int> &q);
void sub(queue<int> &q);

int main(int argc, char *argv[]) {
  if (argc <= 2) {
    cout << "Usage: " << argv[0] << " pub sub" << endl;
    return 1;
  }
  int num_pub = atoi(argv[1]);
  int num_sub = atoi(argv[2]);

  queue<int> q;

#pragma omp parallel num_threads(num_pub + num_sub)
  {
    if (omp_get_thread_num() < num_pub) {
#pragma omp critical
      cout << "spawn publisher " << omp_get_thread_num() << endl;
      pub(q);
    } else {
#pragma omp critical
      cout << "spawn subscriber " << omp_get_thread_num() << endl;
      sub(q);
    }
  } // pragma

  return 0;
}

void pub(queue<int> &q) {
  for (int i = omp_get_thread_num() * 10; i <= omp_get_thread_num() * 10 + 2;
       i++) {
#pragma omp critical
    {
      q.push(i);
      cout << "pub=" << omp_get_thread_num() << ", msg=" << i << endl;
    } // pragma
  }
}

void sub(queue<int> &q) {
  for (;;) {
    optional<int> v;
#pragma omp critical
    {
      if (q.empty()) {
        v.reset();
      } else {
        v = q.front();
        q.pop();
      }
    } // pragma
    if (!v.has_value()) {
      break;
    }

#pragma omp critical
    cout << "sub=" << omp_get_thread_num() << ", msg=" << v.value() << endl;
  }
#pragma omp critical
  cout << "sub=" << omp_get_thread_num() << " thread exit" << endl;
}
