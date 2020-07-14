#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <iterator>
#include <omp.h>
#include <vector>

using std::string;
using std::vector;

typedef double value_t;

template <typename T>
void read_into_vector(std::ifstream &fin, vector<T> &vec) {
  size_t n;
  fin >> n;
  for (size_t i = 0; i < n; i++) {
    T a;
    fin >> a;
    vec.push_back(a);
  }
}
void read_input(std::ifstream &fin, vector<value_t> &values,
                vector<size_t> &col_index, vector<size_t> &row_index,
                vector<value_t> &vec) {
  read_into_vector(fin, values);
  read_into_vector(fin, col_index);
  read_into_vector(fin, row_index);
  read_into_vector(fin, vec);
}

int main(int argc, char *argv[]) {
  if (argc <= 2) {
    std::cout << "Usage: " << argv[0] << " file threads" << std::endl;
    return 1;
  }
  // std::cout << "Reading from `" << argv[1] << "` and writing to `" << argv[2]
  //           << "`." << std::endl;

  int num_threads = atoi(argv[2]);
  omp_set_num_threads(num_threads);

  vector<value_t> values;
  vector<size_t> col_index;
  vector<size_t> row_index; // rows+1
  vector<value_t> vec;      // cols
  vector<value_t> result;

  std::ios::sync_with_stdio(false);
  std::ifstream fin(argv[1]);
  read_input(fin, values, col_index, row_index, vec);
  fin.close();

  result.resize(vec.size());
  // std::cout << "Computing on " << vec.size() << "x" << (row_index.size() - 1)
  // << ", threads=" << num_threads << std::endl;

  auto start = std::chrono::high_resolution_clock::now();
#pragma omp parallel for
  for (size_t row_sub = 0; row_sub < row_index.size() - 1; row_sub++) {
    result[row_sub] = 0;
    for (size_t col_sub = row_index[row_sub]; col_sub < row_index[row_sub + 1];
         col_sub++) {
      result[row_sub] += values[col_sub] * vec[col_index[col_sub]];
    }
  } // pragma
  auto end = std::chrono::high_resolution_clock::now();
  double time_taken =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  std::cout << "Computing on " << vec.size() << "x" << (row_index.size() - 1)
            << ", threads=" << num_threads << ", time=" << time_taken << "ns"
            << std::endl;

  // std::ofstream fout(argv[2]);
  // std::copy(result.begin(), result.end(),
  //           std::ostream_iterator<value_t>(fout, " "));

  return 0;
}
