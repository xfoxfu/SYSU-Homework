#include "csr_formatter.h"
#include <algorithm>
#include <iostream>
#include <iterator>

using std::endl;

int main(int argc, char *argv[]) {
  if (argc <= 2) {
    std::cout << "Usage: " << argv[0] << " matrix vector output" << std::endl;
    return 1;
  }

  CSR sim = assemble_simetric_csr_matrix(argv[1]);

  std::ofstream fout(argv[3]);

  fout << sim.val.size() << " ";
  std::copy(sim.val.begin(), sim.val.end(),
            std::ostream_iterator<double>(fout, " "));
  fout << endl;

  fout << sim.col_ind.size() << " ";
  std::copy(sim.col_ind.begin(), sim.col_ind.end(),
            std::ostream_iterator<double>(fout, " "));
  fout << endl;

  fout << (sim.row_ptr.size() - 1) << " ";
  std::copy(sim.row_ptr.begin(), sim.row_ptr.end() - 1,
            std::ostream_iterator<double>(fout, " "));
  fout << endl;

  std::ifstream fin(argv[2]);
  while (fin.peek() == '%')
    fin.ignore(2048, '\n');
  size_t M, N;
  fin >> M >> N;
  fout << M << " ";
  if (N != 1) {
    std::cout << "Input is not a vector" << endl;
    return 2;
  }
  for (size_t i = 0; i < M; i++) {
    double val;
    fin >> val;
    fout << val << " ";
  }
  fout << endl;

  return 0;
}
