#include "lexer.hpp"
// #include "parser.hpp"
#include "error.hpp"
#include "visitor.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using std::string;
using std::vector;

int main(int argc, char **argv) {
  std::string file;

  if (argc == 2) {
    file = std::string(argv[1]);
  } else if (argc == 1) {
    std::cout << "Filename: " << std::flush;
    std::getline(std::cin, file);
  } else {
    std::cout << "Usage: " << argv[0] << " <filename>" << std::endl;
    return 1;
  }

  std::ifstream fin(file);
  std::string str((std::istreambuf_iterator<char>(fin)),
                  (std::istreambuf_iterator<char>()));

  try {
    auto lexer = Lexer(str);
    lexer.parse();
    // auto parser = Parser(lexer.cbegin(), lexer.cend());
    // auto expr = parser.expr();

    std::cout << "*****Input*****" << std::endl << str << std::endl;
    std::cout << "*****Tokens*****" << std::endl;
    for (const auto &token : lexer.tokens()) {
      std::cout << token.to_string() << std::endl;
    }
  } catch (const std::vector<Error> &errs) {
    for (const auto &e : errs) {
      std::cout << e << std::endl;
    }
  }
  // Printer pr;
  // expr->accept(*&pr);
  // std::cout << "AST: " << std::endl;
  // for (const auto &line : pr.result()) {
  //   for (const auto &value : line) {
  //     std::cout << std::setw(4) << value;
  //   }
  //   std::cout << std::endl;
  // }
  return 0;
}

// 3*(4+5/(2-1))
// 21+42-30/(5+5)*(4-2)
