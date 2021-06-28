#include "error.hpp"
#include "lexer.hpp"
#include "parser.hpp"
#include "semantic.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using std::string;
using std::vector;

void print(const AstNode &node, size_t indent = 0) {
  std::cout << string(indent, ' ') << "- " << node.value << std::endl;
  for (const auto &child : node.children) {
    print(*child, indent + 2);
  }
}

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
    std::cout << "*****Input*****" << std::endl << str << std::endl;

    auto lexer = Lexer(str);
    lexer.parse();
    std::cout << "*****Tokens*****" << std::endl;
    for (const auto &token : lexer.tokens()) {
      std::cout << token.to_string() << std::endl;
    }

    auto parser = Parser(lexer.cbegin(), lexer.cend());
    auto program = parser.Program();
    std::cout << "*****AST*****" << std::endl;
    print(*program);
    std::cout << "*****Type Checking*****" << std::endl;
    SemanticVisitor visitor(lexer.tokens());
    visitor.get_type(*program);
    for (const auto &e : visitor.errors) {
      std::cout << e << std::endl;
    }

  } catch (const std::vector<Error> &errs) {
    for (const auto &e : errs) {
      std::cout << e << std::endl;
    }
  } catch (const Error &err) {
    std::cout << err << std::endl;
  }
  return 0;
}

// 3*(4+5/(2-1))
// 21+42-30/(5+5)*(4-2)
