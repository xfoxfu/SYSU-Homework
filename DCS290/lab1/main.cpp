#include "lexer.hpp"
#include "parser.hpp"
#include "visitor.hpp"
#include <iostream>
#include <sstream>
#include <string>

class Printer final : public Visitor {
  std::ostringstream os;

public:
  Printer() : os(std::ios::app) {}
  virtual void visit(const ast::Integer &value) { os << value.value; }
  virtual void visit(const ast::Binary &value) {
    value.left->accept(*this);
    os << ' ';
    value.right->accept(*this);
    os << ' ';
    if (value.op == ast::Binary::ADD) {
      os << '+';
    } else if (value.op == ast::Binary::SUBTRACT) {
      os << '-';
    } else if (value.op == ast::Binary::MULTIPLY) {
      os << '*';
    } else if (value.op == ast::Binary::DIVIDE) {
      os << '/';
    }
  }
  std::string finalize() { return os.str(); }
};

int main(int argc, char **argv) {
  std::string str;

  if (argc == 2) {
    str = std::string(argv[1]);
  } else if (argc == 1) {
    std::cout << "Expression: " << std::flush;
    std::getline(std::cin, str);
  } else {
    std::cout << "Usage: " << argv[0] << " <expr>" << std::endl;
    return 1;
  }

  auto lexer = Lexer(str);
  lexer.parse();
  auto parser = Parser(lexer.cbegin(), lexer.cend());
  auto expr = parser.expr();

  std::cout << "Previous: " << str << std::endl;
  Printer pr;
  expr->accept(*&pr);
  std::cout << "Post:     " << pr.finalize() << std::endl;
  return 0;
}

// 3*(4+5/(2-1))
// 21+42-30/(5+5)*(4-2)
