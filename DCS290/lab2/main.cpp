#include "lexer.hpp"
#include "parser.hpp"
#include "visitor.hpp"
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using std::string;
using std::vector;

class Printer final : public Visitor {
  size_t _depth;
  std::vector<std::vector<std::string>> _result;

  void _append(string value) {
    while (_result.size() <= _depth) {
      _result.push_back(vector<string>(_result.back().size() - 1));
    }
    _result[_depth].push_back(value);
    for (auto &layer : _result) {
      layer.resize(std::max(layer.size(), _result.back().size()));
    }
  }

public:
  Printer() {
    _depth = 0;
    _result.push_back(vector<string>());
  }
  virtual void visit(const ast::Integer &value) {
    _append(std::to_string(value.value));
  }
  virtual void visit(const ast::Binary &value) {
    string op_str;
    if (value.op == ast::Binary::ADD) {
      op_str = '+';
    } else if (value.op == ast::Binary::SUBTRACT) {
      op_str = '-';
    } else if (value.op == ast::Binary::MULTIPLY) {
      op_str = '*';
    } else if (value.op == ast::Binary::DIVIDE) {
      op_str = '/';
    }
    _append(op_str);

    _depth += 1;
    value.left->accept(*this);
    value.right->accept(*this);
    _depth -= 1;
  }
  const vector<vector<string>> &result() const { return _result; }
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
  std::cout << "AST: " << std::endl;
  for (const auto &line : pr.result()) {
    for (const auto &value : line) {
      std::cout << std::setw(4) << value;
    }
    std::cout << std::endl;
  }
  return 0;
}

// 3*(4+5/(2-1))
// 21+42-30/(5+5)*(4-2)
