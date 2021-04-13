#pragma once

namespace ast {

struct Expr;
struct Integer;
struct Binary;

} // namespace ast

class Visitor {
public:
  virtual void visit(const ast::Integer &value) = 0;
  virtual void visit(const ast::Binary &value) = 0;
};
