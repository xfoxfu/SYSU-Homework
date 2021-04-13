#pragma once

#include "span.hpp"
#include "visitor.hpp"
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#define DECL_INHERIT_FUNCS(T)                                                  \
  void accept(Visitor &visitor) const;                                         \
  bool operator==(const Expr &rhs) const;                                      \
  std::ostream &print(std::ostream &out) const;

std::ostream &operator<<(std::ostream &out, const ast::Expr &value);

namespace ast {

struct Expr : public HasSpan {
  Expr(Span span);

  virtual void accept(Visitor &visitor) const = 0;
  virtual bool operator==(const Expr &rhs) const = 0;
  bool operator!=(const Expr &rhs) const;
  virtual std::ostream &print(std::ostream &out) const = 0;
  friend std::ostream & ::operator<<(std::ostream &out, const Expr &value);
};

struct Integer final : public Expr {
  int value;

  Integer(Span span, int value);

  DECL_INHERIT_FUNCS(Integer)
};

struct Binary final : public Expr {
  typedef enum {
    ADD,
    SUBTRACT,
    MULTIPLY,
    DIVIDE,
  } Op;

  std::shared_ptr<Expr> left;
  Op op;
  std::shared_ptr<Expr> right;

  Binary(Span span, std::shared_ptr<Expr> left, Op op,
         std::shared_ptr<Expr> right);

  DECL_INHERIT_FUNCS(Binary)
};
} // namespace ast
