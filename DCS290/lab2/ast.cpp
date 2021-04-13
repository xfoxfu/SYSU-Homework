#include "ast.hpp"

using namespace std;
using namespace ast;

#define IMPL_ACCEPT(T)                                                         \
  void T::accept(Visitor &visitor) const { visitor.visit(*this); }

// ===== Constructor =====

Expr::Expr(Span span) : HasSpan(span) {}

Integer::Integer(Span span, int value) : Expr(span), value(value) {}

Binary::Binary(Span span, shared_ptr<Expr> left, Op op, shared_ptr<Expr> right)
    : Expr(span), left(left), op(op), right(right) {}

// ===== Equal =====

bool Expr::operator!=(const Expr &rhs) const { return !(*this == rhs); }

bool Integer::operator==(const Expr &rhs) const {
  return typeid(*this) == typeid(rhs) &&
         (value == dynamic_cast<const Integer &>(rhs).value);
}

bool Binary::operator==(const Expr &rhs) const {
  return typeid(*this) == typeid(rhs) &&
         (*left == *dynamic_cast<const Binary &>(rhs).left) &&
         (op == dynamic_cast<const Binary &>(rhs).op) &&
         (*right == *dynamic_cast<const Binary &>(rhs).right);
}

// ===== Accept =====

IMPL_ACCEPT(Integer)
IMPL_ACCEPT(Binary)

// ===== Output =====

ostream &operator<<(ostream &out, const Expr &value) {
  return value.print(out);
}

ostream &Integer::print(ostream &out) const {
  // out << "Integer { " << value << " }";
  out << value;
  return out;
}

ostream &Binary::print(ostream &out) const {
  out << "Binary { " << *left << " (";
  switch (op) {
  case Binary::ADD:
    out << '+';
    break;
  case Binary::SUBTRACT:
    out << '-';
    break;
  case Binary::MULTIPLY:
    out << '*';
    break;
  case Binary::DIVIDE:
    out << "/";
    break;
  }
  out << ") " << *right << " }";
  return out;
}
