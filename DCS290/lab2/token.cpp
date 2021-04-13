#include "token.hpp"

using namespace std;

Token::Token(Type type, Span span) : HasSpan(span), type(type) {}

Token::Token(Type type, string::const_iterator begin,
             string::const_iterator end)
    : Token(type, Span(begin, end)) {}

bool Token::is(Type type) const { return this->type == type; }

bool Token::operator==(const Token &rhs) const {
  if (type == VOID && rhs.type == VOID)
    return get_span() == rhs.get_span();
  return type == rhs.type && get_span() == rhs.get_span();
}

bool Token::operator==(Type type) const { return is(type); }

bool Token::operator!=(Type type) const { return !is(type); }

string Token::to_string() const {
  string str;
  switch (type) {
  case INTEGER:
    str += "INTEGER";
    break;
  case OP_ADD:
    str += "OP_ADD";
    break;
  case OP_SUBTRACT:
    str += "OP_SUBTRACT";
    break;
  case OP_MULTIPLY:
    str += "OP_MULTIPLY";
    break;
  case OP_DIVIDE:
    str += "OP_DIVIDE";
    break;
  case OP_LPAREN:
    str += "OP_LPAREN";
    break;
  case OP_RPAREN:
    str += "OP_RPAREN";
    break;
  case VOID:
    str += "VOID";
    break;
  }
  str += ": (";
  str += HasSpan::to_string();
  str += ")";
  return str;
}

ostream &operator<<(ostream &out, const Token &token) {
  return out << token.to_string();
}
