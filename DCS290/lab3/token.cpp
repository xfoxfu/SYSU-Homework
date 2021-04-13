#include "token.hpp"

using namespace std;

Token::Token(TokenType type, Span span) : HasSpan(span), type(type) {}

Token::Token(TokenType type, string::const_iterator begin,
             string::const_iterator end)
    : Token(type, Span(begin, end)) {}

bool Token::is(TokenType type) const { return this->type == type; }

bool Token::is_keyword(const std::string &name) const {
  return type == TokenType::Keyword && HasSpan::to_string() == name;
}

bool Token::operator==(const Token &rhs) const {
  if (type == TokenType::Void && rhs.type == TokenType::Void)
    return get_span() == rhs.get_span();
  return type == rhs.type && get_span() == rhs.get_span();
}

bool Token::operator==(TokenType type) const { return is(type); }

bool Token::operator!=(TokenType type) const { return !is(type); }

string Token::to_string() const {
  string str;
  switch (type) {
  case TokenType::Void:
    str += "Void";
    break;
  case TokenType::Alpha:
    str += "Alpha";
    break;
  case TokenType::BinDigit:
    str += "BinDigit";
    break;
  case TokenType::OctDigit:
    str += "OctDigit";
    break;
  case TokenType::Digit:
    str += "Digit";
    break;
  case TokenType::HexDigit:
    str += "HexDigit";
    break;
  case TokenType::Ident:
    str += "Ident";
    break;
  case TokenType::Number:
    str += "Number";
    break;
  case TokenType::BinNumber:
    str += "BinNumber";
    break;
  case TokenType::DecNumber:
    str += "DecNumber";
    break;
  case TokenType::OctNumber:
    str += "OctNumber";
    break;
  case TokenType::HexNumber:
    str += "HexNumber";
    break;
  case TokenType::String:
    str += "String";
    break;
  case TokenType::Keyword:
    str += "Keyword";
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
