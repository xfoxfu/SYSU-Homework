#include "token.hpp"

using namespace std;

string token_type_to_string(TokenType type) {
  switch (type) {
  case TokenType::Void:
    return "Void";
  case TokenType::Ident:
    return "Ident";
  case TokenType::Number:
    return "Number";
  case TokenType::String:
    return "String";
  case TokenType::Punct:
    return "Punct";
  case TokenType::Keyword:
    return "Keyword";
  }
}

Token::Token(TokenType type, Span span) : HasSpan(span), type(type) {}

Token::Token(TokenType type, string::const_iterator begin,
             string::const_iterator end)
    : Token(type, Span(begin, end)) {}

bool Token::is(TokenType type) const { return this->type == type; }

bool Token::is_keyword(const std::string &name) const {
  return type == TokenType::Keyword && HasSpan::to_string() == name;
}

string Token::value() const { return HasSpan::to_string(); }

bool Token::operator==(const Token &rhs) const {
  if (type == TokenType::Void && rhs.type == TokenType::Void)
    return get_span() == rhs.get_span();
  return type == rhs.type && get_span() == rhs.get_span();
}

bool Token::operator==(TokenType type) const { return is(type); }

bool Token::operator!=(TokenType type) const { return !is(type); }

bool Token::operator==(const std::string &val) const { return value() == val; }
bool Token::operator!=(const std::string &val) const { return value() != val; }

string Token::to_string() const {
  string str;
  switch (type) {
  case TokenType::Void:
    str += "Void";
    break;
  case TokenType::Ident:
    str += "Ident";
    break;
  case TokenType::Number:
    str += "Number";
    break;
  case TokenType::String:
    str += "String";
    break;
  case TokenType::Punct:
    str += "Punct";
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
