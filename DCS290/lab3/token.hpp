#pragma once

#include "span.hpp"
#include <string>

enum class TokenType {
  Void,

  Ident,
  Number,
  String,
  Punct,

  Keyword,
};

struct Token : public HasSpan {
  TokenType type;

  Token(TokenType type, Span span);
  Token(TokenType type, std::string::const_iterator begin,
        std::string::const_iterator end);
  bool operator==(const Token &rhs) const;
  bool is(TokenType type) const;
  bool is_keyword(const std::string &name) const;
  bool operator==(TokenType type) const;
  bool operator!=(TokenType type) const;
  std::string to_string() const;
  friend std::ostream &operator<<(std::ostream &out, const Token &token);
};
