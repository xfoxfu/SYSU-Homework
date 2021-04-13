#pragma once

#include "span.hpp"
#include <string>

struct Token : public HasSpan {
  typedef enum {
    // Valued
    INTEGER,

    // Operators
    OP_ADD,
    OP_SUBTRACT,
    OP_MULTIPLY,
    OP_DIVIDE,
    OP_LPAREN,
    OP_RPAREN,

    // Void
    VOID,
  } Type;

  Type type;

  Token(Type type, Span span);
  Token(Type type, std::string::const_iterator begin,
        std::string::const_iterator end);
  bool operator==(const Token &rhs) const;
  bool is(Type type) const;
  bool operator==(Type type) const;
  bool operator!=(Type type) const;
  std::string to_string() const;
  friend std::ostream &operator<<(std::ostream &out, const Token &token);
};
