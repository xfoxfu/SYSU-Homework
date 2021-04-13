#pragma once

#include <string>

struct Span {
  std::string::const_iterator begin;
  std::string::const_iterator end;

  Span(std::string::const_iterator begin, std::string::const_iterator end);
  bool operator==(const Span &rhs) const;
  Span operator+(const Span &rhs) const;
  std::string to_string() const;
  friend std::ostream &operator<<(std::ostream &out, const Span &span);
};

class HasSpan {
private:
  Span _span;

public:
  HasSpan(Span span);
  const Span &get_span() const;
  std::string to_string() const;
  friend std::ostream &operator<<(std::ostream &out, const HasSpan &span);
};
