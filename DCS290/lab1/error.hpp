#pragma once

#include "span.hpp"
#include <exception>
#include <string>
#include <vector>

struct Error final {
  std::string total;
  unsigned int begin;
  unsigned int length;
  std::string message;

  Error(std::string total, std::string message, unsigned int begin = 0,
        unsigned int length = 0);
  Error(Span total_span, Span error_span, std::string message);

  std::string to_string() const;
  friend std::ostream &operator<<(std::ostream &out, const Error &error);
};
