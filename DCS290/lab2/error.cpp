#include "error.hpp"
#include <cassert>
#include <exception>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>

using namespace std;

Error::Error(string total, string message, unsigned int begin,
             unsigned int length)
    : total(total), begin(begin), length(length), message(message) {
  if (length <= 0) {
    this->total.push_back(' ');
    this->length = 1;
  }
}

Error::Error(Span total_span, Span error_span, string message)
    : Error(string(total_span.begin, total_span.end), message,
            distance(total_span.begin, error_span.begin),
            distance(error_span.begin, error_span.end)) {}

string Error::to_string() const {
  stringstream ss;
  ss << *this;
  return ss.str();
}

ostream &operator<<(ostream &out, const Error &error) {
  assert(error.length >= 1);
  assert(error.begin + error.length <= error.total.size());

  string hint = "INPUT:" + to_string(error.begin) + ": ";
  out
      // output hint line
      << hint
      // output string before error
      << error.total.substr(0, error.begin)
      // output error section
      << error.total.substr(error.begin, error.length)
      // output rest string
      << error.total.substr(error.begin + error.length)
      << endl
      // output hint line
      << string(hint.length() + error.begin, ' ') << string(error.length, '^')
      << endl
      // output error message
      << "Error: " << error.message;
  return out;
}
