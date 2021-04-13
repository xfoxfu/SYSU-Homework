#include "span.hpp"
#include <algorithm>

using namespace std;

Span::Span(string::const_iterator begin, string::const_iterator end)
    : begin(begin), end(end) {}

// [WARNING] this function is O(logN)
bool Span::operator==(const Span &rhs) const {
  return distance(begin, rhs.begin) == 0 && distance(end, rhs.end) == 0;
}

Span Span::operator+(const Span &rhs) const { return Span(begin, rhs.end); }

string Span::to_string() const { return '\"' + string(begin, end) + '\"'; }

ostream &operator<<(ostream &out, const Span &span) {
  return out << span.to_string();
}

HasSpan::HasSpan(Span span) : _span(span) {}

const Span &HasSpan::get_span() const { return _span; }

string HasSpan::to_string() const { return _span.to_string(); }

ostream &operator<<(ostream &out, const HasSpan &span) {
  return out << span.to_string();
}
