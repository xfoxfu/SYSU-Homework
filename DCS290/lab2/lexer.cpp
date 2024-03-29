#include "lexer.hpp"
#include "error.hpp"
#include "token.hpp"
#include <cassert>
#include <cctype>
#include <functional>
#include <string>
#include <vector>

bool match_digit(char chr) { return isdigit(chr); }
bool match_alpha(char chr) { return isalnum(chr) || chr == '_'; }
bool match_space(char chr) { return isspace(chr); }

#define RETURN_IF_PARSED(fn)                                                   \
  {                                                                            \
    auto token = fn();                                                         \
    if (!token.is(Token::VOID))                                                \
      return token;                                                            \
  }

using namespace std;

// ===== Common Functions =====
char Lexer::match(function<bool(char)> cur_cond) {
  if (finished())
    return '\0';
  auto chr = *_current;
  if (cur_cond(chr)) {
    _current += 1;
    return chr;
  } else
    return '\0';
}

pair<char, char> Lexer::match(function<bool(char, char)> cond) {
  if (finished())
    return make_pair('\0', '\0');
  auto cur = *_current;
  auto next = *(_current + 1);
  if (cond(cur, next)) {
    _current += 2;
    return make_pair(cur, next);
  } else
    return make_pair('\0', '\0');
}

bool Lexer::match(char cur_match) {
  return match([&](char cur) { return cur == cur_match; }) != '\0';
}

bool Lexer::match(char cur_match, char next_match) {
  return match([&](char cur, char next) {
           return cur == cur_match && next == next_match;
         }) != make_pair('\0', '\0');
}

char Lexer::progress() {
  if (finished())
    return '\0';
  return *(_current++);
}

// ===== Constructors =====
Lexer::Lexer(const string &input)
    : _begin(input.cbegin()), _end(input.cend()), _current(_begin) {}

Lexer::Lexer(string::const_iterator begin, string::const_iterator end)
    : _begin(begin), _end(end), _current(_begin) {}

// ===== Accessors =====
vector<Token>::const_iterator Lexer::cbegin() const { return _tokens.cbegin(); }

vector<Token>::const_iterator Lexer::cend() const { return _tokens.cend(); }

string::const_iterator Lexer::cstr_begin() const { return _begin; }

string::const_iterator Lexer::cstr_end() const { return _end; }

string::const_iterator Lexer::cstr_current() const { return _current; }

bool Lexer::finished() const { return _current == _end; }

size_t Lexer::token_count() const { return _tokens.size(); }

// ===== Parsers =====
Token Lexer::integer() {
  string data;
  char to_push;
  auto begin = _current;
  while ((to_push = match(match_digit)) != '\0') {
    data.push_back(to_push);
  }
  auto end = _current;
  if (!data.empty())
    return Token(Token::INTEGER, begin, end);
  else
    return Token(Token::VOID, begin, end);
}

Token Lexer::add() {
  if (match('+'))
    return Token(Token::OP_ADD, _current - 1, _current);
  else
    return Token(Token::VOID, _current, _current);
}

Token Lexer::subtract() {
  if (match('-'))
    return Token(Token::OP_SUBTRACT, _current - 1, _current);
  else
    return Token(Token::VOID, _current, _current);
}

Token Lexer::multiply() {
  if (match('*'))
    return Token(Token::OP_MULTIPLY, _current - 1, _current);
  else
    return Token(Token::VOID, _current, _current);
}

Token Lexer::divide() {
  if (match('/'))
    return Token(Token::OP_DIVIDE, _current - 1, _current);
  else
    return Token(Token::VOID, _current, _current);
}

Token Lexer::lparen() {
  if (match('('))
    return Token(Token::OP_LPAREN, _current - 1, _current);
  else
    return Token(Token::VOID, _current, _current);
}

Token Lexer::rparen() {
  if (match(')'))
    return Token(Token::OP_RPAREN, _current - 1, _current);
  else
    return Token(Token::VOID, _current, _current);
}

void Lexer::whitespace() {
  while (match(match_space) != '\0') {
  }
}

//===== Parse Function =====
void Lexer::parse() {
  vector<Error> errors;
  do {
    auto token = advance();
    if (!token.is(Token::VOID))
      _tokens.push_back(token);
    else if (!finished()) {
      auto chr = progress();
      assert(_current - 1 >= _begin);
      errors.push_back(Error(Span(cstr_begin(), cstr_end()),
                             Span(_current - 1, _current),
                             string("Unexpected token: '") + chr + "'"));
    }
  } while (!finished());
  if (!errors.empty())
    throw errors;
}

Token Lexer::advance() {
  whitespace();
  RETURN_IF_PARSED(integer);
  RETURN_IF_PARSED(add);
  RETURN_IF_PARSED(subtract);
  RETURN_IF_PARSED(multiply);
  RETURN_IF_PARSED(divide);
  RETURN_IF_PARSED(lparen);
  RETURN_IF_PARSED(rparen);
  return Token(Token::VOID, _current, _current);
}
