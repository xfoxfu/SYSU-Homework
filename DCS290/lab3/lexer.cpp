#include "lexer.hpp"
#include "error.hpp"
#include "token.hpp"
#include <cassert>
#include <cctype>
#include <functional>
#include <string>
#include <vector>

bool match_alpha(char chr) { return isalnum(chr); }
bool match_alpha_lodash(char chr) { return isalnum(chr) || chr == '_'; }
bool match_alpha_digit_lodash(char chr) {
  return isalnum(chr) || isdigit(chr) || chr == '_';
}
bool match_bin_digit(char chr) { return chr == '0' || chr == '1'; }
bool match_bin_digit_lodash(char chr) {
  return chr == '0' || chr == '1' || chr == '_';
}
bool match_oct_digit(char chr) {
  return isdigit(chr) && chr != '8' && chr != '9';
}
bool match_oct_digit_lodash(char chr) {
  return (isdigit(chr) && chr != '8' && chr != '9') || chr == '_';
}
bool match_digit(char chr) { return isdigit(chr); }
bool match_digit_lodash(char chr) { return isdigit(chr) || chr == '_'; }
bool match_hex_digit(char chr) { return isxdigit(chr); }
bool match_hex_digit_lodash(char chr) { return isxdigit(chr) || chr == '_'; }
bool match_space(char chr) { return isspace(chr); }

#define RETURN_IF_PARSED(fn)                                                   \
  {                                                                            \
    auto token = fn();                                                         \
    if (!token.is(TokenType::Void))                                            \
      return token;                                                            \
  }

#define FAIL_IF(expr)                                                          \
  if (expr) {                                                                  \
    return Token(TokenType::Void, _current, _current);                         \
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

const vector<Token> &Lexer::tokens() const { return _tokens; }

// ===== Parsers =====
Token Lexer::Ident() {
  auto begin = _current;
  FAIL_IF(match(match_alpha_lodash) == '\0');
  while (match(match_alpha_digit_lodash) != '\0') {
  }
  FAIL_IF(begin == _current);

  TokenType ty = TokenType::Ident;
  std::string value(begin, _current);
  if (value == "BEGIN")
    ty = TokenType::Keyword;
  if (value == "ELSE")
    ty = TokenType::Keyword;
  if (value == "END")
    ty = TokenType::Keyword;
  if (value == "IF")
    ty = TokenType::Keyword;
  if (value == "INT")
    ty = TokenType::Keyword;
  if (value == "MAIN")
    ty = TokenType::Keyword;
  if (value == "REAL")
    ty = TokenType::Keyword;
  if (value == "REF")
    ty = TokenType::Keyword;
  if (value == "RETURN")
    ty = TokenType::Keyword;
  if (value == "READ")
    ty = TokenType::Keyword;
  if (value == "WRITE")
    ty = TokenType::Keyword;
  return Token(ty, begin, _current);
}
Token Lexer::Number() {
  RETURN_IF_PARSED(BinNumber);
  RETURN_IF_PARSED(OctNumber);
  RETURN_IF_PARSED(HexNumber);
  RETURN_IF_PARSED(DecNumber);
  return Token(TokenType::Void, _current, _current);
}
Token Lexer::BinNumber() {
  FAIL_IF(match('0', 'b') == '\0');
  auto begin = _current;
  FAIL_IF(match(match_bin_digit) == '\0');
  while (match(match_bin_digit_lodash) != '\0') {
  }
  FAIL_IF(begin == _current);
  return Token(TokenType::Number, begin, _current);
}
Token Lexer::OctNumber() {
  FAIL_IF(match('0', 'o') == '\0');
  auto begin = _current;
  FAIL_IF(match(match_oct_digit) == '\0');
  while (match(match_oct_digit_lodash) != '\0') {
  }
  FAIL_IF(begin == _current);
  return Token(TokenType::Number, begin, _current);
}
Token Lexer::HexNumber() {
  FAIL_IF(match('0', 'x') == '\0');
  auto begin = _current;
  FAIL_IF(match(match_hex_digit) == '\0');
  while (match(match_hex_digit_lodash) != '\0') {
  }
  FAIL_IF(begin == _current);
  return Token(TokenType::Number, begin, _current);
}
Token Lexer::DecNumber() {
  FAIL_IF(match('0', 'b') == '\0');
  auto begin = _current;
  FAIL_IF(match(match_digit) == '\0');
  while (match(match_digit_lodash) != '\0') {
  }
  FAIL_IF(begin == _current);
  return Token(TokenType::Number, begin, _current);
}
Token Lexer::String() {
  FAIL_IF(match('"') == '\0');
  auto begin = _current;
  while (match([](char chr) { return chr != '"'; }) != '\0') {
  }
  auto end = _current;
  FAIL_IF(begin == end);
  FAIL_IF(match('"') == '\0');
  return Token(TokenType::String, begin, end);
}
void Lexer::whitespace() {
  while (match(match_space) != '\0' || *_current == '{') {
    // handle comments
    if (match('{') != '\0') {
      while (match([](char c) { return c != '}'; })) {
      }
      if (match('}') == '\0') {
        throw Error(Span(cstr_begin(), cstr_end()),
                    Span(_current - 1, _current),
                    string("Expecting close of comment: '}'"));
      }
    }
  }
}
Token Lexer::Punct() {
  if (match('_'))
    return Token(TokenType::Punct, _current - 1, _current);
  if (match('-'))
    return Token(TokenType::Punct, _current - 1, _current);
  if (match(','))
    return Token(TokenType::Punct, _current - 1, _current);
  if (match(';'))
    return Token(TokenType::Punct, _current - 1, _current);
  if (match(':', '='))
    return Token(TokenType::Punct, _current - 2, _current);
  if (match('!', '='))
    return Token(TokenType::Punct, _current - 2, _current);
  if (match('!'))
    return Token(TokenType::Punct, _current - 1, _current);
  if (match('('))
    return Token(TokenType::Punct, _current - 1, _current);
  if (match(')'))
    return Token(TokenType::Punct, _current - 1, _current);
  if (match('*'))
    return Token(TokenType::Punct, _current - 1, _current);
  if (match('/'))
    return Token(TokenType::Punct, _current - 1, _current);
  if (match('&', '&'))
    return Token(TokenType::Punct, _current - 2, _current);
  if (match('+'))
    return Token(TokenType::Punct, _current - 1, _current);
  if (match('<', '='))
    return Token(TokenType::Punct, _current - 2, _current);
  if (match('<'))
    return Token(TokenType::Punct, _current - 1, _current);
  if (match('=', '='))
    return Token(TokenType::Punct, _current - 2, _current);
  if (match('>', '='))
    return Token(TokenType::Punct, _current - 2, _current);
  if (match('>'))
    return Token(TokenType::Punct, _current - 1, _current);
  if (match('|', '|'))
    return Token(TokenType::Punct, _current - 2, _current);
  return Token(TokenType::Void, _current, _current);
}

//===== Parse Function =====
void Lexer::parse() {
  vector<Error> errors;
  do {
    try {
      auto token = advance();
      if (!token.is(TokenType::Void))
        _tokens.push_back(token);
      else if (!finished()) {
        auto chr = progress();
        assert(_current - 1 >= _begin);
        errors.push_back(Error(Span(cstr_begin(), cstr_end()),
                               Span(_current - 1, _current),
                               string("Unexpected token: '") + chr + "'"));
      }
    } catch (Error e) {
      errors.push_back(e);
    }
  } while (!finished());
  if (!errors.empty())
    throw errors;
}

Token Lexer::advance() {
  whitespace();
  RETURN_IF_PARSED(Punct);
  RETURN_IF_PARSED(Number);
  RETURN_IF_PARSED(String);
  RETURN_IF_PARSED(Ident);
  return Token(TokenType::Void, _current, _current);
}
