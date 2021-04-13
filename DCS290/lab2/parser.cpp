#include "parser.hpp"
#include "error.hpp"

using namespace std;
using namespace ast;

#define MARK_SPAN_BEGIN auto span_begin = _current

#define MARK_SPAN_GEN(d) (span_begin->get_span() + (_current - d)->get_span())

// ===== Common Functions =====
Token Parser::match(Token::Type cur_ty) {
  if (_current >= _end)
    throw cur_ty;

  auto &token = *_current;

  if (token != cur_ty)
    throw cur_ty;

  _current++;
  return token;
}

pair<Token, Token> Parser::match(Token::Type cur_ty, Token::Type next_ty) {
  if (_current >= _end - 1)
    throw cur_ty;

  auto &cur = *_current;
  auto &next = *(_current + 1);

  if (cur != cur_ty)
    throw cur_ty;
  if (next != next_ty)
    throw next_ty;

  _current += 2;
  return make_pair(cur, next);
}

bool Parser::peak(Token::Type cur_ty) {
  if (_current >= _end)
    return false;
  return *_current == cur_ty;
}

bool Parser::peak(Token::Type cur_ty, Token::Type next_ty) {
  if (_current + 1 >= _end)
    return false;
  return *_current == cur_ty && *(_current + 1) == next_ty;
}

bool Parser::peak(Token::Type cur_ty, Token::Type next_ty,
                  Token::Type next2_ty) {
  if (_current + 2 >= _end)
    return false;
  return *_current == cur_ty && *(_current + 1) == next_ty &&
         *(_current + 2) == next2_ty;
}

Error Parser::make_error(string message, bool to_end) {
  if (_begin < _end) {
    Span span = _current->get_span();
    if (_current >= _end)
      span = Span((_end - 1)->get_span().end, (_end - 1)->get_span().end);
    else {
      if (to_end)
        span = span + (_end - 1)->get_span();
    }
    return Error(_begin->get_span() + (_end - 1)->get_span(), span, message);
  } else
    return Error(string(), message);
}

// ===== Constructors =====
Parser::Parser(const vector<Token> &input)
    : _begin(input.cbegin()), _end(input.cend()), _current(input.cbegin()) {}

Parser::Parser(vector<Token>::const_iterator begin,
               vector<Token>::const_iterator end)
    : _begin(begin), _end(end), _current(begin) {}

// ===== Utilities =====
void Parser::reset() { _current = _begin; }

// ===== Parsers =====

// Integer    = @{ NUMBER+ }
shared_ptr<Expr> Parser::term_integer() {
  try {
    MARK_SPAN_BEGIN;
    auto token = match(Token::INTEGER);
    return make_shared<Integer>(
        MARK_SPAN_GEN(1),
        std::stoi(string(token.get_span().begin, token.get_span().end)));
  } catch (Token::Type) {
    throw make_error("Expected: integer");
  }
}

// Term       = { ("(" ~ Expr ~ ")") | Integer }
shared_ptr<Expr> Parser::expr_terminal() {
  // MARK_SPAN_BEGIN;
  if (peak(Token::OP_LPAREN)) {
    match(Token::OP_LPAREN);
    auto ret = expr();
    if (peak(Token::OP_RPAREN)) {
      match(Token::OP_RPAREN);
      return ret;
    } else
      throw make_error("Expected: ')'");
  } else if (peak(Token::INTEGER)) {
    return term_integer();
  } else
    throw make_error("Expected: (expr), polynomial, monomial, identifier");
}

// Expr2      = { Expr3 ~ (("*"|"/") ~ Expr3)* }
// shared_ptr<Expr> Parser::expr_multiply_or_divide() {
//   MARK_SPAN_BEGIN;
//   auto expr = expr_terminal();
//   while (peak(Token::OP_MULTIPLY) || peak(Token::OP_DIVIDE)) {
//     Binary::Op op;
//     if (peak(Token::OP_MULTIPLY)) {
//       match(Token::OP_MULTIPLY);
//       op = Binary::MULTIPLY;
//     } else if (peak(Token::OP_DIVIDE)) {
//       match(Token::OP_DIVIDE);
//       op = Binary::DIVIDE;
//     }

//     auto right = expr_multiply_or_divide();
//     if (typeid(*right) == typeid(Binary)) {
//       std::cerr << "Binary" << std::endl;
//       auto rbinary = dynamic_cast<Binary &>(*right);
//       auto mid = rbinary.left;
//       rbinary.left = make_shared<Binary>(MARK_SPAN_GEN(1), expr, op, mid);
//       expr = right;
//     } else {
//       expr = make_shared<Binary>(MARK_SPAN_GEN(1), expr, op, right);
//     }
//   }
//   return expr;
// }
shared_ptr<Expr> Parser::expr_multiply_or_divide() {
  MARK_SPAN_BEGIN;
  auto expr = expr_terminal();
  while (peak(Token::OP_MULTIPLY) || peak(Token::OP_DIVIDE)) {
    if (peak(Token::OP_MULTIPLY)) {
      match(Token::OP_MULTIPLY);
      expr = make_shared<Binary>(MARK_SPAN_GEN(1), expr, Binary::MULTIPLY,
                                 expr_terminal());
    } else if (peak(Token::OP_DIVIDE)) {
      match(Token::OP_DIVIDE);
      expr = make_shared<Binary>(MARK_SPAN_GEN(1), expr, Binary::DIVIDE,
                                 expr_terminal());
    }
  }
  return expr;
}

// Expr1      = { Expr2 ~ (("+"|"-") ~ Expr2)* }
shared_ptr<Expr> Parser::expr_sum_or_subtract() {
  MARK_SPAN_BEGIN;
  auto expr = expr_multiply_or_divide();
  while (peak(Token::OP_ADD) || peak(Token::OP_SUBTRACT)) {
    if (peak(Token::OP_ADD)) {
      match(Token::OP_ADD);
      expr = make_shared<Binary>(MARK_SPAN_GEN(1), expr, Binary::ADD,
                                 expr_multiply_or_divide());
    } else if (peak(Token::OP_SUBTRACT)) {
      match(Token::OP_SUBTRACT);
      expr = make_shared<Binary>(MARK_SPAN_GEN(1), expr, Binary::SUBTRACT,
                                 expr_multiply_or_divide());
    }
  }
  return expr;
}

// Expr       = { Expr1 }
shared_ptr<Expr> Parser::expr() {
  // MARK_SPAN_BEGIN;
  auto expr = expr_sum_or_subtract();
  return expr;
}

void Parser::ensure_finished() {
  if (_current != _end) {
    throw make_error("Expected: END_OF_INPUT", true);
  }
}
