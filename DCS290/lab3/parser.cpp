#include "parser.hpp"
#include "error.hpp"

#define DEBUG_TRACE_OFF
#ifndef DEBUG_TRACE_OFF
#include <iostream>
#define DEBUG_TRACE(s) std::cout << "Parser: " #s << std::endl;
#else
#define DEBUG_TRACE(s)
#endif

using namespace std;

#define MARK_SPAN_BEGIN auto span_begin = _current

#define MARK_SPAN_GEN(d) (span_begin->get_span() + (_current - d)->get_span())

using children_t = vector<shared_ptr<AstNode>>;

// ===== Common Functions =====
Token Parser::match() {
  if (_current >= _end)
    throw make_error("Unexpected EOF.");

  auto &token = *_current;

  _current++;
  return token;
}

Token Parser::match(TokenType cur_ty) {
  if (_current >= _end)
    throw make_error(string("Expected token of: ") +
                     token_type_to_string(cur_ty));

  auto &token = *_current;

  if (token != cur_ty)
    throw make_error(string("Expected token of: ") +
                     token_type_to_string(cur_ty));

  _current++;
  return token;
}

Token Parser::match(TokenType cur_ty, const std::string &cur_val) {
  if (_current >= _end)
    throw make_error(string("Expected token of: ") +
                     token_type_to_string(cur_ty) + " with value " + cur_val +
                     ", found " + _current->to_string());

  auto &token = *_current;

  if (token != cur_ty || token != cur_val)
    throw make_error(string("Expected token of: ") +
                     token_type_to_string(cur_ty) + " with value " + cur_val +
                     ", found " + _current->to_string());

  _current++;
  return token;
}

pair<Token, Token> Parser::match(TokenType cur_ty, TokenType next_ty) {
  if (_current >= _end - 1)
    throw make_error(string("Expected token of: ") +
                     token_type_to_string(cur_ty));

  auto &cur = *_current;
  auto &next = *(_current + 1);

  if (cur != cur_ty)
    throw make_error(string("Expected token of: ") +
                     token_type_to_string(cur_ty));
  if (next != next_ty)
    throw make_error(string("Expected token of: ") +
                     token_type_to_string(next_ty));

  _current += 2;
  return make_pair(cur, next);
}

std::pair<Token, Token> Parser::match(TokenType cur_ty,
                                      const std::string &cur_val,
                                      TokenType next_ty,
                                      const std::string &next_val) {
  if (_current >= _end - 1)
    throw make_error(string("Expected token of: ") +
                     token_type_to_string(cur_ty));

  auto &cur = *_current;
  auto &next = *(_current + 1);

  if (cur != cur_ty || cur != cur_val)
    throw make_error(string("Expected token of: ") +
                     token_type_to_string(cur_ty) + " with value " + cur_val +
                     ", found " + _current->to_string());
  if (next != next_ty || cur != next_val)
    throw make_error(string("Expected token of: ") +
                     token_type_to_string(next_ty) + " with value " + next_val +
                     ", found " + _current->to_string());

  _current += 2;
  return make_pair(cur, next);
}

bool Parser::peek(TokenType cur_ty) {
  if (_current >= _end)
    return false;
  return *_current == cur_ty;
}

bool Parser::peek(TokenType cur_ty, const std::string &cur_val) {
  if (_current >= _end)
    return false;
  return *_current == cur_ty && *_current == cur_val;
}

bool Parser::peek(TokenType cur_ty, TokenType next_ty) {
  if (_current + 1 >= _end)
    return false;
  return *_current == cur_ty && *(_current + 1) == next_ty;
}

bool Parser::peek(TokenType cur_ty, const std::string &cur_val,
                  TokenType next_ty, const std::string &next_val) {
  if (_current + 1 >= _end)
    return false;
  return *_current == cur_ty && *_current == cur_val &&
         *(_current + 1) == next_ty && *(_current + 1) == next_val;
}

bool Parser::peek2(TokenType next_ty) {
  if (_current + 1 >= _end)
    return false;
  return *(_current + 1) == next_ty;
}

bool Parser::peek2(TokenType next_ty, const std::string &next_val) {
  if (_current + 1 >= _end)
    return false;
  return *(_current + 1) == next_ty && *(_current + 1) == next_val;
}

bool Parser::peek(TokenType cur_ty, TokenType next_ty, TokenType next2_ty) {
  if (_current + 2 >= _end)
    return false;
  return *_current == cur_ty && *(_current + 1) == next_ty &&
         *(_current + 2) == next2_ty;
}

bool Parser::peek(TokenType cur_ty, const std::string &cur_val,
                  TokenType next_ty, const std::string &next_val,
                  TokenType next2_ty, const std::string &next2_val) {
  if (_current + 2 >= _end)
    return false;
  return *_current == cur_ty && *_current == cur_val &&
         *(_current + 1) == next_ty && *(_current + 1) == next_val &&
         *(_current + 2) == next2_ty && *(_current + 2) == next2_val;
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

void Parser::progress(size_t incr) { _current += incr; }
void Parser::rollback(size_t incr) { _current -= incr; }

// ===== Constructors =====
Parser::Parser(const vector<Token> &input)
    : _begin(input.cbegin()), _end(input.cend()), _current(input.cbegin()) {}

Parser::Parser(vector<Token>::const_iterator begin,
               vector<Token>::const_iterator end)
    : _begin(begin), _end(end), _current(begin) {}

// ===== Utilities =====
void Parser::reset() { _current = _begin; }

// ===== Parsers =====

// Program      -> (Func | Vars)*
shared_ptr<AstNode> Parser::Program() {
  DEBUG_TRACE(Program);
  MARK_SPAN_BEGIN;
  children_t children;

  while (peek(TokenType::Keyword, "INT") || peek(TokenType::Keyword, "REAL") ||
         peek(TokenType::Keyword, "STRING")) {
    progress(1);
    if (peek(TokenType::Keyword, "MAIN") || peek2(TokenType::Punct, "(")) {
      rollback(1);
      children.push_back(Func());
    } else {
      rollback(1);
      children.push_back(Vars());
    }
  }
  ensure_finished();

  return make_shared<AstNode>(MARK_SPAN_GEN(1), "Program", children);
}
// Func         -> Type [ 'MAIN' ] Ident '(' Params ')' Block
shared_ptr<AstNode> Parser::Func() {
  DEBUG_TRACE(Func);
  MARK_SPAN_BEGIN;
  children_t children;

  children.push_back(Type());
  if (peek(TokenType::Keyword, "MAIN")) {
    children.push_back(make_shared<AstNode>(match(TokenType::Keyword, "MAIN")));
  }
  children.push_back(make_shared<AstNode>(match(TokenType::Ident)));
  match(TokenType::Punct, "(");
  children.push_back(Params());
  match(TokenType::Punct, ")");
  children.push_back(Block());

  return make_shared<AstNode>(MARK_SPAN_GEN(1), "Func", children);
}
// Params -> [Param (',' Param)* ]
shared_ptr<AstNode> Parser::Params() {
  DEBUG_TRACE(Params);
  MARK_SPAN_BEGIN;
  children_t children;

  if (!(peek(TokenType::Keyword, "REF") || peek(TokenType::Keyword, "INT") ||
        peek(TokenType::Keyword, "REAL") ||
        peek(TokenType::Keyword, "STRING"))) {
    return make_shared<AstNode>(MARK_SPAN_GEN(1), "Params");
  }
  children.push_back(Param());
  while (peek(TokenType::Punct, ",")) {
    match(TokenType::Punct, ",");
    children.push_back(Param());
  }

  return make_shared<AstNode>(MARK_SPAN_GEN(1), "Params", children);
}
// Param  -> [ 'REF' ] Type Ident
shared_ptr<AstNode> Parser::Param() {
  DEBUG_TRACE(Param);
  MARK_SPAN_BEGIN;

  shared_ptr<AstNode> is_ref = nullptr;
  if (peek(TokenType::Keyword, "REF")) {
    is_ref = make_shared<AstNode>(match(TokenType::Keyword, "REF"));
  }
  auto type = Type();
  auto ident = make_shared<AstNode>(match(TokenType::Ident));

  auto ast = make_shared<AstNode>(MARK_SPAN_GEN(1), "Param");
  if (is_ref != nullptr) {
    ast->children.push_back(is_ref);
  }
  ast->children.push_back(type);
  ast->children.push_back(ident);
  return ast;
}
// Type         -> 'INT' | 'REAL'
shared_ptr<AstNode> Parser::Type() {
  DEBUG_TRACE(Type);
  MARK_SPAN_BEGIN;
  if (peek(TokenType::Keyword, "INT") || peek(TokenType::Keyword, "REAL") ||
      peek(TokenType::Keyword, "STRING")) {
    return make_shared<AstNode>(match(TokenType::Keyword));
  }
  throw make_error("Expected Type INT, REAL or STRING.");
}
// Vars         -> Type Ident [ ':=' Expression ] ';'
shared_ptr<AstNode> Parser::Vars() {
  DEBUG_TRACE(Vars);
  MARK_SPAN_BEGIN;
  shared_ptr<AstNode> expr = nullptr;

  auto type = Type();
  auto ident = make_shared<AstNode>(match(TokenType::Ident));
  if (peek(TokenType::Punct, ":=")) {
    match(TokenType::Punct, ":=");
    expr = Expression();
  }
  match(TokenType::Punct, ";");

  auto ast = make_shared<AstNode>(MARK_SPAN_GEN(1), "Vars");
  ast->children.push_back(type);
  if (expr != nullptr) {
    ast->children.push_back(expr);
  }
  ast->children.push_back(ident);
  return ast;
}
// Block        -> 'BEGIN' Statement* 'END'
shared_ptr<AstNode> Parser::Block() {
  DEBUG_TRACE(Block);
  MARK_SPAN_BEGIN;
  children_t children;

  match(TokenType::Keyword, "BEGIN");
  while (peek(TokenType::Keyword, "BEGIN") || peek(TokenType::Keyword, "INT") ||
         peek(TokenType::Keyword, "REAL") ||
         peek(TokenType::Keyword, "STRING") || peek(TokenType::Ident) ||
         peek(TokenType::Keyword, "READ") ||
         peek(TokenType::Keyword, "WRITE") ||
         peek(TokenType::Keyword, "RETURN") || peek(TokenType::Keyword, "IF")) {
    children.push_back(Statement());
  }
  match(TokenType::Keyword, "END");

  return make_shared<AstNode>(MARK_SPAN_GEN(1), "Block", children);
}
// Statement    -> Block | Vars | Assignment | ReturnStmt | IfStmt | ExprStmt
shared_ptr<AstNode> Parser::Statement() {
  DEBUG_TRACE(Statement);
  MARK_SPAN_BEGIN;
  if (peek(TokenType::Keyword, "BEGIN")) {
    return Block();
  } else if (peek(TokenType::Keyword, "INT") ||
             peek(TokenType::Keyword, "REAL") ||
             peek(TokenType::Keyword, "STRING")) {
    return Vars();
  } else if (peek(TokenType::Ident) && peek2(TokenType::Punct, ":=")) {
    return Assignment();
  } else if (peek(TokenType::Ident) || peek(TokenType::Keyword, "READ") ||
             peek(TokenType::Keyword, "WRITE")) {
    return ExprStmt();
  } else if (peek(TokenType::Keyword, "RETURN")) {
    return ReturnStmt();
  } else if (peek(TokenType::Keyword, "IF")) {
    return IfStmt();
  } else {
    throw make_error("Expected statement.");
  }
}
// Assignment   -> Ident ':=' Expression ';'
shared_ptr<AstNode> Parser::Assignment() {
  DEBUG_TRACE(Assignment);
  MARK_SPAN_BEGIN;
  shared_ptr<AstNode> expr = nullptr;

  auto ident = make_shared<AstNode>(match(TokenType::Ident));
  match(TokenType::Punct, ":=");
  expr = Expression();
  match(TokenType::Punct, ";");

  auto ast = make_shared<AstNode>(MARK_SPAN_GEN(1), "Assignment");
  ast->children.push_back(expr);
  return ast;
}
// IfStmt       -> 'IF' '(' Expression ')' Statement [ 'ELSE' Statement ]
shared_ptr<AstNode> Parser::IfStmt() {
  DEBUG_TRACE(IfStmt);
  MARK_SPAN_BEGIN;
  shared_ptr<AstNode> el = nullptr;

  match(TokenType::Keyword, "IF");
  match(TokenType::Punct, "(");
  auto expr = Expression();
  match(TokenType::Punct, ")");
  auto stmt = Statement();
  if (peek(TokenType::Keyword, "ELSE")) {
    match(TokenType::Keyword, "ELSE");
    el = Statement();
  }

  auto ast = make_shared<AstNode>(MARK_SPAN_GEN(1), "IfStmt");
  ast->children.push_back(expr);
  ast->children.push_back(stmt);
  if (el != nullptr) {
    ast->children.push_back(el);
  }
  return ast;
}
// ReturnStmt   -> 'RETURN' Expression ';'
shared_ptr<AstNode> Parser::ReturnStmt() {
  DEBUG_TRACE(ReturnStmt);
  MARK_SPAN_BEGIN;

  match(TokenType::Keyword, "RETURN");
  auto expr = Expression();
  match(TokenType::Punct, ";");

  auto ast = make_shared<AstNode>(MARK_SPAN_GEN(1), "ReturnStmt");
  ast->children.push_back(expr);
  return ast;
}
// ExprStmt     -> Expression ';'
shared_ptr<AstNode> Parser::ExprStmt() {
  auto expr = Expression();
  match(TokenType::Punct, ";");
  return expr;
}
// Arguments -> [Expression (',' Expression)*]
shared_ptr<AstNode> Parser::Arguments() {
  DEBUG_TRACE(Arguments);
  MARK_SPAN_BEGIN;
  children_t children;

  if (peek(TokenType::Punct, ")")) {
    return make_shared<AstNode>(MARK_SPAN_GEN(1), "Arguments");
  }
  children.push_back(Expression());
  while (peek(TokenType::Punct, ",")) {
    match(TokenType::Punct, ",");
    children.push_back(Expression());
  }

  return make_shared<AstNode>(MARK_SPAN_GEN(1), "Arguments", children);
}
// Call         -> Ident '(' Arguments ')'
shared_ptr<AstNode> Parser::Call() {
  DEBUG_TRACE(Call);
  MARK_SPAN_BEGIN;
  children_t children;

  if (peek(TokenType::Keyword, "READ") || peek(TokenType::Keyword, "WRITE")) {
    children.push_back(make_shared<AstNode>(match(TokenType::Keyword)));
  } else {
    children.push_back(make_shared<AstNode>(match(TokenType::Ident)));
  }
  match(TokenType::Punct, "(");
  children.push_back(Arguments());
  match(TokenType::Punct, ")");

  return make_shared<AstNode>(MARK_SPAN_GEN(1), "Call", children);
}
// Expr0        -> '(' Expression ')' | Call | Number | Ident | String
shared_ptr<AstNode> Parser::Expr0() {
  DEBUG_TRACE(Expr0);
  MARK_SPAN_BEGIN;
  if (peek(TokenType::Punct, "(")) {
    match(TokenType::Punct, "(");
    auto expr = Expression();
    match(TokenType::Punct, ")");
    return expr;
  } else if (peek(TokenType::Ident) || peek(TokenType::Keyword, "READ") ||
             peek(TokenType::Keyword, "WRITE")) {
    if (peek2(TokenType::Punct, "(")) {
      return Call();
    } else {
      auto ident = vector{make_shared<AstNode>(match(TokenType::Ident))};
      return make_shared<AstNode>(MARK_SPAN_GEN(1), "Variable", ident);
    }
  } else if (peek(TokenType::Number)) {
    return make_shared<AstNode>(match(TokenType::Number));
  } else if (peek(TokenType::String)) {
    return make_shared<AstNode>(match(TokenType::String));
  }

  throw make_error("Expected expression.");
}
// Expr1        -> [('+' | '-' | '!')] Expr0
shared_ptr<AstNode> Parser::Expr1() {
  DEBUG_TRACE(Expr1);
  MARK_SPAN_BEGIN;
  shared_ptr<AstNode> op = nullptr;

  if (peek(TokenType::Punct, "+")) {
    op = make_shared<AstNode>(match(TokenType::Punct, "+"));
  } else if (peek(TokenType::Punct, "-")) {
    op = make_shared<AstNode>(match(TokenType::Punct, "-"));
  } else if (peek(TokenType::Punct, "!")) {
    op = make_shared<AstNode>(match(TokenType::Punct, "!"));
  }
  auto expr = Expr0();

  if (op == nullptr) {
    return expr;
  }
  auto ast = make_shared<AstNode>(MARK_SPAN_GEN(1), "Expr");
  ast->children.push_back(op);
  ast->children.push_back(expr);
  return ast;
}
// Expr2        -> Expr1 (('*' | '/') Expr1)*
shared_ptr<AstNode> Parser::Expr2() {
  DEBUG_TRACE(Expr2);
  MARK_SPAN_BEGIN;
  children_t children;

  children.push_back(Expr1());
  while (peek(TokenType::Punct, "*") || peek(TokenType::Punct, "/")) {
    children.push_back(make_shared<AstNode>(match(TokenType::Punct)));
    children.push_back(Expr1());
  }

  if (children.size() == 1) {
    return children.front();
  }
  return make_shared<AstNode>(MARK_SPAN_GEN(1), "Expr", children);
}
// Expr3        -> Expr2 (('+' | '-') Expr2)*
shared_ptr<AstNode> Parser::Expr3() {
  DEBUG_TRACE(Expr3);
  MARK_SPAN_BEGIN;
  children_t children;

  children.push_back(Expr2());
  while (peek(TokenType::Punct, "+") || peek(TokenType::Punct, "-")) {
    children.push_back(make_shared<AstNode>(match(TokenType::Punct)));
    children.push_back(Expr2());
  }

  if (children.size() == 1) {
    return children.front();
  }
  return make_shared<AstNode>(MARK_SPAN_GEN(1), "Expr", children);
}
// Expr4        -> Expr3 (('>' | '<' | '>=' | '<=') Expr3)*
shared_ptr<AstNode> Parser::Expr4() {
  DEBUG_TRACE(Expr4);
  MARK_SPAN_BEGIN;
  children_t children;

  children.push_back(Expr3());
  while (peek(TokenType::Punct, ">") || peek(TokenType::Punct, "<") ||
         peek(TokenType::Punct, ">=") || peek(TokenType::Punct, "<=")) {
    children.push_back(make_shared<AstNode>(match(TokenType::Punct)));
    children.push_back(Expr3());
  }

  if (children.size() == 1) {
    return children.front();
  }
  return make_shared<AstNode>(MARK_SPAN_GEN(1), "Expr", children);
}
// Expr5        -> Expr4 (('==' | '!=') Expr4)*
shared_ptr<AstNode> Parser::Expr5() {
  DEBUG_TRACE(Expr5);
  MARK_SPAN_BEGIN;
  children_t children;

  children.push_back(Expr4());
  while (peek(TokenType::Punct, "==") || peek(TokenType::Punct, "!=")) {
    children.push_back(make_shared<AstNode>(match(TokenType::Punct)));
    children.push_back(Expr4());
  }

  if (children.size() == 1) {
    return children.front();
  }
  return make_shared<AstNode>(MARK_SPAN_GEN(1), "Expr", children);
}
// Expr6        -> Expr5 (('&&' | '||') Expr5)*
shared_ptr<AstNode> Parser::Expr6() {
  DEBUG_TRACE(Expr6);
  MARK_SPAN_BEGIN;
  children_t children;

  children.push_back(Expr5());
  while (peek(TokenType::Punct, "&&") || peek(TokenType::Punct, "||")) {
    children.push_back(make_shared<AstNode>(match(TokenType::Punct)));
    children.push_back(Expr5());
  }

  if (children.size() == 1) {
    return children.front();
  }
  return make_shared<AstNode>(MARK_SPAN_GEN(1), "Expr", children);
}
// Expression   -> Expr6
shared_ptr<AstNode> Parser::Expression() {
  DEBUG_TRACE(Expression);
  return Expr6();
}

void Parser::ensure_finished() {
  if (_current != _end) {
    throw make_error("Expected: END_OF_INPUT", true);
  }
}
