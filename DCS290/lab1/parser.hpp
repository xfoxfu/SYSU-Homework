#include "ast.hpp"
#include "error.hpp"
#include "token.hpp"
#include <functional>
#include <string>
#include <utility>
#include <vector>

class Parser {
private:
  std::vector<Token>::const_iterator _begin;
  std::vector<Token>::const_iterator _end;
  std::vector<Token>::const_iterator _current;

  // ===== Common Functions =====
  Token match(Token::Type cur_ty);
  std::pair<Token, Token> match(Token::Type cur_ty, Token::Type next_ty);
  bool peak(Token::Type cur_ty);
  bool peak(Token::Type cur_ty, Token::Type next_ty);
  bool peak(Token::Type cur_ty, Token::Type next_ty, Token::Type next2_ty);
  Error make_error(std::string expected, bool to_end = false);

public:
  // ===== Constructors =====
  explicit Parser(const std::vector<Token> &input);
  Parser(std::vector<Token>::const_iterator begin,
         std::vector<Token>::const_iterator end);

  // ===== Utilities =====
  void reset();

  // ===== Parsers =====
  std::shared_ptr<ast::Expr> term_integer();
  std::shared_ptr<ast::Expr> expr_terminal();
  std::shared_ptr<ast::Expr> expr_multiply_or_divide();
  std::shared_ptr<ast::Expr> expr_sum_or_subtract();
  std::shared_ptr<ast::Expr> expr();
  void ensure_finished();
};
