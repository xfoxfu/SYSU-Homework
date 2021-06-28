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
  Token match();
  Token match(TokenType cur_ty);
  Token match(TokenType cur_ty, const std::string &cur_val);
  std::pair<Token, Token> match(TokenType cur_ty, TokenType next_ty);
  std::pair<Token, Token> match(TokenType cur_ty, const std::string &cur_val,
                                TokenType next_ty, const std::string &next_val);
  bool peek(TokenType cur_ty);
  bool peek(TokenType cur_ty, const std::string &cur_val);
  bool peek(TokenType cur_ty, TokenType next_ty);
  bool peek(TokenType cur_ty, const std::string &cur_val, TokenType next_ty,
            const std::string &next_val);
  bool peek2(TokenType next_ty);
  bool peek2(TokenType next_ty, const std::string &next_val);
  bool peek(TokenType cur_ty, TokenType next_ty, TokenType next2_ty);
  bool peek(TokenType cur_ty, const std::string &cur_val, TokenType next_ty,
            const std::string &next_val, TokenType next2_ty,
            const std::string &next2_val);
  Error make_error(std::string expected, bool to_end = false);

  // Move current pointer for ease of peek. This function is highly unsafe!
  void progress(size_t incr);
  // Move current pointer for ease of peek. This function is highly unsafe!
  void rollback(size_t incr);

public:
  // ===== Constructors =====
  explicit Parser(const std::vector<Token> &input);
  Parser(std::vector<Token>::const_iterator begin,
         std::vector<Token>::const_iterator end);

  // ===== Utilities =====
  void reset();

  // ===== Parsers =====
  std::shared_ptr<AstNode> Program();
  std::shared_ptr<AstNode> Func();
  std::shared_ptr<AstNode> Params();
  std::shared_ptr<AstNode> Param();
  std::shared_ptr<AstNode> Type();
  std::shared_ptr<AstNode> Vars();
  std::shared_ptr<AstNode> Block();
  std::shared_ptr<AstNode> Statement();
  std::shared_ptr<AstNode> Assignment();
  std::shared_ptr<AstNode> IfStmt();
  std::shared_ptr<AstNode> ReturnStmt();
  std::shared_ptr<AstNode> ExprStmt();
  std::shared_ptr<AstNode> Arguments();
  std::shared_ptr<AstNode> Call();
  std::shared_ptr<AstNode> Expr0();
  std::shared_ptr<AstNode> Expr1();
  std::shared_ptr<AstNode> Expr2();
  std::shared_ptr<AstNode> Expr3();
  std::shared_ptr<AstNode> Expr4();
  std::shared_ptr<AstNode> Expr5();
  std::shared_ptr<AstNode> Expr6();
  std::shared_ptr<AstNode> Expression();
  void ensure_finished();
};
