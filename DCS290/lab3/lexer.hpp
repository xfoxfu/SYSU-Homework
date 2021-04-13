#include "token.hpp"
#include <functional>
#include <string>
#include <utility>
#include <vector>

class Lexer {
private:
  std::string::const_iterator _begin;
  std::string::const_iterator _end;
  std::string::const_iterator _current;
  std::vector<Token> _tokens;

  // ===== Common Functions =====
  char match(std::function<bool(char)> cur_cond);
  std::pair<char, char> match(std::function<bool(char, char)> cond);
  bool match(char cur);
  bool match(char cur, char next);
  char progress();

public:
  // ===== Constructors =====
  explicit Lexer(const std::string &input);
  Lexer(std::string::const_iterator begin, std::string::const_iterator end);

  // ===== Accessors =====
  std::vector<Token>::const_iterator cbegin() const;
  std::vector<Token>::const_iterator cend() const;
  std::string::const_iterator cstr_begin() const;
  std::string::const_iterator cstr_end() const;
  std::string::const_iterator cstr_current() const;
  bool finished() const;
  size_t token_count() const;
  const std::vector<Token> &tokens() const;

  // ===== Parsers =====
  Token Ident();
  Token Number();
  Token BinNumber();
  Token DecNumber();
  Token OctNumber();
  Token HexNumber();
  Token String();
  Token Punct();
  void whitespace();

  //===== Parse Function =====
  void parse();
  Token advance();
};
