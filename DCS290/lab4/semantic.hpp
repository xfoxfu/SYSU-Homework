#pragma once

#include "ast.hpp"
#include "error.hpp"
#include <queue>
#include <string>
#include <utility>
#include <vector>

struct SemanticSymbolComparator;
struct SemanticSymbol {
  std::string name;
  size_t layer;
  std::vector<std::string> type;

  using comparator = SemanticSymbolComparator;

  SemanticSymbol(std::string name, size_t layer, std::vector<std::string> type);
};

struct SemanticSymbolComparator {
  SemanticSymbolComparator();
  bool operator()(const SemanticSymbol &lhs, const SemanticSymbol &rhs) const;
};

struct SemanticVisitor {
  std::vector<SemanticSymbol> symbol_table;
  size_t current_layer;
  std::vector<Error> errors;
  std::vector<Token>::const_iterator _begin;
  std::vector<Token>::const_iterator _end;
  bool print_trace;
  size_t current_register;
  size_t current_label;

  SemanticVisitor(const std::vector<Token> &total);
  void enter_layer();
  void exit_layer();
  std::vector<std::string> get_type(const AstNode &node);
  void add_symbol(std::string name, std::vector<std::string> type);
  void add_symbol(std::string name, std::vector<std::string> type,
                  size_t layer);
  void emit_error(const AstNode &cur, std::string cause);
  bool has_symbol(const std::string &name);
  std::vector<std::string> get_symbol_type(const std::string &name);
  std::string emit_ir(const AstNode &node);
};
