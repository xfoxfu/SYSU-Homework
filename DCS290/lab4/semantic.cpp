#include "semantic.hpp"
#include "ast.hpp"
#include "fmt/core.h"
#include "fmt/format.h"
#include "fmt/ostream.h"
#include <algorithm>
#include <iostream>
#include <iterator>
#include <optional>
#include <queue>
#include <sstream>
#include <string>
#include <sys/resource.h>
#include <utility>
#include <vector>

using std::string;
using std::vector;

SemanticSymbol::SemanticSymbol(std::string name, size_t layer,
                               std::vector<std::string> type)
    : name(name), layer(layer), type(type) {}

SemanticSymbolComparator::SemanticSymbolComparator() {}

bool SemanticSymbolComparator::operator()(const SemanticSymbol &lhs,
                                          const SemanticSymbol &rhs) const {
  return lhs.layer < rhs.layer;
}

SemanticVisitor::SemanticVisitor(const vector<Token> &total) {
  _begin = total.begin();
  _end = total.end();
  current_layer = 0;
  print_trace = false;
}

void SemanticVisitor::enter_layer() { current_layer += 1; }

void SemanticVisitor::exit_layer() {
  current_layer -= 1;
  while (symbol_table.size() > 0 && symbol_table.back().layer > current_layer) {
    if (print_trace) {
      fmt::print(std::cerr, "Eliminated Symbol [{}] {}: {}\n",
                 symbol_table.back().layer, symbol_table.back().name,
                 fmt::join(symbol_table.back().type, "->"));
    }
    symbol_table.pop_back();
  }
}

void SemanticVisitor::emit_error(const AstNode &cur, string cause) {
  errors.push_back(Error(_begin->get_span() + (_end - 1)->get_span(),
                         cur.get_span(), cause));
}

void SemanticVisitor::add_symbol(std::string name,
                                 std::vector<std::string> type) {
  add_symbol(name, type, current_layer);
}

void SemanticVisitor::add_symbol(std::string name,
                                 std::vector<std::string> type, size_t layer) {
  symbol_table.push_back(SemanticSymbol(name, layer, type));
  if (print_trace) {
    fmt::print(std::cerr, "Added      Symbol [{}] {}: {}\n", layer, name,
               fmt::join(type, "->"));
  }
  std::sort(symbol_table.begin(), symbol_table.end(),
            SemanticSymbolComparator());
}

bool SemanticVisitor::has_symbol(const std::string &name) {
  return std::find_if(symbol_table.begin(), symbol_table.end(),
                      [&name](const SemanticSymbol &val) {
                        return val.name == name;
                      }) != symbol_table.end();
}

vector<string> SemanticVisitor::get_symbol_type(const std::string &name) {
  return std::find_if(
             symbol_table.begin(), symbol_table.end(),
             [&name](const SemanticSymbol &val) { return val.name == name; })
      ->type;
}

bool vector_eq(const vector<string> left, const vector<string> right) {
  if (left.size() != right.size()) {
    return false;
  }
  for (size_t i = 0; i < left.size(); i++) {
    if (left[i] != right[i]) {
      return false;
    }
  }
  return true;
}

string normalize_type(const string &value) {
  if (value == "Keyword: (INT)")
    return "INT";
  if (value == "Keyword: (REAL)")
    return "REAL";
  if (value == "Keyword: (STRING)")
    return "STRING";
  return "VOID";
}

vector<string> SemanticVisitor::get_type(const AstNode &node) {
  if (node.value == "Program") {
    for (const auto &child : node.children) {
      get_type(*child);
    }
    return vector<string>{"VOID"};
  } else if (node.value == "Func") {
    enter_layer();
    auto ret = vector<string>{normalize_type(node.children[0]->value)};
    bool is_main = false;
    if (node.children[1]->value == "Keyword: (MAIN)") {
      is_main = true;
    }
    auto params = get_type(*node.children[is_main ? 3 : 2]);
    auto body = get_type(*node.children[is_main ? 4 : 3]);
    if (!vector_eq(body, ret) && body.front() != "VOID") {
      emit_error(
          node,
          fmt::format("function return type mismatch, expected {}, found {}",
                      ret.front(), body.front()));
    }
    params.push_back(ret.front());
    add_symbol(node.children[is_main ? 2 : 1]->value, params, 0);
    exit_layer();
    return params;
  } else if (node.value == "Params") {
    vector<string> ret;
    for (const auto &param : node.children) {
      ret.push_back(get_type(*param).front());
    }
    return ret;
  } else if (node.value == "Param") {
    auto ty = normalize_type(node.children[0]->value);
    add_symbol(node.children[1]->value, vector<string>{ty});
    return vector<string>{ty};
  } else if (node.value == "Vars") {
    auto var_ty = normalize_type(node.children[0]->value);
    add_symbol(node.children[1]->value, vector<string>{var_ty});
    if (node.children.size() > 2) {
      auto child_ty = get_type(*node.children[2]);
      if (child_ty.front() != var_ty) {
        emit_error(node, fmt::format("expected type {}, found expression of {}",
                                     var_ty, child_ty.front()));
      }
    }
    return vector<string>{"VOID"};
  } else if (node.value == "Block") {
    string type = "VOID";
    enter_layer();
    for (const auto &child : node.children) {
      auto child_ty = get_type(*child);
      if (child->value == "ReturnStmt" && type == "VOID") {
        type = child_ty.front();
      }
    }
    exit_layer();
    return vector<string>{type};
  } else if (node.value == "Assignment") {
    const auto &ident = node.children[0]->value;
    if (!has_symbol(ident)) {
      emit_error(*node.children[0],
                 fmt::format("assign to undefined variable {}", ident));
      return vector<string>{"VOID"};
    }
    const auto &type = get_type(*node.children[1]);
    if (type != get_symbol_type(ident)) {
      emit_error(*node.children[1],
                 fmt::format("assign to variable of {} with expression of {}",
                             get_symbol_type(ident).front(), type.front()));
    }
    return vector<string>{"VOID"};
  } else if (node.value == "IfStmt") {
    // expr, stmt, else
    if (get_type(*node.children[0]).front() != "INT") {
      emit_error(*node.children[0],
                 fmt::format("IF condition must be bool, but found {}",
                             get_type(*node.children[0]).front()));
    }
    get_type(*node.children[1]);
    if (node.children.size() > 2) {
      get_type(*node.children[2]);
    }
    return vector<string>{"VOID"};
  } else if (node.value == "ReturnStmt") {
    return get_type(*node.children[0]);
  } else if (node.value == "Arguments") {
    vector<string> ret;
    for (const auto &child : node.children) {
      ret.push_back(get_type(*child).front());
    }
    return ret;
  } else if (node.value == "Call") {
    const auto &ident = node.children[0]->value;
    if (ident == "Keyword: (READ)" || ident == "Keyword: (WRITE)") {
      return vector<string>{"VOID"};
    }
    if (!has_symbol(ident)) {
      emit_error(*node.children[0], fmt::format("{} is not defined", ident));
      return vector<string>{"VOID"};
    }
    auto fn_type = get_symbol_type(ident);
    if (fn_type.size() <= 1) {
      emit_error(*node.children[0], fmt::format("{} is not callable", ident));
    }
    auto ret_type = fn_type.back();
    auto args = get_type(*node.children[1]);
    size_t index = 0;
    for (const auto &arg_ty : args) {
      if (arg_ty != fn_type[index]) {
        emit_error(
            *node.children[1]->children[index],
            fmt::format("calling arguments mismatch, expected {} but found {}",
                        fn_type[index], arg_ty));
      }
      index += 1;
    }
    return vector<string>{ret_type};
  } else if (node.value == "Variable") {
    const auto &ident = node.children[0]->value;
    if (!has_symbol(ident)) {
      emit_error(*node.children[0],
                 fmt::format("assign to undefined variable {}", ident));
      return vector<string>{"VOID"};
    }
    return get_symbol_type(ident);
  } else if (node.value == "Expr") {
    std::optional<string> lhs_ty;
    if (node.children.size() > 1) {
      lhs_ty = get_type(*node.children[1]).front();
    }
    std::optional<string> rhs_ty;
    if (node.children.size() > 2) {
      rhs_ty = get_type(*node.children[2]).front();
    }
    if ((node.children[0]->value == "Punct: (+)" &&
         node.children.size() == 2) ||
        (node.children[0]->value == "Punct: (-)" &&
         node.children.size() == 2)) {
      if (lhs_ty != "INT") {
        emit_error(*node.children[1],
                   fmt::format("expected {}, found {}", "INT", lhs_ty.value()));
      }
      return vector<string>{lhs_ty.value()};
    } else if ((node.children[0]->value == "Punct: (!)")) {
      if (lhs_ty != "INT" && lhs_ty != "REAL") {
        emit_error(*node.children[1],
                   fmt::format("expected {}, found {}", "INT or REAL",
                               lhs_ty.value()));
      }
      return vector<string>{lhs_ty.value()};
    } else if (node.children[0]->value == "Punct: (*)" ||
               node.children[0]->value == "Punct: (/)" ||
               (node.children[0]->value == "Punct: (+)" &&
                node.children.size() == 3) ||
               (node.children[0]->value == "Punct: (-)" &&
                node.children.size() == 3)) {
      if (lhs_ty != "INT" && lhs_ty != "REAL") {
        emit_error(*node.children[1],
                   fmt::format("expected {}, found {}", "INT or REAL",
                               lhs_ty.value()));
      }
      if (rhs_ty != "INT" && rhs_ty != "REAL") {
        emit_error(*node.children[1],
                   fmt::format("expected {}, found {}", "INT or REAL",
                               rhs_ty.value()));
      }
      if (lhs_ty != rhs_ty) {
        emit_error(node, fmt::format("lhs and rhs type mismatch ({} vs {})",
                                     lhs_ty.value(), rhs_ty.value()));
      }
      return vector<string>{lhs_ty.value()};
    } else if (node.children[0]->value == "Punct: (>)" ||
               node.children[0]->value == "Punct: (<)" ||
               node.children[0]->value == "Punct: (>=)" ||
               node.children[0]->value == "Punct: (<=)" ||
               node.children[0]->value == "Punct: (==)" ||
               node.children[0]->value == "Punct: (!=)") {
      if (lhs_ty != "INT" && lhs_ty != "REAL") {
        emit_error(*node.children[1],
                   fmt::format("expected {}, found {}", "INT or REAL",
                               lhs_ty.value()));
      }
      if (rhs_ty != "INT" && rhs_ty != "REAL") {
        emit_error(*node.children[1],
                   fmt::format("expected {}, found {}", "INT or REAL",
                               rhs_ty.value()));
      }
      if (lhs_ty != rhs_ty.value()) {
        emit_error(node, fmt::format("lhs and rhs type mismatch ({} vs {})",
                                     lhs_ty.value(), rhs_ty.value()));
      }
      return vector<string>{"INT"};
    } else if (node.children[0]->value == "Punct: (&&)" ||
               node.children[0]->value == "Punct: (||)") {
      if (lhs_ty != rhs_ty.value()) {
        if (lhs_ty != "INT") {
          emit_error(*node.children[1], fmt::format("expected {}, found {}",
                                                    "INT", lhs_ty.value()));
        }
        if (rhs_ty != "INT") {
          emit_error(*node.children[1], fmt::format("expected {}, found {}",
                                                    "INT", rhs_ty.value()));
        }
        if (lhs_ty != rhs_ty) {
          emit_error(node, fmt::format("lhs and rhs type mismatch ({} vs {})",
                                       lhs_ty.value(), rhs_ty.value()));
        }
      }
      return vector<string>{"INT"};
    }
    fmt::print(std::cerr, "DEBUG: assert false hit {}",
               node.children[0]->value);
    assert(false);
  } else if (node.value.rfind("Number: ", 0) == 0) {
    return vector<string>{"INT"};
  } else if (node.value.rfind("String: ", 0) == 0) {
    return vector<string>{"STRING"};
  }
  fmt::print(std::cerr, "DEBUG: assert false hit {}", node.value);
  assert(false);
}

std::string convert_type_llvm(const std::string &original) {
  if (original == "INT") {
    return "i32";
  } else if (original == "REAL") {
    return "f64";
  } else if (original == "STRING") {
    return "string";
  } else {
    return "void";
  }
}

std::string convert_token(const std::string &value) {
  if (value.rfind("Number: ", 0) == 0) {
    return value.substr(9, value.size() - 9 - 1);
  } else if (value.rfind("Ident: ", 0) == 0) {
    return value.substr(8, value.size() - 8 - 1);
  } else if (value.rfind("String: ", 0) == 0) {
    return value.substr(9, value.size() - 9 - 1);
  } else if (value.rfind("Punct: ", 0) == 0) {
    return value.substr(8, value.size() - 8 - 1);
  } else if (value.rfind("Keyword: ", 0) == 0) {
    return value.substr(10, value.size() - 10 - 1);
  }
  return value;
}

string SemanticVisitor::emit_ir(const AstNode &node) {
  std::string ir;
  get_type(node);
  if (node.value == "Program") {
    fmt::format_to(std::back_inserter(ir), "source_filename = \"*.tiny\"\n");
    for (const auto &child : node.children) {
      fmt::format_to(std::back_inserter(ir), "{}\n", emit_ir(*child));
    }
  } else if (node.value == "Func") {
    current_register = 0;
    current_label = 0;
    bool is_main = false;
    if (node.children[1]->value == "Keyword: (MAIN)") {
      is_main = true;
    }
    auto type = get_type(node);
    fmt::format_to(std::back_inserter(ir), "define {} @{}({}) {{\n",
                   convert_type_llvm(type.back()),
                   convert_token(node.children[is_main ? 2 : 1]->value),
                   emit_ir(*node.children[is_main ? 3 : 2]));
    fmt::format_to(std::back_inserter(ir), "{}",
                   emit_ir(*node.children[is_main ? 4 : 3]));
    fmt::format_to(std::back_inserter(ir), "\n}}\n");
  } else if (node.value == "Params") {
    vector<string> ret;
    for (const auto &child : node.children) {
      ret.push_back(emit_ir(*child));
    }
    fmt::format_to(std::back_inserter(ir), "{}", fmt::join(ret, ", "));
  } else if (node.value == "Param") {
    auto type = get_type(node).front();
    fmt::format_to(std::back_inserter(ir), "{} %{}", convert_type_llvm(type),
                   convert_token(node.children[1]->value));
  } else if (node.value == "Vars") {
    fmt::format_to(std::back_inserter(ir), "%{} = alloca {}",
                   convert_token(node.children[1]->value),
                   convert_type_llvm(normalize_type(node.children[0]->value)));
    if (node.children.size() > 2) {
      fmt::format_to(std::back_inserter(ir), "\n  {}\n  %{} := %{}",
                     emit_ir(*node.children[2]),
                     convert_token(node.children[1]->value),
                     current_register - 1);
    }
  } else if (node.value == "Block") {
    vector<string> ret;
    for (const auto &child : node.children) {
      ret.push_back(emit_ir(*child));
    }
    fmt::format_to(std::back_inserter(ir), "  {}", fmt::join(ret, "\n  "));
  } else if (node.value == "Assignment") {
    fmt::format_to(
        std::back_inserter(ir), "{}\n  %{} = %{}", emit_ir(*node.children[1]),
        convert_token(node.children[0]->value), current_register - 1);
  } else if (node.value == "IfStmt") {
    size_t then_label = current_label++;
    size_t else_label = current_label++;
    size_t fin_label = current_label++;
    fmt::format_to(std::back_inserter(ir),
                   "{}\n"
                   "  br ir %{}, label %L{}, label %{}\n"
                   "L{}:\n"
                   "{}\n"
                   "  goto %L{}\n"
                   "L{}:\n",
                   emit_ir(*node.children[0]), current_register - 1, then_label,
                   else_label, then_label, emit_ir(*node.children[1]),
                   fin_label, else_label);
    if (node.children.size() > 2) {
      fmt::format_to(std::back_inserter(ir), "{}", emit_ir(*node.children[2]));
    }
    fmt::format_to(std::back_inserter(ir), "\nL{}:", fin_label);
  } else if (node.value == "ReturnStmt") {
    fmt::format_to(std::back_inserter(ir), "{}\n", emit_ir(*node.children[0]));
    fmt::format_to(std::back_inserter(ir), "  return {} %{}",
                   convert_type_llvm(get_type(*node.children[0]).front()),
                   current_register - 1);
  } else if (node.value == "Arguments") {
    assert(node.value == "Arguments");
  } else if (node.value == "Call") {
    vector<string> arg_ir;
    vector<string> arg_reg;
    for (const auto &arg : node.children[1]->children) {
      arg_ir.push_back(emit_ir(*arg));
      arg_reg.push_back(fmt::format("{} %{}",
                                    convert_type_llvm(get_type(*arg).front()),
                                    current_register - 1));
    }
    fmt::format_to(std::back_inserter(ir), "{}", fmt::join(arg_ir, "\n  "));
    const auto &ident = node.children[0]->value;
    std::string fn_type;
    if (get_symbol_type(ident).size() == 0) {
      fn_type = "void";
    } else {
      fn_type = convert_type_llvm(get_symbol_type(ident).back());
    }
    fmt::format_to(std::back_inserter(ir), "\n  %{} := call {} @{} ({})",
                   current_register++, fn_type, convert_token(ident),
                   fmt::join(arg_reg, ", "));

  } else if (node.value == "Variable") {
    fmt::format_to(std::back_inserter(ir), "%{} := %{}", current_register++,
                   convert_token(node.children[0]->value));
  } else if (node.value == "Expr") {
    if (node.children.size() == 2) {
      auto left_ir = emit_ir(*node.children[1]);
      auto left_reg = current_register - 1;
      fmt::format_to(std::back_inserter(ir), "{}\n  %{} := {} %{}", left_ir,
                     current_register, convert_token(node.children[0]->value),
                     left_reg);
    } else if (node.children.size() == 3) {
      auto left_ir = emit_ir(*node.children[1]);
      auto left_reg = current_register - 1;
      auto right_ir = emit_ir(*node.children[2]);
      auto right_reg = current_register - 1;
      fmt::format_to(std::back_inserter(ir), "{}\n  {}\n  %{} := {} %{} %{}",
                     left_ir, right_ir, current_register,
                     convert_token(node.children[0]->value), left_reg,
                     right_reg);
    }
    current_register += 1;
  } else if (node.value.rfind("Number: ", 0) == 0) {
    fmt::format_to(std::back_inserter(ir), "%{} := {}", current_register++,
                   convert_token(node.value));
  } else if (node.value.rfind("String: ", 0) == 0) {
    fmt::format_to(std::back_inserter(ir), "%{} := {}", current_register++,
                   convert_token(node.value));
  } else {
    fmt::print(std::cerr, "DEBUG: assert false hit {}\n", node.value);
    assert(false);
  }
  return ir;
}
