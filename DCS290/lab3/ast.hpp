#pragma once

#include "span.hpp"
#include "visitor.hpp"
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

enum class AstType {
  Alpha,
  BinDigit,
  OctDigit,
  Digit,
  HexDigit,
  Ident,
  Number,
  BinNumber,
  DecNumber,
  OctNumber,
  HexNumber,
  String,
  Program,
  Func,
  FormalParams,
  FormalParam,
  Type,
  Vars,
  Block,
  Statement,
  Assignment,
  IfStmt,
  ReturnStmt,
  ActualParams,
  Call,
  Unit0,
  Unit1,
  Unit2,
  Unit3,
  Unit4,
  Unit5,
  Unit6,
  Expression,
};

struct AstNode : public HasSpan {
  AstType ty;
  std::vector<std::shared_ptr<AstNode>> children;

  AstNode(Span span, AstType ty,
          std::initializer_list<std::shared_ptr<AstNode>> children);
};
