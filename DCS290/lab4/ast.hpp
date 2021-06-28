#pragma once

#include "span.hpp"
#include "token.hpp"
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

struct AstNode : public HasSpan {
  std::string value;
  std::vector<std::shared_ptr<AstNode>> children;

  AstNode(Span span, std::string value);
  AstNode(Span span, std::string value,
          std::vector<std::shared_ptr<AstNode>> children);
  AstNode(Token token);
  Span &span();
};
