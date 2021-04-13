#include "ast.hpp"

using namespace std;

AstNode::AstNode(Span span, AstType ty,
                 std::initializer_list<std::shared_ptr<AstNode>> children)
    : HasSpan(span), ty(ty), children(children) {}
