#include "ast.hpp"

using namespace std;

AstNode::AstNode(Span span, string value) : HasSpan(span), value(value) {}

AstNode::AstNode(Span span, string value, vector<shared_ptr<AstNode>> children)
    : HasSpan(span), value(value), children(children) {}

AstNode::AstNode(Token token) : HasSpan(token), value(token.to_string()) {}
