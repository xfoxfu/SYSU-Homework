# 语义分析程序及中间代码生成实验

17341039 傅禹泽

## 实验目的

构造 TINY+ 的语义分析程序并生成中间代码

## 实验内容

构造符号表，用 C 语言扩展 TINY 的语义分析程序，构造 TINY+ 的语义分析器，构造 TINY+ 的中间代码生成器

## 实验要求

能检查一定的语义错误，将 TINY+ 程序转换成三地址中间代码

## 实验过程

### 扩展 TINY 语法

我基于 [University of Windsor 资料](http://jlu.myweb.cs.uwindsor.ca/214/language.htm)中提出的 TINY 语法，进行了几点改进：

- 将程序 Program 扩充成函数 Func 和变量定义 Vars 的克林闭包。这样 TINY+ 语言允许定义全局变量

      Program -> (Func | Vars)*

- 允许 2 进制（`0b` 开头）、8 进制（`0o` 开头）、16 进制（`0x` 开头）的数字

      Number    -> DecNumber | OctNumber | HexNumber
      BinNumber -> '0b' BinDigit (BinDigit | '_')+
      DecNumber -> Digit (Digit | '_')+
      OctNumber -> '0o' OctDigit (OctDigit | '_')+
      HexNumber -> '0x' HexDigit (HexDigit | '_')+

- 允许数字中出现下划线分割数位

      Digit | '_'

- 允许函数的参数按引用传递

      Param -> [ 'REF' ] Type Ident

- 增加了大量的运算符

      Expr0        -> '(' Expression ')' | Call | Number | Ident | String
      Expr1        -> [('+' | '-' | '!')] Expr0
      Expr2        -> Expr1 (('*' | '/') Expr1)*
      Expr3        -> Expr2 (('+' | '-') Expr2)*
      Expr4        -> Expr3 (('>' | '<' | '>=' | '<=') Expr3)*
      Expr5        -> Expr4 (('==' | '!=') Expr4)*
      Expr6        -> Expr5 (('&&' | '||') Expr5)*
      Expression   -> Expr6

### 词法分析

该语言的词法可以通过 LL(1) 来实现，而不必通过状态机完成。

关键字有 `BEGIN`、`ELSE`、`END`、`IF`、`INT`、`MAIN`、`REAL`、`REF`、`RETURN`、`READ`、`STRING`、`WRITE`。

符号有 `_`、`-`、`,`、`;`、`:=`、`!=`、`!`、`(`、`)`、`*`、`/`、`&&`、`+`、`<=`、`<`、`==`、`>=`、`>`、`||`。

另外定义 `Ident`、`Number`、`String` 三种 Token 如下：

```python
Alpha        -> 'a' | 'b' | ... | 'z' | 'A' | 'B' | ... | 'Z'
BinDigit     -> '0' | '1'
OctDigit     -> BinDigit | '2' | ... | '7'
Digit        -> OctDigit | '8' | '9'
HexDigit     -> Digit | 'a' | 'b' | ... | 'f' | 'A' | 'B' | ... | 'F'

Ident        -> (Alpha | '_') (Alpha | Digit | '_')*
Number       -> DecNumber | OctNumber | HexNumber
BinNumber    -> '0b' BinDigit (BinDigit | '_')+
DecNumber    -> Digit (Digit | '_')+
OctNumber    -> '0o' OctDigit (OctDigit | '_')+
HexNumber    -> '0x' HexDigit (HexDigit | '_')+
String       -> '"' Char* '"'
```

在词法分析阶段，允许出现多个词法错误，并且对错误尝试进行恢复，从而继续解析下一个 Token。

```cpp
void Lexer::parse() {
  vector<Error> errors;
  do {
    try {
      auto token = advance();
      if (!token.is(TokenType::Void))
        _tokens.push_back(token);
      else if (!finished()) {
        auto chr = progress();
        assert(_current - 1 >= _begin);
        errors.push_back(Error(Span(cstr_begin(), cstr_end()),
                               Span(_current - 1, _current),
                               string("Unexpected token: '") + chr + "'"));
      }
    } catch (Error e) {
      errors.push_back(e);
    }
  } while (!finished());
  if (!errors.empty())
    throw errors;
}
```

此外，在词法分析阶段去除代码中的空白与注释。

```cpp
void Lexer::whitespace() {
  while (match(match_space) != '\0' || *_current == '/') {
    // handle comments
    if (match('/', '*') == true) {
      while (match([](char c, char n) { return c != '*' && n != '/'; }).first != '\0') {
      }
      if (match('*', '/') == false) {
        throw Error(Span(cstr_begin(), cstr_end()),
                    Span(_current - 1, _current),
                    string("Expecting close of comment: '}'"));
      }
    }
  }
}
```

因为没有使用状态机，因此较难区分标识符与关键词。这里采用全部作为标识符解析，再判断是否属于关键词的方法。

```cpp
Token Lexer::Ident() {
  auto begin = _current;
  FAIL_IF(match(match_alpha_lodash) == '\0');
  while (match(match_alpha_digit_lodash) != '\0') {
  }
  FAIL_IF(begin == _current);

  TokenType ty = TokenType::Ident;
  std::string value(begin, _current);
  if (value == "BEGIN" || value == "ELSE" || value == "END" || value == "IF" ||
      value == "INT" || value == "MAIN" || value == "REAL" || value == "REF" ||
      value == "RETURN" || value == "READ" || value == "STRING" ||
      value == "WRITE") {
    ty = TokenType::Keyword;
  }
  return Token(ty, begin, _current);
}
```

### 语法分析

语法分析采用 LL(2) 语法分析算法。语法定义为：

```python
Program      -> (Func | Vars)*
Func         -> Type [ 'MAIN' ] Ident '(' Params ')' BlocParams       -> [Param (',' Param)* ]
Param        -> [ 'REF' ] Type Ident
Type         -> 'INT' | 'REAL' | 'STRING'
Vars         -> Type Ident [ ':=' Expression ] ';'
Block        -> 'BEGIN' Statement* 'END'
Statement    -> Block | Vars | Assignment | ReturnStmt | IfStmt | ExprStmt
Assignment   -> Ident ':=' Expression ';'
IfStmt       -> 'IF' '(' Expression ')' Statement [ 'ELSE' Statement ]
ReturnStmt   -> 'RETURN' Expression ';'
ExprStmt     -> Expression ';'
Arguments    -> Expression (',' Expression)*
Call         -> Ident '(' [Arguments] ')'
Expr0        -> '(' Expression ')' | Call | Number | Ident | String
Expr1        -> [('+' | '-' | '!')] Expr0
Expr2        -> Expr1 (('*' | '/') Expr1)*
Expr3        -> Expr2 (('+' | '-') Expr2)*
Expr4        -> Expr3 (('>' | '<' | '>=' | '<=') Expr3)*
Expr5        -> Expr4 (('==' | '!=') Expr4)*
Expr6        -> Expr5 (('&&' | '||') Expr5)*
Expression   -> Expr6
```

具体实现上基本全部参照 LL(2) 算法进行，不再赘述。

## 实验结果

对于以下 TINY+ 程序：

```tiny
/* this is a comment line in the sample program */
INT f2(INT x, INT y )
BEGIN
    INT z;
    z := x*x - y*y;
    RETURN z;
END
INT MAIN f1()
BEGIN
    INT x;
    READ(x, "A41.input");
    INT y;
    READ(y, "A42.input");
    INT z;
    z := f2(x,y) + f2(y,x);
    WRITE (z, "A4.output");
END
```

程序进行词法分析后取得：

```console
Keyword: (INT)
Ident: (f2)
Punct: (()
Keyword: (INT)
Ident: (x)
Punct: (,)
Keyword: (INT)
Ident: (y)
Punct: ())
Keyword: (BEGIN)
Keyword: (INT)
Ident: (z)
Punct: (;)
Ident: (z)
Punct: (:=)
Ident: (x)
Punct: (*)
Ident: (x)
Punct: (-)
Ident: (y)
Punct: (*)
Ident: (y)
Punct: (;)
Keyword: (RETURN)
Ident: (z)
Punct: (;)
Keyword: (END)
Keyword: (INT)
Keyword: (MAIN)
Ident: (f1)
Punct: (()
Punct: ())
Keyword: (BEGIN)
Keyword: (INT)
Ident: (x)
Punct: (;)
Keyword: (READ)
Punct: (()
Ident: (x)
Punct: (,)
String: (A41.input)
Punct: ())
Punct: (;)
Keyword: (INT)
Ident: (y)
Punct: (;)
Keyword: (READ)
Punct: (()
Ident: (y)
Punct: (,)
String: (A42.input)
Punct: ())
Punct: (;)
Keyword: (INT)
Ident: (z)
Punct: (;)
Ident: (z)
Punct: (:=)
Ident: (f2)
Punct: (()
Ident: (x)
Punct: (,)
Ident: (y)
Punct: ())
Punct: (+)
Ident: (f2)
Punct: (()
Ident: (y)
Punct: (,)
Ident: (x)
Punct: ())
Punct: (;)
Keyword: (WRITE)
Punct: (()
Ident: (z)
Punct: (,)
String: (A4.output)
Punct: ())
Punct: (;)
Keyword: (END)
```

程序进行语法分析后取得：

```console
- Program
  - Func
    - Keyword: (INT)
    - Ident: (f2)
    - Params
      - Param
        - Keyword: (INT)
        - Ident: (x)
      - Param
        - Keyword: (INT)
        - Ident: (y)
    - Block
      - Vars
        - Keyword: (INT)
        - Ident: (z)
      - Assignment
        - Ident: (z)
        - Expr
          - Expr
            - Variable
              - Ident: (x)
            - Punct: (*)
            - Variable
              - Ident: (x)
          - Punct: (-)
          - Expr
            - Variable
              - Ident: (y)
            - Punct: (*)
            - Variable
              - Ident: (y)
          - Punct: (+)
          - Number: (123)
      - ReturnStmt
        - Variable
          - Ident: (z)
  - Func
    - Keyword: (INT)
    - Keyword: (MAIN)
    - Ident: (f1)
    - Params
      - Param
        - Keyword: (INT)
        - Ident: (a0)
      - Param
        - Keyword: (REAL)
        - Ident: (a2)
    - Block
      - Vars
        - Keyword: (INT)
        - Ident: (x)
        - Number: (10)
      - Call
        - Keyword: (READ)
        - Arguments
          - Variable
            - Ident: (x)
          - String: (A41.input)
      - Vars
        - Keyword: (INT)
        - Ident: (y)
      - Call
        - Keyword: (READ)
        - Arguments
          - Variable
            - Ident: (y)
          - String: (A42.input)
      - Vars
        - Keyword: (INT)
        - Ident: (z)
      - Assignment
        - Ident: (z)
        - Expr
          - Call
            - Ident: (f2)
            - Arguments
              - Variable
                - Ident: (x)
              - Variable
                - Ident: (y)
          - Punct: (+)
          - Call
            - Ident: (f2)
            - Arguments
              - Variable
                - Ident: (y)
              - Variable
                - Ident: (x)
      - Call
        - Keyword: (WRITE)
        - Arguments
          - Variable
            - Ident: (z)
          - String: (A4.output)
```

当遇到词法错误或语法错误时，程序能够指示错误的位置并给出提示：

```console
INPUT:113:     z := x*x - y*y + 0X123R;
                                ^
Error: Unexpected token: '0'
```

```console
INPUT:109:     int x;
                   ^
Error: Expected token of: Punct with value ;, found Ident: (x)
```

```console
INPUT:95: INT MAIN f1(REF)
                         ^
Error: Expected Type INT, REAL or STRING.
```

```console
INPUT:241:     WRITE (z, "A4.output");
                                      ^
Error: Expected token of: Keyword with value END, found Void: ()
```
