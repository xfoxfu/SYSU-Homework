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

- 增加了大量的运算符

      Expr0        -> '(' Expression ')' | Call | Number | Ident | String
      Expr1        -> [('+' | '-' | '!')] Expr0
      Expr2        -> Expr1 (('*' | '/') Expr1)*
      Expr3        -> Expr2 (('+' | '-') Expr2)*
      Expr4        -> Expr3 (('>' | '<' | '>=' | '<=') Expr3)*
      Expr5        -> Expr4 (('==' | '!=') Expr4)*
      Expr6        -> Expr5 (('&&' | '||') Expr5)*
      Expression   -> Expr6

### 语义分析

我在编译器中实现了类型检查和未定义符号检查。对于类型检查，编译器会尝试推导各个表达式、符号的类型，并且存储为一个 `vector<string>`。函数类型为 `[T0, T1, ..., R]`，而一般的表达式、变量类型为 `[T]`。

#### 符号表

在类型检查阶段，编译器会自顶向下地通过 `get_type` 函数求得语法树中各个成分的类型，同时追踪当前已经定义的符号（`symbol_table`）。考虑到程序中存在语句块，而内层变量离开作用域后就不应该能够被继续访问，所以符号除了名称、类型以外，还要有定义的作用域的层级。

```cpp
struct SemanticSymbol {
  std::string name;
  size_t layer;
  std::vector<std::string> type;
};
```

当编译器进入一个语句块时，会增加当前层级号；而离开语句块时，会移除该层级下的所有符号。

```cpp
void SemanticVisitor::enter_layer() { current_layer += 1; }

void SemanticVisitor::exit_layer() {
  current_layer -= 1;
  while (symbol_table.size() > 0 && symbol_table.back().layer > current_layer) {
    symbol_table.pop_back();
  }
}
```

增加符号时，会纪录其的类型和层号。

```cpp
void SemanticVisitor::add_symbol(std::string name,
                                 std::vector<std::string> type, size_t layer) {
  symbol_table.push_back(SemanticSymbol(name, layer, type));
  std::sort(symbol_table.begin(), symbol_table.end(),
            SemanticSymbolComparator());
}
```

#### 类型推导规则

对于函数、变量，其类型在代码中显式写出。TINY+ 语言不允许隐式类型转换。

因为语法中只定义了整形字面量和字符串字面量，因此它们的类型分别为 `INT` 和 `STRING`。

对于变量赋值，要求表达式的类型和变量的类型一致。

对于条件分支语句，要求条件的类型必须为 `INT`。其中，`0` 代表假值，非 0 代表真值。

对于 `RETURN` 语句，要求其的类型与函数返回值类型一致。

对于函数调用，要求其各个参数的类型与函数签名一致，返回值的类型由函数签名决定。

对于表达式，单目的 `+` 和 `-` 表达式的右操作数必须为 `INT` 或者 `REAL`，返回值类型与右操作数一致。单目 `!` 运算符代表逻辑取反，右操作数必须为 `INT`，返回值类型与右操作数一致。

双目的 `+`、`-`、`*`、`/` 运算符的两个操作数类型一致，并且必须为 `INT` 或 `REAL`，返回值类型与操作数一致。

双目的 `<`、`<=`、`>`、`>=`、`==`、`!=` 运算符的两个操作数类型一致，并且必须为 `INT` 或 `REAL`，返回值类型为 `INT`。

双目的 `&&`、`||` 运算符的两个操作数类型一致，并且必须为 `INT`，返回值类型为 `INT`。

在类型检查中，我纪录了出现过的所有错误，并且在发生错误时使得有错误的语法树节点的类型为 `VOID` 来进行错误恢复。

这一部分实现为 `vector<string> SemanticVisitor::get_type(const AstNode &node);`。

### 中间代码生成

我采用了类似 LLVM IR 的中间代码表示形式。

例如，函数可以表示为：

```llvm-ir
define i32 @f2(i32 %x, i32 %y) {
}
```

一般的运算赋值可以表示为：

```llvm-ir
%2 := == %0 %1
```

本地符号用 `%i` 表示，包括寄存器、栈变量、标签。栈变量通过：

```llvm-ir
%z = alloca i32
```

来分配。

另外，标签表示为：

```llvm-ir
L0:
```

并且可以通过 `br` 来进行条件跳转：

```llvm-ir
br ir %2, label %L0, label %1
```

我将 TINY+ 语言的`INT` 映射为 LLVM IR 的 `i32`，`REAL` 映射为 `f64`，`STRING` 映射为 `string`。

生成中间代码也是一个自顶向下的过程，对于每一个语法树节点，我们调用其子节点的中间代码生成过程，从而获得其子节点的运算结果，再根据子节点的调用结果来生成当前节点的中间代码。

LLVM IR 的寄存器个数是无限的。我们记录当前的寄存器编号（分配到第几个），从而来获得新的寄存器存储结果。我们约定子节点的运算结果一定存储在最后一个分配的寄存器上，从而简化代码。

这一部分实现为 `string SemanticVisitor::emit_ir(const AstNode &node);`。该函数接受一个语法树节点，返回其对应的 IR。

## 实验结果

对于这样一个 TINY+ 代码：

```tiny
/* this is a comment line in the sample program */
INT f2(INT x, INT y )
BEGIN
    INT z;
    IF (z == 10) BEGIN
        z := 30;
    END ELSE BEGIN
        z := 100;
    END
    z := x*x - y*y + 0x123;
    RETURN z;
END
INT MAIN f1_main(INT a0, REAL a2)
BEGIN
    INT x := 10;
    READ(x, "A41.input");
    INT y;
    READ(y, "A42.input");
    INT z;
    z := f2(x,y) + f2(y,x);
    z := f2(x,y) + f2(y,100);
    WRITE (z, "A4.output");
END
```

### 语义分析

程序能够判断出其符合类型规则，并且没有使用未定义的变量。

如果我们将 `z := x*x - y*y + 0x123;` 修改为 `z := x*x - y*y + unknown;`，就使用了未定义的变量 `unknown`，语义分析会报告错误：

```console
INPUT:147:     z := x*x - y*y + unknown;
                                ^^^^^^^
Error: assign to undefined variable Ident: (unknown)
INPUT:135:     z := x*x - y*y + unknown;
                    ^^^^^^^^^
Error: expected INT or REAL, found VOID
INPUT:135:     z := x*x - y*y + unknown;
                    ^^^^^^^^^^^^^^^^^^^
Error: lhs and rhs type mismatch (INT vs VOID)
```

如果我们将 `f2` 的返回值类型修改为 `REAL`，那么表达式的结果就和变量的类型不一致，同时返回语句的类型也和函数签名不一致，会报告错误：

```console
INPUT:0: REAL f2(INT x, INT y )
Error: function return type mismatch, expected REAL, found INT
INPUT:315:     z := f2(x,y) + f2(y,x);
                    ^^^^^^^^^^^^^^^^^
Error: assign to variable of INT with expression of REAL
INPUT:343:     z := f2(x,y) + f2(y,100);
                    ^^^^^^^^^^^^^^^^^^^
Error: assign to variable of INT with expression of REAL
```

如此，不一一列举。

分析过程中，符号表的变化显示为：

```console
Added      Symbol [1] Ident: (x): INT
Added      Symbol [1] Ident: (y): INT
Added      Symbol [2] Ident: (z): INT
Eliminated Symbol [2] Ident: (z): INT
Added      Symbol [0] Ident: (f2): INT->INT->INT
Eliminated Symbol [1] Ident: (y): INT
Eliminated Symbol [1] Ident: (x): INT
Added      Symbol [1] Ident: (a0): INT
Added      Symbol [1] Ident: (a2): REAL
Added      Symbol [2] Ident: (x): INT
Added      Symbol [2] Ident: (y): INT
Added      Symbol [2] Ident: (z): INT
Eliminated Symbol [2] Ident: (z): INT
Eliminated Symbol [2] Ident: (y): INT
Eliminated Symbol [2] Ident: (x): INT
Added      Symbol [0] Ident: (f1_main): INT->REAL->INT
Eliminated Symbol [1] Ident: (a2): REAL
Eliminated Symbol [1] Ident: (a0): INT
```

### 中间代码生成

程序生成中间代码：

```console
source_filename = "*.tiny"
define i32 @f2(i32 %x, i32 %y) {
  %z = alloca i32
  %0 := %z
  %1 := 10
  %2 := == %0 %1
  br ir %2, label %L0, label %1
L0:
  %3 := 30
  %z = %3
  goto %L2
L1:
  %4 := 100
  %z = %4
L2:
  %5 := %x
  %6 := %x
  %7 := * %5 %6
  %8 := %y
  %9 := %y
  %10 := * %8 %9
  %11 := - %7 %10
  %12 := 123
  %13 := + %11 %12
  %z = %13
  %14 := %z
  return i32 %14
}

define i32 @f1_main(i32 %a0, f64 %a2) {
  %x = alloca i32
  %0 := 10
  %x := %0
  %1 := %x
  %2 := A41.input
  %3 := call void @READ (i32 %1, string %2)
  %y = alloca i32
  %4 := %y
  %5 := A42.input
  %6 := call void @READ (i32 %4, string %5)
  %z = alloca i32
  %7 := %x
  %8 := %y
  %9 := call i32 @f2 (i32 %7, i32 %8)
  %10 := %y
  %11 := %x
  %12 := call i32 @f2 (i32 %10, i32 %11)
  %13 := + %9 %12
  %z = %13
  %14 := %x
  %15 := %y
  %16 := call i32 @f2 (i32 %14, i32 %15)
  %17 := %y
  %18 := 100
  %19 := call i32 @f2 (i32 %17, i32 %18)
  %20 := + %16 %19
  %z = %20
  %21 := %z
  %22 := A4.output
  %23 := call void @WRITE (i32 %21, string %22)
}
```

具体列举，对于 `z := x*x - y*y + 0x123;` 其生成：

```console
%5 := %x
%6 := %x
%7 := * %5 %6
%8 := %y
%9 := %y
%10 := * %8 %9
%11 := - %7 %10
%12 := 123
%13 := + %11 %12
%z = %13
```

是该表达式的三地址代码。

对于函数调用 `f2(x,y)`，其生成：

```console
%14 := %x
%15 := %y
%16 := call i32 @f2 (i32 %14, i32 %15)
```

对于条件分支结构

```console
IF (z == 10) BEGIN
    z := 30;
END ELSE BEGIN
    z := 100;
END
```

其也能够生成正确的跳转和标签结构：

```console
  %0 := %z
  %1 := 10
  %2 := == %0 %1
  br ir %2, label %L0, label %1
L0:
  %3 := 30
  %z = %3
  goto %L2
L1:
  %4 := 100
  %z = %4
L2:
```

综上，本次实验完成了预定的实验要求，达到的期望的结果。
