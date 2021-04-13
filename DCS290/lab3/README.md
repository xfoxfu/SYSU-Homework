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

Program      -> (Func | Vars)*
Func         -> Type [ 'MAIN' ] Ident '(' FormalParams ')' Block
FormalParams -> [FormalParam (',' FormalParam)* ]
FormalParam  -> [ 'REF' ] Type Ident
Type         -> 'INT' | 'REAL'
Vars         -> Type Ident [ ':=' Expression ] ';'
Block        -> 'BEGIN' Statement* 'END'
Statement    -> Block | Vars | Assignment | ReturnStmt | IfStmt
Assignment   -> Ident ':=' Expression ';'
IfStmt       -> 'IF' '(' Expression ')' Statement [ 'ELSE' Statement ]
ReturnStmt   -> 'RETURN' Expression ';'
ActualParams -> [Expression (',' Expression)*]
Call         -> Ident '(' ActualParams ')'
Unit0        -> '(' Expression ')' | Call | Number | Ident | String | Char
Unit1        -> [('+' | '-' | '!')] Unit0
Unit2        -> Unit1 (('*' | '/') Unit1)*
Unit3        -> Unit2 (('+' | '-') Unit2)*
Unit4        -> Unit3 (('>' | '<' | '>=' | '<=') Unit3)*
Unit5        -> Unit4 (('==' | '!=') Unit4)*
Unit6        -> Unit5 (('&&' | '||') Unit4)*
Expression   -> Unit6
```
