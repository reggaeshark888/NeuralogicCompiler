grammar Logic;
// Parser Rules
parse
 : expression EOF
 ;

expression
 : LPAREN expression RPAREN                       #parenExpression
 | NOT expression                                 #notExpression
 | left=expression op=binary right=expression     #binaryExpression
 | bool                                           #boolExpression
 | IDENTIFIER                                     #identifierExpression
 ;

binary
 : AND | OR | NAND
 ;

bool
 : TRUE | FALSE
 ;

// Lexer Rules
// logical constants
TRUE: 'true';
FALSE: 'false';

// logical connectives
AND : 'and' ;
OR  : 'or' ;
NAND : 'nand';
NOT : 'not';

IDENTIFIER : [a-z];

LPAREN: '(';
RPAREN: ')';

WS  : [ \t\r\n]+ -> skip ;
