grammar logic;
// Parser Rules
parse
 : expression EOF
 ;

expression
 : LPAREN expression RPAREN                       #parenExpression
 | NOT expression                                 #notExpression
 | left=expression op=comparator right=expression #comparatorExpression
 | left=expression op=binary right=expression     #binaryExpression
 | bool                                           #boolExpression
 | IDENTIFIER                                     #identifierExpression
 ;

binary
 : AND | OR
 ;

bool
 : TRUE | FALSE
 ;

// Lexer Rules
WORD  : [a-z] ;

// logical constants
TRUE: 'true';
FALSE: 'false';

// logical connectives
AND : 'and' ;
OR  : 'or' ;
NOT : 'not';

IDENTIFIER : [a-z];

LPAREN: '(';
RPAREN: ')';

WS  : [ \t\r\n]+ -> skip ;