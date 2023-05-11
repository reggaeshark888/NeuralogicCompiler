# Generated from Logic.g4 by ANTLR 4.12.0
from antlr4 import *
from io import StringIO
import sys
if sys.version_info[1] > 5:
    from typing import TextIO
else:
    from typing.io import TextIO


def serializedATN():
    return [
        4,0,10,61,6,-1,2,0,7,0,2,1,7,1,2,2,7,2,2,3,7,3,2,4,7,4,2,5,7,5,2,
        6,7,6,2,7,7,7,2,8,7,8,2,9,7,9,1,0,1,0,1,0,1,0,1,0,1,1,1,1,1,1,1,
        1,1,1,1,1,1,2,1,2,1,2,1,2,1,3,1,3,1,3,1,4,1,4,1,4,1,4,1,4,1,5,1,
        5,1,5,1,5,1,6,1,6,1,7,1,7,1,8,1,8,1,9,4,9,56,8,9,11,9,12,9,57,1,
        9,1,9,0,0,10,1,1,3,2,5,3,7,4,9,5,11,6,13,7,15,8,17,9,19,10,1,0,2,
        1,0,97,122,3,0,9,10,13,13,32,32,61,0,1,1,0,0,0,0,3,1,0,0,0,0,5,1,
        0,0,0,0,7,1,0,0,0,0,9,1,0,0,0,0,11,1,0,0,0,0,13,1,0,0,0,0,15,1,0,
        0,0,0,17,1,0,0,0,0,19,1,0,0,0,1,21,1,0,0,0,3,26,1,0,0,0,5,32,1,0,
        0,0,7,36,1,0,0,0,9,39,1,0,0,0,11,44,1,0,0,0,13,48,1,0,0,0,15,50,
        1,0,0,0,17,52,1,0,0,0,19,55,1,0,0,0,21,22,5,116,0,0,22,23,5,114,
        0,0,23,24,5,117,0,0,24,25,5,101,0,0,25,2,1,0,0,0,26,27,5,102,0,0,
        27,28,5,97,0,0,28,29,5,108,0,0,29,30,5,115,0,0,30,31,5,101,0,0,31,
        4,1,0,0,0,32,33,5,97,0,0,33,34,5,110,0,0,34,35,5,100,0,0,35,6,1,
        0,0,0,36,37,5,111,0,0,37,38,5,114,0,0,38,8,1,0,0,0,39,40,5,110,0,
        0,40,41,5,97,0,0,41,42,5,110,0,0,42,43,5,100,0,0,43,10,1,0,0,0,44,
        45,5,110,0,0,45,46,5,111,0,0,46,47,5,116,0,0,47,12,1,0,0,0,48,49,
        7,0,0,0,49,14,1,0,0,0,50,51,5,40,0,0,51,16,1,0,0,0,52,53,5,41,0,
        0,53,18,1,0,0,0,54,56,7,1,0,0,55,54,1,0,0,0,56,57,1,0,0,0,57,55,
        1,0,0,0,57,58,1,0,0,0,58,59,1,0,0,0,59,60,6,9,0,0,60,20,1,0,0,0,
        2,0,57,1,6,0,0
    ]

class LogicLexer(Lexer):

    atn = ATNDeserializer().deserialize(serializedATN())

    decisionsToDFA = [ DFA(ds, i) for i, ds in enumerate(atn.decisionToState) ]

    TRUE = 1
    FALSE = 2
    AND = 3
    OR = 4
    NAND = 5
    NOT = 6
    IDENTIFIER = 7
    LPAREN = 8
    RPAREN = 9
    WS = 10

    channelNames = [ u"DEFAULT_TOKEN_CHANNEL", u"HIDDEN" ]

    modeNames = [ "DEFAULT_MODE" ]

    literalNames = [ "<INVALID>",
            "'true'", "'false'", "'and'", "'or'", "'nand'", "'not'", "'('", 
            "')'" ]

    symbolicNames = [ "<INVALID>",
            "TRUE", "FALSE", "AND", "OR", "NAND", "NOT", "IDENTIFIER", "LPAREN", 
            "RPAREN", "WS" ]

    ruleNames = [ "TRUE", "FALSE", "AND", "OR", "NAND", "NOT", "IDENTIFIER", 
                  "LPAREN", "RPAREN", "WS" ]

    grammarFileName = "Logic.g4"

    def __init__(self, input=None, output:TextIO = sys.stdout):
        super().__init__(input, output)
        self.checkVersion("4.12.0")
        self._interp = LexerATNSimulator(self, self.atn, self.decisionsToDFA, PredictionContextCache())
        self._actions = None
        self._predicates = None


