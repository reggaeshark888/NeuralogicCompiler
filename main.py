import sys
from antlr4 import *
from LogicLexer import LogicLexer
from LogicParser import LogicParser
from MyVisiotor import MyVisitor


if __name__ == '__main__':
    input_stream = FileStream('input1.txt')
    lexer = LogicLexer(input_stream)
    stream = CommonTokenStream(lexer)
    parser = LogicParser(stream)
    tree = parser.parse()

    visitor = MyVisitor()
    visitor.visit(tree)
