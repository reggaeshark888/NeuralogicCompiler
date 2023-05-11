# Generated from Logic.g4 by ANTLR 4.12.0
from antlr4 import *
if __name__ is not None and "." in __name__:
    from .LogicParser import LogicParser
else:
    from LogicParser import LogicParser

# This class defines a complete generic visitor for a parse tree produced by LogicParser.

class LogicVisitor(ParseTreeVisitor):

    # Visit a parse tree produced by LogicParser#parse.
    def visitParse(self, ctx:LogicParser.ParseContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by LogicParser#binaryExpression.
    def visitBinaryExpression(self, ctx:LogicParser.BinaryExpressionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by LogicParser#boolExpression.
    def visitBoolExpression(self, ctx:LogicParser.BoolExpressionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by LogicParser#identifierExpression.
    def visitIdentifierExpression(self, ctx:LogicParser.IdentifierExpressionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by LogicParser#notExpression.
    def visitNotExpression(self, ctx:LogicParser.NotExpressionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by LogicParser#parenExpression.
    def visitParenExpression(self, ctx:LogicParser.ParenExpressionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by LogicParser#binary.
    def visitBinary(self, ctx:LogicParser.BinaryContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by LogicParser#bool.
    def visitBool(self, ctx:LogicParser.BoolContext):
        return self.visitChildren(ctx)



del LogicParser