# Generated from Logic.g4 by ANTLR 4.12.0
from antlr4 import *
if __name__ is not None and "." in __name__:
    from .LogicParser import LogicParser
else:
    from LogicParser import LogicParser

# This class defines a complete listener for a parse tree produced by LogicParser.
class LogicListener(ParseTreeListener):

    # Enter a parse tree produced by LogicParser#parse.
    def enterParse(self, ctx:LogicParser.ParseContext):
        pass

    # Exit a parse tree produced by LogicParser#parse.
    def exitParse(self, ctx:LogicParser.ParseContext):
        pass


    # Enter a parse tree produced by LogicParser#binaryExpression.
    def enterBinaryExpression(self, ctx:LogicParser.BinaryExpressionContext):
        pass

    # Exit a parse tree produced by LogicParser#binaryExpression.
    def exitBinaryExpression(self, ctx:LogicParser.BinaryExpressionContext):
        pass


    # Enter a parse tree produced by LogicParser#boolExpression.
    def enterBoolExpression(self, ctx:LogicParser.BoolExpressionContext):
        pass

    # Exit a parse tree produced by LogicParser#boolExpression.
    def exitBoolExpression(self, ctx:LogicParser.BoolExpressionContext):
        pass


    # Enter a parse tree produced by LogicParser#identifierExpression.
    def enterIdentifierExpression(self, ctx:LogicParser.IdentifierExpressionContext):
        pass

    # Exit a parse tree produced by LogicParser#identifierExpression.
    def exitIdentifierExpression(self, ctx:LogicParser.IdentifierExpressionContext):
        pass


    # Enter a parse tree produced by LogicParser#notExpression.
    def enterNotExpression(self, ctx:LogicParser.NotExpressionContext):
        pass

    # Exit a parse tree produced by LogicParser#notExpression.
    def exitNotExpression(self, ctx:LogicParser.NotExpressionContext):
        pass


    # Enter a parse tree produced by LogicParser#parenExpression.
    def enterParenExpression(self, ctx:LogicParser.ParenExpressionContext):
        pass

    # Exit a parse tree produced by LogicParser#parenExpression.
    def exitParenExpression(self, ctx:LogicParser.ParenExpressionContext):
        pass


    # Enter a parse tree produced by LogicParser#binary.
    def enterBinary(self, ctx:LogicParser.BinaryContext):
        pass

    # Exit a parse tree produced by LogicParser#binary.
    def exitBinary(self, ctx:LogicParser.BinaryContext):
        pass


    # Enter a parse tree produced by LogicParser#bool.
    def enterBool(self, ctx:LogicParser.BoolContext):
        pass

    # Exit a parse tree produced by LogicParser#bool.
    def exitBool(self, ctx:LogicParser.BoolContext):
        pass



del LogicParser