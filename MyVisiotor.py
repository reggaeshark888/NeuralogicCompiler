from LogicVisitor import LogicVisitor
from LogicParser import LogicParser

class MyVisitor(LogicVisitor):
    def __init__(self):
        self.weight_matrices = {}
        self.neural_net_weights = []

    def visitParse(self, ctx:LogicParser.ParseContext):
        self.visit(ctx.expression())

        # reverse the matrix 
        for _, layer_weights in self.weight_matrices.items():
            self.neural_net_weights.append(layer_weights)

        self.neural_net_weights.reverse()
        print(self.neural_net_weights)

    def visitBinaryExpression(self, ctx:LogicParser.BinaryExpressionContext):
        if ctx.binary().getText() == "and":
            if ctx.depth() not in self.weight_matrices:
                self.weight_matrices[ctx.depth()] = []
                self.weight_matrices[ctx.depth()].append([1, 1, -1.5])
            else:
                self.weight_matrices[ctx.depth()].append([1, 1, -1.5])
        if ctx.binary().getText() == "nand":
            if ctx.depth() not in self.weight_matrices:
                self.weight_matrices[ctx.depth()] = []
                self.weight_matrices[ctx.depth()].append([-1, -1, 1.5])
            else:
                self.weight_matrices[ctx.depth()].append([-1, -1, 1.5])
        if ctx.binary().getText() == "or":
            if ctx.depth() not in self.weight_matrices:
                self.weight_matrices[ctx.depth()] = []
                self.weight_matrices[ctx.depth()].append([1, 1, -0.5])
            else:
                self.weight_matrices[ctx.depth()].append([1, 1, -0.5])

        return self.visitChildren(ctx)