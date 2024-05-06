class LogicExpression():
    def __init__(self, name):
        self.name = name
        self.level = None
        self.matrix_weigts = []
    def __or__(self, other):
        """Overload operator to represent logical OR."""
        self.matrix_weigts.append([1, 1, -1.5])
        return LogicExpression('x')
    def __and__(self, other):
        """Overload operator to represent logical AND."""
        self.matrix_weigts.append([-1, -1, 1.5])
        return LogicExpression('x')
    def __xor__(self, other):
        """Overload operator to represent logical XOR."""
        self.matrix_weigts.append([1, 1, -0.5])
        return LogicExpression('x')


a = LogicExpression('a')
b = LogicExpression('b')
c = LogicExpression('c')
d = LogicExpression('d')
result = (a & b | a) & (a | (c | d))
print(result)