from sources.experiments.calc_central_charge2 import SimpleExpressionEvaluator
from sources.pdesolver.formula_parser.parser_expression import NumberExpression, UnaryOperatorExpression, \
    BinaryOperatorExpression


class FunctionCall:
    def __init__(self, offset):
        self.offset = offset
        self.prefactorExpr = NumberExpression(1.0)
        self.simpleExpression = True

    def get_offset(self):
        return self.offset

    def get_prefactor(self):
        return self.prefactorExpr

    def negate(self):
        self.prefactorExpr = UnaryOperatorExpression(self.prefactorExpr, '-')

    def multiplyPrefactor(self, expr):
        self.prefactorExpr = BinaryOperatorExpression(expr, self.prefactorExpr, '*')
        self.simpleExpression = False

    def __str__(self):
        return '({0})u({1})'.format(str(self.prefactorExpr), self.offset)

    def __repr__(self):
        return '({0})u({1})'.format(repr(self.prefactorExpr), self.offset)

    def add(self, expr):
        self.prefactorExpr = BinaryOperatorExpression(self.prefactorExpr, expr, '+')

    def makeFunction(self, functionDict):
        def evaluatorFn(col, row):
            evaluator = SimpleExpressionEvaluator(variables={'i':col, 'j':row}, functions=functionDict)
            self.prefactorExpr.accept(evaluator)
            return evaluator.get_result()

        return evaluatorFn
