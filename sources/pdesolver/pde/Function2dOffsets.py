from sources.pdesolver.pde.SimpleExpressionEvaluator import SimpleExpressionEvaluator


class Function2dOffsets:
    def __init__(self, parameter_expr_list):
        assert len(parameter_expr_list)==2
        evaluator_x = self.evaluateParameterExpression(parameter_expr_list[0], 'i')
        evaluator_y = self.evaluateParameterExpression(parameter_expr_list[1], 'j')
        self.x = evaluator_x
        self.y = evaluator_y

    def evaluateParameterExpression(self, parameter_expr, positional_variable_name):
        variables = {positional_variable_name:0.0}
        evaluator = SimpleExpressionEvaluator(variables)
        parameter_expr.accept(evaluator)
        return evaluator.get_result()

    def __eq__(self, another):
        return self.x == another.x and self.y == another.y

    def __hash__(self):
        return hash(self.x) * hash(self.y)

    def __repr__(self):
        return 'i+{0} j+{1}'.format(self.x, self.y)

    def __str__(self):
        return 'i+{0} j+{1}'.format(self.x, self.y)
