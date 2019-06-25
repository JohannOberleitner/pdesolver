from sources.pdesolver.formula_parser.parser_expression import VariableExpression, BinaryOperatorExpression, \
    FunctionCallExpression
from sources.pdesolver.formula_parser.visitor import Visitor
from sources.pdesolver.pde.Sign import Sign
from sources.pdesolver.pde.RewriteFiniteDifferencesVisitor import RewriteFiniteDifferencesVisitor


class VectorCalculusExpressionVisitor(Visitor):

    def __init__(self, vectorVariableNames, dimension):
        self.vectorVariableNames = vectorVariableNames
        self.dimension = dimension
        self.rewritten_expressions = []
        self.rewritten_expressions_count = []

        self.visitor_i_plus = RewriteFiniteDifferencesVisitor('i', sign=Sign.Plus)
        self.visitor_j_plus = RewriteFiniteDifferencesVisitor('j', sign=Sign.Plus)
        self.visitor_k_plus = RewriteFiniteDifferencesVisitor('k', sign=Sign.Plus)
        self.visitor_i_minus = RewriteFiniteDifferencesVisitor('i', sign=Sign.Minus)
        self.visitor_j_minus = RewriteFiniteDifferencesVisitor('j', sign=Sign.Minus)
        self.visitor_k_minus = RewriteFiniteDifferencesVisitor('k', sign=Sign.Minus)

    def get_result(self):
        return self.rewritten_expressions.pop()

    def getVectorCoordinateNames(self, vectorVariableName):
        # at time being ignore all variable names
        if self.dimension == 2:
            return ['i', 'j']
        elif self.dimension == 3:
            return ['i', 'j', 'k']
        else:
            raise Exception('Vectors of dimension 1 or dimension > 3 not supported!')

    def visit_number(self, number_expr):
        self.rewritten_expressions.append(number_expr)
        self.rewritten_expressions_count.append(1)

    def visit_variable(self, variable_expr):
        if variable_expr.get_name() in self.vectorVariableNames:
            for coordinateName in self.getVectorCoordinateNames(variable_expr.get_name()):
                self.rewritten_expressions.append(VariableExpression(coordinateName))
            self.rewritten_expressions_count.append(self.dimension)
        else:
            self.rewritten_expressions.append(variable_expr)
            self.rewritten_expressions_count.append(1)

    def visit_binary_operator(self, binary_expr):
        symbol = binary_expr.get_symbol()

        binary_expr.get_left_child_expr().accept(self)
        binary_expr.get_right_child_expr().accept(self)

        right_expr = self.rewritten_expressions.pop()
        left_expr = self.rewritten_expressions.pop()

        self.rewritten_expressions.append(BinaryOperatorExpression(left_expr, right_expr, symbol))

    def visit_function_call(self, function_call_expr):
        rewritten_argument_list = []
        for parameter in function_call_expr.get_parameter_expr_list():
            parameter.accept(self)
            count = self.rewritten_expressions_count.pop()
            for i in range(count):
                rewritten_argument_list.insert(0, self.rewritten_expressions.pop())

        if function_call_expr.get_function_name() == 'grad':
            self.visit_grad(rewritten_argument_list.pop())

        elif function_call_expr.get_function_name() == 'div':
            self.visit_div(rewritten_argument_list.pop())
        elif function_call_expr.get_function_name() == 'rot':
            raise Exception("rot not implemented")
        else:
            new_function_call_expr = FunctionCallExpression(function_call_expr.get_function_name(),rewritten_argument_list )
            self.rewritten_expressions.append(new_function_call_expr)
            self.rewritten_expressions_count.append(1)

    def apply_finite_differences(self, operand_expr, visitor_plus, visitor_minus):
        expr = operand_expr
        expr.accept(visitor_plus)
        left_expr = visitor_plus.get_result()
        expr.accept(visitor_minus)
        right_expr = visitor_minus.get_result()
        return BinaryOperatorExpression(left_expr, right_expr, '-')

    def visit_div(self, operand_expr):
        # TODO: split on all u functions and combine them
        # TODO: do for eps as well

        #expr_coord_i = operand_expr.get_parameter_expr_list()[0]
        expr_coord_i = operand_expr
        expr_i = self.apply_finite_differences(expr_coord_i, self.visitor_i_plus, self.visitor_i_minus)

        expr_coord_j = operand_expr
        expr_j = self.apply_finite_differences(expr_coord_j, self.visitor_j_plus, self.visitor_j_minus)


        # expr_coord_k = operand.get_parameter_expr_list()[2]

        expr = BinaryOperatorExpression(expr_i, expr_j, '+')

        self.rewritten_expressions.append(expr)
        self.rewritten_expressions_count.append(1)

    def visit_grad(self, operand_expr):

        expr_i = self.apply_finite_differences(operand_expr, self.visitor_i_plus, self.visitor_i_minus)
        expr_j = self.apply_finite_differences(operand_expr, self.visitor_j_plus, self.visitor_j_minus)

        expr = FunctionCallExpression('gradHelper', [expr_i, expr_j])

        self.rewritten_expressions.append(expr)
        self.rewritten_expressions_count.append(1)

        # u(i,j) -> (u(i+1/2,j) - u(i-1/2,j)) + (u(i,j+1/2) - u(i,j-1/2)
