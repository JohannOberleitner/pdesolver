from sources.pdesolver.formula_parser.parser_expression import FunctionCallExpression, BinaryOperatorExpression, \
    NumberExpression, UnaryOperatorExpression, InnerExpression
from sources.pdesolver.formula_parser.visitor import Visitor
from sources.pdesolver.pde.Sign import Sign


class RewriteFiniteDifferencesVisitor(Visitor):

    def __init__(self, variableName, sign):
        self.variableName = variableName
        self.sign = sign
        self.rewritten_expressions = []

    def get_result(self):
        return self.rewritten_expressions.pop()

    def visit_variable(self, variable_expr):
        if variable_expr.get_name() == self.variableName:
            if self.sign == Sign.Plus:
                expr = BinaryOperatorExpression(variable_expr,
                    BinaryOperatorExpression(NumberExpression(1.0), NumberExpression(2.0), '/'),'+')
            elif self.sign == Sign.Minus:
                expr = BinaryOperatorExpression(variable_expr,
                                                BinaryOperatorExpression(NumberExpression(1.0), NumberExpression(2.0),
                                                                         '/'), '-')
            else:
                raise Exception("Invalid Sign")
            self.rewritten_expressions.append(expr)
        else:
            self.rewritten_expressions.append(variable_expr)

    def visit_function_call(self, function_call_expr):
        rewritten_argument_list = []

        if function_call_expr.get_function_name() == 'gradHelper':
            for i,parameter in enumerate(function_call_expr.get_parameter_expr_list()):
                if i==0 and self.variableName == 'i':
                    parameter.accept(self)
                    paramExpr = self.rewritten_expressions.pop()
                elif i==1 and self.variableName == 'j':
                    parameter.accept(self)
                    paramExpr = self.rewritten_expressions.pop()
                elif i == 2 and self.variableName == 'k':
                    parameter.accept(self)
                    paramExpr = self.rewritten_expressions.pop()
                else:
                    pass

            self.rewritten_expressions.append(paramExpr)
        else:

            for parameter in function_call_expr.get_parameter_expr_list():
                parameter.accept(self)
                rewritten_argument_list.append(self.rewritten_expressions.pop())

            new_function_call_expr = FunctionCallExpression(function_call_expr.get_function_name(),
                                                        rewritten_argument_list)
            self.rewritten_expressions.append(new_function_call_expr)

    def visit_binary_operator(self, binary_expr):
        symbol = binary_expr.get_symbol()

        binary_expr.get_left_child_expr().accept(self)
        binary_expr.get_right_child_expr().accept(self)

        right_expr = self.rewritten_expressions.pop()
        left_expr = self.rewritten_expressions.pop()

        self.rewritten_expressions.append(BinaryOperatorExpression(left_expr, right_expr, symbol))

    def visit_unary_operator(self, unary_expr):
        symbol = unary_expr.get_symbol()
        unary_expr.get_child_expr().accept(self)

        child_expr = self.rewritten_expressions.pop()
        self.rewritten_expressions.append(UnaryOperatorExpression(child_expr, symbol))

    def visit_number(self, number_expr):
        self.rewritten_expressions.append(number_expr)

    def visit_child_expression(self, child_expr):
        child_expr.accept(self)
        new_child = self.rewritten_expressions.pop()
        self.rewritten_expressions.append(InnerExpression(new_child))
