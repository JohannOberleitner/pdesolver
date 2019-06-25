
from sources.pdesolver.formula_parser.parser_expression import NumberExpression, UnaryOperatorExpression, \
    BinaryOperatorExpression
from sources.pdesolver.formula_parser.visitor import Visitor
from sources.pdesolver.pde.ExpressionType import ExpressionType


class SimpleExpressionOptimizerVisitor(Visitor):
    def __init__(self):
        self.values = []
        self.valueTypes = []
        self.result = None

    def get_result(self):
        if self.result is None:
            assert len(self.values) == 1 and len(self.valueTypes) == 1
            valueType = self.valueTypes.pop()
            if valueType == ExpressionType.NUMBER:
                self.result = NumberExpression(self.values.pop())
            else:
                self.result = self.values.pop()
        return self.result

    def visit_variable(self, variable_expr):
        self.valueTypes.append(ExpressionType.COMPLICATED)
        self.values.append(variable_expr)

    def visit_number(self, number_expr):
        self.valueTypes.append(ExpressionType.NUMBER)
        self.values.append(number_expr.get_value())

    def visit_child_expression(self, child_expr):
        child_expr.get_child().accept(self)

    def visit_binary_operator(self, binary_expr):
        symbol = binary_expr.get_symbol()

        binary_expr.get_left_child_expr().accept(self)
        binary_expr.get_right_child_expr().accept(self)

        right_value = self.values.pop()
        right_value_type = self.valueTypes.pop()
        left_value = self.values.pop()
        left_value_type = self.valueTypes.pop()

        if left_value_type == ExpressionType.NUMBER and right_value_type == ExpressionType.NUMBER:
            self.values.append(self.calc_binary_operators_on_numbers(left_value, right_value, symbol))
            self.valueTypes.append(ExpressionType.NUMBER)
        elif left_value_type == ExpressionType.COMPLICATED or right_value_type == ExpressionType.COMPLICATED:
            self.values.append(self.calc_binary_operators_on_function_and_number(left_value, left_value_type, right_value, right_value_type, symbol))
            self.valueTypes.append(ExpressionType.COMPLICATED)
        else:
            raise Exception('Unsupported combination:' + symbol)

    def calc_binary_operators_on_numbers(self, left_value, right_value, symbol):
        if symbol == '+':
            return left_value + right_value
        elif symbol == '-':
            return left_value - right_value
        elif symbol == '*':
            return left_value * right_value
        elif symbol == '/':
            return left_value / right_value
        else:
            raise Exception('Unsupported operator symbol:' + symbol)

    def calc_binary_operators_on_function_and_number(self, first, first_type, second, second_type, symbol):
        if symbol == '*':
            if first_type == ExpressionType.NUMBER and first == 1.0:
                return second
            elif second_type == ExpressionType.NUMBER and second == 1.0:
                return first
            elif first_type == ExpressionType.NUMBER and first == -1.0:
                return UnaryOperatorExpression(second, '-')
            elif second_type == ExpressionType.NUMBER and second == -1.0:
                return UnaryOperatorExpression(first, '-')
        elif symbol == '+':
            if type(second) is UnaryOperatorExpression and second.symbol == '-':
                return BinaryOperatorExpression(first, second.get_child_expr(), '-')

        # TODO: + operator

        return BinaryOperatorExpression(first, second, symbol)

    def visit_function_call(self, function_call_expr):
        self.values.append(function_call_expr)
        self.valueTypes.append(ExpressionType.COMPLICATED)

    def visit_unary_operator(self, unary_expr):
        symbol = unary_expr.get_symbol()
        unary_expr.get_child_expr().accept(self)

        child_value = self.values.pop()
        child_value_type = self.valueTypes.pop()

        if symbol == '+':
            self.values.append(child_value)
            self.valueTypes.append(child_value_type )
        elif symbol == '-':
            # check for duplicated negation:
            if type(child_value) is UnaryOperatorExpression and child_value.symbol =='-':
                self.values.append(child_value.get_child_expr())
                self.valueTypes.append(child_value_type)
            elif child_value_type == ExpressionType.NUMBER:
                self.values.append(-child_value )
                self.valueTypes.append(child_value_type)
            else:
                self.values.append(UnaryOperatorExpression(child_value, '-'))
                self.valueTypes.append(ExpressionType.COMPLICATED)
        else:
            raise Exception('Unsupported operator symbol:' + symbol)
