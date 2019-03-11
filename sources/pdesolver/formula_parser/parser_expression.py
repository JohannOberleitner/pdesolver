import abc


class ParserExpression(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def accept(self, visitor):
        """Visitor accept method"""


class NumberExpression(ParserExpression):

    def __init__(self, value):
        self.value = value

    def get_value(self):
        return self.value

    def accept(self, visitor):
        visitor.visit_number(self)

    def __repr__(self):
        return str(self.get_value())

    def __str__(self):
        return str(self.get_value())


class VariableExpression(ParserExpression):

    def __init__(self, name):
        self.name = name

    def get_name(self):
        return self.name

    def accept(self, visitor):
        visitor.visit_variable(self)

    def __repr__(self):
        return str(self.get_name())

    def __str__(self):
        return str(self.get_name())


class InnerExpression(ParserExpression):

    def __init__(self, child):
        self.child = child

    def get_child(self):
        return self.child

    def accept(self, visitor):
        visitor.visit_child_expression(self)

    def __repr__(self):
        return '({0})'.format(self.get_child())

    def __str__(self):
        return '({0})'.format(str(self.get_child()))


class FunctionCallExpression(ParserExpression):

    def __init__(self, function_name, parameter_expr_list):
        self.function_name  = function_name
        self.parameter_expr_list = parameter_expr_list

    def get_function_name(self):
        return self.function_name

    def get_parameter_expr_list(self):
        return self.parameter_expr_list

    def accept(self, visitor):
        visitor.visit_function_call(self)

    def __repr__(self):
        if len(self.parameter_expr_list) == 0:
            return '{0}()'.format(self.get_function_name())
        else:
            return '{0}({1})'.format(self.get_function_name(), ','.join(map(str, self.get_parameter_expr_list())))

    def __str__(self):
        if len(self.parameter_expr_list) == 0:
            return '{0}()'.format(self.get_function_name())
        else:
            return '{0}({1})'.format(self.get_function_name(), ','.join(map(str, self.get_parameter_expr_list())))


class BinaryOperatorExpression(ParserExpression):

    def __init__(self, left_child_expr, right_child_expr, symbol):
        self.left_child_expr = left_child_expr
        self.right_child_expr = right_child_expr
        self.symbol = symbol

    def get_left_child_expr(self):
        return self.left_child_expr

    def get_right_child_expr(self):
        return self.right_child_expr

    def get_symbol(self):
        return self.symbol

    def accept(self, visitor):
        visitor.visit_binary_operator(self)

    def __repr__(self):
        return '{0} {1} {2}'.format(self.get_left_child_expr(), self.get_symbol(), self.get_right_child_expr())

    def __str__(self):
        return '{0} {1} {2}'.format(str(self.get_left_child_expr()), self.get_symbol(), str(self.get_right_child_expr()))

class UnaryOperatorExpression(ParserExpression):

    def __init__(self, child_expr, symbol):
        self.child_expr = child_expr
        self.symbol = symbol

    def get_child_expr(self):
        return self.child_expr

    def get_symbol(self):
        return self.symbol

    def accept(self, visitor):
        visitor.visit_unary_operator(self)

    def __repr__(self):
        return '{0} {1}'.format(self.get_symbol(), self.get_child_expr())

    def __str__(self):
        return '{0} {1}'.format(self.get_symbol(), str(self.get_child_expr()))


