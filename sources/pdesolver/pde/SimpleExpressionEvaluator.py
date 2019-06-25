from sources.pdesolver.formula_parser.visitor import Visitor


class SimpleExpressionEvaluator(Visitor):
    """
    Implements an evaluator for expressions that have been parsed with the formula_parser.
    It is implemented as a visitor that is applied on the parsed formula.

    Attributes
    ------
    values: keeps the set of results and intermediary results used as a stack.



    """


    def __init__(self, variables, functions={}):
        """
        :param variables: Set of variables that may be used for the evaluation
        :param functions: Set of functions that may be used for the evaluation
        """
        self.values = []
        self.variables = variables
        self.functions = functions
        self.result = None

    def get_result(self):
        if self.result is None:
            self.result = self.values.pop()
        return self.result

    def visit_number(self, number_expr):
        self.values.append(number_expr.get_value())

    def visit_function_call(self, function_call_expr):
        parameter_values = []
        for parameter in function_call_expr.get_parameter_expr_list():
            parameter.accept(self)
            parameter_values.append(self.values.pop())

        function_name = function_call_expr.get_function_name()
        if function_name in self.functions:
            fn = self.functions[function_name]
            function_result = fn(parameter_values)
            self.values.append(function_result)
        else:
            raise Exception("Function not provided for evaluation:" + function_name)

    def visit_variable(self, variable_expr):
        name = variable_expr.get_name()
        if name in self.variables:
            self.values.append(self.variables[name])
        else:
            raise Exception("Variable has no value:"+name)

    def visit_child_expression(self, child_expr):
        child_expr.get_child().accept(self)

    def visit_binary_operator(self, binary_expr):
        symbol = binary_expr.get_symbol()

        binary_expr.get_left_child_expr().accept(self)
        binary_expr.get_right_child_expr().accept(self)

        right_value = self.values.pop()
        left_value = self.values.pop()

        if symbol == '+':
            self.values.append(left_value + right_value)
        elif symbol == '-':
            self.values.append(left_value - right_value)
        elif symbol == '*':
            self.values.append(left_value * right_value)
        elif symbol == '/':
            self.values.append(left_value / right_value)
        else:
            raise Exception('Unsupported operator symbol:'+symbol)

    def visit_unary_operator(self, unary_expr):
        symbol = unary_expr.get_symbol()
        unary_expr.get_child_expr().accept(self)

        child_value = self.values.pop()

        if symbol == '+':
            self.values.append(child_value)
        elif symbol == '-':
            self.values.append(-child_value)
        else:
            raise Exception('Unsupported operator symbol:' + symbol)
