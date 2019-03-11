
class Visitor:

    def __init(self, expr):
        self.expr = expr

    def visit_unary_operator(self, unary_expr):
        pass

    def visit_binary_operator(self, binary_expr):
        pass

    def visit_function_call(self, function_call_expr):
        pass

    def visit_child_expression(self, child_expr):
        pass

    def visit_variable(self, variable_expr):
        pass

    def visit_number(self, number_expr):
        pass