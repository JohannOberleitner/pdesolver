from sources.pdesolver.finite_differences_method.FiniteDifferencesSolver_V2 import GridConfiguration, \
    ConstantGridValueProvider, FunctionGridValueProvider
from sources.pdesolver.formula_parser.parser_expression import NumberExpression
from sources.pdesolver.formula_parser.visitor import Visitor
from sources.pdesolver.pde.Function2dOffsets import Function2dOffsets
from sources.pdesolver.pde.FunctionCall import FunctionCall
from sources.pdesolver.pde.SimpleExpressionOptimizerVisitor import SimpleExpressionOptimizerVisitor


class FiniteDifferencesVisitor(Visitor):

    def __init__(self):
        self.solution_function_name = 'u'
        self.values = []

    def get_solution_function_name(self):
        return self.solution_function_name

    def set_solution_function_name(self, new_function_name):
        self.solution_function_name = new_function_name

    # 1. sammle alle Funktionen mit u(...)
    #    in Listen
    #    u(...) mit denselben Parametern sind identisch
    # 2. sammle deren Parameterwerte in einem eigenen Objekt
    # 3. und Werte deren Vorfaktoren
    # 4. summiere die Vorfaktoren
    #
    #
    def visit_function_call(self, expr):
        if expr.get_function_name() == self.solution_function_name:
            parameter_list = expr.get_parameter_expr_list()
            # offets_for_function_call are eg: (i,j+1)
            offsets_for_function_call = Function2dOffsets(parameter_list)
            function_call = FunctionCall(offsets_for_function_call)

            self.values.append([function_call])
        else:
            self.values.append([expr])

    def visit_child_expression(self, child_expr):
        child_expr.get_child().accept(self)

    def visit_binary_operator(self, binary_expr):
        symbol = binary_expr.get_symbol()

        binary_expr.get_left_child_expr().accept(self)
        binary_expr.get_right_child_expr().accept(self)

        right_function_calls = self.values.pop()
        left_function_calls = self.values.pop()

        if symbol == '+':
            left_function_calls.extend(right_function_calls)
            self.values.append(left_function_calls)
        elif symbol == '-':
            for fn in right_function_calls:
                fn.negate()
            left_function_calls.extend(right_function_calls)
            self.values.append(left_function_calls)
        elif symbol == '*':
            if len(left_function_calls) == 1:
                for fn in right_function_calls:
                    fn.multiplyPrefactor(left_function_calls[0])
                self.values.append(right_function_calls)
            else:
                raise Exception("More than 1 element in left_function_calls")
        else:
            raise Exception("Not supported operator:"+symbol)

    def visit_unary_operator(self, unary_expr):
        unary_expr.get_child_expr().accept(self)
        child_function_calls = self.values.pop()

        for fn in child_function_calls:
            fn.negate()

        self.values.append(child_function_calls)

    def combineExpressions(self):
        simplified = {}
        for functionCall in self.values[0]:
            if functionCall.offset in simplified:
                simplified[functionCall.offset].add(functionCall.prefactorExpr)
            else:
                simplified[functionCall.offset] = functionCall
        self.values = list(simplified.values())

    def simplifyExpressions(self):
        for functionCall in self.values:
            simplifier = SimpleExpressionOptimizerVisitor()
            functionCall.prefactorExpr.accept(simplifier)
            functionCall.prefactorExpr = simplifier.get_result()


    def make_grid_config(self, eps):
        gridConfig = GridConfiguration()
        for functionCall in self.values:
            if type(functionCall.prefactorExpr) is NumberExpression:
                gridConfig.add(ConstantGridValueProvider(functionCall.prefactorExpr.value),
                               int(functionCall.offset.x), int(functionCall.offset.y))
            else:
                gridConfig.add(FunctionGridValueProvider(functionCall.makeFunction(eps)),
                               int(functionCall.offset.x), int(functionCall.offset.y))
        return gridConfig
