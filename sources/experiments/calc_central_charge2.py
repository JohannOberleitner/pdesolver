import time
from enum import Enum

import matplotlib.pyplot as plt

from sources.experiments.charges_generators import make_central_charge
from sources.experiments.fdm_helper import plotSurface
from sources.pdesolver.finite_differences_method.FiniteDifferencesSolver_V2 import GridConfiguration, \
    ConstantGridValueProvider, FiniteDifferencesMethod4, FunctionGridValueProvider
from sources.pdesolver.finite_differences_method.boundaryconditions import RectangularBoundaryCondition
from sources.pdesolver.finite_differences_method.charge_distribution import ChargeDistribution
from sources.pdesolver.finite_differences_method.geometry import Geometry
from sources.pdesolver.finite_differences_method.rectangle import Rectangle
from sources.pdesolver.formula_parser.lexer import Lexer
from sources.pdesolver.formula_parser.parser import Parser, NumberExpression, UnaryOperatorExpression, \
    BinaryOperatorExpression
from sources.pdesolver.formula_parser.visitor import Visitor


def solvePDE(geometry, charges, gridConfig):

    g = geometry
    boundaryCondition = RectangularBoundaryCondition(geometry)

    start = time.time()

    fdm = FiniteDifferencesMethod4(g, boundaryCondition, gridConfig, charges)
    fdm.solve()

    resulting_matrix = fdm.values

    duration = time.time() - start
    print('Total duration for solving the PDE lasted {0} sec'.format(duration))

    return resulting_matrix

class SimpleExpressionEvaluator(Visitor):

    def __init__(self, variables, functions={}):
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

class ExpressionType(Enum):
    NUMBER = 0,
    COMPLICATED= 1

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
                self.result = NumberExpression(self.value.pop())
            else:
                self.result = self.values.pop()
        return self.result

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


class Function2DOffsets:
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

    def makeFunction(self, eps):
        def evaluatorFn(col, row):
            evaluator = SimpleExpressionEvaluator(variables={'i':col, 'j':row}, functions={'eps':eps})
            self.prefactorExpr.accept(evaluator)
            return evaluator.get_result()

        return evaluatorFn


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
            offsets_for_function_call = Function2DOffsets(parameter_list)
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


def eps(params):
    col = params[0] #i
    row = params[1] #j
    return col+row

def make_pde_config(expression):
    # expr = 'f(i,2)'

    # div( eps(r)*grad u(r) )

    expression = 'eps(i+1/2,j)*(u(i+1,j)-u(i,j)) - eps(i-1/2,j)*(u(i,j)-u(i-1,j)) + ' + \
           'eps(i,j+1/2)*(u(i,j+1)-u(i,j)) - eps(i,j-1/2)*(u(i,j)-u(i,j-1))'
    #expr = '(u(i+1,j)-u(i,j)) - (u(i,j)-u(i-1,j)) + (u(i,j+1)-u(i,j)) - (u(i,j)-u(i,j-1))'

    lexer = Lexer(expression)
    l = list(lexer.parse())
    parser = Parser(l)
    expr = parser.parse()

    visitor = FiniteDifferencesVisitor()

    expr.accept(visitor)
    visitor.combineExpressions()
    visitor.simplifyExpressions()

    return visitor.make_grid_config(eps)

if __name__ == '__main__':

    delta = 1.0
    rect = Rectangle(0, 0, 64.0, 64.0)

    g = Geometry(rect, delta)
    print(g.numX, g.numY)

    gridConfig = make_pde_config('(u(i+1,j)-u(i,j)) - (u(i,j)-u(i-1,j)) + (u(i,j+1)-u(i,j)) - (u(i,j)-u(i,j-1))')

    charges = make_central_charge(g)

    start = time.clock()
    duration = time.clock() - start

    #fdm = solve_finite_differences(g, boundaryCondition, charges)
    resulting_matrix = solvePDE(g, charges, gridConfig)

    print(duration)



    showGraph = 1

    if showGraph:
        plotSurface(g.X, g.Y, resulting_matrix)
        plt.show()
