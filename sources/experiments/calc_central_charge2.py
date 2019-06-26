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
    BinaryOperatorExpression, VariableExpression, FunctionCallExpression, InnerExpression
from sources.pdesolver.formula_parser.visitor import Visitor
from sources.pdesolver.pde.PDE import PDEExpressionType, PDE


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

# class SimpleExpressionEvaluator(Visitor):
#
#     def __init__(self, variables, functions={}):
#         self.values = []
#         self.variables = variables
#         self.functions = functions
#         self.result = None
#
#     def get_result(self):
#         if self.result is None:
#             self.result = self.values.pop()
#         return self.result
#
#     def visit_number(self, number_expr):
#         self.values.append(number_expr.get_value())
#
#     def visit_function_call(self, function_call_expr):
#         parameter_values = []
#         for parameter in function_call_expr.get_parameter_expr_list():
#             parameter.accept(self)
#             parameter_values.append(self.values.pop())
#
#         function_name = function_call_expr.get_function_name()
#         if function_name in self.functions:
#             fn = self.functions[function_name]
#             function_result = fn(parameter_values)
#             self.values.append(function_result)
#         else:
#             raise Exception("Function not provided for evaluation:" + function_name)
#
#     def visit_variable(self, variable_expr):
#         name = variable_expr.get_name()
#         if name in self.variables:
#             self.values.append(self.variables[name])
#         else:
#             raise Exception("Variable has no value:"+name)
#
#     def visit_child_expression(self, child_expr):
#         child_expr.get_child().accept(self)
#
#     def visit_binary_operator(self, binary_expr):
#         symbol = binary_expr.get_symbol()
#
#         binary_expr.get_left_child_expr().accept(self)
#         binary_expr.get_right_child_expr().accept(self)
#
#         right_value = self.values.pop()
#         left_value = self.values.pop()
#
#         if symbol == '+':
#             self.values.append(left_value + right_value)
#         elif symbol == '-':
#             self.values.append(left_value - right_value)
#         elif symbol == '*':
#             self.values.append(left_value * right_value)
#         elif symbol == '/':
#             self.values.append(left_value / right_value)
#         else:
#             raise Exception('Unsupported operator symbol:'+symbol)
#
#     def visit_unary_operator(self, unary_expr):
#         symbol = unary_expr.get_symbol()
#         unary_expr.get_child_expr().accept(self)
#
#         child_value = self.values.pop()
#
#         if symbol == '+':
#             self.values.append(child_value)
#         elif symbol == '-':
#             self.values.append(-child_value)
#         else:
#             raise Exception('Unsupported operator symbol:' + symbol)

# class ExpressionType(Enum):
#     NUMBER = 0,
#     COMPLICATED= 1

# class SimpleExpressionOptimizerVisitor(Visitor):
#     def __init__(self):
#         self.values = []
#         self.valueTypes = []
#         self.result = None
#
#     def get_result(self):
#         if self.result is None:
#             assert len(self.values) == 1 and len(self.valueTypes) == 1
#             valueType = self.valueTypes.pop()
#             if valueType == ExpressionType.NUMBER:
#                 self.result = NumberExpression(self.values.pop())
#             else:
#                 self.result = self.values.pop()
#         return self.result
#
#     def visit_variable(self, variable_expr):
#         self.valueTypes.append(ExpressionType.COMPLICATED)
#         self.values.append(variable_expr)
#
#     def visit_number(self, number_expr):
#         self.valueTypes.append(ExpressionType.NUMBER)
#         self.values.append(number_expr.get_value())
#
#     def visit_child_expression(self, child_expr):
#         child_expr.get_child().accept(self)
#
#     def visit_binary_operator(self, binary_expr):
#         symbol = binary_expr.get_symbol()
#
#         binary_expr.get_left_child_expr().accept(self)
#         binary_expr.get_right_child_expr().accept(self)
#
#         right_value = self.values.pop()
#         right_value_type = self.valueTypes.pop()
#         left_value = self.values.pop()
#         left_value_type = self.valueTypes.pop()
#
#         if left_value_type == ExpressionType.NUMBER and right_value_type == ExpressionType.NUMBER:
#             self.values.append(self.calc_binary_operators_on_numbers(left_value, right_value, symbol))
#             self.valueTypes.append(ExpressionType.NUMBER)
#         elif left_value_type == ExpressionType.COMPLICATED or right_value_type == ExpressionType.COMPLICATED:
#             self.values.append(self.calc_binary_operators_on_function_and_number(left_value, left_value_type, right_value, right_value_type, symbol))
#             self.valueTypes.append(ExpressionType.COMPLICATED)
#         else:
#             raise Exception('Unsupported combination:' + symbol)
#
#     def calc_binary_operators_on_numbers(self, left_value, right_value, symbol):
#         if symbol == '+':
#             return left_value + right_value
#         elif symbol == '-':
#             return left_value - right_value
#         elif symbol == '*':
#             return left_value * right_value
#         elif symbol == '/':
#             return left_value / right_value
#         else:
#             raise Exception('Unsupported operator symbol:' + symbol)
#
#     def calc_binary_operators_on_function_and_number(self, first, first_type, second, second_type, symbol):
#         if symbol == '*':
#             if first_type == ExpressionType.NUMBER and first == 1.0:
#                 return second
#             elif second_type == ExpressionType.NUMBER and second == 1.0:
#                 return first
#             elif first_type == ExpressionType.NUMBER and first == -1.0:
#                 return UnaryOperatorExpression(second, '-')
#             elif second_type == ExpressionType.NUMBER and second == -1.0:
#                 return UnaryOperatorExpression(first, '-')
#         elif symbol == '+':
#             if type(second) is UnaryOperatorExpression and second.symbol == '-':
#                 return BinaryOperatorExpression(first, second.get_child_expr(), '-')
#
#         # TODO: + operator
#
#         return BinaryOperatorExpression(first, second, symbol)
#
#     def visit_function_call(self, function_call_expr):
#         self.values.append(function_call_expr)
#         self.valueTypes.append(ExpressionType.COMPLICATED)
#
#     def visit_unary_operator(self, unary_expr):
#         symbol = unary_expr.get_symbol()
#         unary_expr.get_child_expr().accept(self)
#
#         child_value = self.values.pop()
#         child_value_type = self.valueTypes.pop()
#
#         if symbol == '+':
#             self.values.append(child_value)
#             self.valueTypes.append(child_value_type )
#         elif symbol == '-':
#             # check for duplicated negation:
#             if type(child_value) is UnaryOperatorExpression and child_value.symbol =='-':
#                 self.values.append(child_value.get_child_expr())
#                 self.valueTypes.append(child_value_type)
#             elif child_value_type == ExpressionType.NUMBER:
#                 self.values.append(-child_value )
#                 self.valueTypes.append(child_value_type)
#             else:
#                 self.values.append(UnaryOperatorExpression(child_value, '-'))
#                 self.valueTypes.append(ExpressionType.COMPLICATED)
#         else:
#             raise Exception('Unsupported operator symbol:' + symbol)


# class Function2DOffsets:
#     def __init__(self, parameter_expr_list):
#         assert len(parameter_expr_list)==2
#         evaluator_x = self.evaluateParameterExpression(parameter_expr_list[0], 'i')
#         evaluator_y = self.evaluateParameterExpression(parameter_expr_list[1], 'j')
#         self.x = evaluator_x
#         self.y = evaluator_y
#
#     def evaluateParameterExpression(self, parameter_expr, positional_variable_name):
#         variables = {positional_variable_name:0.0}
#         evaluator = SimpleExpressionEvaluator(variables)
#         parameter_expr.accept(evaluator)
#         return evaluator.get_result()
#
#     def __eq__(self, another):
#         return self.x == another.x and self.y == another.y
#
#     def __hash__(self):
#         return hash(self.x) * hash(self.y)
#
#     def __repr__(self):
#         return 'i+{0} j+{1}'.format(self.x, self.y)
#
#     def __str__(self):
#         return 'i+{0} j+{1}'.format(self.x, self.y)


# class FunctionCall:
#     def __init__(self, offset):
#         self.offset = offset
#         self.prefactorExpr = NumberExpression(1.0)
#         self.simpleExpression = True
#
#     def get_offset(self):
#         return self.offset
#
#     def get_prefactor(self):
#         return self.prefactorExpr
#
#     def negate(self):
#         self.prefactorExpr = UnaryOperatorExpression(self.prefactorExpr, '-')
#
#     def multiplyPrefactor(self, expr):
#         self.prefactorExpr = BinaryOperatorExpression(expr, self.prefactorExpr, '*')
#         self.simpleExpression = False
#
#     def __str__(self):
#         return '({0})u({1})'.format(str(self.prefactorExpr), self.offset)
#
#     def __repr__(self):
#         return '({0})u({1})'.format(repr(self.prefactorExpr), self.offset)
#
#     def add(self, expr):
#         self.prefactorExpr = BinaryOperatorExpression(self.prefactorExpr, expr, '+')
#
#     def makeFunction(self, functionDict):
#         def evaluatorFn(col, row):
#             evaluator = SimpleExpressionEvaluator(variables={'i':col, 'j':row}, functions=functionDict)
#             self.prefactorExpr.accept(evaluator)
#             return evaluator.get_result()
#
#         return evaluatorFn


# class FiniteDifferencesVisitor(Visitor):
#
#     def __init__(self):
#         self.solution_function_name = 'u'
#         self.values = []
#
#     def get_solution_function_name(self):
#         return self.solution_function_name
#
#     def set_solution_function_name(self, new_function_name):
#         self.solution_function_name = new_function_name
#
#     # 1. sammle alle Funktionen mit u(...)
#     #    in Listen
#     #    u(...) mit denselben Parametern sind identisch
#     # 2. sammle deren Parameterwerte in einem eigenen Objekt
#     # 3. und Werte deren Vorfaktoren
#     # 4. summiere die Vorfaktoren
#     #
#     #
#     def visit_function_call(self, expr):
#         if expr.get_function_name() == self.solution_function_name:
#             parameter_list = expr.get_parameter_expr_list()
#             # offets_for_function_call are eg: (i,j+1)
#             offsets_for_function_call = Function2DOffsets(parameter_list)
#             function_call = FunctionCall(offsets_for_function_call)
#
#             self.values.append([function_call])
#         else:
#             self.values.append([expr])
#
#     def visit_child_expression(self, child_expr):
#         child_expr.get_child().accept(self)
#
#     def visit_binary_operator(self, binary_expr):
#         symbol = binary_expr.get_symbol()
#
#         binary_expr.get_left_child_expr().accept(self)
#         binary_expr.get_right_child_expr().accept(self)
#
#         right_function_calls = self.values.pop()
#         left_function_calls = self.values.pop()
#
#         if symbol == '+':
#             left_function_calls.extend(right_function_calls)
#             self.values.append(left_function_calls)
#         elif symbol == '-':
#             for fn in right_function_calls:
#                 fn.negate()
#             left_function_calls.extend(right_function_calls)
#             self.values.append(left_function_calls)
#         elif symbol == '*':
#             if len(left_function_calls) == 1:
#                 for fn in right_function_calls:
#                     fn.multiplyPrefactor(left_function_calls[0])
#                 self.values.append(right_function_calls)
#             else:
#                 raise Exception("More than 1 element in left_function_calls")
#         else:
#             raise Exception("Not supported operator:"+symbol)
#
#     def visit_unary_operator(self, unary_expr):
#         unary_expr.get_child_expr().accept(self)
#         child_function_calls = self.values.pop()
#
#         for fn in child_function_calls:
#             fn.negate()
#
#         self.values.append(child_function_calls)
#
#     def combineExpressions(self):
#         simplified = {}
#         for functionCall in self.values[0]:
#             if functionCall.offset in simplified:
#                 simplified[functionCall.offset].add(functionCall.prefactorExpr)
#             else:
#                 simplified[functionCall.offset] = functionCall
#         self.values = list(simplified.values())
#
#     def simplifyExpressions(self):
#         for functionCall in self.values:
#             simplifier = SimpleExpressionOptimizerVisitor()
#             functionCall.prefactorExpr.accept(simplifier)
#             functionCall.prefactorExpr = simplifier.get_result()
#
#
#     def make_grid_config(self, eps):
#         gridConfig = GridConfiguration()
#         for functionCall in self.values:
#             if type(functionCall.prefactorExpr) is NumberExpression:
#                 gridConfig.add(ConstantGridValueProvider(functionCall.prefactorExpr.value),
#                                int(functionCall.offset.x), int(functionCall.offset.y))
#             else:
#                 gridConfig.add(FunctionGridValueProvider(functionCall.makeFunction(eps)),
#                                int(functionCall.offset.x), int(functionCall.offset.y))
#         return gridConfig


#
# class Sign(Enum):
#     Plus=1,
#     Minus=2

# class VectorCalculusExpressionVisitor(Visitor):
#
#     def __init__(self, vectorVariableNames, dimension):
#         self.vectorVariableNames = vectorVariableNames
#         self.dimension = dimension
#         self.rewritten_expressions = []
#         self.rewritten_expressions_count = []
#         self.visitor_i_plus = RewriteFiniteDifferencesVisitor('i', sign=Sign.Plus)
#         self.visitor_j_plus = RewriteFiniteDifferencesVisitor('j', sign=Sign.Plus)
#         self.visitor_k_plus = RewriteFiniteDifferencesVisitor('k', sign=Sign.Plus)
#         self.visitor_i_minus = RewriteFiniteDifferencesVisitor('i', sign=Sign.Minus)
#         self.visitor_j_minus = RewriteFiniteDifferencesVisitor('j', sign=Sign.Minus)
#         self.visitor_k_minus = RewriteFiniteDifferencesVisitor('k', sign=Sign.Minus)
#
#     def get_result(self):
#         return self.rewritten_expressions.pop()
#
#     def getVectorCoordinateNames(self, vectorVariableName):
#         # at time being ignore all variable names
#         if self.dimension == 2:
#             return ['i', 'j']
#         elif self.dimension == 3:
#             return ['i', 'j', 'k']
#         else:
#             raise Exception('Vectors of dimension 1 or dimension > 3 not supported!')
#
#     def visit_number(self, number_expr):
#         self.rewritten_expressions.append(number_expr)
#         self.rewritten_expressions_count.append(1)
#
#     def visit_variable(self, variable_expr):
#         if variable_expr.get_name() in self.vectorVariableNames:
#             for coordinateName in self.getVectorCoordinateNames(variable_expr.get_name()):
#                 self.rewritten_expressions.append(VariableExpression(coordinateName))
#             self.rewritten_expressions_count.append(self.dimension)
#         else:
#             self.rewritten_expressions.append(variable_expr)
#             self.rewritten_expressions_count.append(1)
#
#     def visit_binary_operator(self, binary_expr):
#         symbol = binary_expr.get_symbol()
#
#         binary_expr.get_left_child_expr().accept(self)
#         binary_expr.get_right_child_expr().accept(self)
#
#         right_expr = self.rewritten_expressions.pop()
#         left_expr = self.rewritten_expressions.pop()
#
#         self.rewritten_expressions.append(BinaryOperatorExpression(left_expr, right_expr, symbol))
#
#     def visit_function_call(self, function_call_expr):
#         rewritten_argument_list = []
#         for parameter in function_call_expr.get_parameter_expr_list():
#             parameter.accept(self)
#             count = self.rewritten_expressions_count.pop()
#             for i in range(count):
#                 rewritten_argument_list.insert(0, self.rewritten_expressions.pop())
#
#         if function_call_expr.get_function_name() == 'grad':
#             self.visit_grad(rewritten_argument_list.pop())
#
#         elif function_call_expr.get_function_name() == 'div':
#             self.visit_div(rewritten_argument_list.pop())
#         elif function_call_expr.get_function_name() == 'rot':
#             raise Exception("rot not implemented")
#         else:
#             new_function_call_expr = FunctionCallExpression(function_call_expr.get_function_name(),rewritten_argument_list )
#             self.rewritten_expressions.append(new_function_call_expr)
#             self.rewritten_expressions_count.append(1)
#
#     def apply_finite_differences(self, operand_expr, visitor_plus, visitor_minus):
#         expr = operand_expr
#         expr.accept(visitor_plus)
#         left_expr = visitor_plus.get_result()
#         expr.accept(visitor_minus)
#         right_expr = visitor_minus.get_result()
#         return BinaryOperatorExpression(left_expr, right_expr, '-')
#
#     def visit_div(self, operand_expr):
#         # TODO: split on all u functions and combine them
#         # TODO: do for eps as well
#
#         #expr_coord_i = operand_expr.get_parameter_expr_list()[0]
#         expr_coord_i = operand_expr
#         expr_i = self.apply_finite_differences(expr_coord_i, self.visitor_i_plus, self.visitor_i_minus)
#
#         expr_coord_j = operand_expr
#         expr_j = self.apply_finite_differences(expr_coord_j, self.visitor_j_plus, self.visitor_j_minus)
#
#
#         # expr_coord_k = operand.get_parameter_expr_list()[2]
#
#         expr = BinaryOperatorExpression(expr_i, expr_j, '+')
#
#         self.rewritten_expressions.append(expr)
#         self.rewritten_expressions_count.append(1)
#
#     def visit_grad(self, operand_expr):
#
#         expr_i = self.apply_finite_differences(operand_expr, self.visitor_i_plus, self.visitor_i_minus)
#         expr_j = self.apply_finite_differences(operand_expr, self.visitor_j_plus, self.visitor_j_minus)
#
#         expr = FunctionCallExpression('gradHelper', [expr_i, expr_j])
#
#         self.rewritten_expressions.append(expr)
#         self.rewritten_expressions_count.append(1)
#
#         # u(i,j) -> (u(i+1/2,j) - u(i-1/2,j)) + (u(i,j+1/2) - u(i,j-1/2)

# class RewriteFiniteDifferencesVisitor(Visitor):
#
#     def __init__(self, variableName, sign):
#         self.variableName = variableName
#         self.sign = sign
#         self.rewritten_expressions = []
#
#     def get_result(self):
#         return self.rewritten_expressions.pop()
#
#     def visit_variable(self, variable_expr):
#         if variable_expr.get_name() == self.variableName:
#             if self.sign == Sign.Plus:
#                 expr = BinaryOperatorExpression(variable_expr,
#                     BinaryOperatorExpression(NumberExpression(1.0), NumberExpression(2.0), '/'),'+')
#             elif self.sign == Sign.Minus:
#                 expr = BinaryOperatorExpression(variable_expr,
#                                                 BinaryOperatorExpression(NumberExpression(1.0), NumberExpression(2.0),
#                                                                          '/'), '-')
#             else:
#                 raise Exception("Invalid Sign")
#             self.rewritten_expressions.append(expr)
#         else:
#             self.rewritten_expressions.append(variable_expr)
#
#     def visit_function_call(self, function_call_expr):
#         rewritten_argument_list = []
#
#         if function_call_expr.get_function_name() == 'gradHelper':
#             for i,parameter in enumerate(function_call_expr.get_parameter_expr_list()):
#                 if i==0 and self.variableName == 'i':
#                     parameter.accept(self)
#                     paramExpr = self.rewritten_expressions.pop()
#                 elif i==1 and self.variableName == 'j':
#                     parameter.accept(self)
#                     paramExpr = self.rewritten_expressions.pop()
#                 elif i == 2 and self.variableName == 'k':
#                     parameter.accept(self)
#                     paramExpr = self.rewritten_expressions.pop()
#                 else:
#                     pass
#
#             self.rewritten_expressions.append(paramExpr)
#         else:
#
#             for parameter in function_call_expr.get_parameter_expr_list():
#                 parameter.accept(self)
#                 rewritten_argument_list.append(self.rewritten_expressions.pop())
#
#             new_function_call_expr = FunctionCallExpression(function_call_expr.get_function_name(),
#                                                         rewritten_argument_list)
#             self.rewritten_expressions.append(new_function_call_expr)
#
#     def visit_binary_operator(self, binary_expr):
#         symbol = binary_expr.get_symbol()
#
#         binary_expr.get_left_child_expr().accept(self)
#         binary_expr.get_right_child_expr().accept(self)
#
#         right_expr = self.rewritten_expressions.pop()
#         left_expr = self.rewritten_expressions.pop()
#
#         self.rewritten_expressions.append(BinaryOperatorExpression(left_expr, right_expr, symbol))
#
#     def visit_unary_operator(self, unary_expr):
#         symbol = unary_expr.get_symbol()
#         unary_expr.get_child_expr().accept(self)
#
#         child_expr = self.rewritten_expressions.pop()
#         self.rewritten_expressions.append(UnaryOperatorExpression(child_expr, symbol))
#
#     def visit_number(self, number_expr):
#         self.rewritten_expressions.append(number_expr)
#
#     def visit_child_expression(self, child_expr):
#         child_expr.accept(self)
#         new_child = self.rewritten_expressions.pop()
#         self.rewritten_expressions.append(InnerExpression(new_child))


# class PDEExpressionType(Enum):
#     NONE = 0
#     FINITE_DIFFERENCES = 1,
#     VECTOR_CALCULUS = 2
#
# class PDE:
#
#     def __init__(self, gridWidth, gridHeight):
#         self.gridWidth = gridWidth
#         self.gridHeight = gridHeight
#         self.delta = 1.0
#         self.rect = Rectangle(0, 0, gridWidth, gridHeight)
#         self.geometry = Geometry(self.rect, self.delta)
#         self.boundaryCondition = RectangularBoundaryCondition(self.geometry)
#         self.auxiliaryFunctions = {}
#
#     def setEquationExpression(self, expressionType, expressionString):
#         self.expressionType = expressionType
#         lexer = Lexer(expressionString)
#         l = list(lexer.parse())
#         parser = Parser(l)
#         self.expression = parser.parse()
#
#     def setVectorVariable(self, vectorVariableName, dimension=2):
#         if self.expressionType != PDEExpressionType.VECTOR_CALCULUS:
#             raise Exception("Expression type must be set to VECTOR_EXPRESSION")
#
#         self.vectorVariableName = vectorVariableName
#         self.dimension = dimension
#
#     def setAuxiliaryFunctions(self, functionDictionary):
#         self.auxiliaryFunctions = functionDictionary
#
#     def configureGrid(self):
#         if self.expressionType == PDEExpressionType.NONE:
#             raise Exception("Expression not set")
#
#         if self.expressionType == PDEExpressionType.VECTOR_CALCULUS:
#             finiteDifferencesExpression = self.evaluateVectorCalculusExpression(self.expression)
#             self.configureFiniteDifferences(finiteDifferencesExpression)
#         else:
#             self.configureFiniteDifferences(self.expression)
#
#     def evaluateVectorCalculusExpression(self, vectorCalculusExpression):
#         visitor = VectorCalculusExpressionVisitor([self.vectorVariableName], self.dimension)
#         vectorCalculusExpression.accept(visitor)
#         finiteDifferencesExpression = visitor.get_result()
#         return finiteDifferencesExpression
#
#     def configureFiniteDifferences(self, finiteDifferencesExpression):
#         visitor = FiniteDifferencesVisitor()
#         finiteDifferencesExpression.accept(visitor)
#         visitor.combineExpressions()
#         visitor.simplifyExpressions()
#         self.gridConfig = visitor.make_grid_config(self.auxiliaryFunctions)
#
#
#     # TODO: replace charges -> rightSide
#     def solve(self, charges):
#
#         start = time.time()
#
#         fdm = FiniteDifferencesMethod4(self.geometry, self.boundaryCondition, self.gridConfig, charges)
#         fdm.solve()
#
#         resulting_matrix = fdm.values
#
#         self.duration = time.time() - start
#         #print('Total duration for solving the PDE lasted {0} sec'.format(duration))
#         return resulting_matrix

def eps(params):
    col = params[0] #i
    row = params[1] #j
    if (col > 10 and col < 54 and row > 10 and row < 54): # and (col < 28 or col > 36) and (row < 28 or row > 36):
        if col > 15 and row > 15 and col < 49 and row < 48:
            if col > 25 and row > 25 and col < 39 and row < 39:
                return 10.0
            else:
                return 3.0
        else:
            return 20.0
    else:
        return 1.0


def setupPDE_finite_differences():
    # 1. Finite Differences without aux function eps
    pde = PDE(64.0, 64.0)
    pde.setEquationExpression(PDEExpressionType.FINITE_DIFFERENCES,
                              '(u(i+1,j)-u(i,j)) - (u(i,j)-u(i-1,j)) + (u(i,j+1)-u(i,j)) - (u(i,j)-u(i,j-1))')
    pde.configureGrid()
    return pde

def setupPDE_finite_differences_with_eps():
    # 2. Finite Differences with aux function eps
    pde = PDE(64.0, 64.0)
    pde.setEquationExpression(PDEExpressionType.FINITE_DIFFERENCES,
                               'eps(i+1/2,j)*(u(i+1,j)-u(i,j)) - eps(i-1/2,j)*(u(i,j)-u(i-1,j)) + ' + \
                               'eps(i,j+1/2)*(u(i,j+1)-u(i,j)) - eps(i,j-1/2)*(u(i,j)-u(i,j-1))')
    pde.setAuxiliaryFunctions({'eps':eps})
    pde.configureGrid()
    return pde

def setupPDE_vector_calculus():
    # 3. Equation as vector calculus without aux function eps
    pde = PDE(64.0, 64.0)
    pde.setEquationExpression(PDEExpressionType.VECTOR_CALCULUS, "div(grad( u(r) ))")
    pde.setVectorVariable("r", dimension=2)
    pde.configureGrid()
    return pde

def setupPDE_vector_calculus_with_eps():
    # 4. Equation as vector calculus with aux function eps
    pde = PDE(64.0, 64.0)
    pde.setEquationExpression(PDEExpressionType.VECTOR_CALCULUS, "div(eps(r) * grad( u(r) ))")
    pde.setVectorVariable("r", dimension=2)
    pde.setAuxiliaryFunctions({'eps': eps})
    pde.configureGrid()
    return pde


if __name__ == '__main__':

    pdeNr = 4

    pde = None

    if pdeNr == 1:
        pde = setupPDE_finite_differences()
    elif pdeNr == 2:
        pde = setupPDE_finite_differences_with_eps()
    elif pdeNr == 3:
        pde = setupPDE_vector_calculus()
    elif pdeNr == 4:
        pde = setupPDE_vector_calculus_with_eps()
    else:
        raise Exception("Invalid pdeNr:"+pdeNr)

    charges = make_central_charge(pde.geometry)
    resulting_matrix = pde.solve(charges)

    showGraph = 1

    if showGraph:
        plotSurface(pde.geometry.X, pde.geometry.Y, resulting_matrix)
        plt.show()
