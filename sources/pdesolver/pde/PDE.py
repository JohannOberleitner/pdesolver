import time
from enum import Enum

from sources.pdesolver.finite_differences_method.FiniteDifferencesSolver_V2 import FiniteDifferencesMethod4
from sources.pdesolver.finite_differences_method.boundaryconditions import RectangularBoundaryCondition
from sources.pdesolver.finite_differences_method.geometry import Geometry
from sources.pdesolver.finite_differences_method.rectangle import Rectangle
from sources.pdesolver.formula_parser.lexer import Lexer
from sources.pdesolver.formula_parser.parser import Parser
from sources.pdesolver.pde.FiniteDifferencesVisitor import FiniteDifferencesVisitor
from sources.pdesolver.pde.VectorCalculusExpressionVisitor import VectorCalculusExpressionVisitor


class PDEExpressionType(Enum):
    NONE = 0
    FINITE_DIFFERENCES = 1
    VECTOR_CALCULUS = 2

class PDE:
    """
    A class that models PDEs (partial differential equations) that can be solved with
    the finite differences method.
    """

    def __init__(self, gridWidth, gridHeight):
        self.gridWidth = gridWidth
        self.gridHeight = gridHeight
        self.delta = 1.0
        self.rect = Rectangle(0, 0, gridWidth, gridHeight)
        self.geometry = Geometry(self.rect, self.delta)
        self.boundaryCondition = RectangularBoundaryCondition(self.geometry)
        self.auxiliaryFunctions = {}

    def setEquationExpression(self, expressionType, expressionString):
        self.expressionType = expressionType
        lexer = Lexer(expressionString)
        l = list(lexer.parse())
        parser = Parser(l)
        self.expression = parser.parse()

    def setVectorVariable(self, vectorVariableName, dimension=2):
        if self.expressionType != PDEExpressionType.VECTOR_CALCULUS:
            raise Exception("Expression type must be set to VECTOR_EXPRESSION")

        self.vectorVariableName = vectorVariableName
        self.dimension = dimension

    def setAuxiliaryFunctions(self, functionDictionary):
        self.auxiliaryFunctions = functionDictionary

    def configureGrid(self):
        if self.expressionType == PDEExpressionType.NONE:
            raise Exception("Expression not set")

        if self.expressionType == PDEExpressionType.VECTOR_CALCULUS:
            finiteDifferencesExpression = self.evaluateVectorCalculusExpression(self.expression)
            self.configureFiniteDifferences(finiteDifferencesExpression)
        else:
            self.configureFiniteDifferences(self.expression)

    def evaluateVectorCalculusExpression(self, vectorCalculusExpression):
        visitor = VectorCalculusExpressionVisitor([self.vectorVariableName], self.dimension)
        vectorCalculusExpression.accept(visitor)
        finiteDifferencesExpression = visitor.get_result()
        return finiteDifferencesExpression

    def configureFiniteDifferences(self, finiteDifferencesExpression):
        visitor = FiniteDifferencesVisitor()
        finiteDifferencesExpression.accept(visitor)
        visitor.combineExpressions()
        visitor.simplifyExpressions()
        self.gridConfig = visitor.make_grid_config(self.auxiliaryFunctions)


    def solve(self, rightSide):

        start = time.time()

        fdm = FiniteDifferencesMethod4(self.geometry, self.boundaryCondition, self.gridConfig, rightSide)
        fdm.solve()

        resulting_matrix = fdm.values

        self.duration = time.time() - start
        #print('Total duration for solving the PDE lasted {0} sec'.format(duration))
        return resulting_matrix
