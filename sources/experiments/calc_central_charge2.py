import time
import matplotlib.pyplot as plt

from sources.experiments.charges_generators import make_central_charge
from sources.experiments.fdm_helper import plotSurface, solve_finite_differences, solve_finite_differences2, \
    makeStandardPDEConfig
from sources.pdesolver.finite_differences_method.boundaryconditions import RectangularBoundaryCondition
from sources.pdesolver.finite_differences_method.charge_distribution import ChargeDistribution
from sources.pdesolver.finite_differences_method.geometry import Geometry
from sources.pdesolver.finite_differences_method.rectangle import Rectangle
from sources.pdesolver.formula_parser.lexer import Lexer
from sources.pdesolver.formula_parser.parser import Parser


def make_pde_config(expression):
    # expr = 'f(i,2)'

    # div(eps(r)*nabla(u(r)))

    #expr = 'eps(i+1/2,j)*(u(i+1,j)-u(i,j)) - eps(i-1/2,j)*(u(i,j)-u(i-1,j))'
    expr = '(u(i+1,j)-u(i,j)) - (u(i,j)-u(i-1,j)) + (u(i,j+1)-u(i,j)) - (u(i,j)-u(i,j-1))'

    lexer = Lexer(expr)
    l = list(lexer.parse())
    parser = Parser(l)
    expr = parser.parse()

    return expr

if __name__ == '__main__':

    delta = 1.0
    rect = Rectangle(0, 0, 64.0, 64.0)

    g = Geometry(rect, delta)
    print(g.numX, g.numY)

    boundaryCondition = RectangularBoundaryCondition(g)
    gridConfig = make_pde_config('u(i+1,j)-u(i,j)-(u(i,j)-u(i-1,j))')

    charges = make_central_charge(g)

    start = time.clock()
    duration = time.clock() - start

    #fdm = solve_finite_differences(g, boundaryCondition, charges)
    fdm = solve_finite_differences2(g, boundaryCondition, gridConfig, charges)

    print(duration)



    showGraph = 1

    if showGraph:
        plotSurface(fdm.geometry.X, fdm.geometry.Y, fdm.values)
        plt.show()
