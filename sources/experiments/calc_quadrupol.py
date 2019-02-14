import time
import matplotlib.pyplot as plt

from sources.experiments.charges_generators import make_quadrupol
from sources.experiments.fdm_helper import plotSurface, solve_finite_differences, makeStandardPDEConfig, \
    solve_finite_differences2
from sources.pdesolver.finite_differences_method.boundaryconditions import RectangularBoundaryCondition
from sources.pdesolver.finite_differences_method.charge_distribution import ChargeDistribution
from sources.pdesolver.finite_differences_method.geometry import Geometry
from sources.pdesolver.finite_differences_method.rectangle import Rectangle




if __name__ == '__main__':

    delta = 1.0
    rect = Rectangle(0, 0, 64.0, 64.0)

    g = Geometry(rect, delta)
    print(g.numX, g.numY)

    boundaryCondition = RectangularBoundaryCondition(g)
    gridConfig = makeStandardPDEConfig()
    charges = make_quadrupol(g)

    start = time.clock()
    duration = time.clock() - start

    fdm = solve_finite_differences2(g, boundaryCondition, gridConfig, charges)

    print(duration)

    showGraph = 1

    if showGraph:
        plotSurface(fdm.geometry.X, fdm.geometry.Y, fdm.values)
        plt.show()