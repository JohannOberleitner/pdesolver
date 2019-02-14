import time
import matplotlib.pyplot as plt

from sources.experiments.charges_generators import make_quadrupol
from sources.experiments.fdm_helper import plotSurface, solve_finite_differences, makeStandardPDEConfig, \
    solve_finite_differences2
from sources.pdesolver.finite_differences_method.boundaryconditions import RectangularBoundaryCondition
from sources.pdesolver.finite_differences_method.charge_distribution import ChargeDistribution
from sources.pdesolver.finite_differences_method.geometry import Geometry
from sources.pdesolver.finite_differences_method.rectangle import Rectangle

def make_zikzak_charges(g):
    charges = ChargeDistribution(g)
    charge = -1000.0
    charges.add((int)(g.numX/2.), (int)(g.numY/3.), charge)
    charges.add((int)(g.numX/7.*2), (int)(g.numY/3.*2), -charge)
    charges.add((int)(g.numX/7.*4), (int)(g.numY/3.*2), -charge)
    charges.add((int)(g.numX/7.*1), (int)(g.numY/3.), charge)
    charges.add((int)(g.numX/7.*6), (int)(g.numY/3.), charge)
    return charges


if __name__ == '__main__':

    delta = 1.0
    rect = Rectangle(0, 0, 32.0, 16.0)

    g = Geometry(rect, delta)

    boundaryCondition = RectangularBoundaryCondition(g)
    gridConfig = makeStandardPDEConfig()
    charges = make_zikzak_charges(g)

    start = time.clock()
    duration = time.clock() - start

    fdm = solve_finite_differences2(g, boundaryCondition, gridConfig, charges)

    print(duration)

    showGraph = 1

    if showGraph:
        plotSurface(fdm.geometry.X, fdm.geometry.Y, fdm.values)
        plt.show()