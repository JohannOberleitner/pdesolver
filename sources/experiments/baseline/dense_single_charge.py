
from sources.experiments.charges_generators import make_central_charge
from sources.experiments.fdm_helper import plotSurface
from sources.pdesolver.pde.PDE import PDEExpressionType, PDE

import matplotlib.pyplot as plt

def setupPDE_vector_calculus(gridSize):

    pde = PDE(gridSize, gridSize)
    pde.setEquationExpression(PDEExpressionType.VECTOR_CALCULUS, "div(grad( u(r) ))")
    pde.setVectorVariable("r", dimension=2)
    pde.configureGrid()
    return pde


if __name__ == '__main__':

    gridSize = 64.0

    print(PDEExpressionType.VECTOR_CALCULUS)

    pde = setupPDE_vector_calculus(gridSize)

    charges = make_central_charge(pde.geometry)
    resulting_matrix = pde.solve(charges)

    showGraph = 1

    if showGraph:
        plotSurface(pde.geometry.X, pde.geometry.Y, resulting_matrix)
        plt.show()

