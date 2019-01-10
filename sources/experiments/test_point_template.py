

import numpy as np
from sources.experiments.charges_generators import make_central_charge
from sources.experiments.fdm_helper import plotSurface, solve_finite_differences
from sources.pdesolver.finite_differences_method.FiniteDifferencesSolver_V2 import FiniteDifferencesMethod3, \
    GridConfiguration, ConstantGridValueProvider, FunctionGridValueProvider
from sources.pdesolver.finite_differences_method.boundaryconditions import RectangularBoundaryCondition
from sources.pdesolver.finite_differences_method.charge_distribution import ChargeDistribution
from sources.pdesolver.finite_differences_method.geometry import Geometry
from sources.pdesolver.finite_differences_method.rectangle import Rectangle


def eps(i, j):
    if i==0 and j==0:
        return -200.0
    if i == 2 and j==2:
        return 10.0
    if i==1 and j==2:
        return -20.0
    else:
        return 1.0


if __name__ == '__main__':

    np.set_printoptions(threshold=np.nan)

    delta = 1.0
    rect = Rectangle(0, 0, 3.0, 3.0)

    g = Geometry(rect, delta)

    boundaryCondition = RectangularBoundaryCondition(g)

    charges = make_central_charge(g)

    gridConfig = GridConfiguration()

    gridConfig.add(ConstantGridValueProvider(-4.0), 0,0)
    gridConfig.add(ConstantGridValueProvider(1.0), 1, 0)
    gridConfig.add(ConstantGridValueProvider(1.0), -1, 0)
    gridConfig.add(ConstantGridValueProvider(1.0), 0, 1)
    gridConfig.add(ConstantGridValueProvider(1.0), 0, -1)

    fdm = FiniteDifferencesMethod3(g, boundaryCondition, gridConfig, charges)

    fdm.solve()
    fdm.printMatrix()

    #f = (lambda i, j:
    #     -0.5 * 2 * eps(i,j))

    # -0.5 * (4 * eps(i, j) + eps(i + 1, j) + eps(i - 1, j) + eps(i, j + 1) + eps(i, j - 1)))

    #print(eps(0,0))
    #print(f(0,0))


    gridConfig2 = GridConfiguration()
    gridConfig2.add(FunctionGridValueProvider((lambda i, j: 0.5*(eps(i,j)+eps(i+1,j)))),1,0)
    gridConfig2.add(FunctionGridValueProvider((lambda i, j: 0.5*(eps(i,j)+eps(i-1,j)))),-1,0)
    gridConfig2.add(FunctionGridValueProvider((lambda i, j: 0.5*(eps(i,j)+eps(i,j+1)))),0,1)
    gridConfig2.add(FunctionGridValueProvider((lambda i, j: 0.5*(eps(i,j)+eps(i,j-1)))),0,-1)
    gridConfig2.add(FunctionGridValueProvider((lambda i, j:
                                               -0.5*(4*eps(i,j)+eps(i+1,j)+eps(i-1,j)+eps(i,j+1)+eps(i,j-1)))),0,0)

    fdm2 = FiniteDifferencesMethod3(g, boundaryCondition, gridConfig2, charges)

    fdm2.solve()
    fdm2.printMatrix()