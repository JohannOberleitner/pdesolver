
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm

from sources.pdesolver.finite_differences_method.FiniteDifferencesSolver_V2 import FiniteDifferencesMethod2, \
    FiniteDifferencesMethod4, GridConfiguration, ConstantGridValueProvider


def plotSurface(x,y,values):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(x, y, values, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)


def plotSurface_subplot(axes,x,y,values):

    axes.set_zlim(-2.0, 10.0)
    surf = axes.plot_surface(x, y, values, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

def makeStandardPDEConfig():
    # this encodes the linearized PDE: -4u(i,j)+u(i+1,j)+u(i-1,j)+u(i,j+1)+u(i,j-1) = -rho(i,j)
    # u(i+1,j)-u(i,j) - (u(i,j)-u(i-1,j)) + ... (i<->j) = -rho(i,j)
    gridConfig = GridConfiguration()
    gridConfig.add(ConstantGridValueProvider(1.0), 1, 0)
    gridConfig.add(ConstantGridValueProvider(1.0), -1, 0)
    gridConfig.add(ConstantGridValueProvider(1.0), 0, 1)
    gridConfig.add(ConstantGridValueProvider(1.0), 0, -1)
    gridConfig.add(ConstantGridValueProvider(-4.0), 0, 0)
    return gridConfig


def solve_finite_differences(g, boundaryCondition, charges):
    fdm = FiniteDifferencesMethod2(g, boundaryCondition, charges)
    fdm.solve()
    return fdm

def solve_finite_differences2(g, boundaryCondition, gridConfiguration, charges):
    fdm = FiniteDifferencesMethod4(g, boundaryCondition, gridConfiguration, charges)
    fdm.solve()
    return fdm
