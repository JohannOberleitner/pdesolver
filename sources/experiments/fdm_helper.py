
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm

from sources.pdesolver.finite_differences_method.FiniteDifferencesSolver_V2 import FiniteDifferencesMethod2


def plotSurface(x,y,values):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(x, y, values, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)


def solve_finite_differences(g, boundaryCondition, charges):
    fdm = FiniteDifferencesMethod2(g, boundaryCondition, charges)
    fdm.solve()
    return fdm
