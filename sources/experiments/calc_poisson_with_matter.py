
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from matplotlib.widgets import Button

from sources.experiments.data_generation.data_scenario1 import make_test_set, make_representation
from sources.experiments.ellipsis_data_support.make_ellipsis import plot_ellipsis
from sources.experiments.fdm_helper import plotSurface, plotSurface_subplot
from sources.pdesolver.finite_differences_method.FiniteDifferencesSolver_V2 import FiniteDifferencesMethod3, \
    GridConfiguration, ConstantGridValueProvider, FunctionGridValueProvider
from sources.pdesolver.finite_differences_method.boundaryconditions import RectangularBoundaryCondition
from sources.pdesolver.finite_differences_method.charge_distribution import ChargeDistribution
from sources.pdesolver.finite_differences_method.geometry import Geometry
from sources.pdesolver.finite_differences_method.rectangle import Rectangle

def make_charges_in_line(g, count, charge, startX, startY, endX, endY):
    charges = ChargeDistribution(g)
    deltaX = (endX-startX)/count
    deltaY = (endY-startY)/count
    for i in range(count):
        charges.add((int)(startX + i * deltaX), (int)(startY + i*deltaY), charge)

    return charges

#def eps(i, j):
#    #if i==0 and j==0:
#    #    return -200.0
    #if i == 2 and j==2:
    #    return 10.0
    #if i==1 and j==2:
    #    return -20.0
    #else:
#    return 1.0


def make_finite_differences_poisson_equation_in_matter(eps):
    gridConfig = GridConfiguration()
    gridConfig.add(FunctionGridValueProvider((lambda i, j: 0.5 * (eps(i, j) + eps(i + 1, j)))), 1, 0)
    gridConfig.add(FunctionGridValueProvider((lambda i, j: 0.5 * (eps(i, j) + eps(i - 1, j)))), -1, 0)
    gridConfig.add(FunctionGridValueProvider((lambda i, j: 0.5 * (eps(i, j) + eps(i, j + 1)))), 0, 1)
    gridConfig.add(FunctionGridValueProvider((lambda i, j: 0.5 * (eps(i, j) + eps(i, j - 1)))), 0, -1)
    gridConfig.add(FunctionGridValueProvider((lambda i, j:
                                               -0.5 * (4 * eps(i, j) + eps(i + 1, j) + eps(i - 1, j) + eps(i,j + 1) + eps(
                                                   i, j - 1)))), 0, 0)
    return gridConfig

def load_test_set(count):
    test_set = make_test_set(count)
    permeability_matrix = make_permeability_matrix(test_set)

    return (test_set, permeability_matrix)

def make_permeability_matrix(test_set):
    count = len(test_set['majorSemiAxis'])
    permeability_matrix = []
    for i in range(count):
        permeability_matrix.append(
            make_representation(i + 1, 64, 64, 32, 32, test_set['majorSemiAxis'][i], test_set['minorSemiAxis'][i],
                                test_set['permeabilities'][i], test_set['angles'][i]))
    #print(permeability_matrix)
    return permeability_matrix

def generate_permability_function(index, permeability_matrix):

    def eps(i,j):
        if (len(permeability_matrix[index]) <= i):
            return 1.0
        elif (len(permeability_matrix[index][i]) <= j):
            return 1.0

        if (permeability_matrix[index][i,j] == 0.0):
            return 1.0
        else:
            return permeability_matrix[index][i,j]

    return eps

class Index(object):

    def __init__(self, count, current_index, permability_data, permeability_axes, surface_axes, error_axes, generate_finite_differences_solver):
        self.count = count
        self.current_index = current_index
        self.permeability_data = permeability_data
        self.permeability_axes = permeability_axes
        self.surface_axes = surface_axes
        self.error_axes = error_axes
        self.colorbar = 0
        self.cmap = plt.get_cmap('PiYG')
        self.generate_finite_differences_solver = generate_finite_differences_solver


    def next(self, event):
        if self.current_index+1 < count:
            self.update(self.current_index+1)

    def prev(self, event):
        if self.current_index-1 >= 0:
            self.update(self.current_index-1)

    def update(self, new_index):
        self.current_index = new_index
        self.redraw_permeability()
        fdm = self.recalc_finite_differences(self.current_index)
        self.redraw_surface(fdm)
        self.redraw_errors(fdm)
        plt.draw()

    def recalc_finite_differences(self, index):
        fdm = self.generate_finite_differences_solver(index)
        fdm.solve()
        fdm.calcMetrices()
        return fdm

    def redraw_surface(self, fdm):
        self.surface_axes.cla()
        plotSurface_subplot(self.surface_axes, fdm.geometry.X, fdm.geometry.Y, fdm.values)

    def redraw_permeability(self):
        self.permeability_axes.cla()
        pdata = self.permeability_data[0]
        permeability_title = 'id = {0}, eps = {1}, \naxisMaj = {2}, axisMin = {3},\n angle={4:6.4f} '.format(self.current_index, pdata['permeabilities'][self.current_index], pdata['majorSemiAxis'][self.current_index], pdata['minorSemiAxis'][self.current_index], pdata['angles'][self.current_index])
        print(permeability_title)
        plot_ellipsis(self.permeability_axes, self.permeability_data[1][self.current_index], permeability_title)
        plt.draw()

    def redraw_errors(self, fdm):
        self.error_axes.cla()
        if self.colorbar != 0:
            self.colorbar.remove()
            self.colorbar = 0
        levels = MaxNLocator(nbins=20).tick_values(fdm.minValue-2.0, fdm.maxValue+2.0)
        norm = BoundaryNorm(levels, ncolors = self.cmap.N, clip=True)
        im = self.error_axes.pcolormesh(fdm.geometry.X, fdm.geometry.Y, fdm.error, norm = norm, cmap=self.cmap)
        self.error_axes.set_title('Errors', fontsize=9)
        self.colorbar = fig.colorbar(im, ax=self.error_axes)



if __name__ == '__main__':

    np.set_printoptions(threshold=np.nan)

    count = 60
    index = 0

    # setup for finite differences
    delta = 1.0
    rect = Rectangle(0, 0, 64.0, 64.0)
    g = Geometry(rect, delta)
    boundaryCondition = RectangularBoundaryCondition(g)
    charges = make_charges_in_line(g, 11, -10.0, 16.0, 20.0, 48.0, 20.0)
    #charges = make_charges_in_line(g, 32, -10.0, 16.0, 20.0, 48.0, 20.0)

    # setup for plot
    fig = plt.figure()


    permeability_data = load_test_set(count)
    permeability_title = 'eps = {0}'.format(permeability_data[0]['permeabilities'][index])
    permeability_axes = fig.add_subplot(1, 3, 1)
    surface_axes = fig.add_subplot(1, 3, 2, projection='3d')
    error_axes = fig.add_subplot(1, 3, 3)


    def generate_finite_differences_solver(index):

        eps = generate_permability_function(index, permeability_data[1])
        gridConfig = make_finite_differences_poisson_equation_in_matter(eps)

        fdm = FiniteDifferencesMethod3(g, boundaryCondition, gridConfig, charges)
        return fdm


    callback = Index(count, 0, permeability_data, permeability_axes, surface_axes, error_axes, generate_finite_differences_solver)
    callback.update(0)


    #plot_ellipsis(permeability_axes, permeability_data[1][0], permeability_title)
    #xData = []
    #yData = []

    #for x in range(0, 64):
    #    for y in range(0, 64):
    #        item = permeability_data[1][0].item((x, y))
    #        if item > 0:
    #            xData.append(x)
    #            yData.append(y)

    #plt.scatter(xData, yData)
    #plt.axis([0, 64, 0, 64])

    #plt.show()
    #





    #plotSurface_subplot(surface_axes, fdm.geometry.X, fdm.geometry.Y, fdm.values)


    axnext = plt.axes([0.81, 0.0, 0.1, 0.075])
    bnext = Button(axnext, 'Next')
    bnext.on_clicked(callback.next)

    axprev = plt.axes([0.7, 0.0, 0.1, 0.075])
    bprev = Button(axprev, 'Prev')
    bprev.on_clicked(callback.prev)

    plt.show()


    print('ende')