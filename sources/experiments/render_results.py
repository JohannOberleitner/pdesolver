import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from matplotlib.widgets import Button

from sources.experiments.data_generation.results_data import as_ResultsSet

class UICallback(object):

    def __init__(self, permittivity_axes, figure1, figure2, current_index):

            if figure1[1] != None:
                self.figure1 = figure1
                self.figure1_axes = figure1[0]
                self.figure1_results = figure1[1]
                #self.ndValues = np.array(self.figure1_results.resultValues[current_index])
                x = np.linspace(0.0, 64.0, 64)
                y = np.linspace(0.0, 64.0, 64)

                #x = np.linspace(0.0, 32.0, 32)
                #y = np.linspace(0.0, 32.0, 32)

                self.figure1_X, self.figure1_Y = np.meshgrid(x,y)
            else:
                self.figure1 = None

            if figure2[1] != None:
                self.figure2 = figure2
                self.figure2_axes = figure2[0]
                self.figure2_results = figure2[1]
            else:
                self.figure2 = None

            self.current_index = current_index
            self.count = self.figure1_results.count()

            self.colorbar = 0
            self.cmap = plt.get_cmap('PiYG')


    def next(self, event):
        if self.current_index+1 < self.count:
            self.update(self.current_index + 1)

    def prev(self, event):
        if self.current_index - 1 >= 0:
            self.update(self.current_index - 1)

    def plotSurface_subplot(self, axes, x, y, values):
        axes.set_zlim(-5.0, 40.0)
        surf = axes.plot_surface(x, y, values, cmap=cm.coolwarm,
                                         linewidth=0, antialiased=False)

    def redraw_figure(self, results_axes, ndValues, x, y):
        results_axes.cla()
        self.plotSurface_subplot(results_axes, x, y, ndValues)

    def update(self, new_index):
        self.current_index = new_index
        #self.redraw_permittivity()
        self.redraw_figure1()
        self.redraw_figure2()
        plt.draw()

    def redraw_figure1(self):
        if self.figure1 != None:
            self.figure1_values = np.array(self.figure1_results.resultValues[self.current_index])
            self.redraw_figure(self.figure1_axes, self.figure1_values, self.figure1_X, self.figure1_Y)

    def redraw_figure2(self):
        if self.figure2 != None:
            self.figure2_values = np.array(self.figure2_results.resultValues[self.current_index])
            self.redraw_figure(self.figure2_axes, self.figure2_values, self.figure1_X, self.figure1_Y)

    # def redraw_permittivity(self):
    #         self.permittivity_axes.cla()
    #         pdata = self.permittivity_data[0]
    #         permittivity_title = 'id = {0}, eps = {1}, \naxisMaj = {2}, axisMin = {3},\n angle={4:6.4f} '.format(
    #             self.current_index, pdata['permittivities'][self.current_index],
    #             pdata['majorSemiAxis'][self.current_index], pdata['minorSemiAxis'][self.current_index],
    #             pdata['angles'][self.current_index])
    #         print(permittivity_title)
    #         plot_ellipsis(self.permittivity_axes, self.permittivity_data[1][self.current_index], permittivity_title)
    #         plt.draw()
    #
    # def redraw_errors(self, fdm):
    #         self.error_axes.cla()
    #         if self.colorbar != 0:
    #             self.colorbar.remove()
    #             self.colorbar = 0
    #         levels = MaxNLocator(nbins=20).tick_values(fdm.minValue - 2.0, fdm.maxValue + 2.0)
    #         norm = BoundaryNorm(levels, ncolors=self.cmap.N, clip=True)
    #         im = self.error_axes.pcolormesh(fdm.geometry.X, fdm.geometry.Y, fdm.error, norm=norm, cmap=self.cmap)
    #         self.error_axes.set_title('Errors', fontsize=9)
    #         self.colorbar = fig.colorbar(im, ax=self.error_axes)


def loadResultsFile(file):
    if file == None:
        return None
    file = open(file, mode='r')
    results = json.load(file, object_hook=as_ResultsSet)
    return results


def parseArguments(argv):
    supportedOptions = "hp:"
    supportLongOptions = []
    usage = 'render_results.py <inputfile1> <inputfile2>'

    inputFile1 = None
    inputFile2 = None

    if len(argv) == 0:
        print(usage)
        sys.exit(2)

    if len(argv) >= 1:
        inputFile1 = argv[0]

    if len(argv) >= 2:
        inputFile2 = argv[1]

    return inputFile1, inputFile2


if __name__ == '__main__':
    inputFileName1, inputFileName2 = parseArguments(sys.argv[1:])
    print(inputFileName1, inputFileName2)

    resultsFile1 = loadResultsFile(inputFileName1)
    resultsFile2 = loadResultsFile(inputFileName2)
    #permittivities = loadPermittivities

    fig = plt.figure()

    permittivity_axes = fig.add_subplot(1, 3, 1)
    results1_axes = fig.add_subplot(1, 3, 2, projection='3d')
    results2_axes = fig.add_subplot(1, 3, 3, projection='3d')

    callback = UICallback(permittivity_axes, (results1_axes, resultsFile1), (results2_axes, resultsFile2), 0)
    callback.update(0)

    axnext = plt.axes([0.81, 0.0, 0.1, 0.075])
    bnext = Button(axnext, 'Next')
    bnext.on_clicked(callback.next)

    axprev = plt.axes([0.7, 0.0, 0.1, 0.075])
    bprev = Button(axprev, 'Prev')
    bprev.on_clicked(callback.prev)

    plt.show()