from keras.models import model_from_json
import numpy as np

from sources.experiments.charges_generators import make_single_charge
from sources.pdesolver.finite_differences_method.geometry import Geometry
from sources.pdesolver.finite_differences_method.rectangle import Rectangle

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider

def plotMatrixList(g, matrices):
    fig = plt.figure()
    axes = []
    for i,m in enumerate(matrices):
        ax = fig.add_subplot(1, len(matrices), i+1, projection='3d')
        axes.append(ax)
        ax.plot_surface(g.X, g.Y, m, cmap=cm.coolwarm,
                    linewidth=0, antialiased=False)
    plt.show()

def calc_charge_weight_matrix_orig(geometry, charges):
    matrix = np.zeros(shape=(len(geometry.Y)*2,len(geometry.X)*2))
    firstCharge = charges.chargesList[0]
    x = firstCharge[0]+16.0
    y = firstCharge[1]+16.0
    for row in range(0, geometry.numY*2):
        for col in range(0, geometry.numX*2):
            matrix[row, col] = np.sqrt( (x-col)**2 + (y-row)**2 )

    return matrix

def calc_charge_weight_matrix(geometry, charges):
    matrix = np.zeros(shape=(len(geometry.Y)*2,len(geometry.X)*2))
    firstCharge = charges.chargesList[0]
    x = firstCharge[0]+16.0
    y = firstCharge[1]+16.0
    for row in range(0, geometry.numY*2):
        for col in range(0, geometry.numX*2):
            matrix[row, col] = 1./(1.+np.sqrt( (x-col)**2 + (y-row)**2 ))

    return matrix


def loadModel(baseFilename):
    json_file = open(baseFilename+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    #  load weights into new model
    loaded_model.load_weights(baseFilename+'.h5')
    return loaded_model

def makeGeometry():
    delta = 1.0
    gridWidth = 32.0
    gridHeight = 32.0
    rect = Rectangle(0, 0, gridWidth, gridHeight)
    g = Geometry(rect, delta)
    return g

class UICallback(object):
    def __init__(self, g, model):
        self.g = g
        self.model = model

    def setupAxes(self):
        fig = plt.figure()
        self.ax_input = fig.add_subplot(1, 2, 1, projection='3d')
        self.ax_result = fig.add_subplot(1, 2, 2, projection='3d')

    def setupSliders(self, updateFn):
        axcolor = 'lightgoldenrodyellow'
        delta_f = 1.0
        horzAxes = plt.axes([0.25, 0.05, 0.65, 0.03], facecolor=axcolor)
        self.horzSlider = Slider(horzAxes, 'X', 0.0, 32.0, valinit=16.0, valstep=delta_f)
        self.horzSlider.on_changed(updateFn)
        vertAxes = plt.axes([0.25, 0.02, 0.65, 0.03], facecolor=axcolor)
        self.vertSlider = Slider(vertAxes, 'Y', 0.0, 32.0, valinit=16.0, valstep=delta_f)
        self.vertSlider.on_changed(updateFn)


    def update(self):
        charge = self.calcCharges()
        inputMatrix = self.setupInputMatrix(self.g, charge)
        resultMatrix = self.predict(inputMatrix)
        self.plotMatrices(inputMatrix, resultMatrix)

    def calcCharges(self):
        horizontalRelChargePos = self.horzSlider.val / 32.0
        verticalRelChargePos = self.vertSlider.val / 32.0
        charge = make_single_charge(g, horizontalRelChargePos, verticalRelChargePos, -10.0)
        return charge

    def setupInputMatrix(self, g, charge):
        charges_weight_matrix = calc_charge_weight_matrix_orig(g, charge)
        charges = np.zeros(shape=(1, g.numX * 2, g.numY * 2))
        charges[0] = charges_weight_matrix
        c = charges.reshape((1, 64, 64, 1))
        c = c / np.max(c)
        return c

    def predict(self, inputMatrix):
        predictedFunction = self.model.predict(inputMatrix)
        return predictedFunction

    def plotMatrices(self, inputMatrix, predictedMatrix):
        self.ax_input.cla()
        self.ax_input.plot_surface(g.X, g.Y, inputMatrix[0,16:48,16:48,0], cmap=cm.coolwarm,
                            linewidth=0, antialiased=False)
        self.ax_result.cla()
        self.ax_result.plot_surface(g.X, g.Y, predictedMatrix[0,:,:,0], cmap=cm.coolwarm,
                                   linewidth=0, antialiased=False)


if __name__ == '__main__':

    model = loadModel('pde-charges-model_orig')
    g = makeGeometry()

    #charge = make_single_charge(g, 0.5, 0.2, -10)
    #charges_weight_matrix = calc_charge_weight_matrix_orig(g, charge)

    #charges = np.zeros(shape=(1, g.numX * 2, g.numY * 2))
    #charges[0] = charges_weight_matrix
    #c = charges.reshape((1, 64, 64, 1))
    #c = c / np.max(c)

    #predictedFunctions = model.predict(c)

    callback = UICallback(g, model)

    def updateCallback(arg):
        callback.update()

    callback.updateFn = updateCallback
    callback.setupAxes()
    callback.setupSliders(updateCallback)


    callback.update()

    plt.show()
