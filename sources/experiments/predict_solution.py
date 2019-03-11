from keras.models import model_from_json
import numpy as np

from sources.experiments.calc_charge_matrix import solvePDE
from sources.experiments.charges_generators import make_single_charge, make_n_fold_charge
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

def calc_charge_weight_matrix(geometry, charges, index=0):
    matrix = np.zeros(shape=(len(geometry.Y)*2,len(geometry.X)*2))
    firstCharge = charges.chargesList[index]
    x = firstCharge[0]+geometry.numX/2
    y = firstCharge[1]+geometry.numX/2
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

def makeGeometry(gridWidth, gridHeight):
    delta = 1.0
    gridWidth = gridWidth
    gridHeight = gridHeight
    rect = Rectangle(0, 0, gridWidth, gridHeight)
    g = Geometry(rect, delta)
    return g

class UICallback(object):
    def __init__(self, g, model):
        self.g = g
        self.gridWidth = g.numX
        self.gridHeight = g.numY
        self.model = model

    def setupAxes(self):
        fig = plt.figure()
        self.ax_input = fig.add_subplot(1, 2, 1, projection='3d')
        self.ax_input.set_title('Numeric solution')
        self.ax_prediction = fig.add_subplot(1, 2, 2, projection='3d')
        self.ax_prediction.set_title('Predicted solution with ML')

    def setupSliders(self, updateFn):
        axcolor = 'lightgoldenrodyellow'
        delta_f = 1.0
        horzAxes1 = plt.axes([0.25, 0.06, 0.65, 0.03], facecolor=axcolor)
        self.horzSlider1 = Slider(horzAxes1, 'X', 0.0, self.gridWidth, valinit=16.0, valstep=delta_f)
        self.horzSlider1.on_changed(updateFn)
        vertAxes1 = plt.axes([0.25, 0.02, 0.65, 0.03], facecolor=axcolor)
        self.vertSlider1 = Slider(vertAxes1, 'Y', 0.0, self.gridHeight, valinit=16.0, valstep=delta_f)
        self.vertSlider1.on_changed(updateFn)

        horzAxes2 = plt.axes([0.25, 0.14, 0.65, 0.03], facecolor=axcolor)
        self.horzSlider2 = Slider(horzAxes2, 'X', 0.0, self.gridWidth, valinit=16.0, valstep=delta_f)
        self.horzSlider2.on_changed(updateFn)
        vertAxes2 = plt.axes([0.25, 0.10, 0.65, 0.03], facecolor=axcolor)
        self.vertSlider2 = Slider(vertAxes2, 'Y', 0.0, self.gridHeight, valinit=16.0, valstep=delta_f)
        self.vertSlider2.on_changed(updateFn)


    def update(self):
        charge = self.calcCharges()
        inputMatrix = self.setupInputMatrix(self.g, charge)
        resultMatrix = self.predict(inputMatrix)
        self.plotMatrices(resultMatrix)

    def calcCharges(self):
        horizontalRelChargePos = []
        verticalRelChargePos = []

        horizontalRelChargePos.append(self.horzSlider1.val / self.gridWidth)
        verticalRelChargePos.append(self.vertSlider1.val / self.gridHeight)
        horizontalRelChargePos.append(self.horzSlider2.val / self.gridWidth)
        verticalRelChargePos.append(self.vertSlider2.val / self.gridHeight)

        charge = make_n_fold_charge(g, horizontalRelChargePos, verticalRelChargePos, 0, 2, -10.0)

        #charge = make_single_charge(g, horizontalRelChargePos, verticalRelChargePos, -10.0)
        return charge

    def setupInputMatrix(self, g, charge):
        charges_weight_matrix = []
        chargesCount = 2
        for channel in range(0, chargesCount):
            charges_weight_matrix.append(calc_charge_weight_matrix(g, charge, channel))

        #charges_weight_matrix = calc_charge_weight_matrix(g, charge)
        #charges = np.zeros(shape=(1, g.numX * 2, g.numY * 2))
        #charges[0] = charges_weight_matrix
        charges_stacked = np.stack(charges_weight_matrix, axis=-1)
        charges = np.zeros(shape=(1, g.numX * 2, g.numY * 2, chargesCount))
        charges[0] = charges_stacked

        c = charges.reshape((1, int(self.gridWidth*2), int(self.gridHeight*2), chargesCount))
        c = c / np.max(c)
        return c

    def predict(self, inputMatrix):
        predictedFunction = self.model.predict(inputMatrix)
        return predictedFunction

    def plotMatrices(self, predictedMatrix):

        result_matrix = solvePDE(g, self.calcCharges())
        result_matrix /= np.max(result_matrix)
        predictedMatrix /= np.max(predictedMatrix)

        self.ax_input.cla()
        self.ax_input.set_title('Numeric solution')
        self.ax_input.plot_surface(g.X, g.Y, result_matrix, cmap=cm.coolwarm,
                            linewidth=0, antialiased=False)
        self.ax_prediction.cla()
        self.ax_prediction.set_title('Predicted solution with ML')
        self.ax_prediction.plot_surface(g.X, g.Y, predictedMatrix[0,:,:,0], cmap=cm.coolwarm,
                                   linewidth=0, antialiased=False)

        maxZ = np.max(result_matrix)

        self.ax_input.set_zlim(0.0, 1.1)
        self.ax_prediction.set_zlim(0.0, 1.1)



if __name__ == '__main__':

    #model = loadModel('pde-charges-model_orig')
    model = loadModel('model64x64-2charges-v2')
    g = makeGeometry(64,64)

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
