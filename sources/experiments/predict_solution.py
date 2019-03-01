from keras.models import model_from_json
import numpy as np

from sources.experiments.charges_generators import make_single_charge
from sources.pdesolver.finite_differences_method.geometry import Geometry
from sources.pdesolver.finite_differences_method.rectangle import Rectangle

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

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

if __name__ == '__main__':

    model = loadModel('pde-charges-model')
    g = makeGeometry()

    charge = make_single_charge(g, 0.5, 0.2, -10)
    charges_weight_matrix = calc_charge_weight_matrix(g, charge)

    charges = np.zeros(shape=(1, g.numX * 2, g.numY * 2))
    charges[0] = charges_weight_matrix
    c = charges.reshape((1, 64, 64, 1))
    c = c / np.max(c)

    predictedFunctions = model.predict(c)

    plotMatrixList(g, [c[0,16:48,16:48,0], predictedFunctions[0,:,:,0]])