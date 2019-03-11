import time

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


from sources.pdesolver.finite_differences_method.FiniteDifferencesSolver_V2 import GridConfiguration, \
    ConstantGridValueProvider, FiniteDifferencesMethod4
from sources.pdesolver.finite_differences_method.boundaryconditions import RectangularBoundaryCondition
from sources.pdesolver.finite_differences_method.geometry import Geometry
from sources.experiments.charges_generators import make_single_charge, make_double_charge, make_n_fold_charge
from sources.pdesolver.finite_differences_method.rectangle import Rectangle

from keras import layers, optimizers, losses
from keras import models
import keras.backend as K

def calc_charge_weight_matrix_orig(geometry, charges):
    matrix = np.zeros(shape=(len(geometry.Y)*2,len(geometry.X)*2))
    firstCharge = charges.chargesList[0]
    x = firstCharge[0]+geometry.numX/2
    y = firstCharge[1]+geometry.numX/2
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
            matrix[row, col] = 1./(1.2+np.sqrt( (x-col)**2 + (y-row)**2 ))

    return matrix

def calc_charge_weight_matrix_multi(geometry, charges):
    matrix = np.zeros(shape=(len(geometry.Y)*2,len(geometry.X)*2))

    multiplier = 1
    for charge in charges.chargesList:
        x = charge[0] + geometry.numX/2
        y = charge[1] + geometry.numY/2

        if charge[2] < 0:
            multiplier = 1
        else:
            multiplier = -1

        if multiplier == 1:
            for row in range(0, geometry.numY*2):
                for col in range(0, geometry.numX*2):
                    #matrix[row, col] += np.sqrt((x - col) ** 2 + (y - row) ** 2)
                   matrix[row, col] += 1./(1.2+np.sqrt( (x-col)**2 + (y-row)**2 ))
        else:
            for row in range(0, geometry.numY*2):
                for col in range(0, geometry.numX*2):
                    #matrix[row, col] += np.sqrt((x - col) ** 2 + (y - row) ** 2)
                   matrix[row, col] -= 1./(1.2+np.sqrt( (x-col)**2 + (y-row)**2 ))

        multiplier *= -1


    return matrix


def plotChargesWeightMatrix(matrix):

    plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(g.X, g.Y, matrix, cmap=cm.coolwarm,
                    linewidth=0, antialiased=False)
    plt.show()

def plotMatrixList(g, matrices):
    fig = plt.figure()
    axes = []
    for i,m in enumerate(matrices):
        ax = fig.add_subplot(1, len(matrices), i+1, projection='3d')
        axes.append(ax)
        ax.plot_surface(g.X, g.Y, m, cmap=cm.coolwarm,
                    linewidth=0, antialiased=False)

    # scale second last element to last element
    maxZ = np.max(matrices[-1])
    axes[-2].set_zlim(0.0, maxZ*1.1)

    plt.show()

def create_finite_differences_configuration():
    gridConfig = GridConfiguration()
    gridConfig.add(ConstantGridValueProvider(1.0), 1, 0)
    gridConfig.add(ConstantGridValueProvider(1.0), -1, 0)
    gridConfig.add(ConstantGridValueProvider(1.0), 0, 1)
    gridConfig.add(ConstantGridValueProvider(1.0), 0, -1)
    gridConfig.add(ConstantGridValueProvider(-4.0), 0, 0)

    return gridConfig

def solvePDE(geometry, charges):

    #g = Geometry(rect, delta)
    g = geometry
    boundaryCondition = RectangularBoundaryCondition(geometry)

    index = 1

    start = time.time()

    gridConfig = create_finite_differences_configuration()

    fdm = FiniteDifferencesMethod4(g, boundaryCondition, gridConfig, charges)
    fdm.solve()

    resulting_matrix = fdm.values

    index = index + 1

    duration = time.time() - start
    #print('Total duration for solving {0} PDEs lasted :{1}'.format(inputDataset.count(), duration))

    return resulting_matrix

def saveModel(model, filename):
    model_json = model.to_json()
    with open(filename + '.json', "w") as json_file:
        json_file.write(model_json)
    json_file.close()
    model.save_weights(filename + '.h5')

def learn(input, target):

    count = len(input)
    train_count = int(count * 0.6)
    validation_count = int(count * 0.2)
    test_count = int(count*0.2)
    train_input = input[:train_count]
    train_result = target[:train_count]
    validation_input = input[train_count:train_count+validation_count]
    validation_result = target[train_count:train_count+validation_count]
    test_input = input[train_count+validation_count:]
    test_result = target[train_count+validation_count:]

    channelCount = input.shape[-1]
    height = input.shape[-2]
    width = input.shape[-3]

    # 100 = train=60%+validation=20%+test=20%
    model = models.Sequential()
    #model.add(layers.Conv2D(16, (11, 11), activation='relu', input_shape=(width, height, channelCount)))
    #model.add(layers.Conv2D(32, (11,11), activation='relu'))
    #model.add(layers.Conv2D(64, (5, 5), activation='relu'))
    #model.add(layers.Conv2D(64, (5, 5), activation='relu'))
    #model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    #model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    #model.add(layers.Conv2D(64, (1, 1), activation='relu'))
    #model.add(layers.Conv2D(1, (1, 1), activation='relu'))

    #model.add(layers.Conv2D(16, (5, 5), activation='relu', input_shape=(width, height, channelCount)))
    #model.add(layers.Conv2D(32, (5, 5), activation='relu'))
    #model.add(layers.Conv2D(64, (5, 5), activation='relu'))
    #model.add(layers.Conv2D(64, (5, 5), activation='relu'))
    #model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    #model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    #model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    #model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    #model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    #model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    #model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    #model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    #model.add(layers.Conv2D(64, (1, 1), activation='relu'))
    #model.add(layers.Conv2D(1, (1, 1), activation='relu'))

    #model.add(layers.Conv2D(16, (11, 11), activation='relu', input_shape=(width, height, channelCount)))
    #model.add(layers.Conv2D(32, (9, 9), activation='relu'))
    #model.add(layers.Conv2D(64, (7, 7), activation='relu'))
    #model.add(layers.Conv2D(64, (7, 7), activation='relu'))
    #model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    #model.add(layers.Conv2D(64, (1, 1), activation='relu'))
    #model.add(layers.Conv2D(1, (1, 1), activation='relu'))

    #model.add(layers.Conv2D(16, (31, 31), activation='relu', input_shape=(width, height, channelCount)))
    #model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    #model.add(layers.Conv2D(64, (1, 1), activation='relu'))
    #model.add(layers.Conv2D(1, (1, 1), activation='relu'))

    # model.add(layers.Conv2D(16, (63, 63), activation='relu', input_shape=(width, height, channelCount)))
    # model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    # model.add(layers.Conv2D(64, (1, 1), activation='relu'))
    # model.add(layers.Conv2D(1, (1, 1), activation='relu'))

    model.add(layers.Conv2D(16, (31, 31), activation='relu', input_shape=(width, height, channelCount)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.Conv2D(64, (1, 1), activation='relu'))
    model.add(layers.Conv2D(1, (1, 1), activation='relu'))

    #model.add(layers.Conv2D(128, (31, 31), activation='relu', input_shape=(width, height, channelCount)))
    #model.add(layers.Conv2D(512, (2, 2), activation='relu'))
    #model.add(layers.Conv2D(1, (2, 2), activation='relu'))

    #model.add(layers.Conv2D(128, (2, 2), activation='relu'))
    #model.add(layers.Conv2D(96, (2, 2), activation='relu'))
    #model.add(layers.Conv2D(16, (1, 1), activation='relu'))
    #model.add(layers.Conv2D(1, (1, 1), activation='relu'))

    model.summary()

    # model.add(layers.Conv2D(16,(11,11), activation='relu', input_shape=(32,32,1)))
    # model.add(layers.Conv2D(64,(5,5), activation='relu'))
    # model.add(layers.Conv2D(64, (5, 5), activation='relu'))
    # model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    # model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    # model.add(layers.Conv2D(64, (1, 1), activation='relu'))
    # model.add(layers.Conv2D(1, (1, 1), activation='relu'))
    from keras.optimizers import SGD
    #sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
    #modela.compile(optimizer=optimizers.RMSprop(lr=1e-4),
    #              loss=losses.mean_squared_error, metrics=['accuracy'])
    #modela.compile(optimizer=sgd,
    #              loss=losses.mean_squared_error, metrics=['accuracy'])
    #losses.mean_squared_logarithmic_error
    #loss = 'binary_crossentropy'

    from keras.optimizers import SGD
    sgd = SGD() # lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    #sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    lossFn = losses.mean_squared_logarithmic_error

    def coeff(y_true, y_pred, smooth, tresh):
        return K.sqrt(K.sum(K.square(y_true - y_pred)*K.abs(y_true)))

    def my_loss(smooth, thresh):
        def loss1(y_true, y_pred):
            return coeff(y_true, y_pred, smooth, thresh)
        return loss1

    lossFn = my_loss(smooth=1e-5, thresh=0.5)

    model.compile(optimizer=sgd,loss=lossFn,
                   metrics=['mse'])

    epochs = 20

    history = model.fit(x=train_input, y=train_result, epochs=epochs,
              batch_size=1,
              validation_data=(validation_input, validation_result)
              )

    test_predicted = model.predict(test_input)

    #saveModel(model)

    return history, test_input, test_predicted, test_result

def plotHistory(history):

    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.figure(1)
    plt.subplot(211)

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.subplot(212)

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()

if __name__ == '__main__':

    delta = 1.0
    gridWidth = 32.0
    gridHeight = 32.0
    rect = Rectangle(0, 0, gridWidth, gridHeight)
    g = Geometry(rect, delta)

    chargeCount = 3
    channelChargeCount = int(chargeCount)
    count = 400

    x_areaWithoutCharge = 4.0
    y_areaWithoutCharge = 4.0
    x_positions = np.linspace(x_areaWithoutCharge, gridWidth-x_areaWithoutCharge, gridWidth-2*x_areaWithoutCharge)
    y_positions = np.linspace(y_areaWithoutCharge, gridHeight-y_areaWithoutCharge, gridHeight-2*y_areaWithoutCharge)
    x = np.random.choice(x_positions, size=count*chargeCount)/gridWidth
    y = np.random.choice(y_positions, size=count*chargeCount)/gridHeight

    charges = np.zeros(shape=(count,g.numX*2,g.numY*2, channelChargeCount))
    results = np.zeros(shape=(count,g.numX,g.numY))

    for i in range(0, count):
        #charge = make_single_charge(g, x[i], y[i], -10)
        charge = make_n_fold_charge(g, x, y, i, chargeCount, -10, variateSign=False)

        charges_weight_matrix = []
        for channel in range(0, chargeCount):
            charges_weight_matrix.append(calc_charge_weight_matrix(g, charge, channel))
        #charges_weight_matrix.append(calc_charge_weight_matrix_multi(g, charge))

        charges_stacked = np.stack(charges_weight_matrix, axis=-1)
        charges[i] = charges_stacked

        result_matrix = solvePDE(g, charge)
        results[i] = result_matrix

    print(charges.shape)
    print(results.shape)

    c = charges.reshape((count,int(gridWidth*2),int(gridHeight*2),channelChargeCount))
    r = results.reshape((count,int(gridWidth),int(gridHeight),1))

    #c1 = np.min(c)
    #r1 = np.min(r)

    #c -= c1
    #r -= r1

    #c1 = np.min(c)
    #r1 = np.min(r)

    c = c / np.max(c)
    r = r / np.max(r)




    ma = np.max(c)
    mi = np.min(c)

    history, test_input, test_predicted, test_result = learn(c,r)


    #plotHistory(history)

    #c1 = c[:, 16:48,16:48, :]

    ct = test_input[:, int(gridWidth/2):int(gridWidth/2*3), int(gridWidth/2):int(gridWidth/2*3), :]

    #c1 *= -1
    #m = np.min(c1, axis=(1,2))[0]
    #c1 -= m

    #plotMatrixList(g, [r[5,:,:,0], c1[5,:,:,0]])

    idx = 5
    #print(x[i], y[i], (int)(len(g.X) * x[idx]),(int)(len(g.Y) * y[idx]))

    #test_predicted


    #max_numeric_solution = np.max(test_result[idx,:,:,0])
    #max_prediced_solution = np.max(test_predicted[idx,:,:,0])
    #scale_factor = max_numeric_solution / max_prediced_solution
    #test_predicted[idx, :, :, 0] *= scale_factor

    #plotMatrixList(g, [r[idx,:,:,0], c1[idx,:,:,0]])
    #plotMatrixList(g, [ct[idx,:,:,0], test_predicted[idx,:,:,0], test_result[idx,:,:,0]])
    #idx+=1

    plotMatrixList(g, [ct[idx,:,:,0], test_predicted[idx,:,:,0], test_result[idx,:,:,0]])