import time

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


from sources.pdesolver.finite_differences_method.FiniteDifferencesSolver_V2 import GridConfiguration, \
    ConstantGridValueProvider, FiniteDifferencesMethod4
from sources.pdesolver.finite_differences_method.boundaryconditions import RectangularBoundaryCondition
from sources.pdesolver.finite_differences_method.geometry import Geometry
from sources.experiments.charges_generators import make_single_charge
from sources.pdesolver.finite_differences_method.rectangle import Rectangle

from keras import layers, optimizers, losses
from keras import models

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
            #matrix[row, col] = 1./(1+np.sqrt( (x-col)**2 + (y-row)**2 ))
            matrix[row, col] = 1./(1.+np.sqrt( (x-col)**2 + (y-row)**2 ))

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

    g = Geometry(rect, delta)
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

    epochs = 20

    # 100 = train=60%+validation=20%+test=20%
    model = models.Sequential()
    #model.add(layers.Conv2D(32,(11,11), activation='relu', input_shape=(32,32,1)))
    #model.add(layers.Conv2D(64,(7,7), activation='relu'))
    #model.add(layers.Conv2D(96,(5, 5), activation='relu'))
    #model.add(layers.Conv2DTranspose(48, (7,7), activation='relu'))
    #model.add(layers.Conv2DTranspose(16, (5, 5), activation='relu'))
    #model.add(layers.Conv2DTranspose(16, (11, 11), activation='relu'))
    #model.add(layers.Conv2DTranspose(16, (3, 3), activation='relu'))
    #model.add(layers.Conv2D(16, (3,3), activation='relu'))
    #model.add(layers.Dense(1, activation='sigmoid'))

    #model.add(layers.Conv2D(32,(3,3), activation='relu', input_shape=(32,32,1)))
    #model.add(layers.Conv2D(32,(5,5), activation='relu'))
    #model.add(layers.Conv2D(64,(5, 5), activation='relu'))
    #model.add(layers.Conv2DTranspose(48, (7,7), activation='relu'))
    #model.add(layers.Conv2DTranspose(16, (5, 5), activation='relu'))
    #model.add(layers.Dense(1, activation='sigmoid'))

    modela = models.Sequential()
    modela.add(layers.Conv2D(16, (11, 11), activation='relu', input_shape=(64, 64, 1)))
    modela.add(layers.Conv2D(32, (11,11), activation='relu'))
    modela.add(layers.Conv2D(64, (5, 5), activation='relu'))
    modela.add(layers.Conv2D(64, (5, 5), activation='relu'))
    modela.add(layers.Conv2D(64, (3, 3), activation='relu'))
    modela.add(layers.Conv2D(128, (3, 3), activation='relu'))
    modela.add(layers.Conv2D(64, (1, 1), activation='relu'))
    modela.add(layers.Conv2D(1, (1, 1), activation='relu'))
    modela.summary()

    # model.add(layers.Conv2D(16,(11,11), activation='relu', input_shape=(32,32,1)))
    # model.add(layers.Conv2D(64,(5,5), activation='relu'))
    # model.add(layers.Conv2D(64, (5, 5), activation='relu'))
    # model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    # model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    # model.add(layers.Conv2D(64, (1, 1), activation='relu'))
    # model.add(layers.Conv2D(1, (1, 1), activation='relu'))
    #
    #
    #
    # model.summary()
    from keras.optimizers import SGD
    #sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
    #modela.compile(optimizer=optimizers.RMSprop(lr=1e-4),
    #              loss=losses.mean_squared_error, metrics=['accuracy'])
    #modela.compile(optimizer=sgd,
    #              loss=losses.mean_squared_error, metrics=['accuracy'])



    #losses.mean_squared_logarithmic_error
    #loss = 'binary_crossentropy'

    import keras.backend as K
    def dice_coeff(y_true, y_pred, smooth, thresh):
        print(y_true.shape, y_pred.shape)
        result = K.sqrt(K.sum(K.square(y_pred) - K.square(y_true)))
        return result



    def pde_loss(smooth, thresh):
        def dice(y_true, y_pred):
            return -dice_coeff(y_true, y_pred, smooth, thresh)
        return dice
    model_dice = pde_loss(smooth=1e-5, thresh=0.5)

    from keras.optimizers import SGD
    sgd = SGD()
    #losses.mean_squared_logarithmic_error
    modela.compile(optimizer=sgd,loss='mse',
                   metrics=['mse'])


    history = modela.fit(x=train_input, y=train_result, epochs=epochs,
              batch_size=1,
              validation_data=(validation_input, validation_result)
              )

    test_predicted = modela.predict(test_input)

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

    count = 200

    x_areaWithoutCharge = 4.0
    y_areaWithoutCharge = 4.0
    x_positions = np.linspace(x_areaWithoutCharge, gridWidth-x_areaWithoutCharge, gridWidth-2*x_areaWithoutCharge)
    y_positions = np.linspace(y_areaWithoutCharge, gridHeight-y_areaWithoutCharge, gridHeight-2*y_areaWithoutCharge)
    x = np.random.choice(x_positions, size=count)/gridWidth
    y = np.random.choice(y_positions, size=count)/gridHeight

    charges = np.zeros(shape=(count,g.numX*2,g.numY*2))
    results = np.zeros(shape=(count,g.numX,g.numY))

    for i in range(count):
        charge = make_single_charge(g, x[i], y[i], -10)

        charges_weight_matrix = calc_charge_weight_matrix_orig(g, charge)
        charges[i] = charges_weight_matrix

        result_matrix = solvePDE(g, charge)
        results[i] = result_matrix

    #plotMatrixList(g, [result_matrix, charges_weight_matrix])

    print(charges.shape)
    print(results.shape)

    c = charges.reshape((count,64,64,1))
    #c1 = c[:, 16:48, 16:48, :]
    #'c *= -1
    #m = np.min(c, axis=(1, 2))[:,0]
    #for c1, m1 in zip(c, m):
    #    print(m1.shape, m1)
    #   c1 -= m1''

    r = results.reshape((count,32,32,1))

    #r = c[:,16:48,16:48,:]

    c = c/np.max(c)
    r = r/np.max(r)


    ma = np.max(c)
    mi = np.min(c)

    mar = np.max(r)
    mir = np.min(r)

    print(ma, mi, mar, mir)
    #c -= c.mean(axis=0)
    #c /= c.std(axis=0)
    #c -= c.min(axis=0)

    #r -= r.mean(axis=0)
    l2 = np.copy(r)
    s = l2.std(axis=0)
    s[s == 0.] = 1.
    #r = r / s
    #r -= r.min(axis=0)

    #r -= r.mean(axis=0)
    #u1 = r.std(axis=0)
    #print(u1)

    ma = np.max(c)
    mi = np.min(c)

    mar = np.max(r)
    mir = np.min(r)

    print(ma, mi, mar, mir)

    history, test_input, test_predicted, test_result = learn(c,r)
    #plotHistory(history)

    #c1 = c[:, 16:48,16:48, :]

    ct = test_input[:, 16:48, 16:48, :]

    #c1 *= -1
    #m = np.min(c1, axis=(1,2))[0]
    #c1 -= m

    #plotMatrixList(g, [r[5,:,:,0], c1[5,:,:,0]])

    idx = 5
    #print(x[i], y[i], (int)(len(g.X) * x[idx]),(int)(len(g.Y) * y[idx]))

    #test_predicted

    #plotMatrixList(g, [r[idx,:,:,0], c1[idx,:,:,0]])
    plotMatrixList(g, [ct[idx,:,:,0], test_predicted[idx,:,:,0], test_result[idx,:,:,0]])
    #idx+=1
    #plotMatrixList(g, [ct[idx,:,:,0], test_predicted[idx,:,:,0], test_result[idx,:,:,0]])