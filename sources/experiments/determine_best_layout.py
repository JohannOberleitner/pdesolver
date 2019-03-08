import sys
import time

import numpy as np

from sources.pdesolver.finite_differences_method.FiniteDifferencesSolver_V2 import GridConfiguration, \
    ConstantGridValueProvider, FiniteDifferencesMethod4
from sources.pdesolver.finite_differences_method.boundaryconditions import RectangularBoundaryCondition
from sources.pdesolver.finite_differences_method.geometry import Geometry
from sources.experiments.charges_generators import make_single_charge, make_double_charge, make_n_fold_charge
from sources.pdesolver.finite_differences_method.rectangle import Rectangle

from keras import layers, optimizers, losses
from keras import models

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
            matrix[row, col] = 1./(1.+np.sqrt( (x-col)**2 + (y-row)**2 ))

    return matrix

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

def learn(input, target, model, epochs):

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
    #model = models.Sequential()
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

    #model.add(layers.Conv2D(16, (63, 63), activation='relu', input_shape=(width, height, channelCount)))
    #model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    #model.add(layers.Conv2D(64, (1, 1), activation='relu'))
    #model.add(layers.Conv2D(1, (1, 1), activation='relu'))

    #model.summary()

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
    #lossFn = 'mse'
    model.compile(optimizer=sgd,loss=lossFn,
                   metrics=['mse'])


    history = model.fit(x=train_input, y=train_result, epochs=epochs,
              batch_size=1,
              validation_data=(validation_input, validation_result),
              verbose=0
              )

    test_predicted = model.predict(test_input)

    #saveModel(model)

    return history, test_input, test_predicted, test_result

#def expand_models(models):


def split_numbers_1(number):

    results = []
    for i in range(0,number, 2):
        next = []
        next.append(number-i)
        next.extend(split_numbers(i))
        results.append(next)

    return results


def split_numbers(number):

    results = []
    for i in range(0, number, 2):
        if i==0:
            results.append([number])
        else:
            next = split_numbers(number-i)
            for innerlist in next:
                item = []
                item.extend(innerlist)
                item.append(i)
                results.append(item)

    return results

def make_model(input_sizes, width, height, channelCount):

    model = models.Sequential()


    model.add(layers.Conv2D(16, (input_sizes[0]+1, input_sizes[0]+1), activation='relu', input_shape=(width, height, channelCount)))
    if (len(input_sizes) > 2):
        model.add(layers.Conv2D(32, (input_sizes[1]+1, input_sizes[1]+1), activation='relu'))

    for size in input_sizes[2:-1]:
        model.add(layers.Conv2D(64, (size+1, size+1), activation='relu'))
    #    #

    if (len(input_sizes)>1):
        model.add(layers.Conv2D(128, (input_sizes[-1]+1, input_sizes[-1]+1), activation='relu'))


    #model.add(layers.Conv2D(16, (31, 31), activation='relu', input_shape=(width, height, channelCount)))
    # model.add(layers.Conv2D(32, (11,11), activation='relu'))
    # model.add(layers.Conv2D(64, (5, 5), activation='relu'))
    # model.add(layers.Conv2D(64, (5, 5), activation='relu'))
    # model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    #model.add(layers.Conv2D(128, (3, 3), activation='relu'))

    model.add(layers.Conv2D(64, (1, 1), activation='relu'))
    model.add(layers.Conv2D(1, (1, 1), activation='relu'))

    #model.summary()
    return model

def calc_models(length, maxdepth):

    # make_model([32], 64,64,2)
    # make_model([30,2],64,64,1)
    # make_model([28,2,2], 64,64,2)
    # make_model([26, 2, 2,2], 64,64,2)
    # make_model([24, 2, 2, 2, 2], 64, 64, 2)

    all_models=[]
    i=0
    d = split_numbers(length)
    print('maxcount:', len(d))
    for m in d:
        if len(m) <= maxdepth:
            all_models.append((m, make_model(m, length*2, length*2, 2)))
            i+=1
            #print('count:', len(all_models))

    print('count:', i, len(all_models))
    return all_models

if __name__ == '__main__':

    delta = 1.0
    gridWidth = 32.0
    gridHeight = 32.0
    rect = Rectangle(0, 0, gridWidth, gridHeight)
    g = Geometry(rect, delta)

    chargeCount = 2
    count = 600

    x_areaWithoutCharge = 4.0
    y_areaWithoutCharge = 4.0
    x_positions = np.linspace(x_areaWithoutCharge, gridWidth-x_areaWithoutCharge, gridWidth-2*x_areaWithoutCharge)
    y_positions = np.linspace(y_areaWithoutCharge, gridHeight-y_areaWithoutCharge, gridHeight-2*y_areaWithoutCharge)
    x = np.random.choice(x_positions, size=count*chargeCount)/gridWidth
    y = np.random.choice(y_positions, size=count*chargeCount)/gridHeight

    charges = np.zeros(shape=(count,g.numX*2,g.numY*2, chargeCount))
    results = np.zeros(shape=(count,g.numX,g.numY))

    for i in range(0, count):
        #charge = make_single_charge(g, x[i], y[i], -10)
        charge = make_n_fold_charge(g, x, y, i, chargeCount, -10)

        charges_weight_matrix = []
        for channel in range(0, chargeCount):
            charges_weight_matrix.append(calc_charge_weight_matrix(g, charge, channel))

        charges_stacked = np.stack(charges_weight_matrix, axis=-1)
        #charges[i] = charges_weight_matrix
        charges[i] = charges_stacked

        result_matrix = solvePDE(g, charge)
        results[i] = result_matrix

    #plotMatrixList(g, [result_matrix, charges_weight_matrix])

    print(charges.shape)
    print(results.shape)

    c = charges.reshape((count,int(gridWidth*2),int(gridHeight*2),chargeCount))

    r = results.reshape((count,int(gridWidth),int(gridHeight),1))

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

    all_models = calc_models(int(32.0), 2)

    for model in all_models:

        start = time.time()

        history, test_input, test_predicted, test_result = learn(c,r, model[1], epochs=10)

        duration = time.time()-start
        print(model[0], duration, history.history['mean_squared_error'][-1], history.history['loss'][-1], history.history['val_mean_squared_error'][-1], history.history['val_loss'][-1])



