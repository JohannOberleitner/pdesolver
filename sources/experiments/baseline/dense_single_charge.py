import datetime
import getopt
import json
import os
import pickle
import sys
import time
from math import log

from keras import models, layers, losses
import keras.backend as K

from sources.experiments.charges_generators import make_central_charge, make_single_charge, make_n_fold_charge_from_list
from sources.pdesolver.finite_differences_method.charge_distribution import ChargeDistribution
from sources.pdesolver.finite_differences_method.geometry import Geometry
from sources.pdesolver.pde.PDE import PDEExpressionType, PDE

import numpy as np

from sources.experiments.fdm_helper import plotSurface
import matplotlib.pyplot as plt

class Results:

    def __init__(self, inputset_duration, solution_calculation_duration, learning_duration, errors):
        self.inputset_duration = inputset_duration
        self.solutionset_duration = solution_calculation_duration
        self.learning_duration = learning_duration
        self.errors = errors

        # sum, max_value, avg, median, varianc

        sums = np.zeros(shape=(len(errors)))
        max_values = np.zeros(shape=(len(errors)))
        avgs = np.zeros(shape=(len(errors)))
        medians = np.zeros(shape=(len(errors)))
        variances = np.zeros(shape=(len(errors)))

        for i, error_tuple in enumerate(errors):
            sums[i] = error_tuple[0]
            max_values[i] = error_tuple[1]
            avgs[i] = error_tuple[2]
            medians[i] = error_tuple[3]
            variances[i] = error_tuple[4]

        self.avg_error_avg = np.average(avgs)
        self.avg_error_variance = np.var(avgs)

        self.variances_avg = np.average(variances)
        self.variances_variance = np.var(variances)

        self.sum_error = np.average(sums)
        self.sum_error_variance = np.var(sums)

        self.max_values = np.average(max_values)
        self.max_values_variance = np.var(max_values)

        self.median_values = np.average(medians)
        self.median_values_variance = np.var(medians)


    def encode(self):
        return {'__Results__': True,
                'inputset_duration': self.inputset_duration,
                'solutionset_duration': self.solutionset_duration,
                'learning_duration': self.learning_duration,
                'avg_error_avg': self.avg_error_avg, 'avg_error_variance':self.avg_error_variance,
                'variances_avg': self.variances_avg, 'variances_variance':self.variances_variance,
                'sum_error': self.sum_error, 'sum_error_variances': self.sum_error_variance,
                'max_values': self.max_values, 'max_values_variance': self.max_values_variance,
                'median_values': self.median_values, 'median_values_variance': self.max_values_variance,
                'errors': self.errors
                }

class ResultsEncoder(json.JSONEncoder):
    def default(self, data):
        if isinstance(data, Results):
            return data.encode()
        else:
            super().default(self, data)

class TrainingsSetConfig:

    def __init__(self, gridWidth, gridHeight, architectureType, epochs, N, charges, timestamp):
        self.gridWidth = gridWidth
        self.gridHeight = gridHeight
        self.architectureType = architectureType
        self.epochs = epochs
        self.N = N
        self.charges = charges
        self.timestamp = timestamp

    def encode(self):
        return {'__TrainingsSetConfig__': True, 'createdAt': str(self.timestamp), 'gridWidth': self.gridWidth,
            'gridHeight': self.gridHeight, 'architectureType':self.architectureType, 'charges': self.charges, 'N':self.N,
            'epochs': self.epochs,
            'charges': self.charges }

class TrainingsSetConfigEncoder(json.JSONEncoder):
    def default(self, data):
        if isinstance(data, TrainingsSetConfig):
            return data.encode()
        elif isinstance(data, ChargeDistribution):
            return data.chargesList
        else:
            super().default(self, data)

class TrainingsSetConfigDecoder:
    def decode(self, json_data):

        gridWidth = json_data["gridWidth"]
        gridHeight = json_data["gridHeight"]
        architectureType = json_data["architectureType"]
        epochs = json_data["epochs"]
        N = json_data["N"]
        chargesList = json_data["charges"]
        timestamp = json_data["createdAt"]
        trainingsSetConfig = self.init_data(gridWidth, gridHeight, architectureType, epochs, N, chargesList, timestamp)
        return trainingsSetConfig

    def init_data(self, gridWidth, gridHeight, architectureType, epochs, N, chargesList, timestamp):
        delta = 1.0
        #geometry = Geometry(self.rect, delta)
        #charges = ChargeDistribution(geometry)
        #charges.addList(chargesList)
        charges = [] # does not work right now
        return TrainingsSetConfig(gridWidth, gridHeight, architectureType, N, charges)



def setupPDE_vector_calculus(gridSize, equation):

    pde = PDE(gridSize, gridSize)
    pde.setEquationExpression(PDEExpressionType.VECTOR_CALCULUS, equation)
    pde.setVectorVariable("r", dimension=2)
    pde.configureGrid()

    return pde

def calc_charge_weight_matrix(geometry, chargeDistribution, index=0):
    matrix = np.zeros(shape=(len(geometry.Y),len(geometry.X)))
    firstCharge = chargeDistribution.chargesList[index]
    x = firstCharge[0]
    y = firstCharge[1]
    for row in range(0, geometry.numY):
        for column in range(0, geometry.numX):
            matrix[row, column] = 1./(1.2+np.sqrt( (x-column)**2 + (y-row)**2 ))

    return matrix


class SortedChargeCollection:
    """
    Used to generate collections of charges. The sort order is column ASCending, row ASCending
    """

    def __init__(self):
        self.columns = []
        self.rows = []
        self.values = []

    def append(self, column, row, value):
        insertion_index = self.find_position(column, row)
        self.columns.insert(insertion_index, column)
        self.row.insert(insertion_index, row)
        self.values.insert(insertion_index, value)

    def find_position(self, new_column, new_row):
        for index in range(0, len(self.columns)):
            if index > len(self.columns):
                return index

            if self.columns[index] < new_column:
                pass # index weiterschalten, ok
            elif self.columns[index] > new_column:
                return index
            else: # ==
                if self.rows[index] < new_row:
                    pass # index weiterschalten ok
                elif self.rows[index] > new_row:
                    return index
                else:
                    raise ValueError()
        return index


    def get_column(self):
        return self.column

    def get_row(self):
        return self.row

    def get_value(self):
        return self.value

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.column == other.column and self.row == other.row and self.value == other.value
        else:
            return False

    def __hash(self):
        return hash(self.column * self.row * self.value)


class TrainingSet_CreationStrategy:
    def __init__(self):
        self.charge_positions = []
        self.charges_count = 1

    def get_marginWithout_Charge(self):
        return 4

    def get_charge_value(self):
        return -10

    def create_inputSet(self):
        self.prepare()
        self.charges = []
        self.input_set = np.zeros(shape=(self.N, self.gridWidth, self.gridHeight, self.charges_count))

        for index, charges_list in enumerate(self.charge_positions):

            charges_weight_matrix_list = []

            if len(charges_list) == 1:
                charge_tuple = charges_list[0]
                column = charge_tuple[0]
                row = charge_tuple[1]

                charge = make_single_charge(self.geometry, column/self.gridWidth, row/self.gridHeight, self.get_charge_value())
                self.charges.append(charge)
                charges_weight_matrix_list.append(calc_charge_weight_matrix(self.geometry, charge))

            else:
                charges_coordinate_list = []
                for charge_tuple in charges_list:
                    column = charge_tuple[0]
                    row = charge_tuple[1]
                    charges_coordinate_list.append((column/self.gridWidth, row/self.gridHeight))

                charge = make_n_fold_charge_from_list(self.geometry, charges_coordinate_list, self.get_charge_value(), variateSign=False)
                self.charges.append(charge)

                for charge_index in range(0, self.charges_count):
                    charges_weight_matrix_list.append(calc_charge_weight_matrix(self.geometry, charge, charge_index))

            self.input_set[index] = np.stack(charges_weight_matrix_list, axis=-1)

            if (index % 100) == 0:
                print(index, ' input set done')

    def create_solutionSet(self, pde):
        self.solutions = np.zeros(shape=(self.N, self.gridWidth, self.gridHeight))
        for index, charge in enumerate(self.charges):
            self.solutions[index] = pde.solve(charge)
            if (index % 100) == 0:
                print(index, ' solution set done')

    def normalize_input_set(self):
        self.input_set = self.input_set/np.max(self.input_set)

    def normalize_solutions(self):
        self.solutions = self.solutions/np.max(self.solutions)

    def add_channel_axis_to_solutionSet(self):
        self.solutions = self.solutions.reshape((self.solutions.shape[0], self.gridWidth, self.gridHeight, -1))


class TrainingSet_CreationStrategy_Full_SingleCharge(TrainingSet_CreationStrategy):

    def __init__(self, geometry):
        self.charges_count = 1
        self.geometry = geometry
        self.gridWidth = geometry.numX
        self.gridHeight = geometry.numY

    def prepare(self):
        margin = self.get_marginWithout_Charge()
        self.charge_positions = []

        for row in range(margin, self.gridHeight-margin-1):
            for column in range(margin, self.gridWidth-margin-1):
                charge_tuple = (column, row)
                charges_list = []
                charges_list.append(charge_tuple)
                self.charge_positions.append(charges_list)

        self.N = len(self.charge_positions)

        np.random.shuffle(self.charge_positions)

class TrainingSet_CreationStrategy_N_SingleCharge(TrainingSet_CreationStrategy):

    def __init__(self, geometry, N):
        self.charges_count = 1
        self.geometry = geometry
        self.gridWidth = geometry.numX
        self.gridHeight = geometry.numY
        self.N = N

    def prepare(self):
        margin = self.get_marginWithout_Charge()
        duplicate_trainings_elements = True

        while duplicate_trainings_elements:
            self.charge_positions = []
            #rows = np.random.random_integers(margin, self.gridHeight-margin-1, self.N)
            #columns = np.random.random_integers(margin, self.gridWidth-margin-1, self.N)

            rows = np.random.randint(margin, self.gridHeight-margin-1+1, self.N)
            columns = np.random.randint(margin, self.gridWidth-margin-1+1, self.N)

            for row, column in zip(rows, columns):
                charge_tuple = (column, row)
                if charge_tuple in self.charge_positions:
                    continue

                charge_tuple = (column, row)
                charges_list = []
                charges_list.append(charge_tuple)
                self.charge_positions.append(charges_list)

            duplicate_trainings_elements = False


class TrainingSet_CreationStrategy_N_MultiCharge(TrainingSet_CreationStrategy):

    def __init__(self, geometry, N, charges_count):
        self.charges_count = charges_count
        self.geometry = geometry
        self.gridWidth = geometry.numX
        self.gridHeight = geometry.numY
        self.N = N

    def prepare(self):
        margin = self.get_marginWithout_Charge()

        rows = np.random.random_integers(margin, self.gridHeight - margin - 1, self.N * self.charges_count)
        columns = np.random.random_integers(margin, self.gridWidth - margin - 1, self.N * self.charges_count)

        self.charge_positions = []
        for trainingsset_index in range(0, self.N):

            charges_list = []
            for index in range(0, self.charges_count):
                charge_tuple = (rows[trainingsset_index*self.charges_count+index],
                                columns[trainingsset_index*self.charges_count+index],)

                charges_list.append(charge_tuple)

            self.charge_positions.append(charges_list)


def make_model(architectureType, gridWidth, gridHeight, charges_count):
    model = models.Sequential()

    if architectureType == 1:
        model.add(layers.Dense(16, input_shape=(gridWidth, gridHeight, charges_count), activation='relu'))
        model.add(layers.Dense(1, activation='relu'))
    elif architectureType == 2:
        model.add(layers.Dense(64, input_shape=(gridWidth, gridHeight, charges_count), activation='relu'))
        model.add(layers.Dense(1, activation='relu'))
    elif architectureType == 3:
        model.add(layers.Dense(96, input_shape=(gridWidth, gridHeight, charges_count), activation='relu'))
        model.add(layers.Dense(1, activation='relu'))
    elif architectureType == 21:
        model.add(layers.Dense(64, input_shape=(gridWidth, gridHeight, charges_count), activation='relu'))
        model.add(layers.Dense(16, input_shape=(gridWidth, gridHeight, charges_count), activation='relu'))
        model.add(layers.Dense(1, activation='relu'))
    elif architectureType == 22:
        model.add(layers.Dense(64, input_shape=(gridWidth, gridHeight, charges_count), activation='relu'))
        model.add(layers.Dense(128, input_shape=(gridWidth, gridHeight, charges_count), activation='relu'))
        model.add(layers.Dense(1, activation='relu'))
    elif architectureType == 23:
        model.add(layers.Dense(96, input_shape=(gridWidth, gridHeight, charges_count), activation='relu'))
        model.add(layers.Dense(16, input_shape=(gridWidth, gridHeight, charges_count), activation='relu'))
        model.add(layers.Dense(1, activation='relu'))
    elif architectureType == 31:
        model.add(layers.Dense(64, input_shape=(gridWidth,gridHeight,charges_count), activation='relu'))
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(1, activation='relu'))
    elif architectureType == 32:
        model.add(layers.Dense(64, input_shape=(gridWidth,gridHeight,charges_count), activation='relu'))
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(1, activation='relu'))
    elif architectureType == 33:
        model.add(layers.Dense(64, input_shape=(gridWidth, gridHeight, charges_count), activation='relu'))
        model.add(layers.Dense(16, activation='relu'))
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(1, activation='relu'))
    elif architectureType == 34:
        model.add(layers.Dense(64, input_shape=(gridWidth, gridHeight, charges_count), activation='relu'))
        model.add(layers.Dense(16, activation='relu'))
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dense(1, activation='relu'))
    elif architectureType == 35:
        model.add(layers.Dense(64, input_shape=(gridWidth, gridHeight, charges_count), activation='relu'))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(16, activation='relu'))
        model.add(layers.Dense(1, activation='relu'))
    elif architectureType == 36:
        model.add(layers.Dense(96, input_shape=(gridWidth, gridHeight, charges_count), activation='relu'))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(16, activation='relu'))
        model.add(layers.Dense(1, activation='relu'))
    elif architectureType == 41:
        model.add(layers.Dense(64, input_shape=(gridWidth, gridHeight, charges_count), activation='relu'))
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(16, activation='relu'))
        model.add(layers.Dense(1, activation='relu'))
    elif architectureType == 42:
        model.add(layers.Dense(64, input_shape=(gridWidth, gridHeight, charges_count), activation='relu'))
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(16, activation='relu'))
        model.add(layers.Dense(1, activation='relu'))
    elif architectureType == 43:
        model.add(layers.Dense(64, input_shape=(gridWidth, gridHeight, charges_count), activation='relu'))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(1, activation='relu'))
    elif architectureType == 44:
        model.add(layers.Dense(64, input_shape=(gridWidth, gridHeight, charges_count), activation='relu'))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(16, activation='relu'))
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dense(1, activation='relu'))
    elif architectureType == 45:
        model.add(layers.Dense(96, input_shape=(gridWidth, gridHeight, charges_count), activation='relu'))
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dense(96, activation='relu'))
        model.add(layers.Dense(16, activation='relu'))
        model.add(layers.Dense(1, activation='relu'))
    elif architectureType == 46:
        model.add(layers.Dense(96, input_shape=(gridWidth, gridHeight, charges_count), activation='relu'))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(1, activation='relu'))
    elif architectureType == 47:
        model.add(layers.Dense(96, input_shape=(gridWidth, gridHeight, charges_count), activation='relu'))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(16, activation='relu'))
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dense(1, activation='relu'))
    elif architectureType == 48:
        model.add(layers.Dense(96, input_shape=(gridWidth, gridHeight, charges_count), activation='relu'))
        model.add(layers.Dense(2048, activation='relu'))
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(1, activation='relu'))


    # model.add(layers.Dense(92, input_shape=(gridWidth, gridHeight, charges_count), activation='relu'))
    # model.add(layers.Dense(128, activation='relu'))
    # model.add(layers.Dense(64, activation='relu'))
    # model.add(layers.Dense(32, activation='relu'))
    # model.add(layers.Dense(1, activation='relu'))

    #model.add(layers.Dense(92, input_shape=(64,64,1), activation='relu'))
    #model.add(layers.Dense(512, activation='relu'))
    #model.add(layers.Dense(64, activation='relu'))
    ##model.add(layers.Dense(32, activation='relu'))
    #model.add(layers.Dense(1, activation='relu'))

    # model.add(layers.Dense(128, input_shape=(64,64,1), activation='relu'))
    # model.add(layers.Dense(512, activation='relu'))
    # model.add(layers.Dense(2048, activation='relu'))
    # model.add(layers.Dense(512, activation='relu'))
    # model.add(layers.Dense(256, activation='relu'))
    # model.add(layers.Dense(128, activation='relu'))
    # model.add(layers.Dense(1,activation='relu'))

    model.summary()

    return model

def learn(model, epochs, train_input, train_output, validation_input, validation_output):

    from keras.optimizers import SGD
    sgd = SGD()  # lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    # sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    lossFn = losses.mean_squared_logarithmic_error

    def coeff(y_true, y_pred, smooth, tresh):
        return K.sqrt(K.sum(K.square(y_true - y_pred) * K.abs(y_true)))

    def my_loss(smooth, thresh):
        def loss1(y_true, y_pred):
            return coeff(y_true, y_pred, smooth, thresh)

        return loss1

    # lossFn = my_loss(smooth=1e-5, thresh=0.5)

    model.compile(optimizer=sgd, loss=lossFn,
                  metrics=['mse'])

    history = model.fit(x=train_input, y=train_output, epochs=epochs,
                        batch_size=1,
                        validation_data=(validation_input, validation_output),
                        verbose=0
                        )

def calc_square_error_for_matrix(matrix1, matrix2):
    """ calculates errors and metrics for one solution matrix and its prediction"""
    #print (len(matrix1))
    #print (len(matrix1[0]))

    sum = 0.0
    max_value = -1.0

    errors = np.zeros(shape=(len(matrix1), len(matrix1[0])))

    for i in range(0, len(matrix1)):
        for j in range(0, len(matrix1[0])):
            errors[i,j] = (matrix1[i,j,0] - matrix2[i,j,0])**2

    sum = np.sum(errors)
    max_value = np.max(errors)
    avg = np.average(errors)
    median = np.median(errors)
    variance = np.var(errors)

    # sum of all errors on one matrix
    # max_value of an error on one matrix
    return sum, max_value, avg, median, variance

def calc_square_error_for_list(set1, set2):
    comparison_errors = []
    #print(len(set1))
    #print(len(set2))

    for index in range(0, len(set1)):
        comparison_errors.append(calc_square_error_for_matrix(set1[index], set2[index]))

    return comparison_errors

def write_data_configuration(filename, trainings_configuration):
    if filename == None:
        s = json.dumps(trainings_configuration, cls=TrainingsSetConfigEncoder)
        print(s)
    else:
        file = open(filename, 'w')
        json.dump(trainings_configuration, file, cls=TrainingsSetConfigEncoder)
        file.close()

def write_results(filename, results):
    if filename == None:
        s = json.dumps(results, cls=ResultsEncoder)
        print(s)
    else:
        file = open(filename, 'w')
        json.dump(results, file, cls=ResultsEncoder)
        file.close()

def saveModel(model, filename):
    model_json = model.to_json()
    with open(filename + '_model'+'.json', "w") as json_file:
        json_file.write(model_json)
    json_file.close()
    model.save_weights(filename + '.h5')

def makeFilename(directory,filename):
    if filename == None:
        return None
    else:
        return os.path.join(directory,filename)

def deleteExistingFile(filepath):
    if filepath and os.path.exists(filepath):
        os.remove(filepath)

def parseArguments(argv):
    supportedOptions = "hd:o:N:l:e:s:a:p:r:"
    supportLongOptions = ["dir=", "ofile="]
    usage = 'dense_single_charge.py -s <gridSize> -a <architectureType> -e <epochs> -N <count> -d <directory> -o <outputfile> -l <label>'

    outputDirectory = '.'
    outputFile = None
    label = None
    count = 20
    persistFile = None
    readFile = None

    try:
        opts, args = getopt.getopt(argv, supportedOptions, supportLongOptions)
    except getopt.GetoptError:
        print(usage)
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print(usage)
            sys.exit()
        elif opt in ("-a"):
            architectureType = int(arg)
        elif opt in ("-s"):
            gridSize = int(arg)
        elif opt in ("-e"):
            epochs = int(arg)
        elif opt in ("-N"):
            count = int(arg)
        elif opt in ("-d", "--dir"):
            outputDirectory = arg
        elif opt in ("-o", "--ofile"):
            outputFile = arg
        elif opt in ("-l", "--label"):
            label = arg
        elif opt in ("-p"):
            persistFile = arg
        elif opt in ("-r"):
            readFile = arg

    return outputFile, outputDirectory, gridSize, count, epochs, architectureType, persistFile, readFile


if __name__ == '__main__':

    charges_count = 1

    try:
        outputFile, outputDirectory, gridSize, count, epochs, architectureType, persistFile, readFile = parseArguments(sys.argv[1:])

    except:
        outputFile = 'test_data'
        outputDirectory= '.'
        gridSize = 64
        count = 1000
        epochs = 2
        architectureType = 1
        #persistFile = 'my_dump.pickle'
        #readFile = None
        readFile = 'my_dump.pickle'

    fileName = makeFilename(outputDirectory, outputFile)

    print('Write output to:', os.path.abspath(fileName))

    gridSize = float(gridSize)


    model = make_model(architectureType, (int)(gridSize), (int)(gridSize), charges_count)

    poisson_equation = "div(grad( u(r) ))"
    pde = setupPDE_vector_calculus(gridSize, poisson_equation)

    if readFile == None:

        #fill_strategy = TrainingSet_CreationStrategy_Full_SingleCharge(pde.geometry)
        fill_strategy = TrainingSet_CreationStrategy_N_SingleCharge(pde.geometry, N=count)
        #fill_strategy = TrainingSet_CreationStrategy_N_MultiCharge(pde.geometry, N=count, charges_count=charges_count)

        start = time.time()
        fill_strategy.create_inputSet()
        inputset_duration = time.time() - start
        print('duration for creating input set:', inputset_duration)

        fill_strategy.normalize_input_set()

        start = time.time()
        fill_strategy.create_solutionSet(pde)
        solutionset_duration = time.time() - start
        print('duration for calculating solution set:', solutionset_duration)

        fill_strategy.normalize_solutions()
        fill_strategy.add_channel_axis_to_solutionSet()

        if persistFile != None:
            with open(persistFile, 'wb') as f:
                tuple_to_persist = (fill_strategy, inputset_duration, solutionset_duration)
                pickle.dump(tuple_to_persist, f, pickle.HIGHEST_PROTOCOL)

    else:
        with open(readFile, 'rb') as f:

            tuple_to_load = pickle.load(f)
            fill_strategy = tuple_to_load[0]
            inputset_duration = tuple_to_load[1]
            solutionset_duration = tuple_to_load[2]


    #s = fill_strategy.solutions.reshape((fill_strategy.solutions.shape[0], pde.geometry.numX, (int)(pde.geometry.numY), -1))

    print(fill_strategy.input_set.shape)
    print(fill_strategy.solutions.shape)

    trainings_count = int(count * 0.9)
    train_input = fill_strategy.input_set[0:trainings_count]
    train_output = fill_strategy.solutions[0:trainings_count]
    print(trainings_count, train_input.shape, train_output.shape)

    validation_count = int(count * 0.05)
    validation_input = fill_strategy.input_set[trainings_count:trainings_count+validation_count]
    validation_output = fill_strategy.solutions[trainings_count:trainings_count+validation_count]
    print(validation_count, validation_input.shape, validation_output.shape)

    test_count = int(count * 0.05)
    test_input = fill_strategy.input_set[trainings_count+validation_count:]
    test_output = fill_strategy.solutions[trainings_count+validation_count:]
    print(test_count, test_input.shape, test_output.shape)


    start = time.time()

    learn(model, epochs, train_input, train_output, validation_input, validation_output)

    learning_duration = time.time() - start
    print('duration for learning:',  learning_duration)

    prediction = model.predict(test_input)

    #print(prediction.shape)

    errors = calc_square_error_for_list(test_output, prediction)
    print(errors)

    saveModel(model, fileName)
    trainingsSetConfig = TrainingsSetConfig(gridSize, gridSize, "1", epochs, count, fill_strategy.charges, datetime.datetime.utcnow())
    write_data_configuration(fileName, trainingsSetConfig)

    results = Results(inputset_duration, solutionset_duration, learning_duration, errors)
    write_results(fileName+'_results', results)

    showGraph = 1

    if showGraph:
        plotSurface(pde.geometry.X, pde.geometry.Y, prediction[0,:,:,0])
        plt.show()

