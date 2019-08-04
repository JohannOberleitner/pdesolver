import datetime
import getopt
import json
import platform
import os
import pickle
import sys
import time
from math import log, modf

import tensorflow as tf
from keras import models, layers, losses
import keras.backend as K

from sources.experiments.charges_generators import make_central_charge, make_single_charge, make_n_fold_charge_from_list
from sources.experiments.ellipsis_data_support.make_ellipsis import create_ellipsis_grid
from sources.pdesolver.finite_differences_method.charge_distribution import ChargeDistribution
from sources.pdesolver.finite_differences_method.geometry import Geometry
from sources.pdesolver.pde.PDE import PDEExpressionType, PDE

import numpy as np

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

        self.sum_error_avg = np.average(sums)
        self.sum_error_variance = np.var(sums)

        self.sum_error_total = np.sum(sums)

        self.max_values_avg = np.average(max_values)
        self.max_values_variance = np.var(max_values)
        self.max_values_max = np.max(max_values)

        self.median_values = np.average(medians)
        self.median_values_variance = np.var(medians)


    def encode(self):
        return {'__Results__': True,
                'inputset_duration': self.inputset_duration,
                'solutionset_duration': self.solutionset_duration,
                'learning_duration': self.learning_duration,
                'avg_error_avg': self.avg_error_avg, 'avg_error_variance':self.avg_error_variance,
                'variances_avg': self.variances_avg, 'variances_variance':self.variances_variance,
                'sum_error_avg': self.sum_error_avg, 'sum_error_variances': self.sum_error_variance,
                'sum_error_total': self.sum_error_total,
                'max_values_avg': self.max_values_avg, 'max_values_variance': self.max_values_variance,
                'max_values_max': self.max_values_max,
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


def setupPDE_vector_calculus_without_configure(gridSize, equation):

    pde = PDE(gridSize, gridSize)
    pde.setEquationExpression(PDEExpressionType.VECTOR_CALCULUS, equation)
    pde.setVectorVariable("r", dimension=2)
    return pde

def setupPDE_complete(pde, auxiliaryFunctions=None):
    if auxiliaryFunctions != None:
        pde.setAuxiliaryFunctions(auxiliaryFunctions)
    pde.configureGrid()
    return pde


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
    def __init__(self, geometry, charges_count):
        self.charge_positions = []
        self.geometry = geometry
        self.gridWidth = geometry.numX
        self.gridHeight = geometry.numY
        self.charges_count = charges_count

    def get_marginWithout_Charge(self):
        return 4

    def get_charge_value(self):
        return -10

    def create_inputSet(self):
        self.prepare()
        self.charges = []
        self.input_set = np.zeros(shape=(self.N, self.gridWidth, self.gridHeight, self.charges_count))

        for index, charges_list in enumerate(self.charge_positions):

            if len(charges_list) == 1:
                charge_tuple = charges_list[0]
                column = charge_tuple[0]
                row = charge_tuple[1]

                charge = make_single_charge(self.geometry, column/self.gridWidth, row/self.gridHeight, self.get_charge_value())
                self.charges.append(charge)
                charges_weight_matrix_list = self.calc_weight_matrix_list(0, charge, charges_list)

            else:
                charges_coordinate_list = []
                for charge_tuple in charges_list:
                    column = charge_tuple[0]
                    row = charge_tuple[1]
                    charges_coordinate_list.append((column/self.gridWidth, row/self.gridHeight))

                charge = make_n_fold_charge_from_list(self.geometry, charges_coordinate_list, self.get_charge_value(), variateSign=False)

                self.charges.append(charge)
                charges_weight_matrix_list = self.calc_weight_matrix_list(index, charge, charges_list)

            #print(np.stack(charges_weight_matrix_list).shape, len(charges_weight_matrix_list))
            self.input_set[index] = np.stack(charges_weight_matrix_list, axis=-1)

            if (index % 100) == 0:
                print(index, ' input set done')

    def calc_weight_matrix_list(self, index, charge, charges_list):
        charges_weight_matrix_list = []
        if len(charges_list) == 1:
            charges_weight_matrix_list.append(calc_charge_weight_matrix(self.geometry, charge))
        else:
            for charge_index in range(0, self.charges_count):
                charges_weight_matrix_list.append(calc_charge_weight_matrix(self.geometry, charge, charge_index))
        return charges_weight_matrix_list

    def create_solutionSet(self, pde):
        self.solutions = np.zeros(shape=(self.N, self.gridWidth, self.gridHeight))
        for index, charge in enumerate(self.charges):
            self.solutions[index] = self.solve(pde, index, charge)
            if (index % 100) == 0:
                print(index, ' solution set done')

    def solve(self, pde, index, charge):
        return pde.solve(charge)

    def normalize_input_set(self):
        self.input_set = self.input_set/np.max(self.input_set)

    def normalize_solutions(self):
        self.solutions = self.solutions/np.max(self.solutions)

    def add_channel_axis_to_solutionSet(self):
        self.solutions = self.solutions.reshape((self.solutions.shape[0], self.gridWidth, self.gridHeight, -1))


class TrainingSet_CreationStrategy_Full_SingleCharge(TrainingSet_CreationStrategy):

    def __init__(self, geometry):
        super(TrainingSet_CreationStrategy_Full_SingleCharge, self).__init__(geometry, 1)

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
        self.N = N
        super(TrainingSet_CreationStrategy_N_SingleCharge, self).__init__(geometry, 1)

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
        self.N = N
        super(TrainingSet_CreationStrategy_N_MultiCharge, self).__init__(geometry, charges_count)

    def prepare(self):
        margin = self.get_marginWithout_Charge()

        rows = np.random.randint(margin, self.gridHeight - margin - 1+1, self.N * self.charges_count)
        columns = np.random.randint(margin, self.gridWidth - margin - 1+1, self.N * self.charges_count)

        self.charge_positions = []
        for trainingsset_index in range(0, self.N):

            charges_list = []
            for index in range(0, self.charges_count):
                charge_tuple = (rows[trainingsset_index*self.charges_count+index],
                                columns[trainingsset_index*self.charges_count+index],)

                charges_list.append(charge_tuple)

            self.charge_positions.append(charges_list)

class TrainingSet_CreationStrategy_m_MultiCharge_Duplicates(TrainingSet_CreationStrategy):

    def __init__(self, geometry, N, charges_count):
        self.N = N
        super(TrainingSet_CreationStrategy_m_MultiCharge_Duplicates, self).__init__(geometry, charges_count)

    def prepare(self):
        margin = self.get_marginWithout_Charge()

        rows = np.random.randint(margin, self.gridHeight - margin - 1+1, self.N * self.charges_count)
        columns = np.random.randint(margin, self.gridWidth - margin - 1+1, self.N * self.charges_count)
        charges_count_per_trainingsset = np.random.randint(1, self.charges_count + 1, self.N)

        self.charge_positions = []
        for trainingsset_index in range(0, self.N):

            charges_list = []
            m = charges_count_per_trainingsset[trainingsset_index]
            for index in range(0, m):
                charge_tuple = (rows[trainingsset_index*self.charges_count+index],
                                columns[trainingsset_index*self.charges_count+index],)
                charges_list.append(charge_tuple)

            self.charge_positions.append(charges_list)

    def calc_weight_matrix_list(self, index, charge, charges_list):
        charges_weight_matrix_list = []
        m = len(charges_list)

        for charge_index in range(0, m):
            charges_weight_matrix_list.append(calc_charge_weight_matrix(self.geometry, charge, charge_index))

        for index in range(m, self.charges_count):
            charges_weight_matrix_list.append(charges_weight_matrix_list[index % m])

        return charges_weight_matrix_list

class TrainingSet_CreationStrategy_m_MultiCharge_Zeros(TrainingSet_CreationStrategy):

    def __init__(self, geometry, N, charges_count):
        self.N = N
        super(TrainingSet_CreationStrategy_m_MultiCharge_Zeros, self).__init__(geometry, charges_count)

    def prepare(self):
        margin = self.get_marginWithout_Charge()

        rows = np.random.randint(margin, self.gridHeight - margin - 1+1, self.N * self.charges_count)
        columns = np.random.randint(margin, self.gridWidth - margin - 1+1, self.N * self.charges_count)
        charges_count_per_trainingsset = np.random.randint(1, self.charges_count + 1, self.N)

        self.charge_positions = []
        for trainingsset_index in range(0, self.N):

            charges_list = []
            m = charges_count_per_trainingsset[trainingsset_index]
            for index in range(0, m):
                charge_tuple = (rows[trainingsset_index*self.charges_count+index],
                                columns[trainingsset_index*self.charges_count+index],)
                charges_list.append(charge_tuple)

            self.charge_positions.append(charges_list)

    def calc_weight_matrix_list(self, index, charge, charges_list):
        charges_weight_matrix_list = []
        zero_matrix = np.zeros(shape=(len(self.geometry.Y),len(self.geometry.X)))

        m = len(charges_list)

        for charge_index in range(0, m):
            charges_weight_matrix_list.append(calc_charge_weight_matrix(self.geometry, charge, charge_index))

        for index in range(m, self.charges_count):
            charges_weight_matrix_list.append(zero_matrix)

        return charges_weight_matrix_list


class TrainingSet_CreationStrategy_m_MultiCharge_EllipticMatter_Zeros(TrainingSet_CreationStrategy):

    def __init__(self, geometry, N, charges_count):
        self.N = N
        super(TrainingSet_CreationStrategy_m_MultiCharge_EllipticMatter_Zeros, self).__init__(geometry, charges_count)

    def get_channel_count(self):
        return self.charges_count + 1

    def prepare(self):
        margin = self.get_marginWithout_Charge()

        rows = np.random.randint(margin, self.gridHeight - margin - 1 + 1, self.N * self.charges_count)
        columns = np.random.randint(margin, self.gridWidth - margin - 1 + 1, self.N * self.charges_count)
        charges_count_per_trainingsset = np.random.randint(1, self.charges_count + 1, self.N)

        self.charge_positions = []
        for trainingsset_index in range(0, self.N):

            charges_list = []
            m = charges_count_per_trainingsset[trainingsset_index]
            for index in range(0, m):
                charge_tuple = (rows[trainingsset_index * self.charges_count + index],
                                columns[trainingsset_index * self.charges_count + index],)
                charges_list.append(charge_tuple)

            self.charge_positions.append(charges_list)

        self.calc_background()

    def calc_background(self):
        self.permittivity_matrices = []
        angleValuesSet = np.linspace(np.pi / 20.0, np.pi, 20)
        permittivityValueRange = [0.125, 0.25, 0.5, 1.25, 1.5, 2.0]

        angles = np.random.choice(angleValuesSet, size=self.N)
        semiAxes = np.random.randint(1, 21, size=self.N)
        permittivities = np.random.choice(permittivityValueRange, size=self.N)

        centerX = self.gridWidth / 2.0
        centerY = self.gridHeight / 2.0
        majorSemiAxis = self.gridWidth / 4.0
        for angle, minorSemiAxis, permittivity in zip(angles, semiAxes, permittivities):
            background_data = create_ellipsis_grid(self.gridHeight, self.gridWidth, centerX, centerY, majorSemiAxis,
                                                   minorSemiAxis, permittivity, angle)
            self.permittivity_matrices.append(background_data)

    def calc_weight_matrix_list(self, index, charge, charges_list):
        charges_weight_matrix_list = []
        zero_matrix = np.zeros(shape=(len(self.geometry.Y), len(self.geometry.X)))

        m = len(charges_list)

        for charge_index in range(0, m):
            charges_weight_matrix_list.append(calc_charge_weight_matrix(self.geometry, charge, charge_index))

        for index in range(m, self.charges_count):
            charges_weight_matrix_list.append(zero_matrix)

        return charges_weight_matrix_list

    def append_additional_channels(self, index, channel_data_list):
        channel_data_list.append(self.permittivity_matrices[index])

    def get_eps_value(self, index, i, j):
        if len(self.permittivity_matrices[index]) <= i:
            return 1.0
        elif len(self.permittivity_matrices[index][i]) <= j:
            return 1.0

        if self.permittivity_matrices[index][i, j] == 0.0:
            return 1.0
        else:
            return self.permittivity_matrices[index][i, j]

    def get_eps_function(self, index):

        def eps(params):
            i = params[0]  # i
            j = params[1]  # j

            # first, check if average of 2 numbers needs to be taken,
            # eg: i=1.5, j=1.0  => ret = 0.5 * (eps[1.0,j] + eps[2.0,j]
            # #or i=1.5, j=1.5  => ret = 0.5 * (eps[1.0,1.0] + eps[2.0,2.0]

            i_parts = modf(i)
            j_parts = modf(j)
            i_1 = int(i_parts[1])
            i_2 = int(i_parts[1])
            j_1 = int(j_parts[1])
            j_2 = int(j_parts[1])
            i_count = 2
            j_count = 2

            # check for float numbrers
            if (i_parts[0] > 0.2 and i_parts[0] < 0.8):
                i_2 = i_1 + 1
                i_count = 2
            else:
                i_2 = i_1
                i_count = 1

            if (j_parts[0] > 0.2 and j_parts[0] < 0.8):
                j_2 = j_1 + 1
                j_count = 2
            else:
                j_2 = j_1
                j_count = 1

            if i_count == 2 or j_count == 2:
                first_value = 0.0
                if len(self.permittivity_matrices[index]) <= i_1:
                    first_value = 1.0
                elif len(self.permittivity_matrices[index][i_1]) <= j_1:
                    first_value = 1.0
                else:
                    first_value = self.permittivity_matrices[index][i_1, j_1]
                    if first_value == 0.0:
                        first_value = 1.0

                second_value = 0.0
                if len(self.permittivity_matrices[index]) <= i_2:
                    second_value = 1.0
                elif len(self.permittivity_matrices[index][i_2]) <= j_2:
                    second_value = 1.0
                else:
                    second_value = self.permittivity_matrices[index][i_2, j_2]
                    if second_value == 0.0:
                        second_value = 1.0

                return 0.5 * (first_value + second_value)
            else:
                return self.get_eps_value(index, i_1, j_1)

        return eps

    def solve(self, pde, index, charge):
        auxiliaryFunctionsDict = {'eps': self.get_eps_function(index)}
        pde.setAuxiliaryFunctions(auxiliaryFunctionsDict)
        pde.configureGrid()
        return pde.solve(charge)


def make_model(architectureType, gridWidth, gridHeight, charges_count):
    model = models.Sequential()

    if architectureType == 'c31':
        model.add(layers.Conv2D(16, (17, 17), input_shape=(gridWidth, gridHeight, charges_count), activation='relu'))
        model.add(layers.Conv2D(16, (17, 17), activation='relu'))
        model.add(layers.Conv2D(1, (1, 1), activation='relu'))
    elif architectureType == 'c32':
        model.add(layers.Conv2D(64, (17, 17), input_shape=(gridWidth, gridHeight, charges_count), activation='relu'))
        model.add(layers.Conv2D(64, (17, 17), activation='relu'))
        model.add(layers.Conv2D(1, (1, 1), activation='relu'))
    elif architectureType == 'c33':
        model.add(layers.Conv2D(128, (17, 17), input_shape=(gridWidth, gridHeight, charges_count), activation='relu'))
        model.add(layers.Conv2D(128, (17, 17), activation='relu'))
        model.add(layers.Conv2D(1, (1, 1), activation='relu'))
    elif architectureType == 'c51':
        model.add(layers.Conv2D(32, (17, 17), input_shape=(gridWidth, gridHeight, charges_count), activation='relu'))
        model.add(layers.Conv2D(64, (7, 7), activation='relu'))
        model.add(layers.Conv2D(96, (7, 7), activation='relu'))
        model.add(layers.Conv2D(32, (5, 5), activation='relu'))
        model.add(layers.Conv2D(1, (1, 1), activation='relu'))
    elif architectureType == 'c52':
        model.add(layers.Conv2D(64, (17, 17), input_shape=(gridWidth, gridHeight, charges_count), activation='relu'))
        model.add(layers.Conv2D(128, (7, 7), activation='relu'))
        model.add(layers.Conv2D(128, (7, 7), activation='relu'))
        model.add(layers.Conv2D(32, (5, 5), activation='relu'))
        model.add(layers.Conv2D(1, (1, 1), activation='relu'))
    elif architectureType == 'c53':
        model.add(layers.Conv2D(64, (17, 17), input_shape=(gridWidth, gridHeight, charges_count), activation='relu'))
        model.add(layers.Conv2D(128, (7, 7), activation='relu'))
        model.add(layers.Conv2D(256, (7, 7), activation='relu'))
        model.add(layers.Conv2D(64, (5, 5), activation='relu'))
        model.add(layers.Conv2D(1, (1, 1), activation='relu'))
    elif architectureType == 'c54': # rename to c71
        model.add(layers.Conv2D(64, (13, 13), input_shape=(gridWidth, gridHeight, charges_count), activation='relu'))
        model.add(layers.Conv2D(96, (7, 7), activation='relu'))
        model.add(layers.Conv2D(128, (7, 7), activation='relu'))
        model.add(layers.Conv2D(96, (5, 5), activation='relu'))
        model.add(layers.Conv2D(64, (5, 5), activation='relu'))
        model.add(layers.Conv2D(32, (1, 1), activation='relu'))
        model.add(layers.Conv2D(1, (1, 1), activation='relu'))

    elif architectureType == 'c81':  ## Shan Paper Network
        model.add(layers.Conv2D(16, (11, 11), input_shape=(gridWidth, gridHeight, charges_count), activation='relu'))
        model.add(layers.Conv2D(32, (11, 11), activation='relu'))
        model.add(layers.Conv2D(64, (5, 5), activation='relu'))
        model.add(layers.Conv2D(64, (5, 5), activation='relu'))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.Conv2D(64, (1, 1), activation='relu'))
        model.add(layers.Conv2D(1, (1, 1), activation='relu'))
    elif architectureType == 'c82':
        model.add(layers.Conv2D(64, (11, 11), input_shape=(gridWidth, gridHeight, charges_count), activation='relu'))
        model.add(layers.Conv2D(128, (11, 11), activation='relu'))
        model.add(layers.Conv2D(256, (5, 5), activation='relu'))
        model.add(layers.Conv2D(128, (5, 5), activation='relu'))
        model.add(layers.Conv2D(96, (3, 3), activation='relu'))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.Conv2D(32, (1, 1), activation='relu'))
        model.add(layers.Conv2D(1, (1, 1), activation='relu'))

    elif architectureType == 'c83':
        model.add(layers.Conv2D(64, (11, 11), input_shape=(gridWidth, gridHeight, charges_count), activation='relu'))
        model.add(layers.Conv2D(96, (7, 7), activation='relu'))
        model.add(layers.Conv2D(128, (5, 5), activation='relu'))
        model.add(layers.Conv2D(96, (5, 5), activation='relu'))
        model.add(layers.Conv2D(64, (5, 5), activation='relu'))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.Conv2D(32, (3, 3), activation='relu'))
        model.add(layers.Conv2D(1, (1, 1), activation='relu'))

    elif architectureType == 'c91':
        model.add(layers.Conv2D(64, (11, 11), input_shape=(gridWidth, gridHeight, charges_count), activation='relu'))
        model.add(layers.Conv2D(96, (7, 7), activation='relu'))
        model.add(layers.Conv2D(128, (5, 5), activation='relu'))
        model.add(layers.Conv2D(192, (5, 5), activation='relu'))
        model.add(layers.Conv2D(128, (5, 5), activation='relu'))
        model.add(layers.Conv2D(96, (3, 3), activation='relu'))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.Conv2D(32, (1, 1), activation='relu'))
        model.add(layers.Conv2D(1, (1, 1), activation='relu'))


    elif architectureType == 'c92':
        model.add(layers.Conv2D(64, (11, 11), input_shape=(gridWidth, gridHeight, charges_count), activation='relu'))
        model.add(layers.Conv2D(256, (7, 7), activation='relu'))
        model.add(layers.Conv2D(512, (5, 5), activation='relu'))
        model.add(layers.Conv2D(1024, (5, 5), activation='relu'))
        model.add(layers.Conv2D(512, (5, 5), activation='relu'))
        model.add(layers.Conv2D(256, (3, 3), activation='relu'))
        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.Conv2D(64, (1, 1), activation='relu'))
        model.add(layers.Conv2D(1, (1, 1), activation='relu'))

    elif architectureType == 1:
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
    elif architectureType == 101:
        model.add(layers.Reshape((1,64*64*charges_count), input_shape=(gridWidth, gridHeight, charges_count)))
        model.add(layers.Dense(4096, activation='relu'))
        model.add(layers.Reshape((64, 64, 1)))

        #model.add(layers.Dense(1, input_shape=(gridWidth, gridHeight, charges_count)))
        #model.add(layers.Flatten())
        #model.add(layers.Dense(4096))
        #model.add(layers.Reshape((64, 64, 1)))
        #

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
    supportedOptions = "hd:o:N:l:e:s:a:p:r:c:"
    supportLongOptions = ["dir=", "ofile="]
    usage = 'dense_single_charge.py -s <gridSize> -a <architectureType> -e <epochs> -N <count> -d <directory> -o <outputfile> -l <label> -c <charges_count>'

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
            architectureType = arg
        elif opt in ("-s"):
            gridSize = int(arg)
        elif opt in ("-e"):
            epochs = int(arg)
        elif opt in ("-N"):
            count = int(arg)
        elif opt in ("-c"):
            charges_count = int(arg)
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

    return outputFile, outputDirectory, gridSize, count, epochs, architectureType, charges_count, persistFile, readFile


if __name__ == '__main__':

    print(tf.VERSION)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)

    charges_count = 2

    try:
        outputFile, outputDirectory, gridSize, count, epochs, architectureType, charges_count, persistFile, readFile = parseArguments(sys.argv[1:])

    except:
        outputFile = 'test_data'
        outputDirectory= '.'
        gridSize = 64
        count = 1000
        epochs = 2
        architectureType = 'c31'
        persistFile = None
        #persistFile = 'my_dump.pickle'
        readFile = None
        #readFile = 'my_dump.pickle'

    if outputFile != None:
        fileName = makeFilename(outputDirectory, outputFile)
        print('Write output to:', os.path.abspath(fileName))

    gridSize = float(gridSize)


    model = make_model(architectureType, (int)(gridSize), (int)(gridSize), charges_count)

    if readFile == None:

        poisson_equation_with_matter = "div(eps(r) * grad( u(r) ))"
        pde = setupPDE_vector_calculus_without_configure(gridSize, poisson_equation_with_matter)

        #fill_strategy = TrainingSet_CreationStrategy_Full_SingleCharge(pde.geometry)
        #fill_strategy = TrainingSet_CreationStrategy_N_SingleCharge(pde.geometry, N=count)
        #fill_strategy = TrainingSet_CreationStrategy_N_MultiCharge(pde.geometry, N=count, charges_count=charges_count)
        #fill_strategy = TrainingSet_CreationStrategy_m_MultiCharge_Zeros(pde.geometry, N=count, charges_count=charges_count)
        #fill_strategy = TrainingSet_CreationStrategy_m_MultiCharge_Duplicates(pde.geometry, N=count, charges_count=charges_count)
        fill_strategy = TrainingSet_CreationStrategy_m_MultiCharge_EllipticMatter_Zeros(pde.geometry, N=count,
                                                                                        charges_count=charges_count)

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

            if outputFile == None:
                sys.exit(0)

    else:
        with open(readFile, 'rb') as f:

            tuple_to_load = pickle.load(f)
            fill_strategy = tuple_to_load[0]
            inputset_duration = tuple_to_load[1]
            solutionset_duration = tuple_to_load[2]


    #s = fill_strategy.solutions.reshape((fill_strategy.solutions.shape[0], pde.geometry.numX, (int)(pde.geometry.numY), -1))

    print(fill_strategy.input_set.shape)
    print(fill_strategy.solutions.shape)

    trainings_count = int(count * 0.7)
    train_input = fill_strategy.input_set[0:trainings_count]
    train_output = fill_strategy.solutions[0:trainings_count, 16:48, 16:48, :]
    print(trainings_count, train_input.shape, train_output.shape)

    validation_count = int(count * 0.05)
    validation_input = fill_strategy.input_set[trainings_count:trainings_count+validation_count]
    validation_output = fill_strategy.solutions[trainings_count:trainings_count+validation_count, 16:48, 16:48, :]
    print(validation_count, validation_input.shape, validation_output.shape)

    test_count = int(count * 0.25)
    test_input = fill_strategy.input_set[trainings_count+validation_count:]
    test_output = fill_strategy.solutions[trainings_count+validation_count:, 16:48, 16:48, :]
    print(test_count, test_input.shape, test_output.shape)


    start = time.time()

    learn(model, epochs, train_input, train_output, validation_input, validation_output)

    learning_duration = time.time() - start
    print('duration for learning:',  learning_duration)

    prediction = model.predict(test_input)

    prediction_full = model.predict(fill_strategy.input_set)

    #print(prediction.shape)

    errors = calc_square_error_for_list(test_output, prediction)
    errors_full = calc_square_error_for_list(fill_strategy.solutions[:, 16:48, 16:48, :], prediction_full)
    #print(errors)

    saveModel(model, fileName)
    trainingsSetConfig = TrainingsSetConfig(gridSize, gridSize, "1", epochs, count, fill_strategy.charges, datetime.datetime.utcnow())
    write_data_configuration(fileName, trainingsSetConfig)

    results = Results(inputset_duration, solutionset_duration, learning_duration, errors)
    write_results(fileName+'_results', results)

    #results_full = Results(inputset_duration, solutionset_duration, learning_duration, errors_full)
    #write_results(fileName + '_results_full', results_full)

    showGraph = 0
    if platform.uname()[1] != 'qcd':
        showGraph = 1

    if showGraph:
        from sources.experiments.fdm_helper import plotSurface
        import matplotlib.pyplot as plt

        for i in range(0, 5):
            plotSurface(pde.geometry.X[16:48, 16:48], pde.geometry.Y[16:48, 16:48], prediction[i,:,:,0])
            plt.show()

