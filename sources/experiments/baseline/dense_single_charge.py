from keras import models, layers, losses
import keras.backend as K

from sources.experiments.charges_generators import make_central_charge, make_single_charge, make_n_fold_charge_from_list
from sources.experiments.fdm_helper import plotSurface
from sources.pdesolver.pde.PDE import PDEExpressionType, PDE

import numpy as np
import matplotlib.pyplot as plt


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
            rows = np.random.random_integers(margin, self.gridHeight-margin-1, self.N)
            columns = np.random.random_integers(margin, self.gridWidth-margin-1, self.N)

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


def make_model(gridWidth, gridHeight, charges_count):
    model = models.Sequential()
    model.add(layers.Dense(92, input_shape=(gridWidth,gridHeight,charges_count), activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(1, activation='relu'))

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

def learn(model, train_input, train_output, validation_input, validation_output):

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

    epochs = 10


    history = model.fit(x=train_input, y=train_output, epochs=epochs,
                        batch_size=1,
                        validation_data=(validation_input, validation_output)
                        )


if __name__ == '__main__':

    gridSize = 64.0
    charges_count = 3

    model = make_model((int)(gridSize), (int)(gridSize), charges_count)

    poisson_equation = "div(grad( u(r) ))"
    pde = setupPDE_vector_calculus(gridSize, poisson_equation)

    #fill_strategy = TrainingSet_CreationStrategy_Full_SingleCharge(pde.geometry)
    #-> fill_strategy = TrainingSet_CreationStrategy_N_SingleCharge(pde.geometry, N=1000)
    fill_strategy = TrainingSet_CreationStrategy_N_MultiCharge(pde.geometry, N=1000, charges_count=charges_count)

    fill_strategy.create_inputSet()
    fill_strategy.create_solutionSet(pde)

    s = fill_strategy.solutions.reshape((fill_strategy.solutions.shape[0], pde.geometry.numX, (int)(pde.geometry.numY), -1))

    print(fill_strategy.input_set.shape)
    print(s.shape)

    train_input = fill_strategy.input_set[0:950]
    train_output = s[0:950]

    validation_input = fill_strategy.input_set[:-50]
    validation_output = s[:-50]

    learn(model, train_input, train_output, validation_input, validation_output)


    prediction = model.predict(validation_input[0:10])

    print(prediction.shape)

    showGraph = 1

    if showGraph:
        plotSurface(pde.geometry.X, pde.geometry.Y, prediction[0,:,:,0])
        plt.show()

