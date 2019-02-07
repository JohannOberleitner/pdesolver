import time
import numpy as np
import json

from sources.experiments.data_generation.trainings_data import TrainingsSet, TrainingsSetEncoder, \
    as_TrainingsSet
from sources.pdesolver.finite_differences_method.FiniteDifferencesSolver_V2 import FiniteDifferencesMethod3, \
    GridConfiguration, ConstantGridValueProvider, FunctionGridValueProvider
from sources.pdesolver.finite_differences_method.boundaryconditions import RectangularBoundaryCondition
from sources.pdesolver.finite_differences_method.charge_distribution import ChargeDistribution
from sources.pdesolver.finite_differences_method.geometry import Geometry
from sources.pdesolver.finite_differences_method.rectangle import Rectangle

from sources.experiments.ellipsis_data_support.make_ellipsis import create_ellipsis_grid

def make_charges_in_line(g, count, charge, startX, startY, endX, endY):
    charges = ChargeDistribution(g)
    deltaX = (endX-startX)/count
    deltaY = (endY-startY)/count
    for i in range(count):
        charges.add((int)(startX + i * deltaX), (int)(startY + i*deltaY), charge)

    return charges


def make_representation(iterationNumber, gridWidth, gridHeight, innerGridWidth, innerGridHeight, semiMajorAxis, semiMinorAxis, permeability, angle):
    eps_data = create_ellipsis_grid(gridWidth, gridHeight, innerGridWidth, innerGridHeight, semiMajorAxis, semiMinorAxis, permeability, angle)
    s   = ('\nnr={0}\n'\
        + 'major={1}\n'\
        + 'minor={2}\n' \
        + 'eps={3}\n' \
        + 'angle={4}\n'\
        + 'eps_data=\n{5}'
           ).format(iterationNumber, semiMajorAxis, semiMinorAxis, permeability, angle, eps_data)
    return s


def generate_permittivity_function(permittivity_matrix):

    def eps(i,j):
        if (len(permittivity_matrix) <= i):
            return 1.0
        elif (len(permittivity_matrix[i]) <= j):
            return 1.0

        if (permittivity_matrix[i,j] == 0.0):
            return 1.0
        else:
            return permittivity_matrix[i,j]

    return eps

def make_permeability_matrix(test_set):
    count = len(test_set['majorSemiAxis'])
    permeability_matrix = []
    for i in range(count):
        permeability_matrix.append(
            make_representation(i + 1, 64, 64, 32, 32, test_set['majorSemiAxis'][i], test_set['minorSemiAxis'][i],
                                test_set['permeabilities'][i], test_set['angles'][i]))
    #print(permeability_matrix)
    return permeability_matrix

def make_permeability_matrix_one(gridWidth, gridHeight, innerGridWidth, innerGridHeight, majorSemiAxis, minorSemiAxis, permeability, angle):
    eps_data = create_ellipsis_grid(gridWidth, gridHeight, innerGridWidth, innerGridHeight, majorSemiAxis,\
                                    minorSemiAxis, permeability, angle)
    return eps_data

def make_finite_differences_poisson_equation_in_matter(eps):
    gridConfig = GridConfiguration()
    gridConfig.add(FunctionGridValueProvider((lambda i, j: 0.5 * (eps(i, j) + eps(i + 1, j)))), 1, 0)
    gridConfig.add(FunctionGridValueProvider((lambda i, j: 0.5 * (eps(i, j) + eps(i - 1, j)))), -1, 0)
    gridConfig.add(FunctionGridValueProvider((lambda i, j: 0.5 * (eps(i, j) + eps(i, j + 1)))), 0, 1)
    gridConfig.add(FunctionGridValueProvider((lambda i, j: 0.5 * (eps(i, j) + eps(i, j - 1)))), 0, -1)
    gridConfig.add(FunctionGridValueProvider((lambda i, j:
                                               -0.5 * (4 * eps(i, j) + eps(i + 1, j) + eps(i - 1, j) + eps(i,j + 1) + eps(
                                                   i, j - 1)))), 0, 0)
    return gridConfig



def generate_text_presentation(iterationNumber, dataElement, values, errors):
    s = ('\nnr={0}\n' \
         + 'major={1}\n' \
         + 'minor={2}\n' \
         + 'eps={3}\n' \
         + 'angle={4}\n')\
        .format(iterationNumber, dataElement.get_semiMajorAxis(), dataElement.get_semiMinorAxis(), dataElement.get_permittivity(), dataElement.get_angle())

    eps_string = ('eps_data={0}\n').format(dataElement.get_permittivity_matrix())
    #values_string = ('u={0}\n').format(values)
    values_string = ('u={0}\n').format(np.array2string(values, precision=4, separator=',', suppress_small=True))
    errors_string = ('err={0}\n').format(errors)

    return s + eps_string + values_string + errors_string

def calculate_random_parameters(count):
    angleValuesSet = np.linspace(np.pi / 20.0, np.pi, 20)
    permeabilityValuesSet = [0.125, 0.25, 0.5, 2., 4., 6.]

    angles = np.random.choice(angleValuesSet, size=count)
    semiAxes = np.random.randint(1, 21, size=count)
    permeabilities = np.random.choice(permeabilityValuesSet, size=count)

    return TrainingsSet([16]*count, semiAxes/2, permeabilities, angles)

if __name__ == '__main__':

    np.set_printoptions(threshold=np.nan)

    count = 2 # 10000
    index = 0

    # setup for finite differences
    delta = 1.0
    rect = Rectangle(0, 0, 64.0, 64.0)
    g = Geometry(rect, delta)
    boundaryCondition = RectangularBoundaryCondition(g)
    charges = make_charges_in_line(g, 11, -10.0, 16.0, 20.0, 48.0, 20.0)

    start = time.clock()

    data_parameters = calculate_random_parameters(count=count)

    for dataElement in data_parameters:
        dataElement.calc_permittivity_matrix(64,64,32,32)

    s1 = json.dumps(data_parameters, cls=TrainingsSetEncoder)
    s2 = '{"__TrainingsSet__":true,"entry":'+s1+'}'
    reloaded_data_parameters = json.loads(s1, object_hook=as_TrainingsSet)

    index = 1

    for dataElement in data_parameters:
        #dataElement.calc_permeability_matrix(64,64,32,32)
        eps = generate_permittivity_function(dataElement.get_permittivity_matrix())
        gridConfig = make_finite_differences_poisson_equation_in_matter(eps)
        fdm = FiniteDifferencesMethod3(g, boundaryCondition, gridConfig, charges)
        fdm.solve()
        fdm.calcMetrices()

        s = generate_text_presentation(index, dataElement, fdm.values, fdm.error)

        print('Solved:',index)
        index = index+1
        print(s)

    duration = time.clock() - start
    print('Total duration for {0} data elements:{1}'.format(count, duration))

# executables
# 1. TrainingsSet-InputGenerator -> writes into 1 file or directory/file  - InputData-xx.data
#    make_trainingsset_input_scenario1
# 2. NumericSolver               -> reads one file/directory -> writes trainingsset_labels into file/directory - Solution-xx.data
#    solve_poission2d
#      -> requires reader for trainingsset
# 3. read_trainingsset