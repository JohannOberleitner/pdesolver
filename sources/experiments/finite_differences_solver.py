import sys, getopt

import json

from sources.experiments.data_generation.results_data import ResultsSet, ResultsSetEncoder, as_ResultsSet
from sources.experiments.data_generation.trainings_data import as_TrainingsSet
from sources.pdesolver.finite_differences_method.FiniteDifferencesSolver_V2 import GridConfiguration, \
    FunctionGridValueProvider, FiniteDifferencesMethod3
from sources.pdesolver.finite_differences_method.boundaryconditions import RectangularBoundaryCondition
from sources.pdesolver.finite_differences_method.charge_distribution import ChargeDistribution
from sources.pdesolver.finite_differences_method.geometry import Geometry
from sources.pdesolver.finite_differences_method.rectangle import Rectangle


def make_charges_in_line(g, count, charge, startX, startY, endX, endY):
    charges = ChargeDistribution(g)
    deltaX = (endX-startX)/count
    deltaY = (endY-startY)/count
    for i in range(count):
        charges.add((int)(startX + i * deltaX), (int)(startY + i*deltaY), charge)

    return charges

def load_chargedistribution(g, dataset):
    charges = ChargeDistribution(g)
    charges.addList(dataset.chargesList)
    return charges

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


def calcSolutions(inputDataSet):

    delta = 1.0
    rect = Rectangle(0, 0, inputDataSet.geometry.gridWidth, inputDataSet.geometry.gridHeight)
    g = Geometry(rect, delta)
    boundaryCondition = RectangularBoundaryCondition(g)
    charges = load_chargedistribution(g, inputDataSet)

    index = 1

    resultsSet = ResultsSet(label=inputDataSet.label)

    for dataElement in inputDataset:
        eps = generate_permittivity_function(dataElement.get_permittivity_matrix())
        gridConfig = make_finite_differences_poisson_equation_in_matter(eps)

        fdm = FiniteDifferencesMethod3(g, boundaryCondition, gridConfig, charges)
        fdm.solve()
        fdm.calcMetrices()

        resultsSet.add(fdm.results)

        #s = json.dumps(fdm.results.tolist())
        #print(s)


        #s = generate_text_presentation(index, dataElement, fdm.values, fdm.error)
        print('Solved:',index)
        index = index+1

    return resultsSet

def loadInputDataset(filename):
    file = open(filename, mode='r')
    dataset = json.load(file, object_hook=as_TrainingsSet)
    return dataset

def write_dataset(filename, resultsSet):
    if filename == None:
        s = json.dumps(resultsSet, cls=ResultsSetEncoder)
        print(s)
    else:
        file = open(filename, 'w')
        json.dump(resultsSet, file, cls=ResultsSetEncoder)
        file.close()

def parseArguments(argv):
    supportedOptions = "hi:o:"
    supportLongOptions = ["ifile=", "ofile="]
    usage = 'finite_differences_solver.py -i <inputfile> -o <outputfile>'

    inputFile = None
    outputFile = None

    try:
        opts, args = getopt.getopt(argv, supportedOptions, supportLongOptions)
    except getopt.GetoptError:
        print(usage)
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print(usage)
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputFile = arg
        elif opt in ("-o", "--ofile"):
            outputFile = arg

    return inputFile, outputFile


#def loadFileAgain(file):
#    file = open(file, mode='r')
#    data = json.load(file, object_hook=as_ResultsSet)
#    i=1


if __name__ == '__main__':

    inputFile, outputFile = parseArguments(sys.argv[1:])
    inputDataset = loadInputDataset(inputFile)

    try:
        resultsSet = calcSolutions(inputDataset)
        write_dataset(outputFile, resultsSet)

    except TypeError as terr:
        print(terr)
    except AttributeError as aerr:
        print(aerr)

    except:
        print("Unexpected error:", sys.exc_info()[0])

#    loadFileAgain(outputFile)

