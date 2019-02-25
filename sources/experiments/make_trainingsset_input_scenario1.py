import sys, getopt
import time
import json

import numpy as np
import os.path

from sources.pdesolver.finite_differences_method.charge_distribution import ChargeDistribution
from sources.pdesolver.finite_differences_method.geometry import Geometry
from sources.pdesolver.finite_differences_method.rectangle import Rectangle

from sources.experiments.data_generation.trainings_data import TrainingsSet, TrainingsSetEncoder, TrainingsSetGeometry



def calculate_random_parameters(count, geometry, chargesList=None, label=None):

    angleValuesSet = np.linspace(np.pi / 20.0, np.pi, 20)
    permittivityValueRange = [0.125, 0.25, 0.5, 2., 4., 6.]

    angles = np.random.choice(angleValuesSet, size=count)
    semiAxes = np.random.randint(1, 21, size=count)
    permittivities = np.random.choice(permittivityValueRange, size=count)

    return TrainingsSet(geometry, chargesList, [geometry.innerGridWidth/2.0]*count, semiAxes/2, permittivities, angles, label=label)

def make_charges_in_line(g, count, chargeValue, startX, startY, endX, endY):
    charges = ChargeDistribution(g)
    deltaX = (endX-startX)/count
    deltaY = (endY-startY)/count
    for i in range(count):
        charges.add((int)(startX + i * deltaX), (int)(startY + i*deltaY), chargeValue)

    return charges

def make_charges(trainingsSetGeometry):
    rect = Rectangle(0.0, 0.0, trainingsSetGeometry.gridWidth, trainingsSetGeometry.gridHeight)
    delta = 1.0
    g = Geometry(rect, delta)
    countCharges = 11
    chargeValue = -10.0
    yCoord = 20.0
    xCoordLeft = rect.midX() - (trainingsSetGeometry.gridWidth - trainingsSetGeometry.innerGridWidth)/2.0
    xCoordRight = rect.midX() +(trainingsSetGeometry.gridWidth - trainingsSetGeometry.innerGridWidth)/2.0
    return make_charges_in_line(g, countCharges, chargeValue, xCoordLeft, yCoord, xCoordRight, yCoord)

def write_dataset(filename, trainingsSet):
    if filename == None:
        s = json.dumps(trainingsSet, cls=TrainingsSetEncoder)
        print(s)
    else:
        file = open(filename, 'w')
        json.dump(trainingsSet, file, cls=TrainingsSetEncoder)
        file.close()

def generate_dataset(count, trainingsSetGeometry, charges, label):

    #np.set_printoptions(threshold=np.nan)

    index = 0

    # setup for finite differences
    rect = Rectangle(0.0, 0.0, trainingsSetGeometry.gridWidth, trainingsSetGeometry.gridHeight)
    delta = 1.0
    g = Geometry(rect, delta)

    start = time.clock()

    dataset = calculate_random_parameters(count, trainingsSetGeometry, chargesList=charges.chargesList, label=label)

    for dataElement in dataset:
        dataElement.calc_permittivity_matrix(64,64,32,32)

    duration = time.clock() - start
    print('Total duration for generating {0} dataset elements:{1}'.format(count, duration))

    return dataset


def makeTargetPath(directory):
    if directory != '.' and not os.path.exists(directory):
        os.makedirs(directory)

def makeFilename(directory,filename):
    if filename == None:
        return None
    else:
        return os.path.join(directory,filename)

def deleteExistingFile(filepath):
    if filepath and os.path.exists(filepath):
        os.remove(filepath)

def parseArguments(argv):
    supportedOptions = "hd:o:c:l:"
    supportLongOptions = ["dir=", "ofile="]
    usage = 'make_trainingsset_input_scenario1.py -c <count> -d <directory> -o <outputfile> -l <label>'

    outputDirectory = '.'
    outputFile = None
    label = None
    count = 20


    try:
        opts, args = getopt.getopt(argv, supportedOptions, supportLongOptions)
    except getopt.GetoptError:
        print(usage)
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print(usage)
            sys.exit()
        elif opt in ("-c"):
            count = int(arg)
        elif opt in ("-d", "--dir"):
            outputDirectory = arg
        elif opt in ("-o", "--ofile"):
            outputFile = arg
        elif opt in ("-l", "--label"):
            label = arg

    return outputFile, outputDirectory, count, label

if __name__ == '__main__':

    filename, directory, count, label = parseArguments(sys.argv[1:])
    if filename != None:
        makeTargetPath(directory)

    filepath = makeFilename(directory, filename)
    deleteExistingFile(filepath)

    geometry = TrainingsSetGeometry([64.0,64.0,32.0,32.0])
    charges = make_charges(geometry)
    generatedDataset = generate_dataset(count, geometry, charges, label)
    write_dataset(filename, generatedDataset)
