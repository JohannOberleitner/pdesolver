import sys, getopt
import time
import json

import numpy as np
import os.path

from sources.pdesolver.finite_differences_method.charge_distribution import ChargeDistribution
from sources.pdesolver.finite_differences_method.geometry import Geometry
from sources.pdesolver.finite_differences_method.rectangle import Rectangle

from sources.experiments.data_generation.trainings_data import TrainingsSet, TrainingsSetEncoder


def calculate_random_parameters(count):
    angleValuesSet = np.linspace(np.pi / 20.0, np.pi, 20)
    permittivityValueRange = [0.125, 0.25, 0.5, 2., 4., 6.]

    angles = np.random.choice(angleValuesSet, size=count)
    semiAxes = np.random.randint(1, 21, size=count)
    permittivities = np.random.choice(permittivityValueRange, size=count)

    return TrainingsSet([16]*count, semiAxes/2, permittivities, angles)

def write_dataset(filename, trainingsSet):
    if filename == None:
        s = json.dumps(trainingsSet, cls=TrainingsSetEncoder)
        print(s)
    else:
        file = open(filename, 'w')
        json.dump(trainingsSet, file, cls=TrainingsSetEncoder)
        file.close()

def generate_dataset(count):

    np.set_printoptions(threshold=np.nan)

    index = 0

    # setup for finite differences
    delta = 1.0
    rect = Rectangle(0, 0, 64.0, 64.0)
    g = Geometry(rect, delta)

    start = time.clock()

    dataset = calculate_random_parameters(count=count)

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
    if os.path.exists(filepath):
        os.remove(filepath)

def parseArguments(argv):
    supportedOptions = "hd:o:c:"
    supportLongOptions = ["dir=", "ofile="]
    usage = 'make_trainingsset_input_scenario1.py -c <count> -d <directory> -o <outputfile>'

    outputDirectory = '.'
    outputFile = None
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

    return outputFile, outputDirectory, count

if __name__ == '__main__':

    filename, directory, count = parseArguments(sys.argv[1:])
    if filename != None:
        makeTargetPath(directory)

    filepath = makeFilename(directory, filename)
    deleteExistingFile(filepath)

    generatedDataset = generate_dataset(count)
    write_dataset(filename, generatedDataset)
