import numpy as np

from sources.experiments.ellipsis_data_support.make_ellipsis import create_ellipsis_grid

def make_test_set(count):
    angleValuesSet = np.linspace(np.pi / 20.0, np.pi, 20)
    permeabilityValuesSet = [0.125, 0.25, 0.5, 2., 4., 6.]

    angles = np.random.choice(angleValuesSet, size=count)
    semiAxes = np.random.randint(1, 21, size=count)
    permeabilities = np.random.choice(permeabilityValuesSet, size=count)

    return {'majorSemiAxis':[16]*count, 'minorSemiAxis':semiAxes/2.0, 'permeabilities':permeabilities, 'angles':angles}


def make_representation(iterationNumber, gridWidth, gridHeight, innerGridWidth, innerGridHeight, semiMajorAxis, semiMinorAxis, permeability, angle):
    eps_data = create_ellipsis_grid(gridWidth, gridHeight, innerGridWidth, innerGridHeight, semiMajorAxis, semiMinorAxis, permeability, angle)
    return eps_data
