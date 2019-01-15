import time
import numpy as np

from sources.experiments.ellipsis_data_support.make_ellipsis import create_ellipsis_grid

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


if __name__ == '__main__':

    np.set_printoptions(threshold=np.nan)

    count = 2000

    start = time.clock()

    angleValuesSet = np.linspace(np.pi / 20.0, np.pi, 20)
    permeabilityValuesSet = [0.125, 0.25, 0.5, 2., 4., 6.]

    angles = np.random.choice(angleValuesSet, size=count)
    semiAxes = np.random.randint(1, 21, size=count)
    permeabilities = np.random.choice(permeabilityValuesSet, size=count)

    for i in range(0, count):
        s = make_representation(i+1, 64, 64, 32, 32, 16, semiAxes[i]/2, permeabilities[i], angles[i])
        print(s)
        #target = create_ellipsis_grid(64, 64, 32, 32, 16, semiAxes[i] / 2, permeabilities[i], angles[i])
        #print(target)

    duration = time.clock() - start
    print(duration)

    #    xData = []
    #    yData = []

    #    for x in range(0, 64):
    #        for y in range(0, 64):
    #            item = target.item((x, y))
    #            if item > 0:
    #                xData.append(x)
    #                yData.append(y)

    #    plt.scatter(xData, yData)
    #    plt.axis([0, 64, 0, 64])

    #plt.show()