
import numpy as np
import matplotlib.pyplot as plt

def create_ellipsis(targetWidth, targetHeight, centerX, centerY, semiAxisMajor, semiAxisMinor, rotationAngle=0):

    xData = []
    yData = []
    #xData = [0] * targetWidth
    #yData = [0] * targetHeight

    for x in range(semiAxisMajor+1):
        xData.append(x+centerX)
        y = np.sqrt(semiAxisMinor**2 * (1.0-x**2/(semiAxisMajor**2)))
        yData.append(y+centerY)

    for i in range(0,len(xData)):
        xData.append(-(xData[i]-centerX)+centerX)
        yData.append(yData[i])

    for i in range(0,len(xData)):
        xData.append(xData[i])
        yData.append(-(yData[i]-centerY)+centerY)

    #for i in range(0, len(xData)):
    #       xData.append(-(xData[i] - centerX) + centerX)
    #        yData.append(yData[i])


    return (xData,yData)


if __name__ == '__main__':

    target = create_ellipsis(64, 64, 32, 32, 10, 2)
    print(target)

    plt.scatter(target[0], target[1])

    plt.show()