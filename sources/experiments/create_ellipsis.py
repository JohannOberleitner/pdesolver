import time
import numpy as np
import matplotlib.pyplot as plt

def create_ellipsis(targetWidth, targetHeight, centerX, centerY, semiAxisMajor, semiAxisMinor, rotationAngle=0):

    xData = []
    yData = []
    #xData = [0] * targetWidth
    #yData = [0] * targetHeight

    delta = 0.1

    for x in np.linspace(0, semiAxisMajor, endpoint=True):
        y = np.sqrt(semiAxisMinor**2 * (1.0-x**2/(semiAxisMajor**2)))
        xrot = x #x*np.cos(rotationAngle) - y*np.sin(rotationAngle)
        yrot = y #®x*np.sin(rotationAngle) + y*np.cos(rotationAngle)
        xData.append(xrot+centerX)
        yData.append(yrot+centerY)

    for i in range(0,len(xData)):
        xData.append(-(xData[i]-centerX)+centerX)
        yData.append(yData[i])

    for i in range(0,len(xData)):
        xData.append(xData[i])
        yData.append(-(yData[i]-centerY)+centerY)

    for i in range(0,len(xData)):
        x = xData[i]-centerX
        y = yData[i]-centerY
        xData[i] = x*np.cos(rotationAngle) - y*np.sin(rotationAngle) + centerX
        yData[i] = x*np.sin(rotationAngle) + y*np.cos(rotationAngle) + centerY

    #for i in range(0, len(xData)):
    #       xData.append(-(xData[i] - centerX) + centerX)
    #    yData.append(yData[i])

    return (xData,yData)

def create_ellipsis_grid(targetWidth, targetHeight, centerX, centerY, semiAxisMajor, semiAxisMinor, permeability=1.0, rotationAngle=0):

    matrix = np.zeros( (targetWidth, targetHeight) )

    a2 = semiAxisMajor**2
    b2 = semiAxisMinor**2
    cosAngle = np.cos(-rotationAngle)
    sinAngle = np.sin(-rotationAngle)

    # raster around all points
    for xPos in range(0, targetWidth):
        for yPos in range(0, targetHeight):

            # shift the position by midldle as ellipse formula is centered around origin
            x = (xPos - centerX)
            y = (yPos - centerY)

            # undo the rotation with negative angle
            xdash = x*cosAngle - y*sinAngle
            ydash = x*sinAngle + y*cosAngle

            value = (xdash**2)/a2 + (ydash**2)/b2

            if value <= 1:
                matrix[xPos,yPos] = permeability

    return matrix


if __name__ == '__main__':

    np.set_printoptions(threshold=np.nan)

    count = 2000

    start = time.clock()

    angleValuesSet = np.linspace(np.pi/20.0, np.pi, 20)
    permittivityValuesSet = [0.125, 0.25, 0.5, 2.,4.,6.]

    angles = np.random.choice(angleValuesSet, size=count)
    semiAxes = np.random.randint(1,21,size=count)
    permittivity = np.random.choice(permittivityValuesSet, size=count)

    for i in range(0,count):
        target = create_ellipsis_grid(64, 64, 32, 32, 16, semiAxes[i]/2, permittivity[i], angles[i])
    #print(target)

    duration = time.clock() - start
    print(duration)

    xData = []
    yData = []

    for x in range(0,64):
        for y in range(0,64):
            item = target.item((x,y))
            if item > 0:
                xData.append(x)
                yData.append(y)



    plt.scatter(xData, yData)
    plt.axis([0,64,0,64])

    plt.show()