
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

def create_ellipsis_grid(targetWidth, targetHeight, centerX, centerY, semiAxisMajor, semiAxisMinor, rotationAngle=0):

    matrix = np.zeros( (targetWidth, targetHeight) )

    a2 = semiAxisMajor**2
    b2 = semiAxisMinor**2

    for xPos in range(0, targetWidth):
        for yPos in range(0, targetHeight):

            x = (xPos - centerX)
            y = (yPos - centerY)

            xdash = x*np.cos(-rotationAngle) - y*np.sin(-rotationAngle)
            ydash = x*np.sin(-rotationAngle) + y*np.cos(-rotationAngle)

            value = (xdash*xdash)/a2 + (ydash*ydash)/b2

            if value <= 1:
                matrix[xPos,yPos] = 1.0

    return matrix


if __name__ == '__main__':

    np.set_printoptions(threshold=np.nan)


    #target = create_ellipsis(64, 64, 32, 32, 10, 5, np.deg2rad(45))
    target = create_ellipsis_grid(64, 64, 32, 32, 16, 10, np.deg2rad(45))
    #print(target)

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