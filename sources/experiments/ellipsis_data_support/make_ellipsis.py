import numpy as np
import matplotlib.pyplot as plt

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


def plot_ellipsis(ax, ellipsis_data, title):
    xData = []
    yData = []

    ax.set_title(title, fontsize=9)
    for x in range(0, len(ellipsis_data)):
        for y in range(0, len(ellipsis_data[x])):
            item = ellipsis_data.item((x, y))
            if item > 0:
                xData.append(x)
                yData.append(y)

                ax.scatter(xData, yData)
                ax.axis([0, len(ellipsis_data), 0, len(ellipsis_data[0])])