import numpy as np


class Geometry:

    def __init__(self, rect, delta):
        self.delta = delta
        self.rect = rect

        numX = (int)(self.rect.width/delta) #+1
        numY = (int)(self.rect.height/delta) #+1
        self.x = np.linspace(self.rect.x1, self.rect.x2, numX, dtype=np.double)
        self.y = np.linspace(self.rect.y1, self.rect.y2, numY, dtype=np.double)

        self.X, self.Y = np.meshgrid(self.x, self.y)
        self.numX = len(self.X[0])
        self.numY = len(self.X)
        #print(self.X.shape, self.Y.shape, self.numX, self.numY)

    def indexFromCoords(self, x, y):
        i = (int)((x - self.rect.x1) / self.delta)
        j = (int)((y - self.rect.y1) / self.delta)
        # because numpy's meshgrid returns the arrays in different order change this here:
        return (j,i)

    def coordFromHorizontalIndex(self, column):
        return self.x + column * self.delta

    def coordFromVerticalIndex(self, row):
        return self.y + row * self.delta

    def coordFromIndices(self, column, row):
        return self.coordFromHorizontalIndex(self, column), self.coordFromVerticalIndex(self, row)


