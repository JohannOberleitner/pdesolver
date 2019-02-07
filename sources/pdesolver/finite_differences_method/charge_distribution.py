import numpy as np

class ChargeDistribution:
    def __init__(self, geometry):
        self.geometry = geometry
        self.Charges = np.zeros_like(self.geometry.X)
        self.chargesList = []

    def add(self, x, y, value):
        i, j = self.geometry.indexFromCoords(x, y)
        self.Charges[i, j] = value
        self.chargesList.append((x,y,value))

    def addList(self, chargesList):
        self.chargesList = chargesList
        for item in chargesList:
            self.Charges[item[0], item[1]] = item[2]

    def get(self, column, row):
        assert column >= 0 and column <= self.geometry.numX
        assert row >= 0 and row <= self.geometry.numY
        # because numpy's meshgrid returns the arrays in different order change this here:
        return self.Charges[row, column]

    def calcAverage(self, col, row):
        return (self.get(col + 1, row) +
                self.get(col - 1, row) +
                self.get(col, row + 1) +
                self.get(col, row - 1)) / 4.0