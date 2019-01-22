import numpy as np

class GridValueProvider:
    def __init__(self, ):
        pass

    def getValue(self, coordinates, row, col):
        pass

    def apply(self, matrix):
        matrix

class ConstantGridValueProvider(GridValueProvider):
    """A value provider that delivers always a constant value initialized once at construction time
       regardless of the coordinates
    """

    def __init__(self, value):
        """
        :param value: initialization value for the constant value
        """
        self.constantValue = value

    def getValue(self, coordinates, row, col):
        """

        :param coordinates: ignored
        :param row: determines the row for which a value shall be returned
        :param col: determines the column for which a value shall be returned
        :return: the value that will be returned - constant for this value provider
        """
        return self.constantValue

class FunctionGridValueProvider(GridValueProvider):


    def __init__(self, fn):
        self.fn = fn

    def getValue(self, coordinates, row, col):
        return self.fn(col, row)

class GridConfiguration:
    """A class that provides finite difference values for an equation for a grid.

       The values are provided by valueProviders that are added with the add method.
    """
    def __init__(self):
        self.valueProviders = []

    def add(self, valueProvider,xOffset,yOffset):
        """Adds a valueProvider

        :param valueProvider: valueProvider to be added
        :param xOffset: the x coordinate for which this value provider will be used
        :param yOffset: the y coordinate for which this value provider will be used
        :return:
        """
        providerWithCoordinates = dict(x=xOffset, y=yOffset, provider=valueProvider)
        self.valueProviders.append(providerWithCoordinates)

    def setValues(self, matrix, row, col, indexCallback):
        """Fills the matrix on the position row, col with values provided by a value provider

        :param matrix:
        :param row:
        :param col:
        :param indexCallback:
        :return:
        """
        #i=0
        for providerWithCoordinates in self.valueProviders:

            #print(providerWithCoordinates['provider'], providerWithCoordinates['x'], providerWithCoordinates['y'])

            #coordinates = dict(x=providerWithCoordinates['x'], y=providerWithCoordinates['y'])
            value = providerWithCoordinates['provider'].getValue(providerWithCoordinates, row, col)
            matrixIndices = indexCallback(row, col, providerWithCoordinates['x'], providerWithCoordinates['y'])
            #print(value, matrixIndices)
            rowIndex = matrixIndices['rowIndex']
            colIndex = matrixIndices['colIndex']
            if rowIndex >= 0 and rowIndex < len(matrix) and colIndex >= 0 and colIndex < len(matrix[0]):
                #print('write:',i)
                matrix[rowIndex, colIndex] = value

            #i=i+1
            #print(i)

    def getValue(self, values, row, col):
        """Calculates the value of the differences at a certain point when multiplied by the solution at that point"""
        returnValue = 0.0
        for providerWithCoordinates in self.valueProviders:
            value = providerWithCoordinates['provider'].getValue(providerWithCoordinates, row, col)
            colIndex = col + providerWithCoordinates['x']
            rowIndex = row + providerWithCoordinates['y']
            if rowIndex >= 0 and rowIndex < len(values) and colIndex >= 0 and colIndex < len(values[0]):
                returnValue += values[rowIndex, colIndex] * value

        return returnValue


class FiniteDifferencesMethod2:
    def __init__(self, geometry, boundaryCondition, c):
        self.geometry = geometry
        self.boundaryCondition = boundaryCondition
        self.c = c
        self.values = np.zeros_like(geometry.X)

    def dummy(self):
        """ 1.row   (i=1  ,j=1):  F(0  ,  1)+F(2  ,1)+F(1  ,0)+F(1  ,2)-4F(1  ,1)
            # 2.row   (i=2  ,j=1):  F(1  ,  1)+F(3  ,1)+F(2  ,0)+F(2  ,2)-4F(2  ,1)
            # 3.row   (i=3  ,j=1):  F(2  ,  1)+F(4  ,1)+F(3  ,0)+F(3  ,2)-4F(3  ,1)
            # N-1.row (i=N-1,j=1):  F(N-2,  1)+F(N-1,1)+F(N-1,0)+F(N-1,2)-4F(N-1,1
            # N.row   (i=1  ,j=2):  F(0  ,  2)+F(2  ,2)+F(1,  1)+F(1,  3)-4F(1  ,2)
            # N+1.row (i=2  ,j=2):  F(1  ,  2)+F(3  ,2)+F(2,  1)+F(2,  3)-4F(N+1  ,N+1)
            #
            #
            # i.row, j.col: F(i,j): -4; F(i,j-1):1; F(i,j+1): 1;  F(i-1,j): 1;  F(i+1,j):  1
            # ?. row  (i=N-1,j=N-1):F(N-2,N-1)+F(N,N-1)+F(N-1,N-2)+F(N-1,N)-4F(N-1,N-1)
            #
            #


            # All elements with a 0 are boundary conditions -> don't show up in A
            #       (1,1) (2,1) (3,1) (4,1)     (1,2)   (2,2)   (3,2)       ... (1,3) (2,3)
            # A = ( -4     1     0          ...   1
            #        1     -4    1          ...          1
            #        0     1     -4    1    ...                  1
            #
            # N.row  1                      ...  -4      1                       1
            #              1                      1      -4      1                   1
            #
            #   ... ->           ...
        """
        pass

    def calc1DIndex(self, col, row):
        return col + self.geometry.numX * row

    def rowElements(self):
        return self.geometry.numX

    def setupRightSideOfEquation(self):
        """copy the charge distribution of the matrix (x,y) into a vector
        """
        self.bias = np.zeros(shape=(self.geometry.numX * self.geometry.numY, 1))
        for row in range(1, self.geometry.numY - 1):
            for col in range(1, self.geometry.numX - 1):
                self.bias[self.calc1DIndex(col, row)] = self.c.get(col, row)

    def setupMatrices(self):
        """create the coefficient and boundary matrix compatible with the bias vector
           for every point (x,y) there is a row in the matrix.
           The entries are the coefficients of the involved matrix points
        """
        self.matrix = np.zeros(shape=(self.geometry.numX * self.geometry.numY, self.geometry.numX * self.geometry.numY))
        self.elementCountTarget = self.geometry.numX * self.geometry.numY
        self.elementsInRow = self.geometry.numY
        for row in range(0, self.geometry.numY):
            for col in range(0, self.geometry.numX):
                # rowIndex is an index within a target row
                rowIndex = row * self.rowElements() + col

                offsetInRow = rowIndex
                valueForBoundary = self.boundaryCondition.get(col, row)
                # rowInMatrixOffset = row*self.rowElements()
                # print(row, self.rowElements(), rowInMatrixOffset)
                # self.matrix[ row, 0] = 2.0
                # for col in range(0, self.geometry.numX):
                # valueForBoundary = self.boundaryCondition.get(col,row)
                # print(row,col,rowInMatrixOffset,valueForBoundary)
                # offsetInRow = row

                # print(rowIndex,row,col,valueForBoundary)

                # central element

                #for valueProvider in self.gridConfiguration.valueProviders:
                #    valueProvider.apply(col, row, self.matrix)

                newValue = -4.0
                if (row == 0 and col == 0) or (row == self.geometry.numY - 1 and col == self.geometry.numX - 1):
                    newValue = -4.0 if valueForBoundary == None else valueForBoundary

                self.matrix[rowIndex, offsetInRow] = newValue

                # left neighbor element
                newValue = 1.0
                if (row == 0 and col - 1 <= 0) or (row == self.geometry.numY - 1 and col >= self.geometry.numX - 1):
                    newValue = 1.0 if valueForBoundary == None else valueForBoundary

                if offsetInRow - 1 >= 0:
                    self.matrix[rowIndex, offsetInRow - 1] = newValue

                # right neighbor element
                newValue = 1.0

                if (row == 0 and col + 1 <= 1) or (row == self.geometry.numY - 1 and col + 1 >= self.geometry.numX - 1):
                    newValue = 1.0 if valueForBoundary == None else valueForBoundary

                if offsetInRow + 1 < self.elementCountTarget:
                    self.matrix[rowIndex, offsetInRow + 1] = newValue

                # top neighbor element
                newValue = 1.0

                if (row - 1 <= 0 and col == 0) or (row >= self.geometry.numY - 1 and col == self.geometry.numX - 1):
                    newValue = 1.0 if valueForBoundary == None else valueForBoundary

                # print("top:",newValue, row, col, rowIndex, offsetInRow, self.elementsInRow)
                if offsetInRow - self.elementsInRow >= 0:
                    self.matrix[rowIndex, offsetInRow - self.elementsInRow] = newValue

                # bottom neighbor element
                newValue = 1.0

                if (row <= 0 and col == 0):
                    newValue = 1.0 if valueForBoundary == None else valueForBoundary

                # print("top:",newValue, row, col, rowIndex, offsetInRow, self.elementsInRow)
                if offsetInRow + self.elementsInRow <= self.elementCountTarget - 1:
                    self.matrix[rowIndex, offsetInRow + self.elementsInRow] = newValue

                if offsetInRow+self.elementsInRow < self.elementsInRow-1:
                                   self.matrix[ rowIndex, offsetInRow + 2 ] = 1.0

                # TODO: besser die Schleife von 2 bis ... laufen zu lassen
                #      und für die Werte an den Stellen 1 bzw N-1 auszuwerten -> spart sich einen Haufen if
                # else:
                # self.matrix[ rowInMatrixOffset+rowInMatrix, 0 ] = valueForBoundary
                # self.matrix[ rowInMatrixOffset+rowInMatrix, self.geometry.numX ] = valueForBoundary

    def solve(self):
        self.setupRightSideOfEquation()
        self.setupMatrices()
        self.valuesResult = np.linalg.solve(
            self.matrix[1:self.geometry.numX * self.geometry.numY - 1, 1:self.geometry.numX * self.geometry.numY - 1],
            self.bias[1:self.geometry.numX * self.geometry.numY - 1])
        self.convertToMatrix()

    def convertToMatrix(self):
        for row in range(1, self.geometry.numY - 1):
            for col in range(1, self.geometry.numX - 1):
                rowIndex = (row - 1) * self.rowElements() + (col - 1)
                # print(rowIndex, row, col)
                self.values[col, row] = self.valuesResult[rowIndex]

    def printBias(self):
        print(self.bias)

    def printMatrix(self):
        print(self.matrix)

    def printValues(self):
        print(self.valuesResult)


class FiniteDifferencesMethod3:
    def __init__(self, geometry, boundaryCondition, gridConfiguration, c):
        self.geometry = geometry
        self.boundaryCondition = boundaryCondition
        self.c = c
        self.gridConfiguration = gridConfiguration
        self.values = np.zeros_like(geometry.X)

    def dummy(self):
        """ 1.row   (i=1  ,j=1):  F(0  ,  1)+F(2  ,1)+F(1  ,0)+F(1  ,2)-4F(1  ,1)
            # 2.row   (i=2  ,j=1):  F(1  ,  1)+F(3  ,1)+F(2  ,0)+F(2  ,2)-4F(2  ,1)
            # 3.row   (i=3  ,j=1):  F(2  ,  1)+F(4  ,1)+F(3  ,0)+F(3  ,2)-4F(3  ,1)
            # N-1.row (i=N-1,j=1):  F(N-2,  1)+F(N-1,1)+F(N-1,0)+F(N-1,2)-4F(N-1,1
            # N.row   (i=1  ,j=2):  F(0  ,  2)+F(2  ,2)+F(1,  1)+F(1,  3)-4F(1  ,2)
            # N+1.row (i=2  ,j=2):  F(1  ,  2)+F(3  ,2)+F(2,  1)+F(2,  3)-4F(N+1  ,N+1)
            #
            #
            # i.row, j.col: F(i,j): -4; F(i,j-1):1; F(i,j+1): 1;  F(i-1,j): 1;  F(i+1,j):  1
            # ?. row  (i=N-1,j=N-1):F(N-2,N-1)+F(N,N-1)+F(N-1,N-2)+F(N-1,N)-4F(N-1,N-1)
            #
            #


            # All elements with a 0 are boundary conditions -> don't show up in A
            #       (1,1) (2,1) (3,1) (4,1)     (1,2)   (2,2)   (3,2)       ... (1,3) (2,3)
            # A = ( -4     1     0          ...   1
            #        1     -4    1          ...          1
            #        0     1     -4    1    ...                  1
            #
            # N.row  1                      ...  -4      1                       1
            #              1                      1      -4      1                   1
            #
            #   ... ->           ...
        """
        pass

    def calc1DIndex(self, col, row):
        return col + self.geometry.numX * row

    def rowElements(self):
        return self.geometry.numX

    def setupRightSideOfEquation(self):
        """copy the charge distribution of the matrix (x,y) into a vector
        """
        self.bias = np.zeros(shape=(self.geometry.numX * self.geometry.numY, 1))
        for row in range(1, self.geometry.numY - 1):
            for col in range(1, self.geometry.numX - 1):
                self.bias[self.calc1DIndex(col, row)] = self.c.get(col, row)

    def indexCallback(self, col, row, colOffset, rowOffset):
        """Calculates a rowIndex and columnIndex in a 2D array
        """
        rowIndex = row * self.rowElements() + col
        colIndex = rowIndex + colOffset + rowOffset * self.elementsInRow

        return {'rowIndex':rowIndex, 'colIndex':colIndex}

    def setupMatrices(self):
        """create the coefficient and boundary matrix compatible with the bias vector
           For every point (x,y) there is a row in the matrix.
           The entries are the coefficients of the involved matrix points
        """
        self.matrix = np.zeros(shape=(self.geometry.numX * self.geometry.numY, self.geometry.numX * self.geometry.numY))
        self.elementCountTarget = self.geometry.numX * self.geometry.numY
        self.elementsInRow = self.geometry.numY
        for row in range(0, self.geometry.numY):
            for col in range(0, self.geometry.numX):
                # rowIndex is an index within a target row
                #rowIndex = row * self.rowElements() + col

                #offsetInRow = rowIndex
                #valueForBoundary = self.boundaryCondition.get(col, row)

                self.gridConfiguration.setValues(self.matrix, row, col, self.indexCallback)

    def solve(self):
        self.setupRightSideOfEquation()
        self.setupMatrices()
        self.valuesResult = np.linalg.solve(
            self.matrix[1:self.geometry.numX * self.geometry.numY - 1, 1:self.geometry.numX * self.geometry.numY - 1],
            self.bias[1:self.geometry.numX * self.geometry.numY - 1])
        self.convertToMatrix()

    def convertToMatrix(self):
        for row in range(1, self.geometry.numY - 1):
            for col in range(1, self.geometry.numX - 1):
                rowIndex = (row - 1) * self.rowElements() + (col - 1)
                # print(rowIndex, row, col)
                self.values[col, row] = self.valuesResult[rowIndex] / 10.0

    def calcRightside(self):
        for row in range(1, self.geometry.numY - 1):
            for col in range(1, self.geometry.numX - 1):
                self.values[col, row]

    def calcMetrices(self):
        self.error = np.zeros_like(self.geometry.X)
        self.minValue = 100.0
        self.maxValue = -100.0
        sumValue = 0.0
        for row in range(1, self.geometry.numY - 1):
            for col in range(1, self.geometry.numX - 1):
                valueAtPoint = self.gridConfiguration.getValue(self.values, row, col)
                errorAtPoint = valueAtPoint - self.c.get(row, col)
                self.error[row, col] = errorAtPoint
                sumValue = sumValue + abs(valueAtPoint)
                self.maxValue = max(self.maxValue, errorAtPoint)
                self.minValue = min(self.minValue, errorAtPoint)

        print('Max,Min:',self.maxValue,self.minValue)
        print('AvgValue:', sumValue/((self.geometry.numY-2)*(self.geometry.numX-2)))

    def printBias(self):
        print(self.bias)

    def printMatrix(self):
        print(self.matrix)

    def printValues(self):
        print(self.valuesResult)