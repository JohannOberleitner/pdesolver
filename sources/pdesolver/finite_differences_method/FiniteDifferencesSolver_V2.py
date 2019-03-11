import numpy as np
from scipy.sparse import dia_matrix
from scipy.sparse.linalg import splu
from scipy.sparse.linalg import spsolve


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
        for row in range(0, self.geometry.numY):
            for col in range(0, self.geometry.numX):
                self.bias[self.calc1DIndex(col, row)] = self.c.get(col, row)

    def indexCallback(self, row, col, colOffset, rowOffset):
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
        self.elementsInRow = self.geometry.numX
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

        mm3 = self.matrix[64*30:64*30+6]
        self.valuesResult = np.linalg.solve(
            self.matrix,
            self.bias)
        probe = self.matrix.dot(self.valuesResult)


        self.convertToMatrix()

    def convertToMatrix(self):
        for row in range(0, self.geometry.numY):
            for col in range(0, self.geometry.numX):
                rowIndex = row * self.rowElements() + col
                # print(rowIndex, row, col)
                self.values[col, row] = self.valuesResult[rowIndex] / 10.0

    #def calcRightside(self):
    #    for row in range(1, self.geometry.numY - 1):
    #        for col in range(1, self.geometry.numX - 1):
    #            self.values[col, row]

    def calcMetrices(self):
        self.error = np.zeros_like(self.geometry.X)
        self.results = np.zeros_like(self.geometry.X)
        self.minValue = 100.0
        self.maxValue = -100.0
        sumValue = 0.0
        for row in range(1, self.geometry.numY - 1):
            for col in range(1, self.geometry.numX - 1):
                valueAtPoint = self.gridConfiguration.getValue(self.values, row, col)
                errorAtPoint = valueAtPoint - self.c.get(row, col)
                self.error[row, col] = errorAtPoint
                self.results[row, col] = valueAtPoint
                sumValue = sumValue + abs(valueAtPoint)
                self.maxValue = max(self.maxValue, errorAtPoint)
                self.minValue = min(self.minValue, errorAtPoint)

        print('Max,Min:',self.maxValue,self.minValue)
        print('AvgValue:', sumValue/((self.geometry.numY)*(self.geometry.numX)))

    def printBias(self):
        print(self.bias)

    def printMatrix(self):
        print(self.matrix)

    def printValues(self):
        print(self.valuesResult)


class FiniteDifferencesMethod4:
    def __init__(self, geometry, boundaryCondition, gridConfiguration, c):
        self.geometry = geometry
        self.boundaryCondition = boundaryCondition
        self.c = c
        self.gridConfiguration = gridConfiguration
        self.values = np.zeros_like(geometry.X)

    def calc1DIndex(self, col, row):
        return col + self.geometry.numX * row

    def rowElements(self):
        return self.geometry.numX

    def setupRightSideOfEquation(self):
        """copy the charge distribution of the matrix (x,y) into a vector
        """
        self.bias = np.zeros(shape=(self.geometry.numX * self.geometry.numY, 1))
        for charge in self.c:
            col = charge.get_x()
            row = charge.get_y()
            self.bias[self.calc1DIndex(col, row)] = charge.get_value()

        #for row in range(0, self.geometry.numY):
        #    for col in range(0, self.geometry.numX):
        #        self.bias[self.calc1DIndex(col, row)] = self.c.get(col, row)


    def setupMatricesCSR(self):
        """
            https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_column_(CSC_or_CCS)
            Matrix M has NNZ entries
            A=4(0,0) 1(0,0) 1(0,0) 1(0,0) 4(0,0) 1(0,0) 1(0,0) ...
            Matrix IA has row+1 entries
            IA[0] = 0
            IA[i] = I[i] + (number of nonzero elements of i-th row in Matrix M)
            JA[i] = column-index for A (has NNZ entries)

            for our case:
            IA.append([0, 3, 4, 8, 12, ..., ..., k+5, ... , ... l+3]
            JA.append([0,1,64,0,1,2,65,1,2,3,66,2,3,4,67,...,0,63,64,65,64+64,...]

        :return:
        """

    def appendDefaultDirichletConditions(self, dataElements, mainDiagonalIndex, count):
        for index, dataElement in enumerate(dataElements):
            if index == mainDiagonalIndex:
                dataElement += count * [1.0]
            else:
                dataElement += count * [0.0]

    def prepareSetupMatrices(self):
        offsetElements = []
        dataElements = [[] for x in range(0, len(self.gridConfiguration.valueProviders))]

        mainDiagonalIndex = -1
        for index, providerWithCoordinates in enumerate(self.gridConfiguration.valueProviders):
            xOffset = providerWithCoordinates['x']
            yOffset = providerWithCoordinates['y']
            offset = xOffset + yOffset * self.geometry.numX
            offsetElements.append(offset)

            if offset == 0:
                mainDiagonalIndex = index

        return offsetElements, dataElements, mainDiagonalIndex

    def preProcessDiagonalMatrices(self, dataElements):
        for index, providerWithCoordinates in enumerate(self.gridConfiguration.valueProviders):
            xOffset = providerWithCoordinates['x']
            yOffset = providerWithCoordinates['y']
            offset = xOffset + yOffset * self.geometry.numX

            if offset > 0:
                dataElements[index] += offset * [-1.0]

    def postProcessDiagonalMatrices(self, dataElements):

        for index, providerWithCoordinates in enumerate(self.gridConfiguration.valueProviders):
            xOffset = providerWithCoordinates['x']
            yOffset = providerWithCoordinates['y']
            offset = xOffset + yOffset * self.geometry.numX

            if offset < 0:
                dataElements[index] += (-offset) * [-1.0]
                del dataElements[index][0:(-offset)]

        gridLength = self.geometry.numY * self.geometry.numX

        for dataElement in dataElements:
            if len(dataElement) > gridLength:
                del dataElement[gridLength - len(dataElement):]

    def setupMatrices(self):
        """create the coefficient and boundary matrix compatible with the bias vector
           For every point (x,y) there is a row in the matrix.
           The entries are the coefficients of the involved matrix points
        """

        # dataElements is an (initially) empty array of 5 lists - for each diagonal in the coefficient matrix
        # for 3D dataElements would have 7 lists
        # offsetElements defines the offset for each list within each row (+1,-1,+n,-n,0)
        # mainDiagonalIndex references within dataElements to the main diagonal
        offsetElements, dataElements, mainDiagonalIndex = self.prepareSetupMatrices()

        # appends dummy elements (with value = -1) for diagonals right of the main one
        # for the side-diagonal just 1 element is added, for the one for the row above n elements are added
        # this is needed because scipy.dia_matrix requires that all the diagonal elements have the same count
        self.preProcessDiagonalMatrices(dataElements)

        # Boundary condition for row=0 (and every column)
        # For Dirichlet conditions the value at the boundary = 0.
        # To include this in  the equation there need to be an element in the matrix for every column
        # Hence count=numX elements need to be added.
        # The value for the mainDiagonal must be 1.0, for other diagonals it must be 0.0
        self.appendDefaultDirichletConditions(dataElements, mainDiagonalIndex, count=self.geometry.numX)

        for row in range(1, self.geometry.numY-1):

            # Boundary condition in first column of this row
            self.appendDefaultDirichletConditions(dataElements, mainDiagonalIndex, count=1)

            for col in range(1, self.geometry.numX-1):
                for index, providerWithCoordinates in enumerate(self.gridConfiguration.valueProviders):
                    value = providerWithCoordinates['provider'].getValue(None, row, col)
                    dataElements[index].append(value)

            # Boundary condition in last column of this row
            self.appendDefaultDirichletConditions(dataElements, mainDiagonalIndex, count=1)

        # Boundary condition in last row
        self.appendDefaultDirichletConditions(dataElements, mainDiagonalIndex, self.geometry.numX)

        # appends dummy elements (with value = -1) for diagonals left of the main one
        # for the side diagonal just 1 element is added, for the one for the row below n elements are added
        # this is needed because scipy.dia_matrix requires those elements to have the proper amount of elements
        # - additionally the first elements needed to be removed
        self.postProcessDiagonalMatrices(dataElements)

        gridLength = self.geometry.numY * self.geometry.numX

        data = np.array(dataElements)
        offsetData = np.array(offsetElements)
        self.v = dia_matrix((data, offsetData), shape=(gridLength, gridLength))

    def setupMatricesWithoutBoundaryConditions(self):
        """create the coefficient and boundary matrix compatible with the bias vector
           For every point (x,y) there is a row in the matrix.
           The entries are the coefficients of the involved matrix points
        """
        offsetElements = []
        dataElements = [[] for x in range(0, len(self.gridConfiguration.valueProviders))]


        #offsetElements.append(offset)
        i = 0
        for providerWithCoordinates in self.gridConfiguration.valueProviders:
            xOffset = providerWithCoordinates['x']
            yOffset = providerWithCoordinates['y']
            offset = xOffset + yOffset * 64
            offsetElements.append(offset)

            if offset > 0:
                dataElements[i] += offset * [-1.0]
            i = i+1

        for row in range(0, self.geometry.numY):
            for col in range(0, self.geometry.numX):
                i = 0
                for providerWithCoordinates in self.gridConfiguration.valueProviders:
                    dataElementsForDiagonal = dataElements[i]
                    i = i+1
                    #xOffset = providerWithCoordinates['x']
                    #yOffset = providerWithCoordinates['y']
                    provider = providerWithCoordinates['provider']
                    #offset = xOffset + yOffset*64

                    value = provider.getValue(None, row, col)
                    dataElementsForDiagonal.append(value)

        #dataElements[2][2] = 9.9
        #dataElements[2][3] = 12.9
        #dataElements[2][64] = 999.9
        #dataElements[2][65] = -999.9

        #dataElements[3][2] = -9.9
        #dataElements[3][3] = -12.9
        gridLength = self.geometry.numY * self.geometry.numX

        i = 0
        for providerWithCoordinates in self.gridConfiguration.valueProviders:
            xOffset = providerWithCoordinates['x']
            yOffset = providerWithCoordinates['y']
            offset = xOffset + yOffset * 64

            if offset < 0:
                dataElements[i] += (-offset) * [-1.0]
                del dataElements[i][0:(-offset)]

            i = i + 1

        for dataElement in dataElements:
            if len(dataElement) > gridLength:
                del dataElement[gridLength-len(dataElement):]

        data = np.array(dataElements)
        offsetData = np.array(offsetElements)
        self.v = dia_matrix((data, offsetData), shape=(self.geometry.numX*self.geometry.numX, self.geometry.numY*self.geometry.numY))


    def solve(self):
        self.setupRightSideOfEquation()
        self.setupMatrices()

        self.matrix = self.v.tocsc()
        B = splu(self.matrix)
        self.valuesResult = B.solve(self.bias)

        self.convertToMatrix()

    def convertToMatrix(self):

        self.values = np.reshape(self.valuesResult, (self.geometry.numY, self.geometry.numX))

        #for row in range(0, self.geometry.numY):
        #    for col in range(0, self.geometry.numX):
        #        rowIndex = row * self.rowElements() + col
        #        # print(rowIndex, row, col)
        #        self.values[col, row] = self.valuesResult[rowIndex] # / 10.0

    def calcMetrices(self):
        self.error = np.zeros_like(self.geometry.X)
        self.results = np.zeros_like(self.geometry.X)
        self.minValue = 100.0
        self.maxValue = -100.0
        sumValue = 0.0
        for row in range(0, self.geometry.numY):
            for col in range(0, self.geometry.numX):

                valueAtPoint = self.gridConfiguration.getValue(self.values, row, col)
                errorAtPoint = valueAtPoint - self.c.get(row, col)
                self.error[row, col] = errorAtPoint
                self.results[row, col] = valueAtPoint
                sumValue = sumValue + abs(valueAtPoint)
                self.maxValue = max(self.maxValue, abs(errorAtPoint))
                self.minValue = min(self.minValue, abs(errorAtPoint))

        print('Max,Min:',self.maxValue,self.minValue)
        print('AvgValue:', sumValue/((self.geometry.numY-2)*(self.geometry.numX-2)))
