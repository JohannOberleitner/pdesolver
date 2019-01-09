
class BoundaryCondition:
    def __init__(self, geometry):
        self.geometry = geometry

    def get(self,i,j):
        pass

class RectangularBoundaryCondition(BoundaryCondition):
    def get(self,i,j):
        if i == 0 or i == self.geometry.numX-1:
            return 0.0
        elif j == 0 or j == self.geometry.numY-1:
            return 0.0
        else:
            return None