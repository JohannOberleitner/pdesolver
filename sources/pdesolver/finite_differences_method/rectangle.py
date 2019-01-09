

class Rectangle:
    def __init__(self, x, y, width, height):
        self.x1 = x
        self.y1 = y

        self.width = width
        self.height = height

        self.x2 = self.x1 + width
        self.y2 = self.y1 + height

    def midX(self):
        return self.x1 + self.width / 2

    def midY(self):
        return self.y1 + self.height / 2
