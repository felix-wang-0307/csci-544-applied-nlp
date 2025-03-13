class Rectangle:
    __count = 0
    def __init__(self, width, height):
        self.width = width
        self.height = height
        Rectangle.__count += 1
    
    @property
    def area(self):
        return self.width * self.height
    
    def set_width(self, width):
        self.width = width


r1 = Rectangle(10, 20)
r2 = Rectangle(20, 30)
print(r1.area, r2.area)
print(Rectangle.__count) # 2
