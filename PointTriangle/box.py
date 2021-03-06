class Box:
    def __init__(self, left, top, width, height):
        '''
            x1(left,top) -- x2
            |                |
            |h               |
            |        w       |
            x4 ------------ x3
        '''
        self.dims = width, height

        self.top = top
        self.bottom = top - height
        self.left = left
        self.right = left+width

        self.x1 = left, top
        self.x2 = self.right, top
        self.x3 = self.right, self.bottom
        self.x4 = left, self.bottom

    def draw(self, gui):
        c = 0x445566
        gui.line(begin=self.x1, end=self.x2, radius=2, color=c)
        gui.line(begin=self.x1, end=self.x4, radius=2, color=c)
        gui.line(begin=self.x2, end=self.x3, radius=2, color=c)
        gui.line(begin=self.x4, end=self.x3, radius=2, color=c)