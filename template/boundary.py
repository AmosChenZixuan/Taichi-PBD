import taichi as ti


class Box:
    def __init__(self, left, top, width, height, elastic=True):
        '''
            2-D Rectangle boundary 
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

        self.elastic = elastic

    def draw(self, gui):
        c = 0x445566
        gui.line(begin=self.x1, end=self.x2, radius=2, color=c)
        gui.line(begin=self.x1, end=self.x4, radius=2, color=c)
        gui.line(begin=self.x2, end=self.x3, radius=2, color=c)
        gui.line(begin=self.x4, end=self.x3, radius=2, color=c)

    def _v(self, v):
        return v if self.elastic else 0

    @ti.func
    def bound_check(self, x,y,vx,vy,r):
        print(x,y)
        bounced = False
        if x - r <= self.left :
            x = self.left + r
            vx = self._v(abs(vx))
            bounced = True
        elif x + r >= self.right:
            x = self.right - r
            vx = self._v(-abs(vx))
            bounced = True
        if y - r < self.bottom:
            y = self.bottom + r
            vy = self._v(abs(vy))
            bounced = True
        elif y + r > self.top:
            y = self.top - r
            vy = self._v(-abs(vy))
            bounced = True
        return bounced, x,y,vx,vy
