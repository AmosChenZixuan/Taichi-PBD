class Particle:
    COLOR=0xffaa77
    def __init__(self, pid, x, y, radius):
        self.pid = pid
        self._x = x
        self._y = y
        self.r = radius
    
    def at(self, pos=None):
        if pos is not None:
            x,y = pos
            self._x = x
            self._y = y
        return self._x, self._y