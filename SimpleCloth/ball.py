import taichi as ti
#ti.init(arch=ti.cpu)
from particle import Particle

class Ball(Particle):
    COLOR = 0xffffff
    def __init__(self, pid, x, y, r=10):
        super().__init__(pid,x,y,r)
        