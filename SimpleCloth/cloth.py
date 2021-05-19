import taichi as ti
from particle import Particle
ti.reset()
#@ti.data_oriented
class Cloth:
    def __init__(self, dims, topleft, width, height, start_idx=0, radius=12):
        self.w, self.h = dims
        self.start_idx = start_idx
        self.r = radius

        dx, dy = width/self.w, height/dims[1]
        sx, sy = topleft
        self.particles = {}
        for i in range(self.w):
            for j in range(self.h):
                idx = j*self.w + i
                x, y = sx+dx*i , sy-dy*j
                self.particles[idx + start_idx] = Particle(idx+start_idx, x, y, radius=radius)


    def at(self, idx, pos=None):
        return self.particles[idx].at(pos)
    
    def dims(self):
        return self.w, self.h

    def __len__(self):
        return len(self.particles)

    def __iter__(self):
        i = 0
        while i < len(self):
            yield self.start_idx + i
            i += 1

    def __getitem__(self, pid):
        return self.particles[pid]
   



if __name__ == '__main__':
    c = Cloth([20,4], [0.2, 0.6], 0.6, 0.2)

    print(list(iter(c)))