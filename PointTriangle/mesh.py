import taichi as ti 
import math

@ti.data_oriented
class Mesh:
    def __init__(self, width=2, height=2, d=0.025, start=0, pos=(-1.0,-1.0)):
        self.dims = max(width,2), max(height,2)   #mininal dimension (2x2)
        self.n = self.dims[0] * self.dims[1]
        self.start = start
        self.end = start + self.n
        self.d = d
        self.pos = pos
        self.edges = self.get_edges()
        self.squares = self.get_squares()

    def at(self, i, j):
        """
            return: global index based on local mesh location
        """
        return i * self.dims[1] + j + self.start

    def __len__(self):
        return self.n
    
    def __iter__(self):
        for i in range(self.start, self.end):
            yield i

    def get_edges(self):
        """
            4 sides + 2 diagnals for each square(4 vertices)

            return: [(idx1, idx2)]
        """
        ret = []
        w, h = self.dims
        for i in range(w):
            for j in range(h): 
                has_bot = j < h-1
                has_right = i < w-1
                has_left = i > 0
                if has_bot: # below
                    ret.append([self.at(i,j), self.at(i,j+1), self.d, 0.5]) 
                if has_right: # right
                    ret.append([self.at(i,j), self.at(i+1,j), self.d, 0.5]) 
                if has_bot and has_right: # right-bot
                    ret.append([self.at(i,j), self.at(i+1,j+1), self.d*math.sqrt(2), 0.5]) 
                if has_bot and has_left: # left-bot
                    ret.append([self.at(i,j), self.at(i-1,j+1), self.d*math.sqrt(2), 0.5]) 
        return ret

    def get_squares(self):
        """
            return [(lb ,lt, rt, rb)]
        """
        ret = []
        w, h = self.dims
        for i in range(w-1):
            for j in range(h-1):
                sq = [self.at(i,j+1), self.at(i,j),
                      self.at(i+1,j), self.at(i+1,j+1)]
                ret.append(sq)
        return ret


    def draw(self, gui, X):
        for x,y,d,k in self.edges:
            if d == self.d:
                color = 0x990000
            else:
                color = 0x3D85C6
            gui.line(begin=X[x], end=X[y], radius=2, color=color)
        #gui.circles(X[self.start:self.end], color=0xffaa77, radius=3)
        # gui.circle(X[self.start], color=0xffaa77, radius=10)
        # gui.circle(X[self.start+1], color=0xA6640C, radius=10)
        # gui.circle(X[self.start+2], color=0xDC4E75, radius=10)
        # gui.circle(X[self.start+3], color=0x227CAD, radius=10)
        
    @staticmethod
    @ti.func
    def sign():
        return 1 if ti.random(ti.f32) > 0.5 else -1

    @staticmethod
    @ti.func
    def rand(a,b):
        r = ti.random(ti.f32)
        r = max(a, r)
        r = min(b, r)
        return r

    @ti.kernel
    def init(self, X:ti.template(), V:ti.template(), W:ti.template()):
        x, y = self.pos
        if x < 0 or y < 0:
            x, y = self.rand(0.25, 0.75), self.rand(0.25, 0.75)
        for i in range(self.dims[0]):
            for j in range(self.dims[1]):
                idx = self.at(i, j)
                X[idx] = [x + i * self.d , y - j * self.d]
                if self.start!=0 and i == 0 and 0:
                    W[idx] = 0.0
                else:
                    W[idx] = 1.0
                
        mul = 1
        #V[self.at(0,0)] = [mul*self.sign()*ti.random(ti.f32), mul*self.sign()*ti.random(ti.f32)]
        #V[self.at(0,0)] = [0, -9.8]




if __name__ == '__main__':
    b = Mesh(start=4)

    print(b.at(1,1))
