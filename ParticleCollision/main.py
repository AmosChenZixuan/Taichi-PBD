import taichi as ti 
ti.init(arch=ti.gpu)

from constraint import *


dt = 5e-3
gravity = ti.Vector([0, 0])
damping = 1
n_iter = 5
radius = 10

n_balls = 10
X = ti.Vector.field(2, dtype=ti.f32, shape=n_balls)
P = ti.Vector.field(2, dtype=ti.f32, shape=n_balls)
V = ti.Vector.field(2, dtype=ti.f32, shape=n_balls)
W = ti.field(ti.f32, shape=n_balls)


contact_cache = {}
contact_count=0


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

class Balls:
    def __init__(self, start, nums):
        self.start = start
        self.end = start + nums
        self.n = nums
    
    def __len__(self):
        return self.n
    
    def __iter__(self):
        for i in range(self.start, self.end):
            yield i

    def draw(self, gui, X):
        gui.circles(X.to_numpy()[:self.n], color=0xffaa77, radius=12)

balls = Balls(0, n_balls)
box = Box(0.1, 0.9, 0.8, 0.8)

@ti.kernel
def init():
    for i in range(balls.start, balls.end):
        X[i] = [ti.random(ti.f32), ti.random(ti.f32)]
        V[i] = [2*ti.random(ti.f32), 2*ti.random(ti.f32)]
        W[i] = 1
        
def pre_collis():
    global contact_count
    r = range(balls.start, balls.end)
    for i in r:
        for j in r:
            if i==j:
                continue
            key = f'{i}-{j}'
            contact_cache[key] = DistanceConstraint(contact_count, i, j, 20/800)
            contact_count += 1 
            print(contact_count)
            solve(contact_cache[key])


@ti.pyfunc
def p(s):
    print(s)

@ti.kernel
def propose():
    for i in range(balls.start, balls.end):
        V[i] += dt * W[i] * gravity 
        P[i] = X[i] + dt * V[i] * damping

def generate_collisions():
    pos = []
    for i in range(balls.start, balls.end):
        x,y = P[i][0], P[i][1]
        pos.append((x,y,i))
    return sweep_prune(sorted(pos))

def sweep_prune(pos):
    global contact_count
    res = []
    active = [pos[0]]
    for i in range(1, len(pos)):
        cur = pos[i]
        tid = 0 
        while len(active) > 0 and tid < len(active):
            target = active[tid]
            r,d = 10/800, 22/800
            if target[0] + r >= cur[0] - r: #candidate
                # if abs(target[2]- cur[2]) < 5:
                #     print( target[2], cur[2], target[0], cur[0])
                p1 = ti.Vector([target[0], target[1]])
                p2 = ti.Vector([cur[0], cur[1]])
                if (p1-p2).norm() <= d:
                    key = f'{target[2]}-{cur[2]}'
                    c = contact_cache.get(key)
                    if c is None:
                        print(contact_count)
                        c = DistanceConstraint(contact_count, target[2], cur[2], d)
                        contact_cache[key] = c
                        contact_count += 1 
                    res.append(c)
                    break
                tid += 1
            else:
                active.pop(0)
        active.append(cur)

    return res


def project(collis):
    for _ in range(n_iter):
        for c in collis:
            solve(c)

@ti.kernel
def solve(c: ti.template()):
    c.solve(P, W)

@ti.kernel
def update():
    for i in range(balls.start, balls.end):
        V[i] = (P[i] - X[i]) / dt
        X[i] = P[i]

@ti.kernel
def bound_check():
    for i in range(balls.start, balls.end):
        x,y = X[i]
        vx, vy = V[i]
        r = radius/800
        damping = 1
        collided = False
        # bound check
        if x - r <= box.left :
            x = box.left + r
            vx = abs(vx)
            collided = True
        elif x + r >= box.right:
            x = box.right - r
            vx = -abs(vx)
            collided = True
        if y - r < box.bottom:
            y = box.bottom + r
            vy = abs(vy)
            collided = True
        elif y + r > box.top:
            y = box.top - r
            vy = -abs(vy)
            collided = True
        X[i] = x, y
        V[i] = vx, vy
        if collided:
            V[i] *= damping

gui = ti.GUI('BOX', res=(800, 800), background_color=0xdddddd)
init()
pre_collis()
print(balls.start, balls.end)
while gui.running:
    propose()

    collis = generate_collisions()

    project(collis)

    update()
    bound_check()
    #print(X[0][0], X[0][1])


    box.draw(gui)
    balls.draw(gui,X)
    for c in collis:
        if c.CTYPE == 1:
            idx1 = c.xpid
            idx2 = c.ypid
            gui.line(begin=X[idx1], end=X[idx2], radius=2, color=DistanceConstraint.COLOR)

    if gui.get_event(ti.GUI.PRESS):

        if gui.event.key == 'a':
            print('a')
            V[0][0] = -1
        elif gui.event.key == ti.GUI.LMB:
            X[0] = [gui.event.pos[0], gui.event.pos[1]]
            V[0] = [0,0]
        elif gui.event.key == ti.GUI.ESCAPE:
            break
        

    gui.show()
        