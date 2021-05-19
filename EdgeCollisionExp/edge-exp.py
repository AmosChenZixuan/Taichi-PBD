import taichi as ti
from taichi.lang.ops import random, sub
ti.init(arch=ti.gpu)


dt = ti.field(dtype=ti.f32, shape=()) 
substep = ti.field(dtype=ti.i32, shape=()) 
gravity = ti.Vector([0, -2])
#gravity = ti.Vector([0, 0])
damping = 0.99
n_iter = 51
n_mesh = 2
RADIUS = 50/800

S = ti.Vector.field(2, dtype=ti.f32, shape=3)
X = ti.Vector.field(2, dtype=ti.f32, shape=3)
P = ti.Vector.field(2, dtype=ti.f32, shape=3)
V = ti.Vector.field(2, dtype=ti.f32, shape=3)
W = ti.field(ti.f32, shape=3)

@ti.kernel
def init():
    dt[None] = 6e-3
    substep[None] = 1
    X[0] = [0.3, 0.5]
    W[0] = 0.0
    X[1] = [0.7, 0.5]
    W[1] = 0.0
    X[2] = [0.5, 0.8]
    W[2] = 1
    V[2] = [0,0]

@ti.kernel
def propose():
    for i in range(3):
        V[i] += dt[None] * W[i] * gravity 
        P[i] = X[i] + dt[None] * V[i]

@ti.kernel
def project():
    a, b = P[0], P[1]
    c, d = X[2], P[2]
    respond(a,b,c,d)
    # # bound check
    # a, b = [0.1, 0.1], [0.1, 0.9]
    # c, d = X[2], P[2]
    # respond(a,b,c,d)
    # a, b = [0.9, 0.9], [0.9, 0.1]
    # c, d = X[2], P[2]
    # respond(a,b,c,d)
    # a, b = [0.9, 0.9], [0.1, 0.9]
    # c, d = X[2], P[2]
    # respond(a,b,c,d)
    # a, b = [0.1, 0.1], [0.9, 0.1]
    # c, d = X[2], P[2]
    # respond(a,b,c,d)

    

@ti.func
def respond(a,b,c,d):
    ab = b-a #; p(ab)
    cd = d-c #; p(cd)
    inter = intersect(ab, cd, a, c)
    if collided(a,b,inter) and collided(c,d,inter):
        # t = cross(ab, cd)
        # n = t.dot(c-a)  # reflect direction norm n>0:above, n<0 below
        # if n > 0: 
        #     p(1111)
        # else:
        #     p(2)
        x,y = reflect(a, b, d)
        xp, yp = reflect(a, b, c)
        dp = -P[2] + [x,y]
        X[2] = [xp, yp]
        P[2] += dp * 1

@ti.func
def collided(a, b, inter):
    """
    check if intersection is actually on the line segment
    """
    x,y = a
    m,n = b
    if x > m:
        x , m = m , x
    if y > n:
        y , n = n, y
    return  x <= inter[0] <= m and y <= inter[1] <= n

@ti.func
def nonzero(v):
    if v == 0:
        v = 1e-5
    return v

@ti.func
def reflect(x, y, p) :
    """
        x,y are the two points which define the axis y=mx+b
        p is the point to be reflected
    """
    # basis vector of the reflection axis
    axis = y - x
    result = p
    if axis[0] == 0:
        result = [(x[0]-p[0])*2+p[0], p[1]]
    else:
        m = axis[1] / axis[0]
        # householder's transformation
        t = (1/(1+m**2))
        r1 = ti.Vector([1-m**2, 2*m]) * t # row1
        r2 = ti.Vector([2*m, m**2-1]) * t # row2

        b = x[1] - m*x[0]

        r = ti.Vector([p[0], p[1]-b])
        result = [r1.dot(r), r2.dot(r)+b]
    return result


    
@ti.kernel
def update():
    for i in range(3):
        V[i] = (P[i] - X[i]) / dt[None]
        S[i] = X[i]
        X[i] = P[i]

@ti.kernel
def VelocityCorrection():
    V[2] *= damping
    s = max(abs(V[2][0]), abs(V[2][1])) // 5
    if s != substep[None] and s>0:
        substep[None] = s
        dt[None] = 6e-3 / s
        # p(s)
        # p(dt[None])


@ti.kernel
def bound_check():
    for i in range(2,3):
        x,y = X[i]
        vx, vy = V[i]
        r = 10/800
        # bound check
        if x-r <= 0.1 :
            x = 0.1 +r
            vx = abs(vx)
        elif x+r >= 0.9:
            x = 0.9 - r
            vx = -abs(vx)
        if y-r <= 0.1:
            y = 0.1 + r
            vy = abs(vy)
        elif y+r >= 0.9:
            y = 0.9 - r
            vy = -abs(vy)
        X[i] = x, y
        V[i] = vx, vy

@ti.func
def p(s):
    print(s)

@ti.func
def perp(p):
    p[0], p[1] = p[1], -p[0]
    return p

@ti.func
def cross(x,y):
    return perp(x-y)


@ti.func
def intersect(ab, cd, a, c):
    """
    find the intersection of two infinite line
    """
    ca = a - c
    p= perp(ab)

    denom = p.dot(cd)
    num = p.dot(ca)
    return (num / denom) * cd + c


gui = ti.GUI('BOX', res=(800, 800), background_color=0xdddddd)
init()

import time
sleep = 0
show_p = True
speed_m = 1
skip = False
while gui.running:
    if not skip:
        time.sleep(sleep)
    else:
        skip = False
    for _ in range(substep[None]):
        propose()
        for _ in range(n_iter):
            project()
        update()
        bound_check()
        #for _ in range(n_iter):
        project()
    VelocityCorrection()
    

    x = X.to_numpy()
    s = S.to_numpy()
    gui.circles(x[:3],color=0xffaa77, radius=10)
    if show_p:
        gui.circles(s[:3],color=0x3D85C6, radius=10)
    gui.line(begin=x[0], end=x[1], radius=2, color=0x990000)
    gui.line(begin=x[2], end=s[2], radius=2, color=0x990000)
    gui.line(begin=[0.1, 0.1], end=[0.1, 0.9], radius=3, color=0x000000)
    gui.line(begin=[0.1, 0.1], end=[0.9, 0.1], radius=3, color=0x000000)
    gui.line(begin=[0.9, 0.9], end=[0.1, 0.9], radius=3, color=0x000000)
    gui.line(begin=[0.9, 0.9], end=[0.9, 0.1], radius=3, color=0x000000)

    # control
    if gui.get_event(ti.GUI.PRESS):
        skip=True
        if gui.event.key == 'Up':
            speed_m *= 1.1
            print('speed:',speed_m)
        elif gui.event.key == 'Down':
            speed_m /= 1.1
            print('speed:',speed_m)
        elif gui.event.key == 'Left':
            sleep = max(0, sleep - 0.2)
            print('sleep time:', sleep)
        elif gui.event.key == 'Right':
            sleep = min(1, sleep + 0.2)
            print('sleep time:', sleep)
        elif gui.event.key == 'p':
            show_p = not show_p
        elif gui.event.key == 'a':
            V[2][0] = -1 * speed_m
        elif gui.event.key == 'd':
            V[2][0] = 1 * speed_m
        elif gui.event.key == 'w':
            V[2][1] = 1 * speed_m
        elif gui.event.key == 's':
            V[2][1] = -1 * speed_m
        elif gui.event.key == 'q':
            V[2] = [0,0]
        elif gui.event.key == ti.GUI.SPACE:
            X[2] = [gui.event.pos[0], gui.event.pos[1]]
            V[2] = [0,0]
        elif gui.event.key == ti.GUI.LMB:
            X[0] = [gui.event.pos[0], gui.event.pos[1]]
        elif gui.event.key == ti.GUI.RMB:
            X[1] = [gui.event.pos[0], gui.event.pos[1]]

    gui.text(content='Press WASD to move around. Press Space to relocate. Press Q to stop',pos=(0,0.99), color=0x0)
    gui.text(content='Left/Right Click to move the line',pos=(0,0.975), color=0x0)
    gui.text(content=f'Up/Down arrow to change the speed [v:{speed_m}][subsetp:{substep[None]}]{V[2].value}',pos=(0,0.96), color=0x0)
    gui.text(content=f'Left/Right arrow to change frame rate [frame sleep:{sleep}]',pos=(0,0.945), color=0x0)
    gui.text(content=f'Press P to show/unshow the previous postion',pos=(0,0.93), color=0x0)
    gui.show()