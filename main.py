"""
    This is a template program for fast implementation of PBD projects 
"""
import time
import taichi as ti
ti.init(arch=ti.gpu)

from boundary import Box
from mylib import cross, length2, perpend

# Constants
# values that you probably want to stay unchanged during the execution
dt = 1e-3 
gravity = ti.Vector([0, -9.8])
damping = 0.99
n_iter = 1
substeps = 3
window = (800, 800)
p_radius = 5

# Boundary
box = Box(0.1, 0.9, 0.8, 0.8, elastic=True) 

# Particles 
n_particles = 5
X = ti.Vector.field(2, dtype=ti.f32, shape=n_particles)
P = ti.Vector.field(2, dtype=ti.f32, shape=n_particles)
V = ti.Vector.field(2, dtype=ti.f32, shape=n_particles)
W = ti.field(ti.f32, shape=n_particles)

# constraints



# Engine Logics 
@ti.kernel
def init():
    X[0] = [0.3, 0.6]
    W[0] = 1.0
    X[1] = [0.3, 0.5]
    W[1] = 0.0
    X[2] = [0.5, 0.7]
    W[2] = 0.0
    X[3] = [0.7, 0.5]
    W[3] = 0.0
    X[4] = [0.5, 0.3]
    W[4] = 0.0

    idxes = 1,2,3,4,0
    a,b,c,d,p = X[idxes[0]], X[idxes[1]], X[idxes[2]], X[idxes[3]], X[idxes[4]] 
    # tri norms
    nab = cross(b-a, p-a) 
    nbc = cross(c-b, p-b)
    ncd = cross(d-c, p-c)
    nda = cross(a-d, p-d)
    cont = contact(nab, nbc, ncd, nda)


@ti.kernel
def propose():
    """Semi-Implicit Euler Integration"""
    for i in range(n_particles):
        V[i] += dt * W[i] * gravity 
        P[i] = X[i] + dt * V[i] * damping

@ti.kernel
def project():
    """One iteration of constraints solving"""
    idxes = 1,2,3,4,0
    a,b,c,d,p = P[idxes[0]], P[idxes[1]], P[idxes[2]], P[idxes[3]], P[idxes[4]] 
    # tri norms
    nab = cross(b-a, p-a)
    nbc = cross(c-b, p-b)
    ncd = cross(d-c, p-c)
    nda = cross(a-d, p-d)
    cont = contact(nab, nbc, ncd, nda)
    if  cont > 0:
        ialpha, ibeta =  idxes[0], idxes[1]
        if cont == 2 :
            ialpha, ibeta = idxes[1], idxes[2]

        if cont == 3 :
            ialpha, ibeta = idxes[2], idxes[3]

        if cont == 4:
            ialpha, ibeta = idxes[0], idxes[3]

        alpha, beta = P[ialpha], P[ibeta]
        # find direction 
        axis = beta - alpha
        direction =  perpend(axis).normalized() 
        if cont == 4:
            direction = [-direction[0], -direction[1]]
        distance = calc_dis(alpha,beta,p)
        #pp(direction)
        # update
        delta = distance * direction.normalized()
        P[idxes[4]] +=  delta * W[idxes[4]] * 0.15
        P[ialpha] -= delta* W[ialpha] * 0.075
        P[ibeta] -= delta* W[ibeta] * 0.075

@ti.func
def calc_dis(a,b,c):
    """
        Calculate the displacement 
    """
    dis = length2(c,b) - ((b[0]-a[0])*(b[0]-c[0]) + (b[1]-a[1])*(b[1]-c[1]))**2 / length2(a,b)
    if dis <= 0:
        dis = 0
    else: 
        dis = ti.sqrt(dis)
    return dis

@ti.func
def argmin(n1,n2,n3,n4):
    m = n1
    ret = 1
    if n2 < m:
        m = n2
        ret = 2
    if n3 < m:
        m = n3
        ret = 3
    if n4 < m:
        m = n4
        ret = 4
    return ret      

@ti.func
def contact(n1,n2,n3,n4):
    ret = 0
    r = 1e-3
    if (n1>=-r and n2>=-r and n3>=-r and n4>=-r) or\
        (n1<=r and n2<=r and n3<=r and n4<=r):
        ret = argmin(abs(n1),abs(n2),abs(n3),abs(n4))
    print(n1,n2,n3,n4)
    print(ret)
    return ret

@ti.kernel
def update():
    for i in range(n_particles):
        V[i] = (P[i] - X[i]) / dt
        X[i] = P[i]
    
@ti.kernel
def bound_check():
    for i in range(n_particles):
        x,y = X[i]
        vx, vy = V[i]
        r = p_radius/window[0]
        bounced, x,y,vx,vy = box.bound_check(x,y,vx,vy,r)
        if bounced:
            X[i] = x, y
            V[i] = vx, vy

def substep():
    """perform one PBD step"""
    propose()
    for _ in range(n_iter):
        project()
    update()
    bound_check()


def main():
    init()
    gui = ti.GUI('PBD', res=window, background_color=0xdddddd)
    pause=False
    while gui.running:
        # main loop
        if not pause:
            for _ in range(substeps):
                substep()
        # paint 
        box.draw(gui)
        x = X.to_numpy()
        gui.line(x[1], x[2], radius=2, color=0xFF0000)
        gui.line(x[2], x[3], radius=2, color=0xFF0000)
        gui.line(x[3], x[4], radius=2, color=0xFF0000)
        gui.line(x[4], x[1], radius=2, color=0xFF0000)
        gui.circles(x, radius=p_radius, color=0x00BFFF)
        # control
        if gui.get_event(ti.GUI.PRESS):
            if gui.event.key == ti.GUI.LMB:
                X[0] = [gui.event.pos[0], gui.event.pos[1]]
                V[0] = [0,0]
            elif gui.event.key == ti.GUI.SPACE:
                pause = not pause
            elif gui.event.key == 'r':
                init()
                for i in range(n_particles):
                    V[i] = [0,0]
            elif gui.event.key == 'Right':
                substep()
        

        gui.show()

if __name__ == '__main__':
    main()