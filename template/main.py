"""
    This is a template program for fast implementation of PBD projects 
"""
import time
import taichi as ti
ti.init(arch=ti.gpu)

from boundary import Box

# Constants
# values that you probably want to stay unchanged during the execution
dt = 1e-3 
gravity = ti.Vector([0, -9.8])
damping = 0.99
n_iter = 40
substeps = 6
window = (800, 800)
p_radius = 5

# Boundary
box = Box(0.1, 0.9, 0.8, 0.8, elastic=True) 

# Particles 
n_particles = 1
X = ti.Vector.field(2, dtype=ti.f32, shape=n_particles)
P = ti.Vector.field(2, dtype=ti.f32, shape=n_particles)
V = ti.Vector.field(2, dtype=ti.f32, shape=n_particles)
W = ti.field(ti.f32, shape=n_particles)

# constraints


# Engine Logics
def init():
    X[0] = [0.5, 0.5]
    W[0] = 1.0


@ti.kernel
def propose():
    """Semi-Implicit Euler Integration"""
    for i in range(n_particles):
        V[i] += dt * W[i] * gravity 
        P[i] = X[i] + dt * V[i] * damping

@ti.kernel
def project():
    """One iteration of constraints solving"""
    pass


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

    while gui.running:
        # main loop
        for _ in range(substeps):
            substep()
        # paint 
        box.draw(gui)
        x = X.to_numpy()
        gui.circles(x, radius=p_radius, color=0x00BFFF)
        # control
        if gui.get_event(ti.GUI.PRESS):
            if gui.event.key == ti.GUI.LMB:
                X[0] = [gui.event.pos[0], gui.event.pos[1]]
                V[0] = [0,0]
            elif gui.event.key == ti.GUI.SPACE:
                time.sleep(1)
            elif gui.event.key == 'r':
                init()
                for i in range(n_particles):
                    V[i] = [0,0]

        gui.show()

if __name__ == '__main__':
    main()