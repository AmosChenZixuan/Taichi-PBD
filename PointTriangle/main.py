import taichi as ti 
ti.init(arch=ti.gpu)
from box import Box
from mesh import Mesh
from random import randint
import time
# Constant
dt = 1e-3
gravity = ti.Vector([0, -50])
#gravity = ti.Vector([0, 0])
damping = 0.99
n_iter = 40
n_mesh = 15
# Boundary
box = Box(0.1, 0.9, 0.8, 0.8)
# Particles 
meshes = []
n_particles = 0
for i in range(n_mesh):
    if i == 0:
        new_mesh = Mesh(width=2, height=4,start=n_particles, pos=(0.5, 0.9))
    # elif i == 1:
    #     new_mesh = Mesh(width=8, height=3, start=n_particles, pos=(0.6, 0.2))
    else:
        new_mesh = Mesh(width=randint(2,4), height=randint(2,4), d=randint(3,5)/randint(80,100), start=n_particles, pos=(0.1+0.15*(i//4), 0.9-0.15*(i%4)))
    meshes.append(new_mesh)
    n_particles = new_mesh.end
print('Num of particles:', n_particles)
X = ti.Vector.field(2, dtype=ti.f32, shape=n_particles)
P = ti.Vector.field(2, dtype=ti.f32, shape=n_particles)
V = ti.Vector.field(2, dtype=ti.f32, shape=n_particles)
W = ti.field(ti.f32, shape=n_particles)
# constraints
## distance
n_dc = sum([len(m.edges) for m in meshes])
Distance_C = ti.Vector.field(2, dtype=ti.i32, shape=n_dc)
Distance = ti.field(ti.f32, shape=n_dc)
K = ti.field(ti.f32, shape=n_dc)
## triangle collision
n_tc = sum([len(m.squares) for m in meshes]) * n_particles
# [a, b, c, d, point] 
# two triangle in one square: abc, bcd
Triangle_C = ti.Vector.field(5, dtype=ti.i32, shape=n_tc)
TCActivation = ti.field(ti.i32, shape=n_tc)


def init():
    dc_idx = 0
    tc_idx = 0
    for m in meshes:
        m.init(X,V,W)
        for x,y,dis,k in m.edges:
            Distance_C[dc_idx] = [x,y]
            Distance[dc_idx] = dis
            K[dc_idx] = k
            dc_idx += 1

        for a,b,c,d in m.squares:
            for n in range(n_particles):
                if n in [a,b,c,d]:
                    continue
                Triangle_C[tc_idx] = [a, b, c, d, n]
                TCActivation[tc_idx] = 1
                tc_idx += 1
    print(f'Distance Cons(cur/cap): {dc_idx}/{n_dc}')
    print(f'Tri-point Cons(cur/cap): {tc_idx}/{n_tc}')

@ti.kernel
def propose():
    """Semi-Implicit Euler Integration"""
    for i in range(n_particles):
        V[i] += dt * W[i] * gravity 
        P[i] = X[i] + dt * V[i] * damping

@ti.kernel
def project():
    # distance constraint
    for i in range(n_dc):
        idx1 = Distance_C[i][0]
        idx2 = Distance_C[i][1]
        k    = K[i]
        d    = Distance[i]

        p1 = P[idx1]
        p2 = P[idx2]
        n = (p1 - p2).normalized()
        c = (p1 - p2).norm() - d  # |p1-p2|-d
        inv_m1 = W[idx1]
        inv_m2 = W[idx2]
        inv_m_sum = inv_m1 + inv_m2
        delta_p1 = ti.Vector([0.0,0.0])
        delta_p2 = ti.Vector([0.0,0.0])
        if inv_m_sum != 0:
            delta_p1 = - (inv_m1 * c / inv_m_sum) * n
            delta_p2 = (inv_m2 * c / inv_m_sum) * n
        P[idx1] += delta_p1 * k
        P[idx2] += delta_p2 * k
    
    # tri-point collision
    for i in range(n_tc):
        if TCActivation[i]:
            # b---c
            # | / |
            # a---d
            idxes = Triangle_C[i]
            a,b,c,d,p = P[idxes[0]], P[idxes[1]], P[idxes[2]], P[idxes[3]], P[idxes[4]] 
            # tri norms
            nab = cross(b-a, p-a)
            nbc = cross(c-b, p-b)
            ncd = cross(d-c, p-c)
            nda = cross(a-d, p-d)
            if contact(nab, nbc, ncd, nda):
                ialpha, ibeta, dis, flag = idxes[0], idxes[3], length2(X[idxes[4]], mid(a,d)), 1
                if length2(p, mid(b,c)) < dis:
                    ialpha, ibeta, dis, flag = idxes[1], idxes[2], length2(X[idxes[4]], mid(b,c)), 0

                if length2(p, mid(c,d)) < dis:
                    ialpha, ibeta, dis, flag = idxes[2], idxes[3], length2(X[idxes[4]], mid(c,d)), 0

                if length2(p, mid(a,b)) < dis:
                    ialpha, ibeta, dis, flag = idxes[0], idxes[1], length2(X[idxes[4]], mid(a,b)), 0

                alpha, beta = P[ialpha], P[ibeta]
                # find direction 
                axis = beta - alpha
                direction =  perp(axis).normalized() 
                if flag:
                    direction = [-direction[0], -direction[1]]
                distance = calc_dis(alpha,beta,p)
                #pp(direction)
                # update
                delta = distance * direction.normalized()
                P[idxes[4]] +=  delta * W[idxes[4]] * 0.15
                P[ialpha] -= delta* W[ialpha] * 0.075
                P[ibeta] -= delta* W[ibeta] * 0.075

@ti.func
def mid(x,y):
    """
        Return the mid point between x and y
    """
    return (x[0]+y[0])/2, (x[1]+y[1])/2


@ti.func
def length2(x, y):
    """
        Return the sqaure of the length of line segment xy
    """
    return (x[0]-y[0])**2 + (x[1]-y[1])**2

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
def contact(n1,n2,n3,n4):
    ret = 0
    if (n1>=0 and n2>=0 and n3>=0 and n4>=0) or\
        (n1<=0 and n2<=0 and n3<=0 and n4<=0):
        ret= 1
    return ret


@ti.func
def perp(p):
    """
        Return the perpendicular vector
    """
    p[0], p[1] = -p[1], p[0]
    return p


@ti.func
def cross(x,y):
    """
        Return the 2-D cross product of x and y
    """
    product = x[0]*y[1] - x[1]*y[0]
    return product

@ti.func
def pp(s):
    """kernel printer"""
    print('pp', s, s.normalized())
    time.sleep(2)

@ti.func
def empty():
    """kernel printer"""
    print('.',end='\r')


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
        r = 3/800
        # bound check
        if x - r <= box.left :
            x = box.left + r
            vx = 0#abs(vx)
        elif x + r >= box.right:
            x = box.right - r
            vx = 0#-abs(vx)
        if y - r < box.bottom:
            y = box.bottom + r
            vy = 0#abs(vy)
        elif y + r > box.top:
            y = box.top - r
            vy = 0#-abs(vy)
        X[i] = x, y
        V[i] = vx, vy

def substep():
    """
        perform one PBD step
    """
    propose()

    for _ in range(n_iter):
        project()

    update()
    bound_check()



gui = ti.GUI('BOX', res=(800, 800), background_color=0xdddddd)
init()
speed_m = 5
#print(Distance_C)
import time
while gui.running:
    #time.sleep(1)
    for _ in range(3):
        substep()

    box.draw(gui)
    x = X.to_numpy()
    for m in meshes:
        m.draw(gui, x)
    # for i in range(n_dc):
    #     p1 = Distance_C[i][0]
    #     p2 = Distance_C[i][1]
    #     d = Distance[i]
    #     # if (round(d,3) * 10**3)%10 == 0:
    #     #     color = 0x990000
    #     # else:
    #     color = 0x3D85C6
    #     gui.line(begin=x[int(p1)], end=x[int(p2)], radius=2, color=color)

    # Control
    if gui.get_event(ti.GUI.PRESS):
        if gui.event.key == ti.GUI.SPACE:
            X[0] = [gui.event.pos[0], gui.event.pos[1]]
            V[0] = [0,0]
        elif gui.event.key == 'a':
            V[0][0] = -1 * speed_m
            V[1][0] = -1 * speed_m
            V[2][0] = -1 * speed_m
            V[3][0] = -1 * speed_m
        elif gui.event.key == 'd':
            V[0][0] = 1 * speed_m
            V[1][0] = 1 * speed_m
            V[2][0] = 1 * speed_m
            V[3][0] = 1 * speed_m
        elif gui.event.key == 'w':
            V[0][1] = 1 * speed_m
            V[1][1] = 1 * speed_m
            V[2][1] = 1 * speed_m
            V[3][1] = 1 * speed_m
        elif gui.event.key == 's':
            V[0][1] = -1 * speed_m
            V[1][1] = -1 * speed_m
            V[2][1] = -1 * speed_m
            V[3][1] = -1 * speed_m
        elif gui.event.key == 'q':
            V[0] = [0,0]
            V[1] = [0,0]
            V[2] = [0,0]
            V[3] = [0,0]
        elif gui.event.key == 'p':
            print(X)
        elif gui.event.key == 'Right':
            time.sleep(1)
        elif gui.event.key == 'r':
            init()
            for i in range(n_particles):
                V[i] = [0,0]

    gui.circle(x[0], radius=5,color=0xFF0000 )
    # for i in range(n_tc):
    #     if TCActivation[i]:
    #         a,b,c,d,p = Triangle_C[i].value
            
    #         x = X.to_numpy() 
    #         gui.line(x[a],x[b],radius=5,color=0x00BFFF)
    #         gui.line(x[b],x[c],radius=5,color=0x00BFFF)
    #         gui.line(x[c],x[d],radius=5,color=0x00BFFF)
    #         gui.line(x[a],x[d],radius=5,color=0x00BFFF)

    #         gui.circles(x, radius=3, color=0xFF0000)
    #         gui.circle(x[p], radius=10, color=0xFF0000)
    #         gui.show()
    #         time.sleep(1)


    gui.show()

