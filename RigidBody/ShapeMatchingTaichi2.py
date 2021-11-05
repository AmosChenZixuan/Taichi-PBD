from typing import Dict
import taichi as ti
ti.init(arch=ti.gpu)

import math
import time
import numpy as np
from numpy.linalg import inv
from scipy.linalg import sqrtm
from numpy.linalg import eig

# CUDA
DIM = 2  
NUM_VERT = 20
X = ti.Vector.field(DIM, dtype=ti.f32, shape=NUM_VERT)
P = ti.Vector.field(DIM, dtype=ti.f32, shape=NUM_VERT)
V = ti.Vector.field(DIM, dtype=ti.f32, shape=NUM_VERT)
M = ti.field(dtype=ti.f32, shape=NUM_VERT)
CM_0 = ti.Vector.field(DIM, dtype=ti.f32, shape=())
CM = ti.Vector.field(DIM, dtype=ti.f32, shape=())
Q_0 = ti.Vector.field(DIM, dtype=ti.f32, shape=NUM_VERT)
Q = ti.Vector.field(DIM, dtype=ti.f32, shape=NUM_VERT)


Apq = ti.Matrix.field(DIM, DIM, dtype=ti.f32, shape=())
R = ti.Matrix.field(DIM, DIM, dtype=ti.f32, shape=())

# constant
ALPHA = 1.2
dt = 1/240
dt_inv = 1/dt
GRAVITY = ti.Vector([0.0, -9.8])


@ti.kernel
def init(cx:ti.f32, cy:ti.f32, r:ti.f32):
    theta = (2*math.pi) / NUM_VERT
    cos = ti.cos(theta)
    sin = ti.sin(theta)
    x = 0.0
    y = r
    for i in ti.static(range(NUM_VERT)):
        X[i] = x + cx, y + cy
        #X[NUM_VERT-i+1] += i%4*0.05 - 0.02*i , (i//4)*0.01 - 0.02*i
        V[i] = 0.5, -1.0
        M[i] = 1.0
        temp = x
        x = cos*x    - sin*y
        y = sin*temp + cos*y
    # X[0] = 0.55, 0.5
    # X[1] = 0.4, 0.4
    # X[2] = 0.45, 0.3
    # X[3] = 0.6, 0.4

    CM_0[None] = calc_CM(X)
    for i in Q_0:
        rel_pos = X[i] - CM_0[None]
        Q_0[i] = rel_pos
        Q[i] = rel_pos

@ti.func
def calc_CM(x):
    # center of mass
    cm = ti.Vector([0.0, 0.0])
    total_mass = 0.0
    for i in x:
        cm += x[i] * M[i]
        total_mass += M[i]
    cm /= total_mass
    return cm

@ti.func
def update():
    for i in P:
        p = R[None] @ Q_0[i] + CM[None]

        V[i] += ALPHA * (p - P[i]) * dt_inv + dt * GRAVITY 
        X[i] += dt*V[i]


@ti.func
def calc_Apq():
    SUM = ti.Vector([[0.0,0.0],[0.0,0.0]])
    for i in Q:
        #SUM += Q[i].outer_product(Q_0[i])
        SUM += Q[i] @ Q_0[i].transpose()
    Apq[None] = SUM


def calc_R():
    A = Apq[None].value.to_numpy()
    S = sqrtm(A.T@A)
    R[None] = ti.Vector(A @ inv(S))

@ti.func
def box_collision():
    for i in X:
        if X[i][1] < 0.0:
            X[i][1] = 0.0
            V[i] = 0.0, 0.0

@ti.kernel
def step1():
    for i in P:
        V[i] += dt * GRAVITY
        P[i] = X[i] + dt*V[i] 
    CM[None] = calc_CM(P)
    for i in Q:
        Q[i] = P[i] - CM[None]

    calc_Apq()

@ti.kernel
def step2():
    update()
    box_collision()

@ti.func
def p(t):
    print(t)



init(0, 0.5, 0.1)
gui = ti.GUI('Shape Matching',
            res=(600, 600), background_color=0xdddddd)
import time
while gui.running:
    x = X.to_numpy()
    #time.sleep(0.2)
    #print(V)
    gui.circles(x, color=0x328ac1, radius=10.5)   
    gui.circle(pos=CM[None], color=0xff0000, radius=5.5)
    gui.show()

    step1()
    calc_R()
    step2()
    #print('===')
