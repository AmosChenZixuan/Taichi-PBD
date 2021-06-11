import taichi as ti
import numpy as np

ti.init(arch=ti.gpu)

NUM_WATER_ROW = 200
NUM_WATER_COL = 30
NUM_WATER = NUM_WATER_ROW * NUM_WATER_COL
NUM_SMOKE_ROW = 8
NUM_SMOKE_COL = 5
MAX_NUM_PARTICLE = 102400

#
# GUI
WIDTH = 600
HEIGHT = 600
BACKGROUND_COLOUR = 0xf0f0f0
WATER_COLOR = 0x328ac1
SMOKE_COLOR = 0xf99613

# simulator variables 
paused = False
display_water = False
attract = 0
mouse_pos = (0, 0)

# parameters
SUBSTEPS = 2
SOLVE_ITERS = 10
MAX_PARTICLES_IN_A_GRID = 64
MAX_NEIGHBOURS = 64
KERNEL_SIZE = 25     # h in Poly6 kernel and Spiky kernel
KERNEL_SIZE_SQR = KERNEL_SIZE * KERNEL_SIZE
POLY6_CONST = 315 / 64 / np.pi / KERNEL_SIZE ** 9
SPIKY_GRAD_CONST = -45 / np.pi / KERNEL_SIZE ** 6
GRID_SIZE = KERNEL_SIZE
GRID_SHAPE = (WIDTH // GRID_SIZE + 1, HEIGHT // GRID_SIZE + 1)
PARTICLE_MASS = 1.0

COLLISION_EPSILON = 1e-3
LAMBDA_EPSILON = 200
S_CORR_DELTA_Q = 0.3
S_CORR_K = 0.1
S_CORR_N = 4
S_CORR_CONST = 1 / (POLY6_CONST * (KERNEL_SIZE_SQR - KERNEL_SIZE_SQR * S_CORR_DELTA_Q * S_CORR_DELTA_Q) ** 3)
dt = 1 / 60 / SUBSTEPS

# GPU variables
NUM_PARTICLES = ti.field(ti.int32, shape=())
x = ti.Vector.field(2, dtype=ti.f32, shape=MAX_NUM_PARTICLE)
p = ti.Vector.field(2, dtype=ti.f32, shape=MAX_NUM_PARTICLE)
v = ti.Vector.field(2, dtype=ti.f32, shape=MAX_NUM_PARTICLE)
lambda_ = ti.field(ti.f32, shape=MAX_NUM_PARTICLE)
dx = ti.Vector.field(2, dtype=ti.f32, shape=MAX_NUM_PARTICLE)
grid = ti.field(ti.i32, shape=(GRID_SHAPE[0], GRID_SHAPE[1], MAX_PARTICLES_IN_A_GRID))
num_particles_in_grid = ti.field(ti.i32, shape=GRID_SHAPE)
neighbours = ti.field(ti.i32, shape=(MAX_NUM_PARTICLE, MAX_NEIGHBOURS))
num_neighbours = ti.field(ti.i32, shape=MAX_NUM_PARTICLE)
x_display = ti.Vector.field(2, dtype=ti.f32, shape=MAX_NUM_PARTICLE)
RADIUS = ti.field(ti.f32, shape=MAX_NUM_PARTICLE)
mass = ti.field(ti.f32, shape=MAX_NUM_PARTICLE)
rho_0 = ti.field(ti.f32, shape=MAX_NUM_PARTICLE)
active = ti.field(ti.int32, shape=MAX_NUM_PARTICLE)

@ti.func
def calc_rho(m):
    return m * (POLY6_CONST * (KERNEL_SIZE_SQR) ** 3) * 0.5

@ti.kernel
def reset_particles():
    NUM_PARTICLES[None] = NUM_WATER
    for i in range(NUM_WATER_ROW):
        for j in range(NUM_WATER_COL):
            index =  i * NUM_WATER_COL + j
            x[index][0] = 5 + j * KERNEL_SIZE *0.8 + ti.random()
            x[index][1] = 5 + i * KERNEL_SIZE * 0.1 + ti.random()
            v[index] = 0, 0
            RADIUS[index] = 4
            mass[index] = 1
            rho_0[index] = calc_rho(mass[index])
            active[index] = -1

@ti.kernel
def emit_smoke():
    for i in range(NUM_SMOKE_ROW):
        for j in range(NUM_SMOKE_COL):
            index =  NUM_PARTICLES[None] + i * NUM_SMOKE_COL + j
            x[index][0] = 290 + j * KERNEL_SIZE *0.2#+ ti.random()
            x[index][1] = 100 + i * KERNEL_SIZE * 0.3 #+ ti.random()
            v[index] = 0, 0 
            RADIUS[index] = 2
            mass[index] = 0.6
            rho_0[index] = calc_rho(mass[index])
            active[index] = 30
    NUM_PARTICLES[None] += NUM_SMOKE_COL * NUM_SMOKE_ROW

@ti.kernel
def apply_external_forces(mouse_x: ti.f32, mouse_y: ti.f32, attract: ti.i32):
    for i in range(NUM_PARTICLES[None]):
        if not active[i]:
            continue
        gravity = -980
        if i > NUM_WATER:
            gravity = 2048
        v[i][1] = v[i][1] + dt * gravity  

        # mouse interaction
        if attract:
            r = ti.Vector([mouse_x * WIDTH, mouse_y * HEIGHT]) - x[i]
            r_norm = r.norm()
            if r_norm > 15:
                v[i] += attract * dt * 5e6 * r / r_norm ** 3   # F = GMm/|r|^2 * (r/|r|)

        p[i] = x[i] + dt * v[i]

    # pre-stabilization
    box_collision()

@ti.kernel
def find_neighbours():
    """
    We look up neighbours in 3 steps:
    1. clear grid to particle table (set size to 0 for each cell)
    2. put each particle into a grid cell by its position
    3. for each particle, look for neighbour in its closest 9 grid cells, put into neighbour table
    Note on Taichi: the outer-most for loop in each kernel function is parallelized, therefore we need to use
    atomic add to increment shared table values
    """
    for i, j in num_particles_in_grid:
        num_particles_in_grid[i, j] = 0

    for i in range(NUM_PARTICLES[None]):
        if not active[i]:
            continue
        grid_idx = int(p[i] / GRID_SIZE)
        old = ti.atomic_add(num_particles_in_grid[grid_idx], 1)
        if (old < MAX_PARTICLES_IN_A_GRID):
            grid[grid_idx[0], grid_idx[1], old] = i

    for x1 in range(NUM_PARTICLES[None]):
        if not active[x1]:
            continue
        neighbours_idx = 0
        grid_idx = int(p[x1] / GRID_SIZE)
        for grid_y in ti.static(range(-1, 2)):
            if 0 <= grid_idx[1] + grid_y < GRID_SHAPE[1]:
                for grid_x in ti.static(range(-1, 2)):
                    if 0 <= grid_idx[0] + grid_x < GRID_SHAPE[0]:
                        for i in range(num_particles_in_grid[grid_idx[0] + grid_x, grid_idx[1] + grid_y]):
                            x2 = grid[grid_idx[0] + grid_x, grid_idx[1] + grid_y, i]
                            if (p[x2] - p[x1]).norm_sqr() < KERNEL_SIZE_SQR and neighbours_idx < MAX_NEIGHBOURS and x2 != x1:
                                neighbours[x1, neighbours_idx] = x2
                                neighbours_idx += 1
        num_neighbours[x1] = neighbours_idx

@ ti.func
def poly6_kernel(r_sqr):
    ret_val = 0.
    if r_sqr < KERNEL_SIZE_SQR:
        ret_val = POLY6_CONST * (KERNEL_SIZE_SQR - r_sqr) ** 3
    return ret_val


@ ti.func
def spiky_grad_kernel(r):
    ret_val = ti.Vector([0., 0.])
    r_norm = r.norm()
    if 0 < r_norm < KERNEL_SIZE:
        ret_val = r / r_norm * SPIKY_GRAD_CONST * (KERNEL_SIZE - r_norm) ** 2
    return ret_val

@ ti.kernel
def solve_iter():
    """
    We try to satisfy the density constraint here. We compute lambdas in the first loop, then compute dx
    (Delta p in the original paper) in the second part, finally update the position.
    This follows a non-linear Jacobi Iteration pattern. We run multiple solve_iter steps in a sub-step, and
    multiple sub-steps in a frame
    """
    for x1 in range(NUM_PARTICLES[None]):
        if not active[x1]:
            continue
        sum_grad_pk_C_sq = 0.
        rho_i = poly6_kernel(0)
        sum_grad_pi_C = ti.Vector([0., 0.])
        for i in range(num_neighbours[x1]):
            if x1 >= NUM_WATER and i < NUM_WATER:
                continue
            x2 = neighbours[x1, i]
            r = p[x1] - p[x2]
            grad = spiky_grad_kernel(r) / rho_0[x1]
            sum_grad_pi_C += grad
            sum_grad_pk_C_sq += grad.norm_sqr()
            rho_i += poly6_kernel(r.norm_sqr())

        C_i = rho_i / rho_0[x1] - 1
        lambda_[x1] = -C_i / (sum_grad_pk_C_sq + sum_grad_pi_C.norm_sqr() + LAMBDA_EPSILON)

    for x1 in range(NUM_PARTICLES[None]):
        if not active[x1]:
            continue
        dx[x1] = ti.Vector([0., 0.])
        for i in range(num_neighbours[x1]):
            x2 = neighbours[x1, i]
            r = p[x1] - p[x2]
            s_corr = -S_CORR_K * (poly6_kernel(r.norm_sqr()) * S_CORR_CONST) ** S_CORR_N
            dx[x1] += (lambda_[x1] + lambda_[x2] + s_corr) * spiky_grad_kernel(r)

    for x1 in x:
        p[x1] = p[x1] + dx[x1] / rho_0[x1]

@ ti.func
def box_collision():
    """
    In position based dynamics, collision handling is just moving particles back into a valid position, its
    velocity will be implied by the position change
    We add a wall to left, right and bottom. We leave top open for more fun. Potential problem: particles outside of
    the screen does not belong to any grid cell, therefore they don't have liquid properties.
    """
    for i in range(NUM_PARTICLES[None]):
        if not active[i]:
            continue
        if p[i][0] < RADIUS[i]:
            p[i][0] = RADIUS[i] #+ COLLISION_EPSILON * ti.random()
        if p[i][0] > WIDTH - RADIUS[i]:
            p[i][0] = WIDTH - RADIUS[i] #- COLLISION_EPSILON * ti.random()
        if p[i][1] < RADIUS[i]:
            p[i][1] = RADIUS[i] #+ COLLISION_EPSILON * ti.random()
        if i < NUM_WATER:
            if p[i][1] > HEIGHT + 2*RADIUS[i]:
                p[i][1] = HEIGHT + 2*RADIUS[i]
        else:
            if p[i][1] > HEIGHT - RADIUS[i]:
                active[i] = 0

@ ti.kernel
def update():
    for i in range(NUM_PARTICLES[None]):
        if not active[i]:
            continue
        active[i] -= 1
        v[i] = (p[i] - x[i]) / dt * 0.99
        x[i] = p[i]
    

    box_collision()
    for i in range(NUM_PARTICLES[None]):
        x_display[i][0] = x[i][0] / WIDTH
        x_display[i][1] = x[i][1] / HEIGHT

def simulate(mouse_pos, attract):
    for i in range(SUBSTEPS):
        apply_external_forces(mouse_pos[0], mouse_pos[1], attract)
        find_neighbours()
        for _ in range(SOLVE_ITERS):
            solve_iter()

        
        update()

def render(gui):
    q = x_display.to_numpy()
    if display_water:
        for i in range(NUM_WATER):
            gui.circle(pos=q[i], color=WATER_COLOR, radius=RADIUS[i])
    for i in range(NUM_WATER, NUM_PARTICLES[None]):
        if not active[i]:
            continue
        gui.circle(pos=q[i], color=SMOKE_COLOR, radius=RADIUS[i])
    gui.show()

if __name__ == '__main__':
    gui = ti.GUI('Position Based Fluid',
                 res=(WIDTH, HEIGHT), background_color=BACKGROUND_COLOUR)

    reset_particles()

    while True:
        for e in gui.get_events(ti.GUI.PRESS):
            if e.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
                exit()
            elif e.key == 'p':
                paused = not paused
            elif e.key == 'r':
                reset_particles()
            elif e.key == 'e':
                display_water = not display_water
            elif e.key == gui.SPACE:
                emit_smoke()

        if gui.is_pressed(ti.GUI.RMB):
            mouse_pos = gui.get_cursor_pos()
            attract = 1
        elif gui.is_pressed(ti.GUI.LMB):
            mouse_pos = gui.get_cursor_pos()
            attract = -1
        else:
            attract = 0

        if not paused:
            simulate(mouse_pos, attract)

        render(gui)