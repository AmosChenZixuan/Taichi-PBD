import taichi as ti
ti.init(arch=ti.gpu)


from ball import Ball
from cloth import Cloth, Particle
from constraint import PointConstraint, DistanceConstraint

@ti.data_oriented
class Main:
    def INIT(self):
        self.dt = 4e-3
        self.gravity = ti.Vector([0, -9.8]) 
        self.n_iter = 3

    def __init__(self):
        self.INIT()
        # create objects
        self.cloth = Cloth([10,3], [0.35, 0.4], 0.3, 0.09, start_idx=0)
        self.cloth2 = Cloth([5,7], [0.2, 0.8], 0.15, 0.21, start_idx=len(self.cloth))
        self.ball = Ball(pid=len(self.cloth)+len(self.cloth2), x=0.2, y=0.9)
        # allocate variables
        n_p = len(self.cloth)+len(self.cloth2) + 1
        self.num_particles = n_p
        self.x = ti.Vector.field(2, dtype=ti.f32, shape=n_p)
        self.p = ti.Vector.field(2, dtype=ti.f32, shape=n_p)
        self.v = ti.Vector.field(2, dtype=ti.f32, shape=n_p)
        self.inv_m = ti.field(ti.f32, shape=n_p)
        self.C = []
        # init ball particle
        idx = self.ball.pid
        self.x[idx] = self.ball.at()
        self.inv_m[idx] = 1
        # init cloth particles
        self.init_cloth(self.cloth)
        self.init_cloth(self.cloth2)

    def init_cloth(self, cloth):
        w, h = cloth.dims()
        c_idx = 0
        p_idx = list(iter(cloth))
        for i in range(w):
            for j in range(h):
                idx = j*w + i
                pid = p_idx[idx]
                pos = cloth.at(pid)
                self.x[pid] = pos
                if idx == 0:
                    self.inv_m[pid] = 0.0   # cant be moved
                    #self.C.append(PointConstraint(c_idx, pid, pos))
                    c_idx += 0
                elif idx == w-1:
                    self.inv_m[pid] = 0.0   # cant be moved
                    self.C.append(DistanceConstraint(c_idx, p_idx[j*(w) + i - 1], pid, 0.03)) # particle on the left
                    #self.C.append(PointConstraint(c_idx+1, pid, pos))
                    c_idx += 1
                else:
                    self.inv_m[pid] = 1.0
                    if i > 0:
                        self.C.append(DistanceConstraint(c_idx, p_idx[j*(w) + i - 1], pid, 0.03)) # particle on the left
                        c_idx += 1
                        #print(j*(w) + i - 1, idx, 'h', c_idx)
                    if j > 0:
                        self.C.append(DistanceConstraint(c_idx, p_idx[(j-1)*w + i], pid, 0.03)) # particle above
                        c_idx += 1
                        #print((j-1)*w + i, idx, 'v', c_idx)
        print(c_idx)

    @ti.kernel
    def handle_external_force(self):
        for i in range(self.num_particles):
            gravity = self.gravity #if i != self.ball.pid else ti.Vector([0, -9.8]) 
            # Handle gravity:
            self.v[i] = self.v[i] + self.dt * self.inv_m[i] * gravity
            # Damping velocity:
            self.v[i] *= 0.99

    @ti.kernel
    def get_proposed_pos(self):
        for i in range(self.num_particles):
            self.p[i] = self.x[i] + self.dt * self.v[i]
            #self.at(i, self.p[i])

    @ti.kernel
    def update_vel_pos(self):
        for i in range(self.num_particles):
            self.v[i] = (self.p[i] - self.x[i]) / self.dt
            self.x[i] = self.p[i]
            #self.at(i, self.x[i])
        

    @ti.kernel
    def project_constraints(self):
        for c in ti.static(self.C):
            if c.CTYPE == 0:
                pass
                #self.p[c.pid] = c.pos
            elif c.CTYPE == 1:
                p1 = self.p[c.xpid]
                p2 = self.p[c.ypid]
                n = (p1 - p2).normalized()
                C = (p1 - p2).norm() - c.d  # |p1-p2|-d
                inv_m1 = self.inv_m[c.xpid]
                inv_m2 = self.inv_m[c.ypid]
                inv_m_sum = inv_m1 + inv_m2
                delta_p1 = - (inv_m1 * C / inv_m_sum) * n
                delta_p2 = (inv_m2 * C / inv_m_sum) * n
                self.p[c.xpid] += delta_p1 * c.k
                self.p[c.ypid] += delta_p2 * c.k

    

    @ti.kernel
    def bound_check(self):
        pid = self.ball.pid
        x,y = self.x[pid]
        vx, vy = self.v[pid]
        r = 10/720
        damping = 1
        collided = False
        # bound check
        if x - r <= 0 :
            x = r
            vx = abs(vx)
            collided = True
        elif x + r >= 1:
            x = 1 - r
            vx = -abs(vx)
            collided = True
        if y - r < 0:
            y = r
            vy = abs(vy)
            collided = True
        elif y + r > 1:
            y = 1 - r
            vy = -abs(vy)
            collided = True
        self.x[pid] = x, y
        #ball.at(ball.x[None])
        self.v[pid] = vx, vy
        if collided:
            self.v[pid] *= damping


    def gen_colli(self):
        pos = []
        for i in range(self.num_particles):
            x,y = self.x[i][0], self.x[i][1]
            pos.append((x,y,i))
        return self.sweep_prune(sorted(pos))#, key=lambda x:(x[0],-x[1],x[2])))



    def sweep_prune(self, pos):
        res = []
        active = [pos[0]]
        for i in range(1, len(pos)):
            cur = pos[i]
            tid = 0 
            while len(active) > 0 and tid < len(active):
                target = active[tid]
                r,d = 5/720, 15/720
                if self.ball.pid in [target[2], cur[2]]:
                    r,d = 22/720, 23/720
                if target[0] + r >= cur[0] - r: #candidate
                    # if abs(target[2]- cur[2]) < 5:
                    #     print( target[2], cur[2], target[0], cur[0])
                    p1 = ti.Vector([target[0], target[1]])
                    p2 = ti.Vector([cur[0], cur[1]])
                    if (p1-p2).norm() <= d:
                        # k=1
                        # if self.ball.pid in [target[2], cur[2]]:
                        #     k=1
                        #   print('!!',target[2], cur[2], r, d)
                        res.append(DistanceConstraint(99, target[2], cur[2], d))
                        break
                    tid += 1
                else:
                    active.pop(0)
            active.append(cur)

        return res

    @ti.kernel
    def solve(self, c:ti.template()):
        c.solve(self.p, self.inv_m)

        
    
if __name__ == '__main__':
    main = Main()

    gui = ti.GUI('PBD', res=(720, 720), background_color=0xdddddd)
    while gui.running:
        # PBD steps:
        # Handle external force:
        main.handle_external_force()
        # Explicit Euler gets proposed pos:
        main.get_proposed_pos()
        # Generate collision constraints: 
        collis = main.gen_colli()
        # project constraints:
        for _ in range(main.n_iter):
            main.project_constraints()
            for c in collis:
                main.solve(c)
        # for i in range(main.n_iter):
        #     for c in main.C:
        #         main.project_constraints(c)
        #     for c in collis:
        #         main.project_constraints(c)
        # Update velocity and pos:
        main.update_vel_pos()
        # bound check
        main.bound_check()

        # Render:
        x = main.x.to_numpy()
        gui.circles(x[:len(main.cloth)], color=Particle.COLOR, radius=main.cloth.r)
        gui.circles(x[len(main.cloth):len(main.cloth)+len(main.cloth2)], color=Particle.COLOR, radius=main.cloth2.r)
        # Draw distance constraint:
        for c in main.C+ collis:
            if c.CTYPE == 1:
                idx1 = c.xpid
                idx2 = c.ypid
                gui.line(begin=x[idx1], end=x[idx2], radius=2, color=DistanceConstraint.COLOR)
        gui.circle(x[-1], color=Ball.COLOR, radius=main.ball.r)

        # control
        if gui.get_event(ti.GUI.PRESS):
            pid = main.ball.pid
            if gui.event.key == 'a':
                main.v[pid][0] = -1
            elif gui.event.key == 'd':
                main.v[pid][0] = 1
            elif gui.event.key == 'w':
                main.v[pid][1] = 3
            elif gui.event.key == 's':
                main.v[pid][1] = -2
            elif gui.event.key == 'q':
                main.v[pid] = [0,0]
            elif gui.event.key == ti.GUI.LMB:
                main.x[pid] = [gui.event.pos[0], gui.event.pos[1]]
                main.v[pid] = [0,0]

        gui.text(content='Press WASD to move around',pos=(0,0.99), color=0x0)
        gui.text(content='Click to relocate',pos=(0,0.95), color=0x0)
        gui.show()