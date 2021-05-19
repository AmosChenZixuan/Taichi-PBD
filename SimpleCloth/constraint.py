import taichi as ti
#ti.init(arch=ti.cpu)

class Constraint:
    CTYPE = -1
    COLOR=0x445566
    def __init__(self, cid):
        self.cid = cid
        # self.pid=None
        # self.pos=None
        # self.xpid=None
        # self.ypid=None
        # self.d=None
        # self.k=None
    
    @ti.func
    def solve(self, P, W):
        print('        '+str(self.CTYPE))
        pass



class PointConstraint(Constraint):
    CTYPE = 0
    def __init__(self, idx, pid, pos):
        super().__init__(idx)
        self.pid = pid
        self.pos = pos

    @ti.func
    def solve(self, P, W):
        P[self.pid] = self.pos

class DistanceConstraint(Constraint):
    CTYPE = 1
    def __init__(self, idx, xpid, ypid, d,k=1):
        super().__init__(idx)
        self.xpid = xpid
        self.ypid = ypid
        self.d = d
        self.k = k

    @ti.func
    def solve(self, P, W):
        p1 = P[self.xpid]
        p2 = P[self.ypid]
        n = (p1 - p2).normalized()
        c = (p1 - p2).norm() - self.d  # |p1-p2|-d
        inv_m1 = W[self.xpid]
        inv_m2 = W[self.ypid]
        inv_m_sum = inv_m1 + inv_m2
        delta_p1 = - (inv_m1 * c / inv_m_sum) * n
        delta_p2 = (inv_m2 * c / inv_m_sum) * n
        P[self.xpid] = p1 + delta_p1 * self.k
        P[self.ypid] = p2 + delta_p2 * self.k



if __name__ == '__main__':

    # P = ti.Vector.field(2, dtype=ti.f32, shape=200)
    # W = ti.field(ti.f32, shape=200)
    # W[0] = 1.0
    # W[1] = 0.0
    # pc= PointConstraint(0, 0, [1,0])
    # print(pc.CTYPE)
    # print(pc.cid)

    # @ti.kernel
    # def show(c:ti.template()):
    #     c.solve(P, W)
    #     print(P[0], P[1])


    # show(pc)

    # dc = DistanceConstraint(1, 0, 1, 0.05)
    # show(dc)
    # show(dc)
    @ti.data_oriented
    class A:
        def __init__(self):
            self.p = ti.Vector.field(2, dtype=ti.f32, shape=200)
            self.w = ti.field(ti.f32, shape=200)
            self.w[0] = 1.0
            self.w[1] = 0.0
            pc= PointConstraint(0, 0, [1,0])
            dc = DistanceConstraint(1, 0, 1, 0.05)
            self.C = [pc, dc]

        #@ti.kernel
        def show(self, c:ti.template()):
            c.solve(self.p, self.w)
            print(self.p[0].value, self.p[1].value)


    a = A()
    for c in a.C:
        a.show(c)
