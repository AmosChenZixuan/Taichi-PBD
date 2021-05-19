import taichi as ti
#ti.init(arch=ti.cpu)

class Constraint:
    CTYPE = -1
    COLOR=0x445566
    def __init__(self, cid):
        self.cid = cid
    
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
        P[self.xpid] += delta_p1 * self.k
        P[self.ypid] += delta_p2 * self.k
        #print(delta_p1, delta_p2)