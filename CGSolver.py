import taichi as ti

@ti.data_oriented
class CGSolver:
    def __init__(self, m, n, div, u, v, cell_type):
        self.m = m
        self.n = n
        self.div = div
        self.u = u
        self.v = v
        self.cell_type = cell_type

        # rhs of linear system
        self.b = ti.field(dtype=ti.f32, shape=(self.m, self.n))

        # lhs of linear system
        self.A = ti.field(dtype=ti.f32, shape=(self.m, self.n, 5))
        # self.Adiag = ti.field(dtype=ti.f32, shape=(self.m, self.n))
        # self.Ax = ti.field(dtype=ti.f32, shape=(self.m, self.n))
        # self.Ay = ti.field(dtype=ti.f32, shape=(self.m, self.n))

        # cg var
        self.p = ti.field(dtype=ti.f32, shape=(self.m, self.n))
        self.r = ti.field(dtype=ti.f32, shape=(self.m, self.n))
        self.x = ti.field(dtype=ti.f32, shape=(self.m, self.n))
        self.Ap = ti.field(dtype=ti.f32, shape=(self.m, self.n))
        self.sum = ti.field(dtype=ti.f32, shape=())
        self.alpha = ti.field(dtype=ti.f32, shape=())
        self.beta = ti.field(dtype=ti.f32, shape=())
        

    @ti.func
    def GetGridMark(self, x, y):
        res = 2
        if 0 <= x < self.m and 0 <= y < self.n:
            res = self.cell_type[x, y]
        else:
            res = 2
    
        return res


    @ti.kernel
    def system_init_kernel(self, scale_A: ti.f32, scale_b: ti.f32):
        #define right hand side of linear system
        for i, j in ti.ndrange(self.m, self.n):
            if self.GetGridMark(i, j) == 1:
                self.b[i, j] = -self.div[i, j]

        #modify right hand side of linear system to account for solid velocities
        #currently hard code solid velocities to zero
        for i, j in ti.ndrange(self.m, self.n):
            if self.GetGridMark(i, j) == 1:
                if self.GetGridMark(i - 1, j) == 2:
                    self.b[i, j] -= scale_b * (self.u[i, j] - 0)
                if self.GetGridMark(i + 1, j) == 2:
                    self.b[i, j] += scale_b * (self.u[i + 1, j] - 0)

                if self.GetGridMark(i, j - 1) == 2:
                    self.b[i, j] -= scale_b * (self.v[i, j] - 0)
                if self.GetGridMark(i, j + 1) == 2:
                    self.b[i, j] += scale_b * (self.v[i, j + 1] - 0)

        # define left handside of linear system
        for i, j in ti.ndrange(self.m, self.n):
            if self.GetGridMark(i, j) == 1:
                self.A[i, j, 0] = 4 * scale_A
                if self.GetGridMark(i - 1, j) == 1:
                    self.A[i, j, 1] -= scale_A
                elif self.GetGridMark(i - 1, j) == 2:
                    self.A[i, j, 0] -= scale_A

                if self.GetGridMark(i, j + 1) == 1:
                    self.A[i, j, 2] -= scale_A
                elif self.GetGridMark(i, j + 1) == 2:
                    self.A[i, j, 0] -= scale_A

                if self.GetGridMark(i + 1, j) == 1:
                    self.A[i, j, 3] -= scale_A
                elif self.GetGridMark(i + 1, j) == 2:
                    self.A[i, j, 0] -= scale_A

                if self.GetGridMark(i, j - 1) == 1:
                    self.A[i, j, 4] -= scale_A
                elif self.GetGridMark(i, j - 1) == 2:
                    self.A[i, j, 0] -= scale_A


    def system_init(self, scale_A, scale_b):
        self.b.fill(0)
        self.A.fill(0.0)

        self.system_init_kernel(scale_A, scale_b)


    def solve(self, max_iters):
        tol = 1e-8

        self.x.fill(0.0)
        self.compute_r()
        self.p.copy_from(self.r)

        self.reduce(self.r, self.r)
        init_rTr = self.sum[None]

        print("init rTr = {}".format(init_rTr))

        if init_rTr < tol:
            print("Converged: init rtr = {}".format(init_rTr))
        else:
            # p0 = 0
            # r0 = b - Ap0 = b
            # s0 = r0
            old_rTr = init_rTr
            iteration = 0

            for i in range(max_iters):
                # alpha = rTr / pAp
                self.compute_Ap()
                self.reduce(self.p, self.Ap)
                pAp = self.sum[None]
                self.alpha[None] = old_rTr / pAp

                # x = x + alpha * p
                self.update_x()

                # r = r - alpha * Ap
                self.update_r()

                # check for convergence
                self.reduce(self.r, self.r)
                rTr = self.sum[None]
                if rTr < tol:
                    break

                new_rTr = rTr
                self.beta[None] = new_rTr / old_rTr

                # p = r + beta * p
                self.update_p()
                old_rTr = new_rTr
                iteration = i

            print("Converged to {} in {} iterations".format(rTr, iteration))

    @ti.kernel
    def reduce(self, p: ti.template(), q: ti.template()):
        self.sum[None] = 0.0
        for i, j in ti.ndrange(self.m, self.n):
            if self.GetGridMark(i, j) == 1:
                self.sum[None] += p[i, j] * q[i, j]

    @ti.kernel
    def compute_Ap(self):
        for i, j in ti.ndrange(self.m, self.n):
            if self.GetGridMark(i, j) == 1:
                self.Ap[i, j] = self.A[i, j, 0] * self.p[i, j] \
                                + self.A[i, j, 1] * self.p[i - 1, j] \
                                + self.A[i, j, 2] * self.p[i, j + 1] \
                                + self.A[i, j, 3] * self.p[i + 1, j] \
                                + self.A[i, j, 4] * self.p[i, j - 1]

    @ti.kernel
    def update_x(self):
        for i, j in ti.ndrange(self.m, self.n):
            if self.GetGridMark(i, j) == 1:
                self.x[i, j] = self.x[i, j] + self.alpha[None] * self.p[i, j]


    @ti.kernel
    def update_r(self):
        for i, j in ti.ndrange(self.m, self.n):
            if self.GetGridMark(i, j) == 1:
                self.r[i, j] = self.r[i, j] - self.alpha[None] * self.Ap[i, j]

    @ti.kernel
    def update_p(self):
        for i, j in ti.ndrange(self.m, self.n):
            if self.GetGridMark(i, j) == 1:
                self.p[i, j] = self.r[i, j] + self.beta[None] * self.p[i, j]

    
    @ti.kernel
    def compute_r(self):
        for i, j in ti.ndrange(self.m, self.n):
            if self.GetGridMark(i, j) == 1:
                self.r[i, j] = self.b[i, j] \
                                - self.A[i, j, 0] * self.x[i, j] \
                                - self.A[i, j, 1] * self.x[i - 1, j] \
                                - self.A[i, j, 2] * self.x[i, j + 1] \
                                - self.A[i, j, 3] * self.x[i + 1, j] \
                                - self.A[i, j, 4] * self.x[i, j - 1]