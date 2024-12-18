# Fluid solver based on lattice boltzmann method using taichi language
# Author : Wang (hietwll@gmail.com)

import sys
import numpy as np

import taichi as ti
import taichi.math as tm

# ti.init(arch=ti.cpu, cpu_max_num_threads = 1, debug = True)
ti.init(arch=ti.cuda)

@ti.data_oriented
class lbm_solver:
    def __init__(
        self,
        name, # name of the flow case
        bc_type,  # [left,top,right,bottom] boundary conditions: 0 -> Dirichlet ; 1 -> Neumann
        bc_value,  # if bc_type = 0, we need to specify the velocity in bc_value
        ):
        self.name = name
        self.nx = 301 
        self.ny = 301 
        self.W = 7.5
        self.rho_l = 1000.0 # density of water
        self.rho_v = 100.0 # density of vapor
        self.sigma = 1e-4
        self.tau_phi = 0.5
        self.k_water = 0.68
        self.k_vapor = 0.0304
        self.Cp_water = 4.217
        self.Cp_vapor = 1.009

        self.rho = ti.field(float, shape=(self.nx, self.ny))
        self.phi = ti.field(float, shape=(self.nx, self.ny))
        self.vel = ti.Vector.field(2, float, shape=(self.nx, self.ny))
        self.old_temperature = ti.field(float, shape=(self.nx, self.ny))
        self.new_temperature = ti.field(float, shape=(self.nx, self.ny))

        self.h_old = ti.Vector.field(9, float, shape=(self.nx, self.ny))
        self.h_new = ti.Vector.field(9, float, shape=(self.nx, self.ny))
        self.g_old = ti.Vector.field(9, float, shape=(self.nx, self.ny))
        self.g_new = ti.Vector.field(9, float, shape=(self.nx, self.ny))
        self.p_star = ti.field(float, shape=(self.nx, self.ny))

        self.w = ti.field(float, shape=9)
        self.e = ti.Vector.field(2, int, shape=9)
        self.bc_type = ti.field(int, 4)
        self.bc_type.from_numpy(np.array(bc_type, dtype=np.int32))
        self.bc_value = ti.Vector.field(2, float, shape=4)
        self.bc_value.from_numpy(np.array(bc_value, dtype=np.float32))

        self.image = ti.Vector.field(3, dtype=ti.f32, shape=(self.nx, self.ny))  # RGB image

        self.phi_grad = ti.Vector.field(2, float, shape=(self.nx, self.ny))
        self.rho_grad = ti.Vector.field(2, float, shape=(self.nx, self.ny))
        self.vel_grad = ti.Matrix.field(2, 2, float, shape=(self.nx, self.ny))
        self.phi_laplatian = ti.field(float, shape=(self.nx, self.ny))
        self.vel_div = ti.field(float, shape=(self.nx, self.ny))

    
    @ti.func
    def is_INB(self, x, y, k):
        res = False
        nx, ny = x + self.e[k][0], y + self.e[k][1] 
        if nx >= 0 and nx < self.nx and ny >= 0 and ny < self.ny:
            if (self.phi[nx, ny] - 0.5) * (self.phi[x, y] - 0.5) < 0:
                res = True
        
        return res


    @ti.func
    def CalPhiGrad(self, i, j):
        res = ti.Vector([0.0, 0.0]) 
        for k in ti.static(range(9)):
            nxt_x, nxt_y = i + self.e[k][0], j + self.e[k][1]
            if nxt_x < self.nx and nxt_x >= 0 and nxt_y < self.ny and nxt_y >= 0 :
                res += self.w[k] * self.phi[nxt_x, nxt_y] * ti.Vector([self.e[k][0], self.e[k][1]])
            
        return 3.0 * res


    @ti.func
    def CalRhoGrad(self, i:int, j:int):
        res = ti.Vector([0.0, 0.0]) 
        for k in ti.static(range(9)):
            nxt_x, nxt_y = i + self.e[k][0], j + self.e[k][1]
            if nxt_x < self.nx and nxt_x >= 0 and nxt_y < self.ny and nxt_y >= 0 :
                res += self.w[k] * self.rho[nxt_x, nxt_y] * ti.Vector([self.e[k][0], self.e[k][1]])
            
        return 3.0 * res
    

    @ti.func
    def CalRhoGradReal(self, x:float, y:float):
        x1, y1 = int(x), int(y)
        x2, y2 = x1 + 1, y1 + 1
        tx, ty = x - float(x1), y - float(y1)

        rho11 = self.rho[x1, y1]
        rho12 = self.rho[x1, y2]
        rho21 = self.rho[x2, y1]
        rho22 = self.rho[x2, y2]

        rho_dx = (1 - ty) * (rho21 - rho11) + ty * (rho22 - rho12)
        rho_dy = (1 - tx) * (rho12 - rho11) + tx * (rho22 - rho21)
        return ti.Vector([rho_dx, rho_dy])


    @ti.func
    def CalVelGrad(self, i, j):
        res = ti.Matrix([[0.0, 0.0], [0.0, 0.0]]) 
        for k in ti.static(range(9)):
            nxt_x, nxt_y = i + self.e[k][0], j + self.e[k][1]
            if nxt_x < self.nx and nxt_x >= 0 and nxt_y < self.ny and nxt_y >= 0 :
                ele00 = self.vel[nxt_x, nxt_y][0] * self.e[k][0]
                ele01 = self.vel[nxt_x, nxt_y][0] * self.e[k][1]
                ele10 = self.vel[nxt_x, nxt_y][1] * self.e[k][0]
                ele11 = self.vel[nxt_x, nxt_y][1] * self.e[k][1]
                res += self.w[k] * ti.Matrix([[ele00, ele01], [ele10, ele11]])
            
        return 3.0 * res


    @ti.func
    def CalPhiLaplatian(self, i, j):
        res = 0.0
        for k in ti.static(range(9)):
            nxt_x, nxt_y = i + self.e[k][0], j + self.e[k][1]
            if nxt_x < self.nx and nxt_x >= 0 and nxt_y < self.ny and nxt_y >= 0 :
                res += self.w[k] * (self.phi[nxt_x, nxt_y] - self.phi[i, j])
            
        return 6.0 * res

    
    @ti.func
    def CalVelDiv(self, i, j):
        res = 0.0
        if i>=1 and i<self.nx-1 and j>=1 and j<self.ny-1 :
            res = 0.5 * (self.vel[i+1, j][0] + self.vel[i, j+1][1] - self.vel[i-1, j][0] - self.vel[i, j-1][1])
        
        return res
    

    @ti.func
    def CalMFlux(self, i, j, k):
        res = 0.0
        if self.is_INB(i, j, k):
            res = 5e-1
            # T_grad = 0.5 * ti.Vector([self.old_temperature[i + 1, j] - self.old_temperature[i - 1, j],
            #                         self.old_temperature[i, j + 1] - self.old_temperature[i, j - 1]])
            # normal = self.CalPhiGrad(i, j).normalized()
            # h_fg = 2260000.0
        else:
            res = 0.0
        
        return res

    
    @ti.func
    def CalMRate(self, x, y):
        max_idx = 1000
        max_inner = -1000.0
        distribute_u = 0.0

        ti.loop_config(serialize=True)
        for k in range(1, 9):
            if self.is_INB(x, y, k):
                nx, ny = x + self.e[k][0], y + self.e[k][1] 
                if nx >= 0 and nx < self.nx and ny >= 0 and ny < self.ny:
                    u = (0.5 - self.phi[x, y]) / (self.phi[nx, ny] - self.phi[x, y])
                    inter_x, inter_y = float(x) + u * self.e[k][0], float(y) + u * self.e[k][1]
                    rho_grad = self.CalRhoGradReal(inter_x, inter_y)
                    align = tm.dot(rho_grad.normalized(), self.e[k].normalized())
                    if align > max_inner:
                        max_inner = align
                        max_idx = k
                        distribute_u = u
        
        res = 0.0
        if max_idx != 1000:
            flux = self.CalMFlux(x, y, max_idx)
            res = flux * (1.0 - distribute_u)
        else:
            res = 0.0
        
        return res



    @ti.func
    def CalF(self, i, j):
        # return ti.Vector([0.0, 0.0])
        beta = 12.0 * self.sigma / self.W
        kappa = 1.5 * self.W
        phi = self.phi[i, j]
        miu = 2.0 * beta * phi * (1 - phi) * (1 - 2.0 * phi) - kappa * self.phi_laplatian[i, j]
        # F_s = miu * self.phi_grad[i, j]
        F_s = ti.Vector([0.0, 0.0])

        # F_b = (self.rho_l - self.rho[i, j]) * ti.Vector([0.0, -10.0])
        F_b = ti.Vector([0.0, 0.0])

        # F_p = - self.p_star[i, j] * self.rho_grad[i, j] / 3.0
        F_p = ti.Vector([0.0, 0.0])

        niu = (self.tau_phi - 0.5) / 3.0
        u_grad = self.vel_grad[i, j]
        F_eta = niu * (u_grad + u_grad.transpose()) @ self.rho_grad[i, j]
        F_eta = ti.Vector([0.0, 0.0])

        F_a = ti.Vector([0.0, 0.0])
        # if i!=0 and i!=self.nx-1 and j!=0 and j!=self.ny-1:
            # F_a = self.rho[i, j] * self.vel[i, j] * self.vel_div[i, j]
        
        res = F_s + F_b + F_p + F_eta + F_a

        # if i == 121 and j == 153:
            # print(F_s, miu, self.phi_grad[i, j])
        
        return res


    @ti.func
    def h_eq(self, i, j, k):
        eu = tm.dot(self.e[k], self.vel[i, j])
        uv = tm.dot(self.vel[i, j], self.vel[i, j])
        return self.w[k] * self.phi[i, j] * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * uv)

    
    @ti.func
    def g_eq(self, i, j, k):
        eu = tm.dot(self.e[k], self.vel[i, j])
        uv = tm.dot(self.vel[i, j], self.vel[i, j])
        res = self.w[k] * (self.p_star[i, j] + 3.0 * eu + 4.5 * eu * eu - 1.5 * uv)
        return res


    @ti.kernel
    def init(self):
        self.e[0] = ti.Vector([0, 0])
        self.e[1] = ti.Vector([1, 0])
        self.e[2] = ti.Vector([0, 1])
        self.e[3] = ti.Vector([-1, 0])
        self.e[4] = ti.Vector([0, -1])
        self.e[5] = ti.Vector([1, 1])
        self.e[6] = ti.Vector([-1, 1])
        self.e[7] = ti.Vector([-1, -1])
        self.e[8] = ti.Vector([1, -1])

        self.w[0] = 4.0 / 9.0
        self.w[1] = 1.0 / 9.0
        self.w[2] = 1.0 / 9.0
        self.w[3] = 1.0 / 9.0
        self.w[4] = 1.0 / 9.0
        self.w[5] = 1.0 / 36.0
        self.w[6] = 1.0 / 36.0
        self.w[7] = 1.0 / 36.0
        self.w[8] = 1.0 / 36.0

        mid_x = self.nx // 2
        mid_y = self.ny // 2
        r_init = self.nx / 10

        for i, j in self.phi:
            if (i - mid_x)**2 + (j - mid_y)**2 < r_init**2:
                self.phi[i, j] = 0
                self.rho[i, j] = self.rho_v
            else:
                self.phi[i, j] = 1
                self.rho[i, j] = self.rho_l
            
            self.p_star[i, j] = 0.0
            self.vel[i, j] = ti.Vector([0.0, 0.0])
            for k in ti.static(range(9)):
                self.h_old[i, j][k] = self.h_new[i, j][k] = self.h_eq(i, j, k)
                self.g_old[i, j][k] = self.g_new[i, j][k] = self.g_eq(i, j, k)
                self.p_star[i, j] += self.g_new[i, j][k]

    
    @ti.func
    def GetTemp(self, i, j):
        res = 0.0
        if i < 0 or i >= self.nx or j < 0 or j >= self.ny:
            res = 0.0
        elif j == 0:
            res = 100.0
        else:
            res = self.old_temperature[i, j]

        return res

    
    @ti.kernel
    def CalDerivative(self):
        for i, j in ti.ndrange((1, self.nx - 1), (1, self.ny - 1)):
            self.phi_grad[i, j] = self.CalPhiGrad(i, j)
            self.rho_grad[i, j] = self.CalRhoGrad(i, j)
            self.vel_grad[i, j] = self.CalVelGrad(i, j)
            self.phi_laplatian[i, j] = self.CalPhiLaplatian(i, j)
            self.vel_div[i, j] = self.CalVelDiv(i, j)


    @ti.kernel
    def UpdateTemp(self):
        for i, j in ti.ndrange(self.nx, self.ny):
            X_water = self.k_water / (self.rho[i, j] * self.Cp_water)
            X_vapor = self.k_vapor / (self.rho[i, j] * self.Cp_vapor)
            X = 0.0
            if self.phi[i, j] < 0.5 : 
                X = X_vapor
            elif self.phi[i, j] > 0.5 :
                X = X_water
            else :
                X = 0.5 * (X_vapor + X_water)
            
            laplacian = self.GetTemp(i+1, j) + self.GetTemp(i-1, j) + self.GetTemp(i, j+1) + self.GetTemp(i, j-1)\
                        - 4 * self.GetTemp(i, j)
            Tgrad = 0.5 * ti.Vector([self.GetTemp(i+1, j)- self.GetTemp(i-1, j), self.GetTemp(i, j+1) - self.GetTemp(i, j-1)])
            self.new_temperature[i, j] = self.GetTemp(i, j) + X * laplacian - tm.dot(Tgrad, self.vel[i, j])
            
        for i, j in ti.ndrange(self.nx, self.ny):
            self.old_temperature[i, j] = self.new_temperature[i, j]


    @ti.kernel
    def Collision(self):
        for i, j in ti.ndrange((0, self.nx), (0, self.ny)):
            for k in ti.static(range(9)):
                # ---- update phi ----
                heq = self.h_eq(i, j, k)

                # formula (22)
                self.h_old[i, j][k] = self.h_old[i, j][k] - (self.h_old[i, j][k] - heq) / self.tau_phi

                R = 0.0
                F = 0.0

                # if self.is_INB(i, j):
                if True:

                    # formula (24)
                    R = tm.dot(self.w[k] * self.e[k], 4 * self.phi[i, j] * (1 - self.phi[i, j]) / self.W * tm.normalize(self.phi_grad[i, j]))
                    # formula (25)
                    F = self.w[k] * (1 + 3 * (tm.dot(self.e[k], self.vel[i, j]) * (self.tau_phi - 0.5) / self.tau_phi))
                    F *= -self.CalMRate(i, j) / self.rho_l

                    self.h_old[i, j][k] += (2 * self.tau_phi - 1) / (2 * self.tau_phi) * R + F

                
                # ---- update vel ----
                geq = self.g_eq(i, j, k)
                self.g_old[i, j][k] = self.g_old[i, j][k] - (self.g_old[i, j][k] - geq) / self.tau_phi

                P = 0.0
                # if self.is_INB(i, j):
                if True:
                    # formula(32)
                    P = self.w[k] * self.CalMRate(i, j) * (1 / self.rho_v - 1 / self.rho_l)

                # formula(33)
                G = 3 * self.w[k] * tm.dot(ti.Vector([self.e[k][0], self.e[k][1]]), self.CalF(i, j)) / self.rho[i, j]

                # formula(31)
                tau = self.tau_phi
                self.g_old[i, j][k] += (2 * tau - 1) / (2 * tau) * G + P

                # if i == 143 and j == 125 and k == 0:
                    # print(heq, R, F)
    

    @ti.kernel
    def Advection(self):
        for i, j in ti.ndrange((1, self.nx - 1), (1, self.ny - 1)):
            for k in ti.static(range(9)):
                i_source = i - self.e[k][0]
                j_source = j - self.e[k][1]
                self.h_new[i, j][k] = self.h_old[i_source, j_source][k]
                self.g_new[i, j][k] = self.g_old[i_source, j_source][k]

    
    @ti.kernel
    def ApplyBc(self):  # impose boundary conditions
        # left and right
        for j in range(1, self.ny - 1):
            # left: dr = 0; ibc = 0; jbc = j; inb = 1; jnb = j
            self.ApplyBcCore(1, 0, 0, j, 1, j)

            # right: dr = 2; ibc = nx-1; jbc = j; inb = nx-2; jnb = j
            self.ApplyBcCore(1, 2, self.nx - 1, j, self.nx - 2, j)

        # top and bottom
        for i in range(self.nx):
            # top: dr = 1; ibc = i; jbc = ny-1; inb = i; jnb = ny-2
            self.ApplyBcCore(1, 1, i, self.ny - 1, i, self.ny - 2)

            # bottom: dr = 3; ibc = i; jbc = 0; inb = i; jnb = 1
            self.ApplyBcCore(1, 3, i, 0, i, 1)


    @ti.func
    def ApplyBcCore(self, outer, dr, ibc, jbc, inb, jnb):
        if outer == 1:  # handle outer boundary
            if self.bc_type[dr] == 0:
                self.vel[ibc, jbc] = self.bc_value[dr]

            elif self.bc_type[dr] == 1:
                self.vel[ibc, jbc] = self.vel[inb, jnb]

        self.rho[ibc, jbc] = self.rho[inb, jnb]
        for k in ti.static(range(9)):
            self.g_old[ibc, jbc][k] = self.g_eq(ibc, jbc, k) - self.g_eq(inb, jnb, k) + self.g_old[inb, jnb][k]
            self.h_old[ibc, jbc][k] = self.h_eq(ibc, jbc, k) - self.h_eq(inb, jnb, k) + self.h_old[inb, jnb][k]


    @ti.kernel
    def MacroVari(self):
        for i, j in ti.ndrange((1, self.nx - 1), (1, self.ny - 1)):
            self.phi[i, j] = 0.0 
            for k in ti.static(range(9)):
                self.phi[i, j] += self.h_new[i, j][k]
            
            self.h_old[i, j] = self.h_new[i, j]
            self.phi[i, j] = tm.min(self.phi[i, j], 1.0)
            self.phi[i, j] = tm.max(self.phi[i, j], 0.0)

            self.rho[i, j] = self.phi[i, j] * self.rho_l + (1 - self.phi[i, j]) * self.rho_v

        for i, j in ti.ndrange((1, self.nx - 1), (1, self.ny - 1)):
            self.p_star[i, j] = 0.0
            for k in ti.static(range(9)):
                self.p_star[i, j] += self.g_new[i, j][k]

        for i, j in ti.ndrange((1, self.nx - 1), (1, self.ny - 1)):
            self.phi_grad[i, j] = self.CalPhiGrad(i, j)
            self.rho_grad[i, j] = self.CalRhoGrad(i, j)
            self.phi_laplatian[i, j] = self.CalPhiLaplatian(i, j)
            
        for i, j in ti.ndrange((1, self.nx - 1), (1, self.ny - 1)):
            FF = self.CalF(i, j)
            self.vel[i, j] = FF / (2.0 * self.rho[i, j])
            for k in ti.static(range(9)):
                self.vel[i, j] += self.g_new[i, j][k] * ti.Vector([self.e[k][0], self.e[k][1]])
            self.g_old[i, j] = self.g_new[i, j]
            # if i == 121 and j == 153:
                # print(FF, self.vel[i, j])


    @ti.kernel
    def UpdateImage(self):
        col_l = ti.Vector([0.0, 0.0, 0.8])
        col_v = ti.Vector([0.0, 0.8, 0.0])
        for i, j in self.old_temperature:
            self.image[i, j] = self.phi[i, j] * col_l + (1 - self.phi[i, j]) * col_v 
    

    @ti.kernel
    def CalR(self) -> int:
        res = 0
        for i in range(self.ny):
            cnt = 0
            for j in range(self.nx):
                if self.phi[i, j] < 0.5:
                    cnt += 1
            
            ti.atomic_max(res, cnt)
        
        return res
    

    def solve(self):
        gui = ti.GUI(self.name, (self.nx, self.ny))
        self.init()
        frame = 0
        max_r = 0
        while not gui.get_event(ti.GUI.ESCAPE, ti.GUI.EXIT):
            # print(frame)
            sys.stdout.flush()
            frame += 1
            # self.UpdateTemp()
            self.CalDerivative()
            self.Collision()
            self.Advection()
            self.MacroVari()
            # self.ApplyBc()

            cur_r = self.CalR()
            if cur_r > max_r:
                max_r = cur_r
                print(frame, cur_r)
            #     print(self.phi[150, 100])
            
            self.UpdateImage()
            gui.set_image(self.image)
            gui.show()


if __name__ == '__main__':
    lbm = lbm_solver(
        name = "LBM",
        bc_type = [0, 0, 0, 0],
        bc_value = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
    )
    lbm.solve()