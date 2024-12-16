# Fluid solver based on lattice boltzmann method using taichi language
# Author : Wang (hietwll@gmail.com)

import sys
import matplotlib
import numpy as np
from matplotlib import cm

import taichi as ti
import taichi.math as tm
import time

# ti.init(arch=ti.cpu, cpu_max_num_threads = 1, debug = True)
ti.init(arch=ti.cuda)

@ti.data_oriented
class lbm_solver:
    def __init__(
        self,
        name, # name of the flow case
        nx,  # domain size
        ny,
        bc_type,  # [left,top,right,bottom] boundary conditions: 0 -> Dirichlet ; 1 -> Neumann
        bc_value,  # if bc_type = 0, we need to specify the velocity in bc_value
        ):
        self.name = name
        self.nx = nx  # by convention, dx = dy = dt = 1.0 (lattice units)
        self.ny = ny
        self.W = 7.5
        self.rho_l = 1000.0 # density of water
        self.rho_v = 900.0 # density of vapor
        self.sigma = 1e-4
        self.tau_phi = 0.5
        self.k_water = 0.68
        self.k_vapor = 0.0304
        self.Cp_water = 4.217
        self.Cp_vapor = 1.009

        self.rho = ti.field(float, shape=(nx, ny))
        self.phi = ti.field(float, shape=(nx, ny))
        self.vel = ti.Vector.field(2, float, shape=(nx, ny))
        self.old_temperature = ti.field(float, shape=(nx, ny))
        self.new_temperature = ti.field(float, shape=(nx, ny))

        self.h_old = ti.Vector.field(9, float, shape=(nx, ny))
        self.h_new = ti.Vector.field(9, float, shape=(nx, ny))
        self.g_old = ti.Vector.field(9, float, shape=(nx, ny))
        self.g_new = ti.Vector.field(9, float, shape=(nx, ny))
        self.p_star = ti.field(float, shape=(nx, ny))

        self.w = ti.field(float, shape=9)
        self.e = ti.Vector.field(2, int, shape=9)
        self.bc_type = ti.field(int, 4)
        self.bc_type.from_numpy(np.array(bc_type, dtype=np.int32))
        self.bc_value = ti.Vector.field(2, float, shape=4)
        self.bc_value.from_numpy(np.array(bc_value, dtype=np.float32))

        self.image = ti.Vector.field(3, dtype=ti.f32, shape=(nx, ny))  # RGB image

        self.phi_grad = ti.Vector.field(2, float, shape=(nx, ny))
        self.rho_grad = ti.Vector.field(2, float, shape=(nx, ny))
        self.vel_grad = ti.Matrix.field(2, 2, float, shape=(nx, ny))
        self.phi_laplatian = ti.field(float, shape=(nx, ny))
        self.vel_div = ti.field(float, shape=(nx, ny))

    
    @ti.func
    def is_INB(self, x, y):
        res = False
        for dx, dy in ti.ndrange((-1, 2), (-1, 2)):
            nx, ny = x + dx, y + dy
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
    def CalRhoGrad(self, i, j):
        res = ti.Vector([0.0, 0.0]) 
        for k in ti.static(range(9)):
            nxt_x, nxt_y = i + self.e[k][0], j + self.e[k][1]
            if nxt_x < self.nx and nxt_x >= 0 and nxt_y < self.ny and nxt_y >= 0 :
                res += self.w[k] * self.rho[nxt_x, nxt_y] * ti.Vector([self.e[k][0], self.e[k][1]])
            
        return 3.0 * res


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
    def CalM(self, i, j):
        return 5e-1
        T_grad = 0.5 * ti.Vector([self.old_temperature[i + 1, j] - self.old_temperature[i - 1, j],
                                    self.old_temperature[i, j + 1] - self.old_temperature[i, j - 1]])
        normal = self.CalPhiGrad(i, j).normalized()
        h_fg = 2260000.0
        return tm.dot(normal, (self.k_water * T_grad - self.k_vapor * T_grad)) / h_fg



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
        # F_eta = ti.Vector([0.0, 0.0])

        F_a = ti.Vector([0.0, 0.0])
        if i!=0 and i!=self.nx-1 and j!=0 and j!=self.ny-1:
            F_a = self.rho[i, j] * self.vel[i, j] * self.vel_div[i, j]
        
        res = F_s + F_b + F_p + F_eta + F_a
        # if i == 124 and j==188:
            # print("F", F_s, F_eta, F_a, self.vel[i, j])
        
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
    def get_temp(self, i, j):
        res = 0.0
        if i < 0 or i >= self.nx or j < 0 or j >= self.ny:
            res = 0.0
        elif j == 0:
            res = 100.0
        else:
            res = self.old_temperature[i, j]

        return res

    
    @ti.kernel
    def Cal_derivative(self):
        for i, j in ti.ndrange((1, self.nx - 1), (1, self.ny - 1)):
            self.phi_grad[i, j] = self.CalPhiGrad(i, j)
            self.rho_grad[i, j] = self.CalRhoGrad(i, j)
            self.vel_grad[i, j] = self.CalVelGrad(i, j)
            self.phi_laplatian[i, j] = self.CalPhiLaplatian(i, j)
            self.vel_div[i, j] = self.CalVelDiv(i, j)


    @ti.kernel
    def update_temperature(self):
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
            
            laplacian = self.get_temp(i+1, j) + self.get_temp(i-1, j) + self.get_temp(i, j+1) + self.get_temp(i, j-1)\
                        - 4 * self.get_temp(i, j)
            Tgrad = 0.5 * ti.Vector([self.get_temp(i+1, j)- self.get_temp(i-1, j), self.get_temp(i, j+1) - self.get_temp(i, j-1)])
            self.new_temperature[i, j] = self.get_temp(i, j) + X * laplacian - tm.dot(Tgrad, self.vel[i, j])
            
        for i, j in ti.ndrange(self.nx, self.ny):
            self.old_temperature[i, j] = self.new_temperature[i, j]

    
    @ti.kernel
    def update_phi(self):
        for i, j in ti.ndrange((1, self.nx - 1), (1, self.ny - 1)):
            for k in ti.static(range(9)):
                i_s = i - self.e[k][0]
                j_s = j - self.e[k][1]
                heq = self.h_eq(i_s, j_s, k)

                # formula (22)
                self.h_new[i, j][k] = self.h_old[i_s, j_s][k] - (self.h_old[i_s, j_s][k] - heq) / self.tau_phi

                R = 0.0
                F = 0.0

                if self.is_INB(i, j):

                    # formula (24)
                    R = tm.dot(self.w[k] * self.e[k], 4 * self.phi[i_s, j_s] * (1 - self.phi[i_s, j_s]) / self.W * tm.normalize(self.phi_grad[i_s, j_s]))
                    # formula (25)
                    F = self.w[k] * (1 + 3 * (tm.dot(self.e[k], self.vel[i_s, j_s]) * (self.tau_phi - 0.5) / self.tau_phi))
                    F *= -self.CalM(i_s, j_s) / self.rho_l

                    self.h_new[i, j][k] += (2 * self.tau_phi - 1) / (2 * self.tau_phi) * R + F


    @ti.kernel
    def update_vel(self):
        for i, j in ti.ndrange((1, self.nx - 1), (1, self.ny - 1)):
            for k in ti.static(range(9)):
                i_s = i - self.e[k][0]
                j_s = j - self.e[k][1]
                geq = self.g_eq(i_s, j_s, k)

                P = 0.0
                if self.is_INB(i, j):
                    # formula(32)
                    P = self.w[k] * self.CalM(i_s, j_s) * (1 / self.rho_v - 1 / self.rho_l)

                # formula(33)
                G = 3 * self.w[k] * tm.dot(ti.Vector([self.e[k][0], self.e[k][1]]), self.CalF(i_s, j_s)) / self.rho[i_s, j_s]

                # formula(31)
                # tau = self.phi[i_s, j_s] * self.tau_phi + (1 - self.phi[i_s, j_s]) * self.tau_phi
                tau = self.tau_phi
                self.g_new[i, j][k] = self.g_old[i_s, j_s][k] - (self.g_old[i_s, j_s][k] - geq) / tau
                self.g_new[i, j][k] += (2 * tau - 1) / (2 * tau) * G + P


    @ti.kernel
    def apply_bc(self):  # impose boundary conditions
        # left and right
        for j in range(1, self.ny - 1):
            # left: dr = 0; ibc = 0; jbc = j; inb = 1; jnb = j
            self.apply_bc_core(1, 0, 0, j, 1, j)

            # right: dr = 2; ibc = nx-1; jbc = j; inb = nx-2; jnb = j
            self.apply_bc_core(1, 2, self.nx - 1, j, self.nx - 2, j)

        # top and bottom
        for i in range(self.nx):
            # top: dr = 1; ibc = i; jbc = ny-1; inb = i; jnb = ny-2
            self.apply_bc_core(1, 1, i, self.ny - 1, i, self.ny - 2)

            # bottom: dr = 3; ibc = i; jbc = 0; inb = i; jnb = 1
            self.apply_bc_core(1, 3, i, 0, i, 1)


    @ti.func
    def apply_bc_core(self, outer, dr, ibc, jbc, inb, jnb):
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
    def macro_vari(self):
        for i, j in ti.ndrange((1, self.nx - 1), (1, self.ny - 1)):
            self.phi[i, j] = 0.0 
            for k in ti.static(range(9)):
                self.phi[i, j] += self.h_new[i, j][k]
            
            self.h_old[i, j] = self.h_new[i, j]
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
            # temp = self.vel[i, j]
            for k in ti.static(range(9)):
                self.vel[i, j] += self.g_new[i, j][k] * ti.Vector([self.e[k][0], self.e[k][1]])
            self.g_old[i, j] = self.g_new[i, j]
            # if i==124 and j==188:
                # print("vel", temp, FF, self.rho[i, j])


    @ti.kernel
    def update_image(self):
        col_l = ti.Vector([0.0, 0.0, 0.8])
        col_v = ti.Vector([0.0, 0.8, 0.0])
        for i, j in self.old_temperature:
            self.image[i, j] = self.phi[i, j] * col_l + (1 - self.phi[i, j]) * col_v 
    

    @ti.kernel
    def Cal_r(self) -> int:
        res = 0
        for i in range(self.ny):
            cnt = 0
            for j in range(self.nx):
                if self.phi[i, j] < 0.5:
                    cnt += 1
            
            ti.atomic_max(res, cnt)
        
        return res
    

    @ti.kernel
    def check(self):
        check_vel = False
        check_phi = False
        check_h = False
        check_g = False
        for i, j in ti.ndrange(self.nx, self.ny):
            if tm.isnan(self.vel[i, j][0]) or tm.isnan(self.vel[i, j][1]):
                # print("vel nan", i, j)
                check_vel = True
            
            if tm.isnan(self.phi[i, j]):
                check_phi = True

            for k in range(9):
                if tm.isnan(self.h_old[i, j][k]):
                    check_h = True

                if tm.isnan(self.g_old[i, j][k]):
                    check_g = True
        
        # print(check_vel, check_phi, check_h, check_g)

    
    def solve(self):
        gui = ti.GUI(self.name, (self.nx, self.ny))
        self.init()
        frame = 0
        max_r = 0
        while not gui.get_event(ti.GUI.ESCAPE, ti.GUI.EXIT):
            # print(frame)
            sys.stdout.flush()
            frame += 1
            # self.update_temperature()
            self.Cal_derivative()
            self.update_phi()
            self.update_vel()
            self.macro_vari()
            self.apply_bc()
            self.check()

            # print(self.Cal_r())
            # print()
            cur_r = self.Cal_r()
            if cur_r > max_r:
                max_r = cur_r
                print(frame, cur_r)
                print(self.phi[150, 60])
            
            self.update_image()
            gui.set_image(self.image)
            gui.show()


if __name__ == '__main__':
    lbm = lbm_solver(
        name = "LBM",
        nx = 301,
        ny = 301,
        bc_type = [0, 0, 0, 0],
        bc_value = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
    )
    lbm.solve()