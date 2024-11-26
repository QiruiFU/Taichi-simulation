# Fluid solver based on lattice boltzmann method using taichi language
# Author : Wang (hietwll@gmail.com)

import sys
import matplotlib
import numpy as np
from matplotlib import cm

import taichi as ti
import taichi.math as tm
import time

ti.init(arch=ti.cpu, cpu_max_num_threads = 1, debug = True)

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
        self.W = 3
        self.rho_l = 10.0 # density of water
        self.rho_v = 1.0 # density of vapor
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

        self.w = ti.types.vector(9, float)(4, 1, 1, 1, 1, 1 / 4, 1 / 4, 1 / 4, 1 / 4) / 9.0
        self.e = ti.types.matrix(9, 2, int)([0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1])
        self.bc_type = ti.field(int, 4)
        self.bc_type.from_numpy(np.array(bc_type, dtype=np.int32))
        self.bc_value = ti.Vector.field(2, float, shape=4)
        self.bc_value.from_numpy(np.array(bc_value, dtype=np.float32))

        self.image = ti.Vector.field(3, dtype=ti.f32, shape=(nx, ny))  # RGB image

        self.phi_grad = ti.Vector.field(2, float, shape=(nx, ny))
        self.rho_grad = ti.Vector.field(2, float, shape=(nx, ny))
        self.vel_grad = ti.Matrix.field(2, 2, float, shape=(nx, ny))
        self.phi_laplatian = ti.field(float, shape=(nx, ny))


    @ti.func
    def CalPhiGrad(self, i, j):
        res = ti.Vector([0.0, 0.0]) 
        for k in ti.static(range(9)):
            nxt_x, nxt_y = i + self.e[k, 0], j + self.e[k, 1]
            if nxt_x < self.nx and nxt_x >= 0 and nxt_y < self.ny and nxt_y >= 0 :
                res += self.w[k] * self.phi[nxt_x, nxt_y] * ti.Vector([self.e[k, 0], self.e[k, 1]])
            
        return 3.0 * res


    @ti.func
    def CalRhoGrad(self, i, j):
        res = ti.Vector([0.0, 0.0]) 
        for k in ti.static(range(9)):
            nxt_x, nxt_y = i + self.e[k, 0], j + self.e[k, 1]
            if nxt_x < self.nx and nxt_x >= 0 and nxt_y < self.ny and nxt_y >= 0 :
                res += self.w[k] * self.rho[nxt_x, nxt_y] * ti.Vector([self.e[k, 0], self.e[k, 1]])
            
        return 3.0 * res


    @ti.func
    def CalVelGrad(self, i, j):
        res = ti.Matrix([[0.0, 0.0], [0.0, 0.0]]) 
        for k in ti.static(range(9)):
            nxt_x, nxt_y = i + self.e[k, 0], j + self.e[k, 1]
            if nxt_x < self.nx and nxt_x >= 0 and nxt_y < self.ny and nxt_y >= 0 :
                ele00 = self.vel[nxt_x, nxt_y][0] * self.e[k, 0]
                ele01 = self.vel[nxt_x, nxt_y][0] * self.e[k, 1]
                ele10 = self.vel[nxt_x, nxt_y][1] * self.e[k, 0]
                ele11 = self.vel[nxt_x, nxt_y][1] * self.e[k, 1]
                res += self.w[k] * ti.Matrix([[ele00, ele01], [ele10, ele11]])
            
        return 3.0 * res


    @ti.func
    def CalPhiLaplatian(self, i, j):
        res = 0.0
        for k in ti.static(range(9)):
            nxt_x, nxt_y = i + self.e[k, 0], j + self.e[k, 1]
            if nxt_x < self.nx and nxt_x >= 0 and nxt_y < self.ny and nxt_y >= 0 :
                res += self.w[k] * (self.phi[nxt_x, nxt_y] - self.phi[i, j])
            
        return 6.0 * res
    

    @ti.func
    def CalM(self, i, j):
        return 5e-5
        T_grad = 0.5 * ti.Vector([self.old_temperature[i + 1, j] - self.old_temperature[i - 1, j],\
                                    self.old_temperature[i, j + 1] - self.old_temperature[i, j - 1]])
        normal = self.CalPhiGrad(i, j).normalized()
        h_fg = 2260000.0
        return tm.dot(normal, (self.k_water * T_grad - self.k_vapor * T_grad)) / h_fg



    @ti.func
    def CalF(self, i, j):
        beta = 12.0 * self.sigma / self.W
        kappa = 1.5 * self.W
        phi = self.phi[i, j]
        miu = 2.0 * beta * phi * (1 - phi) * (1 - 2.0 * phi) - kappa * self.phi_laplatian[i, j]
        F_s = miu * self.phi_grad[i, j]

        # F_b = (self.rho_l - self.rho[i, j]) * ti.Vector([0.0, -10.0])
        F_b = ti.Vector([0.0, 0.0])

        F_p = - self.p_star[i, j] * self.rho_grad[i, j] / 3.0

        niu = (self.tau_phi - 0.5) / 3.0
        u_grad = self.vel_grad[i, j]
        F_eta = niu * (u_grad + u_grad.transpose()) @ self.rho_grad[i, j]

        F_a = ti.Vector([0.0, 0.0])
        if i!=0 and i!=self.nx-1 and j!=0 and j!=self.ny-1:
            F_a = self.rho[i, j] * self.vel[i, j] * (self.vel[i + 1, j][0] + self.vel[i, j + 1][1] - self.vel[i - 1, j][0] - self.vel[i, j - 1][1]) / 2.0

        return F_s + F_b + F_p + F_eta + F_a


    @ti.func
    def h_eq(self, i, j):
        eu = self.e @ self.vel[i, j]
        uv = tm.dot(self.vel[i, j], self.vel[i, j])
        return self.w * self.phi[i, j] * (1 + 3 * eu + 4.5 * eu * eu - 1.5 * uv)

    
    @ti.func
    def g_eq(self, i, j):
        eu = self.e @ self.vel[i, j]
        uv = tm.dot(self.vel[i, j], self.vel[i, j])
        return self.w * (self.p_star[i, j] + 3 * eu + 4.5 * eu * eu - 1.5 * uv)


    @ti.kernel
    def init(self):
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
            
            self.h_old[i, j] = self.h_new[i, j] = self.h_eq(i, j)
            self.g_old[i, j] = self.g_new[i, j] = self.g_eq(i, j)

        # self.vel.fill(0.0)
        # for i, j in self.rho:
        #     if j > 100 :
        #         self.rho[i, j] = self.rho_v
        #         self.phi[i, j] = 0
        #     else :
        #         self.rho[i, j] = self.rho_l
        #         self.phi[i, j] = 1

        #     self.h_old[i, j] = self.h_new[i, j] = self.h_eq(i, j)
        #     self.g_old[i, j] = self.g_new[i, j] = self.g_eq(i, j)

    
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
                vec_e = ti.Vector([self.e[k, 0], self.e[k, 1]])
                i_s = i - self.e[k, 0]
                j_s = j - self.e[k, 1]
                heq = self.h_eq(i_s, j_s)

                # formula (24)
                vvv = self.phi_grad[i_s, j_s].normalized()
                R = tm.dot(self.w[k] * vec_e, 4 * self.phi[i_s, j_s] * (1 - self.phi[i_s, j_s]) / self.W * vvv)

                # formula (25)
                F = self.w[k] * (1 + 3 * (tm.dot(vec_e, self.vel[i_s, j_s]) * (self.tau_phi - 0.5) / self.tau_phi))
                F *= -self.CalM(i_s, j_s) / self.rho_l

                # formula (22)
                self.h_new[i, j][k] = self.h_old[i_s, j_s][k] - (self.h_old[i_s, j_s][k] - heq[k]) / self.tau_phi
                self.h_new[i, j][k] += (2 * self.tau_phi - 1) / (2 * self.tau_phi) * R
    
        # macro varialbes

        for i, j in ti.ndrange((1, self.nx - 1), (1, self.ny - 1)):
            self.phi[i, j] = 0.0 
            for k in ti.static(range(9)):
                self.phi[i, j] += self.h_new[i, j][k]
            
            self.h_old[i, j] = self.h_new[i, j]


    @ti.kernel
    def update_vel(self):
        for i, j in ti.ndrange((1, self.nx - 1), (1, self.ny - 1)):
            for k in ti.static(range(9)):
                i_s = i - self.e[k, 0]
                j_s = j - self.e[k, 1]
                geq = self.g_eq(i_s, j_s)

                # formula(32)
                P = self.w[k] * self.CalM(i_s, j_s) * (1 / self.rho_v - 1 / self.rho_l)

                # formula(33)
                G = 3 * self.w[k] * tm.dot(ti.Vector([self.e[k, 0], self.e[k, 1]]), self.CalF(i_s, j_s)) / self.rho[i_s, j_s]

                # formula(31)
                tau = self.phi[i_s, j_s] * self.tau_phi + (1 - self.phi[i_s, j_s]) * self.tau_phi
                self.g_new[i, j][k] = self.g_old[i_s, j_s][k] - (self.g_old[i_s, j_s][k] - geq[k]) / tau
                self.g_new[i, j][k] += (2 * tau - 1) / (2 * tau) * G + P

        # macro variables

        for i, j in ti.ndrange((1, self.nx - 1), (1, self.ny - 1)):
            self.p_star[i, j] = 0.0
            rho = self.phi[i, j] * self.rho_l + (1 - self.phi[i, j]) * self.rho_v
            self.vel[i, j] = self.CalF(i, j) / (2.0 * rho)
            for k in ti.static(range(9)):
                self.p_star[i, j] += self.g_new[i, j][k]
                self.vel[i, j] += self.g_new[i, j][k] * ti.Vector([self.e[k, 0], self.e[k, 1]])
            
            self.g_old[i, j] = self.g_new[i, j]


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
        self.g_old[ibc, jbc] = self.g_eq(ibc, jbc) - self.g_eq(inb, jnb) + self.g_old[inb, jnb]


    @ti.kernel
    def update_image(self):
        col_l = ti.Vector([0.0, 0.0, 0.8])
        col_v = ti.Vector([0.1, 0.1, 0.1])
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
            
            res = max(res, cnt)
        
        return res

    
    def solve(self):
        gui = ti.GUI(self.name, (self.nx, self.ny))
        self.init()
        print(self.phi)
        while not gui.get_event(ti.GUI.ESCAPE, ti.GUI.EXIT):
            # self.update_temperature()
            self.Cal_derivative()
            self.update_phi()
            self.update_vel()
            self.apply_bc()

            print(self.Cal_r())
            
            self.update_image()
            gui.set_image(self.image)
            gui.show()

            ##  code fragment displaying vorticity is contributed by woclass
            # vel = self.vel.to_numpy()
            # ugrad = np.gradient(vel[:, :, 0])
            # vgrad = np.gradient(vel[:, :, 1])
            # vor = ugrad[1] - vgrad[0]
            # vel_mag = (vel[:, :, 0] ** 2.0 + vel[:, :, 1] ** 2.0) ** 0.5
            # ## color map
            # colors = [
            #     (1, 1, 0),
            #     (0.953, 0.490, 0.016),
            #     (0, 0, 0),
            #     (0.176, 0.976, 0.529),
            #     (0, 1, 1),
            # ]
            # my_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("my_cmap", colors)
            # vor_img = cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=-0.02, vmax=0.02), cmap=my_cmap).to_rgba(vor)
            # vel_img = cm.plasma(vel_mag / 0.15)
            # img = np.concatenate((vor_img, vel_img), axis=1)
            # gui.set_image(img)
            # gui.show()


if __name__ == '__main__':
    flow_case = 0 if len(sys.argv) < 2 else int(sys.argv[1])
    if (flow_case == 0):  # von Karman vortex street: Re = U*D/niu = 200
        lbm = lbm_solver(
            name = "LBM",
            nx = 301,
            ny = 301,
            bc_type = [0, 0, 0, 1],
            bc_value = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        )
        lbm.solve()
    elif (flow_case == 1):  # lid-driven cavity flow: Re = U*L/niu = 1000
        lbm = lbm_solver(
            "Lid-driven Cavity Flow",
            256,
            256,
            0.0255,
            [0, 0, 0, 0],
            [[0.0, 0.0], [0.1, 0.0], [0.0, 0.0], [0.0, 0.0]])
        lbm.solve()
    else:
        print("Invalid flow case ! Please choose from 0 (Karman Vortex Street) and 1 (Lid-driven Cavity Flow).")