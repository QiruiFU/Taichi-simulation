# Fluid solver based on lattice boltzmann method using taichi language
# Author : Wang (hietwll@gmail.com)

import sys
import matplotlib
import numpy as np
from matplotlib import cm

import taichi as ti
import taichi.math as tm

ti.init(arch=ti.gpu)

@ti.data_oriented
class lbm_solver:
    def __init__(
        self,
        name, # name of the flow case
        nx,  # domain size
        ny,
        niu_water,
        niu_vapor,
        bc_type,  # [left,top,right,bottom] boundary conditions: 0 -> Dirichlet ; 1 -> Neumann
        bc_value,  # if bc_type = 0, we need to specify the velocity in bc_value
        ):
        self.name = name
        self.nx = nx  # by convention, dx = dy = dt = 1.0 (lattice units)
        self.ny = ny
        self.niu_water = niu_water
        self.niu_vapor = niu_vapor
        self.W = 6 * nx
        self.rho_l = 958 # density of water
        self.rho_v = 0.6 # density of vapor
        self.p_star = 0.1

        self.tau_phi = 0.5
        self.sigma = 1e-4

        self.tau_water = 3.0 * niu_water + 0.5
        self.tau_vapor = 3.0 * niu_vapor + 0.5
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
        
        self.w = ti.types.vector(9, float)(4, 1, 1, 1, 1, 1 / 4, 1 / 4, 1 / 4, 1 / 4) / 9.0
        self.e = ti.types.matrix(9, 2, int)([0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1])
        self.bc_type = ti.field(int, 4)
        self.bc_type.from_numpy(np.array(bc_type, dtype=np.int32))
        self.bc_value = ti.Vector.field(2, float, shape=4)
        self.bc_value.from_numpy(np.array(bc_value, dtype=np.float32))

        self.image = ti.Vector.field(3, dtype=ti.f32, shape=(nx, ny))  # RGB image


    @ti.func
    def phi_grad(self, i, j):
        return 0.5 * ti.Vector([self.phi[i + 1, j] - self.phi[i - 1, j], self.phi[i, j + 1] - self.phi[i, j - 1]])

    
    @ti.func
    def CalM(self, i, j):
        return 0.9
    

    @ti.func
    def CalF(self, i, j):
        return ti.Vector([666, 777])


    @ti.func
    def h_eq(self, i, j):
        eu = self.e @ self.vel[i, j]
        uv = tm.dot(self.vel[i, j], self.vel[i, j])
        return self.w * self.phi[i, j] * (1 + 3 * eu + 4.5 * eu * eu - 1.5 * uv)

    
    @ti.func
    def g_eq(self, i, j):
        eu = self.e @ self.vel[i, j]
        uv = tm.dot(self.vel[i, j], self.vel[i, j])
        return self.w * (self.p_star + 3 * eu + 4.5 * eu * eu - 1.5 * uv)


    @ti.kernel
    def init(self):
        self.vel.fill(0)
        self.rho.fill(0.5)
        for i, j in self.rho:
            self.h_old[i, j] = self.h_new[i, j] = self.h_eq(i, j)
            self.g_old[i, j] = self.g_new[i, j] = self.g_eq(i, j)

    
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
                R = tm.dot(self.w[k] * vec_e, 4 * self.phi[i_s, j_s] * (1 - self.phi[i_s, j_s]) / self.W * self.phi_grad(i_s, j_s).normalized())

                # formula (25)
                F = self.w[k] * (1 + 3 * (tm.dot(vec_e, self.vel[i_s, j_s]) * (self.tau_phi - 0.5) / self.tau_phi))
                F *= -self.CalM(i_s, j_s) / self.rho_l

                # formula (22)
                self.h_new[i, j][k] = self.h_old[i_s, j_s][k] - (self.h_old[i_s, j_s][k] - heq[k]) / self.tau_phi
                self.h_new[i, j][k] += (2 * self.tau_phi - 1) / (2 * self.tau_phi) * R
    

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
                tau = self.phi[i_s, j_s] * self.tau_water + (1 - self.phi[i_s, j_s]) * self.tau_vapor
                self.g_new[i, j][k] = self.g_old[i_s, j_s][k] - (self.g_old[i_s, j_s][k] - geq[k]) / tau
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
        self.g_old[ibc, jbc] = self.g_eq(ibc, jbc) - self.g_eq(inb, jnb) + self.g_old[inb, jnb]


    @ti.kernel
    def update_image(self):
        for i, j in self.old_temperature:
            # Normalize temperature to [0, 1]
            temp_normalized = (self.old_temperature[i, j] - 0.0) / (100.0 - 0.0)  # Adjust min/max as needed
            # Map to grayscale
            self.image[i, j] = ti.Vector([temp_normalized, 0, 0])

    def solve(self):
        gui = ti.GUI(self.name, (self.nx, self.ny))
        self.init()
        while not gui.get_event(ti.GUI.ESCAPE, ti.GUI.EXIT):
            for _ in range(10):
                self.update_temperature()
                self.update_phi()
                self.update_vel()
                self.apply_bc()
            
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
            name = "Karman Vortex Street",
            nx = 801,
            ny = 801,
            niu_vapor = 0.02,
            niu_water = 0.2,
            bc_type = [0, 0, 1, 0],
            bc_value = [[0.1, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
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