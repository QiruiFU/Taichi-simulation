import sys
import numpy as np

import taichi as ti
import taichi.math as tm

from matplotlib import cm

# ti.init(arch=ti.cpu, cpu_max_num_threads = 1, debug = True)
ti.init(arch=ti.cuda)

benchmark = 2

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
        self.W = 4
        if benchmark == 2:
            self.W = 4
        self.rho_l = 999.9 # density of water
        self.rho_v = 99.9 # density of vapor
        self.sigma = 1e-4
        self.tau_phi = 0.8
        self.tau = 0.8
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

        self.image = ti.Vector.field(3, dtype=ti.f32, shape=(4 * self.nx, self.ny))  # RGB image

        self.phi_grad = ti.Vector.field(2, float, shape=(self.nx, self.ny))
        self.rho_grad = ti.Vector.field(2, float, shape=(self.nx, self.ny))
        self.phi_laplatian = ti.field(float, shape=(self.nx, self.ny))
        self.vel_div = ti.field(float, shape=(self.nx, self.ny))
        self.vel_grad = ti.Matrix.field(2, 2, float, shape=(self.nx, self.ny))

        self.k_water = 0.68
        self.k_vapor = 0.0304
        self.Cp_water = 4.217
        self.Cp_vapor = 1.009

    
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
                res += self.w[k] * self.phi[nxt_x, nxt_y] * self.e[k]
            
        return 3.0 * res


    @ti.func
    def CalRhoGrad(self, i:int, j:int):
        res = ti.Vector([0.0, 0.0]) 
        for k in ti.static(range(9)):
            nxt_x, nxt_y = i + self.e[k][0], j + self.e[k][1]
            if nxt_x < self.nx and nxt_x >= 0 and nxt_y < self.ny and nxt_y >= 0 :
                res += self.w[k] * self.rho[nxt_x, nxt_y] * self.e[k]
            
        return 3.0 * res
    

    @ti.func
    def CalRhoGradReal(self, x:float, y:float):
        x1, y1 = int(x), int(y)
        x2, y2 = x1 + 1, y1 + 1
        if tm.fract(x) == 0.0:
            x2 = x1
        if tm.fract(y) == 0.0:
            y2 = y1

        tx, ty = x - float(x1), y - float(y1)

        rho11 = self.rho_grad[x1, y1]
        rho12 = self.rho_grad[x1, y2]
        rho21 = self.rho_grad[x2, y1]
        rho22 = self.rho_grad[x2, y2]

        grad_x_1 = (1.0 - tx) * rho11 + tx * rho12
        grad_x_2 = (1.0 - tx) * rho21 + tx * rho22
        grad = (1.0 - ty) * grad_x_1 + ty * grad_x_2
        return grad


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
    def CalMFlux(self, x, y, k):
        res = 0.0
        if self.is_INB(x, y, k):
            bx, by = x - self.e[k][0], y - self.e[k][1]
            nx, ny = x + self.e[k][0], y + self.e[k][1]
            T1 = 373.15
            T2 = self.GetTempPos(bx, by)
            T1_v = self.GetTempPos(x, y)

            u = (0.5 - self.phi[x, y]) / (self.phi[nx, ny] - self.phi[x, y])
            T0 = (2.0 * T1 + (u - 1.0) * T2) / (1 + u)
            e_dot_T = ((1.0 + 2.0 * u) * T0 - 4.0 * u * T1_v - (1.0 - 2.0 * u) * T2) * 0.5

            bx, by = x + 2 * self.e[k][0], y + 2 * self.e[k][1]
            curx, cury = x + self.e[k][0], y + self.e[k][1]
            nx, ny = x, y
            T1 = 373.15
            T2 = self.GetTempPos(bx, by)
            T1_v = self.GetTempPos(curx, cury)

            u = (0.5 - self.phi[curx, cury]) / (self.phi[nx, ny] - self.phi[curx, cury])
            T0 = (2.0 * T1 + (u - 1.0) * T2) / (1 + u)
            e_dot_T_inv = ((1.0 + 2.0 * u) * T0 - 4.0 * u * T1_v - (1.0 - 2.0 * u) * T2) * 0.5
            
            h_fg = 1.6
            if self.phi[x, y] < 0.5:
                res = (-self.k_vapor * e_dot_T + self.k_water * e_dot_T_inv) / (h_fg * self.e[k].norm())
            else:
                res = (self.k_water * e_dot_T - self.k_vapor * e_dot_T_inv) / (h_fg * self.e[k].norm())

            # print(res)
            if benchmark == 1:
                res = 0.5

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
                    p_rho_grad = self.CalRhoGradReal(inter_x, inter_y)
                    align = tm.dot(p_rho_grad.normalized(), self.e[k].normalized())
                    if ti.abs(align) > max_inner:
                        max_inner = align
                        max_idx = k
                        distribute_u = u
                    # if tm.isnan(align) or tm.isinf(align):
                    #     self.CalRhoGradReal(inter_x, inter_y, debug = 1)
        
        res = 0.0
        if max_idx != 1000:
            flux = self.CalMFlux(x, y, max_idx)
            res = flux * (1.0 - distribute_u)
        else:
            res = 0.0

        # if res != 0.0:
            # print(res)
        
        return res


    @ti.func
    def CalF(self, i, j):
        beta = 12.0 * self.sigma / self.W
        kappa = 1.5 * self.W
        phi = self.phi[i, j]
        miu = 2.0 * beta * phi * (1 - phi) * (1 - 2.0 * phi) - kappa * self.phi_laplatian[i, j]
        F_s = miu * self.phi_grad[i, j]

        F_b = (self.rho_l - self.rho[i, j]) * ti.Vector([0.0, 0.0])

        F_p = - self.p_star[i, j] * self.rho_grad[i, j] / 3.0

        niu = (self.tau - 0.5) / 3.0
        F_eta = ti.Vector([0.0, 0.0])

        for k in ti.static(range(9)):
            F_eta[0] += self.e[k][0] * self.e[k][1] * (self.g_new[i, j][k] - self.g_eq(i, j, k))
            F_eta[1] += self.e[k][0] * self.e[k][1] * (self.g_new[i, j][k] - self.g_eq(i, j, k))
        F_eta *= -3.0 * niu / self.tau
        F_eta[0] *= self.phi_grad[i, j][0]
        F_eta[1] *= self.phi_grad[i, j][1]

        F_a = ti.Vector([0.0, 0.0])
        if i>=0 and i<self.nx-1 and j>=0 and j<self.ny-1:
            F_a = self.rho[i, j] * self.vel[i, j] * self.vel_div[i, j]
        
        res = F_s + F_b + F_p + F_eta + F_a
        
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


        for i, j in self.phi:
            if benchmark == 1:
                mid_x = self.nx // 2
                mid_y = self.ny // 2
                r_init = self.nx / 10

                # if ti.abs(i - mid_x) < r_init and ti.abs(j - mid_y) < r_init:
                #     self.phi[i, j] = 0.0
                #     self.rho[i, j] = self.rho_v
                # else:
                #     self.phi[i, j] = 1.0
                #     self.rho[i, j] = self.rho_l

                if (i - mid_x)**2 + (j - mid_y)**2 < r_init**2:
                    self.phi[i, j] = 0.0
                    self.rho[i, j] = self.rho_v
                else:
                    self.phi[i, j] = 1.0
                    self.rho[i, j] = self.rho_l
            
            elif benchmark == 2:
                mid_x = self.nx // 2
                mid_y = self.ny // 2
                r_init = self.nx / 8

                # if ti.abs(i - mid_x) < r_init and ti.abs(j - mid_y) < r_init:
                #     self.phi[i, j] = 0.0
                #     self.rho[i, j] = self.rho_v
                # else:
                #     self.phi[i, j] = 1.0
                #     self.rho[i, j] = self.rho_l

                if (i - mid_x)**2 + (j - mid_y)**2 < r_init**2:
                    self.phi[i, j] = 1.0
                    self.rho[i, j] = self.rho_l
                else:
                    self.phi[i, j] = 0.0
                    self.rho[i, j] = self.rho_v


        for i, j in self.phi:
            self.p_star[i, j] = 0.0
            self.vel[i, j] = ti.Vector([0.0, 0.0])
            for k in ti.static(range(9)):
                self.h_old[i, j][k] = self.h_new[i, j][k] = self.h_eq(i, j, k)
                self.g_old[i, j][k] = self.g_new[i, j][k] = self.g_eq(i, j, k)
                self.p_star[i, j] += self.g_new[i, j][k]
            
            self.old_temperature[i, j] = self.new_temperature[i, j] = 373.15

        

    @ti.func
    def GetTempPos(self, x, y):
        res = 0.0
        inf_positive = 2000.0
        if benchmark == 3:
            if y <= 0:
                res = inf_positive
            elif y > self.ny - 1 or x < 0 or x > self.nx - 1:
                res = 373.15
            else:
                res = self.old_temperature[x, y]
        
        else:
            if y <= 0 or y >= self.ny - 1 or x <= 0 or x >= self.nx - 1:
                res = inf_positive
            else:
                res = self.old_temperature[x, y]
        
        return res
            
    
    @ti.func
    def GetTemp(self, x, y, k):
        res = 0.0
        nx, ny = x + self.e[k][0], y + self.e[k][1]
        if self.is_INB(x, y, k):
            bx, by = x - self.e[k][0], y - self.e[k][1]
            T1 = 373.15
            T2 = self.GetTempPos(bx, by)

            u = (0.5 - self.phi[x, y]) / (self.phi[nx, ny] - self.phi[x, y])
            res = (2.0 * T1 + (u - 1.0) * T2) / (1 + u)
        else:
            res = self.GetTempPos(nx, ny)
        # res = self.GetTempPos(nx, ny)
        
        return res

    
    @ti.kernel
    def CalDerivative(self):
        for i, j in ti.ndrange((1, self.nx - 1), (1, self.ny - 1)):
            self.phi_grad[i, j] = self.CalPhiGrad(i, j)
            self.rho_grad[i, j] = self.CalRhoGrad(i, j)
            self.phi_laplatian[i, j] = self.CalPhiLaplatian(i, j)
            self.vel_div[i, j] = self.CalVelDiv(i, j)
            self.vel_grad[i, j] = self.CalVelGrad(i, j)


    @ti.kernel
    def UpdateTemp(self):
        for i, j in ti.ndrange(self.nx, self.ny):
            # X_water = self.k_water / (self.rho[i, j] * self.Cp_water)
            # X_vapor = self.k_vapor / (self.rho[i, j] * self.Cp_vapor)
            X_water = 0.10
            X_vapor = 0.02
            X = 0.0
            if self.phi[i, j] < 0.5 : 
                X = X_vapor
            elif self.phi[i, j] > 0.5 :
                X = X_water
            else :
                X = 0.5 * (X_vapor + X_water)

            laplacian = 0.0
            Tgrad = ti.Vector([0.0, 0.0])

            inv = ti.Vector([0, 3, 4, 1, 2, 7, 8, 5, 6])

            for k in range(9):
                Tgrad += 1.5 * self.w[k] * (self.GetTemp(i, j, k) - self.GetTemp(i, j, inv[k])) * self.e[k]
                laplacian += 3.0 * self.w[k] * (self.GetTemp(i, j, k) + self.GetTemp(i, j, inv[k]) - 2.0 * self.GetTempPos(i, j))

            self.new_temperature[i, j] = self.old_temperature[i, j] + X * laplacian - tm.dot(Tgrad, self.vel[i, j])
            
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
                normal_grad = tm.normalize(self.phi_grad[i, j])
                if (not tm.isinf(normal_grad[0])) and (not tm.isinf(normal_grad[1])): 
                    # formula (24)
                    R = tm.dot(self.w[k] * self.e[k], 4 * self.phi[i, j] * (1 - self.phi[i, j]) / self.W * tm.normalize(self.phi_grad[i, j]))
                    if tm.isnan(R):
                        R = 0.0
                        # print(tm.normalize(self.phi_grad[i, j]))

                # formula (25)
                F = self.w[k] * (1 + 3 * (tm.dot(self.e[k], self.vel[i, j]) * (self.tau_phi - 0.5) / self.tau_phi))
                F *= -self.CalMRate(i, j) / self.rho_l

                self.h_old[i, j][k] += (2 * self.tau_phi - 1) / (2 * self.tau_phi) * R + F
                
                # ---- update vel ----
                geq = self.g_eq(i, j, k)
                self.g_old[i, j][k] = self.g_old[i, j][k] - (self.g_old[i, j][k] - geq) / self.tau

                # formula(32)
                P = self.w[k] * self.CalMRate(i, j) * (1 / self.rho_v - 1 / self.rho_l)

                # formula(33)
                G = 3 * self.w[k] * tm.dot(self.e[k], self.CalF(i, j)) / self.rho[i, j]

                # formula(31)
                self.g_old[i, j][k] += (2 * self.tau - 1) / (2 * self.tau) * G + P



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

            self.rho[i, j] = self.phi[i, j] * self.rho_l + (1 - self.phi[i, j]) * self.rho_v

        for i, j in ti.ndrange((1, self.nx - 1), (1, self.ny - 1)):
            self.p_star[i, j] = 0.0
            for k in ti.static(range(9)):
                self.p_star[i, j] += self.g_new[i, j][k]

        for i, j in ti.ndrange((1, self.nx - 1), (1, self.ny - 1)):
            self.phi_grad[i, j] = self.CalPhiGrad(i, j)
            self.rho_grad[i, j] = self.CalRhoGrad(i, j)
            self.phi_laplatian[i, j] = self.CalPhiLaplatian(i, j)
            self.vel_div[i, j] = self.CalVelDiv(i, j)
            self.vel_grad[i, j] = self.CalVelGrad(i, j)
            
        for i, j in ti.ndrange((1, self.nx - 1), (1, self.ny - 1)):
            FF = self.CalF(i, j)
            self.vel[i, j] = FF / (2.0 * self.rho[i, j])
            for k in ti.static(range(9)):
                self.vel[i, j] += self.g_new[i, j][k] * ti.Vector([self.e[k][0], self.e[k][1]])
            self.g_old[i, j] = self.g_new[i, j]


    @ti.kernel
    def UpdateImage(self):
        for i, j in self.old_temperature:
            col_l = ti.Vector([0.0, 0.0, 0.8])
            col_v = ti.Vector([0.5, 0.5, 0.5])
            self.image[i, j] = self.phi[i, j] * col_l + (1 - self.phi[i, j]) * col_v 
            
            r_init = self.nx // 8
            if i == 150 and j == 150 + r_init:
                self.image[i, j] = ti.Vector([1.0, 1.0, 1.0])
            if i == 150 and j == 150:
                self.image[i, j] = ti.Vector([1.0, 1.0, 1.0])
        
            color_strength = self.vel[i, j].norm() * 100.0
            self.image[i + self.nx, j] = ti.Vector([color_strength, color_strength, 0])

            temp_color = (self.old_temperature[i, j] - 300) / 500.0
            self.image[i + 2 * self.nx, j] = ti.Vector([temp_color, 0.0, 0.0])



    @ti.kernel
    def CalR(self) -> int:
        res = -1
        for i in range(self.ny):
            cnt = 0
            for j in range(self.nx):
                if self.phi[i, j] > 0.5:
                    cnt += 1
            
            ti.atomic_max(res, cnt)
        
        return res


    def FindCenter(self):
        center_x, center_y = 0, 0

        res = 0
        for i in range(self.ny):
            cnt = 0
            for j in range(self.nx):
                if self.phi[i, j] < 0.5:
                    cnt += 1
            if cnt > res:
                res = cnt
                center_x = i
            
        res = 0
        for i in range(self.ny):
            cnt = 0
            for j in range(self.nx):
                if self.phi[j, i] < 0.5:
                    cnt += 1
            if cnt > res:
                res = cnt
                center_y = i
        
        return center_x, center_y 
    

    @ti.kernel
    def Check(self):
        check_phi = True
        check_vel = True
        check_h = True
        check_g = True
        check_sym = True
        for i, j in self.phi:
            if tm.isnan(self.phi[i, j]):
                check_phi = False
            if tm.isnan(self.vel[i, j][0]) or tm.isnan(self.vel[i, j][1]):
                check_vel = False
            for k in ti.static(range(9)):
                if tm.isnan(self.h_old[i, j][k]):
                    check_h = False
                if tm.isnan(self.g_old[i, j][k]):
                    check_g = False

        
        for i, j in self.phi:
            test = False
            for k in range(9):
                if self.is_INB(i, j, k):
                    test = True
            
            if test:
                print(i, j, self.phi_grad[i, j], self.rho_grad[i, j])


    @ti.kernel
    def SumPhi(self) -> float:
        res = 0.0
        for i, j in self.phi:
            res += self.phi[i, j]
        
        return res
    

    def solve(self):
        gui = ti.GUI(self.name, (4 * self.nx, self.ny))
        self.init()
        frame = 0
        init_r = self.nx / 8

        f1 = open("test2-r-3.txt", "w")
        while not gui.get_event(ti.GUI.ESCAPE, ti.GUI.EXIT):
            sys.stdout.flush()
            for substep in range(100):
                frame += 1
                if benchmark != 1:
                    self.UpdateTemp()
                self.CalDerivative()
                self.Collision()
                self.Advection()
                self.MacroVari()
                self.ApplyBc()

                cur_r = self.CalR() / 2
                print(frame, cur_r / init_r, file=f1)

            # self.Check()
            # print(self.SumPhi())

            if cur_r < init_r / 2:
                # cx, cy = self.FindCenter()
                # for i in range(cy, 301):
                #     print(i, self.vel[cx, i][0], file=f2)
                #     print(i, self.phi[cx, i], file=f3)
                
                break
            
            self.UpdateImage()
            gui.set_image(self.image)
            gui.show()
        
        f1.close()


if __name__ == '__main__':
    lbm = lbm_solver(
        name = "LBM",
        bc_type = [0, 0, 0, 0],
        bc_value = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
    )
    lbm.solve()