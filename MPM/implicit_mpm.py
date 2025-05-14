import numpy as np
import taichi as ti
import taichi.math as tm
from Newton import Newton
from LBFGS import LBFGS
from Gradient import GradientDesent

ti.init(arch=ti.gpu, debug=True)

visualization = 0
implicit = True
benchmark = 1

n_particle = 10 ** 3
nx = ny = 18
nz = 18 

dx = 1 / 16
lenx, leny, lenz = nx * dx, ny * dx, nz * dx

p_vol_0 = dx**3 / (2)**3
p_rho = 1000
p_mass = p_rho * p_vol_0

frame = 60
substep = 1
dt = 1 / (frame * substep)

E_0 = 1.4e5
niu = 0.2
zeta = 10
miu_0, lambda_0 = E_0 / (2 * (1 + niu)), E_0 * niu / ((1 + niu) * (1 - 2 * niu))  # Lame parameters

vel_particle = ti.Vector.field(3, float, shape = n_particle)
pos_particle = ti.Vector.field(3, float, shape = n_particle)
C_particle = ti.Matrix.field(3, 3, float, shape = n_particle)
F_particle = ti.Matrix.field(3, 3, float, shape = n_particle)

grid_vel = ti.Vector.field(3, float, shape = (nx, ny, nz))
grid_vstar = ti.Vector.field(3, float, shape = (nx, ny, nz))
grid_mass = ti.field(float, shape = (nx, ny, nz))


bd_x_idx = ti.field(int, shape = 6)
bd_y_idx = ti.field(int, shape = 6)
bd_z_idx = ti.field(int, shape = 6)

bd_x_idx.from_numpy(np.array([0, 1, 2, nx-3, nx-2, nx-1]))
bd_y_idx.from_numpy(np.array([0, 1, 2, ny-3, ny-2, ny-1]))
bd_z_idx.from_numpy(np.array([0, 1, 2, nz-3, nz-2, nz-1]))


pos_vertex = ti.Vector.field(3, float, shape=4) # floor
pos_vertex.from_numpy(np.array([[2, 2, 0], [2, -2, -0], [-2, 2, 0], [-1, -1, 0]]))
indice = ti.field(int, shape = 6)
indice.from_numpy(np.array([0, 1, 2, 2, 1, 3]))

total_energy = ti.field(float, shape = ())

@ti.func
def N(x):
    x = ti.abs(x)
    res = 0.0
    if x < 0.5 :
        res = 0.75 - x * x
    elif x < 1.5 :
        res = 0.5 * (1.5 - x) * (1.5 - x)
    else:
        res = 0.0

    return res

@ti.func
def dN(x):
    res = 0.0
    if x < -1.5 or x > 1.5:
        res = 0.0
    elif x < -0.5:
        res = x + 1.5
    elif x < 0.5:
        res = -2.0 * x
    else:  # x < 1.5
        res = x - 1.5
    return res


@ti.func
def GridIdx(x, y, z):
    return x * ny * nz + y * nz + z


@ti.func
def InDomain(x):
    in_x = x[0] >= 0.0 and x[0] < lenx
    in_y = x[1] >= 0.0 and x[1] < leny
    in_z = x[2] >= 0.0 and x[2] < lenz
    return in_x and in_y and in_z


@ti.func
def AtBoundary(x:int, y:int, z:int):
    res = False
    if x == nx-1 or x == 0:
        res = True
    if y == ny-1 or y == 0:
        res = True
    if z == nz-1 or z == 0:
        res = True

    return res


@ti.func
def CalVelGradParticle(vel, p):
    vel_grad = ti.Matrix.zero(float, 3, 3)
    center = int(pos_particle[p] / dx) # index of center grid
    for i, j, k in ti.ndrange((-1, 2), (-1, 2), (-1, 2)):
        # compute vel_grad of particle
        grid_pos = float(center + ti.Vector([0.5, 0.5, 0.5]) + ti.Vector([i, j, k])) * dx
        if InDomain(grid_pos):
            dpos = grid_pos - pos_particle[p]

            NX = N(dpos[0] / dx)
            NY = N(dpos[1] / dx)
            NZ = N(dpos[2] / dx)
            dNX = dN(dpos[0] / dx)
            dNY = dN(dpos[1] / dx)
            dNZ = dN(dpos[2] / dx)

            dweight = ti.Vector.zero(float, 3)
            dweight[0] = dNX * NY * NZ
            dweight[1] = NX * dNY * NZ
            dweight[2] = NX * NY * dNZ

            idx = GridIdx(center[0] + i, center[1] + j, center[2] + k)
            v_vel = ti.Vector([vel[idx*3], vel[idx*3+1], vel[idx*3+2]])
            vel_grad += v_vel.outer_product(dweight)

    return vel_grad


@ti.kernel
def Initiate():

    for i in pos_particle:
        cube_len = 10

        if benchmark == 1:
            idx = (i % (cube_len * cube_len)) // cube_len
            idy = (i % (cube_len * cube_len)) % cube_len
            idz = i // (cube_len * cube_len)

            base = ti.Vector([0.45, 0.45, 0.15])
            # rand_dpos = ti.Vector([ti.random(float)-0.5, ti.random(float)-0.5, ti.random(float)-0.5]) * 0.05
            rand_dpos = ti.Vector([0, 0, 0])
            pos_particle[i] = (base + 0.5 * dx * ti.Vector([idx, idy, idz]) + rand_dpos)
            vel_particle[i] = ti.Vector.zero(float, 3)
            F_particle[i] = ti.Matrix.identity(float, 3)
            C_particle[i] = ti.Matrix.zero(float, 3, 3)

            pos_particle[i] = pos_particle[i] - base - 5 * ti.Vector([dx, dx, dx])
            rot_x = ti.Matrix([[1, 0, 0], [0, 0.5, tm.sqrt(3)/2], [0, -tm.sqrt(3)/2, 0.5]]) 
            rot_y = ti.Matrix([[tm.sqrt(2)/2, 0, tm.sqrt(2)/2], [0, 1, 0], [-tm.sqrt(2)/2, 0, tm.sqrt(2)/2]]) 
            pos_particle[i] = rot_y @ rot_x @ pos_particle[i]
            pos_particle[i] = pos_particle[i] + base + 5 * ti.Vector([dx, dx, dx])
        # elif benchmark == 2:
        #     if i < n_particle // 2:
        #         idx = (i % (cube_len * cube_len)) // cube_len
        #         idy = (i % (cube_len * cube_len)) % cube_len
        #         idz = i // (cube_len * cube_len)

        #         base = ti.Vector([1.0, 1.0, 0.75])
        #         # rand_dpos = ti.Vector([ti.random(float)-0.5, ti.random(float)-0.5, ti.random(float)-0.5]) * 0.05
        #         rand_dpos = ti.Vector([0, 0, 0])
        #         pos_particle[i] = (base + 0.5 * dx * ti.Vector([idx, idy, idz]) + rand_dpos)
        #         vel_particle[i] = ti.Vector([4, 4, 1])
        #         F_particle[i] = ti.Matrix.identity(float, 3)
        #         C_particle[i] = ti.Matrix.zero(float, 3, 3)

        #         pos_particle[i] = pos_particle[i] - base - 5 * ti.Vector([dx, dx, dx])
        #         rot_x = ti.Matrix([[1, 0, 0], [0, 0.5, tm.sqrt(3)/2], [0, -tm.sqrt(3)/2, 0.5]]) 
        #         rot_y = ti.Matrix([[tm.sqrt(3)/2, 0, 0.5], [0, 1, 0], [-0.5, 0, tm.sqrt(3)/2]]) 
        #         pos_particle[i] = rot_y @ rot_x @ pos_particle[i]
        #         pos_particle[i] = pos_particle[i] + base + 5 * ti.Vector([dx, dx, dx])
        #     else:
        #         j = i - n_particle // 2
        #         idx = (j % (cube_len * cube_len)) // cube_len
        #         idy = (j % (cube_len * cube_len)) % cube_len
        #         idz = j // (cube_len * cube_len)

        #         base = ti.Vector([2.2, 2.2, 0.8])
        #         # rand_dpos = ti.Vector([ti.random(float)-0.5, ti.random(float)-0.5, ti.random(float)-0.5]) * 0.05
        #         rand_dpos = ti.Vector([0, 0, 0])
        #         pos_particle[i] = (base + 0.5 * dx * ti.Vector([idx, idy, idz]) + rand_dpos)
        #         vel_particle[i] = ti.Vector([-4, -4, 1])
        #         F_particle[i] = ti.Matrix.identity(float, 3)
        #         C_particle[i] = ti.Matrix.zero(float, 3, 3)

        #         pos_particle[i] = pos_particle[i] - base - 5 * ti.Vector([dx, dx, dx])
        #         rot_x = ti.Matrix([[1, 0, 0], [0, 0.5, tm.sqrt(3)/2], [0, -tm.sqrt(3)/2, 0.5]]) 
        #         cos45 = tm.sqrt(2) / 2
        #         rot_y = ti.Matrix([[cos45, 0, cos45], [0, 1, 0], [-cos45, 0, cos45]]) 
        #         pos_particle[i] = rot_y @ rot_x @ pos_particle[i]
        #         pos_particle[i] = pos_particle[i] + base + 5 * ti.Vector([dx, dx, dx])


@ti.kernel
def ComputeEnergy(v: ti.template()) -> float:
    # for idx, j, k in ti.ndrange(6, ny, nz):
    #     i = bd_x_idx[idx]
    #     idx_grid = GridIdx(i, j, k)
    #     v[idx_grid * 3 + 0] = 0
    #     v[idx_grid * 3 + 1] = 0
    #     v[idx_grid * 3 + 2] = 0

    # for i, idx, k in ti.ndrange(nx, 6, nz):
    #     j = bd_y_idx[idx]
    #     idx_grid = GridIdx(i, j, k)
    #     v[idx_grid * 3 + 0] = 0
    #     v[idx_grid * 3 + 1] = 0
    #     v[idx_grid * 3 + 2] = 0

    # for i, j, idx in ti.ndrange(nx, ny, 6):
    #     k = bd_z_idx[idx]
    #     idx_grid = GridIdx(i, j, k)
    #     v[idx_grid * 3 + 0] = 0
    #     v[idx_grid * 3 + 1] = 0
    #     v[idx_grid * 3 + 2] = 0

    for x, y, z in grid_mass:
        if grid_mass[x, y, z] == 0.0:
            idx = GridIdx(x, y, z)
            v[idx * 3 + 0] = 0.0
            v[idx * 3 + 1] = 0.0
            v[idx * 3 + 2] = 0.0

    total_energy[None] = 0.0

    for p in vel_particle:
        grad_vel = CalVelGradParticle(v, p)

        # compute potential energy
        new_F = (ti.Matrix.identity(float, 3) + grad_vel * dt) @ F_particle[p]
        new_J = new_F.determinant()

        term1 = 0.5 * miu_0 * ((new_F.transpose() @ new_F).trace() - 3)
        term2 = - miu_0 * tm.log(new_J) 
        term3 = 0.5 * lambda_0 * tm.log(new_J) * tm.log(new_J)
        total_energy[None] += (term1 + term2 + term3) * p_vol_0
        # old_val = ti.atomic_add(total_energy, (term1+term2+term3) * p_vol_0)
        # assert not tm.isnan(total_energy), f"{term1}, {term2}, {term3}, {total_energy_before}, {p_vol_0}"


    for x, y, z in grid_vel:
        idx = GridIdx(x, y, z)
        v_vel = ti.Vector([v[idx*3], v[idx*3+1], v[idx*3+2]])
        del_v = v_vel - grid_vel[x, y, z]
        total_energy[None] += 0.5 * grid_mass[x, y, z] * (tm.dot(del_v, del_v))
        # assert not tm.isnan(total_energy), f"{v_vel[0]}"
    
    return total_energy[None]


@ti.kernel
def ComputeGrad(v: ti.template(), grad: ti.template()):
    grad.fill(0.0)

    # for idx, j, k in ti.ndrange(6, ny, nz):
    #     i = bd_x_idx[idx]
    #     idx_grid = GridIdx(i, j, k)
    #     v[idx_grid * 3 + 0] = 0
    #     v[idx_grid * 3 + 1] = 0
    #     v[idx_grid * 3 + 2] = 0

    # for i, idx, k in ti.ndrange(nx, 6, nz):
    #     j = bd_y_idx[idx]
    #     idx_grid = GridIdx(i, j, k)
    #     v[idx_grid * 3 + 0] = 0
    #     v[idx_grid * 3 + 1] = 0
    #     v[idx_grid * 3 + 2] = 0

    # for i, j, idx in ti.ndrange(nx, ny, 6):
    #     k = bd_z_idx[idx]
    #     idx_grid = GridIdx(i, j, k)
    #     v[idx_grid * 3 + 0] = 0
    #     v[idx_grid * 3 + 1] = 0
    #     v[idx_grid * 3 + 2] = 0
    for x, y, z in grid_mass:
        if grid_mass[x, y, z] == 0.0:
            idx = GridIdx(x, y, z)
            v[idx * 3 + 0] = 0.0
            v[idx * 3 + 1] = 0.0
            v[idx * 3 + 2] = 0.0

    for p in vel_particle:
        grad_vel = CalVelGradParticle(v, p)
        F = F_particle[p]
        new_F = (ti.Matrix.identity(float, 3) + grad_vel * dt) @ F_particle[p]
        new_J = new_F.determinant()

        stress = miu_0 * (new_F - new_F.inverse().transpose()) + lambda_0 * tm.log(new_J) * new_F.inverse().transpose()
        dE_dgvel = stress @ F.transpose() * p_vol_0

        center = int(pos_particle[p] / dx) # index of center grid
        for i, j, k in ti.ndrange((-1, 2), (-1, 2), (-1, 2)):
            grid_pos = float(center + ti.Vector([0.5, 0.5, 0.5]) + ti.Vector([i, j, k])) * dx
            if InDomain(grid_pos) and grid_mass[center[0]+i, center[1]+j, center[2]+k] > 0.0:
                dpos = grid_pos - pos_particle[p]

                NX = N(dpos[0] / dx)
                NY = N(dpos[1] / dx)
                NZ = N(dpos[2] / dx)
                dNX = dN(dpos[0] / dx)
                dNY = dN(dpos[1] / dx)
                dNZ = dN(dpos[2] / dx)

                dweight = ti.Vector.zero(float, 3)
                dweight[0] = dNX * NY * NZ
                dweight[1] = NX * dNY * NZ
                dweight[2] = NX * NY * dNZ

                grid_grad = dt * dE_dgvel @ dweight
                idx = GridIdx(center[0] + i, center[1] + j, center[2] + k)
                grad[idx * 3 + 0] += grid_grad[0]
                grad[idx * 3 + 1] += grid_grad[1]
                grad[idx * 3 + 2] += grid_grad[2]


    for x, y, z in grid_vel:
        idx = GridIdx(x, y, z)
        v_vel = ti.Vector([v[idx*3], v[idx*3+1], v[idx*3+2]])
        del_v = v_vel - grid_vel[x, y, z]
        grid_grad = grid_mass[x, y, z] * del_v
        grad[idx * 3 + 0] += grid_grad[0]
        grad[idx * 3 + 1] += grid_grad[1]
        grad[idx * 3 + 2] += grid_grad[2]


@ti.kernel
def ComputeHessian(v: ti.template(),  hessian: ti.sparse_matrix_builder()):
    for idx, j, k in ti.ndrange(6, ny, nz):
        i = bd_x_idx[idx]
        idx_grid = GridIdx(i, j, k)
        optimizer.x[idx_grid * 3 + 0] = 0
        optimizer.x[idx_grid * 3 + 1] = 0
        optimizer.x[idx_grid * 3 + 2] = 0

    for i, idx, k in ti.ndrange(nx, 6, nz):
        j = bd_y_idx[idx]
        idx_grid = GridIdx(i, j, k)
        optimizer.x[idx_grid * 3 + 0] = 0
        optimizer.x[idx_grid * 3 + 1] = 0
        optimizer.x[idx_grid * 3 + 2] = 0

    for i, j, idx in ti.ndrange(nx, ny, 6):
        k = bd_z_idx[idx]
        idx_grid = GridIdx(i, j, k)
        optimizer.x[idx_grid * 3 + 0] = 0
        optimizer.x[idx_grid * 3 + 1] = 0
        optimizer.x[idx_grid * 3 + 2] = 0

    for p in vel_particle:
        grad_vel = CalVelGradParticle(v, p)
        F = F_particle[p]
        new_F = (ti.Matrix.identity(float, 3) + grad_vel) @ F_particle[p]
        new_J = new_F.determinant()
        F_inv = new_F.inverse()
        F_inv_T = F_inv.transpose()

        center = int(pos_particle[p] / dx + ti.Vector([0.5, 0.5, 0.5])) # index of center grid
        for i, j, k in ti.ndrange((-1, 2), (-1, 2), (-1, 2)):
            grid_pos = float(center + ti.Vector([i, j, k])) * dx
            if InDomain(grid_pos):
                dpos = grid_pos - pos_particle[p]

                NX = N(dpos[0] / dx)
                NY = N(dpos[1] / dx)
                NZ = N(dpos[2] / dx)
                dNX = dN(dpos[0] / dx)
                dNY = dN(dpos[1] / dx)
                dNZ = dN(dpos[2] / dx)

                dweight = ti.Vector.zero(float, 3)
                dweight[0] = dNX * NY * NZ
                dweight[1] = NX * dNY * NZ
                dweight[2] = NX * NY * dNZ

                FTw = dt * F.transpose() @ dweight
                FWWF = F_inv_T @ FTw.outer_product(FTw) @ F_inv
                h1 = miu_0 * tm.dot(FTw, FTw) * ti.Matrix.identity(float, 3)
                h2 = miu_0 * FWWF
                h3 = lambda_0 * (1 - tm.log(new_J)) * FWWF
                grid_hessian = (h1 + h2 + h3) * p_vol_0

                U, sig, V = ti.svd(grid_hessian)
                for d in range(3):
                    if sig[d, d] < 0:
                        sig[d, d] = 0

                grid_hessian = U @ sig @ V

                idx = GridIdx(center[0] + i, center[1] + j, center[2] + k)
                for d1 in range(3):
                    for d2 in range(3):
                        hessian[idx * 3 + d1, idx * 3 + d2] += grid_hessian[d1, d2]

    for x, y, z in grid_vel:
        # m = 1 if grid_mass[x, y, z] == 0 else grid_mass[x, y, z]
        m = grid_mass[x, y, z]
        idx = GridIdx(x, y, z)
        for d in range(3):
            hessian[idx+d, idx+d] += m


@ti.kernel
def Particle2Grid():
    grid_mass.fill(0.0)
    grid_vel.fill(0.0)

    for p in pos_particle:
        affine = p_mass * C_particle[p]

        # p2g
        center = int(pos_particle[p] / dx) # index of center grid
        for i, j, k in ti.ndrange((-1, 2), (-1, 2), (-1, 2)):
            grid_pos = float(center + ti.Vector([0.5, 0.5, 0.5]) + ti.Vector([i, j, k])) * dx
            if InDomain(grid_pos):
                dpos = grid_pos - pos_particle[p]
                weight = N(dpos[0] / dx) * N(dpos[1] / dx) * N(dpos[2] / dx)
                grid_mass[center[0]+i, center[1]+j, center[2]+k] += weight * p_mass
                grid_vel[center[0]+i, center[1]+j, center[2]+k] += weight * (p_mass * vel_particle[p] + affine @ dpos)


@ti.kernel
def ApplyGravity():
    for x, y, z in grid_vel:
        if grid_mass[x, y, z] > 0.0:
            grid_vel[x, y, z] += dt * ti.Vector([0, 0, -5])


@ti.kernel
def DivideVel():
    for x, y, z in grid_vel:
        if grid_mass[x, y, z] == 0.0:
            grid_vel[x, y, z] = ti.Vector.zero(float, 3)
        else:
            grid_vel[x, y, z] /= grid_mass[x, y, z]


# optimizer = GradientDesent(ComputeEnergy, ComputeGrad, dim=3*nx*ny*nz, c1=0.9, eta=0.010)
optimizer = LBFGS(ComputeEnergy, ComputeGrad, dim=3*nx*ny*nz, alpha=1, beta=0.9, eta=0.10)


@ti.kernel
def ExplicitUpdate():
    for p in pos_particle:

        # compute stress
        F = F_particle[p]
        J = ti.Matrix.determinant(F_particle[p])

        stress = miu_0 * (F - F.inverse().transpose()) + lambda_0 * tm.log(J) * F.inverse().transpose()

        # p2g
        center = int(pos_particle[p] / dx) # index of center grid
        for i, j, k in ti.ndrange((-1, 2), (-1, 2), (-1, 2)):
            grid_pos = float(center + ti.Vector([0.5, 0.5, 0.5]) + ti.Vector([i, j, k])) * dx
            if InDomain(grid_pos):
                dpos = grid_pos - pos_particle[p]
                weight = N(dpos[0] / dx) * N(dpos[1] / dx) * N(dpos[2] / dx)
                grid_vel[center[0]+i, center[1]+j, center[2]+k] += (-4.0 * p_vol_0 * dt * stress @ F.transpose() * weight @ dpos / dx**2) / grid_mass[center[0]+i, center[1]+j, center[2]+k]
                

# @ti.kernel
# def ModExplicitUpdate():
#     for p in pos_particle:

#         # compute stress
#         F = F_particle[p]
#         J = ti.Matrix.determinant(F_particle[p])

#         stress = miu_0 * (F - F.inverse().transpose()) + lambda_0 * tm.log(J) * F.inverse().transpose()

#         center = int(pos_particle[p] / dx) # index of center grid
#         for i, j, k in ti.ndrange((-1, 2), (-1, 2), (-1, 2)):
#             grid_pos = float(center + ti.Vector([0.5, 0.5, 0.5]) + ti.Vector([i, j, k])) * dx
#             if InDomain(grid_pos):
#                 dpos = grid_pos - pos_particle[p]
#                 weight = N(dpos[0] / dx) * N(dpos[1] / dx) * N(dpos[2] / dx)
#                 # grid_vel[center[0]+i, center[1]+j, center[2]+k] += (-4.0 * p_vol_0 * dt * stress @ F.transpose() * weight @ dpos / dx**2) / grid_mass[center[0]+i, center[1]+j, center[2]+k]
#                 dv = (-4.0 * p_vol_0 * dt * stress @ F.transpose() * weight @ dpos / dx**2) / grid_mass[center[0]+i, center[1]+j, center[2]+k]
#                 idx = GridIdx(center[0]+i, center[1]+j, center[2]+k)
#                 optimizer.x[idx * 3 + 0] += dv[0]
#                 optimizer.x[idx * 3 + 1] += dv[1]
#                 optimizer.x[idx * 3 + 2] += dv[2]


def UpdateGrid():
    DivideVel()
    ApplyGravity()
    
    if implicit:
        optimizer.x.from_numpy(grid_vel.to_numpy().reshape(-1))
        # ModExplicitUpdate()
        optimizer.minimize(max_iter=1000)
        grid_vel.from_numpy(optimizer.x.to_numpy().reshape(nx, ny, nz, 3))
    else:
        ExplicitUpdate()
        pass
    

@ti.kernel
def Boundary():
    for x, y, z in grid_vel:
        if grid_mass[x, y, z] > 0.0:
            if x < 3 :
                if grid_vel[x, y, z][0] < 0:
                    grid_vel[x, y, z][0] = 0
                    grid_vel[x, y, z] *= 0.6

            if y < 3 :
                if grid_vel[x, y, z][1] < 0:
                    grid_vel[x, y, z][1] = 0
                    grid_vel[x, y, z] *= 0.6

            if z < 3 :
                if grid_vel[x, y, z][2] < 0:
                    grid_vel[x, y, z][2] = 0
                    grid_vel[x, y, z] *= 0.6

            if x >= nx - 3 :
                if grid_vel[x, y, z][0] > 0:
                    grid_vel[x, y, z][0] = 0
                    grid_vel[x, y, z] *= 0.6

            if y >= ny - 3 :
                if grid_vel[x, y, z][1] > 0:
                    grid_vel[x, y, z][1] = 0
                    grid_vel[x, y, z] *= 0.6

            if z >= nz - 3 :
                if grid_vel[x, y, z][2] > 0:
                    grid_vel[x, y, z][2] = 0
                    grid_vel[x, y, z] *= 0.6

@ti.kernel
def Grid2Particle():
    for p in pos_particle:
        vel_particle[p] = ti.Vector.zero(float, 3)
        C_particle[p] = ti.Matrix.zero(float, 3, 3)

        center = int(pos_particle[p] / dx) # index of center grid
        for i, j, k in ti.ndrange((-1, 2), (-1, 2), (-1, 2)):
            grid_pos = float(center + ti.Vector([0.5, 0.5, 0.5]) + ti.Vector([i, j, k])) * dx
            if InDomain(grid_pos):
                dpos = grid_pos - pos_particle[p]
                weight = N(dpos[0] / dx) * N(dpos[1] / dx) * N(dpos[2] / dx)
                gridv = grid_vel[center[0]+i, center[1]+j, center[2]+k]
                vel_particle[p] += weight * gridv
                C_particle[p] += 4 * weight * gridv.outer_product(dpos) / dx**2
        
        pos_particle[p] += dt * vel_particle[p]

        # update F
        F_particle[p] = (ti.Matrix.identity(float, 3) + dt * C_particle[p]) @ F_particle[p]


def main():
    Initiate()
    gui = ti.ui.Window('MPM', res = (700, 700))
    canvas = gui.get_canvas()
    canvas.set_background_color((1, 1, 1))
    scene = gui.get_scene()
    camera = ti.ui.Camera()

    camera.position(5, 5, 4)
    camera.lookat(0.8, 0.8, 0.8)
    camera.up(0, 0, 1)
    
    cur_frame = 0
    
    while gui.running:
        for _ in range(substep):
            Particle2Grid()
            UpdateGrid()
            Boundary()
            Grid2Particle()

        if visualization == 0:
            scene.particles(centers=pos_particle, radius=0.01, color=(1, 1, 1))
            scene.ambient_light((0.2, 0.2, 0.2))
            scene.point_light(pos=(2, 2, 2), color=(0.7, 0.7, 0.7))
            scene.point_light(pos=(-1, -1, 2), color=(0.7, 0.7, 0.7))
            scene.set_camera(camera)
            scene.mesh(pos_vertex, indice, color=(1, 1, 1))
            canvas.scene(scene)
            gui.show()
            cur_frame += 1
        else:
            np_pos = pos_particle.to_numpy() * 10
            series_prefix = "out/plyfile/implicit_.ply"
            writer = ti.tools.PLYWriter(num_vertices = n_particle)
            writer.add_vertex_pos(np_pos[:n_particle, 0], np_pos[:n_particle, 1], np_pos[:n_particle, 2])
            writer.add_vertex_color(np.full(n_particle, 0.8), np.full(n_particle, 0.8), np.full(n_particle, 0.8))
            writer.export_frame_ascii(cur_frame, series_prefix)
            cur_frame += 1

        print(cur_frame)
        # print(np.min(pos_particle.to_numpy(), axis = 0))
        # print(np.max(pos_particle.to_numpy(), axis = 0))
        # if cur_frame == 1800 :
            # exit()


if __name__ == "__main__":
    main()