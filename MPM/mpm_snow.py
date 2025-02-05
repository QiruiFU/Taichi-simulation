import numpy as np
import taichi as ti
import taichi.math as tm

ti.init(arch=ti.gpu)

visualization = 0
n_particle = 64000
nx = ny = 200
nz = 80 

dx = 1 / 64
lenx, leny, lenz = nx * dx, ny * dx, nz * dx

p_vol_0 = dx**3 / (4)**3
p_rho = 400
p_mass = p_rho * p_vol_0

frame = 60
substep = 300
dt = 1 / (frame * substep)

theta_c = 0.025
theta_s = 0.0075

E_0 = 1.4e5
niu = 0.2
zeta = 10
miu_0, lambda_0 = E_0 / (2 * (1 + niu)), E_0 * niu / ((1 + niu) * (1 - 2 * niu))  # Lame parameters


vel_particle = ti.Vector.field(3, float, shape = n_particle)
pos_particle = ti.Vector.field(3, float, shape = n_particle)
C_particle = ti.Matrix.field(3, 3, float, shape = n_particle)
F_E_particle = ti.Matrix.field(3, 3, float, shape = n_particle)
F_P_particle = ti.Matrix.field(3, 3, float, shape = n_particle)

grid_vel = ti.Vector.field(3, float, shape = (nx, ny, nz))
grid_mass = ti.field(float, shape = (nx, ny, nz))

benchmark = 1


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


@ti.kernel
def Initiate():
    for i in pos_particle:
        cube_len = 40

        if benchmark == 1:
            idx = (i % (cube_len * cube_len)) // cube_len
            idy = (i % (cube_len * cube_len)) % cube_len
            idz = i // (cube_len * cube_len)

            base = ti.Vector([1.0, 1.0, 0.8])
            rand_dpos = ti.Vector([ti.random(float)-0.5, ti.random(float)-0.5, ti.random(float)-0.5]) * 0.05
            pos_particle[i] = (base + 0.5 * dx * ti.Vector([idx, idy, idz]) + rand_dpos)
            vel_particle[i] = ti.Vector([2.0, 2.0, 0.0])
            F_E_particle[i] = ti.Matrix.identity(float, 3)
            F_P_particle[i] = ti.Matrix.identity(float, 3)
            C_particle[i] = ti.Matrix.zero(float, 3, 3)

            pos_particle[i] = pos_particle[i] - base - 5 * ti.Vector([dx, dx, dx])
            rot_x = ti.Matrix([[1, 0, 0], [0, 0.5, tm.sqrt(3)/2], [0, -tm.sqrt(3)/2, 0.5]]) 
            rot_y = ti.Matrix([[tm.sqrt(2)/2, 0, tm.sqrt(2)/2], [0, 1, 0], [-tm.sqrt(2)/2, 0, tm.sqrt(2)/2]]) 
            pos_particle[i] = rot_y @ rot_x @ pos_particle[i]
            pos_particle[i] = pos_particle[i] + base + 5 * ti.Vector([dx, dx, dx])
        elif benchmark == 2:
            if i < n_particle // 2:
                idx = (i % (cube_len * cube_len)) // cube_len
                idy = (i % (cube_len * cube_len)) % cube_len
                idz = i // (cube_len * cube_len)

                base = ti.Vector([1.0, 1.0, 0.75])
                rand_dpos = ti.Vector([ti.random(float)-0.5, ti.random(float)-0.5, ti.random(float)-0.5]) * 0.05
                pos_particle[i] = (base + 0.5 * dx * ti.Vector([idx, idy, idz]) + rand_dpos)
                vel_particle[i] = ti.Vector([4, 4, 1])
                F_E_particle[i] = ti.Matrix.identity(float, 3)
                F_P_particle[i] = ti.Matrix.identity(float, 3)
                C_particle[i] = ti.Matrix.zero(float, 3, 3)

                pos_particle[i] = pos_particle[i] - base - 5 * ti.Vector([dx, dx, dx])
                rot_x = ti.Matrix([[1, 0, 0], [0, 0.5, tm.sqrt(3)/2], [0, -tm.sqrt(3)/2, 0.5]]) 
                rot_y = ti.Matrix([[tm.sqrt(3)/2, 0, 0.5], [0, 1, 0], [-0.5, 0, tm.sqrt(3)/2]]) 
                pos_particle[i] = rot_y @ rot_x @ pos_particle[i]
                pos_particle[i] = pos_particle[i] + base + 5 * ti.Vector([dx, dx, dx])
            else:
                j = i - n_particle // 2
                idx = (j % (cube_len * cube_len)) // cube_len
                idy = (j % (cube_len * cube_len)) % cube_len
                idz = j // (cube_len * cube_len)

                base = ti.Vector([2.2, 2.2, 0.8])
                rand_dpos = ti.Vector([ti.random(float)-0.5, ti.random(float)-0.5, ti.random(float)-0.5]) * 0.05
                pos_particle[i] = (base + 0.5 * dx * ti.Vector([idx, idy, idz]) + rand_dpos)
                vel_particle[i] = ti.Vector([-4, -4, 1])
                F_E_particle[i] = ti.Matrix.identity(float, 3)
                F_P_particle[i] = ti.Matrix.identity(float, 3)
                C_particle[i] = ti.Matrix.zero(float, 3, 3)

                pos_particle[i] = pos_particle[i] - base - 5 * ti.Vector([dx, dx, dx])
                rot_x = ti.Matrix([[1, 0, 0], [0, 0.5, tm.sqrt(3)/2], [0, -tm.sqrt(3)/2, 0.5]]) 
                cos45 = tm.sqrt(2) / 2
                rot_y = ti.Matrix([[cos45, 0, cos45], [0, 1, 0], [-cos45, 0, cos45]]) 
                pos_particle[i] = rot_y @ rot_x @ pos_particle[i]
                pos_particle[i] = pos_particle[i] + base + 5 * ti.Vector([dx, dx, dx])




@ti.kernel
def Particle2Grid():
    grid_mass.fill(0.0)
    grid_vel.fill(0.0)

    for p in pos_particle:
        # update F_E, F_P
        F_E_particle[p] = (ti.Matrix.identity(float, 3) + dt * C_particle[p]) @ F_E_particle[p]
        U, sigma, V = ti.svd(F_E_particle[p])
        for i in range(3):
            sigma[i, i] = tm.clamp(sigma[i, i], 1 - theta_c, 1 + theta_s)
        
        F_P_particle[p] = V @ sigma.inverse() @ U.transpose() @ F_E_particle[p] @ F_P_particle[p]
        F_E_particle[p] = U @ sigma @ V.transpose()

        # compute stress
        J_P = ti.Matrix.determinant(F_P_particle[p])
        J_E = ti.Matrix.determinant(F_E_particle[p])

        multi = tm.clamp(tm.exp(zeta * (1 - J_P)), 0.1, 5.0)
        miu = miu_0 * multi
        la = lambda_0 * multi

        term1 = 2.0 * miu * (F_E_particle[p] - U @ V.transpose()) @ F_E_particle[p].transpose()
        term2 = ti.Matrix.identity(float, 3) * la * J_E * (J_E - 1.0)
        stress = term1 + term2
        affine = p_mass * C_particle[p] - p_vol_0 * 4.0 * dt * stress / dx**2

        # p2g
        center = int(pos_particle[p] / dx + ti.Vector([0.5, 0.5, 0.5])) # index of center grid
        for i, j, k in ti.ndrange((-1, 2), (-1, 2), (-1, 2)):
            grid_pos = float(center + ti.Vector([i, j, k])) * dx
            if InDomain(grid_pos):
                dpos = grid_pos - pos_particle[p]
                weight = N(dpos[0] / dx) * N(dpos[1] / dx) * N(dpos[2] / dx)
                grid_mass[center[0]+i, center[1]+j, center[2]+k] += weight * p_mass
                grid_vel[center[0]+i, center[1]+j, center[2]+k] += weight * (p_mass * vel_particle[p] + affine @ dpos)
        

@ti.kernel
def Boundary():
    for x, y, z in grid_vel:
        if grid_mass[x, y, z] > 0.0:
            grid_vel[x, y, z] /= grid_mass[x, y, z]
            grid_vel[x, y, z] += dt * ti.Vector([0, 0, -9.8])

            if z < 3 :
                if grid_vel[x, y, z][2] < 0:
                    grid_vel[x, y, z][2] = 0
                    grid_vel[x, y, z] *= 0.6


@ti.kernel
def Grid2Particle():
    for p in pos_particle:
        vel_particle[p] = ti.Vector.zero(float, 3)
        C_particle[p] = ti.Matrix.zero(float, 3, 3)

        center = int(pos_particle[p] / dx + ti.Vector([0.5, 0.5, 0.5])) # index of center grid
        for i, j, k in ti.ndrange((-1, 2), (-1, 2), (-1, 2)):
            grid_pos = float(center + ti.Vector([i, j, k])) * dx
            if InDomain(grid_pos):
                dpos = grid_pos - pos_particle[p]
                weight = N(dpos[0] / dx) * N(dpos[1] / dx) * N(dpos[2] / dx)
                gridv = grid_vel[center[0]+i, center[1]+j, center[2]+k]
                vel_particle[p] += weight * gridv
                C_particle[p] += 4 * weight * gridv.outer_product(dpos) / dx**2
        
        pos_particle[p] += dt * vel_particle[p]


def main():
    Initiate()
    gui = ti.ui.Window('SNOW', res = (700, 700))
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
            Boundary()
            Grid2Particle()

        if visualization == 0:
            scene.particles(centers=pos_particle, radius=0.001, color=(1, 1, 1))
            scene.ambient_light((0.2, 0.2, 0.2))
            scene.point_light(pos=(2, 2, 2), color=(0.7, 0.7, 0.7))
            scene.point_light(pos=(-1, -1, 2), color=(0.7, 0.7, 0.7))
            scene.set_camera(camera)
            canvas.scene(scene)
            gui.show()
            cur_frame += 1
        else:
            np_pos = pos_particle.to_numpy() * 10
            series_prefix = "out/plyfile/snow_.ply"
            writer = ti.tools.PLYWriter(num_vertices = n_particle)
            writer.add_vertex_pos(np_pos[:n_particle, 0], np_pos[:n_particle, 1], np_pos[:n_particle, 2])
            writer.add_vertex_color(np.full(n_particle, 0.8), np.full(n_particle, 0.8), np.full(n_particle, 0.8))
            writer.export_frame_ascii(cur_frame, series_prefix)
            cur_frame += 1

        print(cur_frame)
        # print(np.min(pos_particle.to_numpy(), axis = 0))
        # print(np.max(pos_particle.to_numpy(), axis = 0))
        if cur_frame == 180 :
            exit()


if __name__ == "__main__":
    main()