import numpy as np
import taichi as ti
import taichi.math as tm

ti.init(arch=ti.gpu)

visualization = 1
n_particle = 2000
nx = ny = nz = 32

frame = 60
substep = 30
dt = 1 / (frame * substep)

theta_c = 0.025
theta_s = 0.0075

E_0 = 1.4e4
niu = 0.2
miu_0, lambda_0 = E_0 / (2 * (1 + niu)), E_0 * niu / ((1 + niu) * (1 - 2 * niu))  # Lame parameters


vel_particle = ti.Vector.field(3, float, shape = n_particle)
pos_particle = ti.Vector.field(3, float, shape = n_particle)
C_particle = ti.Matrix.field(3, 3, float, shape = n_particle)
F_E_particle = ti.Matrix.field(3, 3, float, shape = n_particle)
F_P_particle = ti.Matrix.field(3, 3, float, shape = n_particle)

grid_vel = ti.Vector.field(3, float, shape = (nx, ny, nz))
grid_mass = ti.field(float, shape = (nx, ny, nz))


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
    in_x = x[0] >= 0.0 and x[0] < nx
    in_y = x[0] >= 0.0 and x[0] < ny
    in_z = x[0] >= 0.0 and x[0] < nz
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
        dx = (i % 100) // 10
        dy = (i % 100) % 10
        dz = i // 100

        base = ti.Vector([10, 10, 15]) + ti.Vector([0.25, 0.25, 0.25])
        dpos = ti.Vector([ti.random(float) - 0.5, ti.random(float) - 0.5, ti.random(float) - 0.5]) * 0.5
        pos_particle[i] = base + 0.5 * ti.Vector([dx, dy, dz]) + dpos
        vel_particle[i] = ti.Vector([0.0, 0.0, 0.0])
        F_E_particle[i] = ti.Matrix.identity(float, 3)
        F_P_particle[i] = ti.Matrix.identity(float, 3)
        C_particle[i] = ti.Matrix.zero(float, 3, 3)


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
        # miu = miu_0 * tm.exp(10.0 * (1 - J_P))
        # la = lambda_0 * tm.exp(10.0 * (1 - J_P))
        miu = miu_0
        la = lambda_0
        term1 = 2.0 * miu * (F_E_particle[p] - U @ V.transpose()) @ F_E_particle[p].transpose()
        term2 = ti.Matrix.identity(float, 3) * la * J_E * (J_E - 1.0)
        stress = term1 + term2
        affine = C_particle[p] - dt * 1.25 * 4.0 * stress

        # p2g
        center = int(pos_particle[p] + ti.Vector([0.5, 0.5, 0.5])) # index of center grid
        for i, j, k in ti.ndrange((-1, 2), (-1, 2), (-1, 2)):
            grid_pos = float(center + ti.Vector([i, j, k]))
            if InDomain(grid_pos):
                dpos = grid_pos - pos_particle[p]
                weight = N(dpos[0]) * N(dpos[1]) * N(dpos[2])
                grid_mass[center[0]+i, center[1]+j, center[2]+k] += weight
                grid_vel[center[0]+i, center[1]+j, center[2]+k] += weight * (vel_particle[p] + affine @ dpos)
        

@ti.kernel
def Boundary():
    for x, y, z in grid_vel:
        if grid_mass[x, y, z] > 0.0:
            grid_vel[x, y, z] /= grid_mass[x, y, z]
            grid_vel[x, y, z] += dt * ti.Vector([0, 0, -9.8])

            if x < 3 and grid_vel[x, y, z][0] < 0:
                grid_vel[x, y, z][0] *= -0.2
            if x > nx - 3 and grid_vel[x, y, z][0] > 0:
                grid_vel[x, y, z][0] *= -0.2

            if y < 3 and grid_vel[x, y, z][1] < 0:
                grid_vel[x, y, z][1] *= -0.2
            if y > ny - 3 and grid_vel[x, y, z][1] > 0:
                grid_vel[x, y, z][1] *= -0.2

            if z < 3 and grid_vel[x, y, z][2] < 0:
                grid_vel[x, y, z][2] *= -0.2
            if z > nz - 3 and grid_vel[x, y, z][2] > 0:
                grid_vel[x, y, z][2] *= -0.2


@ti.kernel
def Grid2Particle():
    for p in pos_particle:
        vel_particle[p] = ti.Vector.zero(float, 3)
        C_particle[p] = ti.Matrix.zero(float, 3, 3)

        center = int(pos_particle[p] + ti.Vector([0.5, 0.5, 0.5])) # index of center grid
        for i, j, k in ti.ndrange((-1, 2), (-1, 2), (-1, 2)):
            grid_pos = float(center) + ti.Vector([i, j, k])
            if InDomain(grid_pos):
                dpos = grid_pos - pos_particle[p]
                weight = N(dpos[0]) * N(dpos[1]) * N(dpos[2])
                gridv = grid_vel[center[0]+i, center[1]+j, center[2]+k]
                vel_particle[p] += weight * gridv
                C_particle[p] += 4 * weight * gridv.outer_product(dpos)
        
        pos_particle[p] += dt * vel_particle[p]

        # eps = 0.05
        # if pos_particle[p][0] < 0:
        #     pos_particle[p][0] = 0.05
        #     vel_particle[p][0] = abs(vel_particle[p][0])
        # if pos_particle[p][1] < 0:
        #     pos_particle[p][1] = 0.05
        #     vel_particle[p][1] = abs(vel_particle[p][1])
        # if pos_particle[p][2] < 0:
        #     pos_particle[p][2] = 0.05
        #     vel_particle[p][2] = abs(vel_particle[p][2])

        # if pos_particle[p][0] > nx - 1:
        #     pos_particle[p][0] = nx - 1 - eps
        #     vel_particle[p][0] = -abs(vel_particle[p][0])
        # if pos_particle[p][1] > ny - 1:
        #     pos_particle[p][1] = ny - 1 - eps
        #     vel_particle[p][1] = -abs(vel_particle[p][1])
        # if pos_particle[p][2] > nz - 1:
        #     pos_particle[p][2] = nz - 1 - eps
        #     vel_particle[p][2] = -abs(vel_particle[p][2])


def main():
    Initiate()
    gui = ti.ui.Window('SNOW', res = (700, 700))
    canvas = gui.get_canvas()
    canvas.set_background_color((1, 1, 1))
    scene = gui.get_scene()
    camera = ti.ui.Camera()

    camera.position(50, 50, 50)
    camera.lookat(1, 1, 1)
    camera.up(0, 0, 1)
    
    cur_frame = 0
    
    while gui.running:
        for s1 in range(substep):
            Particle2Grid()
            Boundary()
            Grid2Particle()

        if visualization == 0:
            scene.particles(centers=pos_particle, radius=0.1, color=(1, 1, 1))
            scene.ambient_light((0.7, 0.7, 0.7))
            scene.set_camera(camera)
            canvas.scene(scene)
            gui.show()
            cur_frame += 1
        else:
            np_pos = pos_particle.to_numpy()
            series_prefix = "out/plyfile/snow_.ply"
            writer = ti.tools.PLYWriter(num_vertices = n_particle)
            writer.add_vertex_pos(np_pos[:n_particle, 0], np_pos[:n_particle, 1], np_pos[:n_particle, 2])
            writer.add_vertex_color(np.full(n_particle, 0), np.full(n_particle, 0), np.full(n_particle, 0.8))
            writer.export_frame_ascii(cur_frame, series_prefix)
            cur_frame += 1

        print(cur_frame)
        # if cur_frame == 1800 :
        #     exit()


if __name__ == "__main__":
    main()