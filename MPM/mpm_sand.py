import numpy as np
import taichi as ti
import taichi.math as tm

ti.init(arch=ti.gpu)

visualization = 0
n_particle = 72000
nx = ny = 40
nz = 180 

dx = 1 / 64
lenx, leny, lenz = nx * dx, ny * dx, nz * dx

p_vol_0 = dx**3 / (4)**3
p_rho = 100
p_mass = p_rho * p_vol_0

frame = 60
substep = 300
dt = 1 / (frame * substep)

# theta_c = 0.025
# theta_s = 0.0075

E_0 = 1e5
niu = 0.2
# zeta = 10
miu_0, lambda_0 = E_0 / (2 * (1 + niu)), E_0 * niu / ((1 + niu) * (1 - 2 * niu))  # Lame parameters

fangle = tm.pi / 6
alpha = ti.sqrt(2 / 3) * (2 * ti.sin(fangle) / (3 - ti.sin(fangle)))


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


ti.func
def MatrixNorm(a:ti.Matrix) -> float:
    res = 0.0
    for i, j in ti.ndrange(3, 3):
        res += a[i, j] * a[i, j]
    return tm.sqrt(res)


@ti.kernel
def Initiate():
    for i in pos_particle:
        a_len = 30 
        height = 80

        if benchmark == 1:
            idx = (i % (a_len * a_len)) // a_len
            idy = (i % (a_len * a_len)) % a_len
            idz = i // (a_len * a_len)

            base = ti.Vector([20, 10, 20]) * dx
            rand_dpos = ti.Vector([ti.random(float)-0.5, ti.random(float)-0.5, ti.random(float)-0.5]) * 0.05
            pos_particle[i] = (base + 0.5 * dx * ti.Vector([idx, idy, idz]) + rand_dpos)
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

        singular = ti.Vector([sigma[0, 0], sigma[1, 1], sigma[2, 2]])
        eps_p = tm.log(singular)
        eps_hat = eps_p - (eps_p.sum() * ti.Vector([1.0, 1.0, 1.0])) / 3
        gamma_p = eps_hat.norm() + ((3 * lambda_0 + 2 * miu_0) / (2 * miu_0)) * eps_p.sum() * alpha

        if gamma_p < 0:
            pass
        elif eps_p.sum() > 0 or eps_hat.norm() < 1e-8:
            sigma = ti.Matrix.identity(float, 3)
        else:
            Hp = eps_p - gamma_p * eps_hat / eps_hat.norm()
            new_singular = ti.exp(Hp)
            sigma[0, 0] = new_singular[0]
            sigma[1, 1] = new_singular[1]
            sigma[2, 2] = new_singular[2]

        # print(sigma)

        
        F_P_particle[p] = V @ sigma.inverse() @ U.transpose() @ F_E_particle[p] @ F_P_particle[p]
        F_E_particle[p] = U @ sigma @ V.transpose()

        # compute stress
        log_sigma = ti.Matrix.zero(float, 3, 3)
        inv_sigma = ti.Matrix.zero(float, 3, 3)
        for dim in range(3):
            log_sigma[dim, dim] = ti.log(sigma[dim, dim])
            inv_sigma[dim, dim] = 1 / sigma[dim, dim]

        term1 = 2 * miu_0 * inv_sigma @ log_sigma
        term2 = lambda_0 * log_sigma.trace() * inv_sigma
        stress = U @ (term1 + term2) @ V.transpose() @ F_E_particle[p].transpose()

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

            if x < 3 :
                if grid_vel[x, y, z][0] < 0:
                    grid_vel[x, y, z] = 0

            if x > nx - 3:
                if grid_vel[x, y, z][0] > 0:
                    grid_vel[x, y, z] = 0

            if y < 3 :
                if grid_vel[x, y, z][1] < 0:
                    grid_vel[x, y, z] = 0

            if y > ny - 3:
                if grid_vel[x, y, z][1] > 0:
                    grid_vel[x, y, z] = 0

            if z < 3 :
                if grid_vel[x, y, z][2] < 0:
                    grid_vel[x, y, z] = 0

            if z > nz - 3:
                if grid_vel[x, y, z][2] > 0:
                    grid_vel[x, y, z] = 0


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
    canvas.set_background_color((0, 0, 0))
    scene = gui.get_scene()
    camera = ti.ui.Camera()

    camera.position(2, 2, 1.5)
    camera.lookat(0.0, 0.0, 0.0)
    camera.up(0, 0, 1)
    
    cur_frame = 0
    
    while gui.running:
        gui.get_event()
        if gui.is_pressed('r'):
            Initiate()

        for _ in range(substep):
            Particle2Grid()
            Boundary()
            Grid2Particle()

        if visualization == 0:
            camera.track_user_inputs(gui, movement_speed=0.01, hold_key=ti.ui.RMB)
            scene.set_camera(camera)

            scene.particles(centers=pos_particle, radius=0.005, color=(194/256, 178/256, 128/256))

            scene.ambient_light((0.2, 0.2, 0.2))
            scene.point_light(pos=(2, 2, 2), color=(1, 1, 1))
            scene.point_light(pos=(-1, -1, 2), color=(1, 1, 1))

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
        # print(pos_particle[0], pos_particle[n_particle-1])
        # print(np.min(pos_particle.to_numpy(), axis = 0))
        # print(np.max(pos_particle.to_numpy(), axis = 0))
        # if cur_frame == 180 :
        #     exit()


if __name__ == "__main__":
    main()