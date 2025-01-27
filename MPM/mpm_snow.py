import numpy as np
import taichi as ti
import taichi.math as tm

ti.init(arch=ti.gpu)

visualization = 0
n_particle = 2000
nx = ny = nz = 32

frame = 60
substep = 30
dt = 1 / (frame * substep)

theta_c = 2.5e-2
theta_s = 7.5e-3

E_0 = 1.4e3
niu = 0.2
miu_0, lambda_0 = E_0 / (2 * (1 + niu)), E_0 * niu / ((1 + niu) * (1 - 2 * niu))  # Lame parameters


vel_particle = ti.Vector.field(3, float, shape = n_particle)
pos_particle = ti.Vector.field(3, float, shape = n_particle)
C_particle = ti.Matrix.field(3, 3, float, shape = n_particle)
F_E_particle = ti.Matrix.field(3, 3, float, shape = n_particle)
F_P_particle = ti.Matrix.field(3, 3, float, shape = n_particle)

grid_vel = ti.Vector.field(3, float, shape = (nx, ny, nz))
grid_mass = ti.field(float, shape = (nx, ny, nz))

@ti.kernel
def Initiate():
    for i in pos_particle:
        dx = (i % 100) // 10
        dy = (i % 100) % 10
        dz = i // 100

        base = ti.Vector([10, 10, 10]) + ti.Vector([0.25, 0.25, 0.25])
        dpos = ti.Vector([ti.random(float) - 0.5, ti.random(float) - 0.5, ti.random(float) - 0.5]) * 0.5
        pos_particle[i] = base + 0.5 * ti.Vector([dx, dy, dz]) + dpos
        vel_particle[i] = ti.Vector([0.0, 0.0, 0.0])


@ti.kernel
def Particle2Grid():
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
        miu = miu_0 * tm.exp(10.0 * (1 - J_P))
        la = lambda_0 * tm.exp(10.0 * (1 - J_P))
        term1 = 2.0 * miu * (F_E_particle[p] - U @ V.transpose()) @ F_E_particle[p].transpose()
        term2 = ti.Matrix.identity(float, 3) * la * J_E * (J_E - 1.0)
        stress = term1 + term2

        center = int(pos_particle[p] + ti.Vector([0.5, 0.5, 0.5]))
        for i, j, k in ti.ndrange((-1, 2), (-1, 2), (-1, 2)):
            grid_pos = center + ti.Vector([i, j, k])
            dpos = grid_pos - pos_particle[p]
            weight = N(dpos[0]) * N(dpos[1]) * N(dpos[2])


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
        if cur_frame == 1800 :
            exit()


if __name__ == "__main__":
    main()