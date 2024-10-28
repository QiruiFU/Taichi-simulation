import taichi as ti
from CGSolver import CGSolver
import vtk
from vtkmodules.util import numpy_support
import numpy as np
ti.init(ti.gpu)

visualization = 0

n_grid = 32 
n_particle = 60 * 60 * 60
boundry_len = 2
dx = boundry_len / n_grid

pos_particle = ti.Vector.field(3, float, shape=n_particle)
vel_particle = ti.Vector.field(3, float, shape=n_particle)
C = ti.Matrix.field(3, 3, float, shape=n_particle)
mss_particle = 1.0

# MAC grid
u_grid = ti.field(float, shape=(n_grid + 1, n_grid, n_grid))
v_grid = ti.field(float, shape=(n_grid, n_grid + 1, n_grid))
w_grid = ti.field(float, shape=(n_grid, n_grid, n_grid + 1))
u_mss_grid = ti.field(float, shape=(n_grid + 1, n_grid, n_grid))
v_mss_grid = ti.field(float, shape=(n_grid, n_grid + 1, n_grid))
w_mss_grid = ti.field(float, shape=(n_grid, n_grid, n_grid + 1))

marker = ti.field(dtype=ti.i32, shape=(n_grid, n_grid, n_grid)) # 0:air, 1:fluid, 2:solid
prs_grid = ti.field(float, shape=(n_grid, n_grid, n_grid)) # store at grid
div_vel_grid = ti.field(float, shape=(n_grid, n_grid, n_grid))
gravity = ti.Vector.field(3, float, shape=1)

frame = 60
rho = 1000
substep = 2
dt = 1 / (frame * substep)
damp = 0.9999

# ------------------- CG ---------------------#
#pressure solver
solver = CGSolver(n_grid, div_vel_grid, u_grid, v_grid, w_grid, marker)


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
def GetGridMark(x, y, z):
    res = 2
    if 0 <= x < n_grid and 0 <= y < n_grid and 0 <= z < n_grid:
        res = marker[x, y, z]
    else:
        res = 2
    
    return res


@ti.func
def Boundry(x):
    eps = dx
    if pos_particle[x][0] <= eps:
        pos_particle[x][0] = eps
        vel_particle[x][0] *= -1
    if pos_particle[x][0] >= 2 - eps:
        pos_particle[x][0] = eps
        vel_particle[x][0] *= -1

    if pos_particle[x][1] <= eps:
        pos_particle[x][1] = eps
        vel_particle[x][1] *= -1
    if pos_particle[x][1] >= 2 - eps:
        pos_particle[x][1] = eps
        vel_particle[x][1] *= -1

    if pos_particle[x][2] <= eps:
        pos_particle[x][2] = eps
        vel_particle[x][2] *= -1
    if pos_particle[x][2] >= 2 - eps:
        pos_particle[x][2] = eps
        vel_particle[x][2] *= -1


@ti.kernel
def initiate():
    gravity[0] = ti.Vector([0, 0, -9.8])
    marker.fill(0)
    for i in pos_particle:
        idz, idy, idx = i // 3600, (i % 3600) // 60, (i % 3600) % 60
        pos = [idx * 0.016 + 0.5, idy * 0.016 + 0.5, idz * 0.016 + 0.5]
        pos_particle[i] = pos
        vel_particle[i] = [0.0, 0.0, 0.0]
        marker[int(pos[0]/dx), int(pos[1]/dx), int(pos[2]/dx)] = 1
    

@ti.kernel
def Particle2Grid():
    u_grid.fill(0.0)
    v_grid.fill(0.0)
    w_grid.fill(0.0)
    u_mss_grid.fill(0.0)
    v_mss_grid.fill(0.0)
    w_mss_grid.fill(0.0)

    for p in pos_particle:
        coord_x, coord_y, coord_z = pos_particle[p][0] / dx, pos_particle[p][1] / dx, pos_particle[p][2] / dx
        idx, idy, idz = int(coord_x + 0.5), int(coord_y), int(coord_z)
        affine = C[p]

        for i, j, k in ti.ndrange((-1, 2), (-1, 2), (-1, 2)):
            new_x, new_y, new_z = idx + i, idy + j, idz + k
            if 0 <= new_x <= n_grid and 0 <= new_y < n_grid and 0 <= new_z < n_grid:
                new_pos = ti.Vector([float(idx), float(idy) + 0.5, float(idz) + 0.5]) + ti.Vector([i, j, k])
                weight = N(coord_x - new_pos[0]) * N(coord_y - new_pos[1]) * N(coord_z - new_pos[2])

                dpos = (new_pos - ti.Vector([coord_x, coord_y, coord_z])) * dx
                u_grid[new_x, new_y, new_z] += (vel_particle[p][0] + (affine @ dpos)[0]) * weight
                # u_grid[new_x, new_y, new_z] += vel_particle[p][0] * weight

                u_mss_grid[new_x, new_y, new_z] += mss_particle * weight

    for p in pos_particle:
        coord_x, coord_y, coord_z = pos_particle[p][0] / dx, pos_particle[p][1] / dx, pos_particle[p][2]/ dx 
        idx, idy, idz = int(coord_x), int(coord_y + 0.5), int(coord_z)
        affine = C[p]

        for i, j, k in ti.ndrange((-1, 2), (-1, 2), (-1, 2)):
            new_x, new_y, new_z = idx + i, idy + j, idz + k
            if 0 <= new_x < n_grid and 0 <= new_y <= n_grid and 0 <= new_z < n_grid:
                new_pos = ti.Vector([float(idx) + 0.5, float(idy), float(idz) + 0.5]) + ti.Vector([i, j, k])
                weight = N(coord_x - new_pos[0]) * N(coord_y - new_pos[1]) * N(coord_z - new_pos[2])

                dpos = (new_pos - ti.Vector([coord_x, coord_y, coord_z])) * dx
                v_grid[new_x, new_y, new_z] += (vel_particle[p][1] + (affine @ dpos)[1]) * weight
                # v_grid[new_x, new_y, new_z] += vel_particle[p][1] * weight

                v_mss_grid[new_x, new_y, new_z] += mss_particle * weight

    for p in pos_particle:
        coord_x, coord_y, coord_z = pos_particle[p][0] / dx, pos_particle[p][1] / dx, pos_particle[p][2] / dx 
        idx, idy, idz = int(coord_x), int(coord_y), int(coord_z + 0.5)
        affine = C[p]

        for i, j, k in ti.ndrange((-1, 2), (-1, 2), (-1, 2)):
            new_x, new_y, new_z = idx + i, idy + j, idz + k
            if 0 <= new_x < n_grid and 0 <= new_y < n_grid and 0 <= new_z <= n_grid:
                new_pos = ti.Vector([float(idx) + 0.5, float(idy) + 0.5, float(idz)]) + ti.Vector([i, j, k])
                weight = N(coord_x - new_pos[0]) * N(coord_y - new_pos[1]) * N(coord_z - new_pos[2])

                dpos = (new_pos - ti.Vector([coord_x, coord_y, coord_z])) * dx
                w_grid[new_x, new_y, new_z] += (vel_particle[p][2] + (affine @ dpos)[2]) * weight
                # w_grid[new_x, new_y, new_z] += vel_particle[p][2] * weight

                w_mss_grid[new_x, new_y, new_z] += mss_particle * weight

    for i, j, k in u_grid:
        if u_mss_grid[i, j, k] > 0.0 :
            u_grid[i, j, k] /= u_mss_grid[i, j, k]
        if i == 0 or i == n_grid:
            u_grid[i, j, k] = 0.0

    for i, j, k in v_grid:
        if v_mss_grid[i, j, k] > 0.0 :
            v_grid[i, j, k] /= v_mss_grid[i, j, k]
        if i == 0 or i == n_grid:
            v_grid[i, j, k] = 0.0
    
    for i, j, k in w_grid:
        if w_mss_grid[i, j, k] > 0.0 :
            w_grid[i, j, k] /= w_mss_grid[i, j, k]
        if i == 0 or i == n_grid:
            w_grid[i, j, k] = 0.0


@ti.kernel
def PreProjection():
    # calculate div_vel_grid
    for i, j, k in div_vel_grid:
        if marker[i, j, k] == 1:
            div_x = (u_grid[i+1, j, k] - u_grid[i, j, k]) / dx
            div_y = (v_grid[i, j+1, k] - v_grid[i, j, k]) / dx
            div_z = (w_grid[i, j, k+1] - w_grid[i, j, k]) / dx
            div_vel_grid[i, j, k] = div_x + div_y + div_z
    

@ti.kernel
def PostProjection():
    for i, j, k in u_grid:
        if i == 0 or i == n_grid:
            u_grid[i, j, k] = 0
        else:
            u_grid[i, j, k] += (prs_grid[i-1, j, k] - prs_grid[i, j, k]) * dt / (rho * dx)
    
    for i, j, k in v_grid:
        if j == 0 or j == n_grid:
            v_grid[i, j, k] = 0
        else:
            v_grid[i, j, k] += (prs_grid[i, j-1, k] - prs_grid[i, j, k]) * dt / (rho * dx)

    for i, j, k in w_grid:
        if k == 0 or k == n_grid:
            w_grid[i, j, k] = 0
        else:
            w_grid[i, j, k] += (prs_grid[i, j, k-1] - prs_grid[i, j, k]) * dt / (rho * dx)


@ti.kernel
def Grid2Particle():
    C.fill(0.0)

    for p in pos_particle:
        coord_x, coord_y, coord_z = pos_particle[p][0] / dx, pos_particle[p][1] / dx, pos_particle[p][2]/ dx 
        idx, idy, idz = int(coord_x + 0.5), int(coord_y), int(coord_z)
        new_u = 0.0

        for i, j, k in ti.ndrange((-1, 2), (-1, 2), (-1, 2)):
            new_x, new_y, new_z = idx + i, idy + j, idz + k
            if 0 <= new_x <= n_grid and 0 <= new_y < n_grid and 0 <= new_z < n_grid:
                new_pos = ti.Vector([float(idx), float(idy) + 0.5, float(idz) + 0.5]) + ti.Vector([i, j, k])
                weight = N(coord_x - new_pos[0]) * N(coord_y - new_pos[1]) * N(coord_z - new_pos[2])
                new_u += u_grid[new_x, new_y, new_z] * weight
                
                dpos = new_pos - ti.Vector([coord_x, coord_y, coord_z])
                C[p] += 4 * weight * ti.Vector([u_grid[new_x, new_y, new_z], 0, 0]).outer_product(dpos) / dx 

        vel_particle[p][0] = new_u * damp

    for p in pos_particle:
        coord_x, coord_y, coord_z = pos_particle[p][0] / dx, pos_particle[p][1] / dx, pos_particle[p][2] / dx 
        idx, idy, idz = int(coord_x), int(coord_y + 0.5), int(coord_z)
        new_v = 0.0

        for i, j, k in ti.ndrange((-1, 2), (-1, 2), (-1, 2)):
            new_x, new_y, new_z = idx + i, idy + j, idz + k
            if 0 <= new_x < n_grid and 0 <= new_y <= n_grid and 0 <= new_z < n_grid:
                new_pos = ti.Vector([float(idx) + 0.5, float(idy), float(idz) + 0.5]) + ti.Vector([i, j, k])
                weight = N(coord_x - new_pos[0]) * N(coord_y - new_pos[1]) * N(coord_z - new_pos[2])
                new_v += v_grid[new_x, new_y, new_z] * weight
                
                dpos = new_pos - ti.Vector([coord_x, coord_y, coord_z])
                C[p] += 4 * weight * ti.Vector([0, v_grid[new_x, new_y, new_z], 0]).outer_product(dpos) / dx 

        vel_particle[p][1] = new_v * damp

    for p in pos_particle:
        coord_x, coord_y, coord_z = pos_particle[p][0] / dx, pos_particle[p][1] / dx, pos_particle[p][2] / dx 
        idx, idy, idz = int(coord_x), int(coord_y), int(coord_z + 0.5)
        new_w = 0.0

        for i, j, k in ti.ndrange((-1, 2), (-1, 2), (-1, 2)):
            new_x, new_y, new_z = idx + i, idy + j, idz + k
            if 0 <= new_x < n_grid and 0 <= new_y < n_grid and 0 <= new_z <= n_grid:
                new_pos = ti.Vector([float(new_x) + 0.5, float(new_y) + 0.5, float(new_z)])
                weight = N(coord_x - new_pos[0]) * N(coord_y - new_pos[1]) * N(coord_z - new_pos[2])
                new_w += w_grid[new_x, new_y, new_z] * weight

                dpos = new_pos - ti.Vector([coord_x, coord_y, coord_z])
                C[p] += 4 * weight * ti.Vector([0, 0, w_grid[new_x, new_y, new_z]]).outer_product(dpos) / dx 

        vel_particle[p][2] = new_w * damp

    for p in vel_particle:
        pos_particle[p] += vel_particle[p] * dt
        # Boundry(p)


@ti.kernel
def MarkCell():
    for i, j, k in marker:
        if not (marker[i, j, k] == 2):
            marker[i, j, k] = 0
    
    for p in pos_particle:
        idx, idy, idz = int(pos_particle[p][0] / dx), int(pos_particle[p][1] / dx), int(pos_particle[p][2] / dx)
        if not (marker[idx, idy, idz] == 2):
            marker[idx, idy, idz] = 1


def SolvePressure():
    scale_A = dt / (rho * dx * dx)
    scale_b = 1 / dx

    solver.system_init(scale_A, scale_b)
    solver.solve(500)

    prs_grid.copy_from(solver.x)


@ti.kernel
def enforce_boundary():
    # u solid
    for i, j, k in u_grid:
        if i == 0 or i == n_grid or marker[i - 1, j, k] == 2 or marker[i, j, k] == 2:
            u_grid[i, j, k] = 0.0

    # v solid
    for i, j, k in v_grid:
        if j == 0 or j == n_grid or marker[i, j - 1, k] == 2 or marker[i, j, k] == 2:
            v_grid[i, j, k] = 0.0

    # w solid
    for i, j, k in v_grid:
        if k == 0 or k == n_grid or marker[i, j, k - 1] == 2 or marker[i, j, k] == 2:
            w_grid[i, j, k] = 0.0


@ti.kernel
def ApplyGravity():
    # update gravity
    for i, j, k in w_grid:
        w_grid[i, j, k] += gravity[0][2] * dt


def main():
    initiate()
    gui = ti.ui.Window('APIC', res = (700, 700))
    canvas = gui.get_canvas()
    canvas.set_background_color((1, 1, 1))
    scene = gui.get_scene()
    camera = ti.ui.Camera()

    camera.position(5, 5, 5)
    camera.lookat(1, 1, 1)
    camera.up(0, 0, 1)
    
    cur_frame = 0
    
    while gui.running:
        for s1 in range(substep):
            ApplyGravity()
            enforce_boundary()
            PreProjection()
            SolvePressure()
            PostProjection()
            enforce_boundary()
            Grid2Particle()
            MarkCell()
            Particle2Grid()
            pass

        if visualization == 0:
            scene.particles(centers=pos_particle, radius=0.02, color=(1, 1, 1))
            scene.ambient_light((0.7, 0.7, 0.7))
            scene.set_camera(camera)
            canvas.scene(scene)
            gui.show()
            cur_frame += 1
        else:
            np_pos = pos_particle.to_numpy()
            series_prefix = "out/plyfile/water_.ply"
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