import taichi as ti
from CGSolver import CGSolver
import math
ti.init(ti.gpu)

n_grid = 256 
n_particle = 250000
dx = 1 / n_grid

pos_particle = ti.Vector.field(2, float, shape=n_particle)
vel_particle = ti.Vector.field(2, float, shape=n_particle)
C = ti.Matrix.field(2, 2, float, shape=n_particle)
mss_particle = 1.0

# MAC grid
u_grid = ti.field(float, shape=(n_grid + 1, n_grid))
v_grid = ti.field(float, shape=(n_grid, n_grid + 1))
u_mss_grid = ti.field(float, shape=(n_grid + 1, n_grid))
v_mss_grid = ti.field(float, shape=(n_grid, n_grid + 1))

marker = ti.field(dtype=ti.i32, shape=(n_grid, n_grid)) # 0:air, 1:fluid, 2:solid
prs_grid = ti.field(float, shape=(n_grid, n_grid)) # store at grid
new_prs = ti.field(float, shape=(n_grid, n_grid))
div_vel_grid = ti.field(float, shape=(n_grid, n_grid))
gravity = ti.Vector.field(2, float, shape=1)

frame = 60
rho = 1000
substep = 2
dt = 1 / (frame * substep)
damp = 0.99

# ------------------- CG ---------------------#
#pressure solver
solver = CGSolver(n_grid, n_grid, div_vel_grid, u_grid, v_grid, marker)


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
def GetGridMark(x, y):
    res = 2
    if 0 <= x < n_grid and 0 <= y < n_grid:
        res = marker[x, y]
    else:
        res = 2
    
    return res


@ti.func
def Boundry(x):
    eps = dx
    if pos_particle[x][0] <= eps:
        pos_particle[x][0] = eps
        vel_particle[x][0] = 0
    if pos_particle[x][0] >= 1 - eps:
        pos_particle[x][0] = eps
        vel_particle[x][0] = 0
    if pos_particle[x][1] <= eps:
        pos_particle[x][1] = eps
        vel_particle[x][1] = 0
    if pos_particle[x][1] >= 1 - eps:
        pos_particle[x][1] = eps
        vel_particle[x][1] = 0


@ti.kernel
def initiate():
    gravity[0] = ti.Vector([0, -9.8])
    marker.fill(0)
    for i in pos_particle:
        idx, idy = i // 500, i % 500
        pos = [idx * dx / 5 + 0.3 + dx / 4, idy * dx / 5 + 0.1 + dx / 4]
        pos_particle[i] = pos
        vel_particle[i] = [0.0, 0.0]
        marker[int(pos[0]/dx), int(pos[1]/dx)] = 1
    
    for i in range(n_grid):
        marker[i, 0] = 2
        marker[i, n_grid-1] = 2
        marker[0, i] = 2
        marker[n_grid-1, i] = 2
    

@ti.kernel
def Particle2Grid():
    u_grid.fill(0.0)
    v_grid.fill(0.0)
    u_mss_grid.fill(0.0)
    v_mss_grid.fill(0.0)

    for p in pos_particle:
        coord_x, coord_y = pos_particle[p][0] * n_grid, pos_particle[p][1] * n_grid
        idx, idy = int(coord_x + 0.5), int(coord_y)
        affine = C[p]

        for i, j in ti.ndrange((-1, 2), (-1, 2)):
            new_x, new_y = idx + i, idy + j
            if 0 <= new_x <= n_grid and 0 <= new_y < n_grid:
                new_pos = ti.Vector([float(idx), float(idy) + 0.5]) + ti.Vector([i, j])
                weight = N(coord_x - new_pos[0]) * N(coord_y - new_pos[1])

                dpos = (new_pos - ti.Vector([coord_x, coord_y])) * dx
                u_grid[new_x, new_y] += (vel_particle[p][0] + (affine @ dpos)[0]) * weight
                # u_grid[new_x, new_y] += vel_particle[p][0] * weight
                u_mss_grid[new_x, new_y] += mss_particle * weight

    
    for p in pos_particle:
        coord_x, coord_y = pos_particle[p][0] * n_grid, pos_particle[p][1] * n_grid
        idx, idy = int(coord_x), int(coord_y + 0.5)
        affine = C[p]

        for i, j in ti.ndrange((-1, 2), (-1, 2)):
            new_x, new_y = idx + i, idy + j
            if 0 <= new_x < n_grid and 0 <= new_y <= n_grid:
                new_pos = ti.Vector([float(idx) + 0.5, float(idy)]) + ti.Vector([i, j])
                weight = N(coord_x - new_pos[0]) * N(coord_y - new_pos[1])

                dpos = (new_pos - ti.Vector([coord_x, coord_y])) * dx
                v_grid[new_x, new_y] += (vel_particle[p][1]  + (affine @ dpos)[1]) * weight
                # v_grid[new_x, new_y] += vel_particle[p][1] * weight
                v_mss_grid[new_x, new_y] += mss_particle * weight

    for i, j in u_grid:
        if u_mss_grid[i, j] > 0.0 :
            u_grid[i, j] /= u_mss_grid[i, j]
        if i == 0 or i == n_grid:
            u_grid[i, j] = 0.0

    for i, j in v_grid:
        if v_mss_grid[i, j] > 0.0 :
            v_grid[i, j] /= v_mss_grid[i, j]
        if j == 0 or j == n_grid:
            v_grid[i, j] = 0.0

    

@ti.kernel
def PreProjection():
    # calculate div_vel_grid
    for i, j in div_vel_grid:
        if marker[i, j] == 1:
            div_x = (u_grid[i+1, j] - u_grid[i, j]) / dx
            div_y = (v_grid[i, j+1] - v_grid[i, j]) / dx
            div_vel_grid[i, j] = div_x + div_y
    

@ti.kernel
def PostProjection():
    for i, j in u_grid:
        if i == 0 or i == n_grid:
            u_grid[i, j] = 0
        else:
            u_grid[i, j] += (prs_grid[i-1, j] - prs_grid[i, j]) * dt / (rho * dx)
    
    for i, j in v_grid:
        if j == 0 or j == n_grid:
            v_grid[i, j] = 0
        else:
            v_grid[i, j] += (prs_grid[i, j-1] - prs_grid[i, j]) * dt / (rho * dx)


@ti.kernel
def Grid2Particle():
    C.fill(0.0)

    for p in pos_particle:
        coord_x, coord_y = pos_particle[p][0] * n_grid, pos_particle[p][1] * n_grid
        idx, idy = int(coord_x + 0.5), int(coord_y)
        new_u = 0.0

        for i, j in ti.ndrange((-1, 2), (-1, 2)):
            new_x, new_y = idx + i, idy + j
            if 0 <= new_x <= n_grid and 0 <= new_y < n_grid:
                new_pos = ti.Vector([float(idx), float(idy) + 0.5]) + ti.Vector([i, j])
                weight = N(coord_x - new_pos[0]) * N(coord_y - new_pos[1])
                new_u += u_grid[new_x, new_y] * weight
                
                dpos = new_pos - ti.Vector([coord_x, coord_y])
                C[p] += 4 * weight * ti.Vector([u_grid[new_x, new_y], 0]).outer_product(dpos) * n_grid

        vel_particle[p][0] = new_u * damp

    for p in pos_particle:
        coord_x, coord_y = pos_particle[p][0] * n_grid, pos_particle[p][1] * n_grid
        idx, idy = int(coord_x), int(coord_y + 0.5)
        new_v = 0.0

        for i, j in ti.ndrange((-1, 2), (-1, 2)):
            new_x, new_y = idx + i, idy + j
            if 0 <= new_x < n_grid and 0 <= new_y <= n_grid:
                new_pos = ti.Vector([float(idx) + 0.5, float(idy)]) + ti.Vector([i, j])
                weight = N(coord_x - new_pos[0]) * N(coord_y - new_pos[1])
                new_v += v_grid[new_x, new_y] * weight

                dpos = new_pos - ti.Vector([coord_x, coord_y])
                C[p] += 4 * weight * ti.Vector([0, v_grid[new_x, new_y]]).outer_product(dpos) * n_grid

        vel_particle[p][1] = new_v * damp

    for p in vel_particle:
        pos_particle[p] += vel_particle[p] * dt
        Boundry(p)


@ti.kernel
def MarkCell():
    for i, j in marker:
        if not (marker[i, j] == 2):
            marker[i, j] = 0
    
    for p in pos_particle:
        idx, idy = int(pos_particle[p][0] / dx), int(pos_particle[p][1] / dx)
        if not (marker[idx, idy] == 2):
            marker[idx, idy] = 1


def SolvePressure():
    scale_A = dt / (rho * dx * dx)
    scale_b = 1 / dx

    solver.system_init(scale_A, scale_b)
    solver.solve(500)

    prs_grid.copy_from(solver.x)


@ti.kernel
def enforce_boundary():
    # u solid
    for i, j in u_grid:
        if i == 0 or i == n_grid or marker[i - 1, j] == 2 or marker[i, j] == 2:
            u_grid[i, j] = 0.0

    # v solid
    for i, j in v_grid:
        if j == 0 or j == n_grid or marker[i, j - 1] == 2 or marker[i, j] == 2:
            v_grid[i, j] = 0.0


@ti.kernel
def ApplyGravity():
    # update gravity
    for i, j in v_grid:
        v_grid[i, j] += gravity[0][1] * dt

    
@ti.kernel
def CntValidParticles() -> int:
    result = 0
    for i in pos_particle:
        if not (pos_particle[i][0] != pos_particle[i][0] or pos_particle[i][1] != pos_particle[i][1]):
            result += 1
        
    return result


def main():
    initiate()
    gui = ti.GUI("APIC")
    
    while gui.running:
        gui.get_event()
        if gui.is_pressed('w'):
            gravity[0] = ti.Vector([0, 9.8])
        elif gui.is_pressed('s'):
            gravity[0] = ti.Vector([0, -9.8])
        elif gui.is_pressed('a'):
            gravity[0] = ti.Vector([-9.8, 0])
        elif gui.is_pressed('d'):
            gravity[0] = ti.Vector([9.8, 0])

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
            print(CntValidParticles())

        gui.circles(pos_particle.to_numpy(), radius=2)
        gui.show()


if __name__ == "__main__":
    main()