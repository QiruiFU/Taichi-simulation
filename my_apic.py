import taichi as ti

ti.init(ti.cpu, debug=False)

n_grid = 128
n_particle = 8192
dx = 1 / n_grid

pos_particle = ti.Vector.field(2, float, shape=n_particle)
vel_particle = ti.Vector.field(2, float, shape=n_particle)
mss_particle = 1.0
vel_grid = ti.Vector.field(2, float, shape=(n_grid, n_grid))
mss_grid = ti.field(float, shape=(n_grid, n_grid))
prs_grid = ti.field(float, shape=(n_grid, n_grid))
div_grid = ti.field(float, shape=(n_grid, n_grid))
new_prs = ti.field(float, shape=(n_grid, n_grid))
gravity = ti.Vector.field(2, float, shape=1)

dt = 0.001
substep = 20
damp = 0.99

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
def Boundary(idx:int):
    eps = 1e-5
    if pos_particle[idx][0] < eps :
        pos_particle[idx][0] = eps
        # if vel[idx][0] < 0.0 :
        #     vel[idx][0] *= -0.7
    
    if pos_particle[idx][1] < eps :
        pos_particle[idx][1] = eps
        # if vel[idx][1] < 0.0 :
        #     vel[idx][1] *= -0.7
    
    if pos_particle[idx][0] > 1 - eps :
        pos_particle[idx][0] = 1 - eps
        # if vel[idx][0] > 0.0 :
        #     vel[idx][0] *= -0.7

    if pos_particle[idx][1] > 1 - eps :
        pos_particle[idx][1] = 1 - eps
        # if vel[idx][1] > 0.0 :
        #     vel[idx][1] *= -0.7


@ti.func
def Corner(i: int, j: int):
    res = False
    if i==0 and j==0 :
        res = True
    if i==n_grid-1 and j==0:
        res = True
    if i==0 and j==n_grid-1:
        res = True
    if i==n_grid-1 and j==n_grid-1:
        res = True

    return res


@ti.func
def PressureIteraion():
    for i, j in ti.ndrange(n_grid, n_grid):
        new_prs[i, j] = (prs_grid[i+1, j] + prs_grid[i-1, j] + prs_grid[i, j+1] + prs_grid[i, j-1] - div_grid[i, j]) / 4

    for i, j in ti.ndrange(n_grid, n_grid):
        if Corner(i, j):
            prs_grid[i, j] = 0
        elif i==0:
            prs_grid[i, j] = new_prs[i+1, j]
        elif i==n_grid-1:
            prs_grid[i, j] = new_prs[n_grid-2, j]
        elif j==0:
            prs_grid[i, j] = new_prs[i, j+1]
        elif j==n_grid-1:
            prs_grid[i, j] = new_prs[i, n_grid-2]
        else:
            prs_grid[i, j] = new_prs[i, j]
        

@ti.func
def SolvePressure():
    for _a in range(1):
        for _b in range(20):
            PressureIteraion()
            if _b==18 or _b==19 :
                print(prs_grid[30, 30])


@ti.kernel
def initiate():
    for i in pos_particle:
        pos_particle[i] = [ti.random() * 0.4 + 0.2, ti.random() * 0.4 + 0.2]
        vel_particle[i] = [0.0, 0.0]


@ti.kernel
def Particle2Grid():
    for i, j in vel_grid:
        vel_grid[i, j] = ti.Vector([0.0, 0.0])
        mss_grid[i, j] = 0.0
        prs_grid[i, j] = 0.0
    
    for p in pos_particle:
        vel_particle[p] += dt * gravity[0]
        idx_x = pos_particle[p][0] / dx
        idx_y = pos_particle[p][1] / dx
        for i in ti.static(range(-1, 2)):
            for j in ti.static(range(-1, 2)):
                new_x = int(idx_x) + i
                new_y = int(idx_y) + j
                if new_x>=0 and new_y>=0 and new_x<n_grid and new_y<n_grid:
                    center_pos = ti.Vector([float(new_x)+0.5, float(new_y)+0.5])
                    weight = N(idx_x - center_pos[0]) * N(idx_y - center_pos[1])
                    vel_grid[new_x, new_y] += vel_particle[p] * weight
                    mss_grid[new_x, new_y] += mss_particle * weight
    
    for i, j in vel_grid:
        if mss_grid[i, j] > 0.0 :
            vel_grid[i, j] /= mss_grid[i, j]


@ti.kernel
def Projection():
    for i, j in div_grid:
        div_grid[i, j] = (vel_grid[i+1, j][0] + vel_grid[i, j+1][1] - vel_grid[i-1, j][0] - vel_grid[i, j-1][1]) * 0.5
        prs_grid[i, j] = 0

    SolvePressure()

    for i, j in vel_grid:
        if i==0 or i==n_grid-1 or j==0 or j==n_grid-1:
            vel_grid[i, j] = ti.Vector([0, 0])
        else:
            # TAG: why without dt
            vel_grid[i, j] -= 0.5 * 0.01 * ti.Vector([prs_grid[i+1,j]-prs_grid[i-1,j], prs_grid[i, j+1]-prs_grid[i, j-1]])



@ti.kernel
def Grid2Particle():
    for p in pos_particle:
        idx_x = pos_particle[p][0] / dx
        idx_y = pos_particle[p][1] / dx
        new_vel = ti.Vector([0.0, 0.0])
        for i in ti.static(range(-1, 2)):
            for j in ti.static(range(-1, 2)):
                new_x = int(idx_x) + i
                new_y = int(idx_y) + j
                if new_x>=0 and new_y>=0 and new_x<n_grid and new_y<n_grid:
                    center_pos = ti.Vector([float(new_x)+0.5, float(new_y)+0.5])
                    weight = N(idx_x - center_pos[0]) * N(idx_y - center_pos[1])
                    new_vel += weight * vel_grid[new_x, new_y]
        
        vel_particle[p] = new_vel * damp
        pos_particle[p] += vel_particle[p] * dt
        Boundary(p)


def main():
    initiate()
    gui = ti.GUI("PIC")
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

        for s in range(substep):
            Particle2Grid()
            Projection()
            Grid2Particle()

        gui.circles(pos_particle.to_numpy(), radius=1.5)
        gui.show()
        

if __name__ == "__main__":
    main()