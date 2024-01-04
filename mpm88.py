# MPM-MLS in 88 lines of Taichi code, originally created by @yuanming-hu
import taichi as ti

ti.init(arch=ti.gpu)

n_particles = 8192
n_grid = 128
dx = 1 / n_grid
dt = 2e-4

p_rho = ti.field(float, n_particles)
p_vol = (dx * 0.5) ** 2
p_mass = ti.field(float, n_particles)
gravity = ti.Vector.field(2, float, shape=1)
bound = 3
E = 400

x = ti.Vector.field(2, float, n_particles)
v = ti.Vector.field(2, float, n_particles)
C = ti.Matrix.field(2, 2, float, n_particles)
J = ti.field(float, n_particles)

grid_v = ti.Vector.field(2, float, (n_grid, n_grid))
grid_m = ti.field(float, (n_grid, n_grid))


@ti.kernel
def substep():
    for i, j in grid_m:
        grid_v[i, j] = [0, 0]
        grid_m[i, j] = 0
    for p in x:
        Xp = x[p] / dx
        base = int(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        stress = -dt * 4 * E * p_vol * (J[p] - 1) / dx**2
        affine = ti.Matrix([[stress, 0], [0, stress]]) + p_mass[p] * C[p]
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            dpos = (offset - fx) * dx
            weight = w[i].x * w[j].y
            grid_v[base + offset] += weight * (p_mass[p] * v[p] + affine @ dpos)
            grid_m[base + offset] += weight * p_mass[p]
    for i, j in grid_m:
        if grid_m[i, j] > 0:
            grid_v[i, j] /= grid_m[i, j]
        grid_v[i, j] += dt * gravity[0]
        if i < bound and grid_v[i, j].x < 0:
            grid_v[i, j].x = 0
        if i > n_grid - bound and grid_v[i, j].x > 0:
            grid_v[i, j].x = 0
        if j < bound and grid_v[i, j].y < 0:
            grid_v[i, j].y = 0
        if j > n_grid - bound and grid_v[i, j].y > 0:
            grid_v[i, j].y = 0
    for p in x:
        Xp = x[p] / dx
        base = int(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        new_v = ti.Vector.zero(float, 2)
        new_C = ti.Matrix.zero(float, 2, 2)
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            dpos = (offset - fx) * dx
            weight = w[i].x * w[j].y
            g_v = grid_v[base + offset]
            new_v += weight * g_v
            new_C += 4 * weight * g_v.outer_product(dpos) / dx**2
        v[p] = new_v
        x[p] += dt * v[p]
        J[p] *= 1 + dt * new_C.trace()
        C[p] = new_C


@ti.kernel
def init():
    for i in range(n_particles):
        x[i] = [ti.random() * 0.4 + 0.2, ti.random() * 0.4 + 0.2]
        v[i] = [0, -1]
        J[i] = 1
        if i*2 < n_particles:
            p_rho[i] = 0.8
        else:
            p_rho[i] = 1.0
        
        p_mass[i] = p_vol * p_rho[i]


init()
gui = ti.GUI("MPM88")
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

    for s in range(50):
        substep()
    gui.clear(0x112F41)
    show_x = x.to_numpy()
    show_water = show_x[:4096]
    show_oil = show_x[4096:]
    gui.circles(show_water, radius=1.5, color=0xFF0000)
    gui.circles(show_oil, radius=1.5, color=0x00FF00)
    gui.show()