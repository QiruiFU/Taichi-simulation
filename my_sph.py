import taichi as ti
import taichi.math as tm

ti.init(arch=ti.cpu)

n = 1000
rho_0 = 1.0
h = 1.1
mass = 1.0
g = ti.Vector.field(2, float, shape=1)
damp = 0.999

boundX = 40.0
boundY = 40.0
waterPosX = 0.01 * boundX
waterPosY = 0.01 * boundY

frame = 100
substep = 5

k1 = 0.5
k2 = 7.0
k3 = 30.0

dt = 1.0 / (frame*substep)

vel = ti.Vector.field(2, float, shape=n)
pos = ti.Vector.field(2, float, shape=n)
acc = ti.Vector.field(2, float, shape=n)
rho = ti.field(float, shape=n)
prs = ti.field(float, shape=n)

# cell
cellSize = 4.0
numCellX = ti.ceil(boundX / cellSize)
numCellY = ti.ceil(boundY / cellSize)
numCell = numCellX * numCellY

ParNum = ti.field(int, shape = int(numCell))
Particals = ti.field(int, shape = (n, n))
NeiNum = ti.field(int, shape = n)
neighbor = ti.field(int, shape = (n, n))


@ti.func
def W(r) -> float:
    res = 0.0
    if 0 < r and r < h:
        x = (h*h - r*r) / (h**3)
        res = 315.0 / 64.0 / tm.pi * x * x * x
    return res


@ti.func
def DW(r): 
    res = ti.Vector([0.0, 0.0])
    r_len = r.norm()
    if 0 < r_len and r_len < h:
        x = (h - r_len) / (h * h * h)
        g_factor = -45.0 / tm.pi * x * x
        res = r * g_factor / r_len
    return res
    

@ti.func
def boundry(idx:int):
    eps = 0.5
    if pos[idx][0] > boundX - eps:
        pos[idx][0] = boundX - eps
        if vel[idx][0] > 0.0:
            vel[idx][0] = - 0.999 * vel[idx][0]
    
    if pos[idx][0] < eps:
        pos[idx][0] = eps
        if vel[idx][0] < 0.0:
            vel[idx][0] = - 0.999 * vel[idx][0]

    if pos[idx][1] > boundY - eps:
        pos[idx][1] = boundY - eps
        if vel[idx][1] > 0.0:
            vel[idx][1] = - 0.999 * vel[idx][1]

    if pos[idx][1] < eps:
        pos[idx][1] = eps
        if vel[idx][1] < 0.0:
            vel[idx][1] = - 0.999 * vel[idx][1]


@ti.kernel
def neighbor_search():
    NeiNum.fill(0)
    ParNum.fill(0)
    Particals.fill(0)
    neighbor.fill(0)

    for i in pos:
        idx = int(pos[i][0]/cellSize-0.5) + int(pos[i][1]/cellSize-0.5) * numCellX
        k = ti.atomic_add(ParNum[int(idx)], 1)
        Particals[int(idx), k] = i

    for i in pos:
        idx_x = int(pos[i][0]/cellSize - 0.5)
        idx_y = int(pos[i][1]/cellSize - 0.5)
        kk = 0
        for j in range(9):
            dx = ti.Vector([1, 1, 0, -1, -1, -1, 0, 1, 0])
            dy = ti.Vector([0, 1, 1, 1, 0, -1, -1, -1, 0])
            new_x = idx_x + dx[j]
            new_y = idx_y + dy[j]
            if new_x<numCellX and new_x>=0 and new_y<numCellY and new_y>=0:
                new_idx = int(new_x) + int(new_y * numCellX)
                cnt = ParNum[new_idx]
                for t in range(cnt):
                    nei = Particals[new_idx, t]
                    if nei!=i and (pos[nei]-pos[i]).norm() < 1.1*h:
                        neighbor[i, kk] = nei
                        kk += 1
        NeiNum[i] = kk


@ti.kernel
def init():
    g[0] = ti.Vector([0.0, -9.8])
    num = int(ti.sqrt(n))
    for i in range(n):
        posx = (i % num) * 0.65
        posy = (i // num) * 0.65
        pos[i] = ti.Vector([waterPosX + posx, waterPosY + posy])


@ti.kernel
def cal_gravity():
    for i in vel:
        acc[i] = g[0]


@ti.kernel
def cal_press():
    for i in rho:
        rho[i] = 0.0
        for nei in range(NeiNum[i]):
            j = neighbor[i, nei]
            r = (pos[i] - pos[j]).norm()
            rho[i] += mass * W(r)
        if rho[i] < 1e-5:
            rho[i] = rho_0

    for i in prs:
        density = ti.max(rho[i], rho_0)
        # prs[i] = k1 * (ti.pow(density/rho_0, k2) - 1.0)
        prs[i] = k3 * (density - rho_0)

    for i in acc:
        for nei in range(NeiNum[i]):
            j = neighbor[i, nei]
            gradW = DW(pos[i]-pos[j])
            res = mass * (prs[i]/rho[i]**2 + prs[j]/rho[j]**2) * gradW
            acc[i] -= res


@ti.kernel
def advect():
    for i in vel:
        vel[i] *= damp
        vel[i] += dt * acc[i]
        pos[i] += dt * vel[i]
        boundry(i)


if __name__ == '__main__':
    init()
    gui = ti.GUI('SPH', res = (500, 500))
    while gui.running:

        gui.get_event()
        if gui.is_pressed('w'):
            g[0] = ti.Vector([0, 9.8])
        elif gui.is_pressed('s'):
            g[0] = ti.Vector([0, -9.8])
        elif gui.is_pressed('a'):
            g[0] = ti.Vector([-9.8, 0])
        elif gui.is_pressed('d'):
            g[0] = ti.Vector([9.8, 0])

        for _ in range(substep):
            neighbor_search()
            cal_gravity()
            cal_press()
            # print(rho[111], rho[222], rho[333], rho[444])
            advect()
        
        pos_show = pos.to_numpy()
        pos_show[:, 0] *= 1.0 / boundX
        pos_show[:, 1] *= 1.0 / boundY
        gui.circles(pos_show, radius=3)
        gui.show()
    