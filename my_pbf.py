import taichi as ti
import taichi.math as tm

ti.init(arch = ti.cpu)

# scene
box = (80.0, 40.0)
nx = 60
n = 20 * nx
dt = 1/30.0
gravity = ti.Vector.field(2, float, shape=1)

# PBF
h = 1.1
# rho_water = 1.0
# rho_oil = 0.8
# mass_water = 1.0
# mass_oil = 0.8
substep = 5
corr_deltaQ_coeff = 0.3
corrK = 0.001

# cell
cellSize = 4.0
numCellX = ti.ceil(box[0] / cellSize)
numCellY = ti.ceil(box[1] / cellSize)
numCell = numCellX * numCellY

ParNum = ti.field(int, shape = int(numCell))
Particals = ti.field(int, shape = (n, n))
NeiNum = ti.field(int, shape = n)
neighbor = ti.field(int, shape = (n, n))

rho = ti.field(float, shape=n)
lam = ti.field(float, shape=n)
pos = ti.Vector.field(2, float, shape=n)
vel = ti.Vector.field(2, float, shape=n)
dp = ti.Vector.field(2, float, shape=n)
oldp = ti.Vector.field(2, float, shape=n)
mass = ti.field(float, shape=n)
rho0 = ti.field(float, shape=n)
 
@ti.func
def poly6_value(s, h):
    result = 0.0
    if 0 < s and s < h:
        x = (h * h - s * s) / (h * h * h)
        result = 315.0 / 64.0 / tm.pi * x * x * x
    return result


@ti.func
def spiky_gradient(r, h):
    result = ti.Vector([0.0, 0.0])
    r_len = r.norm()
    if 0 < r_len and r_len < h:
        x = (h - r_len) / (h * h * h)
        g_factor = -45.0 / tm.pi * x * x
        result = r * g_factor / r_len
    return result


@ti.func
def scorr(r):
    x = poly6_value(r.norm(), h) / poly6_value(corr_deltaQ_coeff * h, h)
    # pow(x, 4)
    x = x * x
    x = x * x
    return (-corrK) * x


@ti.func
def boundary(idx:int):
    eps = 1e-5
    if pos[idx][0] < eps :
        pos[idx][0] = eps
        if vel[idx][0] < 0.0 :
            vel[idx][0] *= -0.7
    
    if pos[idx][1] < eps :
        pos[idx][1] = eps
        if vel[idx][1] < 0.0 :
            vel[idx][1] *= -0.7
    
    if pos[idx][0] > box[0] - eps :
        pos[idx][0] = box[0] - eps
        if vel[idx][0] > 0.0 :
            vel[idx][0] *= -0.7

    if pos[idx][1] > box[1] - eps :
        pos[idx][1] = box[1] - eps
        if vel[idx][1] > 0.0 :
            vel[idx][1] *= -0.7


@ti.kernel
def initiate():
    delta = h * 0.8
    corner = ti.Vector([(box[0]-delta*nx)*0.5, box[1]*0.15])
    for i in pos:
        dx = i % nx
        dy = i // nx
        pos[i] = corner + delta * (dx, dy)
        vel[i] = ti.Vector([0, 0])
        if i*2 < n :
            rho0[i] = 0.8
            mass[i] = 0.8
        else :
            rho0[i] = 1.0
            mass[i] = 1.0
        
        rho[i] = rho0[i]


@ti.kernel
def set_up():
    for i in vel:
        oldp[i] = pos[i]
        vel_now = vel[i] + gravity[0] * dt
        pos[i] += vel_now * dt
        boundary(i)
    

@ti.kernel
def limiting():
    # rho
    for i in rho:
        rho[i] = 0.0
        for nei in range(NeiNum[i]):
            j = neighbor[i, nei]
            s = (pos[i] - pos[j]).norm()
            rho[i]  += mass[j] * poly6_value(s, h)

    # lambda
    for i in lam:
        constrain = rho[i] / rho0[i] - 1.0
        sum_grad = 0.0
        C_i = ti.Vector([0.0, 0.0])
        for nei in range(NeiNum[i]):
            j = neighbor[i, nei]
            r = pos[i] - pos[j]
            C_p = -spiky_gradient(r, h) / rho0[i]
            sum_grad += C_p.dot(C_p)
            C_i -= C_p
        
        sum_grad += C_i.dot(C_i)
        lam[i] = - constrain / (sum_grad + 100.0)
    
    # delta_p
    for i in dp:
        dp[i] = ti.Vector([0.0, 0.0])
        for nei in range(NeiNum[i]):
            j = neighbor[i, nei]
            r = pos[i] - pos[j]
            dp[i] += (lam[i]+lam[j]+scorr(r)) * spiky_gradient(r, h)

        dp[i] /= rho0[i]
    
    for i in pos:
        pos[i] += dp[i]


@ti.kernel
def collision():
    for i in pos:
        vel[i] = (pos[i] - oldp[i]) / dt
        boundary(i)


@ti.kernel
def neighbor_search():
    NeiNum.fill(0)
    ParNum.fill(0)

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


def main():
    initiate()
    gui = ti.GUI("PBF", res=(800, 400))
    while gui.running :
        gui.get_event()
        if gui.is_pressed('w'):
            gravity[0] = ti.Vector([0, 9.8])
        elif gui.is_pressed('s'):
            gravity[0] = ti.Vector([0, -9.8])
        elif gui.is_pressed('a'):
            gravity[0] = ti.Vector([-9.8, 0])
        elif gui.is_pressed('d'):
            gravity[0] = ti.Vector([9.8, 0])

        set_up()
        neighbor_search()
        for _ in range(substep):
            limiting()

        collision()

        # render
        pos_show = pos.to_numpy()
        pos_show[:, 0] /= box[0]
        pos_show[:, 1] /= box[1]
        pos_oil = pos_show[:600]
        pos_water = pos_show[600:]
        gui.circles(pos_oil, radius=3, color=0xFF0000)
        gui.circles(pos_water, radius=3, color=0x00FF00)
        gui.show()

if __name__ == '__main__':
    main()