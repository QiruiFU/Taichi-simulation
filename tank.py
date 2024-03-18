import taichi as ti
import taichi.math as tm
import math

ti.init(arch=ti.gpu)

# boundary
boundX = 50.0
boundY = 100.0

# Wall
wallNumX = int(boundX // 0.4) - 5
wallNumY = int(boundY // 0.4) - 5
wallNum = wallNumX * 3 + (wallNumY - 3) * 6

cur_n = ti.field(int, shape=())
fluid_n = 10000
total_num = fluid_n + wallNum
phase = 1
h = 1.1
g = ti.Vector.field(2, float, shape=1)
damp = 0.9995
tao = 1e-4

miscible = False

frame = 60
substep = 20

k3 = 40.0

dt = 1.0 / (frame*substep)

vel = ti.Vector.field(2, float, shape=total_num)
drift_vel = ti.Vector.field(2, float, shape=(total_num, phase))
pos = ti.Vector.field(2, float, shape=total_num)
acc = ti.Vector.field(2, float, shape=total_num)
prs = ti.field(float, shape=total_num) # prs_k = prs_m
rho_m = ti.field(float, shape=total_num) # rho_m of particle
rho_bar = ti.field(float, shape=total_num) # interpolated rho
rho_0 = ti.field(float, shape=phase) # rho_0 for all phases
alpha = ti.field(float, shape=(total_num, phase))

# cell
cellSize = 4.0
numCellX = ti.ceil(boundX / cellSize)
numCellY = ti.ceil(boundY / cellSize)
numCell = numCellX * numCellY

ParNum = ti.field(int, shape = int(numCell))
Particles = ti.field(int, shape = (int(numCell), total_num))
NeiNum = ti.field(int, shape = total_num)
neighbor = ti.field(int, shape = (total_num, total_num))

# rendering
palette = ti.field(int, shape = total_num)


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
    Particles.fill(0)
    neighbor.fill(0)

    for i in range(wallNum+cur_n[None]):
        idx = int(pos[i][0]/cellSize-0.5) + int(pos[i][1]/cellSize-0.5) * numCellX
        k = ti.atomic_add(ParNum[int(idx)], 1)
        Particles[int(idx), k] = i

    for i in range(wallNum+cur_n[None]):
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
                    nei = Particles[new_idx, t]
                    if nei!=i and (pos[nei]-pos[i]).norm() < 1.1*h:
                        neighbor[i, kk] = nei
                        kk += 1
        NeiNum[i] = kk


@ti.kernel
def init():
    rho_0[0] = 1.0  # water
    # rho_0[1] = 0.5  # oil
    g[0] = ti.Vector([0.0, -9.8])
    # mid = fluid_n
    # num = int(tm.sqrt(mid))

    # for i in range(mid):
    #     posx = (i % num) * 0.65
    #     posy = (i // num) * 0.65
    #     pos[i] = ti.Vector([0.3*boundX + posx, 0.5*boundY + posy])        
    #     alpha[i, 0] = 1.0
    #     # alpha[i, 1] = 0.0

    # for i in range(mid, fluid_n):
    #     j = i - mid
    #     posx = (j % num) * 0.65
    #     posy = (j // num) * 0.65
    #     pos[i] = ti.Vector([0.3*boundX + posx, 0.1*boundY + posy])        
    #     alpha[i, 0] = 0.0
    #     alpha[i, 1] = 1.0
    
    for i in range(wallNumX):
        pos[3*i] = ti.Vector([(i+1) * 0.4, 0.4])
        pos[3*i+1] = ti.Vector([(i+1) * 0.4, 0.8])
        pos[3*i+2] = ti.Vector([(i+1) * 0.4, 1.2])
    
    for i in range(wallNumY-3):
        pos[wallNumX*3+6*i] = ti.Vector([0.4, (i+4) * 0.4])
        pos[wallNumX*3+6*i+1] = ti.Vector([0.8, (i+4) * 0.4])
        pos[wallNumX*3+6*i+2] = ti.Vector([1.2, (i+4) * 0.4])
        pos[wallNumX*3+6*i+3] = ti.Vector([(wallNumX-2)*0.4, (i+4) * 0.4])
        pos[wallNumX*3+6*i+4] = ti.Vector([(wallNumX-1)*0.4, (i+4) * 0.4])
        pos[wallNumX*3+6*i+5] = ti.Vector([(wallNumX-0)*0.4, (i+4) * 0.4])


@ti.kernel
def cal_press():
    for i in rho_m:
        rho_m[i] = 0.0
        for ph in range(phase):
            rho_m[i] += alpha[i, ph] * rho_0[ph]
    
    for i in rho_bar: # we can assume V=1
        rho_bar[i] = 0.0
        for nei in range(NeiNum[i]):
            j = neighbor[i, nei]
            if j >= wallNum: # particle
                rho_bar[i] += rho_m[j] * W((pos[i] - pos[j]).norm())
            else: # Wall
                rho_bar[i] += rho_0[0] * W((pos[i] - pos[j]).norm())

        if rho_bar[i] < 1e-6:
            rho_bar[i] = rho_m[i]
    
    for i in prs:
        density = ti.max(rho_bar[i], rho_m[i])
        prs[i] = k3 * (density - rho_m[i])


@ti.kernel
def cal_drift():
    for i, k in drift_vel:
        first_term = (g[0] - acc[i]) * tao
        coef = rho_0[k]
        for ph in range(phase):
            coef -= alpha[i, ph] * rho_0[ph] * rho_0[ph] / rho_m[i]

        first_term *= coef
        second_term = ti.Vector([0.0, 0.0])
        for ph in range(phase):
            prs_grad = ti.Vector([0.0, 0.0])
            for nei in range(NeiNum[i]):
                j = neighbor[i, nei]
                if j >= wallNum:
                    if miscible:
                        prs_grad += rho_m[j] * (alpha[j, k] * prs[j] - alpha[i, k] * prs[i]) * DW(pos[i] - pos[j]) / rho_bar[j]
                    else:
                        prs_grad += rho_m[j] * (prs[j] - prs[i]) * DW(pos[i] - pos[j]) / rho_bar[j]

            second_term -= alpha[i, ph] * rho_0[ph] * prs_grad / rho_m[i]
            if ph==i:
                second_term += prs_grad
        
        second_term *= tao
        drift_vel[i, k] = first_term - second_term


@ti.kernel
def adv_alpha(): # formula 17, 18
    for i, k in alpha:
        first_term = 0.0
        for nei in range(NeiNum[i]):
            j = neighbor[i, nei]
            if j >= wallNum:
                temp1 = rho_m[j] * (alpha[i, k] + alpha[j, k]) / (2.0 * rho_bar[j])
                temp2 = (vel[j] - vel[i]).dot(DW(pos[i] - pos[j]))
                first_term += temp1 * temp2

        second_term = 0.0
        for nei in range(NeiNum[i]):
            j = neighbor[i, nei]
            if j >= wallNum:
                temp1 = rho_m[j] / rho_bar[j]
                temp2 = (alpha[j, k] * drift_vel[j, k] + alpha[i, k] * drift_vel[i, k]).dot(DW(pos[i] - pos[j]))
                second_term += temp1 * temp2

        alpha[i, k] -= (first_term + second_term) * dt
    

@ti.kernel
def check_alpha():
    for i in range(fluid_n):
        tot = 0.0
        for ph in range(phase):
            if alpha[i, ph] > 0:
                tot += alpha[i, ph]

        del_p = 0.0
        if tot < 1e-6:
            for ph in range(phase):
                cur = alpha[i, ph]
                alpha[i, ph] = 1 / phase
                del_p -= k3 * rho_0[ph] * (alpha[i, ph] - cur)
        else:
            for ph in range(phase):
                cur = alpha[i, ph]
                if alpha[i, ph] < 0:
                    alpha[i, ph] = 0.0
                else:
                    alpha[i, ph] /= tot
                del_p -= k3 * rho_0[ph] * (alpha[i, ph] - cur)
        
        prs[i] += del_p


@ti.kernel
def cal_acc():
    for i in acc:
        acc[i] = g[0]
        prs_grad = ti.Vector([0.0, 0.0])
        Tdm_grad = ti.Vector([0.0, 0.0])

        for nei in range(NeiNum[i]):
            j = neighbor[i, nei]
            if j >= wallNum: # partical
                prs_grad += rho_m[j] * (prs[i] + prs[j]) / (2 * rho_bar[j]) * DW(pos[i] - pos[j])
            else: # Wall
                prs_grad += rho_0[0] * (prs[i] + prs[i]) / (2 * rho_0[0]) * DW(pos[i] - pos[j])

        for nei in range(NeiNum[i]):
            j = neighbor[i, nei]
            if j >= wallNum:
                temp = ti.Vector([0.0, 0.0])
                for k in range(phase):
                    temp1 = alpha[j, k] * drift_vel[j, k] * (drift_vel[j, k].dot(DW(pos[i] - pos[j])))
                    temp2 = alpha[i, k] * drift_vel[i, k] * (drift_vel[i, k].dot(DW(pos[i] - pos[j])))
                    temp += (temp1 + temp2) * rho_0[k]

                Tdm_grad -= (rho_m[j] / rho_bar[j]) * temp
        
        acc[i] += (Tdm_grad - prs_grad) / rho_m[i]

            
@ti.kernel
def advect():
    for i in vel:
        if i >= wallNum:
            vel[i] *= damp
            vel[i] += dt * acc[i]
            pos[i] += dt * vel[i]
            boundry(i)


@ti.kernel
def pre_render():
    for i in pos:
        if i >= wallNum:
            clr = int(alpha[i, 0] * 0xFF) * 0x010000 
            # + int(alpha[i, 1] * 0xFF) * 0x000100
            palette[i] = clr
        else:
            palette[i] = 0xFFFFFF


if __name__ == '__main__':
    init()
    cur_frame = 0
    gui = ti.GUI('SPH', res = (400, 800))
    while gui.running:
        
        if cur_n[None] < fluid_n :
            cur_n[None] += 5
            for idx in range(cur_n[None]-5, cur_n[None]):
                pos[wallNum+idx] = ti.Vector([0.05 * boundX, 0.8 * boundY + (cur_n[None] - idx) * 0.65])
                vel[wallNum+idx] = ti.Vector([40.0, 0.0])
                alpha[wallNum+idx, 0] = 1.0
        
        for _ in range(substep):
            neighbor_search()
            cal_press()
            cal_drift()
            adv_alpha()
            check_alpha()
            cal_acc()
            advect()

        f = open(f"out/jsonfile/water_{cur_frame}.json", "w")
        f.write("[\n")
        for i in range(wallNum, wallNum+cur_n[None]-1):
            f.write(f"[{pos[i][0]}, {pos[i][1]}, 0],\n")
        f.write(f"[{pos[wallNum+cur_n[None]-1][0]}, {pos[wallNum+cur_n[None]-1][1]}, 0]\n")
        f.write("]")
        cur_frame += 1
        # pre_render()
        # pos_show = pos.to_numpy()
        # palette_show = palette.to_numpy()
        # pos_show[:, 0] *= 1.0 / boundX
        # pos_show[:, 1] *= 1.0 / boundY
        # roll = math.ceil((cur_n[None] + wallNum) / 255)
        # for i in range(roll): # you can render up to 255 circles at one time
        #     left = i * 255
        #     right = min(cur_n[None] + wallNum, (i+1)*255)
        #     gui.circles(pos_show[left:right, :], radius=3, palette=palette_show[left:right], palette_indices=[i for i in range(right-left)])
        # gui.show()
    