import taichi as ti
import taichi.math as tm
import math

ti.init(arch=ti.gpu)

visualization = 1

# boundary
boundX = 25.0 * ti.sqrt(10.0)
boundY = 40.0 * ti.sqrt(10.0)

# Wall
wallNumX = int(boundX // 0.5) - 5
wallNumY = int(boundY // 0.5) - 5
wallNum = wallNumX * 3 + (wallNumY - 3) * 6

fluid_n = 5000
total_num = fluid_n + wallNum
phase = 2
h = 1.1
g = ti.Vector.field(2, float, shape=1)
damp = 0.999
tao = 1e-4

miscible = False

frame = 100
substep = 10

k1 = 0.5
k2 = 7.0
k3 = 40.0

dt = 1.0 / (frame*substep)

vel = ti.Vector.field(2, float, shape=fluid_n)
drift_vel = ti.Vector.field(2, float, shape=(fluid_n, phase))
pos = ti.Vector.field(2, float, shape=total_num)
acc = ti.Vector.field(2, float, shape=fluid_n)
prs = ti.field(float, shape=fluid_n) # prs_k = prs_m
rho_m = ti.field(float, shape=fluid_n) # rho_m of particle
rho_bar = ti.field(float, shape=fluid_n) # interpolated rho
rho_0 = ti.field(float, shape=phase) # rho_0 for all phases
alpha = ti.field(float, shape=(fluid_n, phase))

# cell
cellSize = 4.0
numCellX = ti.ceil(boundX / cellSize)
numCellY = ti.ceil(boundY / cellSize)
numCell = numCellX * numCellY

ParNum = ti.field(int, shape = int(numCell))
Particles = ti.field(int, shape = (int(numCell), total_num))
NeiNum = ti.field(int, shape = fluid_n)
neighbor = ti.field(int, shape = (fluid_n, total_num))

# rendering
render_pos = ti.Vector.field(3, float, shape = fluid_n)
palette = ti.field(int, shape = total_num)
ply_palette = ti.Vector.field(3, float, shape = fluid_n)


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

    for i in pos:
        idx = int(pos[i][0]/cellSize-0.5) + int(pos[i][1]/cellSize-0.5) * numCellX
        k = ti.atomic_add(ParNum[int(idx)], 1)
        Particles[int(idx), k] = i

    for i in range(fluid_n):
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
    rho_0[1] = 0.5  # oil
    g[0] = ti.Vector([0.0, -9.8])
    mid = fluid_n / 2
    num = int(tm.sqrt(mid))

    for i in range(mid):
        posx = (i % num) * 0.65
        posy = (i // num) * 0.65
        pos[i] = ti.Vector([0.4*boundX + posx, 0.05*boundY + posy])        
        alpha[i, 0] = 0.0
        alpha[i, 1] = 1.0

    for i in range(mid, fluid_n):
        j = i - mid
        posx = (j % num) * 0.65
        posy = (j // num) * 0.65
        pos[i] = ti.Vector([0.4*boundX + posx, 0.35*boundY + posy])        
        alpha[i, 0] = 1.0
        alpha[i, 1] = 0.0
    
    for i in range(wallNumX):
        pos[fluid_n+3*i] = ti.Vector([(i+1) * 0.5, 0.5])
        pos[fluid_n+3*i+1] = ti.Vector([(i+1) * 0.5, 1.0])
        pos[fluid_n+3*i+2] = ti.Vector([(i+1) * 0.5, 1.5])
    
    for i in range(wallNumY-3):
        pos[fluid_n+wallNumX*3+6*i] = ti.Vector([0.5, (i+4) * 0.5])
        pos[fluid_n+wallNumX*3+6*i+1] = ti.Vector([1.0, (i+4) * 0.5])
        pos[fluid_n+wallNumX*3+6*i+2] = ti.Vector([1.5, (i+4) * 0.5])
        pos[fluid_n+wallNumX*3+6*i+3] = ti.Vector([(wallNumX-2)*0.5, (i+4) * 0.5])
        pos[fluid_n+wallNumX*3+6*i+4] = ti.Vector([(wallNumX-1)*0.5, (i+4) * 0.5])
        pos[fluid_n+wallNumX*3+6*i+5] = ti.Vector([(wallNumX-0)*0.5, (i+4) * 0.5])


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
            if j < fluid_n: # particle
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
                if j < fluid_n:
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
            if j < fluid_n:
                temp1 = rho_m[j] * (alpha[i, k] + alpha[j, k]) / (2.0 * rho_bar[j])
                temp2 = (vel[j] - vel[i]).dot(DW(pos[i] - pos[j]))
                first_term += temp1 * temp2

        second_term = 0.0
        for nei in range(NeiNum[i]):
            j = neighbor[i, nei]
            if j < fluid_n:
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
            if j < fluid_n: # partical
                prs_grad += rho_m[j] * (prs[i] + prs[j]) / (2 * rho_bar[j]) * DW(pos[i] - pos[j])
            else: # Wall
                prs_grad += rho_0[0] * (prs[i] + prs[i]) / (2 * rho_0[0]) * DW(pos[i] - pos[j])

        for nei in range(NeiNum[i]):
            j = neighbor[i, nei]
            if j < fluid_n:
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
        vel[i] *= damp
        vel[i] += dt * acc[i]
        pos[i] += dt * vel[i]
        boundry(i)


@ti.kernel
def pre_render():
    for i in pos:
        if i < fluid_n:
            clr = int(alpha[i, 0] * 0xFF) * 0x010000 + int(alpha[i, 1] * 0xFF) * 0x000100
            palette[i] = clr
        else:
            palette[i] = 0xFFFFFF
    
    for i in ply_palette:
        ply_palette[i] = ti.Vector([alpha[i, 0], alpha[i, 1], 0.0])
        render_pos[i] = ti.Vector([pos[i][0], pos[i][1], 0.0])

    
if __name__ == '__main__':
    init()
    gui = ti.GUI('SPH', res = (500, 800))

    cur_frame = 0

    while gui.running:
        for _ in range(10):
            neighbor_search()
            cal_press()
            cal_drift()
            adv_alpha()
            check_alpha()
            cal_acc()
            advect()
            pass

        pre_render()
        if visualization == 0:
            pos_show = pos.to_numpy()
            palette_show = palette.to_numpy()
            pos_show[:, 0] *= 1.0 / boundX
            pos_show[:, 1] *= 1.0 / boundY
            roll = math.ceil(total_num / 255)
            for i in range(roll): # you can render up to 255 circles at one time
                left = i * 255
                right = min(total_num, (i+1)*255)
                gui.circles(pos_show[left:right, :], radius=3, palette=palette_show[left:right], palette_indices=[i for i in range(right-left)])
            gui.show()
        else:
            series_prefix = "out/plyfile/water_.ply"
            np_pos = render_pos.to_numpy()
            np_palette = ply_palette.to_numpy()
            writer = ti.tools.PLYWriter(num_vertices = fluid_n)
            writer.add_vertex_pos(np_pos[:fluid_n, 0], np_pos[:fluid_n, 1], np_pos[:fluid_n, 2])
            writer.add_vertex_color(np_palette[:fluid_n, 0], np_palette[:fluid_n, 1], np_palette[:fluid_n, 2])
            writer.export_frame_ascii(cur_frame, series_prefix)

        cur_frame += 1
        print(cur_frame)
        if cur_frame == 3000 :
            exit()