import taichi as ti
import taichi.math as tm
import numpy as np
import math

ti.init(arch=ti.gpu)
show_type = 0

# boundary
boundX = 50
boundY = 50
boundZ = 100

# Wall
# wallNumX = int(boundX // 0.5) - 5
# wallNumY = int(boundY // 0.5) - 5
# wallNum = wallNumX * 3 + (wallNumY - 3) * 6
wallNum = 0

fluid_n = 27000
total_num = fluid_n + wallNum
phase = 2
h = 1.5
g = ti.Vector.field(3, float, shape=1)
damp = 0.9993
tao = 1e-4

miscible = False

frame = 60
substep = 20

k3 = 100.0

dt = 1.0 / (frame*substep)

vel = ti.Vector.field(3, float, shape=fluid_n)
drift_vel = ti.Vector.field(3, float, shape=(fluid_n, phase))
pos = ti.Vector.field(3, float, shape=total_num)
acc = ti.Vector.field(3, float, shape=fluid_n)
prs = ti.field(float, shape=fluid_n) # prs_k = prs_m
rho_m = ti.field(float, shape=fluid_n) # rho_m of particle
rho_bar = ti.field(float, shape=fluid_n) # interpolated rho
rho_0 = ti.field(float, shape=phase) # rho_0 for all phases
alpha = ti.field(float, shape=(fluid_n, phase))

# cell
cellSize = 4.0
numCellX = int(ti.ceil(boundX / cellSize))
numCellY = int(ti.ceil(boundY / cellSize))
numCellZ = int(ti.ceil(boundZ / cellSize))
numCell = numCellX * numCellY * numCellZ

ParNum = ti.field(int, shape = (numCellX, numCellY, numCellZ))
Particles = ti.field(int, shape = (numCellX, numCellY, numCellZ, 4000))
NeiNum = ti.field(int, shape = fluid_n)
neighbor = ti.field(int, shape = (fluid_n, 200))

# rendering
palette = ti.Vector.field(3, int, shape = total_num)


@ti.func
def W(r:float) -> float:
    res = 0.0
    if 0 < r and r < h:
        x = (h*h - r*r) / (h**3)
        res = 315.0 / 64.0 / tm.pi * x * x * x
    return res


@ti.func
def DW(r) -> ti.Vector: 
    res = ti.Vector([0.0, 0.0, 0.0])
    r_len = r.norm()
    if 0 < r_len and r_len < h:
        x = (h - r_len) / (h * h * h)
        g_factor = -45.0 / tm.pi * x * x
        res = r * g_factor / r_len
    return res

    # res = ti.Vector([0.0, 0.0, 0.0])
    # r_len = r.norm()
    # if 0 < r_len and r_len < h:
    #     res = r
    #     res *= - 945 / (32 * tm.pi * (h ** 9.0))
    #     res *= (h * h - r_len * r_len) ** 2
    # return res


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

    if pos[idx][2] > boundZ - eps:
        pos[idx][2] = boundZ - eps
        if vel[idx][2] > 0.0:
            vel[idx][2] = - 0.999 * vel[idx][1]

    if pos[idx][2] < eps:
        pos[idx][2] = eps
        if vel[idx][2] < 0.0:
            vel[idx][2] = - 0.999 * vel[idx][1]


@ti.kernel
def neighbor_search():
    NeiNum.fill(0)
    ParNum.fill(0)
    Particles.fill(0)
    neighbor.fill(0)

    for i in pos:
        idx_x = int(pos[i][0] / cellSize - 0.5)
        idx_y = int(pos[i][1] / cellSize - 0.5)
        idx_z = int(pos[i][2] / cellSize - 0.5)
        k = ti.atomic_add(ParNum[idx_x, idx_y, idx_z], 1)
        Particles[idx_x, idx_y, idx_z, k] = i

    for i in range(fluid_n):
        idx_x = int(pos[i][0] / cellSize - 0.5)
        idx_y = int(pos[i][1] / cellSize - 0.5)
        idx_z = int(pos[i][2] / cellSize - 0.5)
        kk = 0
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                for dz in range(-1, 2):
                    new_x = idx_x + dx
                    new_y = idx_y + dy
                    new_z = idx_z + dz
                    if not(new_x < 0 or new_x >= numCellX or new_y < 0 or new_y >= numCellY or new_z < 0 or new_z >= numCellZ):
                        cnt = ParNum[new_x, new_y, new_z]
                        for t in range(cnt):
                            nei = Particles[new_x, new_y, new_z, t]
                            if nei!=i and (pos[nei]-pos[i]).norm() < 1.1*h:
                                neighbor[i, kk] = nei
                                kk += 1
        NeiNum[i] = kk


@ti.kernel
def init():
    rho_0[0] = 1.0  # water
    rho_0[1] = 0.5  # oil
    g[0] = ti.Vector([0.0, 0.0, -9.8])
    mid = fluid_n
    num = 30

    for i in range(mid):
        posz = (i // 900) * 1.2
        plane = i % 900
        posx = (plane % num) * 1.2
        posy = (plane // num) * 1.2
        pos[i] = ti.Vector([0.1*boundX + posx, 0.1*boundY + posy, 0.2*boundZ + posz])        
        alpha[i, 0] = 1.0
        alpha[i, 1] = 0.0

    for i in range(mid, fluid_n):
        j = i - mid
        posz = (j // 900) * 1.2
        plane = j % 900
        posx = (plane % num) * 1.2
        posy = (plane // num) * 1.2
        pos[i] = ti.Vector([0.1*boundX + posx, 0.1*boundY + posy, 0.6*boundZ + posz])        
        alpha[i, 0] = 0.0
        alpha[i, 1] = 1.0
    

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
        second_term = ti.Vector([0.0, 0.0, 0.0])
        for ph in range(phase):
            prs_grad = ti.Vector([0.0, 0.0, 0.0])
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
        prs_grad = ti.Vector([0.0, 0.0, 0.0])
        Tdm_grad = ti.Vector([0.0, 0.0, 0.0])

        for nei in range(NeiNum[i]):
            j = neighbor[i, nei]
            if j < fluid_n: # partical
                prs_grad += rho_m[j] * (prs[i] + prs[j]) / (2 * rho_bar[j]) * DW(pos[i] - pos[j])
            else: # Wall
                prs_grad += rho_0[0] * (prs[i] + prs[i]) / (2 * rho_0[0]) * DW(pos[i] - pos[j])

        for nei in range(NeiNum[i]):
            j = neighbor[i, nei]
            if j < fluid_n:
                temp = ti.Vector([0.0, 0.0, 0.0])
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
            if show_type == 0:
                palette[i][0] = int(alpha[i, 0] * 0xFF)
                palette[i][1] = int(alpha[i, 1] * 0xFF)
                palette[i][2] = 0
            elif show_type == 1 :
                ratio = (prs[i] + 30) / 130.0
                palette[i][0] = int(ratio * 0xFF)
                palette[i][1] = int((1-ratio) * 0xFF)
                palette[i][2] = 0
        else:
            palette[i] = ti.Vector([0xFF, 0xFF, 0xFF])


# @ti.kernel
# def classify_pos():
#     cnt_p0[None] = 0
#     cnt_p1[None] = 0
#     pos_p0.fill(0)
#     pos_p1.fill(0)

#     for i in vel:
#         if alpha[i, 0] > 0.5:
#             temp0 = ti.atomic_add(cnt_p0[None], 1)
#             pos_p0[temp0][0] = pos[i][0]
#             pos_p0[temp0][1] = pos[i][1]
#             pos_p0[temp0][2] = 0.0
#         else:
#             temp1 = ti.atomic_add(cnt_p1[None], 1)
#             pos_p1[temp1][0] = pos[i][0]
#             pos_p1[temp1][1] = pos[i][1]
#             pos_p1[temp1][2] = 0.0
    

if __name__ == '__main__':
    init()
    gui = ti.ui.Window('SPH', res = (700, 700))
    canvas = gui.get_canvas()
    canvas.set_background_color((1, 1, 1))
    scene = gui.get_scene()
    camera = ti.ui.Camera()

    cur_frame = 0

    while True:
        for _ in range(substep):
            neighbor_search()
            cal_press()
            cal_drift()
            adv_alpha()
            check_alpha()
            cal_acc()
            advect()

        # classify_pos()

        # series_prefix = "out/plyfile/water_.ply"
        # np_pos = pos.to_numpy()
        # writer0 = ti.tools.PLYWriter(num_vertices = fluid_n)
        # writer0.add_vertex_pos(np_pos[0:fluid_n, 0], np_pos[0:fluid_n, 1], np_pos[0:fluid_n, 2])
        # writer0.export_frame_ascii(cur_frame, series_prefix)
        # cur_frame += 1
        # print(cur_frame)
        # if cur_frame == 2400 :
        #     exit()

        # np_pos_0 = pos_p0.to_numpy()
        # writer0 = ti.tools.PLYWriter(num_vertices = cnt_p0[None])
        # writer0.add_vertex_pos(np_pos_0[0:cnt_p0[None], 0], np_pos_0[0:cnt_p0[None], 1], np_pos_0[0:cnt_p0[None], 2])
        # writer0.export_frame_ascii(cur_frame, series_prefix)

        # series_prefix = "out/plyfile/phase1_.ply"
        # np_pos_1 = pos_p1.to_numpy()
        # writer1 = ti.tools.PLYWriter(num_vertices = cnt_p1[None])
        # writer1.add_vertex_pos(np_pos_1[0:cnt_p1[None], 0], np_pos_1[0:cnt_p1[None], 1], np_pos_1[0:cnt_p1[None], 2])
        # writer1.export_frame_ascii(cur_frame, series_prefix)

        pre_render()
        camera.position(100, 100, 60)
        camera.lookat(0, 0, 30)
        camera.up(0, 0, 1)
        scene.particles(pos, 0.5, per_vertex_color=palette)
        scene.ambient_light((0.7, 0.7, 0.7))
        scene.set_camera(camera)
        canvas.scene(scene)
        gui.show()
    