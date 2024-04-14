import taichi as ti
import taichi.math as tm
import numpy as np
import math
import time

def cal_fluidn(water_radius:int, wall_radius:int, height:int, dis:float) -> int:
    res_n = 0
    for i in range(1, water_radius + 1):
        cur_r = i * dis
        d_theta = 2.0 * tm.asin(0.5 * dis / cur_r)
        num = int((2 * tm.pi) / d_theta)
        num_circle[i-1] = num
        res_n += num
    
    for i in range(wall_radius):
        cur_r = water_radius * dis + (i+1) * 0.5 * dis
        d_theta = 2.0 * tm.asin(0.5 * dis / cur_r)
        num = int((2 * tm.pi) / d_theta)
        num_circle[i+water_radius] = num
        res_n += num
    
    return res_n * height


ti.init(arch=ti.gpu)
show_type = 0
visualization = 0

# parameters
particle_radius = 1.0
h = particle_radius * 4
dt = 0.2 * min(math.sqrt(h / 60.0), h / 500)
particle_distance = 0.95
# wall_distance = 0.6
damp = 0.999
rot_force = 8

# boundary
rho_wall = 200.0
wall_layer = 8
centrifuge_radius = 16
centrifuge_height = 5
total_radius = (centrifuge_radius + wall_layer) * particle_distance 
total_height = (centrifuge_height + 2 * wall_layer) * particle_distance
boundX = ti.ceil(2 * total_radius) + 4
boundY = ti.ceil(2 * total_radius) + 4
boundZ = ti.ceil(total_height) + 4

num_circle = ti.field(int, shape=centrifuge_radius)
fluid_n = int(cal_fluidn(centrifuge_radius, 0, centrifuge_height, particle_distance))
total_num = int(cal_fluidn(centrifuge_radius, wall_layer, centrifuge_height + 2 * wall_layer, particle_distance))
phase = 2

tao = 1e-7
k1 = 300.0
k2 = 7.0

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
cellSize = 4.5
numCellX = int(ti.ceil(boundX / cellSize))
numCellY = int(ti.ceil(boundY / cellSize))
numCellZ = int(ti.ceil(boundZ / cellSize))
numCell = numCellX * numCellY * numCellZ

ParNum = ti.field(int, shape = (numCellX, numCellY, numCellZ))
Particles = ti.field(int, shape = (numCellX, numCellY, numCellZ, 20000))
NeiNum = ti.field(int, shape = fluid_n)
neighbor = ti.field(int, shape = (fluid_n, 2000))

# rendering
palette = ti.Vector.field(3, float, shape = fluid_n)
render_pos = ti.Vector.field(3, float, shape = fluid_n)

@ti.func
def W(r:float) -> float:
    res = 0.0
    if 0 < r and r < h:
        x = (h*h - r*r) / (h**3)
        res = 315.0 / 64.0 / tm.pi * x * x * x
    return res


@ti.func
def DW_prs(r) -> ti.Vector: 
    res = ti.Vector([0.0, 0.0, 0.0])
    r_len = r.norm()
    if 0 < r_len and r_len < h:
        x = (h - r_len) / (h * h * h)
        g_factor = -45.0 / tm.pi * x * x
        res = r * g_factor / r_len
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
    center = ti.Vector([boundX/2, boundY/2, pos[idx][2]])
    r2p = pos[idx] - center
    if r2p.norm() > centrifuge_radius * particle_distance :
        pos[idx] = (centrifuge_radius * particle_distance - 0.07) * tm.normalize(r2p) + center

    if pos[idx][2] < wall_layer * particle_distance :
        pos[idx][2] = wall_layer * particle_distance
        if vel[idx][2] < 0:
            vel[idx][2] *= -0.95
    
    if pos[idx][2] > (wall_layer+centrifuge_height) * particle_distance + 1 :
        pos[idx][2] = (wall_layer+centrifuge_height) * particle_distance + 1
        if vel[idx][2] > 0:
            vel[idx][2] *= -0.95


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

    for i, dx, dy, dz in ti.ndrange(fluid_n, (-1, 2), (-1, 2), (-1, 2)):
        idx_x = int(pos[i][0] / cellSize - 0.5)
        idx_y = int(pos[i][1] / cellSize - 0.5)
        idx_z = int(pos[i][2] / cellSize - 0.5)
        new_x = idx_x + dx
        new_y = idx_y + dy
        new_z = idx_z + dz
        if not(new_x < 0 or new_x >= numCellX or new_y < 0 or new_y >= numCellY or new_z < 0 or new_z >= numCellZ):
            cnt = ParNum[new_x, new_y, new_z]
            for t in range(cnt):
                nei = Particles[new_x, new_y, new_z, t]
                if nei!=i and (pos[nei]-pos[i]).norm() < 1.1*h:
                    kk = ti.atomic_add(NeiNum[i], 1)
                    neighbor[i, kk] = nei


@ti.kernel
def init():
    rho_0[0] = 1000.0 # water
    rho_0[1] = 500.0  # oil
    center_x = boundX / 2
    center_y = boundY / 2
    fluid_id = 0
    wall_id = total_num - 1
    for i, j in ti.ndrange((1, centrifuge_radius + wall_layer + 1), (1, centrifuge_height + 2*wall_layer + 1)):
        cur_r = 0.0
        if i <= centrifuge_radius :
            cur_r = i * particle_distance
        else :
            cur_r = centrifuge_radius * particle_distance + (i-centrifuge_radius) * 0.5 * particle_distance

        d_theta = 2.0 * tm.asin(0.5 * particle_distance / cur_r)
        center = ti.Vector([center_x, center_y, j * particle_distance])
        cur_theta = 0.0
        num = num_circle[i-1]
        while num > 0 :
            if i <= centrifuge_radius and j > wall_layer and j <= centrifuge_height + wall_layer:
                temp = ti.atomic_add(fluid_id, 1)
                pos[temp] = cur_r * ti.Vector([tm.cos(cur_theta), tm.sin(cur_theta), 0]) + center
                alpha[temp, 0] = 0.6
                alpha[temp, 1] = 0.4
            else:
                temp = ti.atomic_sub(wall_id, 1)
                pos[temp] = cur_r * ti.Vector([tm.cos(cur_theta), tm.sin(cur_theta), 0]) + center

            cur_theta += d_theta
            num -= 1

    print(fluid_id, wall_id)
    
   
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
                rho_bar[i] += rho_wall * W((pos[i] - pos[j]).norm())

        if rho_bar[i] < 1e-6:
            rho_bar[i] = rho_m[i]
    
    for i in prs:
        density = ti.max(rho_bar[i], rho_m[i])
        prs[i] = k1 * rho_m[i] * ((density/rho_m[i])**k2 - 1) / k2
        # prs[i] = k3 * (density - rho_m[i])


@ti.kernel
def cal_drift():
    for i, k in drift_vel:
        r2p = rot_force * (pos[i] - ti.Vector([boundX/2, boundY/2, pos[i][2]]))
        g = tm.cross(r2p, ti.Vector([0, 0, 1]))
        first_term = (g - acc[i]) * tao
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
                    prs_grad += rho_m[j] * (alpha[j, k] * prs[j] - alpha[i, k] * prs[i]) * DW(pos[i] - pos[j]) / rho_bar[j]

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
        if k == 1:
            assert(first_term == 0 and second_term == 0)
    

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
                # del_p -= k3 * rho_0[ph] * (alpha[i, ph] - cur)
                del_p -= k1 * rho_0[ph] * ((k2-1)*((rho_bar[i]/rho_m[i])**k2)+1) * (alpha[i, ph] - cur) / k2
        else:
            for ph in range(phase):
                cur = alpha[i, ph]
                if alpha[i, ph] < 0:
                    alpha[i, ph] = 0.0
                else:
                    alpha[i, ph] /= tot
                # del_p -= k3 * rho_0[ph] * (alpha[i, ph] - cur)
                del_p -= k1 * rho_0[ph] * ((k2-1)*((rho_bar[i]/rho_m[i])**k2)+1) * (alpha[i, ph] - cur) / k2
        
        prs[i] += del_p


@ti.kernel
def cal_acc():
    for i in acc:
        r2p = rot_force * (pos[i] - ti.Vector([boundX/2, boundY/2, pos[i][2]]))
        acc[i] = tm.cross(r2p, ti.Vector([0, 0, 1]))
        prs_grad = ti.Vector([0.0, 0.0, 0.0])
        Tdm_grad = ti.Vector([0.0, 0.0, 0.0])

        for nei in range(NeiNum[i]):
            j = neighbor[i, nei]
            if j < fluid_n: # partical
                prs_grad += rho_m[j] * (prs[i] + prs[j]) / (2 * rho_bar[j]) * DW(pos[i] - pos[j])
            else: # Wall
                prs_grad += rho_wall * (prs[i] + prs[i]) / (2 * rho_0[0]) * DW(pos[i] - pos[j])

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
    for i in range(fluid_n):
        render_pos[i] = pos[i]
        if show_type == 0:
            palette[i] = ti.Vector([alpha[i, 0], alpha[i, 1], 0.0])
        elif show_type == 1 :
            ratio = (prs[i] + 30) / 130.0
            palette[i] = ti.Vector([ratio, 1 - ratio, 0.0])
       

if __name__ == '__main__':
    init()
    gui = ti.ui.Window('SPH', res = (700, 700))
    canvas = gui.get_canvas()
    canvas.set_background_color((1, 1, 1))
    scene = gui.get_scene()
    camera = ti.ui.Camera()

    camera.position(18, 18, 100)
    camera.lookat(18, 18, 0)
    camera.up(0, 1, 0)

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

        if visualization == 0:
            pre_render()
            scene.particles(centers=render_pos, per_vertex_color=palette, radius=0.3)
            scene.ambient_light((0.7, 0.7, 0.7))
            scene.set_camera(camera)
            canvas.scene(scene)
            gui.show()
            cur_frame += 1
        else:
            pre_render()
            series_prefix = "out/plyfile/water_.ply"
            np_pos = pos.to_numpy()
            np_palette = palette.to_numpy()
            writer = ti.tools.PLYWriter(num_vertices = fluid_n)
            writer.add_vertex_pos(np_pos[:fluid_n, 0], np_pos[:fluid_n, 1], np_pos[:fluid_n, 2])
            writer.add_vertex_color(np_palette[:fluid_n, 0], np_palette[:fluid_n, 1], np_palette[:fluid_n, 2])
            writer.export_frame_ascii(cur_frame, series_prefix)
            cur_frame += 1

        print(cur_frame)
        if cur_frame == 840 :
            exit()
