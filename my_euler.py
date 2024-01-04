from taichi.examples.patterns import taichi_logo

import taichi as ti
import taichi.math as tm

ti.init(arch=ti.cpu)

size = ti.Vector([512, 512])
frame = 30.0
substep = 2
dt = 1.0 / (frame*substep)
g = 9.8
iteration = 40
miu = 30

color_field = ti.Vector.field(3, float, shape=(size[0], size[1]))
vel_field = ti.Vector.field(2, float, shape=(size[0], size[1]))
new_color_filed = ti.Vector.field(3, float, shape=(size[0], size[1]))
new_vel_filed = ti.Vector.field(2, float, shape=(size[0], size[1]))
divergence = ti.field(float, shape=(size[0], size[1]))
press = ti.field(float, shape=(size[0], size[1]))
new_press = ti.field(float, shape=(size[0], size[1]))
curl_vel = ti.field(float, shape=(size[0], size[1]))
laplacian = ti.Vector.field(2, float, shape=(size[0], size[1]))


@ti.kernel
def mouse_addspeed(
    cur_posx: int,
    cur_posy: int,
    prev_posx: int,
    prev_posy: int,
    mouseRadius: float,
    vf: ti.template(),
    new_vf: ti.template(),
):
    for i, j in vf:
        vec1 = ti.Vector([cur_posx - prev_posx, cur_posy - prev_posy])
        vec2 = ti.Vector([i - prev_posx, j - prev_posy])
        dotans = tm.dot(vec1, vec2)
        distance = abs(tm.cross(vec1, vec2)) / (tm.length(vec1) + 0.001)
        if (
            dotans >= 0
            and dotans <= 0.5 * tm.length(vec1)
            and distance <= mouseRadius
        ):
            new_vf[i, j] = vf[i, j] + vec1 * 125
        else:
            new_vf[i, j] = vf[i, j]

    for i, j in vf:
        vf[i, j] = new_vf[i, j]


def mouse_interaction(prev_posx: int, prev_posy: int):
    mouse_x, mouse_y = window.get_cursor_pos()
    mousePos_x = int(mouse_x * size[0])
    mousePos_y = int(mouse_y * size[1])
    if prev_posx == 0 and prev_posy == 0:
        prev_posx = mousePos_x
        prev_posy = mousePos_y
    mouseRadius = 0.01 * min(size[0], size[1])

    mouse_addspeed(
        mousePos_x,
        mousePos_y,
        prev_posx,
        prev_posy,
        mouseRadius,
        vel_field,
        new_vel_filed,
    )
    
    prev_posx = mousePos_x
    prev_posy = mousePos_y
    return prev_posx, prev_posy


@ti.func
def Border(i: int, j: int):
    res = False
    if i==0 or i==size[0]-1 or j==0 or j==size[1]-1 :
        res = True

    return res    


@ti.func
def Corner(i: int, j: int):
    res = False
    if i==0 and j==0 :
        res = True
    if i==size[0]-1 and j==0:
        res = True
    if i==0 and j==size[1]-1:
        res = True
    if i==size[0]-1 and j==size[1]-1:
        res = True

    return res


@ti.func
def GridSample(i: int, j: int, field: ti.template()):
    i = ti.max(0, ti.min(i, field.shape[0]-1))
    j = ti.max(0, ti.min(j, field.shape[1]-1))
    return field[i, j]


@ti.func
def Sample(i: float, j: float, field: ti.template()):
    idx_i = int(i+0.5)
    idx_j = int(j+0.5)
    fi = i - idx_i + 0.5
    fj = j - idx_j + 0.5
    assert(fi>=0 and fj>=0)
    v1 = GridSample(idx_i, idx_j, field)
    v2 = GridSample(idx_i-1, idx_j, field)
    v3 = GridSample(idx_i, idx_j-1, field)
    v4 = GridSample(idx_i-1, idx_j-1, field)
    ret = (1-fj)*((1-fi)*v4+fi*v3) + fj*((1-fi)*v2+fi*v1)
    return ret


@ti.func
def PressureIteraion():
    for i, j in press:
        new_press[i, j] = (press[i+1, j] + press[i-1, j] + press[i, j+1] + press[i, j-1] - divergence[i, j]) / 4

    for i, j in press:
        if Corner(i, j):
            press[i, j] = 0
        elif i==0:
            press[i, j] = new_press[i+1, j]
        elif i==size[0]-1:
            press[i, j] = new_press[size[0]-2, j]
        elif j==0:
            press[i, j] = new_press[i, j+1]
        elif j==size[1]-1:
            press[i, j] = new_press[i, size[1]-2]
        else:
            press[i, j] = new_press[i, j]
        

def SolvePressure():
    for _ in range(iteration):
        PressureIteraion()


@ti.kernel
def Initiate():
    for i, j in ti.ndrange(size[0] * 4, size[1] * 4):
        # 4x4 super sampling:
        sample = taichi_logo(ti.Vector([i, j]) / (size[0] * 4))
        color_field[i // 4, j // 4] += sample / 16
        vel_field[i, j] = ti.Vector([0, 0])


@ti.kernel
def Advection():
    # for every grid
    for i, j in color_field:
        cur_pos = ti.Vector([i+0.5, j+0.5])
        pos_mid = cur_pos - 0.5 * dt * Sample(cur_pos[0], cur_pos[1], vel_field)
        last_pos = cur_pos - dt * Sample(pos_mid[0], pos_mid[1], vel_field)
        new_vel_filed[i, j] = Sample(last_pos[0], last_pos[1], vel_field)
        new_color_filed[i, j] = Sample(last_pos[0], last_pos[1], color_field)

    for i, j in color_field:
        color_field[i, j] = new_color_filed[i, j]
        vel_field[i, j] = new_vel_filed[i, j]
        if (i <= 0) or (i >= size[0] - 1) or (j >= size[1] - 1) or (j <= 0):
            vel_field[i, j] = ti.Vector([0.0, 0.0])


@ti.kernel
def ExternelForces():
    for i, j in color_field:
        if Border(i, j):
            curl_vel[i, j] = 0
        else:
            curl_vel[i, j] = 0.5 * ((vel_field[i+1, j][1] - vel_field[i-1, j][1]) - (vel_field[i, j+1][0] - vel_field[i, j-1][0]))
    
    for i, j in curl_vel:
        # TAG : what
        gradcurl = ti.Vector(
            [
                0.5 * (curl_vel[i + 1, j] - curl_vel[i - 1, j]),
                0.5 * (curl_vel[i, j + 1] - curl_vel[i, j - 1]),
                0,
            ]
        )
        GradCurlLength = tm.length(gradcurl)
        if GradCurlLength > 1e-5:
            force = miu * tm.cross(gradcurl / GradCurlLength, ti.Vector([0, 0, 1]))
            new_vel_filed[i, j] = vel_field[i, j] + dt * force[:2]

    # for i, j in color_field:
    #     if Border(i, j):
    #         laplacian[i, j] = ti.Vector([0.0, 0.0])
    #     else:
    #         lapx = (vel_field[i+1, j][0] - 2.0 * vel_field[i, j][0] + vel_field[i-1, j][0]) \
    #                 + (vel_field[i, j+1][0] - 2.0 * vel_field[i, j][0] + vel_field[i, j-1][0])
    #         lapy = (vel_field[i+1, j][1] - 2.0 * vel_field[i, j][1] + vel_field[i-1, j][1]) \
    #                 + (vel_field[i, j+1][1] - 2.0 * vel_field[i, j][1] + vel_field[i, j-1][1])
    #         laplacian[i, j] = ti.Vector([lapx, lapy])
    #         force = miu * laplacian[i, j]
    #         new_vel_filed[i, j] = vel_field[i, j] + dt * force


    for i, j in color_field:
        if Border(i, j):
            vel_field[i, j] = ti.Vector([0, 0])
        else:
            vel_field[i, j] = new_vel_filed[i, j]
        

@ti.kernel
def Projection():
    for i, j in divergence:
        divergence[i, j] = (vel_field[i+1, j][0] + vel_field[i, j+1][1] - vel_field[i-1, j][0] - vel_field[i, j-1][1]) * 0.5
        press[i, j] = 0

    SolvePressure()

    for i, j in vel_field:
        if Border(i, j):
            vel_field[i, j] = ti.Vector([0, 0])
        else:
            # TAG: why without dt
            vel_field[i, j] -= 0.5 * ti.Vector([press[i+1,j]-press[i-1,j], press[i, j+1]-press[i, j-1]])


mouse_prevposx, mouse_prevposy = 0, 0
if __name__ == "__main__":
    Initiate()
    window = ti.GUI("euler", res=(size[0], size[1]))
    while window.running:
        for _ in range(substep):
            Advection()
            # ExternelFoces()

            mouse_prevposx, mouse_prevposy = mouse_interaction(mouse_prevposx, mouse_prevposy)

            Projection()
            

        window.set_image(color_field)
        window.show()