import taichi as ti
from math import *

ti.init(arch=ti.cpu)

N = 21
len_cloth = 10

pos_vertex = ti.Vector.field(3, float, shape=(N*N))
vel_vertex = ti.Vector.field(3, float, shape=(N*N))
fixed = ti.field(bool, shape=(N*N))
cnt_triangle = (N-1) * (N-1) * 2
indice = ti.field(int, shape=cnt_triangle*3)
colors = ti.Vector.field(3, float, shape=(N*N))
sum_cal = ti.field(int, shape=(N*N))
sum_x = ti.Vector.field(3, float, shape=(N*N))
len_edge = len_cloth / (N - 1)
g = 0.98
dt = 1.0/30.0
damp = 0.99
substep = 32

ball_center = ti.Vector.field(3, float, shape=1)
ball_radius = 2.7

@ti.kernel
def init_scence():
    ball_center[0] = ti.Vector([5, 5, 0])

    for i in pos_vertex:
        pos_x = i // N
        pos_y = i % N
        pos_vertex[i] = ti.Vector([pos_x*len_edge, pos_y*len_edge, 10.0])
        vel_vertex[i] = ti.Vector([0, 0, 0])
        if i==0 or i==N*(N-1) :
            fixed[i] = True
        else:
            fixed[i] = False

    for i, j in ti.ndrange(N-1, N-1):
        # first triangle
        idx = (i * (N-1)) + j
        indice[idx*6+0] = (i * N) + j
        indice[idx*6+1] = ((i+1) * N) + j
        indice[idx*6+2] = ((i+1) * N) + j + 1
        # second triangle
        indice[idx*6+3] = (i * N) + j
        indice[idx*6+4] = (i * N) + j + 1
        indice[idx*6+5] = ((i+1) * N) + j + 1

    for i, j in ti.ndrange(N, N):
        if (i // 4 + j // 4) % 2 == 0:
            colors[i * N + j] = (0.22, 0.72, 0.52)
        else:
            colors[i * N + j] = (1, 0.334, 0.52)

    
@ti.kernel
def set_up():
    for i in vel_vertex:
        vel_vertex[i] *= damp
        vel_vertex[i] += ti.Vector([0, 0, -g]) * dt
        if not fixed[i]:
            pos_vertex[i] += dt * vel_vertex[i]


@ti.kernel
def limiting():
    for i in sum_x:
        sum_x[i] = ti.Vector([0, 0, 0])
        sum_cal[i] = 0

    for i in pos_vertex:
        x = i // N
        y = i % N
        dx = ti.Vector([1, 0, 1, -1])
        dy = ti.Vector([0, 1, 1, 1])
        origin_len = ti.Vector([1, 1, ti.sqrt(2.0), ti.sqrt(2.0)])

        for j in range(4):
            new_x = x + dx[j]
            new_y = y + dy[j]
            if new_x>=0 and new_x<N and new_y>=0 and new_y<N:
                new_idx = new_x * N + new_y
                ti.atomic_add(sum_cal[new_idx], 1)
                ti.atomic_add(sum_cal[i], 1)
                dirc = (pos_vertex[i] - pos_vertex[new_idx]).normalized()
                sum_x[i] += 0.5 * (pos_vertex[i] + pos_vertex[new_idx] + origin_len[j] * len_edge * dirc)
                sum_x[new_idx] += 0.5 * (pos_vertex[i] + pos_vertex[new_idx] - origin_len[j] * len_edge * dirc)

    for i in pos_vertex:
        if not fixed[i]:
            new_pos = (0.2 * pos_vertex[i] + sum_x[i]) / (0.2 + sum_cal[i])
            vel_vertex[i] += (new_pos - pos_vertex[i]) / dt / substep
            pos_vertex[i] = new_pos


@ti.kernel
def collision():
    for i in pos_vertex:
        dirc = pos_vertex[i] - ball_center[0]
        if (dirc.norm() < 1.05*ball_radius) and (not fixed[i]):
            new_pos = ball_center[0] + 1.05*ball_radius * dirc.normalized()
            vel_vertex[i] += (new_pos - pos_vertex[i]) / dt
            pos_vertex[i] = new_pos


def main():
    init_scence()
    window = ti.ui.Window("cloth", (768, 768))
    canvas = window.get_canvas()
    canvas.set_background_color((1, 1, 1))
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()
    forward = ti.Vector([-1, 1, 0]).normalized()
    up = ti.Vector([0, 0, 1]).normalized()
    left = ti.Vector([-1, -1, 0]).normalized()
    
    while window.running:
        if window.get_event(ti.ui.PRESS):
            if window.event.key == 'w':
                ball_center[0] += 0.2 * forward
            elif window.event.key == 's':
                ball_center[0] -= 0.2 * forward
            elif window.event.key == 'a':
                ball_center[0] += 0.2 * left
            elif window.event.key == 'd':
                ball_center[0] -= 0.2 * left
            elif window.event.key == 'q':
                ball_center[0] += 0.2 * up
            elif window.event.key == 'e':
                ball_center[0] -= 0.2 * up

        set_up()
        for _ in range(substep):
            limiting()
        
        collision()

        camera.position(20.0, -10.0, 15.0)
        camera.lookat(0, 0, 0)
        camera.up(-2, 1, 10/3)
        scene.mesh(pos_vertex, indice, per_vertex_color=colors)
        scene.particles(ball_center, ball_radius, color=(1,0,0))
        scene.ambient_light((0.7, 0.7, 0.7))
        scene.set_camera(camera)
        canvas.scene(scene)
        window.show()


if __name__ == '__main__':
    main()