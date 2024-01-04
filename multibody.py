import taichi as ti
ti.init(ti.cpu)

G = 0.001
N = 4
mass_sun = 3.3e5 # ratio of mass of sun and earth is about 3.3e5
mass_earth = 1.0
dt = 2e-5
substep = 2

pos = ti.Vector.field(2, float, N)
vel = ti.Vector.field(2, float, N)
force = ti.Vector.field(2, float, N)
M = ti.field(float, N)

@ti.kernel
def initialize():
    for i in pos:
        theta = ti.random() * 2.0 * ti.math.pi
        r = ti.random() / 2.0
        pos[i] = ti.Vector([0.5, 0.5]) + r * ti.Vector([ti.cos(theta), ti.sin(theta)])
        vel[i] = ti.Vector([0.0, 0.0])
        if i==0 :
            M[i] = mass_earth
        else:
            M[i] = mass_sun


@ti.kernel
def compute_force():
    for i in pos:
        force[i] = ti.Vector([0.0, 0.0])

    for i in pos:
        p = pos[i]
        for j in range(N):
            if i != j:
                dis = p - pos[j]
                if dis.norm() > 0.075:
                    f = -G * M[i] * M[j] / (dis.norm() * dis.norm())
                    force[i] += f * dis / dis.norm()


@ti.kernel
def update():
    t = dt / substep
    for i in pos:
        vel[i] += t * force[i] / M[i]
        pos[i] += t * vel[i]


@ti.kernel
def compute_radiance() -> float: #The radiation received by the Earth follows an inverse square law.
    total_dis = 0.0 
    for i in range(1, N):
        d_squre = ((pos[i] - pos[0]).norm()) ** 2
        total_dis += 1 / d_squre
    
    return total_dis


def main():
    initialize()
    my_gui = ti.GUI('3-body problem', (512, 512))
    pause = False

    while my_gui.running:
        my_gui.get_event()
        if my_gui.is_pressed('r'):
            initialize()
        elif my_gui.is_pressed('p'):
            pause = not pause

        for _ in range(substep):
            if not pause:
                compute_force()
                update()

        # calculate the background color
        total_dis = compute_radiance()
        if total_dis < 5000:
            bg_color = int(min(total_dis, 1000) / 1000 * 0xF0)
            my_gui.clear(bg_color*0x010101)
        else: # more than 5000 means catastrophe
            my_gui.clear(0xF00000)

        for i in range(N):
            if i==0:
                my_gui.circle(pos=[pos[i][0], pos[i][1]], color=0x0000FF, radius=2.5)
            else:
                my_gui.circle(pos=[pos[i][0], pos[i][1]], color=0xFFD700, radius=5)

        my_gui.text(content='press \'r\' to reset, \'p\' to pause/start ', pos=[0.05, 0.95], font_size=15, color=0x00FF00)
        my_gui.show()


if __name__ == "__main__":
    main()