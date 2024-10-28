import taichi as ti
import meshio

ti.init(arch=ti.gpu)

g = ti.Vector([0, 0, -9.8])
Young = 50000
Poisson = 0.45
miu = Young / (2 + 2 * Poisson)
lam = Young * Poisson / ((1 + Poisson) * (1 - 2 * Poisson))
dt = 0.001
damp = 0.9999
m = 1

@ti.func
def contain(tet, p):
    res = 0
    if tetra[tet][0] == p or tetra[tet][1] == p or tetra[tet][2] == p or tetra[tet][3] == p:
        res = 1
    else:
        res = 0

    return res


@ti.kernel
def init():
    for i in range(NumPoint):
        pos[i] += ti.Vector([0.0, 0.0, 5.0])
        vel[i] = ti.Vector([0.0, 0.0, 0.0])
        acc[i] = ti.Vector([0.0, 0.0, 0.0])
    
    for i in range(NumTetra):
        dm1 = pos[tetra[i][1]] - pos[tetra[i][0]]
        dm2 = pos[tetra[i][2]] - pos[tetra[i][0]]
        dm3 = pos[tetra[i][3]] - pos[tetra[i][0]]
        Dm = ti.Matrix.cols([dm1, dm2, dm3])
        Dm_inv[i] = Dm.inverse()
        volume[i] = Dm.determinant() / 6


@ti.kernel
def update():
    for i in acc:
        acc[i] = ti.Vector([0.0, 0.0, 0.0])

    for i in range(NumTetra):
        p0 = tetra[i][0]
        p1 = tetra[i][1]
        p2 = tetra[i][2]
        p3 = tetra[i][3]

        ds1 = pos[p1] - pos[p0]
        ds2 = pos[p2] - pos[p0]
        ds3 = pos[p3] - pos[p0]
        Ds = ti.Matrix.cols([ds1, ds2, ds3])

        eye3 = ti.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        F = Ds @ Dm_inv[i]
        G = 0.5 * (((F.transpose()) @ F) - eye3)
        # if contain(i, 2):
            # print("G:", G)
        P = F @ (2.0 * miu * G + lam * (G.trace() * eye3))
        # if contain(i, 2):
            # print("P:", P)
        force = -volume[i] * (P @ (Dm_inv[i].transpose()))
        f1 = ti.Vector([force[0, 0], force[1, 0], force[2, 0]])
        f2 = ti.Vector([force[0, 1], force[1, 1], force[2, 1]])
        f3 = ti.Vector([force[0, 2], force[1, 2], force[2, 2]])
        f0 = -f1 - f2 - f3
        # if contain(i, 2):
            # print("f0:", f0)

        acc[p0] += f0 / m        
        acc[p1] += f1 / m        
        acc[p2] += f2 / m        
        acc[p3] += f3 / m  


@ti.kernel
def advance():
    for i in pos:
        vel[i] += (acc[i] + g) * dt
        vel[i] *= damp
        pos[i] += vel[i] * dt

        if pos[i][2] < 0 :
            pos[i][2] = -pos[i][2]
            if vel[i][2] < 0 :
                vel[i][2] = -0.8 * vel[i][2]
        # vel[i] += (acc[i] + g) * dt * damp
        # if pos[i][2] < 0 and vel[i][2] < 0 :
        #     vel[i][2] = 0
        
        # pos[i] += vel[i] * dt


if __name__ == "__main__":
    # initiate
    mesh = meshio.read("frog.msh")
    points = mesh.points
    cells = mesh.cells_dict['tetra']
    NumPoint = len(points)
    NumTetra = len(cells)
    print(points.min(axis=0))
    # print(NumPoint, NumTetra)

    pos = ti.Vector.field(3, float, shape = NumPoint)
    vel = ti.Vector.field(3, float, shape = NumPoint)
    acc = ti.Vector.field(3, float, shape = NumPoint)
    
    Dm_inv = ti.Matrix.field(n = 3, m = 3, dtype=float, shape = NumTetra) # inverse of initial position Matrix
    volume = ti.field(float, shape = NumTetra) # initial volume of tetrahedron
    tetra = ti.Vector.field(4, int, shape = NumTetra)

    pos.from_numpy(points)
    tetra.from_numpy(cells)
    init()


    for i in range(1000):
        for _ in range(20):
            # print(i, _)
            # print(i, acc.to_numpy().max(axis=0))
            # print(acc.to_numpy().argmax(axis=0))
            update()
            advance()
        
        cur_mesh = meshio.Mesh(pos.to_numpy(), [("tetra", cells)])
        cur_mesh.write(f"out_fem/change{i}.vtk")
        
