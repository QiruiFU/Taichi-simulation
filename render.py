import taichi as ti
import numpy as np
from plyfile import PlyData

ti.init(arch=ti.gpu)

# 1. 读取 PLY 文件中的粒子数据
def load_particles_from_ply(filename):
    plydata = PlyData.read(filename)
    vertex = plydata['vertex']
    positions = np.vstack((vertex['x'], vertex['y'], vertex['z'])).T
    return positions.astype(np.float32)

particle_positions = load_particles_from_ply('out/plyfile/water__001234.ply')
num_particles = particle_positions.shape[0]

# 2. 定义 Taichi 数据结构
screen_res = (800, 600)
particle_radius = 0.05
max_num_particles = 1000000  # 根据需要调整

particle_pos = ti.Vector.field(3, dtype=ti.f32, shape=max_num_particles)
particle_color = ti.Vector.field(3, dtype=ti.f32, shape=max_num_particles)

depth_buffer = ti.field(dtype=ti.f32, shape=screen_res)
normal_buffer = ti.Vector.field(3, dtype=ti.f32, shape=screen_res)
color_buffer = ti.Vector.field(3, dtype=ti.f32, shape=screen_res)

# 初始化粒子数据
particle_pos.from_numpy(particle_positions)
particle_color.fill(1.0)  # 可以根据需要设置粒子颜色

# 相机参数
camera_pos = ti.Vector([4.0, 4.0, 4.0])
camera_dir = ti.Vector([-1.0, -1.0, -1.0])
up = ti.Vector([0.0, 0.0, 1.0])
fov = 60  # 视角

# 3. 清空缓冲区
@ti.kernel
def clear_buffers():
    for i, j in depth_buffer:
        depth_buffer[i, j] = 1e10
        normal_buffer[i, j] = ti.Vector([0.0, 0.0, 0.0])
        color_buffer[i, j] = ti.Vector([0.0, 0.0, 0.0])

# 4. 粒子渲染到深度缓冲区（生成深度图）
@ti.kernel
def render_particles_to_depth():
    for p in range(num_particles):
        pos = particle_pos[p]
        # 将世界坐标转换为相机坐标
        view_pos = pos - camera_pos
        # 投影到屏幕空间
        ndc_x = view_pos[0] / (-view_pos[2]) * ti.tan(fov / 2 * 3.1415926 / 180)
        ndc_y = view_pos[1] / (-view_pos[2]) * ti.tan(fov / 2 * 3.1415926 / 180)
        screen_x = int((ndc_x + 1) * screen_res[0] / 2)
        screen_y = int((ndc_y + 1) * screen_res[1] / 2)
        if 0 <= screen_x < screen_res[0] and 0 <= screen_y < screen_res[1]:
            depth = -view_pos[2]
            ti.atomic_min(depth_buffer[screen_x, screen_y], depth)

# 5. 计算法线缓冲区
@ti.kernel
def compute_normals():
    for i, j in depth_buffer:
        if depth_buffer[i, j] < 1e10:
            dzdx = (depth_buffer[min(i + 1, screen_res[0] - 1), j] - depth_buffer[max(i - 1, 0), j]) * 0.5
            dzdy = (depth_buffer[i, min(j + 1, screen_res[1] - 1)] - depth_buffer[i, max(j - 1, 0)]) * 0.5
            normal = ti.Vector([-dzdx, -dzdy, 1.0]).normalized()
            normal_buffer[i, j] = normal

# 6. 光照和着色
@ti.kernel
def shade():
    light_dir = ti.Vector([0.0, 0.0, -1.0]).normalized()
    for i, j in color_buffer:
        if depth_buffer[i, j] < 1e10:
            normal = normal_buffer[i, j]
            intensity = normal.dot(light_dir)
            intensity = ti.max(0, intensity)
            color_buffer[i, j] = intensity * ti.Vector([0.2, 0.6, 0.9])

# 7. 渲染循环
gui = ti.GUI("Screen Space Fluid Rendering", res=screen_res)

while gui.running:
    clear_buffers()
    render_particles_to_depth()
    compute_normals()
    shade()
    gui.set_image(color_buffer.to_numpy())
    gui.show()
