import taichi as ti
import numpy as np
import time

@ti.data_oriented
class GradientDesent:
    def __init__(self, energy_fn, grad_fn, dim=3, c1=0.5, beta=0.6, eta=1e-3):
        self.dim = dim
        self.energy_fn = energy_fn
        self.grad_fn = grad_fn
        self.c1 = c1
        self.beta = beta
        self.eta = eta

        self.x = ti.field(float, shape=dim)
        self.grad = ti.field(float, shape=dim)
        self.temp_x = ti.field(float, shape=dim)
        self.d = ti.field(float, shape=dim)
        self.f0 = 0.0


    @ti.kernel
    def update_temp(self, a: float):
        for i in range(self.dim):
            self.temp_x[i] = self.x[i] + a * self.d[i]


    @ti.kernel
    def grad_inf_norm(self) -> float:
        res = 0.0
        for i in range(self.dim):
            ti.atomic_max(res, ti.abs(self.grad[i]))
        return res


    @ti.kernel
    def set_direction(self):
        for val in self.grad:
            self.d[val] = -self.grad[val]


    @ti.kernel
    def update_x(self, alpha: float):
        for i in range(self.dim):
            self.x[i] += alpha * self.d[i]


    def line_search(self):
        alpha = 1.0
        while alpha > 1e-6:
            self.update_temp(alpha)
            f_new = self.energy_fn(self.temp_x)
            if f_new <= self.f0 * self.c1:
                break
            alpha *= self.beta
        return alpha


    def minimize(self, max_iter=200):
        for it in range(max_iter):
            # compute energy & gradient
            self.grad_fn(self.x, self.grad)
            self.f0 = self.energy_fn(self.x)

            # print(f"Iteration {it}, Energy: {self.f0:.4e}")

            # 检查收敛
            g_norm = self.grad_inf_norm()
            # print(f"Grad norm: {g_norm:.4e}")
            if g_norm < self.eta:
                # print(f"Converged at iteration {it}")
                break

            self.set_direction()
            alpha = self.line_search()
            # print(f"Step size: {alpha:.4e}")
            self.update_x(alpha)


# 示例使用
if __name__ == "__main__":
    ti.init(arch=ti.gpu)
    dim = 30

    @ti.kernel
    def quadratic_energy(x: ti.template()) -> float:
        f = 0.0
        for i in range(x.shape[0]):
            f += x[i] ** 2
        return f

    # 能量函数和梯度函数
    @ti.kernel
    def quadratic_energy_grad(x: ti.template(), grad: ti.template()) -> float:
        f = 0.0
        for i in range(x.shape[0]):
            f += x[i] ** 2
            grad[i] = 2 * x[i]
        return f
            

    @ti.kernel
    def rosenbrock(x: ti.template()) -> float:
        f_total = 0.0
        for i in x:
            if i % 3 == 0:
                x1, x2, x3 = x[i], x[i+1], x[i+2]
                f_total += (3 - x1)**2 + 7*(x2 - x1**2)**2 + 9*(x3 - x1 - x2**2)**2
        return f_total

    
    @ti.kernel
    def rosenbrock_grad(x: ti.template(), grad: ti.template()):
        for i in x:
            if i % 3 == 0:
                x1, x2, x3 = x[i], x[i+1], x[i+2]
                # 梯度计算
                grad[i] = 2*(x1 - 3) + 28*(x1**2 - x2)*x1 + 18*(-x3 + x1 + x2**2)
                grad[i+1] = 14*(x2 - x1**2) + 18*(x3 - x1 - x2**2)*(-2*x2)
                grad[i+2] = 18*(x3 - x1 - x2**2)


    optimizer = GradientDesent(energy_fn=rosenbrock,
                      grad_fn=rosenbrock_grad,
                      c1 = 1.0,
                      beta = 0.9,
                      dim=dim)
    
    # x_init = np.random.uniform(low=0.0, high=100.0, size=dim)
    x_init = np.zeros(dim)
    # 设置初始值
    optimizer.x.from_numpy(x_init)
    
    # 梯度检查
    # optimizer.check_gradient()
    
    # 执行优化
    optimizer.minimize(max_iter=20000)

    print(f"Final parameters: {optimizer.x.to_numpy()}")
    
    # 绘制能量变化
    # plt.plot(optimizer.f_his)
    # plt.title("Energy History")
    # plt.xlabel("Iteration")
    # plt.ylabel("Energy")
    # plt.show()