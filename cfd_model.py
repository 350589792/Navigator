# models/cfd_model.py

import numpy as np
from scipy.sparse import linalg
from scipy.sparse import diags
import matplotlib.pyplot as plt


class CFDSimulator:
    def __init__(self, config):
        """
        初始化CFD模拟器
        Args:
            config: 配置参数字典
        """
        # 网格参数
        self.nx = config.get('nx', 100)  # x方向网格点数
        self.ny = config.get('ny', 100)  # y方向网格点数
        self.dx = config.get('dx', 0.1)  # x方向网格间距
        self.dy = config.get('dy', 0.1)  # y方向网格间距

        # 物理参数
        self.rho = config.get('density', 1.0)  # 密度
        self.mu = config.get('viscosity', 0.01)  # 粘度
        self.k = config.get('conductivity', 0.1)  # 热导率
        self.cp = config.get('specific_heat', 1000.0)  # 比热容

        # 时间步长
        self.dt = config.get('dt', 0.001)

        # 初始化场变量
        self.initialize_fields()

    def initialize_fields(self):
        """初始化计算场"""
        # 温度场
        self.T = np.zeros((self.ny, self.nx))
        self.T_old = np.zeros((self.ny, self.nx))
        # 速度场
        self.u = np.zeros((self.ny, self.nx))  # x方向速度
        self.u_old = np.zeros((self.ny, self.nx))
        self.v = np.zeros((self.ny, self.nx))  # y方向速度
        self.v_old = np.zeros((self.ny, self.nx))
        # 压力场
        self.p = np.zeros((self.ny, self.nx))
        # 初始化边界条件存储
        self.boundary_conditions = {}

    def set_boundary_conditions(self, boundary_conditions):
        """设置边界条件"""
        # 温度边界条件
        self.T[0, :] = boundary_conditions['bottom_temp']  # 底部
        self.T[-1, :] = boundary_conditions['top_temp']  # 顶部
        self.T[:, 0] = boundary_conditions['left_temp']  # 左侧
        self.T[:, -1] = boundary_conditions['right_temp']  # 右侧

        # 速度边界条件
        self.u[0, :] = 0  # 底部无滑移
        self.u[-1, :] = 0  # 顶部无滑移
        self.v[0, :] = 0  # 底部无滑移
        self.v[-1, :] = 0  # 顶部无滑移

    def solve_momentum_equation(self):
        """求解动量方程"""
        # x方向动量方程
        for i in range(1, self.ny - 1):
            for j in range(1, self.nx - 1):
                # 对流项
                convection = (
                        self.u[i, j] * (self.u[i, j + 1] - self.u[i, j - 1]) / (2 * self.dx) +
                        self.v[i, j] * (self.u[i + 1, j] - self.u[i - 1, j]) / (2 * self.dy)
                )

                # 扩散项
                diffusion = (
                        self.mu * (
                        (self.u[i, j + 1] - 2 * self.u[i, j] + self.u[i, j - 1]) / (self.dx ** 2) +
                        (self.u[i + 1, j] - 2 * self.u[i, j] + self.u[i - 1, j]) / (self.dy ** 2)
                )
                )

                # 压力梯度
                pressure_grad = (self.p[i, j + 1] - self.p[i, j - 1]) / (2 * self.dx)

                # 更新速度
                self.u[i, j] = self.u[i, j] + self.dt * (-convection + diffusion / self.rho - pressure_grad / self.rho)

        # y方向动量方程类似实现...

    def solve_energy_equation(self, max_iter=1000, tolerance=1e-6):
        """求解能量方程
        Args:
            max_iter: 最大迭代次数
            tolerance: 收敛容差
        """
        for iter_count in range(max_iter):
            T_new = np.copy(self.T)
            max_change = 0.0

            for i in range(1, self.ny - 1):
                for j in range(1, self.nx - 1):
                    # 对流项
                    convection = (
                            self.u[i, j] * (self.T[i, j + 1] - self.T[i, j - 1]) / (2 * self.dx) +
                            self.v[i, j] * (self.T[i + 1, j] - self.T[i - 1, j]) / (2 * self.dy)
                    )

                    # 扩散项
                    diffusion = (
                            self.k * (
                            (self.T[i, j + 1] - 2 * self.T[i, j] + self.T[i, j - 1]) / (self.dx ** 2) +
                            (self.T[i + 1, j] - 2 * self.T[i, j] + self.T[i - 1, j]) / (self.dy ** 2)
                    )
                    )

                    # 更新温度
                    T_new[i, j] = self.T[i, j] + self.dt * (-convection + diffusion / (self.rho * self.cp))
                    max_change = max(max_change, abs(T_new[i, j] - self.T[i, j]))

            self.T = T_new

            # 检查收敛性
            if max_change < tolerance:
                return True

        return False  # 达到最大迭代次数仍未收敛

    def solve_continuity_equation(self):
        """求解连续方程"""
        # 构建压力泊松方程系数矩阵
        N = self.nx * self.ny
        A = np.zeros((N, N))
        b = np.zeros(N)

        # 填充系数矩阵
        for i in range(1, self.ny - 1):
            for j in range(1, self.nx - 1):
                idx = i * self.nx + j
                A[idx, idx] = -4
                A[idx, idx + 1] = 1
                A[idx, idx - 1] = 1
                A[idx, idx + self.nx] = 1
                A[idx, idx - self.nx] = 1

                # 构建右端项
                b[idx] = self.rho * (
                        (self.u[i, j + 1] - self.u[i, j - 1]) / (2 * self.dx) +
                        (self.v[i + 1, j] - self.v[i - 1, j]) / (2 * self.dy)
                ) / self.dt

        # 求解压力泊松方程
        p_new = linalg.spsolve(diags(A.diagonal()), b)
        self.p = p_new.reshape((self.ny, self.nx))

    def run_simulation(self, boundary_conditions, max_steps=100):
        """运行完整的CFD模拟
        Args:
            boundary_conditions: 边界条件字典
            max_steps: 最大迭代步数
        Returns:
            模拟结果字典
        """
        try:
            # 设置边界条件
            self.boundary_conditions = boundary_conditions
            self.set_boundary_conditions(boundary_conditions)
            
            # 初始化旧场变量
            self.u_old = np.copy(self.u)
            self.v_old = np.copy(self.v)
            self.T_old = np.copy(self.T)

            for step in range(max_steps):
                # 1. 求解动量方程
                self.solve_momentum_equation()

                # 2. 求解连续方程
                self.solve_continuity_equation()

                # 3. 求解能量方程（带收敛检查）
                energy_converged = self.solve_energy_equation(max_iter=100)
                if not energy_converged:
                    print(f"Warning: Energy equation did not converge at step {step}")

                # 4. 应用边界条件
                self.apply_boundary_conditions()

                # 5. 检查整体收敛性
                if self.check_convergence():
                    print(f"Simulation converged after {step + 1} steps")
                    break

            return self.get_results()
            
        except Exception as e:
            print(f"Error in CFD simulation: {str(e)}")
            # 返回一个基本的结果以避免完全失败
            return {
                'temperature_field': np.copy(self.T),
                'velocity_field_u': np.copy(self.u),
                'velocity_field_v': np.copy(self.v),
                'pressure_field': np.copy(self.p)
            }

    def apply_boundary_conditions(self):
        """应用所有边界条件"""
        # 温度边界条件
        self.apply_temperature_boundary()
        # 速度边界条件
        self.apply_velocity_boundary()
        # 压力边界条件
        self.apply_pressure_boundary()

    def apply_temperature_boundary(self):
        """应用温度边界条件"""
        # 固定温度边界
        self.T[0, :] = self.boundary_conditions['bottom_temp']
        self.T[-1, :] = self.boundary_conditions['top_temp']

        # 绝热边界
        self.T[:, 0] = self.T[:, 1]
        self.T[:, -1] = self.T[:, -2]

    def apply_velocity_boundary(self):
        """应用速度边界条件"""
        # 无滑移边界条件
        self.u[0, :] = 0
        self.u[-1, :] = 0
        self.v[0, :] = 0
        self.v[-1, :] = 0

        # 入口速度条件
        self.u[:, 0] = self.boundary_conditions.get('inlet_velocity', 0)

        # 出口条件
        self.u[:, -1] = self.u[:, -2]  # 零梯度
        self.v[:, -1] = self.v[:, -2]

    def apply_pressure_boundary(self):
        """应用压力边界条件"""
        # 出口压力条件
        self.p[:, -1] = 0

        # 其他边界使用零梯度条件
        self.p[:, 0] = self.p[:, 1]
        self.p[0, :] = self.p[1, :]
        self.p[-1, :] = self.p[-2, :]

    def check_convergence(self, tolerance=1e-6):
        """检查求解是否收敛"""
        # 计算速度场的残差
        u_residual = np.max(np.abs(self.u - self.u_old))
        v_residual = np.max(np.abs(self.v - self.v_old))

        # 计算温度场的残差
        t_residual = np.max(np.abs(self.T - self.T_old))

        # 更新旧值
        self.u_old = np.copy(self.u)
        self.v_old = np.copy(self.v)
        self.T_old = np.copy(self.T)

        # 检查是否所有残差都小于容差
        return max(u_residual, v_residual, t_residual) < tolerance

    def get_results(self):
        """获取模拟结果"""
        return {
            'temperature_field': np.copy(self.T),
            'velocity_field_u': np.copy(self.u),
            'velocity_field_v': np.copy(self.v),
            'pressure_field': np.copy(self.p)
        }

    def calculate_heat_transfer(self):
        """计算传热特性"""
        # 计算温度梯度
        dT_dx = np.gradient(self.T, self.dx, axis=1)
        dT_dy = np.gradient(self.T, self.dy, axis=0)

        # 计算热流密度
        q_x = -self.k * dT_dx
        q_y = -self.k * dT_dy

        # 计算总传热率
        Q_total = np.sum(np.sqrt(q_x ** 2 + q_y ** 2)) * self.dx * self.dy

        return {
            'heat_flux_x': q_x,
            'heat_flux_y': q_y,
            'total_heat_transfer': Q_total
        }

    def save_results(self, filename):
        """保存模拟结果"""
        results = {
            'temperature': self.T,
            'velocity_u': self.u,
            'velocity_v': self.v,
            'pressure': self.p,
            'heat_transfer': self.calculate_heat_transfer(),
            'parameters': {
                'nx': self.nx,
                'ny': self.ny,
                'dx': self.dx,
                'dy': self.dy,
                'dt': self.dt
            }
        }
        np.save(filename, results)
