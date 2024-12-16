import numpy as np
from physics import PhysicsModel
from simulation import BalloonSimulation
from visualization import TrajectoryPlotter
from constants import *

# Создание объектов моделей
physics_model = PhysicsModel()
simulation = BalloonSimulation(0)
plotter = TrajectoryPlotter()

# Начальные условия для прямой задачи
initial_conditions = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, physics_model.air_density(0)], dtype=np.float64)  # x, y, z, vx, vy, vz
T_max = 100
t = np.linspace(0, T_max, T_max)  # Время в секундах (от 0 до T_max)
# dt = 0.1 # Шаг интегрирования

# Запуск симуляции
solution = simulation.run_simulation(initial_conditions, t)
# print(solution[:, -2])


# Начальные условия для обратной задачи
observed_trajectory = solution[:, :3]
# Обратная задача
result = simulation.inverse_problem(solution[:, :3])

# Построение траекторий
plotter.plot_trajectory(solution, result, t)

#
# fig = plt.figure(1, (5, 5))
# ax = fig.add_subplot(111, projection='3d')
# ax.plot(solution[:, 0], solution[:, 1], solution[:, 2], label='Траектория')
# x_max = np.max(np.abs(solution[:, 0]))
# plt.ylim(-x_max, x_max)
# ax.set_xlabel('X (м)')
# ax.set_ylabel('Y (м)')
# ax.set_zlabel('Высота (м)')
# plt.grid(True)
# ax.legend()
# plt.show()
