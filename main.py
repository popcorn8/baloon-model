import numpy as np
from physics import PhysicsModel
from simulation import BalloonSimulation
from visualization import TrajectoryPlotter
from constants import *

# Создание объектов моделей
physics_model = PhysicsModel()
simulation = BalloonSimulation(0)
plotter = TrajectoryPlotter()

# Начальные условия
initial_conditions = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, P0], dtype=np.float64)  # x, y, z, vx, vy, vz
t = np.linspace(0, 1800, 1800)  # Время в секундах (от 0 до 1 часа)
# dt = 0.1  # Шаг интегрирования

# Запуск симуляции
solution = simulation.run_simulation(initial_conditions, t)
# print(solution[:, -2])

# Обратная задача
simulation.inverse_problem(solution[:, :3])

# Построение траекторий
plotter.plot_trajectory(solution, t)
