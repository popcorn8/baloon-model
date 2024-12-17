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
initial_conditions = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, np.linalg.norm(physics_model.wind_profile(0)), physics_model.air_density(0)], dtype=np.float64)  # x, y, z, vx, vy, vz
T_max = 1000
t = np.linspace(0, T_max, T_max)  # Время в секундах (от 0 до T_max)

# Запуск симуляции
solution = simulation.run_simulation(initial_conditions, t)

# Начальные условия для обратной задачи
observed_trajectory = solution[:, :3]
# Обратная задача
result = simulation.inverse_problem(solution[:, :3])

# Построение траекторий
plotter.plot_trajectory(solution, result, t)

