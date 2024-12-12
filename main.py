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
initial_conditions = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, physics_model.air_density(0)], dtype=np.float64)  # x, y, z, vx, vy, vz
T_max = 100
t = np.linspace(0, T_max, T_max)  # Время в секундах (от 0 до T_max)
# dt = 0.1  # Шаг интегрирования

# Запуск симуляции
solution = simulation.run_simulation(initial_conditions, t)
# print(solution[:, -2])

# Запуск обратной задачи
results = simulation.inverse_problem(solution[:, :3])



# Построение траекторий
plotter.plot_trajectory(solution, t)

# Построение графиков
plotter.plot_trajectory_inverse(results["reconstructed_trajectory"])
plotter.plot_inverse_results(solution[:, 2], results['wind_speeds'], results['air_densities'])

