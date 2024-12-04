import numpy as np
import scipy as sc
from physics import PhysicsModel
from simulation import BalloonSimulation
from visualization import TrajectoryPlotter

# Создание объектов моделей
physics_model = PhysicsModel()
simulation = BalloonSimulation()
plotter = TrajectoryPlotter()

# Начальные условия
initial_conditions = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)  # x, y, z, vx, vy, vz
t = np.linspace(0, 3600, 3600)  # Время в секундах (от 0 до 1 часа)
dt = 0.1  # Шаг интегрирования

# Запуск симуляции методом Рунге-Кутта
solution_rk5 = simulation.run_simulation(initial_conditions, t, dt)

# Построение траекторий
plotter.plot_trajectory(solution_rk5, t, labels=['X (м)', 'Y (м)', 'Высота (м)'])
