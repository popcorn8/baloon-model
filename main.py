import numpy as np
from physics import PhysicsModel
from simulation import BalloonSimulation
from visualization import TrajectoryPlotter

# Создание объектов моделей
physics_model = PhysicsModel()
simulation = BalloonSimulation()
plotter = TrajectoryPlotter()

# Начальные условия и параметры
initial_conditions = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)  # x, y, z, vx, vy, vz
time_span = (0, 3600)  # Время в секундах (от 0 до 1 часа)
dt = 0.1  # Шаг интегрирования

# Запуск симуляции методом Рунге-Кутта
t, solution_rk5 = simulation.runge_kutta_5(simulation.equations, initial_conditions, time_span, dt)


# Построение траекторий
plotter.plot_trajectory(solution_rk5, t, labels=['X (м)', 'Y (м)', 'Высота (м)'])
