import numpy as np
from simulation import run_simulation
from visualization import plot_trajectory

# Начальные условия
h0, v0, x0, vx0, y0, vy0, z0, vz0 = 0, 0, 0, 0, 0, 0, 0, 0
initial_conditions = [h0, v0, x0, vx0, y0, vy0, z0, vz0]

# Временной интервал
t = np.linspace(0, 3600, 1000)

# Запуск симуляции
solution = run_simulation(initial_conditions, t)

# Визуализация
plot_trajectory(solution, t)