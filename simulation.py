from scipy.integrate import odeint
import numpy as np

from constants import M_TOTAL
from physics import forces, wind_profile


def equations(y, t):
    """Уравнения движения шара."""
    x, y, z, vx, vy, vz = y
    v = np.array([vx, vy, vz])
    F = forces(z, v, t)

    # TODO: учесть ветер
    dvdt = F / M_TOTAL

    return np.array([v[0], v[1], v[2], dvdt[0], dvdt[1], dvdt[2]])


def run_simulation(initial_conditions, t):
    """Запускает симуляцию."""
    return odeint(equations, initial_conditions, t)
