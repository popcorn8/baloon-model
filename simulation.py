from scipy.integrate import odeint

from constants import M_TOTAL
from physics import forces

def equations(y, t):
    """Уравнения движения шара."""
    h, v, x, vx, y_pos, vy, z, vz = y
    F = forces(h, v, vx, vy, vz, t)

    dvdt = (F["F_buoyancy"] - F["F_gravity"] - F["F_drag_vert"]) / M_TOTAL
    dhdt = v
    dvxdt = (-F["F_drag_hor_x"] / M_TOTAL) + F["wind_vx"] / 100
    dxdt = vx
    dvy_dt = (-F["F_drag_hor_y"] / M_TOTAL) + F["wind_vy"] / 100
    dy_dt = vy
    dvzdt = (-F["F_drag_hor_z"] / M_TOTAL) + F["wind_vz"] / 100
    dzdt = vz

    return [dhdt, dvdt, dxdt, dvxdt, dy_dt, dvy_dt, dzdt, dvzdt]

def run_simulation(initial_conditions, t):
    """Запускает симуляцию."""
    return odeint(equations, initial_conditions, t)
