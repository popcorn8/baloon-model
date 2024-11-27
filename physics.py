import numpy as np

from constants import G, C_D, AREA, VOLUME, M_TOTAL
from environment import air_density, wind_profile

def forces(h, v, vx, vy, vz, t):
    """Вычисляет силы в системе."""
    rho_air = air_density(h)
    F_gravity = M_TOTAL * G
    F_buoyancy = rho_air * VOLUME * G
    F_drag_vert = 0.5 * rho_air * v ** 2 * C_D * AREA * np.sign(v)
    F_drag_hor_x = 0.5 * rho_air * vx ** 2 * C_D * AREA * np.sign(vx)
    F_drag_hor_y = 0.5 * rho_air * vy ** 2 * C_D * AREA * np.sign(vy)
    F_drag_hor_z = 0.5 * rho_air * vz ** 2 * C_D * AREA * np.sign(vz)

    wind_vx, wind_vy, wind_vz = wind_profile(h, t)

    return {
        "F_gravity": F_gravity,
        "F_buoyancy": F_buoyancy,
        "F_drag_vert": F_drag_vert,
        "F_drag_hor_x": F_drag_hor_x,
        "F_drag_hor_y": F_drag_hor_y,
        "F_drag_hor_z": F_drag_hor_z,
        "wind_vx": wind_vx,
        "wind_vy": wind_vy,
        "wind_vz": wind_vz,
    }
