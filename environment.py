import numpy as np
from constants import G, RHO_AIR_SEA_LEVEL

def air_density(h):
    """Модель изменения плотности воздуха с высотой."""
    T0 = 288.15
    L = 0.0065
    R_air = 287.05
    P0 = 101325
    T = T0 - L * h
    P = P0 * (T / T0) ** (G / (R_air * L))
    rho = P / (R_air * T)
    return max(rho, 0)

def wind_profile(h, t):
    """Модель ветра."""
    base_speed = 2 + 0.1 * np.log1p(h + 1)
    turbulence = 0.5 * np.sin(2 * np.pi * t / 600)
    speed = base_speed + turbulence

    base_direction = 45 + 15 * np.sin(2 * np.pi * h / 10000)
    time_variation = 10 * np.sin(2 * np.pi * t / 1800)
    direction = base_direction + time_variation

    base_azimuth = 5 * np.sin(2 * np.pi * h / 5000)
    azimuth_turbulence = 2 * np.cos(2 * np.pi * t / 900)
    azimuth = base_azimuth + azimuth_turbulence

    wind_vx = speed * np.cos(np.radians(direction)) * np.cos(np.radians(azimuth))
    wind_vy = speed * np.sin(np.radians(direction)) * np.cos(np.radians(azimuth))
    wind_vz = speed * np.sin(np.radians(azimuth))

    return wind_vx, wind_vy, wind_vz
