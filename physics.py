import numpy as np

from constants import G, C_D, AREA, VOLUME, M_TOTAL, T0, RHO_0, B, A


def air_density(h):
    """Модель изменения плотности воздуха с высотой."""
    return RHO_0 * np.exp((-B * h * T0) / (T0 - A * h))


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


def forces(h, v, t):
    """Вычисляет силы в системе."""
    rho_air = air_density(h)
    F_G = np.zeros(len(v))
    F_A = np.zeros(len(v))
    F_G[-1] = -(M_TOTAL * G)  # сила тяжести
    F_A[-1] = rho_air * VOLUME * G  # сила Архимеда

    # Учитываем ветер
    wind_vx, wind_vy, wind_vz = wind_profile(h, t)

    # Компоненты силы сопротивления
    F_R = 0.5 * rho_air * v ** 2 * C_D * AREA * (-np.sign(v))

    return F_R + F_A + F_G