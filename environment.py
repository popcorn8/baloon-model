import numpy as np
from constants import G, RHO_AIR_SEA_LEVEL

class AtmosphereModel:
    def __init__(self):
        """Инициализация параметров атмосферы."""
        self.T0 = 288.15  # Температура на уровне моря в Кельвинах
        self.L = 0.0065   # Градиент температуры с высотой в К/м
        self.R_air = 287.05  # Универсальная газовая постоянная для воздуха в Дж/(кг·К)
        self.P0 = 101325  # Давление на уровне моря в Па

    def air_density(self, h):
        """
        Рассчитывает плотность воздуха на заданной высоте.
        :param h: Высота в метрах.
        :return: Плотность воздуха в кг/м³.
        """
        T = self.T0 - self.L * h
        P = self.P0 * (T / self.T0) ** (G / (self.R_air * self.L))
        rho = P / (self.R_air * T)
        return max(rho, 0)

    def wind_profile(self, h, t):
        """
        Рассчитывает параметры ветра на заданной высоте и времени.
        :param h: Высота в метрах.
        :param t: Время в секундах.
        :return: Компоненты скорости ветра (vx, vy, vz) в м/с.
        """
        # Скорость ветра
        base_speed = 2 + 0.1 * np.log1p(h + 1)
        turbulence = 0.5 * np.sin(2 * np.pi * t / 600)
        speed = base_speed + turbulence

        # Направление ветра
        base_direction = 45 + 15 * np.sin(2 * np.pi * h / 10000)
        time_variation = 10 * np.sin(2 * np.pi * t / 1800)
        direction = base_direction + time_variation

        # Азимутальное направление
        base_azimuth = 5 * np.sin(2 * np.pi * h / 5000)
        azimuth_turbulence = 2 * np.cos(2 * np.pi * t / 900)
        azimuth = base_azimuth + azimuth_turbulence

        # Компоненты ветра
        wind_vx = speed * np.cos(np.radians(direction)) * np.cos(np.radians(azimuth))
        wind_vy = speed * np.sin(np.radians(direction)) * np.cos(np.radians(azimuth))
        wind_vz = speed * np.sin(np.radians(azimuth))

        return wind_vx, wind_vy, wind_vz
