import numpy as np
from constants import G, C_D, AREA, VOLUME, M_TOTAL, T0, RHO_0, B, A

class PhysicsModel:
    def __init__(self):
        """Инициализация физической модели."""
        self.G = G  # Ускорение свободного падения
        self.C_D = C_D  # Коэффициент сопротивления
        self.AREA = AREA  # Эффективная площадь
        self.VOLUME = VOLUME  # Объем тела
        self.M_TOTAL = M_TOTAL  # Общая масса
        self.T0 = T0  # Температура на уровне моря (К)
        self.RHO_0 = RHO_0  # Плотность воздуха на уровне моря (кг/м³)
        self.B = B  # Коэффициент модели атмосферы
        self.A = A  # Градиент температуры

    def update_parameters(self, params):
        """
        Обновляет параметры модели на основе переданных значений.
        :param params: Список параметров [C_D, AREA, VOLUME, ...].
        """
        self.C_D, self.AREA, self.VOLUME, self.M_TOTAL = params

    def air_density(self, h):
        """
        Рассчитывает плотность воздуха на заданной высоте.
        :param h: Высота в метрах.
        :return: Плотность воздуха в кг/м³.
        """
        return self.RHO_0 * np.exp((-self.B * h * self.T0) / (self.T0 - self.A * h))

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

    def forces(self, h, v, t):
        """
        Вычисляет силы, действующие на объект.
        :param h: Высота в метрах.
        :param v: Скорость объекта (массив компонентов).
        :param t: Время в секундах.
        :return: Массив сил (по компонентам).
        """
        rho_air = self.air_density(h)

        # Сила тяжести
        F_G = np.zeros(len(v))
        F_G[-1] = -(self.M_TOTAL * self.G)

        # Сила Архимеда
        F_A = np.zeros(len(v))
        F_A[-1] = rho_air * self.VOLUME * self.G

        # Компоненты силы сопротивления
        F_R = 0.5 * rho_air * v ** 2 * self.C_D * self.AREA * (-np.sign(v))

        return F_R + F_A + F_G
