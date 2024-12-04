import numpy as np
from constants import G, C_D, AREA, VOLUME, M_TOTAL, T0, R, M, P0, HELLMAN, WIND1


class PhysicsModel:
    def __init__(self, start_h=1, c_d=C_D, area=AREA, volume=VOLUME, m_total=M_TOTAL, ):
        """Инициализация физической модели."""
        self.C_D = c_d  # Коэффициент сопротивления
        self.AREA = area  # Эффективная площадь
        self.VOLUME = volume  # Объем тела
        self.M_TOTAL = m_total  # Общая масса
        self.START_H = start_h  # Начальная высота
        self.MAX_H = 11000  # Максимальная высота, рассматриваемая в симуляции (м)

    def temperature(self, h):
        """
        Рассчитывает температуру воздуха на заданной высоте
        :param h: Высота в м
        :return: Температура воздуха в К
        """
        return T0 - 0.0065 * h

    def air_density(self, h):
        """
        Рассчитывает плотность воздуха на заданной высоте.
        :param h: Высота в метрах.
        :return: Плотность воздуха в кг/м³.
        """
        # Температура на данной высоте
        T = self.temperature(h)
        # Давление на данной высоте - Барометрическая формула
        p = P0 * np.exp(-(M*G*h) / (R * T))
        # Уравнение Менделеева-Клапейрона
        rho = (p*M) / (R*T)
        return rho

    def wind_profile(self, h):
        """
        Рассчитывает параметры ветра на заданной высоте.
        Для упрощения используется формула для проектирования ветряных турбин
        :param h: Высота в метрах.
        :return: Компоненты скорости ветра (vx, vy, vz) в м/с.
        """
        wind_v = np.zeros(3)

        # Полагаем что скорость ветра до 1 м равна 0
        if h > 1:
            # Для упрощения ветер направлен всегда вдоль оси x
            wind_v[0] = WIND1 * (h**HELLMAN)

        return wind_v

    def forces(self, h, v):
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
        F_G[-1] = -(self.M_TOTAL * G)

        # Сила Архимеда
        F_A = np.zeros(len(v))
        F_A[-1] = rho_air * self.VOLUME * G

        # Компоненты силы сопротивления
        F_R = 0.5 * rho_air * v ** 2 * self.C_D * self.AREA * (-np.sign(v))

        return F_R + F_A + F_G
