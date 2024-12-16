import numpy as np
from constants import G, C_D, AREA, VOLUME, M_TOTAL, T0, R, M, P0, HELLMAN, WIND1


class PhysicsModel:
    def __init__(self, c_d=C_D, area=AREA, volume=VOLUME, m_total=M_TOTAL):
        """Инициализация физической модели."""
        self.C_D = c_d  # Коэффициент сопротивления
        self.AREA = area  # Эффективная площадь
        self.VOLUME = volume  # Объем тела
        self.M_TOTAL = m_total  # Общая масса

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

    def air_density_dh(self, h, dh):
        """
        Производная плотности воздуха по высоте
        :param h:
        :param dh:
        :return:
        """
        air_density = self.air_density(h)
        air_density_higher = self.air_density(h + dh)
        if air_density == 0 and air_density_higher == 0 or air_density == air_density_higher:
            return 0
        return (air_density_higher - air_density) / dh

    def wind_v_dh(self, h, dh):
        """
        Производная скорости ветра по высоте
        :param h:
        :param dh:
        :return:
        """
        wind_v = np.linalg.norm(self.wind_profile(h))
        wind_v_higher = np.linalg.norm(self.wind_profile(h + dh))
        if wind_v == 0 and wind_v_higher == 0 or wind_v == wind_v_higher:
            return 0
        return (wind_v_higher - wind_v) / dh

    def forces(self, h, v, rho_air):
        """
        Вычисляет силы, действующие на объект.
        :param h: Высота в метрах.
        :param v: Скорость объекта (массив компонентов).
        :param t: Время в секундах.
        :return: Массив сил (по компонентам).
        """

        # Сила тяжести
        F_G = np.zeros(len(v))
        F_G[-1] = -(self.M_TOTAL * G)

        # Сила Архимеда
        F_A = np.zeros(len(v))
        F_A[-1] = rho_air * self.VOLUME * G

        # Компоненты силы сопротивления
        F_R = 0.5 * rho_air * v ** 2 * self.C_D * self.AREA * (-np.sign(v))

        return F_R + F_A + F_G
