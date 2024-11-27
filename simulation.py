import numpy as np
from scipy.integrate import odeint
from constants import M_TOTAL
from physics import PhysicsModel


class BalloonSimulation:
    def __init__(self):
        """Инициализация симуляции."""
        self.m_total = M_TOTAL
        self.physics_model = PhysicsModel()  # Экземпляр PhysicsModel

    def equations(self, t, y):
        """Дифференциальные уравнения движения."""
        x, y, z, vx, vy, vz = y
        h = z  # Высота равна текущей координате z
        v = np.array([vx, vy, vz])  # Скорость как вектор

        # Получаем силы из PhysicsModel
        F = self.physics_model.forces(h, v, t)
        dvxdt = F[0] / self.m_total
        dvydt = F[1] / self.m_total
        dvzdt = F[2] / self.m_total

        return np.array([vx, vy, vz, dvxdt, dvydt, dvzdt])

    def runge_kutta_5(self, equations, initial_conditions, time_span, dt):
        """
        Метод Runge-Kutta 5-го порядка для численного интегрирования.

        :param equations: Функция, представляющая систему уравнений.
        :param initial_conditions: Начальные условия системы.
        :param time_span: Временной интервал в формате (t0, tf).
        :param dt: Шаг интегрирования.
        :return: Массив времени и матрица состояний.
        """
        t0, tf = time_span
        t = np.arange(t0, tf + dt, dt)  # Массив времени
        y = np.zeros((len(t), len(initial_conditions)))  # Матрица состояний
        y[0] = initial_conditions  # Установить начальные условия

        for i in range(len(t) - 1):
            k1 = dt * equations(t[i], y[i])
            k2 = dt * equations(t[i] + dt / 4, y[i] + k1 / 4)
            k3 = dt * equations(t[i] + 3 * dt / 8, y[i] + 3 / 32 * k1 + 9 / 32 * k2)
            k4 = dt * equations(t[i] + 12 * dt / 13, y[i] + 1932 / 2197 * k1 - 7200 / 2197 * k2 + 7296 / 2197 * k3)
            k5 = dt * equations(t[i] + dt, y[i] + 439 / 216 * k1 - 8 * k2 + 3680 / 513 * k3 - 845 / 4104 * k4)
            k6 = dt * equations(t[i] + dt / 2, y[i] - 8 / 27 * k1 + 2 * k2 - 3544 / 2565 * k3 + 1859 / 4104 * k4 - 11 / 40 * k5)
            y[i + 1] = y[i] + (16 / 135 * k1 + 6656 / 12825 * k3 + 28561 / 56430 * k4 - 9 / 50 * k5 + 2 / 55 * k6)

        return t, y

    def run_simulation(self, initial_conditions, t):
        """
        Запускает симуляцию с использованием встроенного метода SciPy `odeint`.

        :param initial_conditions: Начальные условия системы.
        :param t: Массив времени.
        :return: Результат численного решения.
        """
        return odeint(self.equations, initial_conditions, t)
