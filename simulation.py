import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize
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
        # Учет ветра
        wind_vx, wind_vy, wind_vz = self.physics_model.wind_profile(h, t)
        v = np.array([vx, vy, vz]) + np.array([wind_vx, wind_vy, wind_vz])  # Скорость как вектор

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

    def inverse_problem(self, observed_trajectory, initial_guess, time_span, dt):
        """
        Решение обратной задачи: подбор параметров модели для минимизации ошибки между
        наблюдаемой и симулированной траекториями.

        :param observed_trajectory: Наблюдаемая траектория {'x': [...], 'y': [...], 'z': [...]}
        :param initial_guess: Начальные предположения для параметров.
        :param time_span: Временной интервал в формате (t0, tf).
        :param dt: Шаг интегрирования.
        :return: Оптимизированные параметры.
        """
        def cost_function(params):
            # Установка новых параметров в модель
            self.physics_model.update_parameters(params)

            # Запуск симуляции с текущими параметрами
            initial_conditions = [0, 0, 0, 0, 0, 0]  # Начальные условия
            t, simulated_trajectory = self.runge_kutta_5(self.equations, initial_conditions, time_span, dt)

            # Интерполяция для сравнения траекторий
            simulated_x = simulated_trajectory[:, 0]
            simulated_y = simulated_trajectory[:, 1]
            simulated_z = simulated_trajectory[:, 2]

            observed_x = observed_trajectory['x']
            observed_y = observed_trajectory['y']
            observed_z = observed_trajectory['z']

            # Выравниваем размеры массивов с помощью интерполяции
            min_len = min(len(observed_x), len(simulated_x))
            observed_x = np.interp(np.linspace(0, 1, min_len), np.linspace(0, 1, len(observed_x)), observed_x)
            observed_y = np.interp(np.linspace(0, 1, min_len), np.linspace(0, 1, len(observed_y)), observed_y)
            observed_z = np.interp(np.linspace(0, 1, min_len), np.linspace(0, 1, len(observed_z)), observed_z)
            simulated_x = simulated_x[:min_len]
            simulated_y = simulated_y[:min_len]
            simulated_z = simulated_z[:min_len]

            # Сумма квадратов отклонений
            error = np.sum((simulated_x - observed_x) ** 2 +
                           (simulated_y - observed_y) ** 2 +
                           (simulated_z - observed_z) ** 2)

            return error

        # Оптимизация с использованием метода 'Nelder-Mead'
        result = minimize(cost_function, initial_guess, method='Nelder-Mead')

        return result.x  # Возвращаем оптимальные параметры
