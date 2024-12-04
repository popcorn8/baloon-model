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
        wind_v = self.physics_model.wind_profile(h)
        v = np.array([vx, vy, vz]) + wind_v  # Скорость как вектор

        # Получаем силы из PhysicsModel
        F = self.physics_model.forces(h, v)
        dvxdt = F[0] / self.m_total
        dvydt = F[1] / self.m_total
        dvzdt = F[2] / self.m_total

        return np.array([vx, vy, vz, dvxdt, dvydt, dvzdt])

    def runge_kutta_5(self, equations, initial_conditions, t, dt):
        """
        Метод Runge-Kutta 5-го порядка для численного интегрирования.

        :param equations: Функция, представляющая систему уравнений.
        :param initial_conditions: Начальные условия системы.
        :param t: Временной интервал.
        :param dt: Шаг интегрирования.
        :return: Массив времени и матрица состояний.
        """
        # t0, tf = time_span
        # t = np.arange(t0, tf + dt, dt)  # Массив времени
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

        return y

    def run_simulation(self, initial_conditions, time_span, dt):
        return self.runge_kutta_5(self.equations, initial_conditions, time_span, dt)

    # TODO: обратную задачу пока нельзя решить, тк нет выборки данных по скорости ветра и плотности воздуха
    """
    а эти выборки мы не можем получить, тк данные изменяются в зависимости от высоты по заданным формулам
    то есть нужно как то добавить в вектор данных dwind_v_dh и drho_air_dh, но тк там у нас переменные зависящие от
    времени, возможно это будет некорректно
    ПОКА ХЗ КАК ЭТО РЕШИТЬ
    """
    def inverse_problem(self):
        """
        Решение обратной задачи: подбор параметров модели для минимизации ошибки между
        наблюдаемой и симулированной траекториями.

        """
        # Исходные данные
        time = np.linspace(0, 100, 100)  # Временная сетка
        x_observed = np.random.rand(100, 3)  # Наблюдаемая траектория (пример)

        # Функция потерь
        def loss_function(params, x_observed, time):
            x_calculated = model_trajectory(params, time)
            error = np.linalg.norm(x_observed - x_calculated, axis=1)
            regularization = np.sum(np.gradient(params[:len(time)]) ** 2)  # Регуляризация
            return np.sum(error ** 2) + 0.01 * regularization

        # Начальное приближение
        initial_params = np.random.rand(2 * len(time))  # Плотность и скорость ветра

        # Оптимизация
        result = minimize(loss_function, initial_params, args=(x_observed, time), method='L-BFGS-B')

        # Результаты
        optimized_params = result.x
        rho_optimized = optimized_params[:len(time)]
        vw_optimized = optimized_params[len(time):]
