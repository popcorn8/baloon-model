import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize
from constants import M_TOTAL
from physics import PhysicsModel


class BalloonSimulation:
    def __init__(self, start_h):
        """Инициализация симуляции."""
        self.m_total = M_TOTAL
        self.physics_model = PhysicsModel()  # Экземпляр PhysicsModel
        self.START_H = start_h  # Начальная высота
        self.MAX_H = 11000  # Максимальная высота, рассматриваемая в симуляции (м)

    def equations(self, y, t):
        """Дифференциальные уравнения движения."""
        x, y, z, vx, vy, vz, wind_v, air_density = y
        h = z  # Высота равна текущей координате z
        # Учет ветра
        wind_v = self.physics_model.wind_profile(h)
        # print(f"WINDV {wind_v}")
        vx, vy, vz = v = np.array([vx, vy, vz]) + wind_v  # Скорость как вектор
        # Плотность воздуха
        rho_air = self.physics_model.air_density(h)
        # Получаем силы из PhysicsModel
        F = self.physics_model.forces(h, v, rho_air)
        dvxdt = F[0] / self.m_total
        dvydt = F[1] / self.m_total
        dvzdt = F[2] / self.m_total

        # Шаг для численного дифференцирования по высоте
        dh = vz

        # Производная скорости ветра по высоте
        dwindt_dh = self.physics_model.wind_v_dh(h, dh)
        # print(f"DWINDV {wind_v}")
        # Производная плотности воздуха по высоте
        dair_density_dh = self.physics_model.air_density_dh(h, dh)

        # Возвращаем изменения координат, скоростей, производные скорости ветра и плотности воздуха
        return np.array([vx, vy, vz, dvxdt, dvydt, dvzdt, dwindt_dh, dair_density_dh])

    def run_simulation(self, initial_conditions, time_span):
        solution = np.array(odeint(self.equations, initial_conditions, time_span))
        coords_noise = np.random.normal(0, 0.01, solution[:, :3].shape)
        solution[:, :3] += coords_noise
        return solution

    def inverse_problem(self, trajectory):
        """
        Решение обратной задачи: восстановление профиля ветра и плотности воздуха
        по траектории шарика.

        :param trajectory: np.array, массив формы (N, 3), где N - количество точек времени,
                           а каждая строка содержит координаты (x, y, z).
        :return: dict, содержащий восстановленные профили скорости ветра и плотности воздуха.
                 {
                     'wind_speeds': np.array (N,),
                     'air_densities': np.array (N,)
                 }
        """
        # Извлекаем высоты (z-координата) из траектории
        heights = trajectory[:, 2]

        # Создаем начальные предположения для плотности воздуха и ветра
        initial_wind_speeds = np.zeros_like(heights)
        initial_air_densities = np.ones_like(heights) * 1.225  # Начальное значение плотности воздуха на уровне моря

        # Функция для расчета разницы между расчетной и реальной траекториями
        def residuals(params):
            wind_speeds, air_densities = unpack_params(params)
            simulated_trajectory = self.simulate_with_params(wind_speeds, air_densities, heights)
            return (simulated_trajectory - trajectory).flatten()  # Плоский массив для метода наименьших квадратов

        # Вспомогательная функция для упаковки параметров
        def pack_params(wind_speeds, air_densities):
            return np.concatenate([wind_speeds, air_densities])

        # Вспомогательная функция для распаковки параметров
        def unpack_params(params):
            wind_speeds = params[:len(heights)]
            air_densities = params[len(heights):]
            return wind_speeds, air_densities

        # Собственная реализация метода наименьших квадратов
        def custom_least_squares(residuals_func, initial_params, max_iter=100, tol=1e-2):
            params = initial_params.copy()
            for _ in range(max_iter):
                residuals = residuals_func(params)
                jacobian = compute_jacobian(residuals_func, params)
                delta = np.linalg.lstsq(jacobian, -residuals, rcond=None)[0]
                params += delta
                if np.linalg.norm(delta) < tol:
                    break
            return params

        # Вычисление Якобиана численным методом
        def compute_jacobian(residuals_func, params, epsilon=1e-2):
            n_params = len(params)
            n_residuals = len(residuals_func(params))
            jacobian = np.zeros((n_residuals, n_params))
            for i in range(n_params):
                params_step = params.copy()
                params_step[i] += epsilon
                jacobian[:, i] = (residuals_func(params_step) - residuals_func(params)) / epsilon
            return jacobian

        # Начальное приближение для параметров
        initial_params = pack_params(initial_wind_speeds, initial_air_densities)

        # Оптимизация параметров с помощью собственной реализации метода наименьших квадратов
        optimized_params = custom_least_squares(residuals, initial_params)

        # Извлекаем оптимальные параметры
        optimized_wind_speeds, optimized_air_densities = unpack_params(optimized_params)

        return {
            'wind_speeds': optimized_wind_speeds,
            'air_densities': optimized_air_densities
        }

    def simulate_with_params(self, wind_speeds, air_densities, heights):
        """
        Симулирует траекторию с заданными профилями ветра и плотности воздуха.

        :param wind_speeds: np.array, массив скоростей ветра на разных высотах.
        :param air_densities: np.array, массив плотностей воздуха на разных высотах.
        :param heights: np.array, массив высот для интерполяции.
        :return: np.array, рассчитанная траектория (x, y, z) формы (N, 3).
        """
        def equations(y, t):
            """Дифференциальные уравнения движения с учетом заданных параметров."""
            x, y, z, vx, vy, vz = y
            h = z  # Высота равна текущей координате z

            # Интерполяция скорости ветра и плотности воздуха по высоте
            wind_v = np.interp(h, heights, wind_speeds)
            air_density = np.interp(h, heights, air_densities)

            # Учет ветра
            vx, vy, vz = v = np.array([vx, vy, vz]) + np.array([wind_v, 0, 0])  # Скорость как вектор

            # Модель сил (пример, замените реальной физической моделью)
            F = self.physics_model.forces(h, v, air_density)
            dvxdt = F[0] / self.m_total
            dvydt = F[1] / self.m_total
            dvzdt = F[2] / self.m_total

            return np.array([vx, vy, vz, dvxdt, dvydt, dvzdt])

        # Начальные условия
        initial_conditions = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)  # x, y, z, vx, vy, vz
        time_span = np.linspace(0, len(heights)-1, len(heights))

        # Решение системы ОДУ
        solution = odeint(equations, initial_conditions, time_span, full_output=True)

        # Возвращаем только координаты (x, y, z)
        return solution[:, :3]

