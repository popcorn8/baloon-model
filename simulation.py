import numpy as np
from scipy.integrate import odeint
from scipy.optimize import least_squares
from constants import M_TOTAL
from physics import PhysicsModel
import matplotlib.pyplot as plt


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

    def inverse_problem(self, observed_trajectory, max_iterations=100, tolerance=1e-6):
        """
        Решает обратную задачу методом наименьших квадратов с использованием итеративного подхода.

        :param observed_trajectory: np.array, наблюдаемая траектория (x, y, z) формы (N, 3).
        :param max_iterations: int, максимальное количество итераций для оптимизации.
        :param tolerance: float, порог ошибки для остановки оптимизации.
        :return: tuple (оптимальные скорости ветра, оптимальные плотности воздуха).
        """
        heights = observed_trajectory[:, 2]
        initial_wind_speeds = np.zeros_like(heights)
        initial_air_densities = np.ones_like(heights) * 1.227  # Плотность воздуха на уровне моря

        def compute_error(params):
            """
            Вычисляет текущую ошибку.
            :param params: np.array, текущие параметры (ветер и плотность).
            :return: ошибка
            """
            wind_speeds = params[:len(heights)]
            air_densities = params[len(heights):]
            simulated_trajectory = self.simulate_with_params(wind_speeds, air_densities, heights)
            return np.linalg.norm(simulated_trajectory - observed_trajectory) ** 2

        def compute_gradient(params):
            """
            Численное вычисление градиента ошибки по параметрам.
            :param params: np.array, текущие параметры (ветер и плотность).
            :return: градиент
            """
            gradient = np.zeros_like(params)
            delta = 1e-5
            base_error = compute_error(params)
            for i in range(len(params)):
                perturbed_params = params.copy()
                perturbed_params[i] += delta
                perturbed_error = compute_error(perturbed_params)
                gradient[i] = (perturbed_error - base_error) / delta
            return gradient

        # Начальные параметры
        params = np.concatenate([initial_wind_speeds, initial_air_densities])

        for iteration in range(max_iterations):
            # Вычисляем текущую ошибку и градиент
            current_error = compute_error(params)
            gradient = compute_gradient(params)

            # Обновляем параметры вручную
            step_size = 1e-3  # Размер шага оптимизации
            params -= step_size * gradient

            print(f"Iteration {iteration + 1}: Error = {current_error}")

            if current_error < tolerance:
                print("Converged successfully.")
                break

        else:
            print("Reached maximum iterations without full convergence.")

        # Разделяем оптимизированные параметры на скорости ветра и плотности воздуха
        optimized_wind_speeds = params[:len(heights)]
        optimized_air_densities = params[len(heights):]

        return optimized_wind_speeds, optimized_air_densities

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
        solution, info = odeint(equations, initial_conditions, time_span, full_output=True)

        # Возвращаем только координаты (x, y, z)
        return solution[:, :3]
