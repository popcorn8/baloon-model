import numpy as np
from scipy.integrate import odeint
from scipy.ndimage import median_filter
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
        F = self.physics_model.forces(v, rho_air)
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

    def inverse_problem(self, observed_trajectory, max_iterations=1000, tolerance=0.11):
        """
        Решает обратную задачу методом Гаусса-Ньютона.

        :param observed_trajectory: np.array, наблюдаемая траектория (x, y, z) формы (N, 3).
        :param max_iterations: int, максимальное количество итераций для оптимизации.
        :param tolerance: float, порог ошибки для остановки оптимизации.
        :return: оптимальные скорости ветра, оптимальные плотности воздуха (вместе с полученной оптимизированной траекторией).
        """
        heights = observed_trajectory[:, 2]
        initial_wind = 0
        initial_air = 100000

        def compute_residuals(params):
            """
            Вычисляет текущую ошибку.
            :param params: np.array, текущие параметры (ветер и плотность).
            :return: ошибка
            """
            wind_param = params[0]
            air_param = params[1]
            simulated_trajectory = self.simulate_with_params(wind_param, air_param, heights)
            residuals = np.zeros_like(heights)
            for i in range(len(heights)):
                residuals[i] = np.linalg.norm(simulated_trajectory[i, :3] - observed_trajectory[i]) ** 2
            return residuals

        def compute_jacobian(params):
            """
            Численное вычисление Якобиана функции ошибки по параметрам.
            :param params: np.array[2], текущие параметры (ветер и плотность).
            :return: Якобиан
            """
            wind_param = params[0]
            air_param = params[1]
            jacobian = np.zeros((len(heights), 2))
            delta = 1e-3
            base_residuals = compute_residuals(params)
            perturbed_wind_params = params.copy()
            perturbed_wind_params[0] += delta
            perturbed_wind_residuals = compute_residuals(perturbed_wind_params)
            perturbed_air_params = params.copy()
            perturbed_air_params[1] += delta
            perturbed_air_residuals = compute_residuals(perturbed_air_params)
            for i in range(len(heights)):
                jacobian[i, 0] = (perturbed_wind_residuals[i] - base_residuals[i]) / delta
                jacobian[i, 1] = (perturbed_air_residuals[i] - base_residuals[i]) / delta
            return jacobian

        def solve_svd(A, b):
            """
            Решает уравнение Ax = b с использованием SVD-разложения.
            :param A: np.array, произвольная квадратная или прямоугольная матрица.
            :param b: np.array, вектор правой части.
            :return: np.array, решение x.
            """
            U, S, Vt = np.linalg.svd(A, full_matrices=False)
            S_inv = np.diag(1 / S)  # Обратные сингулярные числа
            x = Vt.T @ S_inv @ U.T @ b  # Решение с использованием SVD
            return x

        # Начальные параметры
        params = np.array([initial_wind, initial_air], dtype=np.float64)

        lambda_damp = 1e-3  # Начальный дампинг
        last_error = 1e+16
        # Решаем методом Гаусса-Ньютона
        for iteration in range(max_iterations):
            # Вычисляем текущую ошибку и градиент
            current_residuals = compute_residuals(params)
            jacobian = compute_jacobian(params)

            # Записываем компоненты уравнения Ax = b
            A = jacobian.T @ jacobian
            A_damped = jacobian.T @ jacobian + lambda_damp * np.eye(jacobian.shape[1])
            b = -jacobian.T @ current_residuals

            # Решение с помощью SVD разложения
            delta_params = solve_svd(A_damped, b)

            # Обновляем параметры
            new_params = params + delta_params

            # Выводим текущую ошибку
            current_error = np.linalg.norm(current_residuals)
            print(f"Iteration {iteration + 1}: Error = {current_error}")
            if current_error < last_error:
                params = new_params
                lambda_damp *= 0.5  # Уменьшаем дампинг
            else:
                lambda_damp *= 2.0  # Увеличиваем дампинг

            if current_error < tolerance:
                print("Converged successfully.")
                break

        else:
            print("Reached maximum iterations without full convergence.")

        # Разделяем оптимизированные параметры на скорости ветра и плотности воздуха
        optimized_wind_param = params[0]
        optimized_air_param = params[1]

        result = self.simulate_with_params(optimized_wind_param, optimized_air_param, heights)

        return result

    def simulate_with_params(self, wind_param, air_param, heights):
        """
        Симулирует траекторию с заданными профилями ветра и плотности воздуха.

        :param wind_param: скорость ветра.
        :param air_param: плотность воздуха.
        :param heights: массив высот из наблюдений.
        :return: np.array, рассчитанные данные.
        """
        self.physics_model.WIND1 = wind_param
        self.physics_model.P0 = air_param

        # Начальные условия
        initial_conditions = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, np.linalg.norm(self.physics_model.wind_profile(0)), self.physics_model.air_density(0)], dtype=np.float64)
        time_span = np.linspace(0, len(heights)-1, len(heights))

        # Решение системы ОДУ
        solution, info = odeint(self.equations, initial_conditions, time_span, full_output=True)

        return solution
