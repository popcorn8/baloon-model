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
        coords_noise = np.random.normal(0, 0.1, solution[:, :3].shape)
        solution[:, :3] += coords_noise
        return solution

    def inverse_problem(self, observed_trajectory):
        """
        Решает обратную задачу методом наименьших квадратов.

        :param observed_trajectory: np.array, наблюдаемая траектория (x, y, z) формы (N, 3).
        :return: tuple (оптимальные скорости ветра, оптимальные плотности воздуха).
        """
        heights = observed_trajectory[:, 2]
        initial_wind_speeds = 0
        initial_air_densities = 100000

        def error_func(params):
            # Разделяем параметры на скорости ветра и плотности воздуха
            wind_speeds = params[0]
            air_densities = params[1]

            # Симуляция траектории с текущими параметрами
            simulated_trajectory = self.simulate_with_params(wind_speeds, air_densities, heights)[:, :3]

            # Вычисляем разницу между наблюдаемой и симулированной траекториями
            error = np.linalg.norm(simulated_trajectory - observed_trajectory) ** 2
            # print(f"Error: {error}")
            return error

        # Начальные параметры для оптимизации
        initial_params = np.array([initial_wind_speeds, initial_air_densities])

        # Решение задачи минимизации
        result = least_squares(error_func, initial_params)

        # Извлекаем оптимальные параметры
        optimized_wind_speeds = result.x[0]
        optimized_air_densities = result.x[1]

        result = self.simulate_with_params(optimized_wind_speeds, optimized_air_densities, heights)

        return result

    def simulate_with_params(self, wind_speeds, air_densities, heights):
        """
        Симулирует траекторию с заданными профилями ветра и плотности воздуха.

        :param wind_speeds: np.array, массив скоростей ветра на разных высотах.
        :param air_densities: np.array, массив плотностей воздуха на разных высотах.
        :param heights: np.array, массив высот для интерполяции.
        :return: np.array, рассчитанная траектория (x, y, z) формы (N, 3).
        """
        self.physics_model.WIND1 = wind_speeds
        self.physics_model.P0 = air_densities

        def equations(y, t):
            """Дифференциальные уравнения движения с учетом заданных параметров."""
            x, y, z, vx, vy, vz, wind_v, air_density = y
            h = z  # Высота равна текущей координате z

            # Рассчет скорости ветра и плотности воздуха по высоте относительно заданных начальных параметров
            wind_v = self.physics_model.wind_profile(h)
            air_density = self.physics_model.air_density(h)

            # Учет ветра
            vx, vy, vz = v = np.array([vx, vy, vz]) + wind_v  # Скорость как вектор

            # Модель сил (пример, замените реальной физической моделью)
            F = self.physics_model.forces(h, v, air_density)
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

            return np.array([vx, vy, vz, dvxdt, dvydt, dvzdt, dwindt_dh, dair_density_dh])

        # Начальные условия
        initial_conditions = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, self.physics_model.air_density(0)], dtype=np.float64)  # x, y, z, vx, vy, vz
        time_span = np.linspace(0, len(heights)-1, len(heights))

        # Решение системы ОДУ
        solution, info = odeint(equations, initial_conditions, time_span, full_output=True)
        #
        # fig = plt.figure(1, (5, 5))
        # ax = fig.add_subplot(111, projection='3d')
        # ax.plot(solution[:, 0], solution[:, 1], solution[:, 2], label='Траектория')
        # x_max = np.max(np.abs(solution[:, 0]))
        # plt.ylim(-x_max, x_max)
        # ax.set_xlabel('X (м)')
        # ax.set_ylabel('Y (м)')
        # ax.set_zlabel('Высота (м)')
        # plt.grid(True)
        # ax.legend()
        # plt.show()

        # Возвращаем только координаты (x, y, z)
        return solution
