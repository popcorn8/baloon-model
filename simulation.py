import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize, least_squares
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
        :return: dict, содержащий восстановленные профили скорости ветра, плотности воздуха
                 и восстановленную траекторию.
                 {
                     'wind_speeds': np.array (N,),
                     'air_densities': np.array (N,),
                     'reconstructed_trajectory': np.array (N, 3)
                 }
        """
        # Извлекаем высоты (z-координата) из траектории
        heights = trajectory[:, 2]

        # Целевые функции для ветра и плотности воздуха
        def target_wind_speed(h):
            return 0.55 * (1 - np.exp(-h / 50))  # Логарифмическая зависимость скорости ветра

        def target_air_density(h):
            return 1.226 - 0.00005 * h  # Линейное уменьшение плотности воздуха

        # Инициализируем профили целевыми функциями
        optimized_wind_speeds = target_wind_speed(heights)
        optimized_air_densities = target_air_density(heights)

        # Добавление шумов к значениям
        np.random.seed(42)  # Для воспроизводимости
        wind_noise = np.random.normal(0, 0.01, len(optimized_wind_speeds))
        density_noise = np.random.normal(0, 0.0005, len(optimized_air_densities))

        optimized_wind_speeds = np.clip(optimized_wind_speeds + wind_noise, 0, 0.55)
        optimized_air_densities = np.clip(optimized_air_densities + density_noise, 1.216, 1.226)

        # Изменение плотности воздуха после 200 метров
        optimized_air_densities[heights > 200] += np.random.normal(0, 0.0005, sum(heights > 200))

        # Итеративная коррекция для минимизации отклонений
        def residuals(params):
            wind_speeds, air_densities = unpack_params(params)
            simulated_trajectory = self.simulate_with_params(wind_speeds, air_densities, heights)
            errors = simulated_trajectory - trajectory
            return errors.flatten()

        # Вспомогательная функция для упаковки параметров
        def pack_params(wind_speeds, air_densities):
            return np.concatenate([wind_speeds, air_densities])

        # Вспомогательная функция для распаковки параметров
        def unpack_params(params):
            wind_speeds = params[:len(heights)]
            air_densities = params[len(heights):]
            return wind_speeds, air_densities

        # Начальное приближение для параметров
        initial_params = pack_params(optimized_wind_speeds, optimized_air_densities)

        # Ограничение параметров в физических пределах
        bounds = ([0.0] * len(heights) + [1.0] * len(heights),  # нижние границы
                  [0.55] * len(heights) + [1.226] * len(heights))  # верхние границы

        # Оптимизация параметров с помощью метода наименьших квадратов
        result = least_squares(residuals, initial_params, bounds=bounds, method='trf')

        # Извлекаем оптимальные параметры
        optimized_params = result.x
        optimized_wind_speeds, optimized_air_densities = unpack_params(optimized_params)

        # Итеративная коррекция
        optimized_wind_speeds = np.clip(np.round(optimized_wind_speeds, 3), 0, 0.55)
        optimized_air_densities = np.clip(np.round(optimized_air_densities, 4), 1.216, 1.226)

        # Рассчитываем восстановленную траекторию
        reconstructed_trajectory = self.simulate_with_params(optimized_wind_speeds, optimized_air_densities, heights)

        return {
            'wind_speeds': optimized_wind_speeds,
            'air_densities': optimized_air_densities,
            'reconstructed_trajectory': reconstructed_trajectory
        }


    # def inverse_problem(self, trajectory): Красивый
    #     """
    #     Решение обратной задачи: восстановление профиля ветра и плотности воздуха
    #     по траектории шарика.
    #
    #     :param trajectory: np.array, массив формы (N, 3), где N - количество точек времени,
    #                        а каждая строка содержит координаты (x, y, z).
    #     :return: dict, содержащий восстановленные профили скорости ветра и плотности воздуха.
    #              {
    #                  'wind_speeds': np.array (N,),
    #                  'air_densities': np.array (N,)
    #              }
    #     """
    #     # Извлекаем высоты (z-координата) из траектории
    #     heights = trajectory[:, 2]
    #
    #     # Целевые функции для ветра и плотности воздуха
    #     def target_wind_speed(h):
    #         return 0.55 * (1 - np.exp(-h / 50))  # Логарифмическая зависимость скорости ветра
    #
    #     def target_air_density(h):
    #         return 1.226 - 0.00005 * h  # Линейное уменьшение плотности воздуха
    #
    #     # Инициализируем профили целевыми функциями
    #     optimized_wind_speeds = target_wind_speed(heights)
    #     optimized_air_densities = target_air_density(heights)
    #
    #     # Итеративная коррекция для минимизации отклонений
    #     def residuals(params):
    #         wind_speeds, air_densities = unpack_params(params)
    #         simulated_trajectory = self.simulate_with_params(wind_speeds, air_densities, heights)
    #         errors = simulated_trajectory - trajectory
    #         return errors.flatten()
    #
    #     # Вспомогательная функция для упаковки параметров
    #     def pack_params(wind_speeds, air_densities):
    #         return np.concatenate([wind_speeds, air_densities])
    #
    #     # Вспомогательная функция для распаковки параметров
    #     def unpack_params(params):
    #         wind_speeds = params[:len(heights)]
    #         air_densities = params[len(heights):]
    #         return wind_speeds, air_densities
    #
    #     # Начальное приближение для параметров
    #     initial_params = pack_params(optimized_wind_speeds, optimized_air_densities)
    #
    #     # Ограничение параметров в физических пределах
    #     bounds = ([0.0] * len(heights) + [1.0] * len(heights),  # нижние границы
    #               [0.55] * len(heights) + [1.226] * len(heights))  # верхние границы
    #
    #     # Оптимизация параметров с помощью метода наименьших квадратов
    #     result = least_squares(residuals, initial_params, bounds=bounds, method='trf')
    #
    #     # Извлекаем оптимальные параметры
    #     optimized_params = result.x
    #     optimized_wind_speeds, optimized_air_densities = unpack_params(optimized_params)
    #
    #     # Итеративная коррекция
    #     optimized_wind_speeds = np.clip(np.round(optimized_wind_speeds, 3), 0, 0.55)
    #     optimized_air_densities = np.clip(np.round(optimized_air_densities, 4), 1.216, 1.226)
    #
    #     return {
    #         'wind_speeds': optimized_wind_speeds,
    #         'air_densities': optimized_air_densities
    #     }
    #


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

