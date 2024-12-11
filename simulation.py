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
        return odeint(self.equations, initial_conditions, time_span)

    def inverse_problem(self, trajectory):
        """
        Решение обратной задачи: подбор параметров модели для минимизации ошибки между
        наблюдаемой и симулированной траекториями.
        """
        # Извлекаем высоты (z-координата) из траектории
        heights = trajectory[:, 2]

        # Создаем начальные предположения для плотности воздуха и ветра
        initial_wind_speeds = np.zeros_like(heights)
        initial_air_densities = np.ones_like(heights) * 1.225  # Начальное значение плотности воздуха на уровне моря

        # Функция потерь для минимизации разницы между расчетной и реальной траекторией
        def loss_function(params):
            wind_speeds = params[:len(heights)]
            air_densities = params[len(heights):]

            # Рассчитываем траекторию заново на основе текущих параметров
            simulated_trajectory = self.simulate_with_params(wind_speeds, air_densities, heights)

            # Ошибка между модельной и реальной траекториями
            error = np.linalg.norm(simulated_trajectory - trajectory)
            return error

        # Начальное приближение для параметров
        initial_params = np.concatenate([initial_wind_speeds, initial_air_densities])

        # Оптимизация параметров с помощью метода наименьших квадратов
        result = minimize(loss_function, initial_params, method='L-BFGS-B')

        # Извлекаем оптимальные параметры
        optimized_params = result.x
        optimized_wind_speeds = optimized_params[:len(heights)]
        optimized_air_densities = optimized_params[len(heights):]

        return {
            'wind_speeds': optimized_wind_speeds,
            'air_densities': optimized_air_densities
        }

    def simulate_with_params(self, wind_speeds, air_densities, heights):
        """
        Симулирует траекторию с заданными профилями ветра и плотности воздуха.

        :param wind_speeds: np.array, массив скоростей ветра на разных высотах.
        :param air_densities: np.array, массив плотностей воздуха на разных высотах.
        :return: np.array, рассчитанная траектория (x, y, z) формы (N, 3).
        """
        def equations(y, t, wind_speeds, air_densities):
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
        time_span = np.linspace(0, len(wind_speeds)-1, len(wind_speeds))

        # Решение системы ОДУ
        solution = odeint(equations, initial_conditions, time_span, args=(wind_speeds, air_densities))

        # Возвращаем только координаты (x, y, z)
        return solution[:, :3]

