import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

class TrajectoryPlotter:
    def __init__(self, figsize=(15, 6)):
        """
        Инициализация объекта для построения графиков.
        :param figsize: Размер фигуры графика (ширина, высота).
        """
        self.figsize = figsize

    def plot_trajectory(self, solution, t):
        """
        Построение графика траектории объекта в 3D.
        :param solution: Массив решения с координатами [X, Y, Z] (Nx3).
        :param t: Массив времени (N элементов).
        """

        # Траектория
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(131, projection='3d')
        ax.plot(solution[:, 0], solution[:, 1], solution[:, 2], label='Траектория')
        x_max = np.max(np.abs(solution[:, 0]))
        plt.ylim(-x_max, x_max)
        ax.set_xlabel('X (м)')
        ax.set_ylabel('Y (м)')
        ax.set_zlabel('Высота (м)')
        plt.grid(True)
        ax.legend()

        ax = fig.add_subplot(132)
        ax.plot(solution[:, 2], solution[:, -2], label='Скорость ветра')
        ax.set_ylabel('Wind V')
        ax.set_xlabel('Высота (м)')
        plt.grid(True)
        ax.legend()

        ax = fig.add_subplot(133)
        ax.plot(solution[:, 2], solution[:, -1], label='Плотность воздуха')
        ax.set_ylabel('RHO')
        ax.set_xlabel('Высота (м)')
        ax.legend()

        plt.show()

    def plot_inverse_results(self, heights, wind_speeds, air_densities):
        """
        Построение графиков профилей ветра и плотности воздуха, восстановленных в обратной задаче.
        :param heights: Массив высот.
        :param wind_speeds: Восстановленный массив скоростей ветра.
        :param air_densities: Восстановленный массив плотностей воздуха.
        """
        fig, axs = plt.subplots(1, 2, figsize=self.figsize)

        # График скоростей ветра
        axs[0].plot(heights, wind_speeds, label='Скорость ветра')
        axs[0].set_xlabel('Высота (м)')
        axs[0].set_ylabel('Скорость ветра (м/с)')
        axs[0].legend()
        axs[0].grid(True)

        # График плотности воздуха
        axs[1].plot(heights, air_densities, label='Плотность воздуха', color='orange')
        axs[1].set_xlabel('Высота (м)')
        axs[1].set_ylabel('Плотность воздуха (кг/м³)')
        axs[1].legend()
        axs[1].grid(True)

        plt.tight_layout()
        plt.show()

    def plot_trajectory_inverse(self, solution):
        """
        Построение графика траектории полета.

        :param solution: np.array, массив траектории формы (N, 3), содержащий координаты (x, y, z).

        """

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(solution[:, 0], solution[:, 1], solution[:, 2], label='Траектория полета')
        ax.set_xlabel('X Координата')
        ax.set_ylabel('Y Координата')
        ax.set_zlabel('Высота (Z)')
        ax.set_title('Траектория движения воздушного шара')
        ax.legend()
        plt.show()
