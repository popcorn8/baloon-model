import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

class TrajectoryPlotter:
    def __init__(self, figsize=(16, 9)):
        """
        Инициализация объекта для построения графиков траектории.
        :param figsize: Размер фигуры графика (ширина, высота).
        """
        self.figsize = figsize

    def plot_trajectory(self, solution, result, t):
        """
        Построение графика траектории объекта в 3D.
        :param solution: Массив решения с координатами [X, Y, Z] (Nx3).
        :param t: Массив времени (N элементов).
        """

        # Траектория
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(231, projection='3d')
        ax.plot(solution[:, 0], solution[:, 1], solution[:, 2], label='Траектория')
        x_max = np.max(np.abs(solution[:, 0]))
        plt.ylim(-x_max, x_max)
        ax.set_xlabel('X (м)')
        ax.set_ylabel('Y (м)')
        ax.set_zlabel('Высота (м)')
        plt.grid(True)
        ax.legend()

        ax = fig.add_subplot(232)
        ax.plot(solution[:, 2], solution[:, -2], label='Скорость ветра')
        ax.set_ylabel('Wind V')
        ax.set_xlabel('Высота (м)')
        plt.grid(True)
        ax.legend()

        ax = fig.add_subplot(233)
        ax.plot(solution[:, 2], solution[:, -1], label='Плотность воздуха')
        ax.set_ylabel('RHO')
        ax.set_xlabel('Высота (м)')
        ax.legend()

        ax = fig.add_subplot(234, projection='3d')
        ax.plot(result[:, 0], result[:, 1], result[:, 2], label='Оптимизированная траектория')
        x_max = np.max(np.abs(result[:, 0]))
        plt.ylim(-x_max, x_max)
        ax.set_xlabel('X (м)')
        ax.set_ylabel('Y (м)')
        ax.set_zlabel('Высота (м)')
        plt.grid(True)
        ax.legend()

        ax = fig.add_subplot(235)
        ax.plot(solution[:, 2], result[:, -2], label='Оптимизированная скорость ветра')
        ax.set_ylabel('Wind V')
        ax.set_xlabel('Высота (м)')
        plt.grid(True)
        ax.legend()

        ax = fig.add_subplot(236)
        ax.plot(solution[:, 2], result[:, -1], label='Оптимизированная плотность воздуха')
        ax.set_ylabel('RHO')
        ax.set_xlabel('Высота (м)')
        ax.legend()

        plt.show()
