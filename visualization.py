import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

class TrajectoryPlotter:
    def __init__(self, figsize=(15, 6)):
        """
        Инициализация объекта для построения графиков траектории.
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
