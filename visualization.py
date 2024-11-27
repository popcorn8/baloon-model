import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class TrajectoryPlotter:
    def __init__(self, figsize=(10, 7)):
        """
        Инициализация объекта для построения графиков траектории.
        :param figsize: Размер фигуры графика (ширина, высота).
        """
        self.figsize = figsize

    def plot_trajectory(self, solution, t, labels=None):
        """
        Построение графика траектории объекта в 3D.
        :param solution: Массив решения с координатами [X, Y, Z] (Nx3).
        :param t: Массив времени (N элементов).
        :param labels: Список меток осей [xlabel, ylabel, zlabel]. Если None, используются значения по умолчанию.
        """
        if labels is None:
            labels = ['X (м)', 'Y (м)', 'Высота (м)']

        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(solution[:, 0], solution[:, 1], solution[:, 2], label='Траектория')
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.set_zlabel(labels[2])
        ax.legend()
        plt.show()
