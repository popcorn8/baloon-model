import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_trajectory(solution, t):
    """Графики траектории и скорости."""
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(solution[:, 2], solution[:, 4], solution[:, 0], label='Траектория')
    ax.set_xlabel('X (м)')
    ax.set_ylabel('Y (м)')
    ax.set_zlabel('Высота (м)')
    ax.legend()
    plt.show()
