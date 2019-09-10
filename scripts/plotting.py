import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def create_2d_subplots(figsize=(8, 6), dpi=200, **kwargs):
    return plt.subplots(figsize=figsize, dpi=dpi, **kwargs)


def set_2d_plot_extent(ax, x_lims, y_lims):
    ax.set_xlim(*x_lims)
    ax.set_ylim(*y_lims)


def create_3d_plot(figsize=(8, 6), dpi=200, **kwargs):
    fig = plt.figure(figsize=figsize, dpi=dpi, **kwargs)
    ax = fig.add_subplot(111, projection='3d')
    return fig, ax


def set_3d_plot_extent(ax, x_lims, y_lims, z_lims):
    ax.set_xlim(*x_lims)
    ax.set_ylim(*y_lims)
    ax.set_zlim(*z_lims)
    set_3d_axes_equal(ax)


def set_3d_axes_equal(ax):

    def set_axes_radius(ax, origin, radius):
        ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
        ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
        ax.set_zlim3d([origin[2] - radius, origin[2] + radius])

    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])

    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    set_axes_radius(ax, origin, radius)


def render(fig, output_path=None):
    fig.tight_layout()
    if output_path is not None:
        plt.savefig(output_path)
    else:
        plt.show()