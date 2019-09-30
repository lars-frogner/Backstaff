import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mpl_colors
import matplotlib.cm as mpl_cm
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.offsetbox import AnchoredText


def create_2d_subplots(dpi=200, **kwargs):
    return plt.subplots(dpi=dpi, **kwargs)


def set_2d_plot_extent(ax, x_lims, y_lims):
    ax.set_xlim(*x_lims)
    ax.set_ylim(*y_lims)


def create_3d_plot(dpi=200, **kwargs):
    fig = plt.figure(dpi=dpi, **kwargs)
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
    radius = 0.5*np.max(np.abs(limits[:, 1] - limits[:, 0]))
    set_axes_radius(ax, origin, radius)


def set_3d_spatial_axis_labels(ax, unit='Mm'):
    ax.set_xlabel('x [{}]'.format(unit))
    ax.set_ylabel('y [{}]'.format(unit))
    ax.set_zlabel('z [{}]'.format(unit))


def get_default_colors():
    return plt.rcParams['axes.prop_cycle'].by_key()['color']


def get_linear_normalizer(vmin, vmax, clip=False):
    return mpl_colors.Normalize(vmin=vmin, vmax=vmax, clip=clip)


def get_log_normalizer(vmin, vmax, clip=False):
    return mpl_colors.LogNorm(vmin=vmin, vmax=vmax, clip=clip)


def get_symlog_normalizer(vmin, vmax, linthresh, linscale=1.0, clip=False):
    return mpl_colors.SymLogNorm(linthresh,
                                 linscale=linscale,
                                 vmin=vmin,
                                 vmax=vmax,
                                 clip=clip)


def get_normalizer(vmin, vmax, clip=False, log=False):
    return get_log_normalizer(vmin, vmax,
                              clip=clip) if log else get_linear_normalizer(
                                  vmin, vmax, clip=clip)


def get_cmap(name):
    cmap = CUSTOM_COLORMAPS.get(name)
    return plt.get_cmap(name) if cmap is None else cmap


def define_linear_segmented_colormap(name,
                                     colors,
                                     bad_color='white',
                                     N=256,
                                     gamma=1.0):
    cmap = mpl_colors.LinearSegmentedColormap.from_list(name,
                                                        colors,
                                                        N=N,
                                                        gamma=gamma)
    cmap.set_bad(color=bad_color)
    return cmap


def colors_from_values(values,
                       log=False,
                       vmin=None,
                       vmax=None,
                       cmap_name='viridis',
                       alpha=1.0,
                       relative_alpha=True):
    cmap = get_cmap(cmap_name)
    normalized_values = get_normalizer(vmin, vmax, clip=False, log=log)(values)
    colors = cmap(normalized_values)

    if relative_alpha:
        colors[:, -1] = np.maximum(0.0,
                                   np.minimum(alpha, normalized_values*alpha))
    else:
        colors[:, -1] = alpha

    return colors


def add_2d_colorbar(fig, ax, mappeable, pad=0.05, label=None):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=pad)
    fig.colorbar(mappeable, cax=cax, label=label)


def add_3d_colorbar(fig, norm, cmap, label=None):
    sm = mpl_cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, label=label)


def add_3d_line_collection(ax, x, y, z, colors, lw=1.0):
    points = np.array([x, y, z]).T.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = Line3DCollection(segments, colors=colors)
    lc.set_linewidth(lw)
    return ax.add_collection(lc)


def add_textbox(ax, text, loc, pad=0.4):
    textbox = AnchoredText(text, loc, pad=pad)
    ax.add_artist(textbox)


def render(fig, tight_layout=True, output_path=None):
    if tight_layout:
        fig.tight_layout()
    if output_path is not None:
        plt.savefig(output_path)
    else:
        plt.show()


CUSTOM_COLORMAPS = {
    'transport':
    define_linear_segmented_colormap(
        'transport',
        np.vstack(
            (plt.get_cmap('Blues')(np.linspace(1, 0,
                                               128)), [[1.0, 1.0, 1.0, 1.0]],
             plt.get_cmap('Oranges')(np.linspace(0, 1, 128)))),
        bad_color='white',
        N=257)
}
