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


def set_2d_spatial_axis_labels(ax, unit='Mm'):
    ax.set_xlabel('x [{}]'.format(unit))
    ax.set_ylabel('y [{}]'.format(unit))


def set_3d_spatial_axis_labels(ax, unit='Mm'):
    set_2d_spatial_axis_labels(ax, unit=unit)
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


def compute_histogram(values, weights, vmin, vmax, decide_bins_in_log_space,
                      weighted_average):

    min_value = np.nanmin(values)
    max_value = np.nanmax(values)

    if vmin is not None and vmin > min_value:
        min_value = vmin
    if vmax is not None and vmax < max_value:
        max_value = vmax

    if decide_bins_in_log_space:
        values = np.log10(values)
        min_value = np.log10(min_value)
        max_value = np.log10(max_value)

    hist, bin_edges = np.histogram(values,
                                   bins='auto',
                                   range=(min_value, max_value),
                                   weights=weights)

    if weights is not None and weighted_average:
        unweighted_hist, _ = np.histogram(values,
                                          bins=bin_edges,
                                          range=(min_value, max_value))
        hist /= unweighted_hist

    bin_centers = 0.5*(bin_edges[:-1] + bin_edges[1:])

    if decide_bins_in_log_space:
        bin_edges = 10**bin_edges
        bin_centers = 10**bin_centers

    return hist, bin_edges, bin_centers


def compute_histogram_difference(values, weights, vmin, vmax, bins,
                                 decide_bins_in_log_space):

    left_values, right_values = values
    left_weights, right_weights = weights

    min_value = min(np.nanmin(left_values), np.nanmin(right_values))
    max_value = max(np.nanmax(left_values), np.nanmax(right_values))

    if vmin is not None and vmin > min_value:
        min_value = vmin
    if vmax is not None and vmax < max_value:
        max_value = vmax

    if decide_bins_in_log_space:
        left_values = np.log10(left_values)
        right_values = np.log10(right_values)
        min_value = np.log10(min_value)
        max_value = np.log10(max_value)

    left_hist, bin_edges = np.histogram(left_values,
                                        bins=bins,
                                        range=(min_value, max_value),
                                        weights=left_weights)

    right_hist, _ = np.histogram(right_values,
                                 bins=bin_edges,
                                 range=(min_value, max_value),
                                 weights=right_weights)

    bin_centers = 0.5*(bin_edges[:-1] + bin_edges[1:])

    if decide_bins_in_log_space:
        bin_edges = 10**bin_edges
        bin_centers = 10**bin_centers

    return left_hist - right_hist, bin_edges, bin_centers


def compute_2d_histogram(values_x, values_y, weights, vmin_x, vmax_x, vmin_y,
                         vmax_y, log_x, log_y, bins_x, bins_y,
                         weighted_average):

    min_value_x = np.nanmin(values_x) if vmin_x is None else vmin_x
    max_value_x = np.nanmax(values_x) if vmax_x is None else vmax_x
    min_value_y = np.nanmin(values_y) if vmin_y is None else vmin_y
    max_value_y = np.nanmax(values_y) if vmax_y is None else vmax_y

    if vmin_x is not None and vmin_x > min_value_x:
        min_value_x = vmin_x
    if vmax_x is not None and vmax_x < max_value_x:
        max_value_x = vmax_x
    if vmin_y is not None and vmin_y > min_value_y:
        min_value_y = vmin_y
    if vmax_y is not None and vmax_y < max_value_y:
        max_value_y = vmax_y

    if log_x:
        values_x = np.log10(values_x)
        min_value_x = np.log10(min_value_x)
        max_value_x = np.log10(max_value_x)
    if log_y:
        values_y = np.log10(values_y)
        min_value_y = np.log10(min_value_y)
        max_value_y = np.log10(max_value_y)

    hist, bin_edges_x, bin_edges_y = np.histogram2d(
        values_x,
        values_y,
        bins=[bins_x, bins_y],
        range=[[min_value_x, max_value_x], [min_value_y, max_value_y]],
        weights=weights)

    if weights is not None and weighted_average:
        unweighted_hist, _, _ = np.histogram2d(
            values_x,
            values_y,
            bins=[bin_edges_x, bin_edges_y],
            range=[[min_value_x, max_value_x], [min_value_y, max_value_y]])
        hist /= unweighted_hist

    return hist, bin_edges_x, bin_edges_y


def compute_2d_histogram_difference(values_x, values_y, weights, vmin_x,
                                    vmax_x, vmin_y, vmax_y, log_x, log_y,
                                    bins_x, bins_y):

    left_values_x, right_values_x = values_x
    left_values_y, right_values_y = values_y
    left_weights, right_weights = weights

    left_min_value_x = np.nanmin(left_values_x) if vmin_x is None else vmin_x
    left_max_value_x = np.nanmax(left_values_x) if vmax_x is None else vmax_x
    left_min_value_y = np.nanmin(left_values_y) if vmin_y is None else vmin_y
    left_max_value_y = np.nanmax(left_values_y) if vmax_y is None else vmax_y

    right_min_value_x = np.nanmin(right_values_x) if vmin_x is None else vmin_x
    right_max_value_x = np.nanmax(right_values_x) if vmax_x is None else vmax_x
    right_min_value_y = np.nanmin(right_values_y) if vmin_y is None else vmin_y
    right_max_value_y = np.nanmax(right_values_y) if vmax_y is None else vmax_y

    min_value_x = min(left_min_value_x, right_min_value_x)
    max_value_x = max(left_max_value_x, right_max_value_x)
    min_value_y = min(left_min_value_y, right_min_value_y)
    max_value_y = max(left_max_value_y, right_max_value_y)

    if vmin_x is not None and vmin_x > min_value_x:
        min_value_x = vmin_x
    if vmax_x is not None and vmax_x < max_value_x:
        max_value_x = vmax_x
    if vmin_y is not None and vmin_y > min_value_y:
        min_value_y = vmin_y
    if vmax_y is not None and vmax_y < max_value_y:
        max_value_y = vmax_y

    if log_x:
        left_values_x = np.log10(left_values_x)
        right_values_x = np.log10(right_values_x)
        min_value_x = np.log10(min_value_x)
        max_value_x = np.log10(max_value_x)
    if log_y:
        left_values_y = np.log10(left_values_y)
        right_values_y = np.log10(right_values_y)
        min_value_y = np.log10(min_value_y)
        max_value_y = np.log10(max_value_y)

    left_hist, bin_edges_x, bin_edges_y = np.histogram2d(
        left_values_x,
        left_values_y,
        bins=[bins_x, bins_y],
        range=[[min_value_x, max_value_x], [min_value_y, max_value_y]],
        weights=left_weights)

    right_hist, _, _ = np.histogram2d(right_values_x,
                                      right_values_y,
                                      bins=[bin_edges_x, bin_edges_y],
                                      range=[[min_value_x, max_value_x],
                                             [min_value_y, max_value_y]],
                                      weights=right_weights)

    return left_hist - right_hist, bin_edges_x, bin_edges_y


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
