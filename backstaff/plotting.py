import numpy as np
import matplotlib as mpl
#mpl.use('agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mpl_colors
import matplotlib.cm as mpl_cm
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.offsetbox import AnchoredText


def create_2d_subplots(dpi=200, **kwargs):
    return plt.subplots(dpi=dpi, **kwargs)


def set_2d_plot_extent(ax, x_lims, y_lims):
    if x_lims is not None:
        ax.set_xlim(*x_lims)
    if y_lims is not None:
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


def set_2d_axis_labels(ax, xlabel, ylabel, xcolor='k', ycolor='k'):
    ax.set_xlabel(xlabel, color=xcolor)
    ax.set_ylabel(ylabel, color=ycolor)


def set_3d_axis_labels(ax, xlabel, ylabel, zlabel):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)


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


def get_cmap(name, bad_color='w'):
    cmap = CUSTOM_COLORMAPS[
        name] if name in CUSTOM_COLORMAPS else plt.get_cmap(name)
    cmap.set_bad(bad_color)
    return cmap


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


def colors_from_values(values, norm, cmap, alpha=1.0, relative_alpha=True):
    normalized_values = norm(values)
    colors = cmap(normalized_values)

    if relative_alpha:
        colors[:, -1] = np.maximum(0.0,
                                   np.minimum(alpha, normalized_values*alpha))
    else:
        colors[:, -1] = alpha

    return colors


def create_colorbar_axis(ax, loc='right', pad=0.05):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(loc, size='5%', pad=pad)
    return cax


def add_2d_colorbar(fig, ax, mappeable, loc='right', pad=0.05, label=''):
    cax = create_colorbar_axis(ax, loc=loc, pad=pad)
    fig.colorbar(
        mappeable,
        cax=cax,
        label=label,
        orientation=('vertical' if loc in ['left', 'right'] else 'horizontal'),
        ticklocation=loc)


def add_2d_colorbar_from_cmap_and_norm(fig,
                                       ax,
                                       norm,
                                       cmap,
                                       loc='right',
                                       pad=0.05,
                                       label=''):
    cax = create_colorbar_axis(ax, loc=loc, pad=pad)
    sm = mpl_cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(
        sm,
        cax=cax,
        label=label,
        orientation=('vertical' if loc in ['left', 'right'] else 'horizontal'),
        ticklocation=loc)


def add_3d_colorbar(fig, norm, cmap, label=''):
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


def render(fig, tight_layout=True, output_path=None, force_show=False):
    if tight_layout:
        fig.tight_layout()
    if output_path is not None:
        plt.savefig(output_path)
        if force_show:
            plt.show()
    else:
        plt.show()


def plot_2d_field(hor_coords,
                  vert_coords,
                  values,
                  vmin=None,
                  vmax=None,
                  log=False,
                  symlog=False,
                  linthresh=np.inf,
                  linscale=1.0,
                  cmap_name='viridis',
                  cmap_bad_color='white',
                  xlabel=None,
                  ylabel=None,
                  clabel='',
                  output_path=None):
    if symlog:
        norm = get_symlog_normalizer(vmin, vmax, linthresh, linscale=linscale)
    else:
        norm = get_normalizer(vmin, vmax, log=log)

    fig, ax = create_2d_subplots()

    im = ax.pcolormesh(*np.meshgrid(hor_coords, vert_coords),
                       values.T,
                       norm=norm,
                       vmin=vmin,
                       vmax=vmax,
                       cmap=get_cmap(cmap_name, bad_color=cmap_bad_color))

    set_2d_plot_extent(ax, (hor_coords[0], hor_coords[-1]),
                       (vert_coords[0], vert_coords[-1]))
    set_2d_axis_labels(ax, xlabel, ylabel)
    add_2d_colorbar(fig, ax, im, label=clabel)

    ax.set_aspect('equal')

    render(fig, output_path=output_path)


def setup_line_animation(fig,
                         ax,
                         initial_coordinates,
                         update_coordinates,
                         x_lims=None,
                         y_lims=None,
                         invert_xaxis=False,
                         invert_yaxis=False,
                         ds='default',
                         ls='-',
                         lw=1.0,
                         c='navy',
                         alpha=1.0,
                         xlabel=None,
                         ylabel=None,
                         show_time_label=False,
                         extra_init_setup=lambda ax: None):

    line, = ax.plot([], [], ds=ds, ls=ls, lw=lw, c=c, alpha=alpha)
    text = ax.text(0.01, 0.99, '', transform=ax.transAxes, ha='left',
                   va='top') if show_time_label else None

    def init():
        line.set_data(*initial_coordinates)
        extra_init_setup(ax)
        set_2d_plot_extent(ax, x_lims, y_lims)
        set_2d_axis_labels(ax, xlabel, ylabel)
        if invert_xaxis:
            ax.invert_xaxis()
        if invert_yaxis:
            ax.invert_yaxis()
        if text is None:
            return line,
        else:
            text.set_text('t = 0 s')
            return line, text

    def update(frame):
        time, coordinates = update_coordinates()
        line.set_data(*coordinates)
        if text is None:
            return line,
        else:
            text.set_text('t = {:g} s'.format(time))
            return line, text

    return init, update


def setup_scatter_animation(fig,
                            ax,
                            initial_coordinates,
                            update_coordinates,
                            x_lims=None,
                            y_lims=None,
                            invert_xaxis=False,
                            invert_yaxis=False,
                            marker='o',
                            s=1.0,
                            c='navy',
                            edgecolors='none',
                            alpha=1.0,
                            xlabel=None,
                            ylabel=None,
                            show_time_label=False,
                            extra_init_setup=lambda ax: None):

    sc = ax.scatter([], [],
                    marker=marker,
                    s=s,
                    c=c,
                    edgecolors=edgecolors,
                    alpha=alpha)
    text = ax.text(0.01, 0.99, '', transform=ax.transAxes, ha='left',
                   va='top') if show_time_label else None

    def init():
        sc.set_offsets(initial_coordinates)
        extra_init_setup(ax)
        set_2d_plot_extent(ax, x_lims, y_lims)
        set_2d_axis_labels(ax, xlabel, ylabel)
        if invert_xaxis:
            ax.invert_xaxis()
        if invert_yaxis:
            ax.invert_yaxis()
        if text is None:
            return sc,
        else:
            text.set_text('t = 0 s')
            return sc, text

    def update(frame):
        time, coordinates = update_coordinates()
        sc.set_offsets(coordinates)
        if text is None:
            return sc,
        else:
            text.set_text('t = {:g} s'.format(time))
            return sc, text

    return init, update


def animate(fig,
            init_func,
            update_func,
            blit=False,
            fps=30.0,
            video_duration=None,
            tight_layout=False,
            writer='ffmpeg',
            codec='h264',
            dpi=None,
            bitrate=None,
            output_path=None):

    interval = 1e3/fps
    n_frames = None if video_duration is None else int(video_duration*fps)

    anim = animation.FuncAnimation(fig,
                                   update_func,
                                   init_func=init_func,
                                   frames=n_frames,
                                   blit=blit,
                                   interval=interval)

    if tight_layout:
        fig.tight_layout()

    if output_path is None:
        plt.show()
    else:
        assert n_frames is not None
        anim.save(
            output_path,
            writer=writer,
            codec=codec,
            dpi=dpi,
            bitrate=bitrate,
            fps=fps,
            progress_callback=lambda i, n: print(
                'Animation progress: {:4.1f}%'.format(i*100.0/n), end='\r'))


def compute_histogram(values,
                      weights=None,
                      bins='auto',
                      vmin=None,
                      vmax=None,
                      decide_bins_in_log_space=False,
                      weighted_average=False,
                      density=False):

    min_value = np.nanmin(values) if vmin is None else vmin
    max_value = np.nanmax(values) if vmax is None else vmax

    if decide_bins_in_log_space:
        values = np.log10(values)
        min_value = np.log10(min_value)
        max_value = np.log10(max_value)

    hist, bin_edges = np.histogram(values,
                                   bins=bins,
                                   range=(min_value, max_value),
                                   weights=weights,
                                   density=density)

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

    min_value = min(np.nanmin(left_values),
                    np.nanmin(right_values)) if vmin is None else vmin
    max_value = max(np.nanmax(left_values),
                    np.nanmax(right_values)) if vmax is None else vmax

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

    left_hist, bin_edges_x, bin_edges_y = compute_2d_histogram(
        left_values_x, left_values_y, left_weights, vmin_x, vmax_x, vmin_y,
        vmax_y, log_x, log_y, bins_x, bins_y, False)

    right_hist, _, _ = compute_2d_histogram(right_values_x, right_values_y,
                                            right_weights, vmin_x, vmax_x,
                                            vmin_y, vmax_y, log_x, log_y,
                                            bins_x, bins_y, False)

    return left_hist - right_hist, bin_edges_x, bin_edges_y


CUSTOM_COLORMAPS = {
    'transport':
    define_linear_segmented_colormap(
        '',
        np.vstack(
            (plt.get_cmap('Blues')(np.linspace(1, 0,
                                               256)), [[1.0, 1.0, 1.0, 1.0]],
             plt.get_cmap('Oranges')(np.linspace(0, 1, 256)))),
        bad_color='white',
        N=513),
    'afternoon':
    define_linear_segmented_colormap(
        '', ['#8C0004', '#C8000A', '#E8A735', '#E2C499']),
    'timeless':
    define_linear_segmented_colormap(
        '', ['#16253D', '#002C54', '#EFB509', '#CD7213']),
    'arctic':
    define_linear_segmented_colormap(
        '', ['#006C84', '#6EB5C0', '#E2E8E4', '#FFCCBB']),
    'sunkissed':
    define_linear_segmented_colormap(
        '', ['#D24136', '#EB8A3E', '#EBB582', '#785A46']),
    'berry':
    define_linear_segmented_colormap(
        '', ['#D0E1F9', '#4D648D', '#283655', '#1E1F26']),
    'sunset':
    define_linear_segmented_colormap(
        '', ['#363237', '#2D4262', '#73605B', '#D09683']),
    'watery':
    define_linear_segmented_colormap(
        '', ['#021C1E', '#004445', '#2C7873', '#6FB98F']),
    'bright':
    define_linear_segmented_colormap(
        '', ['#061283', '#FD3C3C', '#FFB74C', '#138D90']),
    'school':
    define_linear_segmented_colormap(
        '', ['#81715E', '#FAAE3D', '#E38533', '#E4535E']),
    'golden':
    define_linear_segmented_colormap(
        '', ['#323030', '#CDBEA7', '#C29545', '#882426']),
    'misty':
    define_linear_segmented_colormap(
        '', ['#04202C', '#2C493F', '#5B7065', '#C9D1C8']),
    'coolblues':
    define_linear_segmented_colormap(
        '', ['#003B46', '#07575B', '#66A5AD', '#C4DFE6']),
    'candy':
    define_linear_segmented_colormap(
        '', ['#AD1457', '#D81B60', '#FFA000', '#FDD835', '#FFEE58'])
}

CB_COLOR_CYCLE = [
    '#dc143c', '#377eb8', '#ff7f00', '#4daf4a', '#984ea3', '#a65628',
    '#f781bf', '#999999', '#dede00'
]
