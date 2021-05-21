import copy
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
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.offsetbox import AnchoredText


def create_figure(width=6.0, aspect_ratio=4.0/3.0, dpi=300, **kwargs):
    return plt.figure(figsize=kwargs.pop('figsize',
                                         (width, width/aspect_ratio)),
                      dpi=dpi,
                      **kwargs)


def create_2d_subplots(width=6.0, aspect_ratio=4.0/3.0, dpi=300, **kwargs):
    return plt.subplots(figsize=kwargs.pop('figsize',
                                           (width, width/aspect_ratio)),
                        dpi=dpi,
                        **kwargs)


def set_2d_plot_extent(ax, x_lims, y_lims):
    if x_lims is not None:
        ax.set_xlim(*x_lims)
    if y_lims is not None:
        ax.set_ylim(*y_lims)


def create_3d_plot(width=6.0, aspect_ratio=4.0/3.0, dpi=200, **kwargs):
    fig = plt.figure(figsize=kwargs.pop('figsize',
                                        (width, width/aspect_ratio)),
                     dpi=dpi,
                     **kwargs)
    ax = fig.add_subplot(111, projection='3d')
    return fig, ax


def set_3d_plot_extent(ax, x_lims, y_lims, z_lims, axes_equal=True):
    if x_lims is not None:
        ax.set_xlim(*x_lims)
    if y_lims is not None:
        ax.set_ylim(*y_lims)
    if z_lims is not None:
        ax.set_zlim(*z_lims)
    if axes_equal:
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
                                 clip=clip,
                                 base=np.e)


def get_normalizer(vmin, vmax, clip=False, log=False):
    return get_log_normalizer(vmin, vmax,
                              clip=clip) if log else get_linear_normalizer(
                                  vmin, vmax, clip=clip)


def get_cmap(name, bad_color='w'):
    cmap = copy.copy(CUSTOM_COLORMAPS[name] if name in
                     CUSTOM_COLORMAPS else plt.get_cmap(name))
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


def size_from_values(values, norm, max_size, min_size=1):
    normalized_values = norm(values)
    return np.maximum(min_size,
                      np.minimum(max_size, max_size*normalized_values**2))


def create_colorbar_axis(ax, loc='right', pad=0.05):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(loc, size='5%', pad=pad)
    return cax


def add_2d_colorbar(fig,
                    ax,
                    mappeable,
                    loc='right',
                    pad=0.05,
                    minorticks_on=False,
                    opposite_side_ticks=False,
                    tick_formatter=None,
                    label=''):
    cax = create_colorbar_axis(ax, loc=loc, pad=pad)

    cb = fig.colorbar(
        mappeable,
        cax=cax,
        label=label,
        orientation=('vertical' if loc in ['left', 'right'] else 'horizontal'),
        ticklocation=loc)

    if minorticks_on:
        cb.ax.minorticks_on()

    tick_ax = cb.ax.yaxis if loc in ['left', 'right'] else cb.ax.xaxis

    if opposite_side_ticks:
        side = {
            'left': 'right',
            'right': 'left',
            'bottom': 'top',
            'top': 'bottom'
        }[loc]
        tick_ax.set_label_position(side)
        tick_ax.set_ticks_position(side)

    if tick_formatter is not None:
        tick_ax.set_major_formatter(tick_formatter)


def add_2d_colorbar_inside_from_cmap_and_norm(fig,
                                              ax,
                                              norm,
                                              cmap,
                                              loc='upper left',
                                              tick_loc='bottom',
                                              width='30%',
                                              height='3%',
                                              orientation='horizontal',
                                              label='',
                                              fontsize='small',
                                              minorticks_on=False,
                                              tick_formatter=None):

    cax = inset_axes(ax, width=width, height=height, loc=loc)

    sm = mpl_cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    cb = fig.colorbar(sm,
                      cax=cax,
                      orientation=orientation,
                      ticklocation=tick_loc,
                      label=label)

    cb.set_label(label=label, fontsize=fontsize)

    if minorticks_on:
        cb.ax.minorticks_on()

    tick_ax = cb.ax.yaxis if tick_loc in ['left', 'right'] else cb.ax.xaxis

    cb.ax.tick_params(labelsize=fontsize)

    if tick_formatter is not None:
        tick_ax.set_major_formatter(tick_formatter)


def add_2d_colorbar_from_cmap_and_norm(fig,
                                       ax,
                                       norm,
                                       cmap,
                                       loc='right',
                                       pad=0.05,
                                       minorticks_on=False,
                                       opposite_side_ticks=False,
                                       tick_formatter=None,
                                       label=''):
    cax = create_colorbar_axis(ax, loc=loc, pad=pad)
    sm = mpl_cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(
        sm,
        cax=cax,
        label=label,
        orientation=('vertical' if loc in ['left', 'right'] else 'horizontal'),
        ticklocation=loc)

    if minorticks_on:
        cb.ax.minorticks_on()

    tick_ax = cb.ax.yaxis if loc in ['left', 'right'] else cb.ax.xaxis

    if opposite_side_ticks:
        side = {
            'left': 'right',
            'right': 'left',
            'bottom': 'top',
            'top': 'bottom'
        }[loc]
        tick_ax.set_label_position(side)
        tick_ax.set_ticks_position(side)

    if tick_formatter is not None:
        tick_ax.set_major_formatter(tick_formatter)


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


def render(fig,
           tight_layout=True,
           output_path=None,
           force_show=False,
           close=True):
    if tight_layout:
        fig.tight_layout()
    if output_path is not None:
        plt.savefig(output_path)
        if force_show:
            plt.show()
    else:
        plt.show()
    if close:
        plt.close(fig)


def plot_2d_field(hor_coords,
                  vert_coords,
                  values,
                  fig=None,
                  ax=None,
                  figure_width=6.0,
                  figure_aspect=4.0/3.0,
                  minorticks_on=False,
                  vmin=None,
                  vmax=None,
                  log=False,
                  symlog=False,
                  linthresh=np.inf,
                  linscale=1.0,
                  cmap_name='viridis',
                  cmap_bad_color='white',
                  cbar_loc='right',
                  cbar_pad=0.05,
                  cbar_minorticks_on=False,
                  cbar_opposite_side_ticks=False,
                  contour_levels=None,
                  contour_colors='r',
                  contour_alpha=1.0,
                  log_contour=False,
                  vmin_contour=None,
                  vmax_contour=None,
                  contour_cmap_name='viridis',
                  extra_artists=[],
                  xlabel=None,
                  ylabel=None,
                  clabel='',
                  title=None,
                  rasterized=None,
                  output_path=None,
                  render_now=True):

    if fig is None or ax is None:
        fig, ax = create_2d_subplots(width=figure_width,
                                     aspect_ratio=figure_aspect)

    if symlog:
        norm = get_symlog_normalizer(vmin, vmax, linthresh, linscale=linscale)
    else:
        norm = get_normalizer(vmin, vmax, log=log)

    mesh = ax.pcolormesh(*np.meshgrid(hor_coords, vert_coords),
                         values.T,
                         shading='auto',
                         norm=norm,
                         cmap=get_cmap(cmap_name, bad_color=cmap_bad_color),
                         rasterized=rasterized)

    if contour_levels is not None:
        ax.contourf(hor_coords,
                    vert_coords,
                    values.T,
                    levels=contour_levels,
                    norm=get_normalizer(vmin_contour,
                                        vmax_contour,
                                        log=log_contour),
                    cmap=get_cmap(contour_cmap_name),
                    colors=contour_colors,
                    alpha=contour_alpha,
                    rasterized=rasterized)

    if extra_artists is not None:
        for artist in extra_artists:
            ax.add_artist(artist)

    set_2d_plot_extent(ax, (hor_coords[0], hor_coords[-1]),
                       (vert_coords[0], vert_coords[-1]))
    set_2d_axis_labels(ax, xlabel, ylabel)

    if cbar_loc is not None:
        add_2d_colorbar(fig,
                        ax,
                        mesh,
                        loc=cbar_loc,
                        pad=cbar_pad,
                        minorticks_on=cbar_minorticks_on,
                        opposite_side_ticks=cbar_opposite_side_ticks,
                        label=clabel)

    if minorticks_on:
        ax.minorticks_on()

    ax.set_aspect('equal')

    if title is not None:
        ax.set_title(title)

    if render_now:
        render(fig, output_path=output_path)

    return fig, ax, mesh


def plot_histogram(values,
                   fig=None,
                   ax=None,
                   weighted=False,
                   divided_by_bin_size=False,
                   hist_scale=1.0,
                   bins='auto',
                   log_x=False,
                   log_y=False,
                   vmin=None,
                   vmax=None,
                   plot_type='steps',
                   horizontal=False,
                   fit_limits=None,
                   extra_artists=[],
                   color='k',
                   lw=1.0,
                   ls='-',
                   alpha=1.0,
                   x_lims=None,
                   y_lims=None,
                   label=None,
                   legend_loc=None,
                   xlabel=None,
                   ylabel=None,
                   xlabel_color='k',
                   ylabel_color='k',
                   minorticks_on=False,
                   output_path=None,
                   fig_kwargs={},
                   render_now=True):

    if fig is None or ax is None:
        fig, ax = create_2d_subplots(**fig_kwargs)

    hist, bin_edges, bin_centers = compute_histogram(
        values,
        weights=None,
        bins=bins,
        vmin=vmin,
        vmax=vmax,
        decide_bins_in_log_space=(log_y if horizontal else log_x))

    hist = np.asfarray(hist)

    bin_sizes = bin_edges[1:] - bin_edges[:-1]

    if divided_by_bin_size:
        hist /= bin_sizes

    if weighted:
        hist *= bin_centers*bin_sizes

    if hist_scale != 1.0:
        hist *= hist_scale

    if plot_type == 'steps':
        if horizontal:
            ax.step(hist, bin_edges[:-1], c=color, ls=ls, lw=lw, label=label)
        else:
            ax.step(bin_edges[:-1], hist, c=color, ls=ls, lw=lw, label=label)
    elif plot_type == 'bar':
        if horizontal:
            ax.barh(bin_edges[:-1],
                    hist,
                    align='edge',
                    height=bin_sizes,
                    log=log_x,
                    color=color,
                    alpha=alpha,
                    linewidth=lw,
                    label=label)
        else:
            ax.bar(bin_edges[:-1],
                   hist,
                   align='edge',
                   width=bin_sizes,
                   log=log_y,
                   color=color,
                   alpha=alpha,
                   linewidth=lw,
                   label=label)
    elif plot_type == 'fillstep':
        if horizontal:
            ax.fill_betweenx(bin_edges[:-1],
                             hist,
                             step='pre',
                             color=color,
                             alpha=alpha)
        else:
            ax.fill_between(bin_edges[:-1],
                            hist,
                            step='pre',
                            color=color,
                            alpha=alpha)
        if horizontal:
            ax.step(hist, bin_edges[:-1], c=color, ls=ls, lw=lw, label=label)
        else:
            ax.step(bin_edges[:-1], hist, c=color, ls=ls, lw=lw, label=label)
    elif plot_type == 'fill':
        if horizontal:
            ax.fill_betweenx(bin_centers,
                             hist,
                             color=color,
                             alpha=alpha,
                             label=label)
        else:
            ax.fill_between(bin_centers,
                            hist,
                            color=color,
                            alpha=alpha,
                            label=label)
        if horizontal:
            ax.plot(hist,
                    bin_centers,
                    c=color,
                    ls=ls,
                    lw=lw,
                    alpha=alpha,
                    label=label)
        else:
            ax.plot(bin_centers,
                    hist,
                    c=color,
                    ls=ls,
                    lw=lw,
                    alpha=alpha,
                    label=label)
    elif plot_type == 'points':
        if horizontal:
            ax.plot(hist,
                    bin_centers,
                    c=color,
                    ls=ls,
                    lw=lw,
                    alpha=alpha,
                    marker='o')
        else:
            ax.plot(bin_centers,
                    hist,
                    c=color,
                    ls=ls,
                    lw=lw,
                    alpha=alpha,
                    marker='o')
    else:
        raise ValueError(f'Invalid plot type {plot_type}')

    if extra_artists is not None:
        for artist in extra_artists:
            ax.add_artist(artist)

    if log_x:
        ax.set_xscale('log')
    if log_y:
        ax.set_yscale('log')

    if fit_limits is not None:
        start_idx = np.argmin(np.abs(bin_centers - fit_limits[0]))
        end_idx = np.argmin(np.abs(bin_centers - fit_limits[1]))

        coefs = np.polyfit(
            np.log10(bin_centers[start_idx:end_idx])
            if log_x else bin_centers[start_idx:end_idx],
            np.log10(hist[start_idx:end_idx])
            if log_y else hist[start_idx:end_idx], 1)
        print(f'Slope of fitted line: {coefs[0]}')

        fit_values = np.poly1d(coefs)(
            np.log10(bin_centers) if log_x else bin_centers)
        if log_y:
            fit_values = 10**fit_values
        ax.plot(bin_centers, fit_values, 'k--', alpha=0.3, lw=1.0)

        shift = 3
        xylabel = ((bin_centers[shift] + bin_centers[shift + 1])/2,
                   (fit_values[shift] + fit_values[shift + 1])/2)
        p1 = ax.transData.transform_point(
            (bin_centers[shift], fit_values[shift]))
        p2 = ax.transData.transform_point(
            (bin_centers[shift + 1], fit_values[shift + 1]))
        dy = (p2[1] - p1[1])
        dx = (p2[0] - p1[0])
        rotn = np.degrees(np.arctan2(dy, dx))
        ax.annotate(f'{coefs[0]:.1f}',
                    xy=xylabel,
                    ha='center',
                    va='center',
                    rotation=rotn,
                    backgroundcolor='w',
                    alpha=0.5,
                    fontsize='x-small')

    if minorticks_on:
        ax.minorticks_on()

    set_2d_plot_extent(ax, x_lims, y_lims)
    set_2d_axis_labels(ax,
                       xlabel,
                       ylabel,
                       xcolor=xlabel_color,
                       ycolor=ylabel_color)

    ax.tick_params(axis='x', labelcolor=xlabel_color)
    ax.tick_params(axis='y', labelcolor=ylabel_color)

    if legend_loc:
        ax.legend(loc=legend_loc)

    if render_now:
        render(fig, output_path=output_path)

    return fig, ax


def plot_scatter(values_x,
                 values_y,
                 values_c=None,
                 fig=None,
                 ax=None,
                 log_x=False,
                 log_y=False,
                 log_c=False,
                 vmin_c=None,
                 vmax_c=None,
                 cmap_name='viridis',
                 marker='o',
                 s=5.0,
                 relative_s=False,
                 color='k',
                 edgecolors='none',
                 alpha=1.0,
                 relative_alpha=False,
                 x_lims=None,
                 y_lims=None,
                 xlabel=None,
                 ylabel=None,
                 aspect='auto',
                 minorticks_on=False,
                 internal_cbar=False,
                 cbar_loc='right',
                 cbar_pad=0.05,
                 cbar_minorticks_on=False,
                 cbar_opposite_side_ticks=False,
                 cbar_tick_loc='right',
                 cbar_width='3%',
                 cbar_height='60%',
                 cbar_orientation='vertical',
                 clabel='',
                 label=None,
                 legend_loc=None,
                 extra_artists=None,
                 output_path=None,
                 fig_kwargs={},
                 render_now=True):

    if fig is None or ax is None:
        fig, ax = create_2d_subplots(**fig_kwargs)

    if log_x:
        ax.set_xscale('log')
    if log_y:
        ax.set_yscale('log')

    if values_c is None:
        c = color
    else:
        if vmin_c is None:
            vmin_c = np.nanmin(values_c)
        if vmax_c is None:
            vmax_c = np.nanmax(values_c)

        norm = get_normalizer(vmin_c, vmax_c, log=log_c)
        cmap = get_cmap(cmap_name)
        c = colors_from_values(values_c,
                               norm,
                               cmap,
                               alpha=alpha,
                               relative_alpha=relative_alpha)

        if relative_s:
            s = size_from_values(values_c, norm, s)

        if cbar_loc is not None:
            if internal_cbar:
                add_2d_colorbar_inside_from_cmap_and_norm(
                    fig,
                    ax,
                    norm,
                    cmap,
                    loc=cbar_loc,
                    tick_loc=cbar_tick_loc,
                    width=cbar_width,
                    height=cbar_height,
                    orientation=cbar_orientation,
                    label=clabel)
            else:
                add_2d_colorbar_from_cmap_and_norm(
                    fig,
                    ax,
                    norm,
                    cmap,
                    loc=cbar_loc,
                    pad=cbar_pad,
                    minorticks_on=cbar_minorticks_on,
                    opposite_side_ticks=cbar_opposite_side_ticks,
                    label=clabel)

    ax.scatter(values_x,
               values_y,
               c=c,
               s=s,
               marker=marker,
               edgecolors=edgecolors,
               alpha=alpha,
               label=label)

    if extra_artists is not None:
        for artist in extra_artists:
            ax.add_artist(artist)

    if minorticks_on:
        ax.minorticks_on()

    ax.set_aspect(aspect)

    set_2d_plot_extent(ax, x_lims, y_lims)
    set_2d_axis_labels(ax, xlabel, ylabel)

    if legend_loc is not None:
        ax.legend(loc=legend_loc)

    if render_now:
        render(fig, output_path=output_path)

    return fig, ax


def plot_scatter_with_histograms(values_x,
                                 values_y,
                                 values_c=None,
                                 hist_x_scale=1.0,
                                 hist_y_scale=1.0,
                                 bins_x='auto',
                                 bins_y='auto',
                                 hist_x_divided_by_bin_size=False,
                                 hist_y_divided_by_bin_size=False,
                                 log_x=False,
                                 log_y=False,
                                 log_c=False,
                                 log_hist_x=False,
                                 log_hist_y=False,
                                 vmin_x=None,
                                 vmax_x=None,
                                 vmin_y=None,
                                 vmax_y=None,
                                 vmin_c=None,
                                 vmax_c=None,
                                 cmap_name='viridis',
                                 marker='o',
                                 s=5.0,
                                 relative_s=False,
                                 color='k',
                                 edgecolors='none',
                                 alpha=1.0,
                                 relative_alpha=False,
                                 hist_x_alpha=1.0,
                                 hist_y_alpha=1.0,
                                 hist_x_color='k',
                                 hist_y_color='k',
                                 hist_linewidth=0,
                                 xlabel=None,
                                 ylabel=None,
                                 hist_x_plot_type='bar',
                                 hist_y_plot_type='bar',
                                 hist_x_label=None,
                                 hist_y_label=None,
                                 hist_x_label_color='k',
                                 hist_y_label_color='k',
                                 hist_x_fit_limits=None,
                                 hist_y_fit_limits=None,
                                 spacing=0.015,
                                 hist_size_x=0.3,
                                 hist_size_y=0.3,
                                 left_padding=0.12,
                                 bottom_padding=0.1,
                                 minorticks_on=False,
                                 internal_cbar=False,
                                 cbar_loc='upper left',
                                 cbar_tick_loc='right',
                                 cbar_width='3%',
                                 cbar_height='60%',
                                 cbar_orientation='vertical',
                                 clabel='',
                                 output_path=None,
                                 fig_kwargs={},
                                 render_now=True):

    left = left_padding  # > 0 to make space for labels
    bottom = bottom_padding  # > 0 to make space for labels
    width = 1 - 1.5*left - spacing - hist_size_y
    height = 1 - 1.5*bottom - spacing - hist_size_x

    fig = create_figure(**fig_kwargs)
    ax = fig.add_axes([left, bottom, width, height])
    ax_hist_x = fig.add_axes(
        [left, bottom + height + spacing, width, hist_size_x], sharex=ax)
    ax_hist_y = fig.add_axes(
        [left + width + spacing, bottom, hist_size_y, height], sharey=ax)

    ax_hist_x.tick_params(axis='x', labelbottom=False)
    ax_hist_y.tick_params(axis='y', labelleft=False)

    ax_hist_x.tick_params(axis='y', labelcolor=hist_x_label_color)
    ax_hist_y.tick_params(axis='x', labelcolor=hist_y_label_color)

    if log_x:
        ax.set_xscale('log')
    if log_y:
        ax.set_yscale('log')

    if values_c is None:
        c = color
    else:
        if vmin_c is None:
            vmin_c = np.nanmin(values_c)
        if vmax_c is None:
            vmax_c = np.nanmax(values_c)

        norm = get_normalizer(vmin_c, vmax_c, log=log_c)
        cmap = get_cmap(cmap_name)
        c = colors_from_values(values_c,
                               norm,
                               cmap,
                               alpha=alpha,
                               relative_alpha=relative_alpha)

        if relative_s:
            s = size_from_values(values_c, norm, s)

        if internal_cbar:
            add_2d_colorbar_inside_from_cmap_and_norm(
                fig,
                ax,
                norm,
                cmap,
                loc=cbar_loc,
                tick_loc=cbar_tick_loc,
                width=cbar_width,
                height=cbar_height,
                orientation=cbar_orientation,
                label=clabel)

    ax.scatter(values_x,
               values_y,
               c=c,
               s=s,
               marker=marker,
               edgecolors=edgecolors,
               alpha=alpha)

    if minorticks_on:
        ax.minorticks_on()

    plot_histogram(values_x,
                   fig=fig,
                   ax=ax_hist_x,
                   hist_scale=hist_x_scale,
                   weighted=False,
                   divided_by_bin_size=hist_x_divided_by_bin_size,
                   bins=bins_x,
                   log_x=log_x,
                   log_y=log_hist_x,
                   vmin=vmin_x,
                   vmax=vmax_x,
                   plot_type=hist_x_plot_type,
                   horizontal=False,
                   color=hist_x_color,
                   lw=hist_linewidth,
                   ls='-',
                   alpha=hist_x_alpha,
                   fit_limits=hist_x_fit_limits,
                   minorticks_on=minorticks_on,
                   ylabel=hist_x_label,
                   ylabel_color=hist_x_label_color,
                   render_now=False)

    plot_histogram(values_y,
                   fig=fig,
                   ax=ax_hist_y,
                   hist_scale=hist_y_scale,
                   weighted=False,
                   divided_by_bin_size=hist_y_divided_by_bin_size,
                   bins=bins_y,
                   log_x=log_hist_y,
                   log_y=log_y,
                   vmin=vmin_y,
                   vmax=vmax_y,
                   plot_type=hist_y_plot_type,
                   horizontal=True,
                   color=hist_y_color,
                   lw=hist_linewidth,
                   ls='-',
                   alpha=hist_y_alpha,
                   fit_limits=hist_y_fit_limits,
                   minorticks_on=minorticks_on,
                   xlabel=hist_y_label,
                   xlabel_color=hist_y_label_color,
                   render_now=False)

    set_2d_axis_labels(ax, xlabel, ylabel)

    if render_now:
        render(fig, output_path=output_path)

    return fig, ax, ax_hist_x, ax_hist_y


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
    'transport_inv':
    define_linear_segmented_colormap(
        '',
        np.vstack(([[1.0, 1.0, 1.0,
                     1.0]], plt.get_cmap('Blues_r')(np.linspace(1, 0, 256)),
                   plt.get_cmap('Oranges_r')(np.linspace(0, 1, 256)),
                   [[1.0, 1.0, 1.0, 1.0]])),
        bad_color='white',
        N=514),
    'Orangesw_r':
    define_linear_segmented_colormap('',
                                     np.vstack((plt.get_cmap('Oranges_r')(
                                         np.linspace(0, 1, 256)),
                                                [[1.0, 1.0, 1.0, 1.0]])),
                                     bad_color='white',
                                     N=257),
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
