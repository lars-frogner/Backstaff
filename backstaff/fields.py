import numpy as np
try:
    import backstaff.plotting as plotting
except ModuleNotFoundError:
    import plotting


class Coords3:
    def __init__(self, x_coords, y_coords, z_coords):
        self.x = np.asfarray(x_coords)
        self.y = np.asfarray(y_coords)
        self.z = np.asfarray(z_coords)

    def get_shape(self):
        return self.x.size, self.y.size, self.z.size


class Coords2:
    def __init__(self, x_coords, y_coords):
        self.x = np.asfarray(x_coords)
        self.y = np.asfarray(y_coords)

    def get_shape(self):
        return self.x.size, self.y.size


class ScalarField2:
    @staticmethod
    def from_pickle_file(file_path):
        import backstaff.reading as reading
        return reading.read_2d_scalar_field(file_path)

    def __init__(self, coords, values):
        assert isinstance(coords, Coords2)
        self.coords = coords
        self.values = np.asfarray(values)
        assert self.values.shape == self.coords.get_shape()

    def get_shape(self):
        return self.coords.get_shape()

    def get_values(self):
        return self.values

    def get_horizontal_bounds(self):
        return (self.coords.x[0], self.coords.x[-1])

    def get_vertical_bounds(self, negate=False):
        return (-self.coords.y[-1],
                -self.coords.y[0]) if negate else (self.coords.y[0],
                                                   self.coords.y[-1])

    def add_to_plot(self,
                    ax,
                    invert_horizontal_lims=False,
                    invert_vertical_lims=False,
                    negate_vertical_coords=False,
                    vmin=None,
                    vmax=None,
                    log=False,
                    symlog=False,
                    linthresh=np.inf,
                    linscale=1.0,
                    cmap_name='viridis',
                    cmap_bad_color='w',
                    contour_levels=None,
                    contour_colors='r',
                    contour_alpha=1.0,
                    log_contour=False,
                    vmin_contour=None,
                    vmax_contour=None,
                    contour_cmap_name='viridis'):

        if symlog:
            norm = plotting.get_symlog_normalizer(vmin,
                                                  vmax,
                                                  linthresh,
                                                  linscale=linscale)
        else:
            norm = plotting.get_normalizer(vmin, vmax, log=log)

        values = self.get_values()

        extent = [
            *(self.get_horizontal_bounds()
              [::-1 if invert_horizontal_lims else 1]),
            *(self.get_vertical_bounds(negate=negate_vertical_coords)
              [::-1 if invert_vertical_lims else 1])
        ]

        im = ax.imshow(values.T,
                       norm=norm,
                       cmap=plotting.get_cmap(cmap_name,
                                              bad_color=cmap_bad_color),
                       interpolation='none',
                       extent=extent,
                       aspect='equal')

        if contour_levels is not None:
            ax.contourf(
                np.linspace(*extent[:2], values.shape[0]),
                np.linspace(*extent[2:], values.shape[1]),
                values[:, ::(-1 if negate_vertical_coords else 1)].T,
                levels=contour_levels,
                norm=plotting.get_normalizer(vmin_contour,
                                             vmax_contour,
                                             log=log_contour),
                cmap=plotting.get_cmap(contour_cmap_name),
                #colors=contour_colors,
                alpha=contour_alpha)

        return im


def plot_2d_scalar_field(field,
                         fig=None,
                         ax=None,
                         figsize=None,
                         xlabel=None,
                         ylabel=None,
                         value_description=None,
                         title=None,
                         render=True,
                         output_path=None,
                         **kwargs):

    if fig is None or ax is None:
        fig, ax = plotting.create_2d_subplots(figsize=figsize)

    im = field.add_to_plot(ax, **kwargs)

    plotting.set_2d_plot_extent(
        ax,
        field.get_horizontal_bounds()
        [::-1 if kwargs.get('invert_horizontal_lims', False) else 1],
        field.get_vertical_bounds(
            negate=kwargs.get('negate_vertical_coords', False))
        [::-1 if kwargs.get('invert_vertical_lims', False) else 1])
    plotting.set_2d_axis_labels(ax, xlabel, ylabel)
    plotting.add_2d_colorbar(
        fig,
        ax,
        im,
        label='' if value_description is None else value_description)

    if title is not None:
        ax.set_title(title)

    if render:
        plotting.render(fig, output_path=output_path)
