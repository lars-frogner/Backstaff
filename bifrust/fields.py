import numpy as np
try:
    import bifrust.plotting as plotting
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
        import bifrust.reading as reading
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

    def get_vertical_bounds(self):
        return (self.coords.y[0], self.coords.y[-1])

    def add_to_plot(self,
                    ax,
                    invert_horizontal_lims=False,
                    invert_vertical_lims=False,
                    log=False,
                    vmin=None,
                    vmax=None,
                    cmap_name='viridis'):

        values = self.get_values()
        return ax.imshow(values.T,
                         norm=plotting.get_normalizer(vmin, vmax, log=log),
                         vmin=vmin,
                         vmax=vmax,
                         cmap=plotting.get_cmap(cmap_name),
                         interpolation='none',
                         extent=[
                             *(self.get_horizontal_bounds()
                               [::-1 if invert_horizontal_lims else 1]),
                             *(self.get_vertical_bounds()
                               [::-1 if invert_vertical_lims else 1])
                         ],
                         aspect='auto')


def plot_2d_scalar_field(field,
                         fig=None,
                         ax=None,
                         xlabel=None,
                         ylabel=None,
                         value_description=None,
                         title=None,
                         render=True,
                         output_path=None,
                         **kwargs):

    if fig is None or ax is None:
        fig, ax = plotting.create_2d_subplots()

    im = field.add_to_plot(ax, **kwargs)

    plotting.set_2d_plot_extent(
        ax,
        field.get_horizontal_bounds()
        [::-1 if kwargs.get('invert_horizontal_lims', False) else 1],
        field.get_vertical_bounds()
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
