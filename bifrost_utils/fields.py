import numpy as np
import bifrost_utils.plotting as plotting


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
        import bifrost_utils.reading as reading
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
                             *self.get_horizontal_bounds(),
                             *self.get_vertical_bounds()
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

    ax.set_xlim(*field.get_horizontal_bounds())
    ax.set_ylim(*field.get_vertical_bounds())
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plotting.add_2d_colorbar(
        fig,
        ax,
        im,
        label='' if value_description is None else value_description)

    ax.set_title(title)

    if render:
        plotting.render(fig, output_path=output_path)


if __name__ == "__main__":
    from pathlib import Path
    import bifrost_utils.reading as reading

    fig, axes = plotting.create_2d_subplots(ncols=1)
    reading.read_2d_scalar_field(
        Path(reading.DATA_PATH, 'test_data',
             'slice1.pickle')).add_to_plot(axes, log=True)
    # reading.read_2d_scalar_field(
    #     Path(reading.DATA_PATH, 'phd_run',
    #          'en024031_emer3.0str_ebeam_351.pickle')).add_to_plot(axes[1],
    #                                                               log=False)
    plotting.render(fig)
