import numpy as np
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
    def __init__(self, coords, values):
        assert isinstance(coords, Coords2)
        self.coords = coords
        self.values = np.asfarray(values)
        assert self.values.shape == self.coords.get_shape()

    def get_shape(self):
        return self.coords.get_shape()

    def get_values(self):
        return self.values

    def add_to_plot(self,
                    ax,
                    log=False,
                    vmin=None,
                    vmax=None,
                    cmap_name='viridis'):

        values = self.get_values()
        if log:
            values = np.log10(values)

        return ax.imshow(values.T,
                         norm=plotting.get_normalizer(vmin, vmax, log=log),
                         vmin=vmin,
                         vmax=vmax,
                         cmap=plotting.get_cmap(cmap_name))


if __name__ == "__main__":
    import reading
    from pathlib import Path

    fig, axes = plotting.create_2d_subplots(ncols=2)
    reading.read_2d_scalar_field(
        Path(reading.DATA_PATH, 'phd_run',
             'en024031_emer3.0str_ebeam_tg_351.pickle')).add_to_plot(axes[0],
                                                                     log=True)
    reading.read_2d_scalar_field(
        Path(reading.DATA_PATH, 'phd_run',
             'en024031_emer3.0str_ebeam_351.pickle')).add_to_plot(axes[1],
                                                                  log=False)
    plotting.render(fig)
