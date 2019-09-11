import numpy as np


class Coords3:
    def __init__(self, x_coords, y_coords, z_coords):
        self.x = np.asfarray(x_coords)
        self.y = np.asfarray(y_coords)
        self.z = np.asfarray(z_coords)

    def shape(self):
        return self.x.size, self.y.size, self.z.size


class Coords2:
    def __init__(self, x_coords, y_coords):
        self.x = np.asfarray(x_coords)
        self.y = np.asfarray(y_coords)

    def shape(self):
        return self.x.size, self.y.size


class ScalarField2:
    def __init__(self, coords, values):
        assert isinstance(coords, Coords2)
        self.coords = coords
        self.values = np.asfarray(values)
        assert self.values.shape == self.coords.shape()

    def shape(self):
        return self.coords.shape()

    def add_to_plot(self, ax):
        ax.imshow(np.log10(self.values.T))


if __name__ == "__main__":
    import reading
    import plotting
    from pathlib import Path

    field = reading.read_2d_scalar_field(Path(reading.data_path, 'phd_run', 'en024031_emer3.0str_351.pickle'))
    field_coarse = reading.read_2d_scalar_field(Path(reading.data_path, 'phd_run', 'en024031_emer3.0str_coarse_351.pickle'))
    fig, axes = plotting.create_2d_subplots(ncols=2)
    field.add_to_plot(axes[0])
    field_coarse.add_to_plot(axes[1])
    plotting.render(fig)
