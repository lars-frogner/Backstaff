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

    def add_to_plot(self, ax):
        ax.imshow(np.log10(self.values.T))


if __name__ == "__main__":
    import reading
    import plotting
    from pathlib import Path

    field = reading.read_2d_scalar_field(Path(reading.data_path, 'slice_field.pickle'))
    fig, ax = plotting.create_2d_plot()
    field.add_to_plot(ax)
    plotting.show()