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
    @staticmethod
    def from_bifrost_data(bifrost_data, omitted_axis):
        all_coords = [
            bifrost_data.xdn, bifrost_data.ydn,
            -(2*bifrost_data.z - bifrost_data.zdn)[::-1]
        ]
        all_coords.pop(omitted_axis)
        return Coords2(all_coords[0], all_coords[1])

    def __init__(self, x_coords, y_coords):
        self.x = np.asfarray(x_coords)
        self.y = np.asfarray(y_coords)

    def get_shape(self):
        return self.x.size, self.y.size


class Coords1:
    def __init__(self, coords):
        self.coords = np.asfarray(coords)

    def get_size(self):
        return self.coords.size


class ScalarField1:
    @staticmethod
    def horizontal_average_from_bifrost_data(
        bifrost_data,
        quantities,
        scale=None,
        value_processor=lambda x: x,
    ):
        coords = -(2*bifrost_data.z - bifrost_data.zdn)[::-1]
        if not isinstance(quantities, list) and not isinstance(
                quantities, tuple):
            quantities = [quantities]
        all_values = value_processor(
            *(bifrost_data.get_var(quantity)[:, :, ::-1]
              for quantity in quantities))
        values = np.mean(all_values, axis=(0, 1))
        if scale is not None:
            values *= scale

        return ScalarField1(Coords1(coords), values)

    @staticmethod
    def at_horizontal_indices_in_bifrost_data(
        bifrost_data,
        i,
        j,
        quantities,
        scale=None,
        value_processor=lambda x: x,
    ):
        coords = -(2*bifrost_data.z - bifrost_data.zdn)[::-1]
        if not isinstance(quantities, list) and not isinstance(
                quantities, tuple):
            quantities = [quantities]
        values = value_processor(*(bifrost_data.get_var(quantity)[i, j, ::-1]
                                   for quantity in quantities))
        if scale is not None:
            values *= scale

        return ScalarField1(Coords1(coords), values)

    @staticmethod
    def from_file(file_path):
        data = np.load(file_path)
        return ScalarField1(Coords1(data['coords']), data['values'])

    def __init__(self, coords, values):
        assert isinstance(coords, Coords1)
        self.coords = coords
        self.values = np.asfarray(values)
        assert self.values.size == self.coords.get_size()

    def __add__(self, term):
        if isinstance(term, self.__class__):
            assert np.allclose(self.coords.coords, term.coords.coords)
            return ScalarField1(self.coords, self.values + term.values)
        else:
            return ScalarField1(self.coords, self.values + term)

    def __sub__(self, term):
        if isinstance(term, self.__class__):
            assert np.allclose(self.coords.coords, term.coords.coords)
            return ScalarField1(self.coords, self.values - term.values)
        else:
            return ScalarField1(self.coords, self.values - term)

    def __mul__(self, factor):
        if isinstance(factor, self.__class__):
            assert np.allclose(self.coords.coords, factor.coords.coords)
            return ScalarField1(self.coords, self.values*factor.values)
        else:
            return ScalarField1(self.coords, self.values*factor)

    def __truediv__(self, divisor):
        if isinstance(divisor, self.__class__):
            assert np.allclose(self.coords.coords, divisor.coords.coords)
            return ScalarField1(self.coords, self.values/divisor.values)
        else:
            return ScalarField1(self.coords, self.values/divisor)

    def get_size(self):
        return self.coords.get_size()

    def get_values(self, inverted_vertically=False):
        return self.values[::(-1 if inverted_vertically else 1)]

    def get_coords(self, inverted=False):
        return -self.coords.coords[::-1] if inverted else self.coords.coords

    def get_bounds(self, inverted=False):
        return (-self.coords.coords[-1],
                -self.coords.coords[0]) if inverted else (
                    self.coords.coords[0], self.coords.coords[-1])

    def get_extent(self):
        start, end = self.get_bounds()
        return end - start

    def compute_integral(self):
        values = self.get_values()
        return np.sum(0.5*(values[:-1] + values[1:])*
                      np.diff(self.get_coords()))

    def find_peak_coordinate(self):
        return self.coords.coords[np.argmax(self.values)]

    def plot(self, inverted_vertically=False, **plot_kwargs):
        return plotting.plot_1d_field(
            self.get_coords(inverted=inverted_vertically),
            self.get_values(inverted_vertically=inverted_vertically),
            **plot_kwargs)

    def save(self, file_path, compressed=True):
        if compressed:
            np.savez_compressed(file_path,
                                coords=self.coords.coords,
                                values=self.values)
        else:
            np.savez(file_path, coords=self.coords.coords, values=self.values)


class ScalarField2:
    @staticmethod
    def from_pickle_file(file_path):
        import backstaff.reading as reading
        return reading.read_2d_scalar_field(file_path)

    @staticmethod
    def slice_from_bifrost_data(bifrost_data,
                                quantity,
                                slice_axis=1,
                                slice_coord=7.5,
                                scale=None):
        all_center_coords = [
            bifrost_data.x, bifrost_data.y, -bifrost_data.z[::-1]
        ]
        slice_idx = np.argmin(
            np.abs(all_center_coords[slice_axis] - slice_coord))

        all_slices = [slice(None)]*3
        all_slices[slice_axis] = slice_idx

        values = bifrost_data.get_var(quantity)[:, :, ::-1][all_slices[0],
                                                            all_slices[1],
                                                            all_slices[2]]
        if scale is not None:
            values = values*scale

        return ScalarField2(
            Coords2.from_bifrost_data(bifrost_data, slice_axis), values)

    @staticmethod
    def accumulated_from_bifrost_data(bifrost_data,
                                      quantities,
                                      accum_axis=1,
                                      scale=None,
                                      value_processor=lambda x: x,
                                      accum_operator=np.sum):
        if not isinstance(quantities, list) and not isinstance(
                quantities, tuple):
            quantities = [quantities]
        all_values = value_processor(
            *(bifrost_data.get_var(quantity)[:, :, ::-1]
              for quantity in quantities))
        values = accum_operator(all_values, axis=accum_axis)
        if scale is not None:
            values *= scale

        return ScalarField2(
            Coords2.from_bifrost_data(bifrost_data, accum_axis), values)

    @staticmethod
    def from_file(file_path):
        data = np.load(file_path)
        return ScalarField2(Coords2(data['x'], data['y']), data['values'])

    def __init__(self, coords, values):
        assert isinstance(coords, Coords2)
        self.coords = coords
        self.values = np.asfarray(values)
        assert self.values.shape == self.coords.get_shape()

    def __add__(self, term):
        if isinstance(term, self.__class__):
            assert np.allclose(self.coords.x, term.coords.x)
            assert np.allclose(self.coords.y, term.coords.y)
            return ScalarField2(self.coords, self.values + term.values)
        else:
            return ScalarField2(self.coords, self.values + term)

    def __sub__(self, term):
        if isinstance(term, self.__class__):
            assert np.allclose(self.coords.x, term.coords.x)
            assert np.allclose(self.coords.y, term.coords.y)
            return ScalarField2(self.coords, self.values - term.values)
        else:
            return ScalarField2(self.coords, self.values - term)

    def __mul__(self, factor):
        if isinstance(factor, self.__class__):
            assert np.allclose(self.coords.x, factor.coords.x)
            assert np.allclose(self.coords.y, factor.coords.y)
            return ScalarField2(self.coords, self.values*factor.values)
        else:
            return ScalarField2(self.coords, self.values*factor)

    def __truediv__(self, divisor):
        if isinstance(divisor, self.__class__):
            assert np.allclose(self.coords.x, divisor.coords.x)
            assert np.allclose(self.coords.y, divisor.coords.y)
            return ScalarField2(self.coords, self.values/divisor.values)
        else:
            return ScalarField2(self.coords, self.values/divisor)

    def get_shape(self):
        return self.coords.get_shape()

    def get_values(self, inverted_vertically=False):
        return self.values[:, ::(-1 if inverted_vertically else 1)]

    def get_horizontal_coords(self):
        return self.coords.x

    def get_vertical_coords(self, inverted=False):
        return -self.coords.y[::-1] if inverted else self.coords.y

    def get_horizontal_bounds(self):
        return (self.coords.x[0], self.coords.x[-1])

    def get_vertical_bounds(self, inverted=False):
        return (-self.coords.y[-1],
                -self.coords.y[0]) if inverted else (self.coords.y[0],
                                                     self.coords.y[-1])

    def get_horizontal_extent(self):
        start, end = self.get_horizontal_bounds()
        return end - start

    def get_vertical_extent(self):
        start, end = self.get_vertical_bounds()
        return end - start

    def compute_integral(self):
        values = self.get_values()
        partial = np.sum(0.5*(values[:, :-1] + values[:, 1:])*
                         np.diff(self.get_horizontal_coords())[np.newaxis, :],
                         axis=1)
        return np.sum(0.5*(partial[:-1] + partial[1:])*
                      np.diff(self.get_vertical_coords()))

    def plot(self, inverted_vertically=False, **plot_kwargs):
        figure_width = plot_kwargs.pop('figure_width', 7.2)
        figure_aspect = plot_kwargs.pop(
            'figure_aspect', 5/4 if np.abs(
                (self.get_horizontal_extent() - self.get_vertical_extent())/
                self.get_horizontal_extent()) < 1e-3 else 4.5/3)
        return plotting.plot_2d_field(
            self.get_horizontal_coords(),
            self.get_vertical_coords(inverted=inverted_vertically),
            self.get_values(inverted_vertically=inverted_vertically),
            figure_width=figure_width,
            figure_aspect=figure_aspect,
            **plot_kwargs)

    def save(self, file_path, compressed=True):
        if compressed:
            np.savez_compressed(file_path,
                                x=self.coords.x,
                                y=self.coords.y,
                                values=self.values)
        else:
            np.savez(file_path,
                     x=self.coords.x,
                     y=self.coords.y,
                     values=self.values)
