from os import path
import pathlib
import glob
import re
import numpy as np
import scipy.interpolate
try:
    import backstaff.plotting as plotting
except ModuleNotFoundError:
    import plotting


class Coords3:
    def __init__(self, x_coords, y_coords, z_coords):
        self.x = np.asfarray(x_coords)
        self.y = np.asfarray(y_coords)
        self.z = np.asfarray(z_coords)
        self.coords = (self.x, self.y, self.z)

    def get_shape(self):
        return self.x.size, self.y.size, self.z.size

    def __getitem__(self, idx):
        return self.coords[idx]


class Coords2:
    @staticmethod
    def indices(shape):
        return Coords2(np.arange(shape[0]), np.arange(shape[1]))

    @staticmethod
    def from_bifrost_data(bifrost_data, omitted_axis):
        all_coords = [
            bifrost_data.xdn, bifrost_data.ydn,
            -(2 * bifrost_data.z - bifrost_data.zdn)[::-1]
        ]
        all_coords.pop(omitted_axis)
        return Coords2(all_coords[0], all_coords[1])

    def get_shape(self):
        return self.x.size, self.y.size, self.z.size

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


class ScalarField3:
    @staticmethod
    def from_bifrost_data(
        bifrost_data,
        quantities,
        height_range=None,
        scale=None,
        value_processor=lambda x: x,
    ):
        z_coords = -(2 * bifrost_data.z - bifrost_data.zdn)[::-1]
        if not isinstance(quantities, list) and not isinstance(
                quantities, tuple):
            quantities = [quantities]

        def get_quantity(quantity):
            if isinstance(quantity, str):
                return bifrost_data.get_var(quantity)
            else:
                return quantity

        k_slice = slice(*((None, ) if height_range is None else np.
                          searchsorted(z_coords, height_range)))
        values = value_processor(*(get_quantity(quantity)[:, :, ::-1][:, :,
                                                                      k_slice]
                                   for quantity in quantities))
        z_coords = z_coords[k_slice]
        if scale is not None:
            values = values * scale

        return ScalarField3(
            Coords3(bifrost_data.xdn, bifrost_data.ydn, z_coords), values)

    @staticmethod
    def at_horizontal_indices_in_bifrost_data(
        bifrost_data,
        i_slice,
        j_slice,
        quantities,
        height_range=None,
        scale=None,
        value_processor=lambda x: x,
    ):
        x_coords = bifrost_data.xdn
        y_coords = bifrost_data.ydn
        z_coords = helita_utils.inverted_zdn(bifrost_data)
        if not isinstance(quantities, (list, tuple)):
            quantities = [quantities]
        k_slice = helita_utils.inclusive_coord_slice(z_coords, height_range)
        values = value_processor(
            *(bifrost_data.get_var(quantity)[i_slice, j_slice, ::-1][:, :,
                                                                     k_slice]
              for quantity in quantities))
        x_coords = x_coords[i_slice]
        y_coords = y_coords[j_slice]
        z_coords = z_coords[k_slice]
        if scale is not None:
            values = values * scale

        return ScalarField3(Coords3(x_coords, y_coords, z_coords), values)

    @staticmethod
    def for_subdomain_of_bifrost_data(
        bifrost_data,
        quantities,
        x_range=None,
        y_range=None,
        height_range=None,
        scale=None,
        value_processor=lambda x: x,
    ):
        x_coords = bifrost_data.xdn
        y_coords = bifrost_data.ydn
        z_coords = helita_utils.inverted_zdn(bifrost_data)
        if not isinstance(quantities, (list, tuple)):
            quantities = [quantities]
        i_slice = helita_utils.inclusive_coord_slice(x_coords, x_range)
        j_slice = helita_utils.inclusive_coord_slice(y_coords, y_range)
        k_slice = helita_utils.inclusive_coord_slice(z_coords, height_range)
        values = value_processor(
            *(bifrost_data.get_var(quantity)[i_slice, j_slice, ::-1][:, :,
                                                                     k_slice]
              for quantity in quantities))
        x_coords = x_coords[i_slice]
        y_coords = y_coords[j_slice]
        z_coords = z_coords[k_slice]
        if scale is not None:
            values = values * scale

        return ScalarField3(Coords3(x_coords, y_coords, z_coords), values)

    def __init__(self, coords, values):
        assert isinstance(coords, Coords3)
        self.coords = coords
        self.values = np.asfarray(values)
        assert self.values.shape == self.coords.get_shape()

    def __add__(self, term):
        if isinstance(term, self.__class__):
            assert np.allclose(self.coords.x, term.coords.x)
            assert np.allclose(self.coords.y, term.coords.y)
            return ScalarField3(self.coords, self.values + term.values)
        else:
            return ScalarField3(self.coords, self.values + term)

    def __sub__(self, term):
        if isinstance(term, self.__class__):
            assert np.allclose(self.coords.x, term.coords.x)
            assert np.allclose(self.coords.y, term.coords.y)
            return ScalarField3(self.coords, self.values - term.values)
        else:
            return ScalarField3(self.coords, self.values - term)

    def __mul__(self, factor):
        if isinstance(factor, self.__class__):
            assert np.allclose(self.coords.x, factor.coords.x)
            assert np.allclose(self.coords.y, factor.coords.y)
            return ScalarField3(self.coords, self.values * factor.values)
        else:
            return ScalarField3(self.coords, self.values * factor)

    def __truediv__(self, divisor):
        if isinstance(divisor, self.__class__):
            assert np.allclose(self.coords.x, divisor.coords.x)
            assert np.allclose(self.coords.y, divisor.coords.y)
            return ScalarField3(self.coords, self.values / divisor.values)
        else:
            return ScalarField3(self.coords, self.values / divisor)

    def get_shape(self):
        return self.coords.get_shape()

    def get_values(self):
        return self.values

    def get_values_flat(self):
        return np.ravel(self.values)

    def get_coords(self, inverted_vertically=False):
        return (self.get_horizontal_width_coords(),
                self.get_horizontal_depth_coords(),
                self.get_height_coords(inverted=inverted_vertically))

    def get_horizontal_width_coords(self):
        return self.coords.x

    def get_horizontal_depth_coords(self):
        return self.coords.y

    def get_height_coords(self, inverted=False):
        return -self.coords.z[::-1] if inverted else self.coords.z

    def resampled_along_axis(self,
                             axis,
                             resampling_factor,
                             index_range=None,
                             kind='linear'):
        if index_range is None:
            index_range = (0, self.get_shape()[axis])
        slices = [slice(None)] * 3
        slices[axis] = slice(*index_range)
        slices = tuple(slices)

        axis_size = index_range[1] - index_range[0]
        new_axis_size = int(np.ceil(resampling_factor * axis_size))

        new_coords = [self.coords[i][slice] for i, slice in enumerate(slices)]
        new_coords[axis] = np.linspace(new_coords[axis][0],
                                       new_coords[axis][-1], new_axis_size)

        new_values = scipy.interpolate.interp1d(
            self.coords[axis][slices[axis]],
            self.values[slices],
            axis=axis,
            kind=kind,
            copy=False,
            assume_sorted=True)(new_coords[axis])

        return ScalarField3(Coords3(*new_coords), new_values)

    def resampled_to_coords_along_axis(self,
                                       axis,
                                       new_axis_coords,
                                       kind='linear'):
        new_coords = [self.coords[i] for i in range(self.values.ndim)]
        new_coords[axis] = new_axis_coords

        new_values = scipy.interpolate.interp1d(
            self.coords[axis],
            self.values,
            axis=axis,
            kind=kind,
            copy=False,
            assume_sorted=True)(new_axis_coords)

        return ScalarField3(Coords3(*new_coords), new_values)

    def horizontal_mean(self):
        mean_values = np.nanmean(self.values, axis=(0, 1))
        return ScalarField1(Coords1(self.coords[2]), mean_values)


class ScalarField1:
    @staticmethod
    def horizontal_average_from_bifrost_data(
        bifrost_data,
        quantities,
        height_range=None,
        scale=None,
        value_processor=lambda x: x,
    ):
        coords = -(2 * bifrost_data.z - bifrost_data.zdn)[::-1]
        if not isinstance(quantities, list) and not isinstance(
                quantities, tuple):
            quantities = [quantities]
        k_slice = slice(*((None, ) if height_range is None else np.
                          searchsorted(coords, height_range)))
        all_values = value_processor(
            *(bifrost_data.get_var(quantity)[:, :, ::-1][:, :, k_slice]
              for quantity in quantities))
        values = np.mean(all_values, axis=(0, 1))
        coords = coords[k_slice]
        if scale is not None:
            values *= scale

        return ScalarField1(Coords1(coords), values)

    @staticmethod
    def at_horizontal_indices_in_bifrost_data(
        bifrost_data,
        i,
        j,
        quantities,
        height_range=None,
        scale=None,
        value_processor=lambda x: x,
    ):
        coords = -(2 * bifrost_data.z - bifrost_data.zdn)[::-1]
        if not isinstance(quantities, list) and not isinstance(
                quantities, tuple):
            quantities = [quantities]
        k_slice = slice(*((None, ) if height_range is None else np.
                          searchsorted(coords, height_range)))
        values = value_processor(
            *(bifrost_data.get_var(quantity)[i, j, ::-1][k_slice]
              for quantity in quantities))
        coords = coords[k_slice]
        if scale is not None:
            values *= scale

        return ScalarField1(Coords1(coords), values)

    @staticmethod
    def dz_in_bifrost_data(bifrost_data, height_range=None, scale=None):
        coords = -(2 * bifrost_data.z - bifrost_data.zdn)[::-1]
        k_slice = slice(*((None, ) if height_range is None else np.
                          searchsorted(coords, height_range)))
        values = (bifrost_data.z - bifrost_data.zdn)[::-1][k_slice]
        coords = coords[k_slice]
        scale = 2 if scale is None else (2 * scale)
        values *= scale

        return ScalarField1(Coords1(coords), values)

    @staticmethod
    def volumes_in_bifrost_data(bifrost_data, height_range=None, scale=None):
        coords = -(2 * bifrost_data.z - bifrost_data.zdn)[::-1]
        dx = bifrost_data.params['dx'][0]
        dy = bifrost_data.params['dy'][0]
        k_slice = slice(*((None, ) if height_range is None else np.
                          searchsorted(coords, height_range)))
        values = (bifrost_data.z - bifrost_data.zdn)[::-1][k_slice]
        coords = coords[k_slice]
        scale = (2 * dx * dy) if scale is None else (2 * dx * dy * scale)
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
            return ScalarField1(self.coords, self.values * factor.values)
        else:
            return ScalarField1(self.coords, self.values * factor)

    def __truediv__(self, divisor):
        if isinstance(divisor, self.__class__):
            assert np.allclose(self.coords.coords, divisor.coords.coords)
            return ScalarField1(self.coords, self.values / divisor.values)
        else:
            return ScalarField1(self.coords, self.values / divisor)

    def resampled_to_coords(self, coords, kind='cubic'):
        values = scipy.interpolate.interp1d(self.get_coords(),
                                            self.get_values(),
                                            kind=kind,
                                            copy=False,
                                            fill_value='extrapolate')(coords)
        return ScalarField1(Coords1(coords), values)

    def get_size(self):
        return self.coords.get_size()

    def get_values(self, inverted_vertically=False):
        return self.values[::(-1 if inverted_vertically else 1)]

    def get_values_flat(self, **kwargs):
        return self.get_values(**kwargs)

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
        return np.sum(0.5 * (values[:-1] + values[1:]) *
                      np.diff(self.get_coords()))

    def find_peak_coordinate(self):
        return self.coords.coords[np.argmax(self.values)]

    def plot(self, inverted_vertically=False, **plot_kwargs):
        return plotting.plot_1d_field(
            self.get_coords(inverted=inverted_vertically),
            self.get_values(inverted_vertically=inverted_vertically),
            **plot_kwargs)

    def save(self, file_path, compressed=True, overwrite=True):
        if overwrite or not pathlib.Path(file_path).exists():
            if compressed:
                np.savez_compressed(file_path,
                                    coords=self.coords.coords,
                                    values=self.values)
            else:
                np.savez(file_path,
                         coords=self.coords.coords,
                         values=self.values)


class ScalarField2:
    @staticmethod
    def from_pickle_file(file_path):
        import backstaff.reading as reading
        return reading.read_2d_scalar_field(file_path)

    @staticmethod
    def slice_coord_to_idx(bifrost_data, slice_axis, slice_coord):
        all_center_coords = [
            bifrost_data.x, bifrost_data.y, -bifrost_data.z[::-1]
        ]
        slice_idx = np.argmin(
            np.abs(all_center_coords[slice_axis] - slice_coord))
        return slice_idx

    @staticmethod
    def slice_from_bifrost_data(bifrost_data,
                                quantities,
                                slice_axis=1,
                                slice_coord=7.5,
                                slice_idx=None,
                                scale=None,
                                value_processor=lambda x: x):
        if slice_idx is None:
            slice_idx = ScalarField2.slice_coord_to_idx(
                bifrost_data, slice_axis, slice_coord)

        all_slices = [slice(None)] * 3
        all_slices[slice_axis] = slice_idx

        if not isinstance(quantities, list) and not isinstance(
                quantities, tuple):
            quantities = [quantities]

        values = value_processor(
            *(bifrost_data.get_var(quantity)[:, :, ::-1][all_slices[0],
                                                         all_slices[1],
                                                         all_slices[2]]
              for quantity in quantities))
        coords = Coords2.from_bifrost_data(bifrost_data, slice_axis)

        if isinstance(values, dict):
            if scale is not None and scale != 1.0:
                for name in values:
                    values[name] = values[name] * scale

            return ScalarFieldSet(
                **
                {name: ScalarField2(coords, v)
                 for name, v in values.items()})
        else:
            if scale is not None and scale != 1.0:
                values = values * scale

            return ScalarField2(coords, values)

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

        all_values = value_processor(*(bifrost_data.get_var(quantity)
                                       for quantity in quantities))

        values = accum_operator(all_values, axis=accum_axis)
        coords = Coords2.from_bifrost_data(bifrost_data, accum_axis)

        if isinstance(values, dict):
            if accum_axis != 2:
                for name, v in values.items():
                    values[name] = v[:, ::-1]

            if scale is not None:
                for name in values:
                    values[name] *= scale

            return ScalarFieldSet(
                **
                {name: ScalarField2(coords, v)
                 for name, v in values.items()})
        else:
            if accum_axis != 2:
                values = values[:, ::-1]  # Flip z

            if scale is not None:
                values *= scale

            return ScalarField2(coords, values)

    @staticmethod
    def from_file(file_path, field_name_in_set=None):
        try:
            if field_name_in_set is not None:
                return ScalarFieldSet.field_from_file(ScalarField2, file_path,
                                                      field_name_in_set)
        except FileNotFoundError:
            pass
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
            return ScalarField2(self.coords, self.values * factor.values)
        else:
            return ScalarField2(self.coords, self.values * factor)

    def __truediv__(self, divisor):
        if isinstance(divisor, self.__class__):
            assert np.allclose(self.coords.x, divisor.coords.x)
            assert np.allclose(self.coords.y, divisor.coords.y)
            return ScalarField2(self.coords, self.values / divisor.values)
        else:
            return ScalarField2(self.coords, self.values / divisor)

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
        partial = np.sum(0.5 * (values[:, :-1] + values[:, 1:]) *
                         np.diff(self.get_horizontal_coords())[np.newaxis, :],
                         axis=1)
        return np.sum(0.5 * (partial[:-1] + partial[1:]) *
                      np.diff(self.get_vertical_coords()))

    def plot(self, inverted_vertically=False, **plot_kwargs):
        return plotting.plot_2d_field(
            self.get_horizontal_coords(),
            self.get_vertical_coords(inverted=inverted_vertically),
            self.get_values(inverted_vertically=inverted_vertically),
            **plot_kwargs)

    def save(self, file_path, compressed=True, overwrite=True):
        if overwrite or not pathlib.Path(file_path).exists():
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


class ScalarFieldSet:
    @staticmethod
    def field_from_file(field_class, base_file_path, field_name):
        glob_file_path = ScalarFieldSet._get_field_path(base_file_path,
                                                        '*',
                                                        escaper=glob.escape)
        re_file_path = ScalarFieldSet._get_field_path(base_file_path,
                                                      '(.+)',
                                                      escaper=re.escape)
        for file_path in glob.iglob(glob_file_path):
            match = re.search(re_file_path, file_path)
            if match is not None:
                name = match.groups(1)[0]
                if name == field_name:
                    field_file_path = ScalarFieldSet._get_field_path(
                        base_file_path, name)
                    return field_class.from_file(field_file_path)

        raise FileNotFoundError(
            f'No field named {field_name} exists for base path {base_file_path}'
        )

    def __init__(self, **fields):
        assert len(fields) > 0
        self.fields = fields
        self.field_class = next(iter(self.fields.values())).__class__
        for v in self.fields.values():
            assert isinstance(v, self.field_class)

    def save(self, file_path, **kwargs):
        for name, field in self.fields.items():
            field.save(self.__class__._get_field_path(file_path, name),
                       **kwargs)

    def __getitem__(self, name):
        return self.fields[name]

    def __setitem__(self, name, field):
        assert isinstance(field, self.field_class)
        self.fields[name] = field

    @staticmethod
    def _get_field_path(base_file_path,
                        name,
                        separator='.',
                        escaper=lambda x: x):
        fp = pathlib.Path(base_file_path)
        return f'{escaper(str(fp.parent))}{escaper("" if fp.parent == "/" else "/")}{escaper(fp.stem)}{escaper(separator)}{name}{escaper(fp.suffix)}'
