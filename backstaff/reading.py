import os
import pickle
import numpy as np
try:
    import backstaff.fields as fields
    import backstaff.field_lines as field_lines
    import backstaff.electron_beams as electron_beams
    import backstaff.corks as corks
except ModuleNotFoundError:
    import fields
    import field_lines
    import electron_beams
    import corks

SCRIPTS_PATH = os.path.dirname(os.path.realpath(__file__))
PROJECT_PATH = os.path.dirname(SCRIPTS_PATH)
DATA_PATH = os.path.join(PROJECT_PATH, 'data')


def read_2d_scalar_field(file_path):
    with open(file_path, mode='rb') as f:
        data = pickle.load(f)
    return __parse_scalarfield2(data)


def read_3d_field_line_set_from_single_pickle(file_path, **kwargs):
    with open(file_path, mode='rb') as f:
        data = pickle.load(f)
    return __parse_fieldlineset3_pickle(data, **kwargs)


def read_3d_field_line_set_from_combined_pickles(file_path, **kwargs):
    data = {}
    with open(file_path, mode='rb') as f:
        data['lower_bounds'] = pickle.load(f)
        data['upper_bounds'] = pickle.load(f)
        data['number_of_field_lines'] = pickle.load(f)
        data['fixed_scalar_values'] = pickle.load(f)
        data['fixed_vector_values'] = pickle.load(f)
        data['varying_scalar_values'] = pickle.load(f)
        data['varying_vector_values'] = pickle.load(f)
    return __parse_fieldlineset3_pickle(data, **kwargs)


def read_3d_field_line_set_from_custom_binary_file(file_path,
                                                   memmap=True,
                                                   **kwargs):
    with open(file_path, 'rb') as f:
        field_line_set = field_lines.FieldLineSet3(
            *__parse_custom_field_line_binary_file(f, memmap=memmap), **kwargs)
    return field_line_set


def read_electron_beam_swarm_from_single_pickle(file_path,
                                                acceleration_data_type=None,
                                                **kwargs):
    with open(file_path, mode='rb') as f:
        data = pickle.load(f)
    return __parse_electronbeamswarm_pickle(data, acceleration_data_type,
                                            **kwargs)


def read_electron_beam_swarm_from_combined_pickles(file_path,
                                                   acceleration_data_type=None,
                                                   **kwargs):
    data = {}
    with open(file_path, mode='rb') as f:
        data['lower_bounds'] = pickle.load(f)
        data['upper_bounds'] = pickle.load(f)
        data['number_of_beams'] = pickle.load(f)
        data['fixed_scalar_values'] = pickle.load(f)
        data['fixed_vector_values'] = pickle.load(f)
        data['varying_scalar_values'] = pickle.load(f)
        data['varying_vector_values'] = pickle.load(f)
        if acceleration_data_type is not None:
            data['acceleration_data'] = pickle.load(f)
    return __parse_electronbeamswarm_pickle(data, acceleration_data_type,
                                            **kwargs)


def read_electron_beam_swarm_from_custom_binary_file(
        file_path, acceleration_data_type=None, memmap=True, **kwargs):
    with open(file_path, 'rb') as f:
        electron_beam_swarm = electron_beams.ElectronBeamSwarm(
            *__parse_custom_electron_beam_binary_file(f,
                                                      acceleration_data_type,
                                                      memmap=memmap), **kwargs)
    return electron_beam_swarm


def read_cork_set_from_pickle(file_path, **kwargs):
    with open(file_path, mode='rb') as f:
        data = pickle.load(f)
    return __parse_cork_set_pickle(data, **kwargs)


class _SplittedArray1D:
    def __init__(self, buffer, split_indices):
        self.buffer = buffer
        self.split_indices = split_indices
        self.n_parts = split_indices.size

    def __len__(self):
        return self.n_parts

    def __getitem__(self, idx):
        return self.buffer[self._create_slice(idx)]

    def _create_slice(self, idx):
        if isinstance(idx, int):
            if idx < 0:
                idx = self.n_parts + idx
            start = idx
            end = idx + 1
        elif isinstance(idx, slice):
            start, end, stride = idx.indices(self.n_parts)
            if stride != 1:
                raise ValueError('stride must be 1, not {:d}'.format(stride))
            if start < 0:
                start = self.n_parts + start
            if end < 0:
                end = self.n_parts + end
        else:
            raise TypeError(
                'list indices must be integers or slices, not {}'.format(
                    idx.__class__.__name__))

        return slice(self.split_indices[start],
                     None if end == self.n_parts else self.split_indices[end])


class _SplittedArray2D(_SplittedArray1D):
    def __getitem__(self, idx):
        return self.buffer[self._create_slice(idx), :]


def __parse_custom_field_line_binary_file(f, memmap=True):
    if memmap:
        return __parse_custom_field_line_binary_file_memmap(f)
    else:
        return __parse_custom_field_line_binary_file_load_all(f)


def __parse_custom_field_line_binary_file_memmap(f):
    counts = np.fromfile(f, dtype=np.dtype('<u8'), count=7, sep='')

    float_size, \
        number_of_field_lines, \
        number_of_field_line_elements, \
        number_of_fixed_scalar_quantities, \
        number_of_fixed_vector_quantities, \
        number_of_varying_scalar_quantities, \
        number_of_varying_vector_quantities = tuple(map(int, list(counts)))

    float_dtype = np.dtype('<f{:d}'.format(float_size))

    domain_bounds = np.fromfile(f, dtype=float_dtype, count=6, sep='')
    domain_bounds = list(zip(domain_bounds[0::2], domain_bounds[1::2]))

    fixed_scalar_names = [
        f.readline().decode('utf-8').strip()
        for _ in range(number_of_fixed_scalar_quantities)
    ]
    fixed_vector_names = [
        f.readline().decode('utf-8').strip()
        for _ in range(number_of_fixed_vector_quantities)
    ]
    varying_scalar_names = [
        f.readline().decode('utf-8').strip()
        for _ in range(number_of_varying_scalar_quantities)
    ]
    varying_vector_names = [
        f.readline().decode('utf-8').strip()
        for _ in range(number_of_varying_vector_quantities)
    ]

    def running_memmap(f, dtype, shape):
        byte_offset = f.tell()
        m = np.memmap(f,
                      dtype=dtype,
                      mode='r',
                      offset=byte_offset,
                      shape=shape)
        mapped_bytes = np.product(shape, dtype=int) * dtype.itemsize
        f.seek(byte_offset + mapped_bytes)
        return m

    if number_of_field_line_elements > 0:
        split_indices_of_field_line_elements = running_memmap(
            f, np.dtype('<u8'), (number_of_field_lines, ))

    if number_of_fixed_scalar_quantities > 0:
        fixed_scalar_values_shape = (number_of_fixed_scalar_quantities,
                                     number_of_field_lines)
        fixed_scalar_values = running_memmap(f, float_dtype,
                                             fixed_scalar_values_shape)

    if number_of_fixed_vector_quantities > 0:
        fixed_vector_values_shape = (number_of_fixed_vector_quantities,
                                     number_of_field_lines, 3)
        fixed_vector_values = running_memmap(f, float_dtype,
                                             fixed_vector_values_shape)

    if number_of_varying_scalar_quantities > 0:
        varying_scalar_values_shape = (number_of_varying_scalar_quantities,
                                       number_of_field_line_elements)
        varying_scalar_values = running_memmap(f, float_dtype,
                                               varying_scalar_values_shape)

    if number_of_varying_vector_quantities > 0:
        varying_vector_values_shape = (number_of_varying_vector_quantities,
                                       number_of_field_line_elements, 3)
        varying_vector_values = running_memmap(f, float_dtype,
                                               varying_vector_values_shape)

    fixed_scalar_values = dict(
        zip(fixed_scalar_names,
            (fixed_scalar_values[n, :]
             for n in range(fixed_scalar_values.shape[0])
             ))) if number_of_fixed_scalar_quantities > 0 else {}

    fixed_vector_values = dict(
        zip(fixed_vector_names,
            (fixed_vector_values[n, :, :]
             for n in range(fixed_vector_values.shape[0])
             ))) if number_of_fixed_vector_quantities > 0 else {}

    varying_scalar_values = dict(
        zip(varying_scalar_names,
            (_SplittedArray1D(varying_scalar_values[n, :],
                              split_indices_of_field_line_elements)
             for n in range(varying_scalar_values.shape[0])
             ))) if number_of_varying_scalar_quantities > 0 else {}

    varying_vector_values = dict(
        zip(varying_vector_names,
            (_SplittedArray2D(varying_vector_values[n, :, :],
                              split_indices_of_field_line_elements)
             for n in range(varying_vector_values.shape[0])
             ))) if number_of_varying_vector_quantities > 0 else {}

    return domain_bounds, \
        int(number_of_field_lines), \
        fixed_scalar_values, \
        fixed_vector_values, \
        varying_scalar_values, \
        varying_vector_values


def __parse_custom_field_line_binary_file_load_all(f):
    counts = np.fromfile(f, dtype=np.dtype('<u8'), count=7, sep='')

    float_size, \
        number_of_field_lines, \
        number_of_field_line_elements, \
        number_of_fixed_scalar_quantities, \
        number_of_fixed_vector_quantities, \
        number_of_varying_scalar_quantities, \
        number_of_varying_vector_quantities = tuple(counts)

    float_dtype = np.dtype('<f{:d}'.format(float_size))

    domain_bounds = np.fromfile(f, dtype=float_dtype, count=6, sep='')
    domain_bounds = list(zip(domain_bounds[0::2], domain_bounds[1::2]))

    fixed_scalar_names = [
        f.readline().decode('utf-8').strip()
        for _ in range(number_of_fixed_scalar_quantities)
    ]
    fixed_vector_names = [
        f.readline().decode('utf-8').strip()
        for _ in range(number_of_fixed_vector_quantities)
    ]
    varying_scalar_names = [
        f.readline().decode('utf-8').strip()
        for _ in range(number_of_varying_scalar_quantities)
    ]
    varying_vector_names = [
        f.readline().decode('utf-8').strip()
        for _ in range(number_of_varying_vector_quantities)
    ]

    if number_of_field_line_elements > 0:
        split_indices_of_field_line_elements = np.fromfile(
            f, dtype=np.dtype('<u8'), count=number_of_field_lines, sep='')[1:]

    if number_of_fixed_scalar_quantities > 0:
        fixed_scalar_values_shape = (number_of_fixed_scalar_quantities,
                                     number_of_field_lines)
        fixed_scalar_values = np.fromfile(
            f,
            dtype=float_dtype,
            count=np.product(fixed_scalar_values_shape, dtype=int),
            sep='').reshape(fixed_scalar_values_shape)

    if number_of_fixed_vector_quantities > 0:
        fixed_vector_values_shape = (number_of_fixed_vector_quantities,
                                     number_of_field_lines, 3)
        fixed_vector_values = np.fromfile(
            f,
            dtype=float_dtype,
            count=np.product(fixed_vector_values_shape, dtype=int),
            sep='').reshape(fixed_vector_values_shape)

    if number_of_varying_scalar_quantities > 0:
        varying_scalar_values_shape = (number_of_varying_scalar_quantities,
                                       number_of_field_line_elements)
        varying_scalar_values = np.fromfile(
            f,
            dtype=float_dtype,
            count=np.product(varying_scalar_values_shape, dtype=int),
            sep='').reshape(varying_scalar_values_shape)

    if number_of_varying_vector_quantities > 0:
        varying_vector_values_shape = (number_of_varying_vector_quantities,
                                       number_of_field_line_elements, 3)
        varying_vector_values = np.fromfile(
            f,
            dtype=float_dtype,
            count=np.product(varying_vector_values_shape, dtype=int),
            sep='').reshape(varying_vector_values_shape)

    fixed_scalar_values = dict(
        zip(fixed_scalar_names,
            (fixed_scalar_values[n, :]
             for n in range(fixed_scalar_values.shape[0])
             ))) if number_of_fixed_scalar_quantities > 0 else {}

    fixed_vector_values = dict(
        zip(fixed_vector_names,
            (fixed_vector_values[n, :, :]
             for n in range(fixed_vector_values.shape[0])
             ))) if number_of_fixed_vector_quantities > 0 else {}

    varying_scalar_values = dict(
        zip(varying_scalar_names,
            (np.split(varying_scalar_values[n, :],
                      split_indices_of_field_line_elements)
             for n in range(varying_scalar_values.shape[0])
             ))) if number_of_varying_scalar_quantities > 0 else {}

    varying_vector_values = dict(
        zip(varying_vector_names,
            (np.split(varying_vector_values[n, :, :],
                      split_indices_of_field_line_elements,
                      axis=0) for n in range(varying_vector_values.shape[0])
             ))) if number_of_varying_vector_quantities > 0 else {}

    return domain_bounds, \
        int(number_of_field_lines), \
        fixed_scalar_values, \
        fixed_vector_values, \
        varying_scalar_values, \
        varying_vector_values


def __parse_custom_electron_beam_binary_file(f,
                                             acceleration_data_type,
                                             memmap=True):
    domain_bounds, \
        number_of_beams, \
        fixed_scalar_values, \
        fixed_vector_values, \
        varying_scalar_values, \
        varying_vector_values = __parse_custom_field_line_binary_file(f, memmap=memmap)

    if acceleration_data_type is None:
        acceleration_data = {}
    elif acceleration_data_type == 'acceleration_sites':
        acceleration_data = electron_beams.AccelerationSites(
            *__parse_custom_field_line_binary_file(f, memmap=memmap))
    else:
        raise ValueError(
            'Invalid acceleration data type {}'.format(acceleration_data_type))

    return domain_bounds, \
        number_of_beams, \
        fixed_scalar_values, \
        fixed_vector_values, \
        varying_scalar_values, \
        varying_vector_values, \
        acceleration_data


def __parse_fieldlineset3_pickle(data, **kwargs):
    lower_bounds = data.pop('lower_bounds')
    upper_bounds = data.pop('upper_bounds')
    domain_bounds = list(zip(lower_bounds, upper_bounds))
    number_of_field_lines = data.pop('number_of_field_lines')
    fixed_scalar_values = __parse_map_of_vec_of_float(
        data['fixed_scalar_values'])
    fixed_vector_values = __parse_map_of_vec_of_vec3(
        data.pop('fixed_vector_values'))
    varying_scalar_values = __parse_map_of_vec_of_vec_of_float(
        data.pop('varying_scalar_values'))
    varying_vector_values = __parse_map_of_vec_of_vec_of_vec3(
        data.pop('varying_vector_values'))
    return field_lines.FieldLineSet3(domain_bounds, number_of_field_lines,
                                     fixed_scalar_values, fixed_vector_values,
                                     varying_scalar_values,
                                     varying_vector_values, **kwargs)


def __parse_electronbeamswarm_pickle(data, acceleration_data_type, **kwargs):
    lower_bounds = data.pop('lower_bounds')
    upper_bounds = data.pop('upper_bounds')
    domain_bounds = list(zip(lower_bounds, upper_bounds))
    number_of_beams = data.pop('number_of_beams')
    fixed_scalar_values = __parse_map_of_vec_of_float(
        data['fixed_scalar_values'])
    fixed_vector_values = __parse_map_of_vec_of_vec3(
        data.pop('fixed_vector_values'))
    varying_scalar_values = __parse_map_of_vec_of_vec_of_float(
        data.pop('varying_scalar_values'))
    varying_vector_values = __parse_map_of_vec_of_vec_of_vec3(
        data.pop('varying_vector_values'))
    acceleration_data = __parse_electronbeamswarm_acceleration_data_pickle(
        data['acceleration_data'], acceleration_data_type)
    return electron_beams.ElectronBeamSwarm(domain_bounds, number_of_beams,
                                            fixed_scalar_values,
                                            fixed_vector_values,
                                            varying_scalar_values,
                                            varying_vector_values,
                                            acceleration_data, **kwargs)


def __parse_electronbeamswarm_acceleration_data_pickle(acceleration_data,
                                                       acceleration_data_type):
    if acceleration_data_type is None:
        acceleration_data = {}
    elif acceleration_data_type == 'acceleration_sites':
        acceleration_data = {
            acceleration_data_type:
            __parse_fieldlineset3_pickle(acceleration_data)
        }
    else:
        raise NotImplementedError(
            'Invalid acceleration data type {}'.format(acceleration_data_type))
    return acceleration_data


def __parse_cork_set_pickle(data, **kwargs):
    all_corks = [__parse_cork_pickle(cork) for cork in data.pop('corks')]
    times = __parse_vec_of_float(data.pop('times'))
    lower_bounds = data.pop('lower_bounds')
    upper_bounds = data.pop('upper_bounds')
    domain_bounds = list(zip(lower_bounds, upper_bounds))
    scalar_quantity_names = data.pop('scalar_quantity_names')
    vector_magnitude_names = data.pop('vector_magnitude_names')
    vector_quantity_names = data.pop('vector_quantity_names')
    return corks.CorkSet(all_corks, times, domain_bounds,
                         scalar_quantity_names, vector_magnitude_names,
                         vector_quantity_names, **kwargs)


def __parse_cork_pickle(data, **kwargs):
    positions = __parse_vec_of_vec3(data.pop('positions'))
    velocities = __parse_vec_of_vec3(data.pop('velocities'))
    scalar_field_values = __parse_vec_of_vec_of_float(
        data.pop('scalar_field_values'))
    vector_field_values = __parse_vec_of_vec_of_vec3(
        data.pop('vector_field_values'))
    first_time_idx = data.pop('first_time_idx')
    return corks.Cork(positions, velocities, scalar_field_values,
                      vector_field_values, first_time_idx, **kwargs)


def __parse_scalarfield2(data):
    coords = __parse_coords2(data['coords'])
    values = __parse_ndarray(data['values'])
    return fields.ScalarField2(coords, values)


def __parse_coords2(data):
    return fields.Coords2(__parse_vec_of_float(data[0]),
                          __parse_vec_of_float(data[1]))


def __parse_ndarray(data):
    return np.asfarray(data['data']).reshape(data['dim'])


def __parse_vec_of_float(data):
    return np.asfarray(data)


def __parse_map_of_vec3(data):
    return {name: np.asfarray(vec3) for name, vec3 in data.items()}


def __parse_vec_of_vec3(data):
    return np.asfarray(data).T


def __parse_vec_of_vec3_into_coords3(data):
    arr = __parse_vec_of_vec3(data)
    return fields.Coords3(arr[0, :], arr[1, :], arr[2, :])


def __parse_vec_of_vec_of_vec3(data):
    return [__parse_vec_of_vec3(vec) for vec in data]


def __parse_vec_of_vec_of_float(data):
    return [__parse_vec_of_float(vec) for vec in data]


def __parse_vec_of_vec_of_vec3_into_vec_of_coords3(data):
    return [__parse_vec_of_vec3_into_coords3(vec) for vec in data]


def __parse_map_of_vec_of_float(data):
    return {name: __parse_vec_of_float(vec) for name, vec in data.items()}


def __parse_map_of_vec_of_vec3(data):
    return {name: __parse_vec_of_vec3(vec) for name, vec in data.items()}


def __parse_map_of_vec_of_vec_of_float(data):
    return {
        name: __parse_vec_of_vec_of_float(vec)
        for name, vec in data.items()
    }


def __parse_map_of_vec_of_vec_of_vec3(data):
    return {
        name: __parse_vec_of_vec_of_vec3(vec)
        for name, vec in data.items()
    }
