import os
import pickle
import numpy as np
try:
    import bifrust.fields as fields
    import bifrust.field_lines as field_lines
    import bifrust.electron_beams as electron_beams
except ModuleNotFoundError:
    import fields
    import field_lines
    import electron_beams

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
    return __parse_fieldlineset3(data, **kwargs)


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
    return __parse_fieldlineset3(data, **kwargs)


def read_3d_field_line_set_from_custom_binary_file(file_path, **kwargs):
    with open(file_path, 'rb') as f:
        field_line_set = field_lines.FieldLineSet3(
            *__parse_custom_field_line_binary_file(f), **kwargs)
    return field_line_set


def read_electron_beam_swarm_from_single_pickle(file_path, **kwargs):
    with open(file_path, mode='rb') as f:
        data = pickle.load(f)
    return __parse_electronbeamswarm(data, **kwargs)


def read_electron_beam_swarm_from_combined_pickles(file_path, **kwargs):
    data = {}
    with open(file_path, mode='rb') as f:
        data['lower_bounds'] = pickle.load(f)
        data['upper_bounds'] = pickle.load(f)
        data['number_of_beams'] = pickle.load(f)
        data['fixed_scalar_values'] = pickle.load(f)
        data['fixed_vector_values'] = pickle.load(f)
        data['varying_scalar_values'] = pickle.load(f)
        data['varying_vector_values'] = pickle.load(f)
        data['acceleration_data'] = pickle.load(f)
    return __parse_electronbeamswarm(data, **kwargs)


def read_electron_beam_swarm_from_custom_binary_file(file_path, **kwargs):
    with open(file_path, 'rb') as f:
        electron_beam_swarm = electron_beams.ElectronBeamSwarm(
            *__parse_custom_electron_beam_binary_file(f), **kwargs)
    return electron_beam_swarm


def __parse_custom_field_line_binary_file(f):
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

    split_indices_of_field_line_elements = np.fromfile(
        f, dtype=np.dtype('<u8'), count=number_of_field_lines, sep='')[1:]

    fixed_scalar_values_shape = (number_of_fixed_scalar_quantities,
                                 number_of_field_lines)
    fixed_scalar_values = np.fromfile(
        f,
        dtype=float_dtype,
        count=np.product(fixed_scalar_values_shape, dtype=int),
        sep='').reshape(fixed_scalar_values_shape)

    fixed_vector_values_shape = (number_of_fixed_vector_quantities,
                                 number_of_field_lines, 3)
    fixed_vector_values = np.fromfile(
        f,
        dtype=float_dtype,
        count=np.product(fixed_vector_values_shape, dtype=int),
        sep='').reshape(fixed_vector_values_shape)

    varying_scalar_values_shape = (number_of_varying_scalar_quantities,
                                   number_of_field_line_elements)
    varying_scalar_values = np.fromfile(
        f,
        dtype=float_dtype,
        count=np.product(varying_scalar_values_shape, dtype=int),
        sep='').reshape(varying_scalar_values_shape)

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
             for n in range(fixed_scalar_values.shape[0]))))

    fixed_vector_values = dict(
        zip(fixed_vector_names,
            (fixed_vector_values[n, :, :]
             for n in range(fixed_vector_values.shape[0]))))

    varying_scalar_values = dict(
        zip(varying_scalar_names,
            (np.split(varying_scalar_values[n, :],
                      split_indices_of_field_line_elements)
             for n in range(varying_scalar_values.shape[0]))))

    varying_vector_values = dict(
        zip(varying_vector_names,
            (np.split(varying_vector_values[n, :, :],
                      split_indices_of_field_line_elements,
                      axis=0) for n in range(varying_vector_values.shape[0]))))

    return domain_bounds, \
        int(number_of_field_lines), \
        fixed_scalar_values, \
        fixed_vector_values, \
        varying_scalar_values, \
        varying_vector_values


def __parse_custom_electron_beam_binary_file(f):
    domain_bounds, \
        number_of_beams, \
        fixed_scalar_values, \
        fixed_vector_values, \
        varying_scalar_values, \
        varying_vector_values = __parse_custom_field_line_binary_file(f)
    acceleration_data = __parse_electronbeamswarm_acceleration_data(
        pickle.load(f))
    return domain_bounds, \
        number_of_beams, \
        fixed_scalar_values, \
        fixed_vector_values, \
        varying_scalar_values, \
        varying_vector_values, \
        acceleration_data


def __parse_fieldlineset3(data, **kwargs):
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


def __parse_electronbeamswarm(data, **kwargs):
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
    acceleration_data = __parse_electronbeamswarm_acceleration_data(
        data['acceleration_data'])
    return electron_beams.ElectronBeamSwarm(domain_bounds, number_of_beams,
                                            fixed_scalar_values,
                                            fixed_vector_values,
                                            varying_scalar_values,
                                            varying_vector_values,
                                            acceleration_data, **kwargs)


def __parse_electronbeamswarm_acceleration_data(acceleration_data):
    label = acceleration_data[0]
    values = acceleration_data[1]
    if label == 'rejection_cause_code':
        acceleration_data = {label: np.array(values, dtype=np.ubyte)}
    else:
        raise NotImplementedError(
            'Invalid acceleration data label {}'.format(label))
    return acceleration_data


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
