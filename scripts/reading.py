import os
import pickle
import numpy as np
from fields import Coords3, Coords2, ScalarField2
from field_lines import FieldLineSet3
from electron_beams import ElectronBeamSwarm

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
    return __parse_electronbeamswarm(data, **kwargs)


def read_3d_field_line_set_from_combined_pickles(file_path, **kwargs):
    data = {}
    with open(file_path, mode='rb') as f:
        data['number_of_field_lines'] = pickle.load(f)
        data['fixed_scalar_values'] = pickle.load(f)
        data['fixed_vector_values'] = pickle.load(f)
        data['varying_scalar_values'] = pickle.load(f)
        data['varying_vector_values'] = pickle.load(f)
    return __parse_fieldlineset3(data, **kwargs)


def __parse_fieldlineset3(data, **kwargs):
    number_of_field_lines = data.pop('number_of_field_lines')
    fixed_scalar_values = __parse_map_of_vec_of_float(
        data['fixed_scalar_values'])
    fixed_vector_values = __parse_map_of_vec_of_vec3(
        data.pop('fixed_vector_values'))
    varying_scalar_values = __parse_map_of_vec_of_vec_of_float(
        data.pop('varying_scalar_values'))
    varying_vector_values = __parse_map_of_vec_of_vec_of_vec3(
        data.pop('varying_vector_values'))
    return FieldLineSet3(number_of_field_lines, fixed_scalar_values,
                         fixed_vector_values, varying_scalar_values,
                         varying_vector_values, **kwargs)


def read_electron_beam_swarm_from_single_pickle(file_path, **kwargs):
    with open(file_path, mode='rb') as f:
        data = pickle.load(f)
    return __parse_electronbeamswarm(data, **kwargs)


def read_electron_beam_swarm_from_combined_pickles(file_path, **kwargs):
    data = {}
    with open(file_path, mode='rb') as f:
        data['number_of_beams'] = pickle.load(f)
        data['fixed_scalar_values'] = pickle.load(f)
        data['fixed_vector_values'] = pickle.load(f)
        data['varying_scalar_values'] = pickle.load(f)
        data['varying_vector_values'] = pickle.load(f)
        data['metadata'] = pickle.load(f)
    return __parse_electronbeamswarm(data, **kwargs)


def __parse_electronbeamswarm(data, **kwargs):
    number_of_beams = data.pop('number_of_beams')
    fixed_scalar_values = __parse_map_of_vec_of_float(
        data['fixed_scalar_values'])
    fixed_vector_values = __parse_map_of_vec_of_vec3(
        data.pop('fixed_vector_values'))
    varying_scalar_values = __parse_map_of_vec_of_vec_of_float(
        data.pop('varying_scalar_values'))
    varying_vector_values = __parse_map_of_vec_of_vec_of_vec3(
        data.pop('varying_vector_values'))
    metadata = __parse_electronbeamswarm_metadata(data['metadata'])
    return ElectronBeamSwarm(number_of_beams, fixed_scalar_values,
                             fixed_vector_values, varying_scalar_values,
                             varying_vector_values, metadata, **kwargs)


def __parse_electronbeamswarm_metadata(metadata):
    label = metadata[0]
    values = metadata[1]
    if label == 'rejection_cause_code':
        metadata = {label: np.array(values, dtype=np.ubyte)}
    else:
        raise NotImplementedError('Invalid metadata label {}'.format(label))
    return metadata


def __parse_scalarfield2(data):
    coords = __parse_coords2(data['coords'])
    values = __parse_ndarray(data['values'])
    return ScalarField2(coords, values)


def __parse_coords2(data):
    return Coords2(__parse_vec_of_float(data[0]),
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
    return Coords3(arr[0, :], arr[1, :], arr[2, :])


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
