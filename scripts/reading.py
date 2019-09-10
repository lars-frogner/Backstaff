import os
import pickle
import numpy as np
from pathlib import Path
from fields import Coords3, Coords2, ScalarField2
from field_lines import FieldLine3, FieldLineSet3

scripts_path = os.path.dirname(os.path.realpath(__file__))
project_path = os.path.dirname(scripts_path)
data_path = os.path.join(project_path, 'data')


def read_2d_scalar_field(file_path):
    with file_path.open(mode='rb') as f:
        data = pickle.load(f)
    return __parse_scalarfield2(data)


def read_3d_field_line(file_path):
    with file_path.open(mode='rb') as f:
        data = pickle.load(f)
    return __parse_fieldline3(data)


def read_3d_field_line_set_from_single_pickle(file_path):
    with file_path.open(mode='rb') as f:
        data = pickle.load(f)
    return __parse_fieldlineset3(data)


def read_3d_field_line_set_from_combined_pickles(file_path):
    field_line_set = FieldLineSet3([])
    with file_path.open(mode='rb') as f:
        while True:
            try:
                data = pickle.load(f)
            except EOFError:
                break
            field_line = __parse_fieldline3(data)
            field_line_set.insert(field_line)
    return field_line_set


def __parse_scalarfield2(data):
    coords = __parse_coords2(data['coords'])
    values = __parse_ndarray(data['values'])
    return ScalarField2(coords, values)

def __parse_fieldlineset3(data):
    return FieldLineSet3([__parse_fieldline3(d) for d in data['field_lines']])

def __parse_fieldline3(data):
    positions = __parse_vec_of_vec3(data['positions'])
    scalar_values = __parse_map_of_vec_of_float(data['scalar_values'])
    vector_values = __parse_map_of_vec_of_float(data['vector_values'])
    return FieldLine3(positions, scalar_values, vector_values)

def __parse_ndarray(data):
    return np.asfarray(data['data']).reshape(data['dim'])

def __parse_coords2(data):
    return Coords2(__parse_vec_of_float(data[0]), __parse_vec_of_float(data[1]))

def __parse_vec_of_float(data):
    return np.asfarray(data)

def __parse_vec_of_vec3(data):
    return Coords3(*list(zip(*data)))

def __parse_map_of_vec_of_float(data):
    return {name: np.asfarray(vec) for name, vec in data.items()}

def __parse_map_of_vec_of_vec3(data):
    return {name: __parse_vec_of_vec3(vec) for name, vec in data.items()}