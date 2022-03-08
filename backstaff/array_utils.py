import os
import time
import tempfile
import scipy.interpolate
import numpy as np
from numba import njit
from joblib import Parallel, delayed


class CompactArrayMask:
    def __init__(self, mask):
        self.__mask = mask
        self.__included_flat_indices = np.flatnonzero(mask)
        self.compute_max_axis_ranges()
        self.compute_axis_splits()

    @property
    def mask(self):
        return self.__mask

    @property
    def shape(self):
        return self.mask.shape

    @property
    def size(self):
        return self.mask.size

    @property
    def compact_size(self):
        return self.included_flat_indices.size

    @property
    def included_flat_indices(self):
        return self.__included_flat_indices

    @property
    def axis_splits(self):
        return self.__axis_splits

    @property
    def axis_ranges(self):
        return self.__axis_ranges

    def shape_except_axis(self, axis):
        shape = list(self.shape)
        shape.pop(axis)
        return shape

    def compute_max_axis_ranges(self):
        included_indices = np.unravel_index(self.included_flat_indices,
                                            self.shape)
        self.__axis_ranges = [(np.min(indices_for_axis),
                               np.max(indices_for_axis) + 1)
                              for indices_for_axis in included_indices]

    def compute_axis_splits(self):
        axis_indices = [np.arange(n) for n in self.shape]
        split_multi_indices = []
        for axis in range(len(self.shape)):
            indices = []
            for i in range(0, axis):
                indices.append(axis_indices[i])
            for i in range(axis + 1, len(self.shape)):
                indices.append(axis_indices[i])
            mesh_indices = list(
                map(np.ravel, np.meshgrid(*indices, indexing='ij')))
            mesh_indices.insert(
                axis,
                np.zeros(np.product(self.shape_except_axis(axis)), dtype=int))
            split_multi_indices.append(mesh_indices)

        self.__axis_splits = [
            np.searchsorted(
                self.included_flat_indices,
                np.ravel_multi_index(split_multi_indices[axis], self.shape))
            for axis in range(len(self.shape))
        ]

    def apply(self, arr):
        assert arr.shape == self.shape
        return np.ravel(arr)[self.included_flat_indices]

    def sum_over_axis(self, subsectioned_arr, axis=0):
        assert subsectioned_arr.shape == self.included_flat_indices.shape
        return np.add.reduceat(subsectioned_arr,
                               self.axis_splits[axis]).reshape(
                                   self.shape_except_axis(axis))


class tempmap(np.memmap):
    def __new__(subtype,
                dtype=np.uint8,
                mode='r+',
                offset=0,
                shape=None,
                order='C'):
        filename = tempfile.mkstemp()[1]
        self = np.memmap.__new__(subtype,
                                 filename,
                                 dtype=dtype,
                                 mode=mode,
                                 offset=offset,
                                 shape=shape,
                                 order=order)
        return self

    def __del__(self):
        if self.filename is not None and os.path.isfile(self.filename):
            os.remove(self.filename)


def create_tmp_memmap(shape, dtype, mode='w+'):
    return tempmap(shape=shape, dtype=dtype, mode=mode)


def concurrent_interp2(xp, yp, fp, coords, verbose=False, n_jobs=1, **kwargs):
    n = fp.shape[0]
    f = create_tmp_memmap(shape=(n, coords.shape[0]), dtype=fp.dtype)
    chunk_sizes = np.full(n_jobs, n // n_jobs, dtype=int)
    chunk_sizes[:(n % n_jobs)] += 1
    stop_indices = np.cumsum(chunk_sizes)
    start_indices = stop_indices - chunk_sizes
    Parallel(n_jobs=min(n_jobs, n), verbose=verbose)(
        delayed(do_concurrent_interp2)(
            f, start, stop, xp, yp, fp, coords, verbose=verbose, **kwargs)
        for start, stop in zip(start_indices, stop_indices))
    return f


def do_concurrent_interp2(f,
                          start,
                          stop,
                          xp,
                          yp,
                          fp,
                          coords,
                          verbose=False,
                          **kwargs):
    if verbose and start == 0:
        start_time = time.time()
        f[start, :] = scipy.interpolate.interpn((xp, yp), fp[start, :, :],
                                                coords, **kwargs)
        elapsed_time = time.time() - start_time
        print(
            f'Single interpolation took {elapsed_time:g} s, estimated total interpolation time is {elapsed_time*stop:g} s'
        )
        start += 1
    for idx in range(start, stop):
        f[idx, :] = scipy.interpolate.interpn((xp, yp), fp[idx, :, :], coords,
                                              **kwargs)


@njit
def add_values_in_matrix(matrix, rows, cols, values):
    assert rows.size == cols.size and rows.size == values.size
    for i in range(rows.size):
        matrix[rows[i], cols[i]] += values[i]


@njit
def subtract_values_in_matrix(matrix, rows, cols, values):
    assert rows.size == cols.size and rows.size == values.size
    for i in range(rows.size):
        matrix[rows[i], cols[i]] -= values[i]


@njit
def add_values_in_matrices(matrices, rows, cols, values):
    assert rows.size == cols.size and rows.size == values.shape[0]
    for i in range(rows.size):
        matrices[:, rows[i], cols[i]] += values[i, :]


@njit
def subtract_values_in_matrices(matrices, rows, cols, values):
    assert rows.size == cols.size and rows.size == values.shape[0]
    for i in range(rows.size):
        matrices[:, rows[i], cols[i]] -= values[i, :]


@njit
def set_values_in_matrices(matrices, rows, cols, values):
    assert rows.size == cols.size and rows.size == values.shape[0]
    for i in range(rows.size):
        matrices[:, rows[i], cols[i]] = values[i, :]


@njit
def sum_nondiagonal_elements_in_matrix(matrix):
    assert len(matrix.shape) == 2 and matrix.shape[0] == matrix.shape[1]
    size = matrix.shape[0]
    col_sums = np.zeros(size)
    row_sums = np.zeros(size)
    for i in range(size):
        for j in range(0, i):
            row_sums[i] += matrix[i, j]
            col_sums[i] += matrix[j, i]
        for j in range(i + 1, size):
            row_sums[i] += matrix[i, j]
            col_sums[i] += matrix[j, i]
    return col_sums, row_sums
