import numpy as np
from numba import njit


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
