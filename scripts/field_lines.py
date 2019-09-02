import numpy as np
from fields import Coords3

grid_bounds = ((-0.015625, 23.98438),
               (-0.015625, 23.98438),
               (-14.33274, 2.525689))


class FieldLine3:
    def __init__(self, positions, scalar_values, vector_values):
        assert isinstance(positions, Coords3)
        assert isinstance(scalar_values, dict)
        assert isinstance(vector_values, dict)
        self.positions = positions
        self.scalar_values = scalar_values
        self.vector_values = vector_values

    def add_to_plot(self, ax, s=1, c='k'):
        ax.scatter(self.positions.x, self.positions.y, self.positions.z, s=s, c=c)


class FieldLineSet3:
    def __init__(self, field_lines):
        assert isinstance(field_lines, list)
        self.field_lines = field_lines

    def insert(self, field_line):
        assert isinstance(field_line, FieldLine3)
        self.field_lines.append(field_line)

    def add_to_plot(self, ax, s=1, c='k'):
        for field_line in self.field_lines:
            field_line.add_to_plot(ax, s, c)



if __name__ == "__main__":
    import reading
    from pathlib import Path

    field_line_set = reading.read_3d_field_line_set(Path(reading.data_path, 'regular_field_line_set.pickle'))

    import matplotlib.pyplot as plt
    import plotting
    fig, ax = plotting.create_3d_plot()
    plotting.set_3d_plot_extent(ax, *grid_bounds)
    ax.invert_zaxis()

    field_line_set.add_to_plot(ax)

    plt.show()