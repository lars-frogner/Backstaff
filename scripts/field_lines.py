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

    def add_to_plot_as_scatter(self, ax, c='k', s=1.0):
        ax.scatter(self.positions.x, self.positions.y, self.positions.z, c=c, s=s)

    def add_to_plot_as_line(self, ax, c='k', lw=1.0, alpha=1.0):
        for pos_x, pos_y, pos_z in zip(*self.find_nonwrapping_segments()):
            ax.plot(pos_x, pos_y, pos_z, c=c, lw=lw, alpha=alpha)

    def find_nonwrapping_segments(self, threshold=20.0):
        step_sizes = np.sqrt(np.diff(self.positions.x)**2 + np.diff(self.positions.y)**2 + np.diff(self.positions.z)**2)
        wrap_indices = np.where(step_sizes > threshold*np.mean(step_sizes))[0]
        if wrap_indices.size > 0:
            wrap_indices += 1
            return np.split(self.positions.x, wrap_indices), \
                   np.split(self.positions.y, wrap_indices), \
                   np.split(self.positions.z, wrap_indices)
        else:
            return [self.positions.x], [self.positions.y], [self.positions.z]


class FieldLineSet3:
    def __init__(self, field_lines):
        assert isinstance(field_lines, list)
        self.field_lines = field_lines

    def insert(self, field_line):
        assert isinstance(field_line, FieldLine3)
        self.field_lines.append(field_line)

    def add_to_plot_as_scatter(self, ax, **kwargs):
        for field_line in self.field_lines:
            field_line.add_to_plot_as_scatter(ax, **kwargs)

    def add_to_plot_as_line(self, ax, **kwargs):
        for field_line in self.field_lines:
            field_line.add_to_plot_as_line(ax, **kwargs)


if __name__ == "__main__":
    import reading
    from pathlib import Path

    field_line_set = reading.read_3d_field_line_set_from_combined_pickles(Path(reading.data_path, 'field_line_set.pickle'))

    import matplotlib.pyplot as plt
    import plotting
    fig, ax = plotting.create_3d_plot()
    plotting.set_3d_plot_extent(ax, *grid_bounds)
    ax.invert_zaxis()
    ax.set_axis_off()

    field_line_set.add_to_plot_as_line(ax, lw=0.2, alpha=0.1)

    plotting.render(fig)