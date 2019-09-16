import numpy as np
from fields import Coords3


class ElectronBeam:
    def __init__(self, trajectory, initial_scalar_values, initial_vector_values, evolving_scalar_values, evolving_vector_values):
        assert isinstance(trajectory, Coords3)
        assert isinstance(initial_scalar_values, dict)
        assert isinstance(initial_vector_values, dict)
        assert isinstance(evolving_scalar_values, dict)
        assert isinstance(evolving_vector_values, dict)
        self.trajectory = trajectory
        self.initial_scalar_values = initial_scalar_values
        self.initial_vector_values = initial_vector_values
        self.evolving_scalar_values = evolving_scalar_values
        self.evolving_vector_values = evolving_vector_values

    def add_to_plot_as_scatter(self, ax, c='k', s=1.0):
        ax.scatter(self.trajectory.x, self.trajectory.y, self.trajectory.z, c=c, s=s)

    def add_to_plot_as_line(self, ax, c='k', lw=1.0, alpha=1.0):
        for pos_x, pos_y, pos_z in zip(*self.find_nonwrapping_segments()):
            ax.plot(pos_x, pos_y, pos_z, c=c, lw=lw, alpha=alpha)

    def find_nonwrapping_segments(self, threshold=20.0):
        step_lengths = np.sqrt(np.diff(self.trajectory.x)**2 + np.diff(self.trajectory.y)**2 + np.diff(self.trajectory.z)**2)
        wrap_indices = np.where(step_lengths > threshold*np.mean(step_lengths))[0]
        if wrap_indices.size > 0:
            wrap_indices += 1
            return np.split(self.trajectory.x, wrap_indices), \
                   np.split(self.trajectory.y, wrap_indices), \
                   np.split(self.trajectory.z, wrap_indices)
        else:
            return [self.trajectory.x], [self.trajectory.y], [self.trajectory.z]


class ElectronBeamSwarm:
    def __init__(self, beams):
        assert isinstance(beams, list)
        self.beams = beams

    def insert(self, beam):
        assert isinstance(beam, ElectronBeam)
        self.beams.append(beam)

    def add_to_plot_as_scatter(self, ax, **kwargs):
        for beam in self.beams:
            beam.add_to_plot_as_scatter(ax, **kwargs)

    def add_to_plot_as_line(self, ax, **kwargs):
        for beam in self.beams:
            beam.add_to_plot_as_line(ax, **kwargs)


if __name__ == "__main__":
    import reading
    from pathlib import Path

    grid_bounds = ((-0.015625, 23.98438),
                   (-0.015625, 23.98438),
                   (-14.33274, 2.525689))

    electron_beam_swarm = reading.read_electron_beam_swarm_from_combined_pickles(Path(reading.data_path, 'electron_beam_swarm.pickle'))

    import matplotlib.pyplot as plt
    import plotting
    fig, ax = plotting.create_3d_plot()
    plotting.set_3d_plot_extent(ax, *grid_bounds)
    ax.invert_zaxis()
    ax.set_axis_off()

    electron_beam_swarm.add_to_plot_as_line(ax, lw=0.2, alpha=0.1)

    plotting.render(fig)
