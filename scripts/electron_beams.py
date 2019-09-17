import numpy as np
from fields import Coords3
import plotting


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

    def add_to_plot_as_scatter(self, ax, value_name, log=False, vmin=None, vmax=None, cmap_name='viridis', s=0.02, alpha=1.0, relative_alpha=True):
        values = self.evolving_scalar_values[value_name]
        colors = plotting.colors_from_values(values, log=log, vmin=vmin, vmax=vmax, cmap_name=cmap_name, alpha=alpha, relative_alpha=relative_alpha)
        ax.scatter(self.trajectory.x, self.trajectory.y, self.trajectory.z, c=colors, s=s)

    def add_to_plot_as_lines(self, ax, value_name, log=False, vmin=None, vmax=None, cmap_name='viridis', lw=1.0, alpha=1.0, relative_alpha=True):
        values = self.evolving_scalar_values[value_name]
        colors = plotting.colors_from_values(values, log=log, vmin=vmin, vmax=vmax, cmap_name=cmap_name, alpha=alpha, relative_alpha=relative_alpha)
        plotting.add_3d_line_collection(ax, self.trajectory.x, self.trajectory.y, self.trajectory.z, colors, lw=lw)

    def add_to_plot_as_single_color_scatter(self, ax, c='k', s=0.02, alpha=1.0):
        ax.scatter(self.trajectory.x, self.trajectory.y, self.trajectory.z, c=c, s=s, alpha=alpha)

    def add_to_plot_as_single_color_line(self, ax, c='k', lw=1.0, alpha=1.0):
        for pos_x, pos_y, pos_z in zip(*self.__find_nonwrapping_segments()):
            ax.plot(pos_x, pos_y, pos_z, c=c, lw=lw, alpha=alpha)

    def get_evolving_scalar_value_limits(self, value_name):
        values = self.evolving_scalar_values[value_name]
        min_value = np.nanmin(values)
        max_value = np.nanmax(values)
        return min_value, max_value

    def has_initial_scalar_values(self, value_name):
        return value_name in self.initial_scalar_values

    def has_evolving_scalar_values(self, value_name):
        return value_name in self.evolving_scalar_values

    def __find_nonwrapping_segments(self, threshold=20.0):
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

    def add_initial_values_to_plot(self, ax, value_name, log=False, vmin=None, vmax=None, cmap_name='viridis', s=0.5, alpha=1.0, relative_alpha=True):
        acceleration_positions = self.get_acceleration_positions()
        values = self.get_initial_scalar_values(value_name)
        colors = plotting.colors_from_values(values, log=log, vmin=vmin, vmax=vmax, cmap_name=cmap_name, alpha=alpha, relative_alpha=relative_alpha)
        ax.scatter(acceleration_positions.x, acceleration_positions.y, acceleration_positions.z, c=colors, s=s)
        return plotting.get_normalizer(np.nanmin(values) if vmin is None else vmin,
                                       np.nanmax(values) if vmax is None else vmax, log=log), \
               plotting.get_cmap(cmap_name)


    def add_evolving_values_to_plot(self, ax, value_name, log=False, vmin=None, vmax=None, cmap_name='viridis', scatter=True, **kwargs):
        if vmin is None or vmax is None:
            min_value, max_value = self.get_evolving_scalar_value_limits(value_name)
            if vmin is None:
                vmin = min_value
            if vmax is None:
                vmax = max_value

        if scatter:
            for beam in self.beams:
                beam.add_to_plot_as_scatter(ax, value_name, vmin=vmin, vmax=vmax, **kwargs)
        else:
            for beam in self.beams:
                beam.add_to_plot_as_lines(ax, value_name, vmin=vmin, vmax=vmax, **kwargs)

        return plotting.get_normalizer(vmin, vmax, log=log), plotting.get_cmap(cmap_name)

    def add_to_plot_with_single_color(self, ax, scatter=False, **kwargs):
        if scatter:
            for beam in self.beams:
                beam.add_to_plot_as_single_color_scatter(ax, **kwargs)
        else:
            for beam in self.beams:
                beam.add_to_plot_as_single_color_line(ax, **kwargs)

    def get_number_of_beams(self):
        return len(self.beams)

    def get_acceleration_positions(self):
        x_coordinates = [beam.trajectory.x[0] for beam in self.beams]
        y_coordinates = [beam.trajectory.y[0] for beam in self.beams]
        z_coordinates = [beam.trajectory.z[0] for beam in self.beams]
        return Coords3(x_coordinates, y_coordinates, z_coordinates)

    def get_initial_scalar_values(self, value_name):
        return np.asfarray([beam.initial_scalar_values[value_name] for beam in self.beams])

    def get_initial_scalar_value_limits(self, value_name):
        values = np.asfarray([beam.initial_scalar_values[value_name] for beam in self.beams])
        min_value = np.nanmin(values)
        max_value = np.nanmax(values)
        return min_value, max_value

    def get_evolving_scalar_value_limits(self, value_name):
        min_values, max_values = tuple(zip(*[beam.get_evolving_scalar_value_limits(value_name) for beam in self.beams]))
        min_value = np.nanmin(min_values)
        max_value = np.nanmax(max_values)
        return min_value, max_value

    def has_initial_scalar_values(self, value_name):
        return np.all([beam.has_initial_scalar_values(value_name) for beam in self.beams])

    def has_evolving_scalar_values(self, value_name):
        return np.all([beam.has_evolving_scalar_values(value_name) for beam in self.beams])


def plot_electron_beams(electron_beam_swarm, value_name=None, value_description=None, hide_grid=False, output_path=None, **kwargs):

    grid_bounds = ((-0.015625, 23.98438),
                   (-0.015625, 23.98438),
                   (-14.33274, 2.525689))

    fig, ax = plotting.create_3d_plot()

    plotting.set_3d_plot_extent(ax, *grid_bounds)
    plotting.set_3d_spatial_axis_labels(ax, unit='Mm')
    ax.invert_zaxis()
    if hide_grid:
        ax.set_axis_off()

    if value_name is None:
        electron_beam_swarm.add_to_plot_with_single_color(ax, **kwargs)
    else:

        if electron_beam_swarm.has_initial_scalar_values(value_name):
            norm, cmap = electron_beam_swarm.add_initial_values_to_plot(ax, value_name, **kwargs)
        else:
            norm, cmap = electron_beam_swarm.add_evolving_values_to_plot(ax, value_name, **kwargs)

        plotting.add_3d_colorbar(fig, ax, norm, cmap,
                                 label=value_name if value_description is None else value_description)

    plotting.render(fig, output_path=output_path)


if __name__ == "__main__":
    import reading
    from pathlib import Path
    electron_beam_swarm = reading.read_electron_beam_swarm_from_combined_pickles(Path(reading.data_path, 'phd_run', 'en024031_emer3.0str_ebeam_351_beams.pickle'))
    #plot_electron_beams(electron_beam_swarm, value_name='qbeam', vmin=1e-10, vmax=1e-4, log=True, alpha=1.0)
    plot_electron_beams(electron_beam_swarm, value_name='lower_cutoff_energy', log=True, alpha=1.0)
