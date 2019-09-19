import numpy as np
import functools
from fields import Coords3
import plotting


class ElectronBeam:
    def __init__(self, trajectory, fixed_scalar_values, fixed_vector_values, varying_scalar_values, varying_vector_values):
        assert isinstance(trajectory, Coords3)
        assert isinstance(fixed_scalar_values, dict)
        assert isinstance(fixed_vector_values, dict)
        assert isinstance(varying_scalar_values, dict)
        assert isinstance(varying_vector_values, dict)
        self.trajectory = trajectory
        self.fixed_scalar_values = fixed_scalar_values
        self.fixed_vector_values = fixed_vector_values
        self.varying_scalar_values = varying_scalar_values
        self.varying_vector_values = varying_vector_values

        try:
            self.fixed_scalar_values['underestimated_total_propagation_distance'] = self.fixed_scalar_values['total_propagation_distance'] if self.fixed_scalar_values['total_propagation_distance'] > self.fixed_scalar_values['estimated_depletion_distance'] else np.nan
        except KeyError:
            pass

        try:
            self.varying_scalar_values['remaining_power_density'] = self.fixed_scalar_values['total_power_density'] - np.cumsum(self.varying_scalar_values['deposited_power_density'])
        except KeyError:
            pass

    def add_to_3d_plot_as_scatter(self, ax, value_name, log=False, vmin=None, vmax=None, cmap_name='viridis', s=1.0, marker='o', edgecolors='none', depthshade=False, alpha=1.0, relative_alpha=True):
        values = self.get_varying_scalar_values(value_name)
        colors = plotting.colors_from_values(values, log=log, vmin=vmin, vmax=vmax, cmap_name=cmap_name, alpha=alpha, relative_alpha=relative_alpha)
        ax.scatter(self.trajectory.x, self.trajectory.y, self.trajectory.z, c=colors, s=s, marker=marker, edgecolors=edgecolors, depthshade=depthshade)

    def add_to_3d_plot_as_lines(self, ax, value_name, log=False, vmin=None, vmax=None, cmap_name='viridis', lw=1.0, alpha=1.0, relative_alpha=True):
        values = self.get_varying_scalar_values(value_name)
        colors = plotting.colors_from_values(values, log=log, vmin=vmin, vmax=vmax, cmap_name=cmap_name, alpha=alpha, relative_alpha=relative_alpha)
        plotting.add_3d_line_collection(ax, self.trajectory.x, self.trajectory.y, self.trajectory.z, colors, lw=lw)

    def add_to_3d_plot_as_single_color_scatter(self, ax, c='k', s=0.02, marker='.', edgecolors='none', depthshade=False, alpha=1.0):
        ax.scatter(self.trajectory.x, self.trajectory.y, self.trajectory.z, c=c, s=s, marker=marker, edgecolors=edgecolors, depthshade=depthshade, alpha=alpha)

    def add_to_3d_plot_as_single_color_line(self, ax, c='k', lw=1.0, alpha=1.0):
        for pos_x, pos_y, pos_z in zip(*self.__find_nonwrapping_segments()):
            ax.plot(pos_x, pos_y, pos_z, c=c, lw=lw, alpha=alpha)

    def has_fixed_scalar_values(self, value_name):
        return value_name in self.fixed_scalar_values

    def has_varying_scalar_values(self, value_name):
        return value_name in self.varying_scalar_values

    def get_fixed_scalar_value(self, value_name):
        return self.fixed_scalar_values[value_name]

    def get_varying_scalar_values(self, value_name):
        return self.varying_scalar_values[value_name]

    def get_varying_scalar_value_limits(self, value_name):
        values = self.get_varying_scalar_values(value_name)
        min_value = np.nanmin(values)
        max_value = np.nanmax(values)
        return min_value, max_value

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

    def add_fixed_values_to_3d_plot(self, ax, value_name, log=False, vmin=None, vmax=None, cmap_name='viridis', s=2, marker='o', edgecolors='none', depthshade=False, alpha=1.0, relative_alpha=True):
        acceleration_positions = self.get_acceleration_positions()
        values = self.get_fixed_scalar_values(value_name)
        colors = plotting.colors_from_values(values, log=log, vmin=vmin, vmax=vmax, cmap_name=cmap_name, alpha=alpha, relative_alpha=relative_alpha)
        ax.scatter(acceleration_positions.x, acceleration_positions.y, acceleration_positions.z, c=colors, s=s, marker=marker, edgecolors=edgecolors, depthshade=depthshade)
        return plotting.get_normalizer(np.nanmin(values) if vmin is None else vmin,
                                       np.nanmax(values) if vmax is None else vmax, log=log), \
               plotting.get_cmap(cmap_name)


    def add_fixed_values_to_histogram(self, ax, value_name, vmin=None, vmax=None, c='k', lw=1.0):
        values = self.get_fixed_scalar_values(value_name)
        vmin = np.nanmin(values) if vmin is None else vmin
        vmax = np.nanmax(values) if vmax is None else vmax
        hist, bin_edges = np.histogram(values, bins='auto', range=(vmin, vmax))
        ax.step(bin_edges[:-1], hist, where='pre', c=c, lw=lw)


    def add_varying_values_to_3d_plot(self, ax, value_name, log=False, vmin=None, vmax=None, cmap_name='viridis', scatter=True, **kwargs):
        if vmin is None or vmax is None:
            min_value, max_value = self.get_varying_scalar_value_limits(value_name)
            if vmin is None:
                vmin = min_value
            if vmax is None:
                vmax = max_value

        if scatter:
            for beam in self.beams:
                beam.add_to_3d_plot_as_scatter(ax, value_name, vmin=vmin, vmax=vmax, **kwargs)
        else:
            for beam in self.beams:
                beam.add_to_3d_plot_as_lines(ax, value_name, vmin=vmin, vmax=vmax, **kwargs)

        return plotting.get_normalizer(vmin, vmax, log=log), plotting.get_cmap(cmap_name)

    def add_to_3d_plot_with_single_color(self, ax, scatter=False, **kwargs):
        if scatter:
            for beam in self.beams:
                beam.add_to_3d_plot_as_single_color_scatter(ax, **kwargs)
        else:
            for beam in self.beams:
                beam.add_to_3d_plot_as_single_color_line(ax, **kwargs)

    def get_number_of_beams(self):
        return len(self.beams)

    def get_acceleration_positions(self):
        x_coordinates = [beam.trajectory.x[0] for beam in self.beams]
        y_coordinates = [beam.trajectory.y[0] for beam in self.beams]
        z_coordinates = [beam.trajectory.z[0] for beam in self.beams]
        return Coords3(x_coordinates, y_coordinates, z_coordinates)

    def has_fixed_scalar_values(self, value_name):
        return np.all([beam.has_fixed_scalar_values(value_name) for beam in self.beams])

    def has_varying_scalar_values(self, value_name):
        return np.all([beam.has_varying_scalar_values(value_name) for beam in self.beams])

    def get_fixed_scalar_values(self, value_name):
        return np.asfarray([beam.get_fixed_scalar_value(value_name) for beam in self.beams])

    def get_fixed_scalar_value_limits(self, value_name):
        values = np.asfarray([beam.get_fixed_scalar_value(value_name) for beam in self.beams])
        min_value = np.nanmin(values)
        max_value = np.nanmax(values)
        return min_value, max_value

    def get_varying_scalar_value_limits(self, value_name):
        min_values, max_values = tuple(zip(*[beam.get_varying_scalar_value_limits(value_name) for beam in self.beams]))
        min_value = np.nanmin(min_values)
        max_value = np.nanmax(max_values)
        return min_value, max_value


class DistributionRejectionMap:
    def __init__(self, positions, rejection_causes, rejection_codes):
        assert isinstance(positions, Coords3)
        assert rejection_causes.shape == positions.x.shape
        assert rejection_causes.dtype == np.ubyte
        self.positions = positions
        self.rejection_causes = rejection_causes
        self._process_rejection_codes(rejection_codes)

    def add_to_3d_plot_as_scatter(self, ax, included_codes=[], excluded_codes=[], limited_number=None, s=1.0, marker='o', edgecolors='none', depthshade=False, alpha=1.0):
        included_codes = list(map(int, included_codes))
        excluded_codes = list(map(int, excluded_codes))
        assert set(included_codes).issubset(self.valid_code_set)
        assert set(excluded_codes).issubset(self.valid_code_set)
        assert len(set(included_codes).intersection(set(excluded_codes))) == 0

        inclusion_bitflag = self._create_filtering_bitflag(included_codes) if len(included_codes) > 0 else 1 << 8
        exclusion_bitflag = self._create_filtering_bitflag(excluded_codes)

        inclusion_mask = np.logical_and(self.rejection_causes & inclusion_bitflag != 0, self.rejection_causes & exclusion_bitflag == 0)
        rejection_causes = self.rejection_causes[inclusion_mask]
        x_coordinates = self.positions.x[inclusion_mask]
        y_coordinates = self.positions.y[inclusion_mask]
        z_coordinates = self.positions.z[inclusion_mask]

        print('Conditions satisfied by {:d}/{:d} points ({:g} %)'\
              .format(rejection_causes.size, self.rejection_causes.size, (100.0*rejection_causes.size)/self.rejection_causes.size))

        indices = slice(None) if limited_number is None else np.random.shuffle(np.arange(rejection_causes.size))[:min(rejection_causes.size, int(limited_number))]
        rejection_causes = rejection_causes[indices]
        x_coordinates = x_coordinates[indices]
        y_coordinates = y_coordinates[indices]
        z_coordinates = z_coordinates[indices]

        colors = [self.combination_colors[idx] for idx in rejection_causes]
        ax.scatter(x_coordinates, y_coordinates, z_coordinates,
                   c=colors, s=s, marker=marker, edgecolors=edgecolors, depthshade=depthshade, alpha=alpha)

        self._add_labels(ax, inclusion_bitflag, exclusion_bitflag, marker=marker)

    def get_cause_code_legend_text(self):
        return '\n'.join(['{:d}: {}'.format(code, name) for code, name in zip(self.codes, self.code_names)])

    def _create_filtering_bitflag(self, codes):
        bitflag = functools.reduce(lambda a, b: a | b, [1 << (code - 1) for code in codes], 0)
        return bitflag

    def _add_labels(self, ax, inclusion_bitflag, exclusion_bitflag, marker='.'):
        for index in self.combination_indices:
            if index & inclusion_bitflag != 0 and index & exclusion_bitflag == 0:
                ax.plot([], [], linestyle='none', marker=marker, color=self.combination_colors[index], label=self.combination_labels[index])

    def _process_rejection_codes(self, rejection_codes):
        self.code_names, code_bitflags = tuple(zip(*list(rejection_codes.items())))
        self.codes = range(len(code_bitflags))
        number_of_combinations = functools.reduce(lambda a, b: a | b, code_bitflags) + 1
        self.valid_code_set = set(range(1, len(self.codes)))
        self.combination_indices = range(number_of_combinations)
        self.combination_colors = plotting.get_default_colors()[:number_of_combinations]
        combinations = []
        for index in self.combination_indices:
            combinations.append([])
            for n in range(len(code_bitflags)):
                if (index & code_bitflags[n]) != 0:
                    combinations[-1].append(str(n))
        self.combination_labels = [' & '.join(combination) if len(combination) > 0 else '0' for combination in combinations]


def plot_electron_beams(electron_beam_swarm, value_name=None, value_description=None, title=None, hide_grid=False, output_path=None, **kwargs):

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
        electron_beam_swarm.add_to_3d_plot_with_single_color(ax, **kwargs)
    else:

        if electron_beam_swarm.has_fixed_scalar_values(value_name):
            norm, cmap = electron_beam_swarm.add_fixed_values_to_3d_plot(ax, value_name, **kwargs)
        else:
            norm, cmap = electron_beam_swarm.add_varying_values_to_3d_plot(ax, value_name, **kwargs)

        plotting.add_3d_colorbar(fig, ax, norm, cmap,
                                 label=value_name if value_description is None else value_description)

    ax.set_title(title)

    plotting.render(fig, output_path=output_path)


def plot_fixed_beam_value_histogram(electron_beam_swarm, value_name=None, xlog=False, ylog=False, value_description=None, title=None, output_path=None, **kwargs):

    fig, ax = plotting.create_2d_subplots()

    electron_beam_swarm.add_fixed_values_to_histogram(ax, value_name, **kwargs)

    if xlog:
        ax.set_xscale('log')
    if ylog:
        ax.set_yscale('log')
    ax.set_xlabel(value_name if value_description is None else value_description)
    ax.set_ylabel('Number of values')
    ax.set_title(title)

    plotting.render(fig, output_path=output_path)


def plot_rejection_map(rejection_map, hide_grid=False, output_path=None, **kwargs):

    grid_bounds = ((-0.015625, 23.98438),
                   (-0.015625, 23.98438),
                   (-14.33274, 2.525689))

    fig, ax = plotting.create_3d_plot()

    plotting.set_3d_plot_extent(ax, *grid_bounds)
    plotting.set_3d_spatial_axis_labels(ax, unit='Mm')
    ax.invert_zaxis()
    if hide_grid:
        ax.set_axis_off()

    rejection_map.add_to_3d_plot_as_scatter(ax, **kwargs)
    ax.legend(loc='best')
    plotting.add_textbox(ax, rejection_map.get_cause_code_legend_text(), 2)

    plotting.render(fig, tight_layout=False, output_path=output_path)


if __name__ == "__main__":
    import reading
    from pathlib import Path
    electron_beam_swarm = reading.read_electron_beam_swarm_from_combined_pickles(Path(reading.data_path, 'phd_run', 'en024031_emer3.0str_coarse_ebeam_351_beams.pickle'))
    #plot_electron_beams(electron_beam_swarm, value_name='deposited_power_density', vmin=1e-10, vmax=1e-4, log=True, alpha=1.0)
    #plot_electron_beams(electron_beam_swarm, value_name='remaining_power_density', vmax=1, log=True, alpha=1.0)
    #plot_electron_beams(electron_beam_swarm, value_name='total_propagation_distance', log=True, alpha=1.0)
    #plot_electron_beams(electron_beam_swarm, value_name='estimated_depletion_distance', log=True, alpha=1.0)
    #plot_electron_beams(electron_beam_swarm, value_name='underestimated_total_propagation_distance', log=True, alpha=1.0, relative_alpha=False)
    plot_fixed_beam_value_histogram(electron_beam_swarm, value_name='krec', xlog=True, ylog=True)

    #rejection_map = reading.read_electron_distribution_rejection_map(Path('data', 'phd_run', 'en024031_emer3.0str_coarse_ebeam_351_rejection_map.pickle'))
    #plot_rejection_map(rejection_map, included_codes=[3], excluded_codes=[2], alpha=0.01)
