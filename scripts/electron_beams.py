import functools
import numpy as np
from fields import Coords3
import plotting


class ElectronBeamSwarm:
    def __init__(self,
                 number_of_beams,
                 fixed_scalar_values,
                 fixed_vector_values,
                 varying_scalar_values,
                 varying_vector_values,
                 metadata,
                 derived_quantities=[]):
        assert isinstance(number_of_beams, int)
        assert isinstance(fixed_scalar_values, dict)
        assert isinstance(fixed_vector_values, dict)
        assert isinstance(varying_scalar_values, dict)
        assert isinstance(varying_vector_values, dict)
        assert isinstance(metadata, dict)
        self.number_of_beams = number_of_beams
        self.fixed_scalar_values = fixed_scalar_values
        self.fixed_vector_values = fixed_vector_values
        self.varying_scalar_values = varying_scalar_values
        self.varying_vector_values = varying_vector_values
        self.metadata = metadata
        self.__derive_quantities(derived_quantities)

    def add_fixed_values_to_3d_plot(self,
                                    ax,
                                    value_name,
                                    log=False,
                                    vmin=None,
                                    vmax=None,
                                    cmap_name='viridis',
                                    s=2,
                                    marker='o',
                                    edgecolors='none',
                                    depthshade=False,
                                    alpha=1.0,
                                    relative_alpha=True):

        values = self.get_fixed_scalar_values(value_name)
        colors = plotting.colors_from_values(values,
                                             log=log,
                                             vmin=vmin,
                                             vmax=vmax,
                                             cmap_name=cmap_name,
                                             alpha=alpha,
                                             relative_alpha=relative_alpha)

        ax.scatter(self.get_fixed_scalar_values('x0'),
                   self.get_fixed_scalar_values('y0'),
                   self.get_fixed_scalar_values('z0'),
                   c=colors,
                   s=s,
                   marker=marker,
                   edgecolors=edgecolors,
                   depthshade=depthshade)

        return plotting.get_normalizer(np.nanmin(values) if vmin is None else vmin,
                                       np.nanmax(values) if vmax is None else vmax, log=log), \
            plotting.get_cmap(cmap_name)

    def add_fixed_values_as_line_histogram(self,
                                           ax,
                                           value_name,
                                           vmin=None,
                                           vmax=None,
                                           c='k',
                                           lw=1.0):
        values = self.get_fixed_scalar_values(value_name)

        vmin = np.nanmin(values) if vmin is None else vmin
        vmax = np.nanmax(values) if vmax is None else vmax

        hist, bin_edges = np.histogram(values, bins='auto', range=(vmin, vmax))

        ax.step(bin_edges[:-1], hist, where='pre', c=c, lw=lw)

    def add_fixed_values_as_2d_histogram_image(self,
                                               ax,
                                               value_name_x,
                                               value_name_y,
                                               bins_x=20,
                                               bins_y=20,
                                               log_x=False,
                                               log_y=False,
                                               vmin_x=None,
                                               vmax_x=None,
                                               vmin_y=None,
                                               vmax_y=None,
                                               log=False,
                                               vmin=None,
                                               vmax=None,
                                               cmap_name='viridis'):

        values_x = self.get_fixed_scalar_values(value_name_x)
        values_y = self.get_fixed_scalar_values(value_name_y)

        vmin_x = np.nanmin(values_x) if vmin_x is None else vmin_x
        vmax_x = np.nanmax(values_x) if vmax_x is None else vmax_x
        vmin_y = np.nanmin(values_y) if vmin_y is None else vmin_y
        vmax_y = np.nanmax(values_y) if vmax_y is None else vmax_y

        if log_x:
            values_x = np.log10(values_x)
            vmin_x = np.log10(vmin_x)
            vmax_x = np.log10(vmax_x)
        if log_y:
            values_y = np.log10(values_y)
            vmin_y = np.log10(vmin_y)
            vmax_y = np.log10(vmax_y)

        hist, bin_edges_x, bin_edges_y = np.histogram2d(
            values_x,
            values_y,
            bins=[bins_x, bins_y],
            range=[[vmin_x, vmax_x], [vmin_y, vmax_y]])

        return ax.pcolormesh(*np.meshgrid(bin_edges_x, bin_edges_y),
                             hist,
                             norm=plotting.get_normalizer(vmin, vmax, log=log),
                             vmin=vmin,
                             vmax=vmax,
                             cmap=plotting.get_cmap(cmap_name))

    def add_varying_values_to_3d_plot(self,
                                      ax,
                                      value_name,
                                      log=False,
                                      vmin=None,
                                      vmax=None,
                                      cmap_name='viridis',
                                      s=1.0,
                                      marker='o',
                                      edgecolors='none',
                                      depthshade=False,
                                      alpha=1.0,
                                      relative_alpha=True):

        x, y, z = self.get_concatenated_trajectories()
        values = self.get_concatenated_varying_scalar_values(value_name)

        if vmin is None or vmax is None:
            if vmin is None:
                vmin = np.nanmin(values)
            if vmax is None:
                vmax = np.nanmax(values)

        c = plotting.colors_from_values(values,
                                        log=log,
                                        vmin=vmin,
                                        vmax=vmax,
                                        cmap_name=cmap_name,
                                        alpha=alpha,
                                        relative_alpha=relative_alpha)

        ax.scatter(x,
                   y,
                   z,
                   c=c,
                   s=s,
                   marker=marker,
                   edgecolors=edgecolors,
                   depthshade=depthshade)

        return plotting.get_normalizer(vmin, vmax,
                                       log=log), plotting.get_cmap(cmap_name)

    def add_to_3d_plot_with_single_color(self,
                                         ax,
                                         scatter=False,
                                         c='k',
                                         lw=1.0,
                                         s=0.02,
                                         marker='.',
                                         edgecolors='none',
                                         depthshade=False,
                                         alpha=1.0):
        if scatter:
            x, y, z = self.get_concatenated_trajectories()
            ax.scatter(x,
                       y,
                       z,
                       c=c,
                       s=s,
                       marker=marker,
                       edgecolors=edgecolors,
                       depthshade=depthshade,
                       alpha=alpha)
        else:
            trajectories_x = self.get_varying_scalar_values('x')
            trajectories_y = self.get_varying_scalar_values('y')
            trajectories_z = self.get_varying_scalar_values('z')
            for beam_idx in range(self.get_number_of_beams()):
                for x, y, z in zip(*self.__find_nonwrapping_segments(
                        trajectories_x[beam_idx], trajectories_y[beam_idx],
                        trajectories_z[beam_idx])):
                    ax.plot(x, y, z, c=c, lw=lw, alpha=alpha)

    def add_rejection_cause_codes_to_3d_plot_as_scatter(self, *args, **kwargs):
        rejection_map = DistributionRejectionMap(
            self.get_rejection_cause_codes())
        rejection_map.add_to_3d_plot_as_scatter(
            self.get_fixed_scalar_values('x0'),
            self.get_fixed_scalar_values('y0'),
            self.get_fixed_scalar_values('z0'), *args, **kwargs)

    def add_rejected_fixed_values_as_line_histograms(self, value_name, *args,
                                                     **kwargs):
        rejection_map = DistributionRejectionMap(
            self.get_rejection_cause_codes())
        rejection_map.add_fixed_values_as_line_histograms(
            self.get_fixed_scalar_values(value_name), *args, **kwargs)

    def has_fixed_scalar_values(self, value_name):
        return value_name in self.fixed_scalar_values

    def has_varying_scalar_values(self, value_name):
        return value_name in self.varying_scalar_values

    def get_number_of_beams(self):
        return self.number_of_beams

    def get_initial_positions(self):
        return Coords3(self.fixed_scalar_values['x0'],
                       self.fixed_scalar_values['y0'],
                       self.fixed_scalar_values['z0'])

    def get_concatenated_trajectories(self):
        return np.concatenate(
            self.get_varying_scalar_values('x')), np.concatenate(
                self.get_varying_scalar_values('y')), np.concatenate(
                    self.get_varying_scalar_values('z'))

    def get_fixed_scalar_values(self, value_name):
        return self.fixed_scalar_values[value_name]

    def get_fixed_vector_values(self, value_name):
        return self.fixed_vector_values[value_name]

    def get_varying_scalar_values(self, value_name):
        return self.varying_scalar_values[value_name]

    def get_concatenated_varying_scalar_values(self, value_name):
        return np.concatenate(self.varying_scalar_values[value_name])

    def get_varying_vector_values(self, value_name):
        return self.varying_vector_values[value_name]

    def get_metadata(self, metadata_label):
        return self.metadata[metadata_label]

    def get_rejection_cause_codes(self):
        return self.get_metadata('rejection_cause_code')

    def __derive_quantities(self, derived_quantities):

        if 'underestimated_total_propagation_distance' in derived_quantities:
            self.fixed_scalar_values['underestimated_total_propagation_distance'] = \
                np.where(self.fixed_scalar_values['total_propagation_distance'] > self.fixed_scalar_values['estimated_depletion_distance'], self.fixed_scalar_values['total_propagation_distance'], np.nan)

        if 'remaining_power_density' in derived_quantities:
            self.varying_scalar_values['remaining_power_density'] = [self.fixed_scalar_values['total_power_density'][i] - \
                np.cumsum(
                    self.varying_scalar_values['deposited_power_density'][i]) for i in range(self.get_number_of_beams())]

    def __find_nonwrapping_segments(self,
                                    trajectory_x,
                                    trajectory_y,
                                    trajectory_z,
                                    threshold=20.0):
        step_lengths = np.sqrt(
            np.diff(trajectory_x)**2 + np.diff(trajectory_y)**2 +
            np.diff(trajectory_z)**2)
        wrap_indices = np.where(
            step_lengths > threshold * np.mean(step_lengths))[0]
        if wrap_indices.size > 0:
            wrap_indices += 1
            return np.split(trajectory_x, wrap_indices), \
                np.split(trajectory_y, wrap_indices), \
                np.split(trajectory_z, wrap_indices)
        else:
            return [trajectory_x], [trajectory_y], [trajectory_z]


class DistributionRejectionMap:
    def __init__(self, rejection_cause_codes):
        assert rejection_cause_codes.dtype == np.ubyte
        self.rejection_cause_codes = rejection_cause_codes
        self.__init_rejectors()

    def add_to_3d_plot_as_scatter(self,
                                  x0,
                                  y0,
                                  z0,
                                  ax,
                                  included_rejectors=[1, 2, 3],
                                  excluded_rejectors=[],
                                  limited_number=None,
                                  s=1.0,
                                  marker='o',
                                  edgecolors='none',
                                  depthshade=False,
                                  alpha=1.0,
                                  textbox_loc=2):

        inclusion_bitflag, exclusion_bitflag = self.__create_bitflags(
            included_rejectors, excluded_rejectors)

        inclusion_mask = self.__create_inclusion_mask(inclusion_bitflag,
                                                      exclusion_bitflag)

        rejection_cause_codes = self.rejection_cause_codes[inclusion_mask]
        x0 = x0[inclusion_mask]
        y0 = y0[inclusion_mask]
        z0 = z0[inclusion_mask]

        indices = self.__sample_limited_indices(rejection_cause_codes.size,
                                                limited_number)
        rejection_cause_codes = rejection_cause_codes[indices]
        x0 = x0[indices]
        y0 = y0[indices]
        z0 = z0[indices]

        colors = [
            self.rejection_code_colors[code - 1]
            for code in rejection_cause_codes
        ]
        ax.scatter(x0,
                   y0,
                   z0,
                   c=colors,
                   s=s,
                   marker=marker,
                   edgecolors=edgecolors,
                   depthshade=depthshade,
                   alpha=alpha)

        self.__add_labels_for_scatter(ax,
                                      inclusion_bitflag,
                                      exclusion_bitflag,
                                      marker=marker)

        plotting.add_textbox(ax, self.__get_rejector_legend_text(),
                             textbox_loc)

    def add_fixed_values_as_line_histograms(self,
                                            values,
                                            ax,
                                            included_rejectors=[1, 2, 3],
                                            excluded_rejectors=[],
                                            limited_number=None,
                                            vmin=None,
                                            vmax=None,
                                            lw=1.0,
                                            textbox_loc=2):

        inclusion_bitflag, exclusion_bitflag = self.__create_bitflags(
            included_rejectors, excluded_rejectors)

        inclusion_mask = self.__create_inclusion_mask(inclusion_bitflag,
                                                      exclusion_bitflag)

        rejection_cause_codes = self.rejection_cause_codes[inclusion_mask]
        values = values[inclusion_mask]

        indices = self.__sample_limited_indices(rejection_cause_codes.size,
                                                limited_number)
        rejection_cause_codes = rejection_cause_codes[indices]
        values = values[indices]

        vmin = np.nanmin(values) if vmin is None else vmin
        vmax = np.nanmax(values) if vmax is None else vmax

        for code in self.__get_included_rejection_cause_codes(
                inclusion_bitflag, exclusion_bitflag):
            values_for_code = values[rejection_cause_codes == code]

            hist, bin_edges = np.histogram(values_for_code,
                                           bins='auto',
                                           range=(vmin, vmax))

            ax.step(bin_edges[:-1],
                    hist,
                    where='pre',
                    lw=lw,
                    c=self.rejection_code_colors[code - 1],
                    label=self.rejection_code_labels[code - 1])

        ax.legend(loc='best')

        plotting.add_textbox(ax, self.__get_rejector_legend_text(),
                             textbox_loc)

    def __init_rejectors(self):
        self.rejector_names = [
            'Too low total power density', 'Too short depletion distance',
            'Too perpendicular direction'
        ]
        number_of_rejectors = len(self.rejector_names)
        number_of_valid_codes = (1 << number_of_rejectors) - 1
        self.rejector_identifiers = range(1, number_of_rejectors + 1)
        self.valid_rejection_cause_codes = range(1, number_of_valid_codes + 1)

        rejector_bitflags = [1 << n for n in range(number_of_rejectors)]
        combinations = []
        for code in self.valid_rejection_cause_codes:
            combinations.append([])
            for identifier in self.rejector_identifiers:
                if (code & rejector_bitflags[identifier - 1]) != 0:
                    combinations[-1].append(str(identifier))

        self.rejection_code_labels = [
            ' & '.join(combination) for combination in combinations
        ]
        self.rejection_code_colors = plotting.get_default_colors(
        )[:number_of_valid_codes]

    def __create_bitflags(self, included_rejectors, excluded_rejectors):
        included_rejectors = list(map(int, included_rejectors))
        excluded_rejectors = list(map(int, excluded_rejectors))
        assert set(included_rejectors).issubset(set(self.rejector_identifiers))
        assert set(excluded_rejectors).issubset(set(self.rejector_identifiers))
        assert len(
            set(included_rejectors).intersection(set(excluded_rejectors))) == 0

        inclusion_bitflag = self.__create_filtering_bitflag(included_rejectors)
        exclusion_bitflag = self.__create_filtering_bitflag(excluded_rejectors)

        return inclusion_bitflag, exclusion_bitflag

    def __create_inclusion_mask(self, inclusion_bitflag, exclusion_bitflag):
        inclusion_mask = np.logical_and(
            self.rejection_cause_codes & inclusion_bitflag != 0,
            self.rejection_cause_codes & exclusion_bitflag == 0)
        number_of_included_values = np.sum(inclusion_mask)
        print('Conditions satisfied by {:d}/{:d} points ({:g} %)'.format(
            number_of_included_values, self.rejection_cause_codes.size,
            (100.0 * number_of_included_values) /
            self.rejection_cause_codes.size))
        return inclusion_mask

    def __get_included_rejection_cause_codes(self, inclusion_bitflag,
                                             exclusion_bitflag):
        return list(
            filter(
                lambda code: code & inclusion_bitflag != 0 and code &
                exclusion_bitflag == 0, self.valid_rejection_cause_codes))

    def __sample_limited_indices(self, total_number, limited_number):
        return slice(None) if limited_number is None else \
            np.random.permutation(total_number)[
            :min(total_number, int(limited_number))]

    def __get_rejector_legend_text(self):
        return '\n'.join([
            '{:d}: {}'.format(identifier, name) for identifier, name in zip(
                self.rejector_identifiers, self.rejector_names)
        ])

    def __create_filtering_bitflag(self, rejector_identifiers):
        bitflag = functools.reduce(
            lambda a, b: a | b,
            [1 << (identifier - 1) for identifier in rejector_identifiers], 0)
        return bitflag

    def __add_labels_for_scatter(self,
                                 ax,
                                 inclusion_bitflag,
                                 exclusion_bitflag,
                                 marker='.'):
        for code in self.__get_included_rejection_cause_codes(
                inclusion_bitflag, exclusion_bitflag):
            ax.plot([], [],
                    linestyle='none',
                    marker=marker,
                    color=self.rejection_code_colors[code - 1],
                    label=self.rejection_code_labels[code - 1])
        ax.legend(loc='best')


VALUE_DESCRIPTIONS = {
    'x': 'x [Mm]',
    'y': 'y [Mm]',
    'z': 'z [Mm]',
    'total_power_density': 'Total power density [erg/(cm^3 s)]',
    'lower_cutoff_energy': 'Lower cut-off energy [keV]',
    'estimated_depletion_distance': 'Estimated depletion distance [Mm]',
    'total_propagation_distance': 'Total propagation distance [Mm]',
    'deposited_power_density': 'Deposited power density [erg/(cm^3 s)]',
    'remaining_power_density': 'Remaining power density [erg/(cm^3 s)]',
    'krec': 'Reconnection factor [Bifrost units]'
}


def process_value_description(value_name, value_description):
    if value_description is None:
        if value_name in VALUE_DESCRIPTIONS:
            return VALUE_DESCRIPTIONS[value_name]
        else:
            return value_name
    else:
        return value_description


def plot_electron_beams(electron_beam_swarm,
                        value_name=None,
                        value_description=None,
                        title=None,
                        hide_grid=False,
                        output_path=None,
                        **kwargs):

    grid_bounds = ((-0.015625, 23.98438), (-0.015625, 23.98438), (-14.33274,
                                                                  2.525689))

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
            norm, cmap = electron_beam_swarm.add_fixed_values_to_3d_plot(
                ax, value_name, **kwargs)
        else:
            norm, cmap = electron_beam_swarm.add_varying_values_to_3d_plot(
                ax, value_name, **kwargs)

        plotting.add_3d_colorbar(fig,
                                 norm,
                                 cmap,
                                 label=process_value_description(
                                     value_name, value_description))

    ax.set_title(title)

    plotting.render(fig, output_path=output_path)


def plot_fixed_beam_value_histogram(electron_beam_swarm,
                                    value_name,
                                    xlog=False,
                                    ylog=False,
                                    value_description=None,
                                    title=None,
                                    output_path=None,
                                    **kwargs):

    fig, ax = plotting.create_2d_subplots()

    electron_beam_swarm.add_fixed_values_as_line_histogram(
        ax, value_name, **kwargs)

    if xlog:
        ax.set_xscale('log')
    if ylog:
        ax.set_yscale('log')
    ax.set_xlabel(process_value_description(value_name, value_description))
    ax.set_ylabel('Number of values')
    ax.set_title(title)

    plotting.render(fig, output_path=output_path)


def plot_fixed_beam_value_2d_histogram(electron_beam_swarm,
                                       value_name_x,
                                       value_name_y,
                                       log_x=False,
                                       log_y=False,
                                       value_description_x=None,
                                       value_description_y=None,
                                       title=None,
                                       output_path=None,
                                       **kwargs):

    fig, ax = plotting.create_2d_subplots()

    im = electron_beam_swarm.add_fixed_values_as_2d_histogram_image(
        ax, value_name_x, value_name_y, log_x=log_x, log_y=log_y, **kwargs)

    ax.set_xlabel('{}{}'.format(
        'log10 ' if log_x else '',
        process_value_description(value_name_x, value_description_x)))
    ax.set_ylabel('{}{}'.format(
        'log10 ' if log_y else '',
        process_value_description(value_name_y, value_description_y)))
    plotting.add_2d_colorbar(fig, ax, im, label='Number of values')
    ax.set_title(title)

    plotting.render(fig, output_path=output_path)


def plot_rejection_map(electron_beam_swarm,
                       hide_grid=False,
                       title=None,
                       output_path=None,
                       **kwargs):

    grid_bounds = ((-0.015625, 23.98438), (-0.015625, 23.98438), (-14.33274,
                                                                  2.525689))

    fig, ax = plotting.create_3d_plot()

    plotting.set_3d_plot_extent(ax, *grid_bounds)
    plotting.set_3d_spatial_axis_labels(ax, unit='Mm')
    ax.invert_zaxis()
    if hide_grid:
        ax.set_axis_off()

    electron_beam_swarm.add_rejection_cause_codes_to_3d_plot_as_scatter(
        ax, **kwargs)

    ax.set_title(title)

    plotting.render(fig, output_path=output_path)


def plot_rejected_fixed_beam_value_histogram(electron_beam_swarm,
                                             value_name,
                                             xlog=False,
                                             ylog=False,
                                             value_description=None,
                                             title=None,
                                             output_path=None,
                                             **kwargs):

    fig, ax = plotting.create_2d_subplots()

    electron_beam_swarm.add_rejected_fixed_values_as_line_histograms(
        value_name, ax, **kwargs)

    if xlog:
        ax.set_xscale('log')
    if ylog:
        ax.set_yscale('log')
    ax.set_xlabel(process_value_description(value_name, value_description))
    ax.set_ylabel('Number of values')
    ax.set_title(title)

    plotting.render(fig, output_path=output_path)


if __name__ == "__main__":
    import reading
    from pathlib import Path

    electron_beam_swarm = reading.read_electron_beam_swarm_from_combined_pickles(
        Path(reading.DATA_PATH, 'phd_run',
             'en024031_emer3.0str_coarse_ebeam_351_beams.pickle'),
        derived_quantities=[])

    # plot_electron_beams(electron_beam_swarm,
    #                     value_name='deposited_power_density',
    #                     vmin=1e-10,
    #                     vmax=1e-4,
    #                     log=True,
    #                     alpha=1.0)

    # plot_electron_beams(electron_beam_swarm,
    #                     value_name='remaining_power_density',
    #                     vmax=1,
    #                     log=True,
    #                     alpha=1.0)

    # plot_electron_beams(electron_beam_swarm,
    #                     value_name='total_propagation_distance',
    #                     log=True,
    #                     alpha=1.0)

    # plot_electron_beams(electron_beam_swarm,
    #                     value_name='estimated_depletion_distance',
    #                     log=True,
    #                     alpha=1.0)

    # plot_electron_beams(electron_beam_swarm,
    #                     value_name='underestimated_total_propagation_distance',
    #                     log=True,
    #                     alpha=1.0,
    #                     relative_alpha=False)

    # plot_fixed_beam_value_histogram(electron_beam_swarm,
    #                                 value_name='krec',
    #                                 xlog=True,
    #                                 ylog=True)

    # plot_fixed_beam_value_2d_histogram(electron_beam_swarm,
    #                                    'krec',
    #                                    'total_power_density',
    #                                    log_x=True,
    #                                    log_y=True,
    #                                    log=True)

    plot_rejection_map(electron_beam_swarm,
                       included_rejectors=[1, 2, 3],
                       excluded_rejectors=[],
                       alpha=0.01,
                       limited_number=None)
