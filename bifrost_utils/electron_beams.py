import collections
import functools
import numpy as np
from pathlib import Path
import bifrost_utils.plotting as plotting
import bifrost_utils.field_lines as field_lines


class ElectronBeamSwarm(field_lines.FieldLineSet3):

    VALUE_DESCRIPTIONS = {
        'x': r'$x$ [Mm]',
        'y': r'$y$ [Mm]',
        'z': r'$z$ [Mm]',
        'total_power_density': r'Total power density [erg/(cm$^3\;$s)]',
        'lower_cutoff_energy': 'Lower cut-off energy [keV]',
        'estimated_depletion_distance': 'Estimated depletion distance [Mm]',
        'total_propagation_distance': 'Total propagation distance [Mm]',
        'deposited_power_density':
        r'Deposited power density [erg/(cm$^3\;$s)]',
        'remaining_power_density':
        r'Remaining power density [erg/(cm$^3\;$s)]',
        'r': r'Mass density [10$^{-7}\:$g/cm$^3$]',
        'tg': 'Temperature [K]',
        'krec': 'Reconnection factor [Bifrost units]',
        'r0': r'Mass density [10$^{-7}\:$g/cm$^3$]',
        'tg0': 'Temperature [K]',
    }

    @staticmethod
    def from_file(file_path, derived_quantities=[], verbose=False):
        import bifrost_utils.reading as reading
        file_path = Path(file_path)
        extension = file_path.suffix
        if extension == '.pickle':
            electron_beam_swarm = reading.read_electron_beam_swarm_from_combined_pickles(
                file_path,
                derived_quantities=derived_quantities,
                verbose=verbose)
        elif extension == '.fl':
            electron_beam_swarm = reading.read_electron_beam_swarm_from_custom_binary_file(
                file_path,
                derived_quantities=derived_quantities,
                verbose=verbose)
        else:
            raise ValueError(
                'Invalid file extension {} for electron beam data.'.format(
                    extension))

        return electron_beam_swarm

    def __init__(self,
                 number_of_beams,
                 fixed_scalar_values,
                 fixed_vector_values,
                 varying_scalar_values,
                 varying_vector_values,
                 metadata,
                 derived_quantities=[],
                 verbose=False):
        super().__init__(number_of_beams,
                         fixed_scalar_values,
                         fixed_vector_values,
                         varying_scalar_values,
                         varying_vector_values,
                         derived_quantities=derived_quantities,
                         verbose=verbose)
        assert isinstance(metadata, dict)
        self.number_of_beams = number_of_beams
        self.metadata = metadata

        if self.verbose:
            print('Metadata:\n    {}'.format('\n    '.join(
                self.metadata.keys())))

    def add_values_as_line_histogram(self,
                                     ax,
                                     value_name,
                                     value_name_weights=None,
                                     rejected=False,
                                     **kwargs):

        if rejected:
            values, weights = self.get_scalar_values(value_name,
                                                     value_name_weights)

            rejection_map = DistributionRejectionMap(
                self.get_rejection_cause_codes())

            rejection_map.add_values_as_line_histograms(
                ax, values, weights, **kwargs)
        else:
            super().add_values_as_line_histogram(
                ax,
                value_name,
                value_name_weights=value_name_weights,
                **kwargs)

    def add_values_as_2d_histogram_image(self,
                                         ax,
                                         value_name_x,
                                         value_name_y,
                                         value_name_weights=None,
                                         rejected=False,
                                         **kwargs):

        if rejected:
            values_x, values_y, weights = self.get_scalar_values(
                value_name_x, value_name_y, value_name_weights)

            rejection_map = DistributionRejectionMap(
                self.get_rejection_cause_codes())

            return rejection_map.add_values_as_2d_histogram_image(
                ax, values_x, values_y, weights, **kwargs)
        else:
            return super().add_values_as_2d_histogram_image(
                ax,
                value_name_x,
                value_name_y,
                value_name_weights=value_name_weights,
                **kwargs)

    def add_rejection_cause_codes_to_3d_plot_as_scatter(self, ax, **kwargs):

        rejection_map = DistributionRejectionMap(
            self.get_rejection_cause_codes())

        x0 = self.get_fixed_scalar_values('x0')
        y0 = self.get_fixed_scalar_values('y0')
        z0 = self.get_fixed_scalar_values('z0')

        rejection_map.add_to_3d_plot_as_scatter(ax, x0, y0, z0, **kwargs)

    def get_number_of_beams(self):
        return self.number_of_beams

    def get_metadata(self, metadata_label):
        return self.metadata[metadata_label]

    def get_rejection_cause_codes(self):
        return self.get_metadata('rejection_cause_code')

    def _derive_quantities(self, derived_quantities):

        for value_name in filter(
                lambda name: name[-1] == '0' and self.
                has_varying_scalar_values(name[:-1]) and not self.
                has_fixed_scalar_values(name[:-1]), derived_quantities):

            self.fixed_scalar_values[value_name] = np.asfarray([
                values[0]
                for values in self.get_varying_scalar_values(value_name[:-1])
            ])

        if 'underestimated_total_propagation_distance' in derived_quantities:
            self.fixed_scalar_values['underestimated_total_propagation_distance'] = \
                np.where(self.fixed_scalar_values['total_propagation_distance'] > self.fixed_scalar_values['estimated_depletion_distance'], self.fixed_scalar_values['total_propagation_distance'], np.nan)

        if 'remaining_power_density' in derived_quantities:
            self.varying_scalar_values['remaining_power_density'] = [self.fixed_scalar_values['total_power_density'][i] - \
                np.cumsum(
                    self.varying_scalar_values['deposited_power_density'][i]) for i in range(self.get_number_of_beams())]

        if 'distance_weighted_deposited_power_density' in derived_quantities:
            self.varying_scalar_values[
                'distance_weighted_deposited_power_density'] = [
                    deposited_power_densities*
                    np.arange(deposited_power_densities.size)
                    for deposited_power_densities in
                    self.varying_scalar_values['deposited_power_density']
                ]


class DistributionRejectionMap:
    def __init__(self, rejection_cause_codes):
        assert rejection_cause_codes.dtype == np.ubyte
        self.rejection_cause_codes = rejection_cause_codes
        self.__init_rejectors()

    def add_to_3d_plot_as_scatter(self,
                                  ax,
                                  x0,
                                  y0,
                                  z0,
                                  included_rejectors=[1, 2, 3],
                                  excluded_rejectors=[],
                                  aggregated=False,
                                  limited_number=None,
                                  s=1.0,
                                  marker='o',
                                  edgecolors='none',
                                  depthshade=False,
                                  alpha=1.0,
                                  textbox_loc=2):

        included_codes, rejection_cause_codes, x0, y0, z0 = self.__get_filtered_codes_and_values(
            included_rejectors, excluded_rejectors, aggregated, limited_number,
            x0, y0, z0)

        colors = self.__select_colors(aggregated)
        c = [colors[code] for code in rejection_cause_codes]

        ax.scatter(x0,
                   y0,
                   z0,
                   c=c,
                   s=s,
                   marker=marker,
                   edgecolors=edgecolors,
                   depthshade=depthshade,
                   alpha=alpha)

        self.__add_labels_for_scatter(ax,
                                      included_codes,
                                      aggregated,
                                      marker=marker)

        if not aggregated:
            plotting.add_textbox(ax, self.__get_rejector_legend_text(),
                                 textbox_loc)

    def add_values_as_line_histograms(self,
                                      ax,
                                      values,
                                      weights,
                                      included_rejectors=[1, 2, 3],
                                      excluded_rejectors=[],
                                      aggregated=False,
                                      limited_number=None,
                                      vmin=None,
                                      vmax=None,
                                      weighted_average=False,
                                      decide_bins_in_log_space=False,
                                      lw=1.0,
                                      textbox_loc=9):

        included_codes, rejection_cause_codes, values = self.__get_filtered_codes_and_values(
            included_rejectors, excluded_rejectors, aggregated, limited_number,
            values)

        labels = self.__select_labels(aggregated)
        colors = self.__select_colors(aggregated)

        for code in included_codes:

            mask = (rejection_cause_codes & code != 0) if aggregated else (
                rejection_cause_codes == code)

            hist, bin_edges, _ = plotting.compute_histogram(
                values[mask], None if weights is None else weights[mask], vmin,
                vmax, decide_bins_in_log_space, weighted_average)

            ax.step(bin_edges[:-1],
                    hist,
                    where='pre',
                    lw=lw,
                    c=colors[code],
                    label=labels[code])

        ax.legend(loc='best')

        if not aggregated:
            plotting.add_textbox(ax, self.__get_rejector_legend_text(),
                                 textbox_loc)

    def add_values_as_2d_histogram_image(self,
                                         ax,
                                         values_x,
                                         values_y,
                                         weights,
                                         included_rejectors=[1, 2, 3],
                                         limited_number=None,
                                         bins_x=256,
                                         bins_y=256,
                                         log_x=False,
                                         log_y=False,
                                         vmin_x=None,
                                         vmax_x=None,
                                         vmin_y=None,
                                         vmax_y=None,
                                         weighted_average=False,
                                         log=False,
                                         vmin=None,
                                         vmax=None,
                                         cmap_name='viridis',
                                         textbox_loc=9):

        excluded_rejectors = []
        aggregated = True

        included_codes, rejection_cause_codes, values_x, values_y = self.__get_filtered_codes_and_values(
            included_rejectors, excluded_rejectors, aggregated, limited_number,
            values_x, values_y)

        mask = rejection_cause_codes & functools.reduce(
            lambda a, b: a | b, included_codes, 0) != 0

        hist, bin_edges_x, bin_edges_y = plotting.compute_2d_histogram(
            values_x[mask], values_y[mask],
            None if weights is None else weights[mask], vmin_x, vmax_x, vmin_y,
            vmax_y, log_x, log_y, bins_x, bins_y, weighted_average)

        plotting.add_textbox(
            ax, '\n'.join([
                self.rejector_names[identifier - 1]
                for identifier in included_rejectors
            ]), textbox_loc)

        return ax.pcolormesh(*np.meshgrid(bin_edges_x, bin_edges_y),
                             hist.T,
                             norm=plotting.get_normalizer(vmin, vmax, log=log),
                             vmin=vmin,
                             vmax=vmax,
                             cmap=plotting.get_cmap(cmap_name))

    def __init_rejectors(self):
        self.rejector_names = [
            'Too low total power density', 'Too short depletion distance',
            'Too perpendicular direction'
        ]
        number_of_rejectors = len(self.rejector_names)
        number_of_valid_codes = (1 << number_of_rejectors) - 1
        self.rejector_identifiers = range(1, number_of_rejectors + 1)
        self.valid_rejection_cause_codes = range(1, number_of_valid_codes + 1)
        self.valid_aggregated_rejection_cause_codes = [
            1 << (identifier - 1) for identifier in self.rejector_identifiers
        ]

        rejector_bitflags = [1 << n for n in range(number_of_rejectors)]
        combinations = []
        for code in self.valid_rejection_cause_codes:
            combinations.append([])
            for identifier in self.rejector_identifiers:
                if (code & rejector_bitflags[identifier - 1]) != 0:
                    combinations[-1].append(str(identifier))

        self.rejection_code_labels = collections.OrderedDict(
            zip(self.valid_rejection_cause_codes, [
                ' & '.join(combination) +
                (' only' if len(combination) < number_of_rejectors else '')
                for combination in combinations
            ]))

        self.rejection_code_colors = collections.OrderedDict(
            zip(self.valid_rejection_cause_codes,
                plotting.get_default_colors()[:number_of_valid_codes]))

        self.aggregated_code_labels = collections.OrderedDict(
            zip(self.valid_aggregated_rejection_cause_codes,
                self.rejector_names))

        self.aggregated_code_colors = collections.OrderedDict(
            zip(
                self.valid_aggregated_rejection_cause_codes,
                plotting.get_default_colors()[number_of_valid_codes:(
                    number_of_valid_codes + number_of_rejectors)]))

    def __get_filtered_codes_and_values(self, included_rejectors,
                                        excluded_rejectors, aggregated,
                                        limited_number, *values):

        if aggregated:
            excluded_rejectors = []

        inclusion_bitflag, exclusion_bitflag = self.__create_bitflags(
            included_rejectors, excluded_rejectors)

        included_codes = self.__get_included_rejection_cause_codes(
            inclusion_bitflag, exclusion_bitflag, aggregated)

        inclusion_mask = self.__create_inclusion_mask(inclusion_bitflag,
                                                      exclusion_bitflag)

        rejection_cause_codes = self.rejection_cause_codes[inclusion_mask]
        values = [v[inclusion_mask] for v in values]

        indices = self.__sample_limited_indices(rejection_cause_codes.size,
                                                limited_number)
        rejection_cause_codes = rejection_cause_codes[indices]
        values = [v[indices] for v in values]

        return tuple([included_codes, rejection_cause_codes] + values)

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
            (100.0*number_of_included_values)/self.rejection_cause_codes.size))
        return inclusion_mask

    def __get_included_rejection_cause_codes(self, inclusion_bitflag,
                                             exclusion_bitflag, aggregated):
        valid_codes = self.valid_aggregated_rejection_cause_codes if aggregated else self.valid_rejection_cause_codes
        return list(
            filter(
                lambda code: code & inclusion_bitflag != 0 and code &
                exclusion_bitflag == 0, valid_codes))

    def __sample_limited_indices(self, total_number, limited_number):
        return slice(None) if limited_number is None else \
            np.random.permutation(total_number)[
            :min(total_number, int(limited_number))]

    def __select_labels(self, aggregated):
        return self.aggregated_code_labels if aggregated else self.rejection_code_labels

    def __select_colors(self, aggregated):
        return self.aggregated_code_colors if aggregated else self.rejection_code_colors

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
                                 included_codes,
                                 aggregated,
                                 marker='.'):

        labels = self.__select_labels(aggregated)
        colors = self.__select_colors(aggregated)

        for code in included_codes:
            ax.plot([], [],
                    linestyle='none',
                    marker=marker,
                    color=colors[code],
                    label=labels[code])

        ax.legend(loc='best')


def plot_electron_beams(*args, **kwargs):
    field_lines.plot_field_lines(*args, **kwargs)


def plot_beam_value_histogram(*args, **kwargs):
    field_lines.plot_field_line_value_histogram(*args, **kwargs)


def plot_beam_value_histogram_difference(*args, **kwargs):
    field_lines.plot_field_line_value_histogram_difference(*args, **kwargs)


def plot_rejection_map(electron_beam_swarm,
                       fig=None,
                       ax=None,
                       hide_grid=False,
                       title=None,
                       render=True,
                       output_path=None,
                       **kwargs):

    if fig is None or ax is None:
        fig, ax = plotting.create_3d_plot()

    plotting.set_3d_plot_extent(ax, *ElectronBeamSwarm.GRID_BOUNDS)
    plotting.set_3d_spatial_axis_labels(ax, unit='Mm')
    ax.invert_zaxis()
    if hide_grid:
        ax.set_axis_off()

    electron_beam_swarm.add_rejection_cause_codes_to_3d_plot_as_scatter(
        ax, **kwargs)

    ax.set_title(title)

    if render:
        plotting.render(fig, output_path=output_path)


def plot_rejected_beam_value_histogram(electron_beam_swarm,
                                       value_name,
                                       value_name_weights=None,
                                       fig=None,
                                       ax=None,
                                       value_description=None,
                                       value_description_weights=None,
                                       title=None,
                                       render=True,
                                       output_path=None,
                                       log_x=False,
                                       log_y=False,
                                       **kwargs):

    if fig is None or ax is None:
        fig, ax = plotting.create_2d_subplots()

    electron_beam_swarm.add_rejected_values_as_line_histograms(
        ax,
        value_name,
        value_name_weights=value_name_weights,
        decide_bins_in_log_space=log_x,
        **kwargs)

    if log_x:
        ax.set_xscale('log')
    if log_y:
        ax.set_yscale('log')
    ax.set_xlabel(
        electron_beam_swarm.process_value_description(value_name,
                                                      value_description))
    ax.set_ylabel('Number of values' if value_name_weights is None else
                  electron_beam_swarm.process_value_description(
                      value_name_weights, value_description_weights))
    ax.set_title(title)

    if render:
        plotting.render(fig, output_path=output_path)


def plot_beam_value_2d_histogram(*args, **kwargs):
    field_lines.plot_field_line_value_2d_histogram(*args, **kwargs)


def plot_beam_value_2d_histogram_difference(*args, **kwargs):
    field_lines.plot_field_line_value_2d_histogram_difference(*args, **kwargs)


def plot_beam_value_2d_histogram_comparison(*args, **kwargs):
    field_lines.plot_field_line_value_2d_histogram_comparison(*args, **kwargs)


if __name__ == "__main__":
    import bifrost_utils.reading as reading

    electron_beam_swarm = ElectronBeamSwarm.from_file(
        Path(
            reading.DATA_PATH,
            'phd_run',
            #'en024031_emer3.0str_ebeam_351_beams_no_distance_threshold.pickle'),
            'en024031_emer3.0str_ebeam_351_beams.pickle'),
        derived_quantities=[
            'r0', 'tg0', 'distance_weighted_deposited_power_density'
        ])

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

    # plot_beam_value_histogram(electron_beam_swarm,
    #                           value_name='krec',
    #                           log_x=True,
    #                           log_y=True)

    # plot_beam_value_2d_histogram(electron_beam_swarm,
    #                              'krec',
    #                              'total_power_density',
    #                              log_x=True,
    #                              log_y=True,
    #                              log=True)

    # plot_rejection_map(electron_beam_swarm,
    #                    included_rejectors=[2],
    #                    excluded_rejectors=[1, 3],
    #                    alpha=0.01,
    #                    limited_number=None)

    # plot_beam_value_histogram(electron_beam_swarm,
    #                           'z0',
    #                           log_x=False,
    #                           rejected=True,
    #                           included_rejectors=[2],
    #                           excluded_rejectors=[1, 3],
    #                           limited_number=None,
    #                           aggregated=False)

    # plot_beam_value_2d_histogram(
    #     electron_beam_swarm,
    #     'krec',
    #     'z0',
    #     log_x=True,
    #     #log_y=True,
    #     rejected=True,
    #     included_rejectors=[2],
    #     limited_number=None)

    # plot_beam_value_2d_histogram_comparison(
    #     electron_beam_swarm, ('r0', 'r'), ('tg0', 'tg'),
    #     ('total_power_density', 'deposited_power_density'),
    #     bins_x=180,
    #     bins_y=180,
    #     vmin_x=10**(-7.5),
    #     vmax_x=10**(1.0),
    #     vmin_y=10**(3.2),
    #     vmax_y=10**(6.2),
    #     log_x=True,
    #     log_y=True,
    #     vmin=1e-5,
    #     vmax=1e3,
    #     log=True)

    # plot_beam_value_2d_histogram_difference(
    #     electron_beam_swarm, ('r', 'r0'), ('tg', 'tg0'),
    #     value_names_weights=('deposited_power_density', 'total_power_density'),
    #     bins_x=300,
    #     bins_y=300,
    #     log_x=True,
    #     log_y=True,
    #     symlog=True,
    #     linthresh=1e-4,
    #     cmap_name='transport')

    # plot_beam_value_2d_histogram_difference(
    #     electron_beam_swarm, ('r', 'r0'), ('tg', 'tg0'),
    #     value_names_weights=('deposited_power_density', 'total_power_density'),
    #     bins_x=256,
    #     bins_y=256,
    #     log_x=True,
    #     log_y=True,
    #     symlog=True,
    #     linthresh=1e-4,
    #     cmap_name='transport',
    #     vmin=-1e3,
    #     vmax=1e3)

    # plot_beam_value_histogram_difference(
    #     electron_beam_swarm, ('r', 'r0'),
    #     value_names_weights=('deposited_power_density', 'total_power_density'),
    #     bins=500,
    #     log_x=True,
    #     symlog_y=True,
    #     linethresh_y=1e-1)

    # plot_beam_value_2d_histogram_comparison(
    #     electron_beam_swarm, ('r', 'r'), ('tg', 'tg'),
    #     ('deposited_power_density',
    #      'distance_weighted_deposited_power_density'),
    #     bins_x=256,
    #     bins_y=256,
    #     vmin_x=10**(-8.0),
    #     vmax_x=10**(1.0),
    #     vmin_y=10**(3.2),
    #     vmax_y=10**(6.6),
    #     log_x=True,
    #     log_y=True,
    #     log=True)

    plot_beam_value_2d_histogram(
        electron_beam_swarm,
        'r0',
        'tg0',
        value_name_weights='total_propagation_distance',
        log_x=True,
        log_y=True,
        log=True,
        weighted_average=True)
