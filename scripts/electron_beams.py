import collections
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
                 derived_quantities=[],
                 verbose=True):
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
        self.verbose = bool(verbose)

        if self.verbose:
            print('Fixed scalar values:\n    {}'.format('\n    '.join(
                self.fixed_scalar_values.keys())))
            print('Fixed vector values:\n    {}'.format('\n    '.join(
                self.fixed_vector_values.keys())))
            print('Varying scalar values:\n    {}'.format('\n    '.join(
                self.varying_scalar_values.keys())))
            print('Varying vector values:\n    {}'.format('\n    '.join(
                self.varying_vector_values.keys())))
            print('Metadata:\n    {}'.format('\n    '.join(
                self.metadata.keys())))

    def add_values_to_3d_plot(self,
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

        suffix = '0' if self.has_fixed_scalar_values(value_name) else ''
        values, x, y, z = self.get_scalar_values(
            value_name, *[dim + suffix for dim in ['x', 'y', 'z']])

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
            x, y, z = self.get_scalar_values('x', 'y', 'z')
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

    def add_values_as_line_histogram(self,
                                     ax,
                                     value_name,
                                     value_name_weights=None,
                                     rejected=False,
                                     **kwargs):

        values, weights = self.get_scalar_values(value_name,
                                                 value_name_weights)

        if rejected:
            rejection_map = DistributionRejectionMap(
                self.get_rejection_cause_codes())

            rejection_map.add_values_as_line_histograms(
                ax, values, weights, **kwargs)
        else:
            self.__add_values_as_line_histogram(ax, values, weights, **kwargs)

    def add_values_as_line_histogram_difference(self,
                                                ax,
                                                value_names,
                                                value_names_weights=None,
                                                **kwargs):

        left_value_name, right_value_name = value_names
        left_value_name_weights, right_value_name_weights = value_names_weights

        left_values, left_weights = self.get_scalar_values(
            left_value_name, left_value_name_weights)

        right_values, right_weights = self.get_scalar_values(
            right_value_name, right_value_name_weights)

        return self.__add_values_as_line_histogram_difference(
            ax, (left_values, right_values), (left_weights, right_weights),
            **kwargs)

    def add_values_as_2d_histogram_image(self,
                                         ax,
                                         value_name_x,
                                         value_name_y,
                                         value_name_weights=None,
                                         rejected=False,
                                         **kwargs):

        values_x, values_y, weights = self.get_scalar_values(
            value_name_x, value_name_y, value_name_weights)

        if rejected:
            rejection_map = DistributionRejectionMap(
                self.get_rejection_cause_codes())

            return rejection_map.add_values_as_2d_histogram_image(
                ax, values_x, values_y, weights, **kwargs)
        else:
            return self.__add_values_as_2d_histogram_image(
                ax, values_x, values_y, weights, **kwargs)

    def add_values_as_2d_histogram_difference_image(self,
                                                    ax,
                                                    value_names_x,
                                                    value_names_y,
                                                    value_names_weights=(None,
                                                                         None),
                                                    **kwargs):

        left_value_name_x, right_value_name_x = value_names_x
        left_value_name_y, right_value_name_y = value_names_y
        left_value_name_weights, right_value_name_weights = value_names_weights

        left_values_x, left_values_y, left_weights = self.get_scalar_values(
            left_value_name_x, left_value_name_y, left_value_name_weights)

        right_values_x, right_values_y, right_weights = self.get_scalar_values(
            right_value_name_x, right_value_name_y, right_value_name_weights)

        return self.__add_values_as_2d_histogram_difference_image(
            ax, (left_values_x, right_values_x),
            (left_values_y, right_values_y), (left_weights, right_weights),
            **kwargs)

    def add_rejection_cause_codes_to_3d_plot_as_scatter(self, ax, **kwargs):

        rejection_map = DistributionRejectionMap(
            self.get_rejection_cause_codes())

        x0 = self.get_fixed_scalar_values('x0')
        y0 = self.get_fixed_scalar_values('y0')
        z0 = self.get_fixed_scalar_values('z0')

        rejection_map.add_to_3d_plot_as_scatter(ax, x0, y0, z0, **kwargs)

    def has_fixed_scalar_values(self, value_name):
        return value_name in self.fixed_scalar_values

    def has_varying_scalar_values(self, value_name):
        return value_name in self.varying_scalar_values

    def get_number_of_beams(self):
        return self.number_of_beams

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

    def get_scalar_values(self, *value_names):
        assert len(value_names) > 0 and value_names[0] is not None

        value_name = value_names[0]
        if self.has_fixed_scalar_values(value_name):
            assert not self.has_varying_scalar_values(
                value_name), 'Ambiguous value name {}'.format(value_name)
            getter = self.get_fixed_scalar_values
        else:
            assert self.has_varying_scalar_values(
                value_name), 'No values for value name {}'.format(value_name)
            getter = self.get_concatenated_varying_scalar_values

        return tuple([(None if value_name is None else getter(value_name))
                      for value_name in value_names])

    def get_metadata(self, metadata_label):
        return self.metadata[metadata_label]

    def get_rejection_cause_codes(self):
        return self.get_metadata('rejection_cause_code')

    def __derive_quantities(self, derived_quantities):

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

    def __find_nonwrapping_segments(self,
                                    trajectory_x,
                                    trajectory_y,
                                    trajectory_z,
                                    threshold=20.0):
        step_lengths = np.sqrt(
            np.diff(trajectory_x)**2 + np.diff(trajectory_y)**2 +
            np.diff(trajectory_z)**2)
        wrap_indices = np.where(
            step_lengths > threshold*np.mean(step_lengths))[0]
        if wrap_indices.size > 0:
            wrap_indices += 1
            return np.split(trajectory_x, wrap_indices), \
                np.split(trajectory_y, wrap_indices), \
                np.split(trajectory_z, wrap_indices)
        else:
            return [trajectory_x], [trajectory_y], [trajectory_z]

    def __add_values_as_line_histogram(self,
                                       ax,
                                       values,
                                       weights,
                                       vmin=None,
                                       vmax=None,
                                       weighted_average=False,
                                       decide_bins_in_log_space=False,
                                       scatter=False,
                                       c='k',
                                       lw=1.0,
                                       s=1.0):

        hist, bin_edges, bin_centers = _compute_histogram(
            values, weights, vmin, vmax, decide_bins_in_log_space,
            weighted_average)

        if scatter:
            ax.scatter(bin_centers, hist, c=c, s=s)
        else:
            ax.step(bin_edges[:-1], hist, where='pre', c=c, lw=lw)

    def __add_values_as_line_histogram_difference(
            self,
            ax,
            values,
            weights,
            bins=500,
            vmin=None,
            vmax=None,
            decide_bins_in_log_space=False,
            scatter=True,
            c='k',
            lw=1.0,
            s=1.0):

        hist, bin_edges, bin_centers = _compute_histogram_difference(
            values, weights, vmin, vmax, bins, decide_bins_in_log_space)

        if scatter:
            ax.scatter(bin_centers, hist, c=c, s=s)
            if decide_bins_in_log_space:
                ax.set_xlim(bin_centers[0], bin_centers[-1])
        else:
            ax.step(bin_edges[:-1], hist, where='pre', c=c, lw=lw)

    def __add_values_as_2d_histogram_image(self,
                                           ax,
                                           values_x,
                                           values_y,
                                           weights,
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
                                           cmap_name='viridis'):

        hist, bin_edges_x, bin_edges_y = _compute_2d_histogram(
            values_x, values_y, weights, vmin_x, vmax_x, vmin_y, vmax_y, log_x,
            log_y, bins_x, bins_y, weighted_average)

        return ax.pcolormesh(*np.meshgrid(bin_edges_x, bin_edges_y),
                             hist.T,
                             norm=plotting.get_normalizer(vmin, vmax, log=log),
                             vmin=vmin,
                             vmax=vmax,
                             cmap=plotting.get_cmap(cmap_name))

    def __add_values_as_2d_histogram_difference_image(self,
                                                      ax,
                                                      values_x,
                                                      values_y,
                                                      weights,
                                                      bins_x=256,
                                                      bins_y=256,
                                                      log_x=False,
                                                      log_y=False,
                                                      vmin_x=None,
                                                      vmax_x=None,
                                                      vmin_y=None,
                                                      vmax_y=None,
                                                      symlog=False,
                                                      linthresh=np.inf,
                                                      linscale=1.0,
                                                      vmin=None,
                                                      vmax=None,
                                                      cmap_name='viridis'):

        hist_diff, bin_edges_x, bin_edges_y = _compute_2d_histogram_difference(
            values_x, values_y, weights, vmin_x, vmax_x, vmin_y, vmax_y, log_x,
            log_y, bins_x, bins_y)

        if symlog:
            norm = plotting.get_symlog_normalizer(vmin,
                                                  vmax,
                                                  linthresh,
                                                  linscale=linscale)
        else:
            norm = plotting.get_linear_normalizer(vmin, vmax)

        return ax.pcolormesh(*np.meshgrid(bin_edges_x, bin_edges_y),
                             hist_diff.T,
                             norm=norm,
                             vmin=vmin,
                             vmax=vmax,
                             cmap=plotting.get_cmap(cmap_name))


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

            hist, bin_edges, _ = _compute_histogram(
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

        hist, bin_edges_x, bin_edges_y = _compute_2d_histogram(
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


def _compute_histogram(values, weights, vmin, vmax, decide_bins_in_log_space,
                       weighted_average):

    min_value = np.nanmin(values)
    max_value = np.nanmax(values)

    if vmin is not None and vmin > min_value:
        min_value = vmin
    if vmax is not None and vmax < max_value:
        max_value = vmax

    if decide_bins_in_log_space:
        values = np.log10(values)
        min_value = np.log10(min_value)
        max_value = np.log10(max_value)

    hist, bin_edges = np.histogram(values,
                                   bins='auto',
                                   range=(min_value, max_value),
                                   weights=weights)

    if weights is not None and weighted_average:
        unweighted_hist, _ = np.histogram(values,
                                          bins=bin_edges,
                                          range=(min_value, max_value))
        hist /= unweighted_hist

    bin_centers = 0.5*(bin_edges[:-1] + bin_edges[1:])

    if decide_bins_in_log_space:
        bin_edges = 10**bin_edges
        bin_centers = 10**bin_centers

    return hist, bin_edges, bin_centers


def _compute_histogram_difference(values, weights, vmin, vmax, bins,
                                  decide_bins_in_log_space):

    left_values, right_values = values
    left_weights, right_weights = weights

    min_value = min(np.nanmin(left_values), np.nanmin(right_values))
    max_value = max(np.nanmax(left_values), np.nanmax(right_values))

    if vmin is not None and vmin > min_value:
        min_value = vmin
    if vmax is not None and vmax < max_value:
        max_value = vmax

    if decide_bins_in_log_space:
        left_values = np.log10(left_values)
        right_values = np.log10(right_values)
        min_value = np.log10(min_value)
        max_value = np.log10(max_value)

    left_hist, bin_edges = np.histogram(left_values,
                                        bins=bins,
                                        range=(min_value, max_value),
                                        weights=left_weights)

    right_hist, _ = np.histogram(right_values,
                                 bins=bin_edges,
                                 range=(min_value, max_value),
                                 weights=right_weights)

    bin_centers = 0.5*(bin_edges[:-1] + bin_edges[1:])

    if decide_bins_in_log_space:
        bin_edges = 10**bin_edges
        bin_centers = 10**bin_centers

    return left_hist - right_hist, bin_edges, bin_centers


def _compute_2d_histogram(values_x, values_y, weights, vmin_x, vmax_x, vmin_y,
                          vmax_y, log_x, log_y, bins_x, bins_y,
                          weighted_average):

    min_value_x = np.nanmin(values_x) if vmin_x is None else vmin_x
    max_value_x = np.nanmax(values_x) if vmax_x is None else vmax_x
    min_value_y = np.nanmin(values_y) if vmin_y is None else vmin_y
    max_value_y = np.nanmax(values_y) if vmax_y is None else vmax_y

    if vmin_x is not None and vmin_x > min_value_x:
        min_value_x = vmin_x
    if vmax_x is not None and vmax_x < max_value_x:
        max_value_x = vmax_x
    if vmin_y is not None and vmin_y > min_value_y:
        min_value_y = vmin_y
    if vmax_y is not None and vmax_y < max_value_y:
        max_value_y = vmax_y

    if log_x:
        values_x = np.log10(values_x)
        min_value_x = np.log10(min_value_x)
        max_value_x = np.log10(max_value_x)
    if log_y:
        values_y = np.log10(values_y)
        min_value_y = np.log10(min_value_y)
        max_value_y = np.log10(max_value_y)

    hist, bin_edges_x, bin_edges_y = np.histogram2d(
        values_x,
        values_y,
        bins=[bins_x, bins_y],
        range=[[min_value_x, max_value_x], [min_value_y, max_value_y]],
        weights=weights)

    if weights is not None and weighted_average:
        unweighted_hist, _, _ = np.histogram2d(
            values_x,
            values_y,
            bins=[bin_edges_x, bin_edges_y],
            range=[[min_value_x, max_value_x], [min_value_y, max_value_y]])
        hist /= unweighted_hist

    return hist, bin_edges_x, bin_edges_y


def _compute_2d_histogram_difference(values_x, values_y, weights, vmin_x,
                                     vmax_x, vmin_y, vmax_y, log_x, log_y,
                                     bins_x, bins_y):

    left_values_x, right_values_x = values_x
    left_values_y, right_values_y = values_y
    left_weights, right_weights = weights

    left_min_value_x = np.nanmin(left_values_x) if vmin_x is None else vmin_x
    left_max_value_x = np.nanmax(left_values_x) if vmax_x is None else vmax_x
    left_min_value_y = np.nanmin(left_values_y) if vmin_y is None else vmin_y
    left_max_value_y = np.nanmax(left_values_y) if vmax_y is None else vmax_y

    right_min_value_x = np.nanmin(right_values_x) if vmin_x is None else vmin_x
    right_max_value_x = np.nanmax(right_values_x) if vmax_x is None else vmax_x
    right_min_value_y = np.nanmin(right_values_y) if vmin_y is None else vmin_y
    right_max_value_y = np.nanmax(right_values_y) if vmax_y is None else vmax_y

    min_value_x = min(left_min_value_x, right_min_value_x)
    max_value_x = max(left_max_value_x, right_max_value_x)
    min_value_y = min(left_min_value_y, right_min_value_y)
    max_value_y = max(left_max_value_y, right_max_value_y)

    if vmin_x is not None and vmin_x > min_value_x:
        min_value_x = vmin_x
    if vmax_x is not None and vmax_x < max_value_x:
        max_value_x = vmax_x
    if vmin_y is not None and vmin_y > min_value_y:
        min_value_y = vmin_y
    if vmax_y is not None and vmax_y < max_value_y:
        max_value_y = vmax_y

    if log_x:
        left_values_x = np.log10(left_values_x)
        right_values_x = np.log10(right_values_x)
        min_value_x = np.log10(min_value_x)
        max_value_x = np.log10(max_value_x)
    if log_y:
        left_values_y = np.log10(left_values_y)
        right_values_y = np.log10(right_values_y)
        min_value_y = np.log10(min_value_y)
        max_value_y = np.log10(max_value_y)

    left_hist, bin_edges_x, bin_edges_y = np.histogram2d(
        left_values_x,
        left_values_y,
        bins=[bins_x, bins_y],
        range=[[min_value_x, max_value_x], [min_value_y, max_value_y]],
        weights=left_weights)

    right_hist, _, _ = np.histogram2d(right_values_x,
                                      right_values_y,
                                      bins=[bin_edges_x, bin_edges_y],
                                      range=[[min_value_x, max_value_x],
                                             [min_value_y, max_value_y]],
                                      weights=right_weights)

    return left_hist - right_hist, bin_edges_x, bin_edges_y


VALUE_DESCRIPTIONS = {
    'x': r'$x$ [Mm]',
    'y': r'$y$ [Mm]',
    'z': r'$z$ [Mm]',
    'total_power_density': r'Total power density [erg/(cm$^3\;$s)]',
    'lower_cutoff_energy': 'Lower cut-off energy [keV]',
    'estimated_depletion_distance': 'Estimated depletion distance [Mm]',
    'total_propagation_distance': 'Total propagation distance [Mm]',
    'deposited_power_density': r'Deposited power density [erg/(cm$^3\;$s)]',
    'remaining_power_density': r'Remaining power density [erg/(cm$^3\;$s)]',
    'r': r'Mass density [10$^{-7}\:$g/cm$^3$]',
    'tg': 'Temperature [K]',
    'krec': 'Reconnection factor [Bifrost units]',
    'r0': r'Mass density [10$^{-7}\:$g/cm$^3$]',
    'tg0': 'Temperature [K]',
}

GRID_BOUNDS = ((-0.015625, 23.98438), (-0.015625, 23.98438), (-14.33274,
                                                              2.525689))


def __process_value_description(value_name, value_description):
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
                        fig=None,
                        ax=None,
                        title=None,
                        hide_grid=False,
                        render=True,
                        output_path=None,
                        **kwargs):

    if fig is None or ax is None:
        fig, ax = plotting.create_3d_plot()

    plotting.set_3d_plot_extent(ax, *GRID_BOUNDS)
    plotting.set_3d_spatial_axis_labels(ax, unit='Mm')
    ax.invert_zaxis()
    if hide_grid:
        ax.set_axis_off()

    if value_name is None:
        electron_beam_swarm.add_to_3d_plot_with_single_color(ax, **kwargs)
    else:
        norm, cmap = electron_beam_swarm.add_values_to_3d_plot(
            ax, value_name, **kwargs)

        plotting.add_3d_colorbar(fig,
                                 norm,
                                 cmap,
                                 label=__process_value_description(
                                     value_name, value_description))

    ax.set_title(title)

    if render:
        plotting.render(fig, output_path=output_path)


def plot_beam_value_histogram(electron_beam_swarm,
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

    electron_beam_swarm.add_values_as_line_histogram(
        ax,
        value_name,
        value_name_weights=value_name_weights,
        decide_bins_in_log_space=log_x,
        **kwargs)

    if log_x:
        ax.set_xscale('log')
    if log_y:
        ax.set_yscale('log')
    ax.set_xlabel(__process_value_description(value_name, value_description))
    ax.set_ylabel('Number of values' if
                  value_name_weights is None else __process_value_description(
                      value_name_weights, value_description_weights))
    ax.set_title(title)

    if render:
        plotting.render(fig, output_path=output_path)


def plot_beam_value_histogram_difference(electron_beam_swarm,
                                         value_names,
                                         value_names_weights=None,
                                         fig=None,
                                         ax=None,
                                         value_description=None,
                                         value_description_weights=None,
                                         title=None,
                                         render=True,
                                         output_path=None,
                                         log_x=False,
                                         symlog_y=False,
                                         linethresh_y=np.inf,
                                         **kwargs):

    if fig is None or ax is None:
        fig, ax = plotting.create_2d_subplots()

    electron_beam_swarm.add_values_as_line_histogram_difference(
        ax,
        value_names,
        value_names_weights=value_names_weights,
        decide_bins_in_log_space=log_x,
        **kwargs)

    if log_x:
        ax.set_xscale('log')
    if symlog_y:
        ax.set_yscale('symlog', linthreshy=linethresh_y)
    ax.set_xlabel(
        __process_value_description(value_names[0], value_description))
    ax.set_ylabel('Number of values' if value_names_weights[0] is None else
                  __process_value_description(value_names_weights[0],
                                              value_description_weights))
    ax.set_title(title)

    if render:
        plotting.render(fig, output_path=output_path)


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

    plotting.set_3d_plot_extent(ax, *GRID_BOUNDS)
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
    ax.set_xlabel(__process_value_description(value_name, value_description))
    ax.set_ylabel('Number of values' if
                  value_name_weights is None else __process_value_description(
                      value_name_weights, value_description_weights))
    ax.set_title(title)

    if render:
        plotting.render(fig, output_path=output_path)


def plot_beam_value_2d_histogram(electron_beam_swarm,
                                 value_name_x,
                                 value_name_y,
                                 value_name_weights=None,
                                 fig=None,
                                 ax=None,
                                 value_description_x=None,
                                 value_description_y=None,
                                 value_description_weights=None,
                                 title=None,
                                 render=True,
                                 output_path=None,
                                 log_x=False,
                                 log_y=False,
                                 **kwargs):

    if fig is None or ax is None:
        fig, ax = plotting.create_2d_subplots()

    im = electron_beam_swarm.add_values_as_2d_histogram_image(
        ax,
        value_name_x,
        value_name_y,
        value_name_weights=value_name_weights,
        log_x=log_x,
        log_y=log_y,
        **kwargs)

    ax.set_xlabel('{}{}'.format(
        r'$\log_{10}$ ' if log_x else '',
        __process_value_description(value_name_x, value_description_x)))
    ax.set_ylabel('{}{}'.format(
        r'$\log_{10}$ ' if log_y else '',
        __process_value_description(value_name_y, value_description_y)))
    plotting.add_2d_colorbar(
        fig,
        ax,
        im,
        label=('Number of values'
               if value_name_weights is None else __process_value_description(
                   value_name_weights, value_description_weights)))

    ax.set_title(title)

    if render:
        plotting.render(fig, output_path=output_path)


def plot_beam_value_2d_histogram_difference(electron_beam_swarm,
                                            value_names_x,
                                            value_names_y,
                                            value_names_weights=None,
                                            fig=None,
                                            ax=None,
                                            value_description_x=None,
                                            value_description_y=None,
                                            value_description_weights=None,
                                            title=None,
                                            render=True,
                                            output_path=None,
                                            log_x=False,
                                            log_y=False,
                                            **kwargs):

    if fig is None or ax is None:
        fig, ax = plotting.create_2d_subplots()

    im = electron_beam_swarm.add_values_as_2d_histogram_difference_image(
        ax,
        value_names_x,
        value_names_y,
        value_names_weights=value_names_weights,
        log_x=log_x,
        log_y=log_y,
        **kwargs)

    ax.set_xlabel('{}{}'.format(
        r'$\log_{10}$ ' if log_x else '',
        __process_value_description(value_names_x[0], value_description_x)))
    ax.set_ylabel('{}{}'.format(
        r'$\log_{10}$ ' if log_y else '',
        __process_value_description(value_names_y[0], value_description_y)))
    plotting.add_2d_colorbar(
        fig,
        ax,
        im,
        label=('Number of values' if
               value_names_weights[0] is None else __process_value_description(
                   value_names_weights[0], value_description_weights)))

    ax.set_title(title)

    if render:
        plotting.render(fig, output_path=output_path)


def plot_beam_value_2d_histogram_comparison(electron_beam_swarm, value_names_x,
                                            value_names_y, value_names_weights,
                                            **kwargs):

    fig, axes = plotting.create_2d_subplots(ncols=2, figsize=(8, 4))

    plot_beam_value_2d_histogram(electron_beam_swarm,
                                 value_names_x[0],
                                 value_names_y[0],
                                 value_name_weights=value_names_weights[0],
                                 fig=fig,
                                 ax=axes[0],
                                 render=False,
                                 **kwargs)

    plot_beam_value_2d_histogram(electron_beam_swarm,
                                 value_names_x[1],
                                 value_names_y[1],
                                 value_name_weights=value_names_weights[1],
                                 fig=fig,
                                 ax=axes[1],
                                 **kwargs)


if __name__ == "__main__":
    import reading
    from pathlib import Path

    electron_beam_swarm = reading.read_electron_beam_swarm_from_combined_pickles(
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
