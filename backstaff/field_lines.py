import numpy as np
import scipy.ndimage as ndimage
import scipy.interpolate as interpolate
from pathlib import Path
try:
    import backstaff.units as units
    import backstaff.plotting as plotting
except ModuleNotFoundError:
    import units
    import plotting


class FieldLineSet3:

    VALUE_DESCRIPTIONS = {
        'x': r'$x$ [Mm]',
        'y': r'$y$ [Mm]',
        'z': r'$z$ [Mm]',
        'r': r'Mass density [g/cm$^3$]',
        'tg': 'Temperature [K]',
        'r0': r'Mass density [g/cm$^3$]',
        'tg0': 'Temperature [K]',
    }

    VALUE_UNIT_CONVERTERS = {
        'r': lambda f: f*units.U_R,
        'r0': lambda f: f*units.U_R,
    }

    @staticmethod
    def from_file(file_path, params={}, derived_quantities=[], verbose=False):
        import backstaff.reading as reading
        file_path = Path(file_path)
        extension = file_path.suffix
        if extension == '.pickle':
            field_line_set = reading.read_3d_field_line_set_from_combined_pickles(
                file_path,
                params=params,
                derived_quantities=derived_quantities,
                verbose=verbose)
        elif extension == '.fl':
            field_line_set = reading.read_3d_field_line_set_from_custom_binary_file(
                file_path,
                params=params,
                derived_quantities=derived_quantities,
                verbose=verbose)
        else:
            raise ValueError(
                'Invalid file extension {} for field line data.'.format(
                    extension))

        return field_line_set

    def __init__(self,
                 domain_bounds,
                 number_of_field_lines,
                 fixed_scalar_values,
                 fixed_vector_values,
                 varying_scalar_values,
                 varying_vector_values,
                 params={},
                 derived_quantities=[],
                 verbose=False):
        assert all([upper >= lower for lower, upper in domain_bounds])
        assert isinstance(number_of_field_lines, int)
        assert isinstance(fixed_scalar_values, dict)
        assert isinstance(fixed_vector_values, dict)
        assert isinstance(varying_scalar_values, dict)
        assert isinstance(varying_vector_values, dict)
        assert isinstance(derived_quantities, list)
        assert isinstance(params, dict)
        self.bounds_x, self.bounds_y, self.bounds_z = tuple(domain_bounds)
        self.bounds_z = (-self.bounds_z[1], -self.bounds_z[0]
                         )  # Use height instead of depth
        self.number_of_field_lines = number_of_field_lines
        self.fixed_scalar_values = fixed_scalar_values
        self.fixed_vector_values = fixed_vector_values
        self.varying_scalar_values = varying_scalar_values
        self.varying_vector_values = varying_vector_values
        self.params = params
        self._derive_quantities(derived_quantities)
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
            print('Parameters:\n    {}'.format('\n    '.join(
                self.params.keys())))

    def get_subset(self,
                   quantities,
                   *args,
                   included_field_lines_finder=None,
                   **kwargs):
        derived_quantities = list(quantities)
        if included_field_lines_finder is None:
            return self.__class__(
                self.get_domain_bounds(),
                self.get_number_of_field_lines(), {
                    derived_quantities.pop(derived_quantities.index(k)): v
                    for k, v in self.fixed_scalar_values.items()
                    if k in derived_quantities
                }, {
                    derived_quantities.pop(derived_quantities.index(k)): v
                    for k, v in self.fixed_vector_values.items()
                    if k in derived_quantities
                }, {
                    derived_quantities.pop(derived_quantities.index(k)): v
                    for k, v in self.varying_scalar_values.items()
                    if k in derived_quantities
                }, {
                    derived_quantities.pop(derived_quantities.index(k)): v
                    for k, v in self.varying_vector_values.items()
                    if k in derived_quantities
                },
                *args,
                params=self.params,
                derived_quantities=derived_quantities,
                verbose=self.verbose,
                **kwargs)
        else:
            included_field_line_indices = included_field_lines_finder(
                self.fixed_scalar_values, self.varying_scalar_values)
            return self.__class__(
                self.get_domain_bounds(),
                len(included_field_line_indices), {
                    derived_quantities.pop(derived_quantities.index(k)):
                    v[included_field_line_indices]
                    for k, v in self.fixed_scalar_values.items()
                    if k in derived_quantities
                }, {
                    derived_quantities.pop(derived_quantities.index(k)):
                    v[included_field_line_indices]
                    for k, v in self.fixed_vector_values.items()
                    if k in derived_quantities
                }, {
                    derived_quantities.pop(derived_quantities.index(k)):
                    [v[i] for i in included_field_line_indices]
                    for k, v in self.varying_scalar_values.items()
                    if k in derived_quantities
                }, {
                    derived_quantities.pop(derived_quantities.index(k)):
                    [v[i] for i in included_field_line_indices]
                    for k, v in self.varying_vector_values.items()
                    if k in derived_quantities
                },
                *args,
                params=self.params,
                derived_quantities=derived_quantities,
                verbose=self.verbose,
                **kwargs)

    def compute_aggregate_value(self,
                                value_name,
                                aggregator,
                                do_conversion=True,
                                included_field_lines_finder=None,
                                included_points_finder=None,
                                result_if_empty=np.nan):
        values, = self.get_scalar_values(
            value_name,
            included_field_lines_finder=included_field_lines_finder,
            included_points_finder=included_points_finder)
        if values is None:
            return result_if_empty
        values = self._convert_values(value_name, values, do_conversion)
        return aggregator(values)

    def add_values_to_3d_plot(self,
                              ax,
                              value_name,
                              do_conversion=True,
                              included_field_lines_finder=None,
                              included_points_finder=None,
                              log=False,
                              vmin=None,
                              vmax=None,
                              symlog=False,
                              linthresh=np.inf,
                              linscale=1.0,
                              cmap_name='viridis',
                              cmap_bad_color='w',
                              s=1.0,
                              marker='o',
                              edgecolors='none',
                              depthshade=False,
                              alpha=1.0,
                              relative_alpha=True):

        suffix = '0' if self.has_fixed_scalar_values(value_name) else ''
        values, x, y, z = self.get_scalar_values(
            value_name,
            *[dim + suffix for dim in ['x', 'y', 'z']],
            included_field_lines_finder=included_field_lines_finder,
            included_points_finder=included_points_finder)

        values = self._convert_values(value_name, values, do_conversion)

        if vmin is None:
            vmin = np.nanmin(values)
        if vmax is None:
            vmax = np.nanmax(values)

        if symlog:
            norm = plotting.get_symlog_normalizer(vmin,
                                                  vmax,
                                                  linthresh,
                                                  linscale=linscale)
        else:
            norm = plotting.get_normalizer(vmin, vmax, log=log)

        c = plotting.colors_from_values(values,
                                        norm,
                                        plotting.get_cmap(
                                            cmap_name,
                                            bad_color=cmap_bad_color),
                                        alpha=alpha,
                                        relative_alpha=relative_alpha)

        ax.scatter(self._convert_values('x', x, do_conversion),
                   self._convert_values('y', y, do_conversion),
                   self._convert_values('z', z, do_conversion),
                   c=c,
                   s=s,
                   marker=marker,
                   edgecolors=edgecolors,
                   depthshade=depthshade)

        return plotting.get_normalizer(vmin, vmax,
                                       log=log), plotting.get_cmap(cmap_name)

    def add_to_3d_plot_with_single_color(self,
                                         ax,
                                         do_conversion=True,
                                         included_field_lines_finder=None,
                                         included_points_finder=None,
                                         scatter=False,
                                         c='k',
                                         lw=1.0,
                                         s=0.02,
                                         marker='.',
                                         edgecolors='none',
                                         depthshade=False,
                                         alpha=1.0):
        if scatter:
            x, y, z = self.get_scalar_values(
                'x',
                'y',
                'z',
                included_field_lines_finder=included_field_lines_finder,
                included_points_finder=included_points_finder)
            ax.scatter(self._convert_values('x', x, do_conversion),
                       self._convert_values('y', y, do_conversion),
                       self._convert_values('z', z, do_conversion),
                       c=c,
                       s=s,
                       marker=marker,
                       edgecolors=edgecolors,
                       depthshade=depthshade,
                       alpha=alpha)
        else:
            paths_x = self._convert_values('x',
                                           self.get_varying_scalar_values('x'),
                                           do_conversion)
            paths_y = self._convert_values('y',
                                           self.get_varying_scalar_values('y'),
                                           do_conversion)
            paths_z = self._convert_values('z',
                                           self.get_varying_scalar_values('z'),
                                           do_conversion)
            for field_line_idx in range(self.get_number_of_field_lines()):
                for x, y, z in zip(*self.__find_nonwrapping_segments(
                        paths_x[field_line_idx], paths_y[field_line_idx],
                        paths_z[field_line_idx])):
                    ax.plot(x, y, z, c=c, lw=lw, alpha=alpha)

    def add_values_as_2d_property_plot(self,
                                       ax,
                                       value_name_x,
                                       value_name_y,
                                       value_name_color=None,
                                       do_conversion=True,
                                       included_field_lines_finder=None,
                                       included_points_finder=None,
                                       varying_points_processor=None,
                                       mode='instant',
                                       save_path=None,
                                       **kwargs):

        if mode == 'load':
            data = np.load(save_path)
            values_x = data['values_x']
            values_y = data['values_y']
            values_color = data['values_color'] if 'values_color' in data.keys(
            ) else None
            print('Loaded {}'.format(save_path))
        else:
            values_x, values_y, values_color = self.get_scalar_values(
                value_name_x,
                value_name_y,
                value_name_color,
                included_field_lines_finder=included_field_lines_finder,
                included_points_finder=included_points_finder,
                varying_points_processor=varying_points_processor)

            values_x = self._convert_values(value_name_x, values_x,
                                            do_conversion)
            values_y = self._convert_values(value_name_y, values_y,
                                            do_conversion)
            if values_color is not None:
                values_color = self._convert_values(value_name_color,
                                                    values_color,
                                                    do_conversion)

        if mode == 'save':
            if values_color is None:
                np.savez_compressed(save_path,
                                    values_x=values_x,
                                    values_y=values_y)
            else:
                np.savez_compressed(save_path,
                                    values_x=values_x,
                                    values_y=values_y,
                                    values_color=values_color)
            print('Saved {}'.format(save_path))
            return (None, None)
        else:
            return self.__add_values_as_2d_property_plot(
                ax, values_x, values_y, values_color, **kwargs)

    def add_values_as_line_histogram(self,
                                     ax,
                                     value_name,
                                     value_name_weights=None,
                                     do_conversion=True,
                                     included_field_lines_finder=None,
                                     included_points_finder=None,
                                     mode='instant',
                                     **kwargs):

        if mode == 'load':
            values, weights = (None, None)
        else:
            values, weights = self.get_scalar_values(
                value_name,
                value_name_weights,
                included_field_lines_finder=included_field_lines_finder,
                included_points_finder=included_points_finder)

            values = self._convert_values(value_name, values, do_conversion)
            if weights is not None:
                weights = self._convert_values(value_name_weights, weights,
                                               do_conversion)

        return self.__add_values_as_line_histogram(ax,
                                                   values,
                                                   weights,
                                                   mode=mode,
                                                   **kwargs)

    def add_values_as_line_histogram_difference(
            self,
            ax,
            value_names,
            value_names_weights=None,
            do_conversion=True,
            included_field_lines_finder=None,
            included_points_finder=None,
            mode='instant',
            **kwargs):

        if mode == 'load':
            left_values, right_values, left_weights, right_weights = (None,
                                                                      None,
                                                                      None,
                                                                      None)
        else:
            left_value_name, right_value_name = value_names
            left_value_name_weights, right_value_name_weights = value_names_weights

            left_values, left_weights = self.get_scalar_values(
                left_value_name,
                left_value_name_weights,
                included_field_lines_finder=included_field_lines_finder,
                included_points_finder=included_points_finder)

            right_values, right_weights = self.get_scalar_values(
                right_value_name,
                right_value_name_weights,
                included_field_lines_finder=included_field_lines_finder,
                included_points_finder=included_points_finder)

            left_values = self._convert_values(left_value_name, left_values,
                                               do_conversion)
            if left_weights is not None:
                left_weights = self._convert_values(left_value_name_weights,
                                                    left_weights,
                                                    do_conversion)

            right_values = self._convert_values(right_value_name, right_values,
                                                do_conversion)
            if right_weights is not None:
                right_weights = self._convert_values(right_value_name_weights,
                                                     right_weights,
                                                     do_conversion)

        return self.__add_values_as_line_histogram_difference(
            ax, (left_values, right_values), (left_weights, right_weights),
            mode=mode,
            **kwargs)

    def add_values_as_2d_histogram_image(self,
                                         ax,
                                         value_name_x,
                                         value_name_y,
                                         value_name_weights=None,
                                         weight_scale=None,
                                         do_conversion=True,
                                         included_field_lines_finder=None,
                                         included_points_finder=None,
                                         mode='instant',
                                         **kwargs):

        if mode == 'load':
            values_x, values_y, weights = (None, None, None)
        else:
            values_x, values_y, weights = self.get_scalar_values(
                value_name_x,
                value_name_y,
                value_name_weights,
                included_field_lines_finder=included_field_lines_finder,
                included_points_finder=included_points_finder)

            values_x = self._convert_values(value_name_x, values_x,
                                            do_conversion)
            values_y = self._convert_values(value_name_y, values_y,
                                            do_conversion)
            if weights is not None:
                weights = self._convert_values(value_name_weights, weights,
                                               do_conversion)
                if weight_scale is not None:
                    weights *= weight_scale

        return self.__add_values_as_2d_histogram_image(ax,
                                                       values_x,
                                                       values_y,
                                                       weights,
                                                       mode=mode,
                                                       **kwargs)

    def add_values_as_2d_histogram_contour(self,
                                           ax,
                                           value_name_x,
                                           value_name_y,
                                           value_name_weights=None,
                                           do_conversion=True,
                                           included_field_lines_finder=None,
                                           included_points_finder=None,
                                           **kwargs):

        values_x, values_y, weights = self.get_scalar_values(
            value_name_x,
            value_name_y,
            value_name_weights,
            included_field_lines_finder=included_field_lines_finder,
            included_points_finder=included_points_finder)

        values_x = self._convert_values(value_name_x, values_x, do_conversion)
        values_y = self._convert_values(value_name_y, values_y, do_conversion)
        if weights is not None:
            weights = self._convert_values(value_name_weights, weights,
                                           do_conversion)

        return self.__add_values_as_2d_histogram_contour(
            ax, values_x, values_y, weights, **kwargs)

    def add_values_as_2d_histogram_difference_image(
            self,
            ax,
            value_names_x,
            value_names_y,
            value_names_weights=(None, None),
            weight_scale=None,
            do_conversion=True,
            included_field_lines_finder=None,
            included_points_finder=None,
            mode='instant',
            **kwargs):

        if mode == 'load':
            left_values_x, right_values_x, left_values_y, right_values_y, left_weights, right_weights = (
                None, None, None, None, None, None)
        else:
            left_value_name_x, right_value_name_x = value_names_x
            left_value_name_y, right_value_name_y = value_names_y
            left_value_name_weights, right_value_name_weights = value_names_weights

            left_values_x, left_values_y, left_weights = self.get_scalar_values(
                left_value_name_x,
                left_value_name_y,
                left_value_name_weights,
                included_field_lines_finder=included_field_lines_finder,
                included_points_finder=included_points_finder)

            right_values_x, right_values_y, right_weights = self.get_scalar_values(
                right_value_name_x,
                right_value_name_y,
                right_value_name_weights,
                included_field_lines_finder=included_field_lines_finder,
                included_points_finder=included_points_finder)

            left_values_x = self._convert_values(left_value_name_x,
                                                 left_values_x, do_conversion)
            left_values_y = self._convert_values(left_value_name_y,
                                                 left_values_y, do_conversion)
            if left_weights is not None:
                left_weights = self._convert_values(left_value_name_weights,
                                                    left_weights,
                                                    do_conversion)
                if weight_scale is not None:
                    left_weights *= weight_scale

            right_values_x = self._convert_values(right_value_name_x,
                                                  right_values_x,
                                                  do_conversion)
            right_values_y = self._convert_values(right_value_name_y,
                                                  right_values_y,
                                                  do_conversion)
            if right_weights is not None:
                right_weights = self._convert_values(right_value_name_weights,
                                                     right_weights,
                                                     do_conversion)
                if weight_scale is not None:
                    right_weights *= weight_scale

        return self.__add_values_as_2d_histogram_difference_image(
            ax, (left_values_x, right_values_x),
            (left_values_y, right_values_y), (left_weights, right_weights),
            mode=mode,
            **kwargs)

    def has_fixed_scalar_values(self, value_name):
        return value_name in self.fixed_scalar_values

    def has_varying_scalar_values(self, value_name):
        return value_name in self.varying_scalar_values

    def get_domain_bounds(self):
        return self.bounds_x, self.bounds_y, self.bounds_z

    def get_number_of_field_lines(self):
        return self.number_of_field_lines

    def get_fixed_scalar_values(self,
                                value_name,
                                included_field_line_indices=None,
                                included_points_finder=None):
        assert included_points_finder is None
        if included_field_line_indices is None:
            values = self.fixed_scalar_values[value_name]
        else:
            values = self.fixed_scalar_values[value_name][
                included_field_line_indices]
        return values if len(values) > 0 else None

    def get_fixed_vector_values(self, value_name):
        return self.fixed_vector_values[value_name]

    def get_varying_scalar_values(self, value_name):
        return self.varying_scalar_values[value_name]

    def get_concatenated_varying_scalar_values(
            self,
            value_name,
            included_field_line_indices=None,
            included_points_finder=None,
            points_processor=None):
        if points_processor is not None:
            indices = range(
                self.get_number_of_field_lines()
            ) if included_field_line_indices is None else included_field_line_indices
            values = [
                points_processor(self.varying_scalar_values, value_name, i)
                for i in indices
            ]
        elif included_field_line_indices is None:
            values = self.varying_scalar_values[value_name]
        else:
            values = [
                self.varying_scalar_values[value_name][i]
                for i in included_field_line_indices
            ]

        if included_points_finder is not None:
            values = list(
                map(
                    lambda v, field_line_idx: v[included_points_finder(
                        self.varying_scalar_values, field_line_idx)], values,
                    range(self.get_number_of_field_lines())
                    if included_field_line_indices is None else
                    included_field_line_indices))

        return np.concatenate(values) if len(values) > 0 else None

    def get_varying_vector_values(self, value_name):
        return self.varying_vector_values[value_name]

    def get_scalar_values(self,
                          *value_names,
                          included_field_lines_finder=None,
                          included_points_finder=None,
                          varying_points_processor=None):
        assert len(value_names) > 0 and value_names[0] is not None

        included_field_line_indices = None if included_field_lines_finder is None else included_field_lines_finder(
            self.fixed_scalar_values, self.varying_scalar_values)

        value_name = value_names[0]
        if self.has_fixed_scalar_values(value_name):
            assert not self.has_varying_scalar_values(
                value_name), 'Ambiguous value name {}'.format(value_name)
            getter = self.get_fixed_scalar_values
        else:
            assert self.has_varying_scalar_values(
                value_name), 'No values for value name {}'.format(value_name)
            getter = lambda value_name, included_field_line_indices, included_points_finder: self.get_concatenated_varying_scalar_values(
                value_name,
                included_field_line_indices,
                included_points_finder,
                points_processor=varying_points_processor)

        return tuple([(None if value_name is None else getter(
            value_name, included_field_line_indices, included_points_finder))
                      for value_name in value_names])

    def has_param(self, param_name):
        return param_name in self.params

    def get_param(self, param_name):
        return self.params[param_name]

    def process_value_description(self, value_name, value_description):
        if value_description is None:
            if value_name in self.VALUE_DESCRIPTIONS:
                return self.VALUE_DESCRIPTIONS[value_name]
            else:
                return value_name
        else:
            return value_description

    def _convert_values(self, value_name, values, do_conversion):
        if do_conversion and value_name in self.VALUE_UNIT_CONVERTERS:
            return self.VALUE_UNIT_CONVERTERS[value_name](values)
        else:
            return values

    def _derive_quantities(self, derived_quantities):

        for value_name in filter(
                lambda name: name[-1] == '0' and self.
                has_varying_scalar_values(name[:-1]) and not self.
                has_fixed_scalar_values(name[:-1]), derived_quantities):

            self.fixed_scalar_values[value_name] = np.asfarray([
                values[0]
                for values in self.get_varying_scalar_values(value_name[:-1])
            ])

    def __find_nonwrapping_segments(self,
                                    path_x,
                                    path_y,
                                    path_z,
                                    threshold=20.0):
        step_lengths = np.sqrt(
            np.diff(path_x)**2 + np.diff(path_y)**2 + np.diff(path_z)**2)
        wrap_indices = np.where(
            step_lengths > threshold*np.mean(step_lengths))[0]
        if wrap_indices.size > 0:
            wrap_indices += 1
            return np.split(path_x, wrap_indices), \
                np.split(path_y, wrap_indices), \
                np.split(path_z, wrap_indices)
        else:
            return [path_x], [path_y], [path_z]

    def __add_values_as_2d_property_plot(self,
                                         ax,
                                         values_x,
                                         values_y,
                                         values_color,
                                         force_scatter=False,
                                         symlog=False,
                                         linthresh=np.inf,
                                         linscale=1.0,
                                         log=False,
                                         vmin=None,
                                         vmax=None,
                                         color='k',
                                         cmap_name='viridis',
                                         cmap_bad_color='w',
                                         s=1.0,
                                         lw=1.0,
                                         marker='o',
                                         edgecolors='none',
                                         alpha=1.0,
                                         relative_alpha=True,
                                         rasterized=None):

        if values_color is None and not force_scatter:
            ax.plot(values_x,
                    values_y,
                    c=color,
                    lw=lw,
                    alpha=alpha,
                    rasterized=rasterized)
            return (None, None)

        else:

            if values_color is None:
                c = color
                norm = None
                cmap = None
            else:
                if vmin is None:
                    vmin = np.nanmin(values_color)
                if vmax is None:
                    vmax = np.nanmax(values_color)

                if symlog:
                    norm = plotting.get_symlog_normalizer(vmin,
                                                          vmax,
                                                          linthresh,
                                                          linscale=linscale)
                else:
                    norm = plotting.get_normalizer(vmin, vmax, log=log)

                cmap = plotting.get_cmap(cmap_name, bad_color=cmap_bad_color)

                c = plotting.colors_from_values(values_color,
                                                norm,
                                                cmap,
                                                alpha=alpha,
                                                relative_alpha=relative_alpha)

            ax.scatter(values_x,
                       values_y,
                       c=c,
                       s=s,
                       marker=marker,
                       edgecolors=edgecolors,
                       alpha=alpha,
                       rasterized=rasterized)

            return (norm, cmap)

    def __add_values_as_line_histogram(self,
                                       ax,
                                       values,
                                       weights,
                                       bins='auto',
                                       vmin=None,
                                       vmax=None,
                                       weighted_average=False,
                                       decide_bins_in_log_space=False,
                                       density=False,
                                       scatter=False,
                                       ds='steps-pre',
                                       c='k',
                                       lw=1.0,
                                       ls='-',
                                       s=1.0,
                                       alpha=1.0,
                                       legend_label=None,
                                       zorder=2,
                                       rasterized=None,
                                       mode='instant',
                                       save_path=None):

        if mode == 'load':
            data = np.load(save_path)
            hist = data['hist']
            bin_edges = data['bin_edges']
            bin_centers = data['bin_centers']
            print('Loaded {}'.format(save_path))
        else:
            hist, bin_edges, bin_centers = plotting.compute_histogram(
                values,
                weights=weights,
                bins=bins,
                vmin=vmin,
                vmax=vmax,
                decide_bins_in_log_space=decide_bins_in_log_space,
                weighted_average=weighted_average,
                density=density)

        if mode == 'save':
            np.savez_compressed(save_path,
                                hist=hist,
                                bin_edges=bin_edges,
                                bin_centers=bin_centers)
            print('Saved {}'.format(save_path))
            return None
        else:
            if scatter:
                return ax.scatter(bin_centers,
                                  hist,
                                  c=c,
                                  s=s,
                                  alpha=alpha,
                                  zorder=zorder,
                                  rasterized=rasterized)
            else:
                if ds == 'steps-pre':
                    return ax.step(bin_edges[:-1],
                                   hist,
                                   c=c,
                                   ls=ls,
                                   lw=lw,
                                   alpha=alpha,
                                   label=legend_label,
                                   zorder=zorder,
                                   rasterized=rasterized)[0]
                else:
                    return ax.plot(bin_centers,
                                   hist,
                                   c=c,
                                   ls=ls,
                                   lw=lw,
                                   alpha=alpha,
                                   label=legend_label,
                                   zorder=zorder,
                                   rasterized=rasterized)[0]

    def __add_values_as_line_histogram_difference(
            self,
            ax,
            values,
            weights,
            bins=500,
            vmin=None,
            vmax=None,
            decide_bins_in_log_space=False,
            scatter=False,
            c='k',
            lw=1.0,
            s=1.0,
            alpha=1.0,
            legend_label=None,
            rasterized=None,
            mode='instant',
            save_path=None):

        if mode == 'load':
            data = np.load(save_path)
            hist = data['hist']
            bin_edges = data['bin_edges']
            bin_centers = data['bin_centers']
            print('Loaded {}'.format(save_path))
        else:
            hist, bin_edges, bin_centers = plotting.compute_histogram_difference(
                values, weights, vmin, vmax, bins, decide_bins_in_log_space)

        if mode == 'save':
            np.savez_compressed(save_path,
                                hist=hist,
                                bin_edges=bin_edges,
                                bin_centers=bin_centers)
            print('Saved {}'.format(save_path))
            return None
        else:
            if scatter:
                sc = ax.scatter(bin_centers,
                                hist,
                                c=c,
                                s=s,
                                alpha=alpha,
                                rasterized=rasterized)
                if decide_bins_in_log_space:
                    plotting.set_2d_plot_extent(ax, bin_centers[0],
                                                bin_centers[-1])
                return sc
            else:
                return ax.step(bin_edges[:-1],
                               hist,
                               where='pre',
                               c=c,
                               lw=lw,
                               alpha=alpha,
                               label=legend_label,
                               rasterized=rasterized)[0]

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
                                           symlog=False,
                                           linthresh=np.inf,
                                           linscale=1.0,
                                           weighted_average=False,
                                           log=False,
                                           vmin=None,
                                           vmax=None,
                                           cmap_name='viridis',
                                           cmap_bad_color='w',
                                           rasterized=None,
                                           mode='instant',
                                           save_path=None):

        if mode == 'load':
            data = np.load(save_path)
            hist = data['hist']
            bin_edges_x = data['bin_edges_x']
            bin_edges_y = data['bin_edges_y']
            print('Loaded {}'.format(save_path))
        else:
            hist, bin_edges_x, bin_edges_y = plotting.compute_2d_histogram(
                values_x, values_y, weights, vmin_x, vmax_x, vmin_y, vmax_y,
                log_x, log_y, bins_x, bins_y, weighted_average)

        if mode == 'save':
            np.savez_compressed(save_path,
                                hist=hist,
                                bin_edges_x=bin_edges_x,
                                bin_edges_y=bin_edges_y)
            print('Saved {}'.format(save_path))
            return None
        else:
            if symlog:
                norm = plotting.get_symlog_normalizer(vmin,
                                                      vmax,
                                                      linthresh,
                                                      linscale=linscale)
            else:
                norm = plotting.get_normalizer(vmin, vmax, log=log)

            return ax.pcolormesh(*np.meshgrid(bin_edges_x, bin_edges_y),
                                 hist.T,
                                 norm=norm,
                                 vmin=vmin,
                                 vmax=vmax,
                                 cmap=plotting.get_cmap(
                                     cmap_name, bad_color=cmap_bad_color),
                                 rasterized=rasterized)

    def __add_values_as_2d_histogram_contour(self,
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
                                             weighted_average=True,
                                             log=False,
                                             vmin=None,
                                             vmax=None,
                                             gaussian_filter_sigma=None,
                                             filled=False,
                                             extend='neither',
                                             levels=None,
                                             colors=None,
                                             cmap_name='viridis',
                                             linewidths=1.0,
                                             linestyles='solid',
                                             alpha=1.0,
                                             label_levels=None,
                                             fontsize=None,
                                             inline=True,
                                             inline_spacing=5,
                                             fmt='%1.3f',
                                             rightside_up=True,
                                             rasterized=None):

        hist, bin_edges_x, bin_edges_y = plotting.compute_2d_histogram(
            values_x, values_y, weights, vmin_x, vmax_x, vmin_y, vmax_y, log_x,
            log_y, bins_x, bins_y, weighted_average)

        bin_centers_x = 0.5*(bin_edges_x[1:] + bin_edges_x[:-1])
        bin_centers_y = 0.5*(bin_edges_y[1:] + bin_edges_y[:-1])

        if gaussian_filter_sigma is not None:
            hist = ndimage.gaussian_filter(hist, gaussian_filter_sigma)

        contour_func = ax.contourf if filled else ax.contour

        cs = contour_func(
            *np.meshgrid(bin_centers_x, bin_centers_y),
            hist.T,
            levels=levels,
            colors=colors,
            cmap=None if colors is not None else plotting.get_cmap(cmap_name),
            norm=plotting.get_normalizer(vmin, vmax, log=log),
            linewidths=linewidths,
            linestyles=linestyles,
            alpha=alpha,
            extend=extend,
            rasterized=rasterized)

        ax.clabel(cs,
                  cs.levels if label_levels is None else label_levels,
                  fontsize=fontsize,
                  inline=inline,
                  inline_spacing=inline_spacing,
                  fmt=fmt,
                  rightside_up=rightside_up)

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
                                                      cmap_name='viridis',
                                                      cmap_bad_color='w',
                                                      rasterized=None,
                                                      mode='instant',
                                                      save_path=None):

        if mode == 'load':
            data = np.load(save_path)
            hist_diff = data['hist_diff']
            bin_edges_x = data['bin_edges_x']
            bin_edges_y = data['bin_edges_y']
            print('Loaded {}'.format(save_path))
        else:
            hist_diff, bin_edges_x, bin_edges_y = plotting.compute_2d_histogram_difference(
                values_x, values_y, weights, vmin_x, vmax_x, vmin_y, vmax_y,
                log_x, log_y, bins_x, bins_y)

        if mode == 'save':
            np.savez_compressed(save_path,
                                hist_diff=hist_diff,
                                bin_edges_x=bin_edges_x,
                                bin_edges_y=bin_edges_y)
            print('Saved {}'.format(save_path))
            return None
        else:
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
                                 cmap=plotting.get_cmap(
                                     cmap_name, bad_color=cmap_bad_color),
                                 rasterized=rasterized)


def find_field_lines_passing_near_point(point, max_distance,
                                        initial_position_bounds,
                                        fixed_scalar_values,
                                        varying_scalar_values):
    x_lims, y_lims, z_lims = initial_position_bounds
    return list(
        set([
            i for i, (x, y, z) in enumerate(
                zip(varying_scalar_values['x'], varying_scalar_values['y'],
                    varying_scalar_values['z']))
            if np.any((x - point[0])**2 + (y - point[1])**2 +
                      (z - point[2])**2 <= max_distance**2)
        ]).intersection([
            i for i, (x, y, z) in enumerate(
                zip(fixed_scalar_values['x0'], fixed_scalar_values['y0'],
                    fixed_scalar_values['z0']))
            if x >= x_lims[0] and x <= x_lims[1] and y >= y_lims[0]
            and y <= y_lims[1] and z >= z_lims[0] and z <= z_lims[1]
        ]))


def find_field_lines_starting_in_coords(x_coords,
                                        y_coords,
                                        z_coords,
                                        fixed_scalar_values,
                                        max_distance=1e-5):
    return [
        i for i, (x, y, z) in enumerate(
            zip(fixed_scalar_values['x0'], fixed_scalar_values['y0'],
                fixed_scalar_values['z0']))
        if np.any((x - x_coords)**2 + (y - y_coords)**2 +
                  (z - z_coords)**2 <= max_distance**2)
    ]


def find_field_lines_contained_in_box(x_lims, y_lims, z_lims,
                                      varying_scalar_values):
    return [
        i for i, (x, y, z) in enumerate(
            zip(varying_scalar_values['x'], varying_scalar_values['y'],
                varying_scalar_values['z']))
        if np.all((x >= x_lims[0])*(x <= x_lims[1])*(y >= y_lims[0])*
                  (y <= y_lims[1])*(z >= z_lims[0])*(z <= z_lims[1]))
    ]


def find_field_line_points_below_depth(min_depth, varying_scalar_values,
                                       field_line_idx):
    return varying_scalar_values['z'][field_line_idx] > min_depth


def find_field_line_points_between_depths(min_depth, max_depth,
                                          varying_scalar_values,
                                          field_line_idx):
    return np.logical_and(
        varying_scalar_values['z'][field_line_idx] >= min_depth,
        varying_scalar_values['z'][field_line_idx] < max_depth)


def find_field_line_points_after_distance(min_distance, varying_scalar_values,
                                          field_line_idx):
    return varying_scalar_values['s'][field_line_idx] > min_distance


def find_field_line_points_above_density(min_density, varying_scalar_values,
                                         field_line_idx):
    return varying_scalar_values['r'][field_line_idx] > min_density/units.U_R


def find_field_line_points_below_temperature(max_temperature,
                                             varying_scalar_values,
                                             field_line_idx):
    return varying_scalar_values['tg'][field_line_idx] < max_temperature


def find_field_line_point_at_max_depth(varying_scalar_values, field_line_idx):
    return np.argmax(varying_scalar_values['z'][field_line_idx])


def find_last_field_line_point(_varying_scalar_values, _field_line_idx):
    return slice(-1, None, None)


def resample_varying_points(resample_coords,
                            resample_coord_name,
                            resample_val_names,
                            varying_scalar_values,
                            value_name,
                            field_line_idx,
                            log_x=False,
                            log_y=False,
                            start_idx_finder=lambda x: 0):
    if value_name in resample_val_names:
        x = varying_scalar_values[resample_coord_name][field_line_idx]
        y = varying_scalar_values[value_name][field_line_idx]
        if log_x:
            x = np.log10(x)
            coords = np.log10(resample_coords)
        else:
            coords = resample_coords
        start_idx = start_idx_finder(x)
        x = x[start_idx:]
        y = y[start_idx:]
        if hasattr(log_y, '__iter__'):
            log_y = log_y[resample_val_names.index(value_name)]
        if log_y:
            y = np.log10(y)
            vals = 10**interpolate.interp1d(x,
                                            y,
                                            bounds_error=False,
                                            copy=False)(coords)
        else:
            vals = interpolate.interp1d(x, y, bounds_error=False,
                                        copy=False)(coords)
        return vals
    elif value_name == resample_coord_name:
        return resample_coords
    else:
        return varying_scalar_values[value_name][field_line_idx]


def plot_field_lines(field_line_set,
                     value_name=None,
                     fig=None,
                     ax=None,
                     value_description=None,
                     title=None,
                     hide_grid=False,
                     render=True,
                     output_path=None,
                     **kwargs):

    if fig is None or ax is None:
        fig, ax = plotting.create_3d_plot()

    plotting.set_3d_plot_extent(ax, *field_line_set.get_domain_bounds())
    plotting.set_3d_axis_labels(ax, field_line_set.VALUE_DESCRIPTIONS['x'],
                                field_line_set.VALUE_DESCRIPTIONS['y'],
                                field_line_set.VALUE_DESCRIPTIONS['z'])
    ax.invert_zaxis()
    if hide_grid:
        ax.set_axis_off()

    if value_name is None:
        field_line_set.add_to_3d_plot_with_single_color(ax, **kwargs)
    else:
        norm, cmap = field_line_set.add_values_to_3d_plot(
            ax, value_name, **kwargs)

        plotting.add_3d_colorbar(
            fig,
            norm,
            cmap,
            label=field_line_set.process_value_description(
                value_name, value_description))

    if title is not None:
        ax.set_title(title)

    if render:
        plotting.render(fig, output_path=output_path)


def plot_field_line_properties(field_line_set,
                               value_name_x=None,
                               value_name_y=None,
                               value_name_color=None,
                               fig=None,
                               ax=None,
                               figure_width=6.0,
                               figure_aspect=4.0/3.0,
                               invert_xaxis=False,
                               invert_yaxis=False,
                               value_description_x=None,
                               value_description_y=None,
                               value_description_color=None,
                               xlabel_color='k',
                               ylabel_color='k',
                               colorbar_loc='right',
                               colorbar_pad=0.05,
                               no_colorbar=False,
                               title=None,
                               render=True,
                               output_path=None,
                               log_x=False,
                               log_y=False,
                               vmin_x=None,
                               vmax_x=None,
                               vmin_y=None,
                               vmax_y=None,
                               extra_artists=None,
                               mode='instant',
                               save_path=None,
                               **kwargs):

    if (fig is None or ax is None) and mode != 'save':
        fig, ax = plotting.create_2d_subplots(width=figure_width,
                                              aspect_ratio=figure_aspect)

    if mode != 'instant' and save_path is None and output_path is not None:
        save_path = '{}.npz'.format('.'.join(output_path.split('.')[:-1]))

    norm, cmap = field_line_set.add_values_as_2d_property_plot(
        ax,
        value_name_x,
        value_name_y,
        value_name_color=value_name_color,
        mode=mode,
        save_path=save_path,
        **kwargs)

    if mode == 'save':
        return

    if extra_artists is not None:
        for artist in extra_artists:
            ax.add_artist(artist)

    if log_x:
        ax.set_xscale('log')
    if log_y:
        ax.set_yscale('log')

    plotting.set_2d_plot_extent(ax, (vmin_x, vmax_x), (vmin_y, vmax_y))

    if invert_xaxis:
        ax.invert_xaxis()
    if invert_yaxis:
        ax.invert_yaxis()

    if norm is not None and cmap is not None and not no_colorbar:
        plotting.add_2d_colorbar_from_cmap_and_norm(
            fig,
            ax,
            norm,
            cmap,
            loc=colorbar_loc,
            pad=colorbar_pad,
            label=field_line_set.process_value_description(
                value_name_color, value_description_color))

    plotting.set_2d_axis_labels(
        ax,
        field_line_set.process_value_description(value_name_x,
                                                 value_description_x),
        field_line_set.process_value_description(value_name_y,
                                                 value_description_y),
        xcolor=xlabel_color,
        ycolor=ylabel_color)

    if title is not None:
        ax.set_title(title)

    if render:
        plotting.render(fig, output_path=output_path)


def plot_field_line_value_histogram(field_line_set,
                                    value_name,
                                    value_name_weights=None,
                                    fig=None,
                                    ax=None,
                                    figure_width=6.0,
                                    figure_aspect=4.0/3.0,
                                    invert_xaxis=False,
                                    invert_yaxis=False,
                                    value_description=None,
                                    value_description_weights=None,
                                    set_axis_labels=True,
                                    legend_loc=None,
                                    title=None,
                                    render=True,
                                    output_path=None,
                                    log_x=False,
                                    log_y=False,
                                    symlog_y=False,
                                    linthresh_y=np.inf,
                                    vmin_x=None,
                                    vmax_x=None,
                                    vmin_y=None,
                                    vmax_y=None,
                                    extra_artists=None,
                                    mode='instant',
                                    save_path=None,
                                    **kwargs):

    if (fig is None or ax is None) and mode != 'save':
        fig, ax = plotting.create_2d_subplots(width=figure_width,
                                              aspect_ratio=figure_aspect)

    if mode != 'instant' and save_path is None and output_path is not None:
        save_path = '{}.npz'.format('.'.join(output_path.split('.')[:-1]))

    handle = field_line_set.add_values_as_line_histogram(
        ax,
        value_name,
        value_name_weights=value_name_weights,
        decide_bins_in_log_space=log_x,
        mode=mode,
        save_path=save_path,
        **kwargs)

    if mode == 'save':
        return

    if extra_artists is not None:
        for artist in extra_artists:
            ax.add_artist(artist)

    if log_x:
        ax.set_xscale('log')
    if log_y:
        ax.set_yscale('log')

    if symlog_y:
        ax.set_yscale('symlog', linthreshy=linthresh_y)

    plotting.set_2d_plot_extent(ax, (vmin_x, vmax_x), (vmin_y, vmax_y))

    if invert_xaxis:
        ax.invert_xaxis()
    if invert_yaxis:
        ax.invert_yaxis()

    if set_axis_labels:
        plotting.set_2d_axis_labels(
            ax,
            field_line_set.process_value_description(value_name,
                                                     value_description),
            ('Probability density' if kwargs.get('density', False) else
             'Number of values') if value_name_weights is None else
            field_line_set.process_value_description(
                value_name_weights, value_description_weights))

    if legend_loc is not None:
        if extra_artists is None:
            ax.legend(loc=legend_loc)
        else:
            ax.legend([handle] + extra_artists, [handle.get_label()] +
                      [artist.get_label() for artist in extra_artists],
                      loc=legend_loc)

    if title is not None:
        ax.set_title(title)

    if render:
        plotting.render(fig, output_path=output_path)


def plot_field_line_value_histogram_difference(field_line_set,
                                               value_names,
                                               value_names_weights=None,
                                               fig=None,
                                               ax=None,
                                               figure_width=6.0,
                                               figure_aspect=4.0/3.0,
                                               invert_xaxis=False,
                                               invert_yaxis=False,
                                               value_description=None,
                                               value_description_weights=None,
                                               legend_loc=None,
                                               title=None,
                                               render=True,
                                               output_path=None,
                                               log_x=False,
                                               symlog_y=False,
                                               linethresh_y=np.inf,
                                               vmin_x=None,
                                               vmax_x=None,
                                               vmin_y=None,
                                               vmax_y=None,
                                               extra_artists=None,
                                               mode='instant',
                                               save_path=None,
                                               **kwargs):

    if (fig is None or ax is None) and mode != 'save':
        fig, ax = plotting.create_2d_subplots(width=figure_width,
                                              aspect_ratio=figure_aspect)

    if mode != 'instant' and save_path is None and output_path is not None:
        save_path = '{}.npz'.format('.'.join(output_path.split('.')[:-1]))

    handle = field_line_set.add_values_as_line_histogram_difference(
        ax,
        value_names,
        value_names_weights=value_names_weights,
        decide_bins_in_log_space=log_x,
        mode=mode,
        save_path=save_path,
        **kwargs)

    if mode == 'save':
        return

    if extra_artists is not None:
        for artist in extra_artists:
            ax.add_artist(artist)

    if log_x:
        ax.set_xscale('log')
    if symlog_y:
        ax.set_yscale('symlog', linthreshy=linethresh_y)

    plotting.set_2d_plot_extent(ax, (vmin_x, vmax_x), (vmin_y, vmax_y))

    if invert_xaxis:
        ax.invert_xaxis()
    if invert_yaxis:
        ax.invert_yaxis()

    plotting.set_2d_axis_labels(
        ax,
        field_line_set.process_value_description(value_names[0],
                                                 value_description),
        'Number of values' if value_names_weights[0] is None
        else field_line_set.process_value_description(
            value_names_weights[0], value_description_weights))

    if legend_loc is not None:
        if extra_artists is None:
            ax.legend(loc=legend_loc)
        else:
            ax.legend([handle] + extra_artists, [handle.get_label()] +
                      [artist.get_label() for artist in extra_artists],
                      loc=legend_loc)

    if title is not None:
        ax.set_title(title)

    if render:
        plotting.render(fig, output_path=output_path)


def plot_field_line_value_2d_histogram(field_line_set,
                                       value_name_x,
                                       value_name_y,
                                       value_name_weights=None,
                                       fig=None,
                                       ax=None,
                                       figure_width=6.0,
                                       figure_aspect=4.0/3.0,
                                       invert_xaxis=False,
                                       invert_yaxis=False,
                                       aspect='auto',
                                       value_description_x=None,
                                       value_description_y=None,
                                       value_description_weights=None,
                                       title=None,
                                       render=True,
                                       output_path=None,
                                       log_x=False,
                                       log_y=False,
                                       contour_kwargs={},
                                       extra_artists=None,
                                       mode='instant',
                                       save_path=None,
                                       **kwargs):

    if (fig is None or ax is None) and mode != 'save':
        fig, ax = plotting.create_2d_subplots(width=figure_width,
                                              aspect_ratio=figure_aspect)

    if mode != 'instant' and save_path is None and output_path is not None:
        save_path = '{}.npz'.format('.'.join(output_path.split('.')[:-1]))

    im = field_line_set.add_values_as_2d_histogram_image(
        ax,
        value_name_x,
        value_name_y,
        value_name_weights=value_name_weights,
        log_x=log_x,
        log_y=log_y,
        mode=mode,
        save_path=save_path,
        **kwargs)

    if mode == 'save':
        return

    if contour_kwargs:
        contour_field_line_set = contour_kwargs.pop('dataset', field_line_set)
        contour_value_name_x = contour_kwargs.pop('value_name_x', value_name_x)
        contour_value_name_y = contour_kwargs.pop('value_name_y', value_name_y)
        contour_kwargs['log_x'] = contour_kwargs.pop('log_x', log_x)
        contour_kwargs['log_y'] = contour_kwargs.pop('log_y', log_y)
        contour_field_line_set.add_values_as_2d_histogram_contour(
            ax, contour_value_name_x, contour_value_name_y, **contour_kwargs)

    if extra_artists is not None:
        for artist in extra_artists:
            ax.add_artist(artist)

    if invert_xaxis:
        ax.invert_xaxis()
    if invert_yaxis:
        ax.invert_yaxis()

    ax.set_aspect(aspect)

    plotting.set_2d_axis_labels(
        ax, '{}{}'.format(
            r'$\log_{10}$ ' if log_x else '',
            field_line_set.process_value_description(value_name_x,
                                                     value_description_x)),
        '{}{}'.format(
            r'$\log_{10}$ ' if log_y else '',
            field_line_set.process_value_description(value_name_y,
                                                     value_description_y)))
    plotting.add_2d_colorbar(
        fig,
        ax,
        im,
        label=('Number of values' if value_name_weights is None else
               field_line_set.process_value_description(
                   value_name_weights, value_description_weights)))

    if title is not None:
        ax.set_title(title)

    if render:
        plotting.render(fig, output_path=output_path)


def plot_field_line_value_2d_histogram_difference(
        field_line_set,
        value_names_x,
        value_names_y,
        value_names_weights=None,
        fig=None,
        ax=None,
        figure_width=6.0,
        figure_aspect=4.0/3.0,
        invert_xaxis=False,
        invert_yaxis=False,
        aspect='auto',
        value_description_x=None,
        value_description_y=None,
        value_description_weights=None,
        title=None,
        render=True,
        output_path=None,
        log_x=False,
        log_y=False,
        contour_kwargs={},
        extra_artists=None,
        mode='instant',
        save_path=None,
        **kwargs):

    if (fig is None or ax is None) and mode != 'save':
        fig, ax = plotting.create_2d_subplots(width=figure_width,
                                              aspect_ratio=figure_aspect)

    if mode != 'instant' and save_path is None and output_path is not None:
        save_path = '{}.npz'.format('.'.join(output_path.split('.')[:-1]))

    im = field_line_set.add_values_as_2d_histogram_difference_image(
        ax,
        value_names_x,
        value_names_y,
        value_names_weights=value_names_weights,
        log_x=log_x,
        log_y=log_y,
        mode=mode,
        save_path=save_path,
        **kwargs)

    if mode == 'save':
        return

    if contour_kwargs:
        contour_field_line_set = contour_kwargs.pop('dataset', field_line_set)
        contour_value_name_x = contour_kwargs.pop('value_name_x',
                                                  value_names_x[0])
        contour_value_name_y = contour_kwargs.pop('value_name_y',
                                                  value_names_y[0])
        contour_kwargs['log_x'] = contour_kwargs.pop('log_x', log_x)
        contour_kwargs['log_y'] = contour_kwargs.pop('log_y', log_y)
        contour_field_line_set.add_values_as_2d_histogram_contour(
            ax, contour_value_name_x, contour_value_name_y, **contour_kwargs)

    if extra_artists is not None:
        for artist in extra_artists:
            ax.add_artist(artist)

    if invert_xaxis:
        ax.invert_xaxis()
    if invert_yaxis:
        ax.invert_yaxis()

    ax.set_aspect(aspect)

    plotting.set_2d_axis_labels(
        ax, '{}{}'.format(
            r'$\log_{10}$ ' if log_x else '',
            field_line_set.process_value_description(value_names_x[0],
                                                     value_description_x)),
        '{}{}'.format(
            r'$\log_{10}$ ' if log_y else '',
            field_line_set.process_value_description(value_names_y[0],
                                                     value_description_y)))
    plotting.add_2d_colorbar(
        fig,
        ax,
        im,
        label=('Number of values' if value_names_weights[0] is None else
               field_line_set.process_value_description(
                   value_names_weights[0], value_description_weights)))

    if title is not None:
        ax.set_title(title)

    if render:
        plotting.render(fig, output_path=output_path)


def plot_field_line_value_2d_histogram_comparison(field_line_set,
                                                  value_names_x, value_names_y,
                                                  value_names_weights,
                                                  **kwargs):

    fig, axes = plotting.create_2d_subplots(ncols=2, figsize=(8, 4))

    plot_field_line_value_2d_histogram(
        field_line_set,
        value_names_x[0],
        value_names_y[0],
        value_name_weights=value_names_weights[0],
        fig=fig,
        ax=axes[0],
        render=False,
        **kwargs)

    plot_field_line_value_2d_histogram(
        field_line_set,
        value_names_x[1],
        value_names_y[1],
        value_name_weights=value_names_weights[1],
        fig=fig,
        ax=axes[1],
        **kwargs)
