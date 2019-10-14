import numpy as np
from pathlib import Path
import bifrost_utils.plotting as plotting


class FieldLineSet3:

    VALUE_DESCRIPTIONS = {
        'x': r'$x$ [Mm]',
        'y': r'$y$ [Mm]',
        'z': r'$z$ [Mm]',
        'r': r'Mass density [10$^{-7}\:$g/cm$^3$]',
        'tg': 'Temperature [K]',
        'r0': r'Mass density [10$^{-7}\:$g/cm$^3$]',
        'tg0': 'Temperature [K]',
    }

    @staticmethod
    def from_file(file_path, derived_quantities=[], verbose=False):
        import bifrost_utils.reading as reading
        file_path = Path(file_path)
        extension = file_path.suffix
        if extension == '.pickle':
            field_line_set = reading.read_3d_field_line_set_from_combined_pickles(
                file_path,
                derived_quantities=derived_quantities,
                verbose=verbose)
        elif extension == '.fl':
            field_line_set = reading.read_3d_field_line_set_from_custom_binary_file(
                file_path,
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
                 derived_quantities=[],
                 verbose=False):
        assert all([upper >= lower for lower, upper in domain_bounds])
        assert isinstance(number_of_field_lines, int)
        assert isinstance(fixed_scalar_values, dict)
        assert isinstance(fixed_vector_values, dict)
        assert isinstance(varying_scalar_values, dict)
        assert isinstance(varying_vector_values, dict)
        self.bounds_x, self.bounds_y, self.bounds_z = tuple(domain_bounds)
        self.number_of_field_lines = number_of_field_lines
        self.fixed_scalar_values = fixed_scalar_values
        self.fixed_vector_values = fixed_vector_values
        self.varying_scalar_values = varying_scalar_values
        self.varying_vector_values = varying_vector_values
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
            paths_x = self.get_varying_scalar_values('x')
            paths_y = self.get_varying_scalar_values('y')
            paths_z = self.get_varying_scalar_values('z')
            for field_line_idx in range(self.get_number_of_field_lines()):
                for x, y, z in zip(*self.__find_nonwrapping_segments(
                        paths_x[field_line_idx], paths_y[field_line_idx],
                        paths_z[field_line_idx])):
                    ax.plot(x, y, z, c=c, lw=lw, alpha=alpha)

    def add_values_as_line_histogram(self,
                                     ax,
                                     value_name,
                                     value_name_weights=None,
                                     **kwargs):

        values, weights = self.get_scalar_values(value_name,
                                                 value_name_weights)

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
                                         **kwargs):

        values_x, values_y, weights = self.get_scalar_values(
            value_name_x, value_name_y, value_name_weights)

        return self.__add_values_as_2d_histogram_image(ax, values_x, values_y,
                                                       weights, **kwargs)

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

    def has_fixed_scalar_values(self, value_name):
        return value_name in self.fixed_scalar_values

    def has_varying_scalar_values(self, value_name):
        return value_name in self.varying_scalar_values

    def get_domain_bounds(self):
        return self.bounds_x, self.bounds_y, self.bounds_z

    def get_number_of_field_lines(self):
        return self.number_of_field_lines

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

    def process_value_description(self, value_name, value_description):
        if value_description is None:
            if value_name in self.VALUE_DESCRIPTIONS:
                return self.VALUE_DESCRIPTIONS[value_name]
            else:
                return value_name
        else:
            return value_description

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

        hist, bin_edges, bin_centers = plotting.compute_histogram(
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

        hist, bin_edges, bin_centers = plotting.compute_histogram_difference(
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

        hist, bin_edges_x, bin_edges_y = plotting.compute_2d_histogram(
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

        hist_diff, bin_edges_x, bin_edges_y = plotting.compute_2d_histogram_difference(
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


def plot_field_lines(field_line_set,
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

    plotting.set_3d_plot_extent(ax, *field_line_set.get_domain_bounds())
    plotting.set_3d_spatial_axis_labels(ax, unit='Mm')
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

    ax.set_title(title)

    if render:
        plotting.render(fig, output_path=output_path)


def plot_field_line_value_histogram(field_line_set,
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

    field_line_set.add_values_as_line_histogram(
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
        field_line_set.process_value_description(value_name,
                                                 value_description))
    ax.set_ylabel('Number of values' if value_name_weights is None else
                  field_line_set.process_value_description(
                      value_name_weights, value_description_weights))
    ax.set_title(title)

    if render:
        plotting.render(fig, output_path=output_path)


def plot_field_line_value_histogram_difference(field_line_set,
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

    field_line_set.add_values_as_line_histogram_difference(
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
        field_line_set.process_value_description(value_names[0],
                                                 value_description))
    ax.set_ylabel('Number of values' if value_names_weights[0] is None else
                  field_line_set.process_value_description(
                      value_names_weights[0], value_description_weights))
    ax.set_title(title)

    if render:
        plotting.render(fig, output_path=output_path)


def plot_field_line_value_2d_histogram(field_line_set,
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

    im = field_line_set.add_values_as_2d_histogram_image(
        ax,
        value_name_x,
        value_name_y,
        value_name_weights=value_name_weights,
        log_x=log_x,
        log_y=log_y,
        **kwargs)

    ax.set_xlabel('{}{}'.format(
        r'$\log_{10}$ ' if log_x else '',
        field_line_set.process_value_description(value_name_x,
                                                 value_description_x)))
    ax.set_ylabel('{}{}'.format(
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

    im = field_line_set.add_values_as_2d_histogram_difference_image(
        ax,
        value_names_x,
        value_names_y,
        value_names_weights=value_names_weights,
        log_x=log_x,
        log_y=log_y,
        **kwargs)

    ax.set_xlabel('{}{}'.format(
        r'$\log_{10}$ ' if log_x else '',
        field_line_set.process_value_description(value_names_x[0],
                                                 value_description_x)))
    ax.set_ylabel('{}{}'.format(
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
