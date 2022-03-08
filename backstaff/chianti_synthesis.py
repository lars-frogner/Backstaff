import os
import pathlib
import time
import re
import numpy as np
import scipy.interpolate
import scipy.integrate
from numba import njit
from tabulate import tabulate
import tempfile
import joblib
from joblib import Parallel, delayed

import ChiantiPy.tools.util as ch_util
import ChiantiPy.tools.io as ch_io
import ChiantiPy.tools.data as ch_data

try:
    import backstaff.units as units
    import backstaff.array_utils as array_utils
    import backstaff.fields as fields
    import backstaff.plotting as plotting
except ModuleNotFoundError:
    import units
    import backstaff.array_utils as array_utils
    import fields
    import plotting


class SpeciesRatiosCacher:
    def __init__(self):
        self.__ratios = {}

    def __call__(self, abundance_file, verbose=False):
        if abundance_file not in self.__ratios:
            self.__ratios[abundance_file] = SpeciesRatios(abundance_file,
                                                          verbose=verbose)
        return self.__ratios[abundance_file]


SPECIES_RATIOS = SpeciesRatiosCacher()


class SpeciesRatios:
    @staticmethod
    def default(verbose=False):
        return SPECIES_RATIOS('sun_coronal_2012_schmelz_ext', verbose=verbose)

    def __init__(self, abundance_file, verbose=False):
        self.__abundance_file = abundance_file
        self.verbose = verbose
        self.__compute_splines_for_proton_and_hydrogen_to_electron_ratios()

    @property
    def abundance_file(self):
        return self.__abundance_file

    def info(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def compute_proton_to_electron_ratios(self,
                                          temperatures,
                                          temperature_is_log=False):
        return scipy.interpolate.splev(
            temperatures if temperature_is_log else np.log10(temperatures),
            self.__proton_to_electron_ratio_spline,
            ext=1)

    def compute_hydrogen_to_electron_ratios(self,
                                            temperatures,
                                            temperature_is_log=False):
        return scipy.interpolate.splev(
            temperatures if temperature_is_log else np.log10(temperatures),
            self.__hydrogen_to_electron_ratio_spline,
            ext=0)

    def __compute_splines_for_proton_and_hydrogen_to_electron_ratios(self):
        self.info('Constructing splines for species ratios')
        start_time = time.time()

        all_abundances = np.asarray(
            ch_io.abundanceRead(
                abundancename=self.abundance_file)['abundance'])
        abundances = all_abundances[all_abundances > 0]

        electron_number = np.zeros(len(ch_data.IoneqAll['ioneqTemperature']))
        for i in range(abundances.size):
            for z in range(1, i + 2):
                electron_number += z * ch_data.IoneqAll['ioneqAll'][
                    i, z, :] * abundances[i]

        proton_to_electron_ratios = abundances[0] * ch_data.IoneqAll[
            'ioneqAll'][0, 1, :] / electron_number
        neutral_hydrogen_to_electron_ratios = abundances[0] * ch_data.IoneqAll[
            'ioneqAll'][0, 0, :] / electron_number
        hydrogen_to_electron_ratios = proton_to_electron_ratios + neutral_hydrogen_to_electron_ratios

        self.__proton_to_electron_ratio_spline = scipy.interpolate.splrep(
            np.log10(ch_data.IoneqAll['ioneqTemperature']),
            proton_to_electron_ratios,
            s=0)

        self.__hydrogen_to_electron_ratio_spline = scipy.interpolate.splrep(
            np.log10(ch_data.IoneqAll['ioneqTemperature']),
            hydrogen_to_electron_ratios,
            s=0)

        self.info(f'Took {time.time() - start_time:g} s')


class IonAtmosphere:
    def __init__(self,
                 temperatures,
                 electron_densities,
                 species_ratios=None,
                 radiation_temperature=None,
                 distance_from_center=None,
                 compute_proton_densities=True,
                 compute_hydrogen_densities=True,
                 verbose=False):
        self.__temperatures = temperatures  # [K]
        self.__electron_densities = electron_densities  # [1/cm^3]
        self.verbose = verbose
        self.__species_ratios = SpeciesRatios.default(
            verbose=verbose) if species_ratios is None else species_ratios
        self.__proton_densities = self.compute_proton_densities(
            self.temperatures,
            self.electron_densities) if compute_proton_densities else None
        self.__hydrogen_densities = self.compute_hydrogen_densities(
            self.temperatures,
            self.electron_densities) if compute_hydrogen_densities else None
        self.__radiation_temperature = radiation_temperature  # [K]
        self.__distance_from_center = distance_from_center  # [stellar radii]

    @property
    def temperatures(self):
        return self.__temperatures

    @property
    def electron_densities(self):
        return self.__electron_densities

    @property
    def proton_densities(self):
        return self.__proton_densities

    @property
    def hydrogen_densities(self):
        return self.__hydrogen_densities

    @property
    def n_values(self):
        return self.temperatures.size

    @property
    def species_ratios(self):
        return self.__species_ratios

    @property
    def abundance_file(self):
        return self.species_ratios.abundance_file

    @property
    def radiation_temperature(self):
        return self.__radiation_temperature

    @property
    def distance_from_center(self):
        return self.__distance_from_center

    def info(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def save(self, file_path):
        with open(file_path, 'wb') as f:
            joblib.dump(self, f)

    @staticmethod
    def load(file_path):
        with open(file_path, 'rb') as f:
            atmos = joblib.load(f)
        return atmos

    def compute_proton_densities(self, temperatures, electron_densities,
                                 **kwargs):
        self.info(f'Computing {temperatures.size} proton densities')
        start_time = time.time()

        proton_densities = self.species_ratios.compute_proton_to_electron_ratios(
            temperatures, **kwargs) * electron_densities

        self.info(f'Took {time.time() - start_time:g} s')
        return proton_densities

    def compute_hydrogen_densities(self, temperatures, electron_densities,
                                   **kwargs):
        self.info(f'Computing {temperatures.size} hydrogen ratios')
        start_time = time.time()

        hydrogen_densities = self.species_ratios.compute_hydrogen_to_electron_ratios(
            temperatures, **kwargs) * electron_densities

        self.info(f'Took {time.time() - start_time:g} s')
        return hydrogen_densities


class IonAtmosphere1D(IonAtmosphere):
    def __init__(self, z_coords, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(z_coords, np.ndarray) and z_coords.ndim == 1
        self.__z_coords = z_coords  # [Mm]

    @property
    def z_coords(self):
        return self.__z_coords


class IonAtmosphere3D(IonAtmosphere):
    def __init__(self, x_coords, y_coords, z_coords, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(x_coords, np.ndarray) and x_coords.ndim == 1
        assert isinstance(y_coords, np.ndarray) and y_coords.ndim == 1
        assert isinstance(z_coords, np.ndarray) and z_coords.ndim == 1
        self.__x_coords = x_coords  # [Mm]
        self.__y_coords = y_coords  # [Mm]
        self.__z_coords = z_coords  # [Mm]

    @property
    def x_coords(self):
        return self.__x_coords

    @property
    def y_coords(self):
        return self.__y_coords

    @property
    def z_coords(self):
        return self.__z_coords

    @property
    def shape(self):
        return (self.x_coords.size, self.y_coords.size, self.z_coords.size)

    def unravel(self, values):
        return values.reshape(values.shape[:-1] + self.shape)


class IonProperties:
    def __init__(self, ion_name, abundance):
        self.__set_ion_name(ion_name)
        self.__find_ionization_potential()
        if isinstance(abundance, str):
            abundance_file = abundance
            self.__read_abundance(abundance_file)
        else:
            self.__abundance = abundance

    @property
    def ion_name(self):
        return self.__ion_name

    @property
    def nuclear_charge(self):
        return self.__nuclear_charge

    @property
    def ionization_stage(self):
        return self.__ionization_stage

    @property
    def spectroscopic_name(self):
        return self.__spectroscopic_name

    @property
    def abundance(self):
        return self.__abundance

    def compute_thermal_line_variance(self, central_wavelength, temperature):
        return (units.KBOLTZMANN / self.compute_atomic_mass() *
                (central_wavelength / units.CLIGHT)**2) * temperature  # [cm^2]

    def compute_atomic_mass(self):
        return 2 * self.nuclear_charge * units.AMU  # [g]

    @staticmethod
    def get_line_name(ion_name, central_wavelength):
        return f'{ion_name}_{central_wavelength:.3f}'

    @staticmethod
    def parse_line_name(line_name):
        splitted = line_name.split('_')
        ion_name = '_'.join(splitted[:2])
        central_wavelength = float(splitted[2])
        return ion_name, central_wavelength

    @staticmethod
    def get_line_description(spectroscopic_name, central_wavelength):
        return f'{spectroscopic_name} {central_wavelength:.3f} Å'

    @staticmethod
    def line_name_to_description(line_name):
        ion_name, central_wavelength = IonProperties.parse_line_name(line_name)
        _, _, spectroscopic_name = IonProperties.parse_ion_name(ion_name)
        return IonProperties.get_line_description(spectroscopic_name,
                                                  central_wavelength)

    @staticmethod
    def parse_ion_name(ion_name):
        parsed = ch_util.convertName(ion_name)
        nuclear_charge = parsed['Z']
        ionization_stage = parsed['Ion']
        spectroscopic_name = ch_util.zion2spectroscopic(
            nuclear_charge, ionization_stage)
        return nuclear_charge, ionization_stage, spectroscopic_name

    def __set_ion_name(self, ion_name):
        self.__ion_name = ion_name
        self.__nuclear_charge, self.__ionization_stage, self.__spectroscopic_name = self.__class__.parse_ion_name(
            ion_name)

    def __find_ionization_potential(self):
        if self.ionization_stage <= self.nuclear_charge:
            self.__ionization_potential = ch_data.Ip[self.nuclear_charge - 1,
                                                     self.ionization_stage - 1]
            self.__first_ionization_potential = ch_data.Ip[
                self.nuclear_charge - 1, 0]

    def __read_abundance(self, abundance_file):
        self.__abundance_file = abundance_file
        # Abundance of this element relative to hydrogen
        self.__abundance = ch_data.Abundance[abundance_file]['abundance'][
            self.nuclear_charge - 1]


class Ion:
    def __init__(self,
                 ion_name,
                 atmosphere,
                 max_levels_before_pruning=None,
                 probe_atmosphere=None,
                 pruning_min_in_to_average_ratio=1e-1,
                 pruning_min_in_to_out_ratio=1e-1,
                 verbose=False):
        self.__filter_levels = False
        self.verbose = verbose

        self.__atmos = atmosphere
        self.__properties = IonProperties(ion_name, atmosphere.abundance_file)
        self.__read_ion_data()

        if max_levels_before_pruning is not None and self.n_levels > max_levels_before_pruning:
            probe_atmosphere = atmosphere if probe_atmosphere is None else probe_atmosphere
            self.info(
                f'Pruning {self.n_levels} levels for {probe_atmosphere.n_values} conditions'
            )
            important_levels = self.find_important_levels(
                probe_atmosphere,
                min_in_to_average_ratio=pruning_min_in_to_average_ratio,
                min_in_to_out_ratio=pruning_min_in_to_out_ratio)
            self.restrict_to_levels(important_levels)
            self.info(f'Pruned to {important_levels.size} levels')

    @property
    def properties(self):
        return self.__properties

    @property
    def ion_name(self):
        return self.properties.ion_name

    @property
    def nuclear_charge(self):
        return self.properties.nuclear_charge

    @property
    def ionization_stage(self):
        return self.properties.ionization_stage

    @property
    def abundance_file(self):
        return self.atmos.abundance_file

    @property
    def atmos(self):
        return self.__atmos

    @property
    def ionization_fractions(self):
        return self.__ionization_fractions

    @property
    def spectroscopic_name(self):
        return self.properties.spectroscopic_name

    @property
    def levels(self):
        return self.__levels

    @property
    def n_levels(self):
        return self.levels.size

    @property
    def levels_mask(self):
        return self.__getattr('levels_mask', slice(None))

    @property
    def n_lines(self):
        return self.upper_levels.size

    @property
    def upper_levels(self):
        return self.__upper_levels

    @property
    def lower_levels(self):
        return self.__lower_levels

    @property
    def upper_level_indices(self):
        return self.__find_level_indices(self.upper_levels)

    @property
    def lower_level_indices(self):
        return self.__find_level_indices(self.lower_levels)

    @property
    def scups_upper_levels(self):
        return self.__scups_upper_levels

    @property
    def scups_lower_levels(self):
        return self.__scups_lower_levels

    @property
    def scups_upper_level_indices(self):
        return self.__find_level_indices(self.scups_upper_levels)

    @property
    def scups_lower_level_indices(self):
        return self.__find_level_indices(self.scups_lower_levels)

    @property
    def psplups_upper_levels(self):
        return self.__psplups_upper_levels

    @property
    def psplups_lower_levels(self):
        return self.__psplups_lower_levels

    @property
    def psplups_upper_level_indices(self):
        return self.__find_level_indices(self.psplups_upper_levels)

    @property
    def psplups_lower_level_indices(self):
        return self.__find_level_indices(self.psplups_lower_levels)

    @property
    def central_wavelengths(self):
        return self.__central_wavelengths

    @property
    def transition_probabilities(self):
        return self.__transition_probabilities

    @property
    def abundance(self):
        return self.properties.abundance

    @property
    def populations(self):
        return self.__getattr('populations', None)

    @property
    def emissivities(self):
        return self.__getattr('emissivities', None)

    @property
    def emissivity_line_indices(self):
        return self.__getattr('emissivity_line_indices', None)

    @property
    def n_lines_with_emissivity(self):
        emissivity_line_indices = self.emissivity_line_indices
        if emissivity_line_indices is None:
            return None
        return self.n_lines if emissivity_line_indices == slice(
            None) else emissivity_line_indices.size

    def save(self, file_path):
        with open(file_path, 'wb') as f:
            joblib.dump(self, f)

    @staticmethod
    def load(file_path):
        with open(file_path, 'rb') as f:
            ion = joblib.load(f)
        return ion

    def info(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def find_line_index(self, wavelength):
        line_idx = np.argmin(
            np.abs(self.central_wavelengths - wavelength * 1e-8))
        closest_wavelength = self.central_wavelengths[line_idx] * 1e8  # [Å]
        if not np.allclose(closest_wavelength, wavelength):
            print(
                f'Warning: No line sufficiently close to requested wavelength of {wavelength:.3f} Å (smallest difference is {(closest_wavelength - wavelength):g} Å)'
            )
        return line_idx

    def find_line_indices(self, wavelengths):
        if wavelengths is None:
            indices = np.arange(self.n_lines)
        else:
            indices = np.array([
                self.find_line_index(wavelength) for wavelength in wavelengths
            ],
                               dtype=int)
        return indices

    def find_levels_for_lines(self, line_indices):
        return np.unique(
            np.concatenate((self.lower_levels[line_indices],
                            self.upper_levels[line_indices])))

    def print_lines(self):
        print(f'{self.spectroscopic_name} lines')
        for line_idx in range(self.n_lines):
            self.print_line(line_idx)

    def print_line(self, line_idx):
        print(
            f'{line_idx:3d}: λ = {self.central_wavelengths[line_idx]*1e8:8.3f} Å ({self.upper_levels[line_idx]:3d} -> {self.lower_levels[line_idx]:3d})'
        )

    def find_important_levels(self,
                              atmosphere,
                              min_in_to_average_ratio=1e-1,
                              min_in_to_out_ratio=1e-1):
        self.unrestrict()

        original_atmos = self.atmos
        self.__atmos = atmosphere
        rate_matrices = self.__construct_rate_matrices()
        self.__atmos = original_atmos

        mean_rate_matrix = np.mean(rate_matrices, axis=0)
        mean_in_rates, mean_out_rates, mean_rate = compute_transition_rate_statistics(
            mean_rate_matrix)

        # An important level must satisfy at least one of the following criteria:
        # 1. There are significant rates into the level, so the presence of this level provides a drainage for the population in other levels.
        # 2. The incoming rates are not too small compared to the outgoing ones, so the population of this level will not be negligible.
        # Otherwise the level will have a small population and will not influence the populations of other levels.
        important_levels = np.flatnonzero(
            np.logical_or(
                mean_in_rates > min_in_to_average_ratio * mean_rate,
                mean_in_rates > min_in_to_out_ratio * mean_out_rates)) + 1
        return important_levels

    def restrict_to_levels(self, included_levels):
        self.__levels_mask = np.flatnonzero(
            np.isin(self.levels, included_levels))

        self.__transitions_mask = np.flatnonzero(
            np.logical_and(np.isin(self.lower_levels, included_levels),
                           np.isin(self.upper_levels, included_levels)))

        if self.__scups_data is not None:
            self.__scups_transitions_mask = np.flatnonzero(
                np.logical_and(
                    np.isin(self.scups_lower_levels, included_levels),
                    np.isin(self.scups_upper_levels, included_levels)))

        if self.__psplups_data is not None:
            self.__psplups_transitions_mask = np.flatnonzero(
                np.logical_and(
                    np.isin(self.psplups_lower_levels, included_levels),
                    np.isin(self.psplups_upper_levels, included_levels)))

        self.restrict()

    def restrict(self):
        assert self.__hasattr('levels_mask')
        self.__unrestricted_populations = self.populations
        self.__populations = self.__getattr('restricted_populations', None)
        self.__filter_levels = True
        self.__read_ion_data()
        return self

    def unrestrict(self):
        if not self.__filter_levels:
            return self
        self.__restricted_populations = self.populations
        self.__populations = self.__unrestricted_populations
        self.__filter_levels = False
        self.__read_ion_data()
        return self

    def compute_populations(self, precomputed_rate_matrices=None):
        if precomputed_rate_matrices is None:
            rate_matrices = self.__construct_rate_matrices()
        else:
            rate_matrices = precomputed_rate_matrices

        rhs = np.zeros(rate_matrices.shape[1], np.float64)

        # Constrain equation by requiring that the sum of all level populations is 1.0
        rate_matrices[:, -1, :] = 1.0
        rhs[-1] = 1

        self.info(
            f'Solving {rate_matrices.shape[0]} sets of level population equations'
        )
        start_time = time.time()

        # Proportion of ions in the given ionization state that are in each energy level (n_levels, n_conditions)
        self.__populations = np.linalg.solve(rate_matrices,
                                             rhs[np.newaxis, :]).T

        self.__populations[self.__populations < 0] = 0

        self.info(f'Took {time.time() - start_time:g} s')

    def with_populations(self, populations):
        assert isinstance(populations, np.ndarray)
        assert populations.shape == (self.n_levels, self.atmos.n_values)
        self.__populations = populations
        return self

    def drop_populations(self):
        self.__populations = None

    def sample_ionization_fractions(self, temperatures=None):
        if temperatures is None:
            temperatures = self.atmos.temperatures

        is_flat = temperatures.ndim == 1
        if not is_flat:
            shape = temperatures.shape
            temperatures = np.ravel(temperatures)

        self.info('Sampling ionization fractions')
        start_time = time.time()

        table_temperatures = ch_data.IoneqAll['ioneqTemperature']
        table_ionization_fractions = ch_data.IoneqAll['ioneqAll'][
            self.nuclear_charge - 1, self.ionization_stage - 1].squeeze()

        valid = table_ionization_fractions > 0
        inside_lower = temperatures >= table_temperatures[valid].min()
        inside_upper = temperatures <= table_temperatures[valid].max()
        inside = np.logical_and(inside_lower, inside_upper)

        spline = scipy.interpolate.splrep(
            np.log(table_temperatures[valid]),
            np.log(table_ionization_fractions[valid]),
            s=0)
        log_ionization_fractions = scipy.interpolate.splev(
            np.log(temperatures[inside]), spline)

        # Proportion of ions of this element that are in the given ionization state (n_conditions,)
        ionization_fractions = np.zeros_like(temperatures)
        ionization_fractions[inside] = np.exp(log_ionization_fractions)

        if not is_flat:
            ionization_fractions = ionization_fractions.reshape(shape)

        self.info(f'Took {time.time() - start_time:g} s')

        return ionization_fractions

    def compute_emissivities(self, line_indices=None):
        if self.__getattr('populations', None) is None:
            self.compute_populations()
        if self.__getattr('ionization_fractions', None) is None:
            self.__ionization_fractions = self.sample_ionization_fractions()

        if line_indices is None:
            line_indices = slice(None)
            self.__emissivity_line_indices = line_indices
        else:
            self.__emissivity_line_indices = np.asarray(line_indices,
                                                        dtype=int)

        self.info('Computing emissivites')
        start_time = time.time()

        self.__emissivities = (
            (self.abundance * units.HPLANCK * units.CLIGHT /
             (4 * np.pi)) / self.central_wavelengths[line_indices, np.newaxis]
        ) * self.populations[
            self.
            upper_level_indices[line_indices], :] * self.ionization_fractions[
                np.newaxis, :] * self.atmos.hydrogen_densities[
                    np.newaxis, :] * self.transition_probabilities[
                        line_indices, np.newaxis]

        self.info(f'Took {time.time() - start_time:g} s')

    def plot(self,
             quantity_x,
             quantity_y,
             quantity_c=None,
             index=None,
             get_label=lambda x: x,
             **kwargs):
        def process_quantity_input(quantity):
            if quantity is not None and not isinstance(quantity, np.ndarray):
                quantity_name = quantity
                quantity = getattr(self, quantity_name,
                                   getattr(self.atmos, quantity_name, None))
                if quantity is None:
                    raise ValueError(
                        f'Quantity {quantity_name} is unavailable')
                elif not isinstance(quantity, np.ndarray):
                    raise ValueError(
                        f'Quantity {quantity_name} is not an array')
            return quantity

        quantity_x = process_quantity_input(quantity_x)
        quantity_y = process_quantity_input(quantity_y)
        quantity_c = process_quantity_input(quantity_c)

        assert quantity_x.ndim == 1

        if quantity_c is not None:
            assert quantity_y.ndim == 1
            if quantity_c.ndim == 3:
                assert index is not None
                assert quantity_x.size == quantity_c.shape[1]
                assert quantity_y.size == quantity_c.shape[2]
                return plotting.plot_2d_field(quantity_x, quantity_y,
                                              quantity_c[index, :, :],
                                              **kwargs)
            elif quantity_c.ndim == 2:
                assert quantity_x.size == quantity_c.shape[0]
                assert quantity_y.size == quantity_c.shape[1]
                return plotting.plot_2d_field(quantity_x, quantity_y,
                                              quantity_c, **kwargs)

        if quantity_y.ndim == 2:
            if index is None:
                assert quantity_x.size in quantity_y.shape
                quantities_y = quantity_y if quantity_y.shape.index(
                    quantity_x.size) == 1 else quantity_y.T

                fig, ax = plotting.create_2d_subplots(
                    **kwargs.pop('fig_kwargs', {}))
                plotting.set_color_cycle_from_cmap(
                    ax, quantities_y.shape[0],
                    kwargs.pop('cmap_name', 'viridis'))
                for i in range(quantities_y.shape[0]):
                    plotting.plot_1d_field(
                        quantity_x,
                        quantities_y[i, :],
                        label=get_label(i),
                        color=None,
                        fig=fig,
                        ax=ax,
                        render_now=(i == quantity_y.shape[0] - 1),
                        **kwargs)
                return fig, ax
            else:
                assert quantity_x.shape == quantity_y.shape[1]
                return plotting.plot_1d_field(quantity_x,
                                              quantity_y[index, :],
                                              label=get_label(index),
                                              **kwargs)

        elif quantity_y.ndim == 1:
            assert quantity_x.shape == quantity_y.shape
            return plotting.plot_1d_field(quantity_x, quantity_y, **kwargs)

    def __getattr(self, name, *args):
        return getattr(self, f'_Ion__{name}', *args)

    def __hasattr(self, name):
        return hasattr(self, f'_Ion__{name}')

    def __read_ion_data(self):
        self.__read_wgfa_file()
        self.__read_elvlc_file()
        self.__read_scups_file()
        self.__read_psplups_file()

    def __read_elvlc_file(self):
        if not self.__hasattr('elevlc_data'):
            self.__elevlc_data = ch_io.elvlcRead(self.ion_name)

        levels = np.asarray(self.__elevlc_data['lvl'], int)
        mask = np.isin(levels, self.__levels_with_transitions)

        self.__levels = self.__levels_filtered_arr(levels[mask])
        self.__levels_missing = self.__levels.size < levels.size

        self.__elvlc_eryd = self.__levels_filtered_arr(
            np.asarray(self.__elevlc_data['eryd'])[mask])
        self.__elvlc_erydth = self.__levels_filtered_arr(
            np.asarray(self.__elevlc_data['erydth'])[mask])
        self.__elvlc_mult = self.__levels_filtered_arr(
            np.asarray(self.__elevlc_data['mult'])[mask])
        self.__elvlc_ecm = self.__levels_filtered_arr(
            np.asarray(self.__elevlc_data['ecm'])[mask])

        self.__levels_sorter = np.argsort(self.__levels)

    def __find_level_indices(self, levels):
        return self.__levels_sorter[np.searchsorted(
            self.__levels, levels, sorter=self.__levels_sorter
        )] if self.__levels_missing else levels - 1

    def __read_wgfa_file(self):
        if not self.__hasattr('wgfa_data'):
            self.__wgfa_data = ch_io.wgfaRead(self.ion_name)

        central_wavelengths = np.asarray(self.__wgfa_data['wvl'], np.float64)
        mask = central_wavelengths > 0  # Excludes autoionization lines

        lower_levels = np.asarray(self.__wgfa_data['lvl1'], int)[mask]
        upper_levels = np.asarray(self.__wgfa_data['lvl2'], int)[mask]

        self.__levels_with_transitions = np.unique(
            np.concatenate((lower_levels, upper_levels)))

        # Wavelength of each transition from upper to lower energy level (n_lines,) [cm]
        self.__central_wavelengths = self.__transitions_filtered_arr(
            central_wavelengths[mask] * 1e-8)

        # All lower and upper energy levels involved in each transition (n_lines,)
        self.__lower_levels = self.__transitions_filtered_arr(lower_levels)
        self.__upper_levels = self.__transitions_filtered_arr(upper_levels)

        # Spontaneous transition probabilities (n_lines,) [1/s]
        self.__transition_probabilities = self.__transitions_filtered_arr(
            np.asarray(self.__wgfa_data['avalue'], np.float64)[mask])

    def __read_scups_file(self):
        if os.path.isfile(ch_util.ion2filename(self.ion_name) + '.scups'):
            if not self.__hasattr('scups_data'):
                self.__scups_data = ch_io.scupsRead(self.ion_name)

            scups_lower_levels = np.asarray(self.__scups_data['lvl1'], int)
            scups_upper_levels = np.asarray(self.__scups_data['lvl2'], int)

            mask = np.logical_and(
                np.isin(scups_lower_levels, self.__levels_with_transitions),
                np.isin(scups_upper_levels, self.__levels_with_transitions))

            self.__scups_lower_levels = self.__scups_transitions_filtered_arr(
                scups_lower_levels[mask])
            self.__scups_upper_levels = self.__scups_transitions_filtered_arr(
                scups_upper_levels[mask])
            self.__scups_ttype = self.__scups_transitions_filtered_arr(
                np.asarray(self.__scups_data['ttype'], int)[mask])
            self.__scups_cups = self.__scups_transitions_filtered_arr(
                np.asarray(self.__scups_data['cups'])[mask])
            self.__scups_xs = self.__scups_transitions_filtered_list([
                x for i, x in enumerate(self.__scups_data['btemp']) if mask[i]
            ])
            self.__scups_scups = self.__scups_transitions_filtered_list([
                x for i, x in enumerate(self.__scups_data['bscups']) if mask[i]
            ])
            self.__scups_de = self.__scups_transitions_filtered_arr(
                np.asarray(self.__scups_data['de'])[mask])
        else:
            self.__scups_data = None

    def __read_psplups_file(self):
        if os.path.isfile(ch_util.ion2filename(self.ion_name) + '.psplups'):
            if not self.__hasattr('psplups_data'):
                self.__psplups_data = ch_io.splupsRead(self.ion_name,
                                                       filetype='psplups')

            psplups_lower_levels = np.asarray(self.__psplups_data['lvl1'], int)
            psplups_upper_levels = np.asarray(self.__psplups_data['lvl2'], int)

            mask = np.logical_and(
                np.isin(psplups_lower_levels, self.__levels_with_transitions),
                np.isin(psplups_upper_levels, self.__levels_with_transitions))

            self.__psplups_lower_levels = self.__psplups_transitions_filtered_arr(
                psplups_lower_levels[mask])
            self.__psplups_upper_levels = self.__psplups_transitions_filtered_arr(
                psplups_upper_levels[mask])
            self.__psplups_ttype = self.__psplups_transitions_filtered_arr(
                np.asarray(self.__psplups_data['ttype'], int)[mask])
            self.__psplups_cups = self.__psplups_transitions_filtered_arr(
                np.asarray(self.__psplups_data['cups'])[mask])
            self.__psplups_nspls = self.__psplups_transitions_filtered_list([
                x for i, x in enumerate(self.__psplups_data['nspl']) if mask[i]
            ])
            self.__psplups_splups = self.__psplups_transitions_filtered_list([
                x for i, x in enumerate(self.__psplups_data['splups'])
                if mask[i]
            ])
        else:
            self.__psplups_data = None

    def __construct_rate_matrices(self):
        self.info(
            f'Building {self.atmos.n_values} rate matrices with shape {self.n_levels} x {self.n_levels}'
        )
        start_time = time.time()

        matrix_shape = (self.n_levels, self.n_levels)
        rate_matrix = np.zeros(matrix_shape, np.float64)

        l1 = self.lower_level_indices
        l2 = self.upper_level_indices
        array_utils.add_values_in_matrix(rate_matrix, l1, l2,
                                         self.transition_probabilities)
        array_utils.subtract_values_in_matrix(rate_matrix, l2, l2,
                                              self.transition_probabilities)

        # Photo-excitation and stimulated emission
        if self.atmos.radiation_temperature is not None:
            self.info(
                f'Including photoexcitation and stimulated emission at {self.atmos.radiation_temperature} K'
            )
            assert self.atmos.distance_from_center is not None
            dilute = ch_util.dilute(self.atmos.distance_from_center)

            # Don't include autoionization lines
            mask = np.abs(self.central_wavelengths) > 0
            l1 = l1[mask]
            l2 = l2[mask]

            de = (units.HPLANCK * units.CLIGHT) * (self.__elvlc_ecm[l2] -
                                                   self.__elvlc_ecm[l1])
            dekt = de / (units.KBOLTZMANN * self.atmos.radiation_temperature)

            # Photoexcitation
            phex_values = self.transition_probabilities[mask] * dilute * (
                self.__elvlc_mult[l2] /
                self.__elvlc_mult[l1]) / (np.exp(dekt) - 1.0)

            array_utils.add_values_in_matrix(rate_matrix, l2, l1, phex_values)
            array_utils.subtract_values_in_matrix(rate_matrix, l1, l1,
                                                  phex_values)

            # Stimulated emission
            stem_values = self.transition_probabilities[mask] * dilute / (
                np.exp(-dekt) - 1.0)
            array_utils.add_values_in_matrix(rate_matrix, l1, l2, stem_values)
            array_utils.subtract_values_in_matrix(rate_matrix, l2, l2,
                                                  stem_values)

        rate_matrices = np.repeat(rate_matrix[np.newaxis, :],
                                  self.atmos.n_values,
                                  axis=0)

        if self.__scups_data is not None:
            self.info('Including electron collisions')
            _, excitation_rates, deexcitation_rates = self.__compute_collision_strengths(
                for_proton=False)
            l1_scups = self.scups_lower_level_indices
            l2_scups = self.scups_upper_level_indices
            dex_values = self.atmos.electron_densities[
                np.newaxis, :] * deexcitation_rates
            ex_values = self.atmos.electron_densities[
                np.newaxis, :] * excitation_rates
            array_utils.add_values_in_matrices(rate_matrices, l1_scups,
                                               l2_scups, dex_values)
            array_utils.add_values_in_matrices(rate_matrices, l2_scups,
                                               l1_scups, ex_values)
            array_utils.subtract_values_in_matrices(rate_matrices, l1_scups,
                                                    l1_scups, ex_values)
            array_utils.subtract_values_in_matrices(rate_matrices, l2_scups,
                                                    l2_scups, dex_values)

        if self.__psplups_data is not None:
            self.info('Including proton collisions')
            _, excitation_rates, deexcitation_rates = self.__compute_collision_strengths(
                for_proton=True)
            l1_psplups = self.psplups_lower_level_indices
            l2_psplups = self.psplups_upper_level_indices
            pdex_values = self.atmos.proton_densities[
                np.newaxis, :] * deexcitation_rates
            pex_values = self.atmos.proton_densities[
                np.newaxis, :] * excitation_rates
            array_utils.add_values_in_matrices(rate_matrices, l1_psplups,
                                               l2_psplups, pdex_values)
            array_utils.add_values_in_matrices(rate_matrices, l2_psplups,
                                               l1_psplups, pex_values)
            array_utils.subtract_values_in_matrices(rate_matrices, l1_psplups,
                                                    l1_psplups, pex_values)
            array_utils.subtract_values_in_matrices(rate_matrices, l2_psplups,
                                                    l2_psplups, pdex_values)

        self.info(f'Took {time.time() - start_time:g} s')

        return rate_matrices

    def __compute_collision_strengths(self, for_proton=False):
        if for_proton:
            assert self.__psplups_data is not None
            n_transitions = self.psplups_lower_levels.size
            lower_levels = self.psplups_lower_levels
            upper_levels = self.psplups_upper_levels
            ttypes = self.__psplups_ttype
            cups = self.__psplups_cups
            xs = [
                np.arange(nspl) / (nspl - 1) for nspl in self.__psplups_nspls
            ]
            scups = self.__psplups_splups
        else:
            assert self.__scups_data is not None
            n_transitions = self.scups_lower_levels.size
            lower_levels = self.scups_lower_levels
            upper_levels = self.scups_upper_levels
            ttypes = self.__scups_ttype
            cups = self.__scups_cups
            xs = self.__scups_xs
            scups = self.__scups_scups

        collision_strengths = np.zeros((n_transitions, self.atmos.n_values),
                                       np.float64)
        excitation_rates = np.zeros((n_transitions, self.atmos.n_values),
                                    np.float64)
        deexcitation_rates = np.zeros((n_transitions, self.atmos.n_values),
                                      np.float64)

        elvlc = np.where(self.__elvlc_eryd >= 0, self.__elvlc_eryd,
                         self.__elvlc_erydth)

        lower_level_indices = self.__find_level_indices(lower_levels)
        upper_level_indices = self.__find_level_indices(upper_levels)

        if for_proton:
            de = elvlc[upper_level_indices] - elvlc[lower_level_indices]
        else:
            de = self.__scups_de

        kte = units.KBOLTZMANN * self.atmos.temperatures[np.newaxis, :] / (
            de[:, np.newaxis] * units.RYD_TO_ERG)

        compute_st_1_4 = lambda c, k: 1.0 - np.log(c) / np.log(k + c)
        compute_st_2_3_5_6 = lambda c, k: k / (k + c)
        compute_st = [
            compute_st_1_4, compute_st_2_3_5_6, compute_st_2_3_5_6,
            compute_st_1_4, compute_st_2_3_5_6, compute_st_2_3_5_6
        ]

        compute_cs = [
            lambda c, k, s: s * np.log(k + np.e), lambda c, k, s: s,
            lambda c, k, s: s / (k + 1.0), lambda c, k, s: s * np.log(k + c),
            lambda c, k, s: s / k, lambda c, k, s: 10**s
        ]

        st = np.zeros_like(kte)
        for i, ttype in enumerate(range(1, 7)):
            mask = ttypes == ttype
            st[mask, :] = compute_st[i](cups[mask][:, np.newaxis],
                                        kte[mask, :])

        for i in range(n_transitions):
            spline = scipy.interpolate.splrep(xs[i], scups[i], s=0)
            sups = scipy.interpolate.splev(st[i, :], spline)
            collision_strengths[i, :] = compute_cs[ttypes[i] - 1](cups[i],
                                                                  kte[i, :],
                                                                  sups)

        collision = units.HPLANCK**2 / (
            (2. * np.pi * units.M_ELECTRON)**1.5 * np.sqrt(units.KBOLTZMANN))

        de = np.abs(elvlc[upper_level_indices] - elvlc[lower_level_indices])
        ekt = (de[:, np.newaxis] * units.RYD_TO_ERG) / (
            units.KBOLTZMANN * self.atmos.temperatures[np.newaxis, :])

        sqrt_temperatures = np.sqrt(self.atmos.temperatures)[np.newaxis, :]
        deexcitation_rates = collision * collision_strengths / (
            self.__elvlc_mult[upper_level_indices][:, np.newaxis] *
            sqrt_temperatures)
        excitation_rates = collision * collision_strengths * np.exp(-ekt) / (
            self.__elvlc_mult[lower_level_indices][:, np.newaxis] *
            sqrt_temperatures)

        collision_strengths[collision_strengths < 0] = 0
        return collision_strengths, excitation_rates, deexcitation_rates

    def __levels_filtered_arr(self, full):
        return self.__filtered_arr(
            full, self.__levels_mask) if self.__filter_levels else full

    def __transitions_filtered_arr(self, full):
        return self.__filtered_arr(
            full, self.__transitions_mask) if self.__filter_levels else full

    def __scups_transitions_filtered_arr(self, full):
        return self.__filtered_arr(
            full,
            self.__scups_transitions_mask) if self.__filter_levels else full

    def __scups_transitions_filtered_list(self, full):
        return self.__filtered_list(
            full,
            self.__scups_transitions_mask) if self.__filter_levels else full

    def __psplups_transitions_filtered_arr(self, full):
        return self.__filtered_arr(
            full,
            self.__psplups_transitions_mask) if self.__filter_levels else full

    def __psplups_transitions_filtered_list(self, full):
        return self.__filtered_list(
            full,
            self.__psplups_transitions_mask) if self.__filter_levels else full

    def __filtered_arr(self, full, mask):
        return full[mask]

    def __filtered_list(self, full, indices):
        return [full[i] for i in indices]


@njit
def compute_transition_rate_statistics(matrix):
    assert len(matrix.shape) == 2 and matrix.shape[0] == matrix.shape[1]
    size = matrix.shape[0]

    mean_in_rates = np.zeros(size)
    mean_out_rates = np.zeros(size)
    mean_rate = 0
    for i in range(size):
        for j in range(0, i):
            mean_in_rates[i] += matrix[i, j]
            mean_out_rates[i] += matrix[j, i]
            mean_rate += matrix[i, j]
        for j in range(i + 1, size):
            mean_in_rates[i] += matrix[i, j]
            mean_out_rates[i] += matrix[j, i]
            mean_rate += matrix[i, j]
        mean_out_rates[i] /= size - 1
        mean_in_rates[i] /= size - 1
    mean_rate /= size * (size - 1)

    return mean_in_rates, mean_out_rates, mean_rate


class LookupIonAtmosphere(IonAtmosphere):
    def __init__(self,
                 log_temperature_limits=(3.0, 7.0),
                 log_electron_density_limits=(8.0, 13.0),
                 n_temperature_points=100,
                 n_electron_density_points=100,
                 compute_proton_densities=True,
                 compute_hydrogen_densities=False,
                 **kwargs):
        self.__log_table_temperatures = np.linspace(*log_temperature_limits,
                                                    n_temperature_points)
        self.__log_table_electron_densities = np.linspace(
            *log_electron_density_limits, n_electron_density_points)

        table_temperatures = 10**self.log_table_temperatures
        table_electron_densities = 10**self.log_table_electron_densities
        temperature_mesh, electron_density_mesh = np.meshgrid(
            table_temperatures, table_electron_densities, indexing='ij')

        super().__init__(np.ravel(temperature_mesh),
                         np.ravel(electron_density_mesh),
                         compute_proton_densities=False,
                         compute_hydrogen_densities=False,
                         **kwargs)

        if compute_proton_densities:
            self.info(
                f'Computing {n_temperature_points*n_electron_density_points} proton densities'
            )
            start_time = time.time()

            proton_to_electron_ratios = self.species_ratios.compute_proton_to_electron_ratios(
                self.log_table_temperatures, temperature_is_log=True)
            self.__proton_densities = np.ravel(
                np.outer(proton_to_electron_ratios, table_electron_densities))

            self.info(f'Took {time.time() - start_time:g} s')

        if compute_hydrogen_densities:
            self.info(
                f'Computing {n_temperature_points*n_electron_density_points} hydrogen densities'
            )
            start_time = time.time()

            hydrogen_to_electron_ratios = self.species_ratios.compute_hydrogen_to_electron_ratios(
                self.log_table_temperatures, temperature_is_log=True)
            self.__hydrogen_densities = np.ravel(
                np.outer(hydrogen_to_electron_ratios,
                         table_electron_densities))

            self.info(f'Took {time.time() - start_time:g} s')

    @property
    def n_table_temperatures(self):
        return self.log_table_temperatures.size

    @property
    def n_table_electron_densities(self):
        return self.table_electron_densities.size

    @property
    def table_shape(self):
        return (self.n_table_temperatures, self.n_table_electron_densities)

    @property
    def log_table_temperatures(self):
        return self.__log_table_temperatures

    @property
    def table_temperatures(self):
        return 10**self.log_table_temperatures

    @property
    def log_table_electron_densities(self):
        return self.__log_table_electron_densities

    @property
    def table_electron_densities(self):
        return 10**self.log_table_electron_densities

    @property
    def proton_densities(self):
        return self.__proton_densities

    @property
    def hydrogen_densities(self):
        return self.__hydrogen_densities


class LookupIon(Ion):
    @property
    def table_populations(self):
        pop = self.populations
        return None if pop is None else pop.reshape(
            (self.n_levels, *self.atmos.table_shape))

    def in_atmosphere(self,
                      atmosphere,
                      included_levels=None,
                      use_memmap=True,
                      **kwargs):
        ion = Ion(self.ion_name, atmosphere, verbose=self.verbose)
        if included_levels is not None:
            ion.restrict_to_levels(included_levels)

        shape = (atmosphere.n_values, 2)
        dtype = atmosphere.temperatures.dtype
        evaluation_coordinates = create_tmp_memmap(
            shape, dtype) if use_memmap else np.empty(shape, dtype=dtype)
        np.stack((np.ravel(np.log10(atmosphere.temperatures)),
                  np.ravel(np.log10(atmosphere.electron_densities))),
                 axis=1,
                 out=evaluation_coordinates)

        populations = self.__lookup_populations(evaluation_coordinates,
                                                ion.levels_mask,
                                                use_memmap=use_memmap,
                                                **kwargs)
        return ion.with_populations(populations)

    def __lookup_populations(self,
                             evaluation_coordinates,
                             included_level_indices,
                             use_memmap=True,
                             method='linear',
                             bounds_error=False,
                             fill_value=None,
                             n_jobs=1):
        assert evaluation_coordinates.ndim == 2 and evaluation_coordinates.shape[
            1] == 2
        if self.table_populations is None:
            self.compute_populations()

        self.info(
            f'Interpolating populations for {evaluation_coordinates.shape[0]} conditions'
        )
        start_time = time.time()

        included_table_populations = self.table_populations[
            included_level_indices, :, :]

        shape = (included_table_populations.shape[0],
                 evaluation_coordinates.shape[0])
        dtype = included_table_populations.dtype
        populations = create_tmp_memmap(
            shape, dtype) if use_memmap else np.empty(shape, dtype=dtype)

        do_concurrent_interp2(populations,
                              0,
                              shape[0],
                              self.atmos.log_table_temperatures,
                              self.atmos.log_table_electron_densities,
                              included_table_populations,
                              evaluation_coordinates,
                              method=method,
                              bounds_error=bounds_error,
                              fill_value=fill_value,
                              verbose=self.verbose)
        populations[populations < 0.0] = 0.0
        populations[populations > 1.0] = 1.0

        self.info(f'Took {time.time() - start_time:g} s')
        return populations


class tempmap(np.memmap):
    def __new__(subtype,
                dtype=np.uint8,
                mode='r+',
                offset=0,
                shape=None,
                order='C'):
        filename = tempfile.mkstemp()[1]
        self = np.memmap.__new__(subtype,
                                 filename,
                                 dtype=dtype,
                                 mode=mode,
                                 offset=offset,
                                 shape=shape,
                                 order=order)
        return self

    def __del__(self):
        if self.filename is not None and os.path.isfile(self.filename):
            os.remove(self.filename)


def create_tmp_memmap(shape, dtype, mode='w+'):
    return tempmap(shape=shape, dtype=dtype, mode=mode)


def do_concurrent_interp2(f,
                          start,
                          stop,
                          xp,
                          yp,
                          fp,
                          coords,
                          verbose=False,
                          **kwargs):
    if verbose and start == 0:
        start_time = time.time()
        f[start, :] = scipy.interpolate.interpn((xp, yp), fp[start, :, :],
                                                coords, **kwargs)
        elapsed_time = time.time() - start_time
        print(
            f'Single interpolation took {elapsed_time:g} s, estimated total interpolation time is {elapsed_time*stop:g} s'
        )
        start += 1
    for idx in range(start, stop):
        f[idx, :] = scipy.interpolate.interpn((xp, yp), fp[idx, :, :], coords,
                                              **kwargs)


def concurrent_interp2(xp, yp, fp, coords, verbose=False, n_jobs=1, **kwargs):
    n = fp.shape[0]
    f = create_tmp_memmap(shape=(n, coords.shape[0]), dtype=fp.dtype)
    chunk_sizes = np.full(n_jobs, n // n_jobs, dtype=int)
    chunk_sizes[:(n % n_jobs)] += 1
    stop_indices = np.cumsum(chunk_sizes)
    start_indices = stop_indices - chunk_sizes
    Parallel(n_jobs=min(n_jobs, n), verbose=verbose)(
        delayed(do_concurrent_interp2)(
            f, start, stop, xp, yp, fp, coords, verbose=verbose, **kwargs)
        for start, stop in zip(start_indices, stop_indices))
    return f


class IonEmissivities:
    @staticmethod
    def from_lookup(lookup_ion,
                    bifrost_data,
                    height_range=(-np.inf, np.inf),
                    min_ionization_fraction=None,
                    z_upsampling_factor=1,
                    included_line_wavelengths=None,
                    use_memmap=False,
                    drop_populations=True,
                    n_jobs=1):
        tg = fields.ScalarField3.from_bifrost_data(bifrost_data,
                                                   'tg',
                                                   height_range=height_range)
        nel = fields.ScalarField3.from_bifrost_data(bifrost_data,
                                                    'nel',
                                                    height_range=height_range)

        if min_ionization_fraction is None:
            mask = None
            coords = tg.get_coords()
            tg = tg.get_values_flat()
            nel = nel.get_values_flat()
        else:
            ionization_fractions = lookup_ion.sample_ionization_fractions(
                tg.get_values())
            mask = array_utils.CompactArrayMask(
                ionization_fractions > min_ionization_fraction)
            lookup_ion.info(
                f'Proportion of positions with significant ionization fractions is {1e2*mask.compact_size/mask.size:.1f}%'
            )

            if z_upsampling_factor > 1:
                index_range = mask.axis_ranges[2]
                lookup_ion.info(
                    f'Upsampling to {z_upsampling_factor*1e2:g}% along z-axis for k in [{index_range[0]}, {index_range[1]})'
                )
                start_time = time.time()

                tg = tg.resampled_along_axis(2,
                                             z_upsampling_factor,
                                             index_range=index_range,
                                             kind='linear')
                nel = nel.resampled_along_axis(2,
                                               z_upsampling_factor,
                                               index_range=index_range,
                                               kind='linear')

                ionization_fractions = lookup_ion.sample_ionization_fractions(
                    tg.get_values())
                mask = array_utils.CompactArrayMask(
                    ionization_fractions > min_ionization_fraction)

                lookup_ion.info(f'Took {time.time() - start_time:g} s')

            coords = tg.get_coords()
            tg = mask.apply(tg.get_values())
            nel = mask.apply(nel.get_values())

            lookup_ion.info(
                f'Compact atmosphere representation size is {1e2*tg.size/(bifrost_data.x.size*bifrost_data.y.size*bifrost_data.z.size):.1f}%'
            )

        atmos = IonAtmosphere3D(*coords,
                                tg,
                                nel,
                                species_ratios=lookup_ion.atmos.species_ratios,
                                compute_proton_densities=False,
                                verbose=lookup_ion.atmos.verbose)

        included_levels = lookup_ion.find_levels_for_lines(
            lookup_ion.find_line_indices(included_line_wavelengths))

        ion = lookup_ion.in_atmosphere(atmos,
                                       included_levels,
                                       use_memmap=use_memmap)

        ion.compute_emissivities(
            line_indices=ion.find_line_indices(included_line_wavelengths))
        if drop_populations:
            ion.drop_populations()  # May be too big to keep

        central_wavelengths = ion.central_wavelengths[
            ion.emissivity_line_indices]
        emissivities = ion.atmos.unravel(
            ion.emissivities) if mask is None else ion.emissivities

        return IonEmissivities(ion.properties,
                               bifrost_data,
                               height_range,
                               ion.atmos.x_coords,
                               ion.atmos.y_coords,
                               ion.atmos.z_coords,
                               central_wavelengths,
                               mask,
                               emissivities,
                               verbose=lookup_ion.verbose,
                               n_jobs=n_jobs)

    @staticmethod
    def get_optimization_tag(min_ionization_fraction, z_upsampling_factor):
        if min_ionization_fraction is None:
            return None
        else:
            return f'mi={min_ionization_fraction}_zu={z_upsampling_factor}'

    def __init__(self,
                 ion_properties,
                 bifrost_data,
                 height_range,
                 x_coords,
                 y_coords,
                 z_coords,
                 central_wavelengths,
                 mask,
                 emissivities,
                 n_jobs=1,
                 verbose=False):
        self.__ion_properties = ion_properties
        self.__bifrost_data = bifrost_data
        self.__bifrost_data_class = type(bifrost_data)
        self.__height_range = height_range
        self.__x_coords = x_coords
        self.__y_coords = y_coords
        self.__z_coords = z_coords
        self.__central_wavelengths = central_wavelengths
        self.__mask = mask
        self.__emissivities = emissivities
        self.n_jobs = n_jobs
        self.verbose = verbose

    @property
    def ion_properties(self):
        return self.__ion_properties

    @property
    def bifrost_data(self):
        return self.__bifrost_data

    @property
    def x_coords(self):
        return self.__x_coords

    @property
    def y_coords(self):
        return self.__y_coords

    @property
    def z_coords(self):
        return self.__z_coords

    @property
    def horizontal_coords(self):
        return (self.x_coords, self.y_coords)

    @property
    def height_range(self):
        return self.__height_range

    @property
    def central_wavelengths(self):
        return self.__central_wavelengths

    @property
    def central_wavelengths_Å(self):
        return self.central_wavelengths * 1e8

    @property
    def n_lines(self):
        return self.central_wavelengths.size

    @property
    def mask(self):
        return self.__mask

    @property
    def emissivities(self):
        return self.__emissivities

    def info(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def drop_bifrost_data(self):
        self.__bd_root_name = self.bifrost_data.root_name
        self.__bd_snap_num = self.bifrost_data.snap
        self.__bd_input_path = self.bifrost_data.fdir
        self.__bd_verbose = self.bifrost_data.verbose
        self.__bifrost_data = None

    def reopen_dropped_bifrost_data(self):
        self.__bifrost_data = self.__bifrost_data_class(
            self.__bd_root_name,
            snap=self.__bd_snap_num,
            fdir=self.__bd_input_path,
            verbose=self.__bd_verbose)

    def save(self, file_path):
        bd = self.__bifrost_data
        self.drop_bifrost_data()
        with open(file_path, 'wb') as f:
            joblib.dump(self, f)
        self.__bifrost_data = bd

    @staticmethod
    def load(file_path):
        with open(file_path, 'rb') as f:
            ion_emissivities = joblib.load(f)
        ion_emissivities.reopen_dropped_bifrost_data()
        return ion_emissivities

    def save_data(self, file_path):
        with open(file_path, 'wb') as f:
            np.savez_compressed(
                f,
                ion_name=self.ion_properties.ion_name,
                abundance=self.ion_properties.abundance,
                height_range=self.height_range,
                x_coords=self.x_coords,
                y_coords=self.y_coords,
                z_coords=self.z_coords,
                central_wavelengths=self.central_wavelengths,
                mask=(0 if self.mask is None else self.mask.mask),
                emissivities=self.emissivities)

    @staticmethod
    def from_data_file(file_path, bifrost_data, **kwargs):
        data = np.load(file_path)
        return IonEmissivities(
            IonProperties(data['ion_name'].item(),
                          data['abundance'].item()), bifrost_data,
            tuple(data['height_range']), data['x_coords'], data['y_coords'],
            data['z_coords'], data['central_wavelengths'],
            None if data['mask'].shape == () else array_utils.CompactArrayMask(
                data['mask']), data['emissivities'], **kwargs)

    def synthesize_spectral_lines(self,
                                  spectrum,
                                  extra_quantity_names=None,
                                  min_intensity=1.0):
        self.info(f'Synthesizing {self.n_lines} spectral lines')
        start_time = time.time()

        temperatures = fields.ScalarField3.from_bifrost_data(
            self.bifrost_data, 'tg', height_range=self.height_range)
        vertical_speeds = fields.ScalarField3.from_bifrost_data(
            self.bifrost_data, 'uz',
            height_range=self.height_range)  # Positive velocity is downward

        extra_quantity_fields = {} if extra_quantity_names is None else {
            quantity_name: fields.ScalarField3.from_bifrost_data(
                self.bifrost_data,
                quantity_name,
                height_range=self.height_range)
            for quantity_name in extra_quantity_names
        }

        atmos_shape = temperatures.get_shape()

        if atmos_shape[2] != self.z_coords.size:
            temperatures = temperatures.resampled_to_coords_along_axis(
                2, self.z_coords)
            vertical_speeds = vertical_speeds.resampled_to_coords_along_axis(
                2, self.z_coords)
            for quantity_name, quantity_field in extra_quantity_fields.items():
                extra_quantity_fields[
                    quantity_name] = quantity_field.resampled_to_coords_along_axis(
                        2, self.z_coords)

            dz = np.zeros_like(self.z_coords)
            dz[:-1] = self.z_coords[1:] - self.z_coords[:-1]
            dz[-1] = self.z_coords[-1] - self.z_coords[-2]
            dz *= units.U_L

            atmos_shape = atmos_shape[:2] + (self.z_coords.size, )
        else:
            dz = fields.ScalarField1.dz_in_bifrost_data(
                self.bifrost_data,
                height_range=self.height_range,
                scale=units.U_L).get_values()  # [cm]

        temperatures = temperatures.get_values()
        vertical_speeds = vertical_speeds.get_values()
        for quantity_name, quantity in extra_quantity_fields.items():
            extra_quantity_fields[quantity_name] = quantity.get_values()

        if self.mask is None:
            intensity_contributions = self.emissivities * dz[np.newaxis,
                                                             np.newaxis,
                                                             np.newaxis, :]

            def integrate_emissivities(line_idx, weights=None):
                if weights is None:
                    return np.sum(intensity_contributions[line_idx, :, :, :],
                                  axis=2)
                else:
                    return np.sum(weights *
                                  intensity_contributions[line_idx, :, :, :],
                                  axis=2)
        else:
            temperatures = self.mask.apply(temperatures)
            vertical_speeds = self.mask.apply(vertical_speeds)
            for quantity_name, quantity in extra_quantity_fields.items():
                extra_quantity_fields[quantity_name] = self.mask.apply(
                    quantity)
            intensity_contributions = self.emissivities * self.mask.apply(
                np.broadcast_to(dz, atmos_shape))[np.newaxis, :]

            def integrate_emissivities(line_idx, weights=None):
                if weights is None:
                    return self.mask.sum_over_axis(
                        intensity_contributions[line_idx, :], axis=2)
                else:
                    return self.mask.sum_over_axis(
                        weights * intensity_contributions[line_idx, :], axis=2)

        def synthesize_line(line_idx):
            central_wavelength = self.central_wavelengths[line_idx]
            local_doppler_shifts = vertical_speeds * (
                units.U_U * central_wavelength / units.CLIGHT
            )  # [cm] (negative sign omitted so that positive is upwards)
            thermal_variances = self.ion_properties.compute_thermal_line_variance(
                central_wavelength, temperatures)  # [cm^2]

            intensities = integrate_emissivities(line_idx)  # [erg/s/sr/cm^2]
            invalid = intensities < min_intensity
            doppler_shifts = integrate_emissivities(
                line_idx, weights=local_doppler_shifts) / intensities  # [cm]
            doppler_shifts[invalid] = np.nan
            variances = integrate_emissivities(
                line_idx,
                weights=(thermal_variances + local_doppler_shifts**
                         2)) / intensities - doppler_shifts**2  # [cm^2]
            variances[invalid] = np.nan

            doppler_shifts *= 1e8  # [Å]
            widths = (2 * np.sqrt(2 * np.log(2)) * 1e8) * np.sqrt(
                variances)  # [Å] (FWHM)

            quantities = dict(intensity=intensities,
                              doppler_shift=doppler_shifts,
                              width=widths)

            for quantity_name, quantity_field in extra_quantity_fields.items():
                quantity = integrate_emissivities(
                    line_idx, weights=quantity_field) / intensities
                quantity[invalid] = np.nan
                quantities[quantity_name] = quantity

            return SpectralLine(self.ion_properties,
                                self.central_wavelengths_Å[line_idx],
                                quantities)

        for spectral_line in Parallel(
                n_jobs=min(self.n_jobs, self.n_lines), verbose=self.verbose)(
                    (delayed(synthesize_line)(line_idx)
                     for line_idx in range(self.n_lines))):
            spectrum[spectral_line.name, self.bifrost_data.root_name,
                     self.bifrost_data.snap] = spectral_line

        self.info(f'Took {time.time() - start_time:g} s')

    def load_spectral_lines(self, spectrum, dir_path, tag=None):
        dir_path = pathlib.Path(dir_path)
        tag = join_tags(tag, as_str=True)
        ion_name = self.ion_properties.ion_name
        atmosphere_name = self.bifrost_data.root_name
        snap_num = self.bifrost_data.snap
        for central_wavelength in self.central_wavelengths_Å:
            spectral_line = SpectralLine.load(
                dir_path /
                f'{ion_name}_{central_wavelength:.3f}_{atmosphere_name}_{snap_num:03d}{tag}.npz'
            )
            spectrum[spectral_line.name, atmosphere_name,
                     snap_num] = spectral_line

    def plot(self, line_idx, height, **kwargs):
        assert self.mask is None, 'Cannot plot emissivities in compact representation'
        k = min(self.z_coords.size - 1,
                max(0, np.searchsorted(self.z_coords, height)))
        return fields.ScalarField2(
            fields.Coords2(self.x_coords, self.y_coords),
            self.emissivities[line_idx, :, :, k]).plot(**kwargs)

    def obtain_mean_intensity_contribution_field(self, line_idx):
        assert self.mask is None, 'Cannot plot emissivities in compact representation'
        dz = fields.ScalarField1.dz_in_bifrost_data(
            self.bifrost_data, height_range=self.height_range, scale=units.U_L)
        return dz * np.mean(self.emissivities[line_idx, :, :, :], axis=(0, 1))


class SpectralLine:
    def __init__(self,
                 ion_properties,
                 central_wavelength,
                 quantities,
                 selected_quantity_name='intensity'):
        self.__ion_properties = ion_properties
        self.__central_wavelength = central_wavelength  # [Å]
        self.__quantities = quantities

        self.__derived_quantities = dict(
            doppler_velocity=self.__compute_doppler_velocity,
            width_velocity=self.__compute_width_velocity)

        self.__selected_quantity_name = selected_quantity_name
        self.__selected_quantity_params = {}

    @property
    def ion_properties(self):
        return self.__ion_properties

    @property
    def ion_name(self):
        return self.ion_properties.ion_name

    @property
    def central_wavelength(self):
        return self.__central_wavelength

    @property
    def quantities(self):
        return self.__quantities

    @property
    def selected_quantity_name(self):
        return self.__selected_quantity_name

    @property
    def selected_quantity_params(self):
        return self.__selected_quantity_params

    @property
    def name(self):
        return IonProperties.get_line_name(self.ion_name,
                                           self.central_wavelength)

    @property
    def description(self):
        return IonProperties.get_line_description(
            self.ion_properties.spectroscopic_name, self.central_wavelength)

    def get_tag(self, atmosphere_name, snap_num, tag=None):
        tag = join_tags(tag, as_str=True)
        return f'{self.ion_name}_{self.central_wavelength:.3f}_{atmosphere_name}_{snap_num:03d}{tag}'

    @staticmethod
    def get_tag_regex(ion_name=None,
                      central_wavelength=None,
                      atmosphere_name=None,
                      snap_num=None,
                      tag=None):
        ion_name_r = r'([a-z]+_\d+)' if ion_name is None else re.escape(
            ion_name)
        central_wavelength_r = r'(\d+\.\d+)' if central_wavelength is None else re.escape(
            f'{central_wavelength:.3f}')
        atmosphere_name_r = r'(.+)' if atmosphere_name is None else re.escape(
            atmosphere_name)
        snap_num_r = r'(\d+)' if snap_num is None else re.escape(snap_num)
        tag_r = re.escape(join_tags(tag, as_str=True))
        return ion_name_r + '_' + central_wavelength_r + '_' + atmosphere_name_r + '_' + snap_num_r + tag_r

    def save(self, file_path):
        with open(file_path, 'wb') as f:
            np.savez_compressed(f,
                                ion_name=self.ion_name,
                                abundance=self.ion_properties.abundance,
                                central_wavelength=self.central_wavelength,
                                **self.__quantities)

    @staticmethod
    def load(file_path):
        data = np.load(file_path)
        non_quantity_names = ['ion_name', 'central_wavelength']
        extra_quantity_names = [
            name for name in data.files if name not in non_quantity_names
        ]
        ion_name = data['ion_name'].item()
        abundance = data['abundance'].item()
        central_wavelength = data['central_wavelength'].item()
        spectral_line = SpectralLine(
            IonProperties(ion_name, abundance), central_wavelength,
            {name: data[name]
             for name in extra_quantity_names})
        data.close()
        return spectral_line

    def derive_quantity(self, quantity_name):
        quantity = self.__derived_quantities[quantity_name]()
        self.__quantities[quantity_name] = quantity
        return quantity

    def get(self, name, mapper=None, scale=None):
        if name in self.__quantities:
            quantity = self.__quantities[name]
        elif name in self.__derived_quantities:
            quantity = self.derive_quantity(name)
        else:
            raise ValueError(f'Invalid spectral line quantity {name}')
        quantity = quantity if mapper is None else mapper(quantity)
        return quantity if scale is None else quantity * scale

    def select(self, quantity_name, **kwargs):
        if quantity_name is not None and quantity_name not in self.__quantities:
            self.derive_quantity(quantity_name)
        self.__selected_quantity_name = quantity_name
        self.__selected_quantity_params = kwargs
        return self

    def __merge_quantities(self, quantities, operation):
        return {
            name: operation(self.quantities[name], quantities[name])
            for name in set(self.quantities).intersection(quantities)
        }

    def __perform_operation(self, other, operation, inplace=False):
        if isinstance(other, self.__class__):
            if self.selected_quantity_name is None and other.selected_quantity_name is None:
                if inplace:
                    raise ValueError(
                        'Operation requires a quantity to be selected')
                return SpectralLine(
                    self.ion_properties, self.central_wavelength,
                    self.__merge_quantities(other.quantities, operation))
            else:
                if self.selected_quantity_name is not None and other.selected_quantity_name is not None:
                    assert other.selected_quantity_name == self.selected_quantity_name
                    assert other.selected_quantity_params == self.selected_quantity_params
                elif self.selected_quantity_name is None:
                    self.select(other.selected_quantity_name,
                                **other.selected_quantity_params)
                else:
                    other.select(self.selected_quantity_name,
                                 **self.selected_quantity_params)

                this_quantity = self.get(self.selected_quantity_name,
                                         **self.selected_quantity_params)
                other_quantity = other.get(self.selected_quantity_name,
                                           **self.selected_quantity_params)
                result = operation(this_quantity, other_quantity)
                return self if inplace else result
        elif other is not None:
            assert self.selected_quantity_name is not None
            this_quantity = self.get(self.selected_quantity_name,
                                     **self.selected_quantity_params)
            result = operation(this_quantity, other)
            return self if inplace else result
        else:
            return None

    def __add__(self, term):
        return self.__perform_operation(term, lambda a, b: a + b)

    def __sub__(self, term):
        return self.__perform_operation(term, lambda a, b: a - b)

    def __mul__(self, factor):
        return self.__perform_operation(factor, lambda a, b: a * b)

    def __truediv__(self, divisor):
        return self.__perform_operation(divisor, lambda a, b: a / b)

    def __pow__(self, power):
        return self.__perform_operation(power, lambda a, b: a**b)

    def __iadd__(self, term):
        return self.__perform_operation(term,
                                        lambda a, b: a.__iadd__(b),
                                        inplace=True)

    def __isub__(self, term):
        return self.__perform_operation(term,
                                        lambda a, b: a.__isub__(b),
                                        inplace=True)

    def __imul__(self, term):
        return self.__perform_operation(term,
                                        lambda a, b: a.__imul__(b),
                                        inplace=True)

    def __itruediv__(self, term):
        return self.__perform_operation(term,
                                        lambda a, b: a.__itruediv__(b),
                                        inplace=True)

    def compute_statistics(self):
        return dict(spectral_line=self.description,
                    mean_intensity=np.nanmean(self.get('intensity')),
                    mean_doppler_velocity=np.nanmean(
                        self.get('doppler_velocity', scale=1e-5)),
                    mean_width_velocity=np.nanmean(
                        self.get('width_velocity', scale=1e-5)))

    def print_statistics(self):
        quantities = dict(spectral_line='Spectral line',
                          mean_intensity='Mean intensity [erg/s/sr/cm^2]',
                          mean_doppler_velocity='Mean Doppler velocity [km/s]',
                          mean_width_velocity='Mean line width [km/s]')
        statistics = self.compute_statistics()
        headers = []
        entries = []
        for quantity, description in quantities.items():
            headers.append(description)
            entries.append(statistics[quantity])
        print(tabulate([entries], headers=headers, tablefmt='orgtbl'))

    def plot(self,
             x_coords,
             y_coords,
             quantity_name,
             mapper=None,
             scale=None,
             **kwargs):
        return fields.ScalarField2(
            fields.Coords2(x_coords, y_coords),
            self.get(quantity_name, mapper=mapper, scale=scale)).plot(**kwargs)

    def __compute_doppler_velocity(self):
        return -units.CLIGHT * self.get(
            'doppler_shift'
        ) / self.central_wavelength  # Positive upwards [cm/s]

    def __compute_width_velocity(self):
        return units.CLIGHT * self.get('width') / self.central_wavelength


class SpectralLineArray(np.ndarray):
    def __new__(cls, a):
        obj = np.asarray(a).view(cls)
        return obj

    def select(self, quantity_name, **kwargs):
        np.vectorize(lambda spectral_line: None
                     if spectral_line is None else spectral_line.select(
                         quantity_name, **kwargs),
                     otypes=[object])(self)
        return self

    def get(self, quantity_name, **kwargs):
        return np.vectorize(lambda spectral_line: None
                            if spectral_line is None else spectral_line.get(
                                quantity_name, **kwargs),
                            otypes=[object])(self)

    def applied(self, func):
        return np.vectorize(lambda item: None if item is None else func(item),
                            otypes=[object])(self)

    def to_dense_array(self):
        dense_arr = np.ravel(self)
        inner_shape = dense_arr[0].shape
        return np.concatenate(dense_arr).reshape(*self.shape, *inner_shape)

    @classmethod
    def from_dense_array(cls, dense_arr, n_inner_dims=2):
        outer_shape = dense_arr.shape[:-n_inner_dims]
        inner_shape = dense_arr.shape[n_inner_dims:]
        arr = np.empty(np.prod(outer_shape), dtype=object)
        arr[:] = list(dense_arr.reshape((-1, *inner_shape)))
        return cls(arr.reshape(outer_shape))


class Spectrum:
    def __init__(self,
                 line_names,
                 atmosphere_names,
                 snap_nums,
                 spectral_lines=None):
        self.__line_names = np.atleast_1d(line_names)
        self.__atmosphere_names = np.atleast_1d(atmosphere_names)
        self.__snap_nums = np.atleast_1d(snap_nums)

        self.__mesh_line_names, self.__mesh_atmosphere_names, self.__mesh_snap_nums = np.meshgrid(
            self.__line_names,
            self.__atmosphere_names,
            self.__snap_nums,
            indexing='ij')
        shape = self.__mesh_line_names.shape

        self.__line_name_indices = {
            line_name: idx
            for idx, line_name in enumerate(self.line_names)
        }
        self.__atmosphere_name_indices = {
            atmosphere_name: idx
            for idx, atmosphere_name in enumerate(self.atmosphere_names)
        }
        self.__snap_num_indices = {
            snap_num: idx
            for idx, snap_num in enumerate(self.snap_nums)
        }

        if spectral_lines is None:
            self.__spectral_lines = SpectralLineArray(np.full(shape, None))
        else:
            assert isinstance(
                spectral_lines,
                SpectralLineArray) and spectral_lines.shape == shape
            self.__spectral_lines = spectral_lines

    @property
    def line_names(self):
        return self.__line_names

    @property
    def atmosphere_names(self):
        return self.__atmosphere_names

    @property
    def snap_nums(self):
        return self.__snap_nums

    @property
    def spectral_lines(self):
        return self.__spectral_lines

    def save(self, file_path):
        with open(file_path, 'wb') as f:
            joblib.dump(self, f)

    @staticmethod
    def load(file_path):
        with open(file_path, 'rb') as f:
            spectrum = joblib.load(f)
        return spectrum

    def save_spectral_lines(self, dir_path, identifier=None, tag=None):
        dir_path = pathlib.Path(dir_path)

        if identifier is None:
            for idx, spectral_line in np.ravel(self.spectral_lines):
                if spectral_line is None:
                    continue
                _, atmosphere_name, snap_num = self.identifier(idx)
                spectral_line.save(
                    dir_path /
                    f'{spectral_line.get_tag(atmosphere_name, snap_num, tag=tag)}.npz'
                )
        else:
            idx = self.idx(*identifier)
            for atmosphere_name, snap_num, spectral_line in zip(
                    np.atleast_1d(self.__mesh_atmosphere_names[idx]),
                    np.atleast_1d(self.__mesh_snap_nums[idx]),
                    np.atleast_1d(self.spectral_lines[idx])):
                if spectral_line is None:
                    continue
                spectral_line.save(
                    dir_path /
                    f'{spectral_line.get_tag(atmosphere_name, snap_num, tag=tag)}.npz'
                )

    @staticmethod
    def from_files(dir_path,
                   line_names,
                   atmosphere_names,
                   snap_nums,
                   tag=None):
        dir_path = pathlib.Path(dir_path)

        pattern = re.compile(SpectralLine.get_tag_regex(tag=tag) + r'\.npz')

        def parse_filename(filename):
            match = re.match(pattern, filename)
            if match:
                ion_name = match[1]
                central_wavelength = float(match[2])
                atmosphere_name = match[3]
                snap_num = int(match[4])
                return ion_name, central_wavelength, atmosphere_name, snap_num
            else:
                return None

        spectrum = Spectrum(line_names, atmosphere_names, snap_nums)

        for filename in next(os.walk(dir_path))[2]:
            parsed = parse_filename(filename)
            if parsed:
                line_name = IonProperties.get_line_name(*parsed[:2])
                atmosphere_name = parsed[2]
                snap_num = parsed[3]
                if line_name in spectrum.line_names and atmosphere_name in spectrum.atmosphere_names and snap_num in spectrum.snap_nums:
                    spectral_line = SpectralLine.load(dir_path / filename)
                    spectrum[spectral_line.name, atmosphere_name,
                             snap_num] = spectral_line

        return spectrum

    def idx(self, line_name, atmosphere_name, snap_num):
        if isinstance(line_name, list):
            line_name_idx = [
                self.__line_name_indices[name] for name in line_name
            ]
        elif isinstance(line_name, str):
            line_name_idx = self.__line_name_indices[line_name]
        else:
            line_name_idx = line_name

        if isinstance(atmosphere_name, list):
            atmosphere_name_idx = [
                self.__atmosphere_name_indices[name]
                for name in atmosphere_name
            ]
        elif isinstance(atmosphere_name, str):
            atmosphere_name_idx = self.__atmosphere_name_indices[
                atmosphere_name]
        else:
            atmosphere_name_idx = atmosphere_name

        if isinstance(snap_num, list):
            snap_num_idx = [self.__snap_num_indices[name] for name in snap_num]
        elif isinstance(snap_num, (int, np.int32, np.int64)):
            snap_num_idx = self.__snap_num_indices[snap_num]
        else:
            snap_num_idx = snap_num

        return (line_name_idx, atmosphere_name_idx, snap_num_idx)

    def identifier(self, idx):
        line_idx, atmos_idx, snap_idx = np.unravel_index(
            idx, self.spectral_lines.shape)
        return self.line_names[line_idx], self.atmosphere_names[
            atmos_idx], self.snap_nums[snap_idx]

    def get_line_names_for_ion(self, ion_name):
        return list(
            filter(lambda name: name.startswith(ion_name), self.line_names))

    def __getitem__(self, identifier):
        return self.spectral_lines[self.idx(*identifier)]

    def __setitem__(self, identifier, spectral_line):
        self.spectral_lines[self.idx(*identifier)] = spectral_line

    def print_statistics(self, atmosphere_name, snap_num):
        quantities = dict(spectral_line='Spectral line',
                          mean_intensity='Mean intensity [erg/s/sr/cm^2]',
                          mean_doppler_velocity='Mean Doppler velocity [km/s]',
                          mean_width_velocity='Mean line width [km/s]')
        headers = list(quantities.values())
        entries = []
        spectral_lines = self[:, atmosphere_name, snap_num]
        for spectral_line in spectral_lines:
            if spectral_line is None:
                continue
            statistics = spectral_line.compute_statistics()
            entries.append([])
            for quantity in quantities:
                entries[-1].append(statistics[quantity])
        print(tabulate(entries, headers=headers, tablefmt='orgtbl'))


def join_tags(*tags, as_str=False):
    combined = ''
    for tag in tags:
        if tag is not None:
            combined += f'_{tag}'
    if not as_str:
        if combined == '':
            return None
        else:
            return combined[1:]
    else:
        return combined
