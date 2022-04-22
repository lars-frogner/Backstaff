import sys
import os
import time
import pickle
import contextlib

import numpy as np
import scipy.interpolate
from numba import njit

with contextlib.redirect_stdout(None):
    import ChiantiPy.tools.util as ch_util
    import ChiantiPy.tools.io as ch_io
    import ChiantiPy.tools.data as ch_data

try:
    import backstaff.units as units
    import backstaff.array_utils as array_utils
except ModuleNotFoundError:
    import units
    import array_utils


class SpeciesRatiosCacher:
    def __init__(self):
        self.__ratios = {}

    def __call__(self, abundance_file, verbose=False):
        if abundance_file not in self.__ratios:
            self.__ratios[abundance_file] = SpeciesRatios(
                abundance_file, verbose=verbose
            )
        return self.__ratios[abundance_file]


SPECIES_RATIOS = SpeciesRatiosCacher()


class SpeciesRatios:
    @staticmethod
    def default(verbose=False):
        return SPECIES_RATIOS("sun_coronal_2012_schmelz_ext", verbose=verbose)

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

    def compute_proton_to_electron_ratios(self, temperatures, temperature_is_log=False):
        return scipy.interpolate.splev(
            temperatures if temperature_is_log else np.log10(temperatures),
            self.__proton_to_electron_ratio_spline,
            ext=1,
        )

    def compute_hydrogen_to_electron_ratios(
        self, temperatures, temperature_is_log=False
    ):
        return scipy.interpolate.splev(
            temperatures if temperature_is_log else np.log10(temperatures),
            self.__hydrogen_to_electron_ratio_spline,
            ext=0,
        )

    def __compute_splines_for_proton_and_hydrogen_to_electron_ratios(self):
        self.info("Constructing splines for species ratios")
        start_time = time.time()

        all_abundances = np.asarray(
            ch_io.abundanceRead(abundancename=self.abundance_file)["abundance"]
        )
        abundances = all_abundances[all_abundances > 0]

        electron_number = np.zeros(len(ch_data.IoneqAll["ioneqTemperature"]))
        for i in range(abundances.size):
            for z in range(1, i + 2):
                electron_number += (
                    z * ch_data.IoneqAll["ioneqAll"][i, z, :] * abundances[i]
                )

        proton_to_electron_ratios = (
            abundances[0] * ch_data.IoneqAll["ioneqAll"][0, 1, :] / electron_number
        )
        neutral_hydrogen_to_electron_ratios = (
            abundances[0] * ch_data.IoneqAll["ioneqAll"][0, 0, :] / electron_number
        )
        hydrogen_to_electron_ratios = (
            proton_to_electron_ratios + neutral_hydrogen_to_electron_ratios
        )

        self.__proton_to_electron_ratio_spline = scipy.interpolate.splrep(
            np.log10(ch_data.IoneqAll["ioneqTemperature"]),
            proton_to_electron_ratios,
            s=0,
        )

        self.__hydrogen_to_electron_ratio_spline = scipy.interpolate.splrep(
            np.log10(ch_data.IoneqAll["ioneqTemperature"]),
            hydrogen_to_electron_ratios,
            s=0,
        )

        self.info(f"Took {time.time() - start_time:g} s")


class IonAtmosphere:
    def __init__(
        self,
        temperatures,
        electron_densities,
        species_ratios=None,
        radiation_temperature=None,
        distance_from_center=None,
        compute_proton_densities=True,
        compute_hydrogen_densities=True,
        verbose=False,
    ):
        self.__temperatures = temperatures  # [K]
        self.__electron_densities = electron_densities  # [1/cm^3]
        self.verbose = verbose
        self.__species_ratios = (
            SpeciesRatios.default(verbose=verbose)
            if species_ratios is None
            else species_ratios
        )
        self.__proton_densities = (
            self.compute_proton_densities(self.temperatures, self.electron_densities)
            if compute_proton_densities
            else None
        )
        self.__hydrogen_densities = (
            self.compute_hydrogen_densities(self.temperatures, self.electron_densities)
            if compute_hydrogen_densities
            else None
        )
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
        with open(file_path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(file_path):
        with open(file_path, "rb") as f:
            atmos = pickle.load(f)
        return atmos

    def compute_proton_densities(self, temperatures, electron_densities, **kwargs):
        self.info(f"Computing {temperatures.size} proton densities")
        start_time = time.time()

        proton_densities = (
            self.species_ratios.compute_proton_to_electron_ratios(
                temperatures, **kwargs
            )
            * electron_densities
        )

        self.info(f"Took {time.time() - start_time:g} s")
        return proton_densities

    def compute_hydrogen_densities(self, temperatures, electron_densities, **kwargs):
        self.info(f"Computing {temperatures.size} hydrogen ratios")
        start_time = time.time()

        hydrogen_densities = (
            self.species_ratios.compute_hydrogen_to_electron_ratios(
                temperatures, **kwargs
            )
            * electron_densities
        )

        self.info(f"Took {time.time() - start_time:g} s")
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
        return (
            units.KBOLTZMANN
            / self.compute_atomic_mass()
            * (central_wavelength / units.CLIGHT) ** 2
        ) * temperature  # [cm^2]

    def compute_atomic_mass(self):
        return 2 * self.nuclear_charge * units.AMU  # [g]

    @staticmethod
    def get_line_name(ion_name, central_wavelength):
        return f"{ion_name}_{central_wavelength:.3f}"

    @staticmethod
    def parse_line_name(line_name):
        splitted = line_name.split("_")
        ion_name = "_".join(splitted[:2])
        central_wavelength = float(splitted[2])
        return ion_name, central_wavelength

    @staticmethod
    def get_line_description(spectroscopic_name, central_wavelength):
        return f"{spectroscopic_name} {central_wavelength:.3f} Å"

    @staticmethod
    def line_name_to_description(line_name):
        ion_name, central_wavelength = IonProperties.parse_line_name(line_name)
        _, _, spectroscopic_name = IonProperties.parse_ion_name(ion_name)
        return IonProperties.get_line_description(
            spectroscopic_name, central_wavelength
        )

    @staticmethod
    def parse_ion_name(ion_name):
        parsed = ch_util.convertName(ion_name)
        nuclear_charge = parsed["Z"]
        ionization_stage = parsed["Ion"]
        spectroscopic_name = ch_util.zion2spectroscopic(
            nuclear_charge, ionization_stage
        )
        return nuclear_charge, ionization_stage, spectroscopic_name

    def __set_ion_name(self, ion_name):
        self.__ion_name = ion_name
        (
            self.__nuclear_charge,
            self.__ionization_stage,
            self.__spectroscopic_name,
        ) = self.__class__.parse_ion_name(ion_name)

    def __find_ionization_potential(self):
        if self.ionization_stage <= self.nuclear_charge:
            self.__ionization_potential = ch_data.Ip[
                self.nuclear_charge - 1, self.ionization_stage - 1
            ]
            self.__first_ionization_potential = ch_data.Ip[self.nuclear_charge - 1, 0]

    def __read_abundance(self, abundance_file):
        self.__abundance_file = abundance_file
        # Abundance of this element relative to hydrogen
        self.__abundance = ch_data.Abundance[abundance_file]["abundance"][
            self.nuclear_charge - 1
        ]


class Ion:
    def __init__(
        self,
        ion_name,
        atmosphere,
        max_levels_before_pruning=None,
        probe_atmosphere=None,
        pruning_min_in_to_average_ratio=1e-1,
        pruning_min_in_to_out_ratio=1e-1,
        verbose=False,
    ):
        self.__filter_levels = False
        self.verbose = verbose

        self.__atmos = atmosphere
        self.__properties = IonProperties(ion_name, atmosphere.abundance_file)
        self.__read_ion_data()

        if (
            max_levels_before_pruning is not None
            and self.n_levels > max_levels_before_pruning
        ):
            probe_atmosphere = (
                atmosphere if probe_atmosphere is None else probe_atmosphere
            )
            self.info(
                f"Pruning {self.n_levels} levels for {probe_atmosphere.n_values} conditions"
            )
            important_levels = self.find_important_levels(
                probe_atmosphere,
                min_in_to_average_ratio=pruning_min_in_to_average_ratio,
                min_in_to_out_ratio=pruning_min_in_to_out_ratio,
            )
            self.restrict_to_levels(important_levels)
            self.info(f"Pruned to {important_levels.size} levels")

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
        return self.__getattr("levels_mask", slice(None))

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
        return self.__getattr("populations", None)

    @property
    def emissivities(self):
        return self.__getattr("emissivities", None)

    @property
    def emissivity_line_indices(self):
        return self.__getattr("emissivity_line_indices", None)

    @property
    def n_lines_with_emissivity(self):
        emissivity_line_indices = self.emissivity_line_indices
        if emissivity_line_indices is None:
            return None
        return (
            self.n_lines
            if emissivity_line_indices == slice(None)
            else emissivity_line_indices.size
        )

    def save(self, file_path):
        with open(file_path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(file_path):
        with open(file_path, "rb") as f:
            ion = pickle.load(f)
        return ion

    def info(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def find_line_index(self, wavelength):
        line_idx = np.argmin(np.abs(self.central_wavelengths - wavelength * 1e-8))
        closest_wavelength = self.central_wavelengths[line_idx] * 1e8  # [Å]
        if not np.allclose(closest_wavelength, wavelength):
            print(
                f"Warning: No line sufficiently close to requested wavelength of {wavelength:.3f} Å\nfor ion {self.ion_name}, using closest at {closest_wavelength:g} Å",
                file=sys.stderr,
            )
        return line_idx

    def find_line_indices(self, wavelengths):
        if wavelengths is None:
            indices = np.arange(self.n_lines)
        else:
            indices = np.array(
                [self.find_line_index(wavelength) for wavelength in wavelengths],
                dtype=int,
            )
        return indices

    def find_levels_for_lines(self, line_indices):
        return np.unique(
            np.concatenate(
                (self.lower_levels[line_indices], self.upper_levels[line_indices])
            )
        )

    def print_lines(self):
        print(f"{self.spectroscopic_name} lines")
        for line_idx in range(self.n_lines):
            self.print_line(line_idx)

    def print_line(self, line_idx):
        print(
            f"{line_idx:3d}: λ = {self.central_wavelengths[line_idx]*1e8:8.3f} Å ({self.upper_levels[line_idx]:3d} -> {self.lower_levels[line_idx]:3d})"
        )

    def find_important_levels(
        self, atmosphere, min_in_to_average_ratio=1e-1, min_in_to_out_ratio=1e-1
    ):
        self.unrestrict()

        original_atmos = self.atmos
        self.__atmos = atmosphere
        rate_matrices = self.__construct_rate_matrices()
        self.__atmos = original_atmos

        mean_rate_matrix = np.mean(rate_matrices, axis=0)
        mean_in_rates, mean_out_rates, mean_rate = compute_transition_rate_statistics(
            mean_rate_matrix
        )

        # An important level must satisfy at least one of the following criteria:
        # 1. There are significant rates into the level, so the presence of this level provides a drainage for the population in other levels.
        # 2. The incoming rates are not too small compared to the outgoing ones, so the population of this level will not be negligible.
        # Otherwise the level will have a small population and will not influence the populations of other levels.
        important_levels = (
            np.flatnonzero(
                np.logical_or(
                    mean_in_rates > min_in_to_average_ratio * mean_rate,
                    mean_in_rates > min_in_to_out_ratio * mean_out_rates,
                )
            )
            + 1
        )
        return important_levels

    def restrict_to_levels(self, included_levels):
        self.__levels_mask = np.flatnonzero(np.isin(self.levels, included_levels))

        self.__transitions_mask = np.flatnonzero(
            np.logical_and(
                np.isin(self.lower_levels, included_levels),
                np.isin(self.upper_levels, included_levels),
            )
        )

        if self.__scups_data is not None:
            self.__scups_transitions_mask = np.flatnonzero(
                np.logical_and(
                    np.isin(self.scups_lower_levels, included_levels),
                    np.isin(self.scups_upper_levels, included_levels),
                )
            )

        if self.__psplups_data is not None:
            self.__psplups_transitions_mask = np.flatnonzero(
                np.logical_and(
                    np.isin(self.psplups_lower_levels, included_levels),
                    np.isin(self.psplups_upper_levels, included_levels),
                )
            )

        self.restrict()

    def restrict(self):
        assert self.__hasattr("levels_mask")
        self.__unrestricted_populations = self.populations
        self.__populations = self.__getattr("restricted_populations", None)
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
            f"Solving {rate_matrices.shape[0]} sets of level population equations"
        )
        start_time = time.time()

        # Proportion of ions in the given ionization state that are in each energy level (n_levels, n_conditions)
        self.__populations = np.linalg.solve(rate_matrices, rhs[np.newaxis, :]).T

        self.__populations[self.__populations < 0] = 0

        self.info(f"Took {time.time() - start_time:g} s")

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

        self.info("Sampling ionization fractions")
        start_time = time.time()

        table_temperatures = ch_data.IoneqAll["ioneqTemperature"]
        table_ionization_fractions = ch_data.IoneqAll["ioneqAll"][
            self.nuclear_charge - 1, self.ionization_stage - 1
        ].squeeze()

        valid = table_ionization_fractions > 0
        inside_lower = temperatures >= table_temperatures[valid].min()
        inside_upper = temperatures <= table_temperatures[valid].max()
        inside = np.logical_and(inside_lower, inside_upper)

        spline = scipy.interpolate.splrep(
            np.log10(table_temperatures[valid]),
            np.log10(table_ionization_fractions[valid]),
            s=0,
        )
        log_ionization_fractions = scipy.interpolate.splev(
            np.log10(temperatures[inside]), spline
        )

        # Proportion of ions of this element that are in the given ionization state (n_conditions,)
        ionization_fractions = np.zeros_like(temperatures)
        ionization_fractions[inside] = 10**log_ionization_fractions

        if not is_flat:
            ionization_fractions = ionization_fractions.reshape(shape)

        self.info(f"Took {time.time() - start_time:g} s")

        return ionization_fractions

    def compute_emissivities(self, line_indices=None):
        if self.__getattr("populations", None) is None:
            self.compute_populations()
        if self.__getattr("ionization_fractions", None) is None:
            self.__ionization_fractions = self.sample_ionization_fractions()

        if line_indices is None:
            line_indices = slice(None)
            self.__emissivity_line_indices = line_indices
        else:
            self.__emissivity_line_indices = np.asarray(line_indices, dtype=int)

        self.info("Computing emissivites")
        start_time = time.time()

        self.__emissivities = (
            (
                (self.abundance * units.HPLANCK * units.CLIGHT / (4 * np.pi))
                / self.central_wavelengths[line_indices, np.newaxis]
            )
            * self.populations[self.upper_level_indices[line_indices], :]
            * self.ionization_fractions[np.newaxis, :]
            * self.atmos.hydrogen_densities[np.newaxis, :]
            * self.transition_probabilities[line_indices, np.newaxis]
        )

        self.info(f"Took {time.time() - start_time:g} s")

        return self.__emissivities

    def __getattr(self, name, *args):
        return getattr(self, f"_Ion__{name}", *args)

    def __hasattr(self, name):
        return hasattr(self, f"_Ion__{name}")

    def __read_ion_data(self):
        self.__read_wgfa_file()
        self.__read_elvlc_file()
        self.__read_scups_file()
        self.__read_psplups_file()

    def __read_elvlc_file(self):
        if not self.__hasattr("elevlc_data"):
            self.__elevlc_data = ch_io.elvlcRead(self.ion_name)

        levels = np.asarray(self.__elevlc_data["lvl"], int)
        mask = np.isin(levels, self.__levels_with_transitions)

        self.__levels = self.__levels_filtered_arr(levels[mask])
        self.__levels_missing = self.__levels.size < levels.size

        self.__elvlc_eryd = self.__levels_filtered_arr(
            np.asarray(self.__elevlc_data["eryd"])[mask]
        )
        self.__elvlc_erydth = self.__levels_filtered_arr(
            np.asarray(self.__elevlc_data["erydth"])[mask]
        )
        self.__elvlc_mult = self.__levels_filtered_arr(
            np.asarray(self.__elevlc_data["mult"])[mask]
        )
        self.__elvlc_ecm = self.__levels_filtered_arr(
            np.asarray(self.__elevlc_data["ecm"])[mask]
        )

        self.__levels_sorter = np.argsort(self.__levels)

    def __find_level_indices(self, levels):
        return (
            self.__levels_sorter[
                np.searchsorted(self.__levels, levels, sorter=self.__levels_sorter)
            ]
            if self.__levels_missing
            else levels - 1
        )

    def __read_wgfa_file(self):
        if not self.__hasattr("wgfa_data"):
            self.__wgfa_data = ch_io.wgfaRead(self.ion_name)

        central_wavelengths = np.asarray(self.__wgfa_data["wvl"], np.float64)
        mask = central_wavelengths > 0  # Excludes autoionization lines

        lower_levels = np.asarray(self.__wgfa_data["lvl1"], int)[mask]
        upper_levels = np.asarray(self.__wgfa_data["lvl2"], int)[mask]

        self.__levels_with_transitions = np.unique(
            np.concatenate((lower_levels, upper_levels))
        )

        # Wavelength of each transition from upper to lower energy level (n_lines,) [cm]
        self.__central_wavelengths = self.__transitions_filtered_arr(
            central_wavelengths[mask] * 1e-8
        )

        # All lower and upper energy levels involved in each transition (n_lines,)
        self.__lower_levels = self.__transitions_filtered_arr(lower_levels)
        self.__upper_levels = self.__transitions_filtered_arr(upper_levels)

        # Spontaneous transition probabilities (n_lines,) [1/s]
        self.__transition_probabilities = self.__transitions_filtered_arr(
            np.asarray(self.__wgfa_data["avalue"], np.float64)[mask]
        )

    def __read_scups_file(self):
        if os.path.isfile(ch_util.ion2filename(self.ion_name) + ".scups"):
            if not self.__hasattr("scups_data"):
                self.__scups_data = ch_io.scupsRead(self.ion_name)

            scups_lower_levels = np.asarray(self.__scups_data["lvl1"], int)
            scups_upper_levels = np.asarray(self.__scups_data["lvl2"], int)

            mask = np.logical_and(
                np.isin(scups_lower_levels, self.__levels_with_transitions),
                np.isin(scups_upper_levels, self.__levels_with_transitions),
            )

            self.__scups_lower_levels = self.__scups_transitions_filtered_arr(
                scups_lower_levels[mask]
            )
            self.__scups_upper_levels = self.__scups_transitions_filtered_arr(
                scups_upper_levels[mask]
            )
            self.__scups_ttype = self.__scups_transitions_filtered_arr(
                np.asarray(self.__scups_data["ttype"], int)[mask]
            )
            self.__scups_cups = self.__scups_transitions_filtered_arr(
                np.asarray(self.__scups_data["cups"])[mask]
            )
            self.__scups_xs = self.__scups_transitions_filtered_list(
                [x for i, x in enumerate(self.__scups_data["btemp"]) if mask[i]]
            )
            self.__scups_scups = self.__scups_transitions_filtered_list(
                [x for i, x in enumerate(self.__scups_data["bscups"]) if mask[i]]
            )
            self.__scups_de = self.__scups_transitions_filtered_arr(
                np.asarray(self.__scups_data["de"])[mask]
            )
        else:
            self.__scups_data = None

    def __read_psplups_file(self):
        if os.path.isfile(ch_util.ion2filename(self.ion_name) + ".psplups"):
            if not self.__hasattr("psplups_data"):
                self.__psplups_data = ch_io.splupsRead(
                    self.ion_name, filetype="psplups"
                )

            psplups_lower_levels = np.asarray(self.__psplups_data["lvl1"], int)
            psplups_upper_levels = np.asarray(self.__psplups_data["lvl2"], int)

            mask = np.logical_and(
                np.isin(psplups_lower_levels, self.__levels_with_transitions),
                np.isin(psplups_upper_levels, self.__levels_with_transitions),
            )

            self.__psplups_lower_levels = self.__psplups_transitions_filtered_arr(
                psplups_lower_levels[mask]
            )
            self.__psplups_upper_levels = self.__psplups_transitions_filtered_arr(
                psplups_upper_levels[mask]
            )
            self.__psplups_ttype = self.__psplups_transitions_filtered_arr(
                np.asarray(self.__psplups_data["ttype"], int)[mask]
            )
            self.__psplups_cups = self.__psplups_transitions_filtered_arr(
                np.asarray(self.__psplups_data["cups"])[mask]
            )
            self.__psplups_nspls = self.__psplups_transitions_filtered_list(
                [x for i, x in enumerate(self.__psplups_data["nspl"]) if mask[i]]
            )
            self.__psplups_splups = self.__psplups_transitions_filtered_list(
                [x for i, x in enumerate(self.__psplups_data["splups"]) if mask[i]]
            )
        else:
            self.__psplups_data = None

    def __construct_rate_matrices(self):
        self.info(
            f"Building {self.atmos.n_values} rate matrices with shape {self.n_levels} x {self.n_levels}"
        )
        start_time = time.time()

        matrix_shape = (self.n_levels, self.n_levels)
        rate_matrix = np.zeros(matrix_shape, np.float64)

        l1 = self.lower_level_indices
        l2 = self.upper_level_indices
        array_utils.add_values_in_matrix(
            rate_matrix, l1, l2, self.transition_probabilities
        )
        array_utils.subtract_values_in_matrix(
            rate_matrix, l2, l2, self.transition_probabilities
        )

        # Photo-excitation and stimulated emission
        if self.atmos.radiation_temperature is not None:
            self.info(
                f"Including photoexcitation and stimulated emission at {self.atmos.radiation_temperature} K"
            )
            assert self.atmos.distance_from_center is not None
            dilute = ch_util.dilute(self.atmos.distance_from_center)

            # Don't include autoionization lines
            mask = np.abs(self.central_wavelengths) > 0
            l1 = l1[mask]
            l2 = l2[mask]

            de = (units.HPLANCK * units.CLIGHT) * (
                self.__elvlc_ecm[l2] - self.__elvlc_ecm[l1]
            )
            dekt = de / (units.KBOLTZMANN * self.atmos.radiation_temperature)

            # Photoexcitation
            phex_values = (
                self.transition_probabilities[mask]
                * dilute
                * (self.__elvlc_mult[l2] / self.__elvlc_mult[l1])
                / (np.exp(dekt) - 1.0)
            )

            array_utils.add_values_in_matrix(rate_matrix, l2, l1, phex_values)
            array_utils.subtract_values_in_matrix(rate_matrix, l1, l1, phex_values)

            # Stimulated emission
            stem_values = (
                self.transition_probabilities[mask] * dilute / (np.exp(-dekt) - 1.0)
            )
            array_utils.add_values_in_matrix(rate_matrix, l1, l2, stem_values)
            array_utils.subtract_values_in_matrix(rate_matrix, l2, l2, stem_values)

        rate_matrices = np.repeat(
            rate_matrix[np.newaxis, :], self.atmos.n_values, axis=0
        )

        if self.__scups_data is not None:
            self.info("Including electron collisions")
            (
                _,
                excitation_rates,
                deexcitation_rates,
            ) = self.__compute_collision_strengths(for_proton=False)
            l1_scups = self.scups_lower_level_indices
            l2_scups = self.scups_upper_level_indices
            dex_values = (
                self.atmos.electron_densities[np.newaxis, :] * deexcitation_rates
            )
            ex_values = self.atmos.electron_densities[np.newaxis, :] * excitation_rates
            array_utils.add_values_in_matrices(
                rate_matrices, l1_scups, l2_scups, dex_values
            )
            array_utils.add_values_in_matrices(
                rate_matrices, l2_scups, l1_scups, ex_values
            )
            array_utils.subtract_values_in_matrices(
                rate_matrices, l1_scups, l1_scups, ex_values
            )
            array_utils.subtract_values_in_matrices(
                rate_matrices, l2_scups, l2_scups, dex_values
            )

        if self.__psplups_data is not None:
            self.info("Including proton collisions")
            (
                _,
                excitation_rates,
                deexcitation_rates,
            ) = self.__compute_collision_strengths(for_proton=True)
            l1_psplups = self.psplups_lower_level_indices
            l2_psplups = self.psplups_upper_level_indices
            pdex_values = (
                self.atmos.proton_densities[np.newaxis, :] * deexcitation_rates
            )
            pex_values = self.atmos.proton_densities[np.newaxis, :] * excitation_rates
            array_utils.add_values_in_matrices(
                rate_matrices, l1_psplups, l2_psplups, pdex_values
            )
            array_utils.add_values_in_matrices(
                rate_matrices, l2_psplups, l1_psplups, pex_values
            )
            array_utils.subtract_values_in_matrices(
                rate_matrices, l1_psplups, l1_psplups, pex_values
            )
            array_utils.subtract_values_in_matrices(
                rate_matrices, l2_psplups, l2_psplups, pdex_values
            )

        self.info(f"Took {time.time() - start_time:g} s")

        return rate_matrices

    def __compute_collision_strengths(self, for_proton=False):
        if for_proton:
            assert self.__psplups_data is not None
            n_transitions = self.psplups_lower_levels.size
            lower_levels = self.psplups_lower_levels
            upper_levels = self.psplups_upper_levels
            ttypes = self.__psplups_ttype
            cups = self.__psplups_cups
            xs = [np.arange(nspl) / (nspl - 1) for nspl in self.__psplups_nspls]
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

        collision_strengths = np.zeros((n_transitions, self.atmos.n_values), np.float64)
        excitation_rates = np.zeros((n_transitions, self.atmos.n_values), np.float64)
        deexcitation_rates = np.zeros((n_transitions, self.atmos.n_values), np.float64)

        elvlc = np.where(self.__elvlc_eryd >= 0, self.__elvlc_eryd, self.__elvlc_erydth)

        lower_level_indices = self.__find_level_indices(lower_levels)
        upper_level_indices = self.__find_level_indices(upper_levels)

        if for_proton:
            de = elvlc[upper_level_indices] - elvlc[lower_level_indices]
        else:
            de = self.__scups_de

        kte = (
            units.KBOLTZMANN
            * self.atmos.temperatures[np.newaxis, :]
            / (de[:, np.newaxis] * units.RYD_TO_ERG)
        )

        compute_st_1_4 = lambda c, k: 1.0 - np.log(c) / np.log(k + c)
        compute_st_2_3_5_6 = lambda c, k: k / (k + c)
        compute_st = [
            compute_st_1_4,
            compute_st_2_3_5_6,
            compute_st_2_3_5_6,
            compute_st_1_4,
            compute_st_2_3_5_6,
            compute_st_2_3_5_6,
        ]

        compute_cs = [
            lambda c, k, s: s * np.log(k + np.e),
            lambda c, k, s: s,
            lambda c, k, s: s / (k + 1.0),
            lambda c, k, s: s * np.log(k + c),
            lambda c, k, s: s / k,
            lambda c, k, s: 10**s,
        ]

        st = np.zeros_like(kte)
        for i, ttype in enumerate(range(1, 7)):
            mask = ttypes == ttype
            st[mask, :] = compute_st[i](cups[mask][:, np.newaxis], kte[mask, :])

        for i in range(n_transitions):
            spline = scipy.interpolate.splrep(xs[i], scups[i], s=0)
            sups = scipy.interpolate.splev(st[i, :], spline)
            collision_strengths[i, :] = compute_cs[ttypes[i] - 1](
                cups[i], kte[i, :], sups
            )

        collision = units.HPLANCK**2 / (
            (2.0 * np.pi * units.M_ELECTRON) ** 1.5 * np.sqrt(units.KBOLTZMANN)
        )

        de = np.abs(elvlc[upper_level_indices] - elvlc[lower_level_indices])
        ekt = (de[:, np.newaxis] * units.RYD_TO_ERG) / (
            units.KBOLTZMANN * self.atmos.temperatures[np.newaxis, :]
        )

        sqrt_temperatures = np.sqrt(self.atmos.temperatures)[np.newaxis, :]
        deexcitation_rates = (
            collision
            * collision_strengths
            / (
                self.__elvlc_mult[upper_level_indices][:, np.newaxis]
                * sqrt_temperatures
            )
        )
        excitation_rates = (
            collision
            * collision_strengths
            * np.exp(-ekt)
            / (
                self.__elvlc_mult[lower_level_indices][:, np.newaxis]
                * sqrt_temperatures
            )
        )

        collision_strengths[collision_strengths < 0] = 0
        return collision_strengths, excitation_rates, deexcitation_rates

    def __levels_filtered_arr(self, full):
        return (
            self.__filtered_arr(full, self.__levels_mask)
            if self.__filter_levels
            else full
        )

    def __transitions_filtered_arr(self, full):
        return (
            self.__filtered_arr(full, self.__transitions_mask)
            if self.__filter_levels
            else full
        )

    def __scups_transitions_filtered_arr(self, full):
        return (
            self.__filtered_arr(full, self.__scups_transitions_mask)
            if self.__filter_levels
            else full
        )

    def __scups_transitions_filtered_list(self, full):
        return (
            self.__filtered_list(full, self.__scups_transitions_mask)
            if self.__filter_levels
            else full
        )

    def __psplups_transitions_filtered_arr(self, full):
        return (
            self.__filtered_arr(full, self.__psplups_transitions_mask)
            if self.__filter_levels
            else full
        )

    def __psplups_transitions_filtered_list(self, full):
        return (
            self.__filtered_list(full, self.__psplups_transitions_mask)
            if self.__filter_levels
            else full
        )

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
    def __init__(
        self,
        log_temperature_limits=(3.0, 7.0),
        log_electron_density_limits=(8.0, 13.0),
        n_temperature_points=100,
        n_electron_density_points=100,
        compute_proton_densities=True,
        compute_hydrogen_densities=False,
        dtype=np.float32,
        **kwargs,
    ):
        self.__log_table_temperatures = np.linspace(
            *log_temperature_limits, n_temperature_points, dtype=dtype
        )
        self.__log_table_electron_densities = np.linspace(
            *log_electron_density_limits, n_electron_density_points, dtype=dtype
        )

        table_temperatures = 10**self.log_table_temperatures
        table_electron_densities = 10**self.log_table_electron_densities
        temperature_mesh, electron_density_mesh = np.meshgrid(
            table_temperatures, table_electron_densities, indexing="ij"
        )

        super().__init__(
            np.ravel(temperature_mesh),
            np.ravel(electron_density_mesh),
            compute_proton_densities=False,
            compute_hydrogen_densities=False,
            **kwargs,
        )

        if compute_proton_densities:
            self.info(
                f"Computing {n_temperature_points*n_electron_density_points} proton densities"
            )
            start_time = time.time()

            proton_to_electron_ratios = (
                self.species_ratios.compute_proton_to_electron_ratios(
                    self.log_table_temperatures, temperature_is_log=True
                )
            )
            self.__proton_densities = np.ravel(
                np.outer(proton_to_electron_ratios, table_electron_densities)
            )

            self.info(f"Took {time.time() - start_time:g} s")

        if compute_hydrogen_densities:
            self.info(
                f"Computing {n_temperature_points*n_electron_density_points} hydrogen densities"
            )
            start_time = time.time()

            hydrogen_to_electron_ratios = (
                self.species_ratios.compute_hydrogen_to_electron_ratios(
                    self.log_table_temperatures, temperature_is_log=True
                )
            )
            self.__hydrogen_densities = np.ravel(
                np.outer(hydrogen_to_electron_ratios, table_electron_densities)
            )

            self.info(f"Took {time.time() - start_time:g} s")

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


def compute_emissivity_tables(
    ion_line_name_map,
    ion_line_wavelength_map,
    dtype=np.float32,
    verbose=False,
    **kwargs,
):
    table_atmosphere = LookupIonAtmosphere(
        compute_proton_densities=True,
        compute_hydrogen_densities=True,
        verbose=False,
        dtype=dtype,
        **kwargs,
    )

    line_names = []
    wavelengths = []
    emissivity_tables = []

    for ion_name, requested_central_wavelengths in ion_line_wavelength_map.items():
        if verbose:
            print(f"Computing emissivity tables for {ion_name}")

        ion = Ion(ion_name, table_atmosphere, verbose=False)

        line_indices = ion.find_line_indices(requested_central_wavelengths)

        emissivities = ion.compute_emissivities(line_indices=line_indices).astype(dtype)

        for idx, line_idx in enumerate(line_indices):
            line_name = ion_line_name_map[ion_name][idx]
            central_wavelength = ion.central_wavelengths[line_idx]

            line_names.append(line_name)
            wavelengths.append(central_wavelength)
            emissivity_tables.append(
                emissivities[idx, :].reshape(table_atmosphere.table_shape)
            )

    wavelengths = np.array(wavelengths, dtype=dtype)

    return (
        table_atmosphere.log_table_temperatures,
        table_atmosphere.log_table_electron_densities,
        line_names,
        wavelengths,
        emissivity_tables,
    )
