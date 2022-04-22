import os
import time
import pathlib
import re
import pickle

import numpy as np
from tabulate import tabulate
from joblib import Parallel, delayed

try:
    import backstaff.units as units
    import backstaff.array_utils as array_utils
    import backstaff.fields as fields
    import backstaff.chianti_synthesis as core
except ModuleNotFoundError:
    import units
    import array_utils as array_utils
    import fields
    import chianti_synthesis as core


class LookupIon(core.Ion):
    @property
    def table_populations(self):
        pop = self.populations
        return (
            None
            if pop is None
            else pop.reshape((self.n_levels, *self.atmos.table_shape))
        )

    def in_atmosphere(
        self, atmosphere, included_levels=None, use_memmap=True, **kwargs
    ):
        ion = core.Ion(self.ion_name, atmosphere, verbose=self.verbose)
        if included_levels is not None:
            ion.restrict_to_levels(included_levels)

        shape = (atmosphere.n_values, 2)
        dtype = atmosphere.temperatures.dtype
        evaluation_coordinates = (
            array_utils.create_tmp_memmap(shape, dtype)
            if use_memmap
            else np.empty(shape, dtype=dtype)
        )
        np.stack(
            (
                np.ravel(np.log10(atmosphere.temperatures)),
                np.ravel(np.log10(atmosphere.electron_densities)),
            ),
            axis=1,
            out=evaluation_coordinates,
        )

        populations = self.__lookup_populations(
            evaluation_coordinates, ion.levels_mask, use_memmap=use_memmap, **kwargs
        )
        return ion.with_populations(populations)

    def __lookup_populations(
        self,
        evaluation_coordinates,
        included_level_indices,
        use_memmap=True,
        method="linear",
        bounds_error=False,
        fill_value=0.0,
    ):
        assert evaluation_coordinates.ndim == 2 and evaluation_coordinates.shape[1] == 2
        if self.table_populations is None:
            self.compute_populations()

        self.info(
            f"Interpolating populations for {evaluation_coordinates.shape[0]} conditions"
        )
        start_time = time.time()

        included_table_populations = self.table_populations[
            included_level_indices, :, :
        ]

        shape = (included_table_populations.shape[0], evaluation_coordinates.shape[0])
        dtype = included_table_populations.dtype
        populations = (
            array_utils.create_tmp_memmap(shape, dtype)
            if use_memmap
            else np.empty(shape, dtype=dtype)
        )

        array_utils.do_concurrent_interp2(
            populations,
            0,
            shape[0],
            self.atmos.log_table_temperatures,
            self.atmos.log_table_electron_densities,
            included_table_populations,
            evaluation_coordinates,
            method=method,
            bounds_error=bounds_error,
            fill_value=fill_value,
            verbose=self.verbose,
        )
        populations[populations < 0.0] = 0.0
        populations[populations > 1.0] = 1.0

        self.info(f"Took {time.time() - start_time:g} s")
        return populations


class IonEmissivities:
    @staticmethod
    def from_lookup(
        lookup_ion,
        bifrost_data,
        height_range=(-np.inf, np.inf),
        min_ionization_fraction=None,
        z_upsampling_factor=1,
        included_line_wavelengths=None,
        use_memmap=False,
        drop_populations=True,
        n_jobs=1,
    ):
        tg = fields.ScalarField3.from_bifrost_data(
            bifrost_data, "tg", height_range=height_range
        )
        nel = fields.ScalarField3.from_bifrost_data(
            bifrost_data, "nel", height_range=height_range
        )

        if min_ionization_fraction is None:
            mask = None
            coords = tg.get_coords()
            tg = tg.get_values_flat()
            nel = nel.get_values_flat()
        else:
            ionization_fractions = lookup_ion.sample_ionization_fractions(
                tg.get_values()
            )
            mask = array_utils.CompactArrayMask(
                ionization_fractions > min_ionization_fraction
            )
            lookup_ion.info(
                f"Proportion of positions with significant ionization fractions is {1e2*mask.compact_size/mask.size:.1f}%"
            )

            if z_upsampling_factor > 1:
                index_range = mask.axis_ranges[2]
                lookup_ion.info(
                    f"Upsampling to {z_upsampling_factor*1e2:g}% along z-axis for k in [{index_range[0]}, {index_range[1]})"
                )
                start_time = time.time()

                tg = tg.resampled_along_axis(
                    2, z_upsampling_factor, index_range=index_range, kind="linear"
                )
                nel = nel.resampled_along_axis(
                    2, z_upsampling_factor, index_range=index_range, kind="linear"
                )

                ionization_fractions = lookup_ion.sample_ionization_fractions(
                    tg.get_values()
                )
                mask = array_utils.CompactArrayMask(
                    ionization_fractions > min_ionization_fraction
                )

                lookup_ion.info(f"Took {time.time() - start_time:g} s")

            coords = tg.get_coords()
            tg = mask.apply(tg.get_values())
            nel = mask.apply(nel.get_values())

            lookup_ion.info(
                f"Compact atmosphere representation size is {1e2*tg.size/(bifrost_data.x.size*bifrost_data.y.size*bifrost_data.z.size):.1f}%"
            )

        atmos = core.IonAtmosphere3D(
            *coords,
            tg,
            nel,
            species_ratios=lookup_ion.atmos.species_ratios,
            compute_proton_densities=False,
            verbose=lookup_ion.atmos.verbose,
        )

        included_levels = lookup_ion.find_levels_for_lines(
            lookup_ion.find_line_indices(included_line_wavelengths)
        )

        ion = lookup_ion.in_atmosphere(atmos, included_levels, use_memmap=use_memmap)

        ion.compute_emissivities(
            line_indices=ion.find_line_indices(included_line_wavelengths)
        )
        if drop_populations:
            ion.drop_populations()  # May be too big to keep

        central_wavelengths = ion.central_wavelengths[ion.emissivity_line_indices]
        emissivities = (
            ion.atmos.unravel(ion.emissivities) if mask is None else ion.emissivities
        )

        return IonEmissivities(
            ion.properties,
            bifrost_data,
            height_range,
            ion.atmos.x_coords,
            ion.atmos.y_coords,
            ion.atmos.z_coords,
            central_wavelengths,
            mask,
            emissivities,
            verbose=lookup_ion.verbose,
            n_jobs=n_jobs,
        )

    @staticmethod
    def get_optimization_tag(min_ionization_fraction, z_upsampling_factor):
        if min_ionization_fraction is None:
            return None
        else:
            return f"mi={min_ionization_fraction}_zu={z_upsampling_factor}"

    def __init__(
        self,
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
        verbose=False,
    ):
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
            verbose=self.__bd_verbose,
        )

    def save(self, file_path):
        bd = self.__bifrost_data
        self.drop_bifrost_data()
        with open(file_path, "wb") as f:
            pickle.dump(self, f)
        self.__bifrost_data = bd

    @staticmethod
    def load(file_path):
        with open(file_path, "rb") as f:
            ion_emissivities = pickle.load(f)
        ion_emissivities.reopen_dropped_bifrost_data()
        return ion_emissivities

    def save_data(self, file_path):
        with open(file_path, "wb") as f:
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
                emissivities=self.emissivities,
            )

    @staticmethod
    def from_data_file(file_path, bifrost_data, **kwargs):
        data = np.load(file_path)
        return IonEmissivities(
            core.IonProperties(data["ion_name"].item(), data["abundance"].item()),
            bifrost_data,
            tuple(data["height_range"]),
            data["x_coords"],
            data["y_coords"],
            data["z_coords"],
            data["central_wavelengths"],
            None
            if data["mask"].shape == ()
            else array_utils.CompactArrayMask(data["mask"]),
            data["emissivities"],
            **kwargs,
        )

    def compute_intensity_contributions(self):
        atmos_shape = (self.bifrost_data.nx, self.bifrost_data.ny, self.bifrost_data.nz)

        if atmos_shape[2] != self.z_coords.size:
            dz = np.zeros_like(self.z_coords)
            dz[:-1] = self.z_coords[1:] - self.z_coords[:-1]
            dz[-1] = self.z_coords[-1] - self.z_coords[-2]
            dz *= units.U_L

            atmos_shape = atmos_shape[:2] + (self.z_coords.size,)
        else:
            dz = fields.ScalarField1.dz_in_bifrost_data(
                self.bifrost_data, height_range=self.height_range, scale=units.U_L
            ).get_values()  # [cm]

        if self.mask is None:
            intensity_contributions = (
                self.emissivities * dz[np.newaxis, np.newaxis, np.newaxis, :]
            )
        else:
            intensity_contributions = (
                self.emissivities
                * self.mask.apply(np.broadcast_to(dz, atmos_shape))[np.newaxis, :]
            )

        return intensity_contributions

    def synthesize_spectral_lines(
        self, spectrum, extra_quantity_names=None, min_intensity=1.0
    ):
        self.info(f"Synthesizing {self.n_lines} spectral lines")
        start_time = time.time()

        temperatures = fields.ScalarField3.from_bifrost_data(
            self.bifrost_data, "tg", height_range=self.height_range
        )
        vertical_speeds = fields.ScalarField3.from_bifrost_data(
            self.bifrost_data, "uz", height_range=self.height_range
        )  # Positive velocity is downward

        extra_quantity_fields = (
            {}
            if extra_quantity_names is None
            else {
                quantity_name: fields.ScalarField3.from_bifrost_data(
                    self.bifrost_data, quantity_name, height_range=self.height_range
                )
                for quantity_name in extra_quantity_names
            }
        )

        atmos_shape = temperatures.get_shape()

        if atmos_shape[2] != self.z_coords.size:
            temperatures = temperatures.resampled_to_coords_along_axis(2, self.z_coords)
            vertical_speeds = vertical_speeds.resampled_to_coords_along_axis(
                2, self.z_coords
            )
            for quantity_name, quantity_field in extra_quantity_fields.items():
                extra_quantity_fields[
                    quantity_name
                ] = quantity_field.resampled_to_coords_along_axis(2, self.z_coords)

        temperatures = temperatures.get_values()
        vertical_speeds = vertical_speeds.get_values()
        for quantity_name, quantity in extra_quantity_fields.items():
            extra_quantity_fields[quantity_name] = quantity.get_values()

        intensity_contributions = self.compute_intensity_contributions()

        if self.mask is None:

            def integrate_emissivities(line_idx, weights=None):
                if weights is None:
                    return np.sum(intensity_contributions[line_idx, :, :, :], axis=2)
                else:
                    return np.sum(
                        weights * intensity_contributions[line_idx, :, :, :], axis=2
                    )

        else:
            temperatures = self.mask.apply(temperatures)
            vertical_speeds = self.mask.apply(vertical_speeds)
            for quantity_name, quantity in extra_quantity_fields.items():
                extra_quantity_fields[quantity_name] = self.mask.apply(quantity)

            def integrate_emissivities(line_idx, weights=None):
                if weights is None:
                    return self.mask.sum_over_axis(
                        intensity_contributions[line_idx, :], axis=2
                    )
                else:
                    return self.mask.sum_over_axis(
                        weights * intensity_contributions[line_idx, :], axis=2
                    )

        def synthesize_line(line_idx):
            central_wavelength = self.central_wavelengths[line_idx]
            local_doppler_shifts = vertical_speeds * (
                units.U_U * central_wavelength / units.CLIGHT
            )  # [cm] (negative sign omitted so that positive is upwards)
            thermal_variances = self.ion_properties.compute_thermal_line_variance(
                central_wavelength, temperatures
            )  # [cm^2]

            intensities = integrate_emissivities(line_idx)  # [erg/s/sr/cm^2]
            invalid = intensities < min_intensity
            doppler_shifts = (
                integrate_emissivities(line_idx, weights=local_doppler_shifts)
                / intensities
            )  # [cm]
            doppler_shifts[invalid] = np.nan
            variances = (
                integrate_emissivities(
                    line_idx, weights=(thermal_variances + local_doppler_shifts**2)
                )
                / intensities
                - doppler_shifts**2
            )  # [cm^2]
            variances[invalid] = np.nan

            doppler_shifts *= 1e8  # [Å]
            widths = (2 * np.sqrt(2 * np.log(2)) * 1e8) * np.sqrt(
                variances
            )  # [Å] (FWHM)

            quantities = dict(
                intensity=intensities, doppler_shift=doppler_shifts, width=widths
            )

            for quantity_name, quantity_field in extra_quantity_fields.items():
                quantity = (
                    integrate_emissivities(line_idx, weights=quantity_field)
                    / intensities
                )
                quantity[invalid] = np.nan
                quantities[quantity_name] = quantity

            return SpectralLine(
                self.ion_properties, self.central_wavelengths_Å[line_idx], quantities
            )

        for spectral_line in Parallel(
            n_jobs=min(self.n_jobs, self.n_lines), verbose=self.verbose
        )((delayed(synthesize_line)(line_idx) for line_idx in range(self.n_lines))):
            spectrum[
                spectral_line.name, self.bifrost_data.root_name, self.bifrost_data.snap
            ] = spectral_line

        self.info(f"Took {time.time() - start_time:g} s")

    def load_spectral_lines(self, spectrum, dir_path, tag=None):
        dir_path = pathlib.Path(dir_path)
        tag = join_tags(tag, as_str=True)
        ion_name = self.ion_properties.ion_name
        atmosphere_name = self.bifrost_data.root_name
        snap_num = self.bifrost_data.snap
        for central_wavelength in self.central_wavelengths_Å:
            spectral_line = SpectralLine.load(
                dir_path
                / f"{ion_name}_{central_wavelength:.3f}_{atmosphere_name}_{snap_num:03d}{tag}.npz"
            )
            spectrum[spectral_line.name, atmosphere_name, snap_num] = spectral_line

    def plot(self, line_idx, height, **kwargs):
        assert self.mask is None, "Cannot plot emissivities in compact representation"
        k = min(self.z_coords.size - 1, max(0, np.searchsorted(self.z_coords, height)))
        return fields.ScalarField2(
            fields.Coords2(self.x_coords, self.y_coords),
            self.emissivities[line_idx, :, :, k],
        ).plot(**kwargs)

    def obtain_mean_intensity_contribution_field(self, line_idx):
        assert self.mask is None, "Cannot plot emissivities in compact representation"
        dz = fields.ScalarField1.dz_in_bifrost_data(
            self.bifrost_data, height_range=self.height_range, scale=units.U_L
        )
        return dz * np.mean(self.emissivities[line_idx, :, :, :], axis=(0, 1))


class SpectralLine:
    def __init__(
        self,
        ion_properties,
        central_wavelength,
        quantities,
        selected_quantity_name="intensity",
    ):
        self.__ion_properties = ion_properties
        self.__central_wavelength = central_wavelength  # [Å]
        self.__quantities = quantities

        self.__derived_quantities = dict(
            doppler_velocity=self.__compute_doppler_velocity,
            width_velocity=self.__compute_width_velocity,
        )

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
        return core.IonProperties.get_line_name(self.ion_name, self.central_wavelength)

    @property
    def description(self):
        return core.IonProperties.get_line_description(
            self.ion_properties.spectroscopic_name, self.central_wavelength
        )

    def get_tag(self, atmosphere_name, snap_num, tag=None):
        tag = join_tags(tag, as_str=True)
        return f"{self.ion_name}_{self.central_wavelength:.3f}_{atmosphere_name}_{snap_num:03d}{tag}"

    @staticmethod
    def get_tag_regex(
        ion_name=None,
        central_wavelength=None,
        atmosphere_name=None,
        snap_num=None,
        tag=None,
    ):
        ion_name_r = r"([a-z]+_\d+)" if ion_name is None else re.escape(ion_name)
        central_wavelength_r = (
            r"(\d+\.\d+)"
            if central_wavelength is None
            else re.escape(f"{central_wavelength:.3f}")
        )
        atmosphere_name_r = (
            r"(.+)" if atmosphere_name is None else re.escape(atmosphere_name)
        )
        snap_num_r = r"(\d+)" if snap_num is None else re.escape(snap_num)
        tag_r = re.escape(join_tags(tag, as_str=True))
        return (
            ion_name_r
            + "_"
            + central_wavelength_r
            + "_"
            + atmosphere_name_r
            + "_"
            + snap_num_r
            + tag_r
        )

    def save(self, file_path):
        with open(file_path, "wb") as f:
            np.savez_compressed(
                f,
                ion_name=self.ion_name,
                abundance=self.ion_properties.abundance,
                central_wavelength=self.central_wavelength,
                **self.__quantities,
            )

    @staticmethod
    def load(file_path):
        data = np.load(file_path)
        non_quantity_names = ["ion_name", "central_wavelength"]
        extra_quantity_names = [
            name for name in data.files if name not in non_quantity_names
        ]
        ion_name = data["ion_name"].item()
        abundance = data["abundance"].item()
        central_wavelength = data["central_wavelength"].item()
        spectral_line = SpectralLine(
            core.IonProperties(ion_name, abundance),
            central_wavelength,
            {name: data[name] for name in extra_quantity_names},
        )
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
            raise ValueError(f"Invalid spectral line quantity {name}")
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
            if (
                self.selected_quantity_name is None
                and other.selected_quantity_name is None
            ):
                if inplace:
                    raise ValueError("Operation requires a quantity to be selected")
                return SpectralLine(
                    self.ion_properties,
                    self.central_wavelength,
                    self.__merge_quantities(other.quantities, operation),
                )
            else:
                if (
                    self.selected_quantity_name is not None
                    and other.selected_quantity_name is not None
                ):
                    assert other.selected_quantity_name == self.selected_quantity_name
                    assert (
                        other.selected_quantity_params == self.selected_quantity_params
                    )
                elif self.selected_quantity_name is None:
                    self.select(
                        other.selected_quantity_name, **other.selected_quantity_params
                    )
                else:
                    other.select(
                        self.selected_quantity_name, **self.selected_quantity_params
                    )

                this_quantity = self.get(
                    self.selected_quantity_name, **self.selected_quantity_params
                )
                other_quantity = other.get(
                    self.selected_quantity_name, **self.selected_quantity_params
                )
                result = operation(this_quantity, other_quantity)
                return self if inplace else result
        elif other is not None:
            assert self.selected_quantity_name is not None
            this_quantity = self.get(
                self.selected_quantity_name, **self.selected_quantity_params
            )
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
        return self.__perform_operation(term, lambda a, b: a.__iadd__(b), inplace=True)

    def __isub__(self, term):
        return self.__perform_operation(term, lambda a, b: a.__isub__(b), inplace=True)

    def __imul__(self, term):
        return self.__perform_operation(term, lambda a, b: a.__imul__(b), inplace=True)

    def __itruediv__(self, term):
        return self.__perform_operation(
            term, lambda a, b: a.__itruediv__(b), inplace=True
        )

    def compute_statistics(self):
        return dict(
            spectral_line=self.description,
            mean_intensity=np.nanmean(self.get("intensity")),
            mean_doppler_velocity=np.nanmean(self.get("doppler_velocity", scale=1e-5)),
            mean_width_velocity=np.nanmean(self.get("width_velocity", scale=1e-5)),
        )

    def print_statistics(self):
        quantities = dict(
            spectral_line="Spectral line",
            mean_intensity="Mean intensity [erg/s/sr/cm^2]",
            mean_doppler_velocity="Mean Doppler velocity [km/s]",
            mean_width_velocity="Mean line width [km/s]",
        )
        statistics = self.compute_statistics()
        headers = []
        entries = []
        for quantity, description in quantities.items():
            headers.append(description)
            entries.append(statistics[quantity])
        print(tabulate([entries], headers=headers, tablefmt="orgtbl"))

    def plot(
        self, x_coords, y_coords, quantity_name, mapper=None, scale=None, **kwargs
    ):
        return fields.ScalarField2(
            fields.Coords2(x_coords, y_coords),
            self.get(quantity_name, mapper=mapper, scale=scale),
        ).plot(**kwargs)

    def __compute_doppler_velocity(self):
        return (
            -units.CLIGHT * self.get("doppler_shift") / self.central_wavelength
        )  # Positive upwards [cm/s]

    def __compute_width_velocity(self):
        return units.CLIGHT * self.get("width") / self.central_wavelength


class SpectralLineArray(np.ndarray):
    def __new__(cls, a):
        obj = np.asarray(a).view(cls)
        return obj

    def select(self, quantity_name, **kwargs):
        np.vectorize(
            lambda spectral_line: None
            if spectral_line is None
            else spectral_line.select(quantity_name, **kwargs),
            otypes=[object],
        )(self)
        return self

    def get(self, quantity_name, **kwargs):
        return np.vectorize(
            lambda spectral_line: None
            if spectral_line is None
            else spectral_line.get(quantity_name, **kwargs),
            otypes=[object],
        )(self)

    def applied(self, func):
        return np.vectorize(
            lambda item: None if item is None else func(item), otypes=[object]
        )(self)

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
    def __init__(self, line_names, atmosphere_names, snap_nums, spectral_lines=None):
        self.__line_names = np.atleast_1d(line_names)
        self.__atmosphere_names = np.atleast_1d(atmosphere_names)
        self.__snap_nums = np.atleast_1d(snap_nums)

        (
            self.__mesh_line_names,
            self.__mesh_atmosphere_names,
            self.__mesh_snap_nums,
        ) = np.meshgrid(
            self.__line_names, self.__atmosphere_names, self.__snap_nums, indexing="ij"
        )
        shape = self.__mesh_line_names.shape

        self.__line_name_indices = {
            line_name: idx for idx, line_name in enumerate(self.line_names)
        }
        self.__atmosphere_name_indices = {
            atmosphere_name: idx
            for idx, atmosphere_name in enumerate(self.atmosphere_names)
        }
        self.__snap_num_indices = {
            snap_num: idx for idx, snap_num in enumerate(self.snap_nums)
        }

        if spectral_lines is None:
            self.__spectral_lines = SpectralLineArray(np.full(shape, None))
        else:
            assert (
                isinstance(spectral_lines, SpectralLineArray)
                and spectral_lines.shape == shape
            )
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
        with open(file_path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(file_path):
        with open(file_path, "rb") as f:
            spectrum = pickle.load(f)
        return spectrum

    def save_spectral_lines(self, dir_path, identifier=None, tag=None):
        dir_path = pathlib.Path(dir_path)

        if identifier is None:
            for idx, spectral_line in np.ravel(self.spectral_lines):
                if spectral_line is None:
                    continue
                _, atmosphere_name, snap_num = self.identifier(idx)
                spectral_line.save(
                    dir_path
                    / f"{spectral_line.get_tag(atmosphere_name, snap_num, tag=tag)}.npz"
                )
        else:
            idx = self.idx(*identifier)
            for atmosphere_name, snap_num, spectral_line in zip(
                np.atleast_1d(self.__mesh_atmosphere_names[idx]),
                np.atleast_1d(self.__mesh_snap_nums[idx]),
                np.atleast_1d(self.spectral_lines[idx]),
            ):
                if spectral_line is None:
                    continue
                spectral_line.save(
                    dir_path
                    / f"{spectral_line.get_tag(atmosphere_name, snap_num, tag=tag)}.npz"
                )

    @staticmethod
    def from_files(dir_path, line_names, atmosphere_names, snap_nums, tag=None):
        dir_path = pathlib.Path(dir_path)

        pattern = re.compile(SpectralLine.get_tag_regex(tag=tag) + r"\.npz")

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
                line_name = core.IonProperties.get_line_name(*parsed[:2])
                atmosphere_name = parsed[2]
                snap_num = parsed[3]
                if (
                    line_name in spectrum.line_names
                    and atmosphere_name in spectrum.atmosphere_names
                    and snap_num in spectrum.snap_nums
                ):
                    spectral_line = SpectralLine.load(dir_path / filename)
                    spectrum[
                        spectral_line.name, atmosphere_name, snap_num
                    ] = spectral_line

        return spectrum

    def idx(self, line_name, atmosphere_name, snap_num):
        if isinstance(line_name, list):
            line_name_idx = [self.__line_name_indices[name] for name in line_name]
        elif isinstance(line_name, str):
            line_name_idx = self.__line_name_indices[line_name]
        else:
            line_name_idx = line_name

        if isinstance(atmosphere_name, list):
            atmosphere_name_idx = [
                self.__atmosphere_name_indices[name] for name in atmosphere_name
            ]
        elif isinstance(atmosphere_name, str):
            atmosphere_name_idx = self.__atmosphere_name_indices[atmosphere_name]
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
        line_idx, atmos_idx, snap_idx = np.unravel_index(idx, self.spectral_lines.shape)
        return (
            self.line_names[line_idx],
            self.atmosphere_names[atmos_idx],
            self.snap_nums[snap_idx],
        )

    def get_line_names_for_ion(self, ion_name):
        return list(filter(lambda name: name.startswith(ion_name), self.line_names))

    def __getitem__(self, identifier):
        return self.spectral_lines[self.idx(*identifier)]

    def __setitem__(self, identifier, spectral_line):
        self.spectral_lines[self.idx(*identifier)] = spectral_line

    def print_statistics(self, atmosphere_name, snap_num):
        quantities = dict(
            spectral_line="Spectral line",
            mean_intensity="Mean intensity [erg/s/sr/cm^2]",
            mean_doppler_velocity="Mean Doppler velocity [km/s]",
            mean_width_velocity="Mean line width [km/s]",
        )
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
        print(tabulate(entries, headers=headers, tablefmt="orgtbl"))


def join_tags(*tags, as_str=False):
    combined = ""
    for tag in tags:
        if tag is not None:
            combined += f"_{tag}"
    if not as_str:
        if combined == "":
            return None
        else:
            return combined[1:]
    else:
        return combined
