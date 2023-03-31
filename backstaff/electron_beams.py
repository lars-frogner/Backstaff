import collections
import functools
import numpy as np
import scipy.signal as signal
from pathlib import Path

try:
    import backstaff.units as units
    import backstaff.plotting as plotting
    import backstaff.field_lines as field_lines
    import backstaff.beam_heating as beam_heating
except ModuleNotFoundError:
    import units
    import plotting
    import field_lines
    import beam_heating


class ElectronBeamSwarm(field_lines.FieldLineSet3):

    VALUE_DESCRIPTIONS = {
        "x": r"$x$ [Mm]",
        "y": r"$y$ [Mm]",
        "z": "Height [Mm]",
        "z0": "Initial height [Mm]",
        "s": r"$s$ [Mm]",
        "sz0": r"$s(z=0) - s$ [Mm]",
        "initial_pitch_angle_cosine": r"$\mu_0$",
        "collisional_pitch_angle_cosine": r"$\mu$",
        "adiabatic_pitch_angle_cosine": r"$\mu$",
        "initial_pitch_angle": r"$\beta_0$ [deg]",
        "electric_field_angle_cosine": "Electric field angle cosine",
        "total_power": "Total power [erg/s]",
        "total_power_density": r"Total power density [erg/s/cm$^3$]",
        "total_energy_density": r"Total energy density [erg/cm$^3$]",
        "lower_cutoff_energy": r"$E_\mathrm{{c}}$ [keV]",
        "acceleration_volume": r"Acceleration site volume [cm$^3$]",
        "estimated_depletion_distance": r"$\tilde{{s}}_\mathrm{{dep}}$ [Mm]",
        "total_propagation_distance": r"$s_\mathrm{{dep}}$ [Mm]",
        "residual_factor": r"$r$",
        "acceleration_height": "Acceleration site height [Mm]",
        "depletion_height": "Depletion height [Mm]",
        "beam_electron_fraction": "Beam electrons relative to total electrons",
        "return_current_speed_fraction": "Speed relative to speed of light",
        "estimated_electron_density": r"Electron density [electrons/cm$^3$]",
        "deposited_power": "Deposited power [erg/s]",
        "deposited_power_per_dist": r"$\mathrm{{d}}\mathcal{{E}}/\mathrm{{d}}s$ [erg/s/cm]",
        "deposited_power_density": r"$Q$ [erg/s/cm$^3$]",
        "power_change": "Power change [erg/s]",
        "power_density_change": r"Power density change [erg/s/cm$^3$]",
        "beam_flux": r"Energy flux [erg/s/cm$^2$]",
        "conduction_flux": r"Energy flux [erg/s/cm$^2$]",
        "remaining_power": "Remaining power [erg/s]",
        "relative_cumulative_power": r"$\mathcal{{E}}/P_\mathrm{{beam}}$ [$\%$]",
        "r": r"$\rho$ [g/cm$^3$]",
        "tg": r"$T$ [K]",
        "nel": r"$n_\mathrm{{e}}$ [electrons/cm$^3$]",
        "krec": r"$K$ [Bifrost units]",
        "qspitz": r"Power density change [erg/s/cm$^3$]",
        "r0": r"$\rho$ [g/cm$^3$]",
        "tg0": r"$T$ [K]",
        "p": r"$P$ [dyn/cm$^2$]",
        "b": r"$|B|$ [G]",
        "beta": r"$\beta$",
        "ux": r"$u_x$ [cm/s]",
        "uy": r"$u_y$ [cm/s]",
        "uz": r"$u_z$ [cm/s]",
        "us": r"$u_s$ [cm/s]",
        "uhor": r"$u_\mathrm{h}$ [cm/s]",
        "magnetic_flux_density": r"$|B|$ [G]",
    }

    VALUE_UNIT_CONVERTERS = {
        "r": lambda f: f * units.U_R,
        "qspitz": lambda f: f * (units.U_E / units.U_T),
        "qjoule": lambda f: f * (units.U_E / units.U_T),
        "dedt": lambda f: f * (units.U_E / units.U_T),
        "r0": lambda f: f * units.U_R,
        "z": lambda f: -f,
        "z0": lambda f: -f,
        "bx": lambda f: f * units.U_B,
        "by": lambda f: f * units.U_B,
        "bz": lambda f: f * units.U_B,
        "b": lambda f: f * units.U_B,
        "p": lambda f: f * units.U_E,
        "ux": lambda f: f * units.U_U,
        "uy": lambda f: f * units.U_U,
        "uz": lambda f: f * units.U_U,
        "us": lambda f: f * units.U_U,
        "uhor": lambda f: f * units.U_U,
    }

    @staticmethod
    def from_file(
        file_path,
        acceleration_data_type=None,
        params={},
        derived_quantities=[],
        verbose=False,
    ):
        import backstaff.reading as reading

        file_path = Path(file_path)
        extension = file_path.suffix
        if extension == ".pickle":
            electron_beam_swarm = (
                reading.read_electron_beam_swarm_from_combined_pickles(
                    file_path,
                    acceleration_data_type=acceleration_data_type,
                    params=params,
                    derived_quantities=derived_quantities,
                    verbose=verbose,
                )
            )
        elif extension == ".fl":
            electron_beam_swarm = (
                reading.read_electron_beam_swarm_from_custom_binary_file(
                    file_path,
                    acceleration_data_type=acceleration_data_type,
                    params=params,
                    derived_quantities=derived_quantities,
                    verbose=verbose,
                )
            )
        else:
            raise ValueError(
                "Invalid file extension {} for electron beam data.".format(extension)
            )

        return electron_beam_swarm

    @staticmethod
    def dummy(domain_bounds):
        return ElectronBeamSwarm(domain_bounds, 0, {}, {}, {}, {}, {})

    def __init__(
        self,
        domain_bounds,
        number_of_beams,
        fixed_scalar_values,
        fixed_vector_values,
        varying_scalar_values,
        varying_vector_values,
        acceleration_data,
        params={},
        derived_quantities=[],
        verbose=False,
    ):
        assert isinstance(acceleration_data, dict)
        self.number_of_beams = number_of_beams
        self.acceleration_data = acceleration_data
        super().__init__(
            domain_bounds,
            number_of_beams,
            fixed_scalar_values,
            fixed_vector_values,
            varying_scalar_values,
            varying_vector_values,
            params=params,
            derived_quantities=derived_quantities,
            verbose=verbose,
        )

        if self.verbose:
            print(
                "Acceleration data:\n    {}".format(
                    "\n    ".join(self.acceleration_data.keys())
                )
            )

    def get_subset(
        self, only_quantities=None, included_field_lines_finder=None, **kwargs
    ):
        return super().get_subset(
            self.acceleration_data,
            only_quantities=only_quantities,
            included_field_lines_finder=included_field_lines_finder,
            **kwargs
        )

    def get_number_of_beams(self):
        return self.number_of_beams

    def compute_number_of_sites(self):
        return np.unique(
            np.stack(
                (
                    self.fixed_scalar_values["x0"],
                    self.fixed_scalar_values["y0"],
                    self.fixed_scalar_values["z0"],
                ),
                axis=1,
            ),
            axis=0,
        ).shape[0]

    def get_acceleration_data(self, acceleration_data_type):
        return self.acceleration_data[acceleration_data_type]

    def get_acceleration_sites(self):
        return self.get_acceleration_data("acceleration_sites")

    def _derive_quantities(self, derived_quantities):

        super()._derive_quantities(derived_quantities)

        if "initial_pitch_angle" in derived_quantities:
            self.fixed_scalar_values["initial_pitch_angle"] = (
                np.arccos(self.get_fixed_scalar_values("initial_pitch_angle_cosine"))
                * 180.0
                / np.pi
            )

        if "total_power_density" in derived_quantities:
            self.fixed_scalar_values[
                "total_power_density"
            ] = self.get_fixed_scalar_values(
                "total_power"
            ) / self.get_fixed_scalar_values(
                "acceleration_volume"
            )

        if "total_energy_density" in derived_quantities:
            self._obtain_total_energy_densities()

        if "beam_electron_density" in derived_quantities:
            self._obtain_beam_electron_densities()

        if "non_thermal_energy_per_thermal_electron" in derived_quantities:
            self.fixed_scalar_values[
                "non_thermal_energy_per_thermal_electron"
            ] = self._obtain_total_energy_densities() / self.get_fixed_scalar_values(
                "nel0" if self.has_fixed_scalar_values("nel0") else "nel"
            )

        if "mean_electron_energy" in derived_quantities:
            self._obtain_mean_electron_energies()

        if "mean_electron_speed" in derived_quantities:
            self._obtain_mean_electron_speeds()

        if "acceleration_height" in derived_quantities:
            self.fixed_scalar_values["acceleration_height"] = np.asfarray(
                [-z[0] for z in self.get_varying_scalar_values("z")]
            )

        if "depletion_height" in derived_quantities:
            self.fixed_scalar_values["depletion_height"] = np.asfarray(
                [-z[-1] for z in self.get_varying_scalar_values("z")]
            )

        if "acceleration_site_electron_density" in derived_quantities:
            self._obtain_acceleration_site_electron_densities()

        if "beam_electron_fraction" in derived_quantities:
            self._obtain_beam_electron_fractions()

        if "return_current_speed_fraction" in derived_quantities:
            mean_electron_energies = (
                self._obtain_mean_electron_energies() * units.KEV_TO_ERG
            )
            mean_electron_speed_fractions = np.sqrt(
                1.0 - 1.0 / (1.0 + mean_electron_energies / units.MC2_ELECTRON) ** 2
            )
            beam_electron_fractions = self._obtain_beam_electron_fractions()
            self.fixed_scalar_values["return_current_speed_fraction"] = (
                beam_electron_fractions * mean_electron_speed_fractions
            )

        if "acceleration_current" in derived_quantities:
            self._obtain_acceleration_currents()

        if "acceleration_induced_magnetic_field" in derived_quantities:
            self._obtain_acceleration_induced_magnetic_fields()

        if "acceleration_ambient_magnetic_field" in derived_quantities:
            self._obtain_acceleration_ambient_magnetic_field()

        if "relative_acceleration_induced_magnetic_field" in derived_quantities:
            B_induced = self._obtain_acceleration_induced_magnetic_fields()
            B = self._obtain_acceleration_ambient_magnetic_field()
            self.fixed_scalar_values["relative_acceleration_induced_magnetic_field"] = (
                B_induced / B
            )

        if "acceleration_induced_electric_field" in derived_quantities:
            self._obtain_acceleration_induced_electric_fields()

        if "parallel_electric_field" in derived_quantities:
            self._obtain_parallel_electric_fields()

        if "relative_acceleration_induced_electric_field" in derived_quantities:
            E_induced = self._obtain_acceleration_induced_electric_fields()
            E = np.abs(self._obtain_parallel_electric_fields())
            self.fixed_scalar_values["relative_acceleration_induced_electric_field"] = (
                E_induced / E
            )

        if "return_current_heating_ratio" in derived_quantities:
            self._obtain_return_current_heating_ratio()

        if "estimated_electron_density" in derived_quantities:
            assert self.has_varying_scalar_values("r")
            self.varying_scalar_values["estimated_electron_density"] = [
                self.varying_scalar_values["r"][i]
                * units.U_R
                * units.MASS_DENSITY_TO_ELECTRON_DENSITY
                for i in range(self.get_number_of_beams())
            ]

        if "deposited_power_per_dist" in derived_quantities:
            scale = 1.0 / (self.get_param("dense_step_length") * units.U_L)
            self.varying_scalar_values["deposited_power_per_dist"] = [
                arr * scale for arr in self.varying_scalar_values["deposited_power"]
            ]

        if "power_change" in derived_quantities:
            self.varying_scalar_values["power_change"] = [
                arr.copy() for arr in self.varying_scalar_values["deposited_power"]
            ]
            for i in range(self.get_number_of_beams()):
                self.varying_scalar_values["power_change"][i][
                    0
                ] -= self.fixed_scalar_values["total_power"][i]

        if "power_density_change" in derived_quantities:
            self.varying_scalar_values["power_density_change"] = [
                arr.copy()
                for arr in self.varying_scalar_values["deposited_power_density"]
            ]
            for i in range(self.get_number_of_beams()):
                self.varying_scalar_values["power_density_change"][i][0] -= (
                    self.get_fixed_scalar_values("total_power")[i]
                    / self.get_fixed_scalar_values("acceleration_volume")[i]
                )

        if "padded_total_power_density" in derived_quantities:
            self.varying_scalar_values["padded_total_power_density"] = [
                np.zeros_like(arr) for arr in self.varying_scalar_values["x"]
            ]
            for i in range(self.get_number_of_beams()):
                self.varying_scalar_values["padded_total_power_density"][i][0] += (
                    self.get_fixed_scalar_values("total_power")[i]
                    / self.get_fixed_scalar_values("acceleration_volume")[i]
                )

        if "beam_flux" in derived_quantities:
            self.varying_scalar_values["beam_flux"] = [
                arr.copy()
                for arr in self.varying_scalar_values["deposited_power_density"]
            ]
            for i in range(self.get_number_of_beams()):
                self.varying_scalar_values["beam_flux"][i][0] -= (
                    self.get_fixed_scalar_values("total_power")[i]
                    / self.get_fixed_scalar_values("acceleration_volume")[i]
                )
                self.varying_scalar_values["beam_flux"][i] *= (
                    self.get_param("dense_step_length") * units.U_L
                )

        if "conduction_flux" in derived_quantities:
            self.varying_scalar_values["conduction_flux"] = [
                arr.copy() for arr in self.varying_scalar_values["qspitz"]
            ]
            for i in range(self.get_number_of_beams()):
                self.varying_scalar_values["conduction_flux"][i] *= (
                    self.get_param("dense_step_length")
                    * units.U_L
                    * units.U_E
                    / units.U_T
                )

        if "cumulative_power" in derived_quantities:
            self.varying_scalar_values["cumulative_power"] = [
                np.cumsum(self.varying_scalar_values["deposited_power"][i])
                for i in range(self.get_number_of_beams())
            ]

        if "relative_cumulative_power" in derived_quantities:
            self.varying_scalar_values["relative_cumulative_power"] = [
                100
                * np.cumsum(self.varying_scalar_values["deposited_power"][i])
                / self.fixed_scalar_values["total_power"][i]
                for i in range(self.get_number_of_beams())
            ]

        if "remaining_power" in derived_quantities:
            self.varying_scalar_values["remaining_power"] = [
                self.fixed_scalar_values["total_power"][i]
                - np.cumsum(self.varying_scalar_values["deposited_power"][i])
                for i in range(self.get_number_of_beams())
            ]

        if "pitch_angle_intersection_energy" in derived_quantities:
            assert self.has_param("power_law_delta")
            delta = self.get_param("power_law_delta")
            self.varying_scalar_values["pitch_angle_intersection_energy"] = [
                Ec
                / np.sqrt(
                    (
                        1.0
                        - (
                            np.sqrt(np.maximum(0.0, 1.0 - b * (1.0 - mu0**2) / b[0]))
                            / mu0
                        )
                        ** 3
                    )
                    / r ** (-2 / delta)
                )
                for Ec, mu0, b, r in zip(
                    self.get_fixed_scalar_values("lower_cutoff_energy"),
                    self.get_fixed_scalar_values("initial_pitch_angle_cosine"),
                    self.get_varying_scalar_values("b"),
                    self.get_varying_scalar_values("residual_factor"),
                )
            ]

        # if 'collisional_pitch_angle_cosine' in derived_quantities:
        #     assert self.has_param('power_law_delta')
        #     delta = self.get_param('power_law_delta')
        #     self.varying_scalar_values['collisional_pitch_angle_cosine'] = [
        #         mu0*np.cbrt(np.maximum(0.0, 1.0 - r**(-2/delta)))
        #         for mu0, r in zip(
        #             self.get_fixed_scalar_values('initial_pitch_angle_cosine'),
        #             self.get_varying_scalar_values('residual_factor'))
        #     ]

        if "adiabatic_pitch_angle_cosine" in derived_quantities:
            self.varying_scalar_values["adiabatic_pitch_angle_cosine"] = [
                np.sqrt(np.maximum(0.0, 1.0 - b * (1.0 - mu0**2) / b[0]))
                for mu0, b in zip(
                    self.get_fixed_scalar_values("initial_pitch_angle_cosine"),
                    self.get_varying_scalar_values("b"),
                )
            ]

        if "total_hydrogen_density" in derived_quantities:
            self.varying_scalar_values["total_hydrogen_density"] = [
                beam_heating.compute_total_hydrogen_density(
                    self.varying_scalar_values["r"][i] * units.U_R
                )
                for i in range(self.get_number_of_beams())
            ]

        if "ionization_fraction" in derived_quantities:
            self.varying_scalar_values["ionization_fraction"] = [
                beam_heating.compute_equilibrium_hydrogen_ionization_fraction(
                    self.varying_scalar_values["tg"][i],
                    self.varying_scalar_values["nel"][i],
                )
                for i in range(self.get_number_of_beams())
            ]

        if "magnetic_flux_density" in derived_quantities:
            self.varying_scalar_values["magnetic_flux_density"] = [
                np.sqrt(
                    self.varying_scalar_values["bx"][i] ** 2
                    + self.varying_scalar_values["by"][i] ** 2
                    + self.varying_scalar_values["bz"][i] ** 2
                )
                * units.U_B
                for i in range(self.get_number_of_beams())
            ]

        if "relative_magnetic_flux_density_change" in derived_quantities:

            def compute_relative_gradient(coords, values):
                grad = np.gradient(values, coords) / values
                grad[values < 0.1] = 0.0
                return grad

            self.varying_scalar_values["relative_magnetic_flux_density_change"] = [
                compute_relative_gradient(
                    self.varying_scalar_values["s"][i],
                    np.sqrt(
                        self.varying_scalar_values["bx"][i] ** 2
                        + self.varying_scalar_values["by"][i] ** 2
                        + self.varying_scalar_values["bz"][i] ** 2
                    )
                    * units.U_B,
                )
                for i in range(self.get_number_of_beams())
            ]

    def _obtain_mean_electron_energies(self):
        if not self.has_fixed_scalar_values("mean_electron_energy"):
            assert self.has_param("power_law_delta")
            delta = self.get_param("power_law_delta")
            self.fixed_scalar_values["mean_electron_energy"] = (
                (delta - 0.5) / (delta - 1.5)
            ) * self.get_fixed_scalar_values("lower_cutoff_energy")

        return self.get_fixed_scalar_values("mean_electron_energy")

    def _obtain_mean_electron_speeds(self):
        if not self.has_fixed_scalar_values("mean_electron_speed"):
            assert self.has_param("power_law_delta")
            delta = self.get_param("power_law_delta")
            self.fixed_scalar_values["mean_electron_speed"] = (
                (delta - 0.5) / (delta - 1)
            ) * np.sqrt(
                2
                * self.get_fixed_scalar_values("lower_cutoff_energy")
                * units.KEV_TO_ERG
                / units.M_ELECTRON
            )  # [cm/s]

        return self.get_fixed_scalar_values("mean_electron_speed")

    def _obtain_beam_electron_densities(self):
        if not self.has_fixed_scalar_values("beam_electron_density"):
            assert self.has_param("power_law_delta")
            delta = self.get_param("power_law_delta")
            self.fixed_scalar_values["beam_electron_density"] = (
                (
                    ((2 * delta - 3.0) / (2 * delta - 1))
                    / (
                        self.get_fixed_scalar_values("lower_cutoff_energy")
                        * units.KEV_TO_ERG
                    )
                )
                * self.get_fixed_scalar_values("total_power")
                / self.get_fixed_scalar_values("acceleration_volume")
            )

    def _obtain_acceleration_site_electron_densities(self):
        if not self.has_fixed_scalar_values("acceleration_site_electron_density"):
            assert self.has_fixed_scalar_values("r0") or self.has_fixed_scalar_values(
                "r"
            )
            self.fixed_scalar_values["acceleration_site_electron_density"] = (
                self.get_fixed_scalar_values(
                    "r0" if self.has_fixed_scalar_values("r0") else "r"
                )
                * units.U_R
                * units.MASS_DENSITY_TO_ELECTRON_DENSITY
            )

        return self.get_fixed_scalar_values("acceleration_site_electron_density")

    def _obtain_beam_electron_fractions(self):
        if not self.has_fixed_scalar_values("beam_electron_fraction"):
            assert self.has_param("particle_energy_fraction")
            assert (
                self.has_fixed_scalar_values("bx")
                and self.has_fixed_scalar_values("by")
                and self.has_fixed_scalar_values("bz")
            )
            assert (
                self.has_fixed_scalar_values("ix")
                and self.has_fixed_scalar_values("iy")
                and self.has_fixed_scalar_values("iz")
            )

            bx = self.get_fixed_scalar_values("bx") * units.U_B
            by = self.get_fixed_scalar_values("by") * units.U_B
            bz = self.get_fixed_scalar_values("bz") * units.U_B
            ix = self.get_fixed_scalar_values("ix")
            iy = self.get_fixed_scalar_values("iy")
            iz = self.get_fixed_scalar_values("iz")
            free_energy = (
                bx * bx
                + by * by
                + bz * bz
                - (bx * ix + by * iy + bz * iz) ** 2 / (ix * ix + iy * iy + iz * iz)
            ) / (8.0 * np.pi)
            mean_electron_energies = (
                self._obtain_mean_electron_energies() * units.KEV_TO_ERG
            )
            electron_densities = self._obtain_acceleration_site_electron_densities()
            self.fixed_scalar_values["beam_electron_fraction"] = (
                self.get_param("particle_energy_fraction")
                * free_energy
                / (mean_electron_energies * electron_densities)
            )

        return self.get_fixed_scalar_values("beam_electron_fraction")

    def _obtain_total_energy_densities(self):
        if not self.has_fixed_scalar_values("total_energy_density"):
            assert self.has_param("acceleration_duration")
            self.fixed_scalar_values["total_energy_density"] = (
                self.get_fixed_scalar_values("total_power")
                * self.get_param("acceleration_duration")
                / self.get_fixed_scalar_values("acceleration_volume")
            )

        return self.get_fixed_scalar_values("total_energy_density")

    def _obtain_acceleration_currents(self):
        if not self.has_fixed_scalar_values("acceleration_current"):
            u_acc = self._obtain_total_energy_densities()
            E_mean = self._obtain_mean_electron_energies() * units.KEV_TO_ERG
            v_mean = self._obtain_mean_electron_speeds()
            self.fixed_scalar_values["acceleration_current"] = (
                units.Q_ELECTRON * u_acc * v_mean / E_mean
            )

        return self.get_fixed_scalar_values("acceleration_current")

    def _obtain_acceleration_induced_magnetic_fields(self):
        if not self.has_fixed_scalar_values("acceleration_induced_magnetic_field"):
            j = self._obtain_acceleration_currents()
            L = np.cbrt(self.get_fixed_scalar_values("acceleration_volume"))
            self.fixed_scalar_values["acceleration_induced_magnetic_field"] = (
                np.pi * j * L / units.CLIGHT
            )

        return self.get_fixed_scalar_values("acceleration_induced_magnetic_field")

    def _obtain_resistivities(self):
        if not self.has_fixed_scalar_values("resistivity"):
            nel = self.get_fixed_scalar_values("nel0")
            tg = self.get_fixed_scalar_values("tg0")
            r = self.get_fixed_scalar_values("r0") * units.U_R
            x = beam_heating.compute_equilibrium_hydrogen_ionization_fraction(tg, nel)
            nH = beam_heating.compute_total_hydrogen_density(r)

            self.fixed_scalar_values["resistivity"] = (
                7.26e-9 * x / tg ** (3 / 2)
            ) * np.log(
                3
                * np.sqrt((units.KBOLTZMANN * tg) ** 3 / (np.pi * nH))
                / (2 * units.Q_ELECTRON**3)
            ) + 7.6e-18 * (
                1 - x
            ) * np.sqrt(
                tg
            ) / x

        return self.get_fixed_scalar_values("resistivity")

    def _obtain_acceleration_induced_electric_fields(self):
        if not self.has_fixed_scalar_values("acceleration_induced_electric_field"):
            j = self._obtain_acceleration_currents()
            eta = self._obtain_resistivities()
            self.fixed_scalar_values["acceleration_induced_electric_field"] = (
                eta * j * units.STATV_TO_V * 1e2
            )

        return self.get_fixed_scalar_values("acceleration_induced_electric_field")

    def _obtain_parallel_electric_fields(self):
        if not self.has_fixed_scalar_values("parallel_electric_field"):
            bx = self.get_fixed_scalar_values("bx0") * units.U_B
            by = self.get_fixed_scalar_values("by0") * units.U_B
            bz = self.get_fixed_scalar_values("bz0") * units.U_B
            ex = self.get_fixed_scalar_values("ex0") * units.U_EL
            ey = self.get_fixed_scalar_values("ey0") * units.U_EL
            ez = self.get_fixed_scalar_values("ez0") * units.U_EL

            self.fixed_scalar_values["parallel_electric_field"] = (
                bx * ex + by * ey + bz * ez
            ) / np.sqrt(bx**2 + by**2 + bz**2)

        return self.get_fixed_scalar_values("parallel_electric_field")

    def _obtain_return_current_heating_ratio(self):
        if not self.has_fixed_scalar_values("return_current_heating_ratio"):
            E_mean = self._obtain_mean_electron_energies()  # [keV]
            mu = self.get_fixed_scalar_values("initial_pitch_angle_cosine")

            nel = self.get_fixed_scalar_values("nel0")
            tg = self.get_fixed_scalar_values("tg0")
            r = self.get_fixed_scalar_values("r0") * units.U_R
            x = beam_heating.compute_equilibrium_hydrogen_ionization_fraction(tg, nel)
            r = self.get_fixed_scalar_values("r0") * units.U_R
            nH = beam_heating.compute_total_hydrogen_density(r)  # [1/cm^3]

            electron_coulomb_logarithm = (
                beam_heating.compute_electron_coulomb_logarithm(nel, E_mean)
            )
            neutral_hydrogen_coulomb_logarithm = (
                beam_heating.compute_neutral_hydrogen_coulomb_logarithm(E_mean)
            )
            gamma = beam_heating.compute_effective_coulomb_logarithm(
                x,
                electron_coulomb_logarithm,
                neutral_hydrogen_coulomb_logarithm,
            )

            E = self._obtain_acceleration_induced_electric_fields() / (
                units.STATV_TO_V * 1e2
            )  # [statV/cm]
            e = units.Q_ELECTRON  # [statC]

            self.fixed_scalar_values["return_current_heating_ratio"] = (E * e / nH) / (
                2 * np.pi * e**4 * gamma / (mu * E_mean * units.KEV_TO_ERG)
            )

        return self.get_fixed_scalar_values("return_current_heating_ratio")

    def _obtain_acceleration_ambient_magnetic_field(self):
        if not self.has_fixed_scalar_values("acceleration_ambient_magnetic_field"):
            bx = self.get_fixed_scalar_values("bx0")
            by = self.get_fixed_scalar_values("by0")
            bz = self.get_fixed_scalar_values("bz0")

            B = np.sqrt(bx**2 + by**2 + bz**2) * units.U_B

            self.fixed_scalar_values["acceleration_ambient_magnetic_field"] = B

        return self.get_fixed_scalar_values("acceleration_ambient_magnetic_field")


class AccelerationSites(field_lines.FieldLineSet3):

    VALUE_DESCRIPTIONS = ElectronBeamSwarm.VALUE_DESCRIPTIONS
    VALUE_UNIT_CONVERTERS = ElectronBeamSwarm.VALUE_UNIT_CONVERTERS

    def __init__(
        self,
        domain_bounds,
        number_of_sites,
        fixed_scalar_values,
        fixed_vector_values,
        varying_scalar_values,
        varying_vector_values,
        params={},
        derived_quantities=[],
        verbose=False,
    ):
        self.number_of_sites = number_of_sites
        super().__init__(
            domain_bounds,
            number_of_sites,
            fixed_scalar_values,
            fixed_vector_values,
            varying_scalar_values,
            varying_vector_values,
            params=params,
            derived_quantities=derived_quantities,
            verbose=verbose,
        )

    def get_number_of_sites(self):
        return self.number_of_sites


def find_beams_starting_in_coords(
    x_coords,
    y_coords,
    z_coords,
    propagation_senses,
    fixed_scalar_values,
    max_propagation_sense_diff=1e-6,
    max_distance=1e-5,
):
    return [
        i
        for i, (x, y, z, s) in enumerate(
            zip(
                fixed_scalar_values["x0"],
                fixed_scalar_values["y0"],
                fixed_scalar_values["z0"],
                fixed_scalar_values["propagation_sense"],
            )
        )
        if np.any(
            np.logical_and(
                (x - x_coords) ** 2 + (y - y_coords) ** 2 + (z - z_coords) ** 2
                <= max_distance**2,
                np.abs(s - propagation_senses) < max_propagation_sense_diff,
            )
        )
    ]


def find_beams_propagating_longer_than_distance(min_distance, fixed_scalar_values):
    return list(
        np.nonzero(fixed_scalar_values["total_propagation_distance"] > min_distance)[0]
    )


def find_beams_in_temperature_height_region(
    height_lims, tg_lims, varying_scalar_values
):
    return [
        i
        for i, (z, tg) in enumerate(
            zip(varying_scalar_values["z"], varying_scalar_values["tg"])
        )
        if np.any(
            (z > -height_lims[1])
            * (z < -height_lims[0])
            * (tg > tg_lims[0])
            * (tg < tg_lims[1])
        )
    ]


def find_peak_deposition_point(varying_scalar_values, field_line_idx):
    deposited_power = varying_scalar_values["deposited_power"][field_line_idx]
    indices, _ = signal.find_peaks(
        deposited_power / np.mean(deposited_power), prominence=2
    )
    return slice(np.max(indices) if indices.size > 0 else -1, None, None)


def plot_electron_beams(*args, **kwargs):
    return field_lines.plot_field_lines(*args, **kwargs)


def plot_electron_beam_properties(*args, **kwargs):
    return field_lines.plot_field_line_properties(*args, **kwargs)


def plot_beam_value_histogram(*args, **kwargs):
    return field_lines.plot_field_line_value_histogram(*args, **kwargs)


def plot_beam_value_histogram_difference(*args, **kwargs):
    return field_lines.plot_field_line_value_histogram_difference(*args, **kwargs)


def plot_beam_value_2d_histogram(*args, **kwargs):
    return field_lines.plot_field_line_value_2d_histogram(*args, **kwargs)


def plot_beam_value_2d_histogram_difference(*args, **kwargs):
    return field_lines.plot_field_line_value_2d_histogram_difference(*args, **kwargs)


def plot_beam_value_2d_histogram_comparison(*args, **kwargs):
    return field_lines.plot_field_line_value_2d_histogram_comparison(*args, **kwargs)
