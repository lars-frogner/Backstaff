//! Simple model for acceleration of non-thermal electron beams described by power-law distributions.

use super::super::super::super::accelerator::Accelerator;
use super::super::super::super::detection::ReconnectionSiteDetector;
use super::super::super::super::feb;
use super::super::{PowerLawDistribution, PowerLawDistributionConfig, PowerLawDistributionData};
use crate::constants::{INFINITY, KBOLTZMANN, KEV_TO_ERG, PI};
use crate::geometry::{Dim3, Idx3, Point3, Vec3};
use crate::grid::Grid3;
use crate::interpolation::Interpolator3;
use crate::io::snapshot::{fdt, SnapshotCacher3, SnapshotReader3};
use crate::io::Verbose;
use crate::plasma::ionization;
use crate::tracing::stepping::{StepperFactory3, SteppingSense};
use crate::units::solar::{U_E, U_L, U_L3, U_R, U_T};
use rand::{self, Rng};
use rayon::prelude::*;
use std::io;
use Dim3::{X, Y, Z};

/// Configuration parameters for the simple power-law acceleration model.
#[derive(Clone, Debug)]
pub struct SimplePowerLawAccelerationConfig {
    /// Duration of the acceleration events [s].
    pub acceleration_duration: feb,
    /// Fraction of the released reconnection energy going into acceleration of electrons.
    pub particle_energy_fraction: feb,
    /// Exponent of the inverse power-law describing the non-thermal electron distribution.
    pub power_law_delta: feb,
    /// Distributions with total power densities smaller than this value are discarded [erg/(cm^3 s)].
    pub min_total_power_density: feb,
    /// Distributions with an estimated depletion distance smaller than this value
    /// are discarded [Mm].
    pub min_depletion_distance: feb,
    /// Distributions with initial absolute pitch angles larger than this are discarded [deg].
    pub max_pitch_angle: feb,
    /// Distributions with electric field directions angled more than this away from
    /// the magnetic field axis are discarded [deg].
    pub max_electric_field_angle: feb,
    /// Distributions exceeding the maximum mass density are discarded if they have a
    /// temperature smaller than this value [K].
    pub min_temperature: feb,
    /// Distributions below the minimum temperature are discarded if they have a mass
    /// density higher than this value [g/cm^3].
    pub max_mass_density: feb,
    /// Accepted distributions will be included with this probability.
    pub inclusion_probability: feb,
    /// Initial guess to use when estimating lower cut-off energy [keV].
    pub initial_cutoff_energy_guess: feb,
    /// Target relative error when estimating lower cut-off energy.
    pub acceptable_root_finding_error: feb,
    /// Maximum number of iterations when estimating lower cut-off energy.
    pub max_root_finding_iterations: i32,
}

// Simple acceleration process producing power-law distributions of non-thermal electrons.
#[derive(Clone, Debug)]
pub struct SimplePowerLawAccelerator {
    distribution_config: PowerLawDistributionConfig,
    config: SimplePowerLawAccelerationConfig,
    pitch_angle_cosine_threshold: feb,
    electric_field_angle_cosine_threshold: feb,
}

impl SimplePowerLawAccelerator {
    /// How many adjacent grid cells in each direction to include when
    /// computing the average electric field around the acceleration site.
    const ELECTRIC_FIELD_PROBING_SPAN: isize = 0;

    /// Smallest mean electron energy that will be used to compute Coulomb logarithms [keV].
    const MIN_COULOMB_LOG_MEAN_ENERGY: feb = 0.1;

    fn determine_total_power_density<G>(
        &self,
        snapshot: &SnapshotCacher3<G>,
        indices: &Idx3<usize>,
    ) -> feb
    where
        G: Grid3<fdt>,
    {
        let joule_heating_field = snapshot.cached_scalar_field("qjoule");
        let joule_heating = feb::from(joule_heating_field.value(indices));
        let joule_heating = feb::max(0.0, joule_heating * U_E / U_T); // [erg/(cm^3 s)]

        self.config.particle_energy_fraction * joule_heating
    }

    fn determine_acceleration_volume<G>(
        &self,
        snapshot: &SnapshotCacher3<G>,
        indices: &Idx3<usize>,
    ) -> feb
    where
        G: Grid3<fdt>,
    {
        feb::from(snapshot.reader().grid().grid_cell_volume(indices)) * U_L3 // [cm^3]
    }

    fn determine_acceleration_position<G>(
        snapshot: &SnapshotCacher3<G>,
        indices: &Idx3<usize>,
    ) -> Point3<fdt>
    where
        G: Grid3<fdt>,
    {
        snapshot.reader().grid().centers().point(indices)
    }

    fn determine_electric_field_direction<G>(
        snapshot: &SnapshotCacher3<G>,
        indices: &Idx3<usize>,
    ) -> Option<Vec3<fdt>>
    where
        G: Grid3<fdt>,
    {
        let electric_field = snapshot.cached_vector_field("e");
        let grid = electric_field.grid();

        let lower_indices = Idx3::new(
            indices[X] as isize - Self::ELECTRIC_FIELD_PROBING_SPAN,
            indices[Y] as isize - Self::ELECTRIC_FIELD_PROBING_SPAN,
            indices[Z] as isize - Self::ELECTRIC_FIELD_PROBING_SPAN,
        );
        let upper_indices = Idx3::new(
            indices[X] as isize + Self::ELECTRIC_FIELD_PROBING_SPAN + 1,
            indices[Y] as isize + Self::ELECTRIC_FIELD_PROBING_SPAN + 1,
            indices[Z] as isize + Self::ELECTRIC_FIELD_PROBING_SPAN + 1,
        );

        let mut total_electric_vector = Vec3::zero();

        for &k in grid
            .create_idx_range_list(Z, lower_indices[Z], upper_indices[Z])
            .iter()
        {
            for &j in grid
                .create_idx_range_list(Y, lower_indices[Y], upper_indices[Y])
                .iter()
            {
                for &i in grid
                    .create_idx_range_list(X, lower_indices[X], upper_indices[X])
                    .iter()
                {
                    total_electric_vector =
                        total_electric_vector + electric_field.vector(&Idx3::new(i, j, k));
                }
            }
        }
        let squared_total_electric_vector = total_electric_vector.squared_length();

        if squared_total_electric_vector > std::f32::EPSILON {
            Some(total_electric_vector / fdt::sqrt(squared_total_electric_vector))
        } else {
            None
        }
    }

    fn determine_magnetic_field_direction<G, I>(
        snapshot: &SnapshotCacher3<G>,
        interpolator: &I,
        acceleration_position: &Point3<fdt>,
    ) -> Vec3<fdt>
    where
        G: Grid3<fdt>,
        I: Interpolator3,
    {
        let magnetic_field = snapshot.cached_vector_field("b");
        let mut magnetic_field_direction = interpolator
            .interp_vector_field(magnetic_field, acceleration_position)
            .expect_inside();
        magnetic_field_direction.normalize();
        magnetic_field_direction
    }

    fn compute_electric_field_angle_cosine(
        &self,
        magnetic_field_direction: &Vec3<fdt>,
        electric_field_direction: &Vec3<fdt>,
    ) -> feb {
        feb::from(electric_field_direction.dot(magnetic_field_direction))
    }

    fn compute_power_density_partition(
        &self,
        total_power_density: feb,
        electric_field_angle_cosine: feb,
    ) -> (Option<feb>, Option<feb>) {
        (
            {
                let backward_power_density =
                    0.5 * (1.0 + electric_field_angle_cosine) * total_power_density;
                if backward_power_density >= self.config.min_total_power_density {
                    Some(backward_power_density)
                } else {
                    None
                }
            },
            {
                let forward_power_density =
                    0.5 * (1.0 - electric_field_angle_cosine) * total_power_density;
                if forward_power_density >= self.config.min_total_power_density {
                    Some(forward_power_density)
                } else {
                    None
                }
            },
        )
    }

    fn determine_temperature<G>(snapshot: &SnapshotCacher3<G>, indices: &Idx3<usize>) -> feb
    where
        G: Grid3<fdt>,
    {
        feb::from(snapshot.cached_scalar_field("tg").value(indices))
    }

    fn determine_electron_density<G>(snapshot: &SnapshotCacher3<G>, indices: &Idx3<usize>) -> feb
    where
        G: Grid3<fdt>,
    {
        feb::from(snapshot.cached_scalar_field("nel").value(indices)) // [1/cm^3]
    }

    fn determine_mass_density<G>(snapshot: &SnapshotCacher3<G>, indices: &Idx3<usize>) -> feb
    where
        G: Grid3<fdt>,
    {
        feb::from(snapshot.cached_scalar_field("r").value(indices)) * U_R // [g/cm^3]
    }

    /// Estimates the lower cut-off energy of the non-thermal distribution by
    /// determining where the power-law intersects the thermal Maxwell-Boltzmann
    /// distribution.
    ///
    /// More precisely, the method determines the cut-off energy `Ec` such that
    /// `ne*P_MB(Ec) = n_acc(Ec)*P_PL(Ec)`, where `n_acc(Ec) = e_acc/(Ec*(delta - 1)/(delta - 2))`
    /// is the number density of non-thermal electrons, `e_acc` is their energy density and
    /// `P_MB` and `P_PL` are respectively the Maxwell-Boltzmann and power-law probability
    /// distributions.
    fn compute_lower_cutoff_energy(
        &self,
        temperature: feb,
        electron_density: feb,
        total_power_density: feb,
    ) -> Option<feb> {
        let total_energy_density = self.compute_total_energy_density(total_power_density);
        if total_energy_density < std::f64::EPSILON {
            return None;
        }
        let beta = KEV_TO_ERG / (KBOLTZMANN * temperature); // [1/keV]
        let thermal_fraction =
            KEV_TO_ERG * electron_density * feb::sqrt(4.0 * feb::powi(beta, 3) / PI)
                / (total_energy_density * (self.config.power_law_delta - 2.0)); // [1/keV^(5/2)]
        let ln_thermal_fraction = feb::ln(thermal_fraction);

        // Make sure the initial guess satisfies E > 5/(2*beta), so that we find the solution
        // on the correct side
        let minimum_energy = 2.5 / beta + 1e-4;
        let mut energy = feb::max(minimum_energy, self.config.initial_cutoff_energy_guess);

        let mut number_of_iterations = 0;

        loop {
            let difference = ln_thermal_fraction + 2.5 * feb::ln(energy) - beta * energy;

            // Stop when the difference between the distributions is sufficiently close to zero
            if feb::abs(difference) < self.config.acceptable_root_finding_error {
                break;
            }

            energy -= difference / (2.5 / energy - beta);

            // If we reach the wrong side of the thermal distribution peak there is no solution
            if energy < minimum_energy {
                return None;
            }

            number_of_iterations += 1;

            if number_of_iterations > self.config.max_root_finding_iterations {
                println!(
                    "Cut-off energy estimation reached maximum number of iterations with error {}",
                    difference
                );
                return None;
            }
        }

        Some(energy)
    }

    fn compute_total_energy_density(&self, total_power_density: feb) -> feb {
        total_power_density * self.config.acceleration_duration // [erg/cm^3]
    }

    fn compute_initial_pitch_angle_cosine(
        &self,
        temperature: feb,
        delta: feb,
        lower_cutoff_energy: feb,
    ) -> Option<feb> {
        let squared_perpendicular_fraction = (8.0 * KBOLTZMANN * temperature / PI)
            / (feb::powi((2.0 * delta - 2.0) / (2.0 * delta - 3.0), 2)
                * 2.0
                * lower_cutoff_energy
                * KEV_TO_ERG);
        if squared_perpendicular_fraction <= 1.0 {
            let pitch_angle_cosine = feb::sqrt(1.0 - squared_perpendicular_fraction);
            if pitch_angle_cosine >= self.pitch_angle_cosine_threshold {
                Some(pitch_angle_cosine)
            } else {
                None
            }
        } else {
            None
        }
    }
}

impl SimplePowerLawAccelerator {
    /// Creates a new simple power-law accelerator.
    ///
    /// # Parameters
    ///
    /// - `distribution_config`: Configuration parameters for the distribution.
    /// - `config`: Configuration parameters for the accelerator.
    ///
    /// # Returns
    ///
    /// A new `SimplePowerLawAccelerator`.
    pub fn new(
        distribution_config: PowerLawDistributionConfig,
        config: SimplePowerLawAccelerationConfig,
    ) -> Self {
        distribution_config.validate();
        config.validate();

        let pitch_angle_cosine_threshold = feb::cos(config.max_pitch_angle.to_radians());
        let electric_field_angle_cosine_threshold =
            feb::cos(config.max_electric_field_angle.to_radians());

        SimplePowerLawAccelerator {
            distribution_config,
            config,
            pitch_angle_cosine_threshold,
            electric_field_angle_cosine_threshold,
        }
    }
}

impl Accelerator for SimplePowerLawAccelerator {
    type DistributionType = PowerLawDistribution;
    type AccelerationDataCollectionType = ();

    fn generate_distributions<G, D, I, StF>(
        &self,
        snapshot: &mut SnapshotCacher3<G>,
        detector: D,
        interpolator: &I,
        _stepper_factory: &StF,
        verbose: Verbose,
    ) -> io::Result<(
        Vec<Self::DistributionType>,
        Self::AccelerationDataCollectionType,
    )>
    where
        G: Grid3<fdt>,
        D: ReconnectionSiteDetector,
        I: Interpolator3,
        StF: StepperFactory3 + Sync,
    {
        let seeder = detector.detect_reconnection_sites(snapshot, verbose);

        if verbose.is_yes() {
            println!("Computing total beam powers");
        }
        snapshot.cache_scalar_field("qjoule")?;
        let properties: Vec<_> = seeder
            .into_par_iter()
            .filter_map(|indices| {
                if self.config.inclusion_probability < 1.0
                    && rand::thread_rng().gen::<feb>() >= self.config.inclusion_probability
                {
                    None
                } else {
                    let total_power_density =
                        self.determine_total_power_density(snapshot, &indices);
                    if total_power_density < self.config.min_total_power_density {
                        None
                    } else {
                        Some((indices, total_power_density))
                    }
                }
            })
            .collect();
        snapshot.drop_scalar_field("qjoule");

        if verbose.is_yes() {
            println!("Computing magnetic and electric field directions");
        }
        snapshot.cache_vector_field("b")?;
        snapshot.cache_vector_field("e")?;
        let properties: Vec<_> = properties
            .into_par_iter()
            .filter_map(|(indices, total_power_density)| {
                let acceleration_position =
                    Self::determine_acceleration_position(snapshot, &indices);
                match Self::determine_electric_field_direction(snapshot, &indices) {
                    Some(electric_field_direction) => {
                        let magnetic_field_direction = Self::determine_magnetic_field_direction(
                            snapshot,
                            interpolator,
                            &acceleration_position,
                        );
                        let electric_field_angle_cosine = self.compute_electric_field_angle_cosine(
                            &magnetic_field_direction,
                            &electric_field_direction,
                        );
                        let partitioned_power_densities = self.compute_power_density_partition(
                            total_power_density,
                            electric_field_angle_cosine,
                        );
                        if let (None, None) = partitioned_power_densities {
                            None
                        } else {
                            Some((
                                indices,
                                total_power_density,
                                acceleration_position,
                                partitioned_power_densities,
                                electric_field_angle_cosine,
                            ))
                        }
                    }
                    None => None,
                }
            })
            .collect();
        snapshot.drop_vector_field("e");

        if verbose.is_yes() {
            println!("Computing lower cutoff energies and estimating stopping distances");
        }
        snapshot.cache_scalar_field("nel")?;
        snapshot.cache_scalar_field("r")?;
        snapshot.cache_scalar_field("tg")?;
        let distributions = properties
            .into_par_iter()
            .filter_map(
                |(
                    indices,
                    total_power_density,
                    acceleration_position,
                    partitioned_power_densities,
                    electric_field_angle_cosine,
                )| {
                    let electron_density = Self::determine_electron_density(snapshot, &indices);
                    assert!(
                        electron_density > 0.0,
                        "Electron density must be larger than zero."
                    );

                    let temperature = Self::determine_temperature(snapshot, &indices);
                    assert!(temperature > 0.0, "Temperature must be larger than zero.");

                    let lower_cutoff_energy = match self.compute_lower_cutoff_energy(
                        temperature,
                        electron_density,
                        total_power_density,
                    ) {
                        Some(energy) => energy,
                        None => return None,
                    };
                    let mean_energy = PowerLawDistribution::compute_mean_energy(
                        self.config.power_law_delta,
                        lower_cutoff_energy,
                    );

                    let initial_pitch_angle_cosine = match self.compute_initial_pitch_angle_cosine(
                        temperature,
                        self.config.power_law_delta,
                        lower_cutoff_energy,
                    ) {
                        Some(initial_pitch_angle_cosine) => initial_pitch_angle_cosine,
                        None => return None,
                    };

                    let coulomb_logarithm_energy =
                        feb::max(mean_energy, Self::MIN_COULOMB_LOG_MEAN_ENERGY);

                    let electron_coulomb_logarithm =
                        PowerLawDistribution::compute_electron_coulomb_logarithm(
                            electron_density,
                            coulomb_logarithm_energy,
                        );
                    let neutral_hydrogen_coulomb_logarithm =
                        PowerLawDistribution::compute_neutral_hydrogen_coulomb_logarithm(
                            coulomb_logarithm_energy,
                        );

                    let ionization_fraction =
                        ionization::compute_equilibrium_hydrogen_ionization_fraction(
                            temperature,
                            electron_density,
                        );
                    let effective_coulomb_logarithm =
                        PowerLawDistribution::compute_effective_coulomb_logarithm(
                            ionization_fraction,
                            electron_coulomb_logarithm,
                            neutral_hydrogen_coulomb_logarithm,
                        );
                    let mass_density = Self::determine_mass_density(snapshot, &indices);
                    let total_hydrogen_density =
                        PowerLawDistribution::compute_total_hydrogen_density(mass_density);

                    let acceleration_volume =
                        self.determine_acceleration_volume(snapshot, &indices);

                    let stopping_ionized_column_depth =
                        PowerLawDistribution::compute_stopping_column_depth(
                            initial_pitch_angle_cosine,
                            lower_cutoff_energy,
                            electron_coulomb_logarithm,
                        );

                    if temperature < self.config.min_temperature
                        && mass_density > self.config.max_mass_density
                    {
                        None
                    } else {
                        let mut distributions = Vec::with_capacity(2);
                        if let (Some(backward_power_density), _) = partitioned_power_densities {
                            let backward_power = PowerLawDistribution::compute_total_power(
                                backward_power_density,
                                acceleration_volume,
                            );
                            let heating_scale = PowerLawDistribution::compute_heating_scale(
                                backward_power,
                                self.config.power_law_delta,
                                initial_pitch_angle_cosine,
                                lower_cutoff_energy,
                            );
                            let estimated_depletion_distance =
                                PowerLawDistribution::estimate_depletion_distance(
                                    self.config.power_law_delta,
                                    self.distribution_config.min_residual_factor,
                                    self.distribution_config.min_deposited_power_per_distance,
                                    total_hydrogen_density,
                                    effective_coulomb_logarithm,
                                    electron_coulomb_logarithm,
                                    stopping_ionized_column_depth,
                                    heating_scale,
                                );
                            if estimated_depletion_distance
                                >= self.config.min_depletion_distance * U_L
                            {
                                let distribution_data = PowerLawDistributionData {
                                    delta: self.config.power_law_delta,
                                    initial_pitch_angle_cosine,
                                    total_power: backward_power,
                                    total_power_density: backward_power_density,
                                    lower_cutoff_energy,
                                    acceleration_position: acceleration_position.clone(),
                                    acceleration_volume,
                                    propagation_sense: SteppingSense::Opposite,
                                    electron_coulomb_logarithm,
                                    neutral_hydrogen_coulomb_logarithm,
                                    heating_scale,
                                    stopping_ionized_column_depth,
                                    estimated_depletion_distance,
                                    electric_field_angle_cosine,
                                };
                                distributions.push(PowerLawDistribution::new(
                                    self.distribution_config.clone(),
                                    distribution_data,
                                ));
                            }
                        }
                        if let (_, Some(forward_power_density)) = partitioned_power_densities {
                            let forward_power = PowerLawDistribution::compute_total_power(
                                forward_power_density,
                                acceleration_volume,
                            );
                            let heating_scale = PowerLawDistribution::compute_heating_scale(
                                forward_power,
                                self.config.power_law_delta,
                                initial_pitch_angle_cosine,
                                lower_cutoff_energy,
                            );
                            let estimated_depletion_distance =
                                PowerLawDistribution::estimate_depletion_distance(
                                    self.config.power_law_delta,
                                    self.distribution_config.min_residual_factor,
                                    self.distribution_config.min_deposited_power_per_distance,
                                    total_hydrogen_density,
                                    effective_coulomb_logarithm,
                                    electron_coulomb_logarithm,
                                    stopping_ionized_column_depth,
                                    heating_scale,
                                );
                            if estimated_depletion_distance
                                >= self.config.min_depletion_distance * U_L
                            {
                                let distribution_data = PowerLawDistributionData {
                                    delta: self.config.power_law_delta,
                                    initial_pitch_angle_cosine,
                                    total_power: forward_power,
                                    total_power_density: forward_power_density,
                                    lower_cutoff_energy,
                                    acceleration_position,
                                    acceleration_volume,
                                    propagation_sense: SteppingSense::Same,
                                    electron_coulomb_logarithm,
                                    neutral_hydrogen_coulomb_logarithm,
                                    heating_scale,
                                    stopping_ionized_column_depth,
                                    estimated_depletion_distance,
                                    electric_field_angle_cosine,
                                };
                                distributions.push(PowerLawDistribution::new(
                                    self.distribution_config.clone(),
                                    distribution_data,
                                ));
                            }
                        }
                        if distributions.is_empty() {
                            None
                        } else {
                            Some(distributions)
                        }
                    }
                },
            )
            .flatten()
            .collect();

        Ok((distributions, ()))
    }
}

impl SimplePowerLawAccelerationConfig {
    pub const DEFAULT_ACCELERATION_DURATION: feb = 1.0; // [s]
    pub const DEFAULT_PARTICLE_ENERGY_FRACTION: feb = 0.2;
    pub const DEFAULT_POWER_LAW_DELTA: feb = 4.0;
    pub const DEFAULT_MIN_TOTAL_POWER_DENSITY: feb = 1e-2; // [erg/(cm^3 s)]
    pub const DEFAULT_MIN_DEPLETION_DISTANCE: feb = 0.5; // [Mm]
    pub const DEFAULT_MAX_PITCH_ANGLE: feb = 70.0; // [deg]
    pub const DEFAULT_MAX_ELECTRIC_FIELD_ANGLE: feb = 90.0; // [deg]
    pub const DEFAULT_MIN_TEMPERATURE: feb = 0.0; // [K]
    pub const DEFAULT_MAX_MASS_DENSITY: feb = INFINITY; // [g/cm^3]
    pub const DEFAULT_INCLUSION_PROBABILITY: feb = 1.0;
    pub const DEFAULT_INITIAL_CUTOFF_ENERGY_GUESS: feb = 2.0; // [keV]
    pub const DEFAULT_ACCEPTABLE_ROOT_FINDING_ERROR: feb = 1e-3;
    pub const DEFAULT_MAX_ROOT_FINDING_ITERATIONS: i32 = 100;

    /// Creates a set of simple power law accelerator configuration parameters with
    /// values read from the specified parameter file when available, otherwise
    /// falling back to the hardcoded defaults.
    pub fn with_defaults_from_param_file<G: Grid3<fdt>>(reader: &SnapshotReader3<G>) -> Self {
        let acceleration_duration = reader
            .get_converted_numerical_param_or_fallback_to_default_with_warning(
                "acceleration_duration",
                "dt",
                &|dt: feb| dt * U_T,
                Self::DEFAULT_ACCELERATION_DURATION,
            );
        let particle_energy_fraction = reader
            .get_converted_numerical_param_or_fallback_to_default_with_warning(
                "particle_energy_fraction",
                "qjoule_acc_frac",
                &|qjoule_acc_frac: feb| qjoule_acc_frac,
                Self::DEFAULT_PARTICLE_ENERGY_FRACTION,
            );
        let power_law_delta = reader
            .get_converted_numerical_param_or_fallback_to_default_with_warning(
                "power_law_delta",
                "power_law_index",
                &|power_law_index| power_law_index,
                Self::DEFAULT_POWER_LAW_DELTA,
            );
        let min_total_power_density = reader
            .get_converted_numerical_param_or_fallback_to_default_with_warning(
                "min_total_power_density",
                "min_beam_en",
                &|min_beam_en: feb| min_beam_en * U_E / U_T,
                Self::DEFAULT_MIN_TOTAL_POWER_DENSITY,
            );
        let min_depletion_distance = reader
            .get_converted_numerical_param_or_fallback_to_default_with_warning(
                "min_depletion_distance",
                "min_stop_dist",
                &|min_stop_dist: feb| min_stop_dist,
                Self::DEFAULT_MIN_DEPLETION_DISTANCE,
            );
        let max_pitch_angle = reader
            .get_converted_numerical_param_or_fallback_to_default_with_warning(
                "max_pitch_angle",
                "max_pitch_ang",
                &|max_pitch_ang: feb| max_pitch_ang,
                Self::DEFAULT_MAX_PITCH_ANGLE,
            );
        let max_electric_field_angle = reader
            .get_converted_numerical_param_or_fallback_to_default_with_warning(
                "max_electric_field_angle",
                "max_efield_ang",
                &|max_efield_ang: feb| max_efield_ang,
                Self::DEFAULT_MAX_PITCH_ANGLE,
            );
        SimplePowerLawAccelerationConfig {
            acceleration_duration,
            particle_energy_fraction,
            power_law_delta,
            min_total_power_density,
            min_depletion_distance,
            max_pitch_angle,
            max_electric_field_angle,
            ..Self::default()
        }
    }

    /// Panics if any of the configuration parameter values are invalid.
    fn validate(&self) {
        assert!(
            self.acceleration_duration >= 0.0,
            "Duration must be larger than or equal to zero."
        );
        assert!(
            self.particle_energy_fraction >= 0.0 && self.particle_energy_fraction <= 1.0,
            "Particle energy fraction must be in the range [0, 1]."
        );
        assert!(
            self.power_law_delta > 2.0,
            "Power-law delta must be larger than two."
        );
        assert!(
            self.min_total_power_density >= 0.0,
            "Minimum total power density must be larger than or equal to zero."
        );
        assert!(
            self.min_depletion_distance >= 0.0,
            "Minimum stopping distance must be larger than or equal to zero."
        );
        assert!(
            self.max_pitch_angle >= 0.0 && self.max_pitch_angle < 90.0,
            "Maximum pitch angle must be in the range [0, 90)."
        );
        assert!(
            self.max_electric_field_angle >= 0.0 && self.max_electric_field_angle <= 90.0,
            "Maximum electric field angle must be in the range [0, 90]."
        );
        assert!(
            self.min_temperature >= 0.0,
            "Minimum temperature must be larger than or equal to zero."
        );
        assert!(
            self.max_mass_density >= 0.0,
            "Maximum mass density must be larger than or equal to zero."
        );
        assert!(
            self.inclusion_probability >= 0.0 && self.inclusion_probability <= 1.0,
            "Inclusion probability must be in the range [0, 1]."
        );
        assert!(
            self.initial_cutoff_energy_guess > 0.0,
            "Initial cut-off energy guess must be larger than zero."
        );
        assert!(
            self.acceptable_root_finding_error > 0.0,
            "Acceptable root finding error must be larger than zero."
        );
        assert!(
            self.max_root_finding_iterations > 0,
            "Maximum number of root finding iterations must be larger than zero."
        );
    }
}

impl Default for SimplePowerLawAccelerationConfig {
    fn default() -> Self {
        SimplePowerLawAccelerationConfig {
            acceleration_duration: Self::DEFAULT_ACCELERATION_DURATION,
            particle_energy_fraction: Self::DEFAULT_PARTICLE_ENERGY_FRACTION,
            power_law_delta: Self::DEFAULT_POWER_LAW_DELTA,
            min_total_power_density: Self::DEFAULT_MIN_TOTAL_POWER_DENSITY,
            min_depletion_distance: Self::DEFAULT_MIN_DEPLETION_DISTANCE,
            max_pitch_angle: Self::DEFAULT_MAX_PITCH_ANGLE,
            max_electric_field_angle: Self::DEFAULT_MAX_ELECTRIC_FIELD_ANGLE,
            min_temperature: Self::DEFAULT_MIN_TEMPERATURE,
            max_mass_density: Self::DEFAULT_MAX_MASS_DENSITY,
            inclusion_probability: Self::DEFAULT_INCLUSION_PROBABILITY,
            initial_cutoff_energy_guess: Self::DEFAULT_INITIAL_CUTOFF_ENERGY_GUESS,
            acceptable_root_finding_error: Self::DEFAULT_ACCEPTABLE_ROOT_FINDING_ERROR,
            max_root_finding_iterations: Self::DEFAULT_MAX_ROOT_FINDING_ITERATIONS,
        }
    }
}
