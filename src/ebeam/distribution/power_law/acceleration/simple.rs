//! Simple model for acceleration of non-thermal electron beams described by power-law distributions.

use super::super::super::super::accelerator::Accelerator;
use super::super::super::super::detection::ReconnectionSiteDetector;
use super::super::super::super::feb;
use super::super::{PowerLawDistribution, PowerLawDistributionConfig, PowerLawDistributionData};
use crate::constants::{KBOLTZMANN, KEV_TO_ERG, MC2_ELECTRON, PI};
use crate::geometry::{Dim3, Idx3, Point3, Vec3};
use crate::grid::Grid3;
use crate::interpolation::Interpolator3;
use crate::io::snapshot::{fdt, SnapshotCacher3, SnapshotReader3};
use crate::io::Verbose;
use crate::plasma::ionization;
use crate::tracing::stepping::{StepperFactory3, SteppingSense};
use crate::units::solar::{U_E, U_L, U_L3, U_R, U_T};
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
    /// Distributions with lower cut-off energies lower than this value are discarded [keV].
    pub min_lower_cutoff_energy: feb,
    /// Distributions with an estimated thermalization distance smaller than this value
    /// are discarded [Mm].
    pub min_thermalization_distance: feb,
    /// Distributions with initial absolute pitch angles larger than this are discarded [deg].
    pub max_pitch_angle: feb,
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
}

impl SimplePowerLawAccelerator {
    /// How many adjacent grid cells in each direction to include when
    /// computing the average electric field around the acceleration site.
    const ELECTRIC_FIELD_PROBING_SPAN: isize = 0;

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

    fn determine_acceleration_direction<G>(
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
            // Electrons are accelerated in the opposite direction as the electric field.
            Some(total_electric_vector / (-fdt::sqrt(squared_total_electric_vector)))
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
    /// More precisely, the method determines the cut-off energy `E_cut` such that
    /// `e_therm*P_MB(E_cut) = e_acc*P_PL(E_cut)`,
    /// where `e_therm = 3*ne/(2*beta)` is the energy density of thermal electrons,
    /// `e_acc` is the energy density of non-thermal electrons and `P_MB` and `P_PL`
    /// are respectively the Maxwell-Boltzmann and power-law probability distributions.
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
        let thermal_fraction = KEV_TO_ERG * 3.0 * electron_density * feb::sqrt(beta / PI)
            / (total_energy_density * (self.config.power_law_delta - 1.0)); // [1/keV^(3/2)]
        let ln_thermal_fraction = feb::ln(thermal_fraction);

        // Make sure the initial guess satisfies E > 3/(2*beta), so that we find the solution
        // on the correct side
        let minimum_energy = 1.5 / beta + 1e-4;
        let mut energy = feb::max(minimum_energy, self.config.initial_cutoff_energy_guess);

        let mut number_of_iterations = 0;

        loop {
            let difference = ln_thermal_fraction + 1.5 * feb::ln(energy) - beta * energy;

            // Stop when the difference between the distributions is sufficiently close to zero
            if feb::abs(difference) < self.config.acceptable_root_finding_error {
                break;
            }

            energy -= difference / (1.5 / energy - beta);

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

        if energy >= self.config.min_lower_cutoff_energy {
            let lower_cutoff_energy = energy * KEV_TO_ERG / MC2_ELECTRON;
            Some(lower_cutoff_energy)
        } else {
            None
        }
    }

    fn compute_total_energy_density(&self, total_power_density: feb) -> feb {
        total_power_density * self.config.acceleration_duration // [erg/cm^3]
    }

    fn compute_initial_pitch_angle_cosine(
        magnetic_field_direction: &Vec3<fdt>,
        acceleration_direction: &Vec3<fdt>,
    ) -> feb {
        feb::from(acceleration_direction.dot(magnetic_field_direction))
    }

    fn find_propagation_sense(pitch_angle_cosine: feb) -> SteppingSense {
        if pitch_angle_cosine >= 0.0 {
            SteppingSense::Same
        } else {
            SteppingSense::Opposite
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

        SimplePowerLawAccelerator {
            distribution_config,
            config,
            pitch_angle_cosine_threshold,
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
                let total_power_density = self.determine_total_power_density(snapshot, &indices);
                if total_power_density < self.config.min_total_power_density {
                    None
                } else {
                    Some((indices, total_power_density))
                }
            })
            .collect();
        snapshot.drop_scalar_field("qjoule");

        if verbose.is_yes() {
            println!("Computing initial pitch angles");
        }
        snapshot.cache_vector_field("b")?;
        snapshot.cache_vector_field("e")?;
        let properties: Vec<_> = properties
            .into_par_iter()
            .filter_map(|(indices, total_power_density)| {
                let acceleration_position =
                    Self::determine_acceleration_position(snapshot, &indices);
                match Self::determine_acceleration_direction(snapshot, &indices) {
                    Some(acceleration_direction) => {
                        let magnetic_field_direction = Self::determine_magnetic_field_direction(
                            snapshot,
                            interpolator,
                            &acceleration_position,
                        );
                        let initial_pitch_angle_cosine = Self::compute_initial_pitch_angle_cosine(
                            &magnetic_field_direction,
                            &acceleration_direction,
                        );
                        if feb::abs(initial_pitch_angle_cosine) < self.pitch_angle_cosine_threshold
                        {
                            None
                        } else {
                            let propagation_sense =
                                Self::find_propagation_sense(initial_pitch_angle_cosine);
                            Some((
                                indices,
                                total_power_density,
                                acceleration_position,
                                initial_pitch_angle_cosine,
                                propagation_sense,
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
                    initial_pitch_angle_cosine,
                    propagation_sense,
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

                    let electron_coulomb_logarithm =
                        PowerLawDistribution::compute_electron_coulomb_logarithm(
                            electron_density,
                            mean_energy,
                        );
                    let neutral_hydrogen_coulomb_logarithm =
                        PowerLawDistribution::compute_neutral_hydrogen_coulomb_logarithm(
                            mean_energy,
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

                    let total_power = PowerLawDistribution::compute_total_power(
                        total_power_density,
                        acceleration_volume,
                    );

                    let heating_scale = PowerLawDistribution::compute_heating_scale(
                        total_power,
                        self.config.power_law_delta,
                        initial_pitch_angle_cosine,
                        lower_cutoff_energy,
                    );

                    let stopping_ionized_column_depth =
                        PowerLawDistribution::compute_stopping_column_depth(
                            initial_pitch_angle_cosine,
                            lower_cutoff_energy,
                            electron_coulomb_logarithm,
                        );

                    let estimated_thermalization_distance =
                        PowerLawDistribution::estimate_depletion_distance(
                            self.distribution_config.max_stopping_length_traversals,
                            total_hydrogen_density,
                            effective_coulomb_logarithm,
                            electron_coulomb_logarithm,
                            stopping_ionized_column_depth,
                        );

                    if estimated_thermalization_distance
                        < self.config.min_thermalization_distance * U_L
                    {
                        None
                    } else {
                        let distribution_data = PowerLawDistributionData {
                            delta: self.config.power_law_delta,
                            initial_pitch_angle_cosine,
                            total_power,
                            total_power_density,
                            lower_cutoff_energy,
                            acceleration_position,
                            acceleration_volume,
                            propagation_sense,
                            electron_coulomb_logarithm,
                            neutral_hydrogen_coulomb_logarithm,
                            heating_scale,
                            stopping_ionized_column_depth,
                            estimated_thermalization_distance,
                        };
                        Some(PowerLawDistribution::new(
                            self.distribution_config.clone(),
                            distribution_data,
                        ))
                    }
                },
            )
            .collect();

        Ok((distributions, ()))
    }
}

impl SimplePowerLawAccelerationConfig {
    pub const DEFAULT_ACCELERATION_DURATION: feb = 1.0; // [s]
    pub const DEFAULT_PARTICLE_ENERGY_FRACTION: feb = 0.5;
    pub const DEFAULT_POWER_LAW_DELTA: feb = 4.0;
    pub const DEFAULT_IGNORE_REJECTION: bool = false;
    pub const DEFAULT_MIN_TOTAL_POWER_DENSITY: feb = 1e-2; // [erg/(cm^3 s)]
    pub const DEFAULT_MIN_LOWER_CUTOFF_ENERGY: feb = 0.1; // [keV]
    pub const DEFAULT_MIN_THERMALIZATION_DISTANCE: feb = 0.3; // [Mm]
    pub const DEFAULT_MAX_PITCH_ANGLE: feb = 70.0; // [deg]
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
        let min_lower_cutoff_energy = reader
            .get_converted_numerical_param_or_fallback_to_default_with_warning(
                "min_lower_cutoff_energy",
                "min_cutoff_en",
                &|min_cutoff_en| min_cutoff_en,
                Self::DEFAULT_MIN_LOWER_CUTOFF_ENERGY,
            );
        let min_thermalization_distance = reader
            .get_converted_numerical_param_or_fallback_to_default_with_warning(
                "min_thermalization_distance",
                "min_stop_dist",
                &|min_stop_dist: feb| min_stop_dist,
                Self::DEFAULT_MIN_THERMALIZATION_DISTANCE,
            );
        SimplePowerLawAccelerationConfig {
            acceleration_duration,
            particle_energy_fraction,
            power_law_delta,
            min_total_power_density,
            min_lower_cutoff_energy,
            min_thermalization_distance,
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
            self.min_lower_cutoff_energy >= 0.0,
            "Minimum lower cut-off energy must be larger than or equal to zero."
        );
        assert!(
            self.min_thermalization_distance >= 0.0,
            "Minimum stopping distance must be larger than or equal to zero."
        );
        assert!(
            self.max_pitch_angle >= 0.0 && self.max_pitch_angle <= 90.0,
            "Maximum pitch angle must be in the range [0, 90]."
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
            min_lower_cutoff_energy: Self::DEFAULT_MIN_LOWER_CUTOFF_ENERGY,
            min_thermalization_distance: Self::DEFAULT_MIN_THERMALIZATION_DISTANCE,
            max_pitch_angle: Self::DEFAULT_MAX_PITCH_ANGLE,
            initial_cutoff_energy_guess: Self::DEFAULT_INITIAL_CUTOFF_ENERGY_GUESS,
            acceptable_root_finding_error: Self::DEFAULT_ACCEPTABLE_ROOT_FINDING_ERROR,
            max_root_finding_iterations: Self::DEFAULT_MAX_ROOT_FINDING_ITERATIONS,
        }
    }
}
