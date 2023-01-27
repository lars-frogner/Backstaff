//! Simple model for acceleration of non-thermal electron beams described by power-law distributions.

use super::super::PowerLawDistribution;
use crate::{
    constants::{INFINITY, KBOLTZMANN, KEV_TO_ERG, PI},
    ebeam::{
        accelerator::Accelerator, detection::ReconnectionSiteDetector, feb, propagation::Propagator,
    },
    field::CachingScalarFieldProvider3,
    geometry::{Dim3, Idx3, Point3, Vec3},
    grid::{fgr, Grid3},
    interpolation::Interpolator3,
    io::{
        snapshot::{self, fdt, SnapshotParameters},
        Verbosity,
    },
    tracing::{
        field_line::basic::FieldLineTracingSense,
        stepping::{DynStepper3, SteppingSense},
    },
    units::solar::{U_B, U_E, U_EL, U_L3, U_R, U_T},
};
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
    /// Direction(s) to trace the trajectory of the distribution relative to the magnetic
    /// field direction.
    pub tracing_sense: FieldLineTracingSense,
}

// Simple acceleration process producing power-law distributions of non-thermal electrons.
#[derive(Clone, Debug)]
pub struct SimplePowerLawAccelerator {
    config: SimplePowerLawAccelerationConfig,
    pitch_angle_cosine_threshold: feb,
    #[allow(dead_code)]
    electric_field_angle_cosine_threshold: feb,
}

impl SimplePowerLawAccelerator {
    /// How many adjacent grid cells in each direction to include when
    /// computing the average electric field around the acceleration site.
    const ELECTRIC_FIELD_PROBING_SPAN: isize = 0;

    fn determine_total_power_density(
        &self,
        snapshot: &dyn CachingScalarFieldProvider3<fdt>,
        indices: &Idx3<usize>,
    ) -> feb {
        let joule_heating_field = snapshot.cached_scalar_field("qjoule");
        #[allow(clippy::useless_conversion)]
        let joule_heating = feb::from(joule_heating_field.value(indices));
        let joule_heating = feb::max(0.0, joule_heating * U_E / U_T); // [erg/(cm^3 s)]

        self.config.particle_energy_fraction * joule_heating
    }

    fn determine_acceleration_volume(
        &self,
        snapshot: &dyn CachingScalarFieldProvider3<fdt>,
        indices: &Idx3<usize>,
    ) -> feb {
        snapshot.grid().grid_cell_volume(indices) * U_L3 // [cm^3]
    }

    fn determine_acceleration_position(
        snapshot: &dyn CachingScalarFieldProvider3<fdt>,
        indices: &Idx3<usize>,
    ) -> Point3<fgr> {
        snapshot.grid().centers().point(indices)
    }

    fn determine_electric_field_strength_and_direction(
        snapshot: &dyn CachingScalarFieldProvider3<fdt>,
        indices: &Idx3<usize>,
    ) -> Option<(fdt, Vec3<fdt>)> {
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

        if squared_total_electric_vector > fdt::EPSILON {
            let electric_field_strength = fdt::sqrt(squared_total_electric_vector);
            Some((
                electric_field_strength,
                total_electric_vector / electric_field_strength,
            ))
        } else {
            None
        }
    }

    fn determine_magnetic_field_strength_and_direction(
        snapshot: &dyn CachingScalarFieldProvider3<fdt>,
        interpolator: &dyn Interpolator3<fdt>,
        acceleration_position: &Point3<fgr>,
    ) -> (fgr, Vec3<fdt>) {
        let magnetic_field = snapshot.cached_vector_field("b");
        let mut magnetic_field_direction = interpolator
            .interp_vector_field(magnetic_field, acceleration_position)
            .expect_inside();
        let magnetic_field_strength = magnetic_field_direction.normalize_and_get_length();
        (magnetic_field_strength, magnetic_field_direction.cast())
    }

    fn compute_electric_field_angle_cosine(
        &self,
        magnetic_field_direction: &Vec3<fdt>,
        electric_field_direction: &Vec3<fdt>,
    ) -> feb {
        #[allow(clippy::useless_conversion)]
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

    #[allow(clippy::useless_conversion)]
    fn determine_temperature(
        snapshot: &dyn CachingScalarFieldProvider3<fdt>,
        indices: &Idx3<usize>,
    ) -> feb {
        feb::from(snapshot.cached_scalar_field("tg").value(indices))
    }

    #[allow(clippy::useless_conversion)]
    fn determine_electron_density(
        snapshot: &dyn CachingScalarFieldProvider3<fdt>,
        indices: &Idx3<usize>,
    ) -> feb {
        feb::from(snapshot.cached_scalar_field("nel").value(indices)) // [1/cm^3]
    }

    #[allow(clippy::useless_conversion)]
    fn determine_mass_density(
        snapshot: &dyn CachingScalarFieldProvider3<fdt>,
        indices: &Idx3<usize>,
    ) -> feb {
        feb::from(snapshot.cached_scalar_field("r").value(indices)) * U_R // [g/cm^3]
    }

    /// Estimates the lower cut-off energy of the non-thermal distribution by
    /// determining where the power-law intersects the thermal Maxwell-Boltzmann
    /// distribution.
    ///
    /// More precisely, the method determines the cut-off energy `Ec` such that
    /// `ne*P_MB(Ec) = n_acc(Ec)*P_PL(Ec)`, where `n_acc(Ec) = e_acc/(Ec*(2*delta - 1)/(2*delta - 3))`
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
        if total_energy_density < feb::EPSILON {
            return None;
        }
        let beta = KEV_TO_ERG / (KBOLTZMANN * temperature); // [1/keV]
        let thermal_fraction =
            KEV_TO_ERG * electron_density * feb::sqrt(4.0 * feb::powi(beta, 3) / PI)
                / (total_energy_density * (self.config.power_law_delta - 1.5)); // [1/keV^(5/2)]
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
                eprintln!(
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
            / (feb::powi((delta - 0.5) / (delta - 1.0), 2)
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
    /// - `config`: Configuration parameters for the accelerator.
    ///
    /// # Returns
    ///
    /// A new `SimplePowerLawAccelerator`.
    pub fn new(config: SimplePowerLawAccelerationConfig) -> Self {
        config.validate();

        let pitch_angle_cosine_threshold = feb::cos(config.max_pitch_angle.to_radians());
        let electric_field_angle_cosine_threshold =
            feb::cos(config.max_electric_field_angle.to_radians());

        SimplePowerLawAccelerator {
            config,
            pitch_angle_cosine_threshold,
            electric_field_angle_cosine_threshold,
        }
    }
}

impl Accelerator for SimplePowerLawAccelerator {
    type DistributionType = PowerLawDistribution;
    type AccelerationDataCollectionType = ();

    fn generate_propagators_with_distributions<P>(
        &self,
        propagator_config: P::Config,
        snapshot: &mut dyn CachingScalarFieldProvider3<fdt>,
        detector: &dyn ReconnectionSiteDetector,
        interpolator: &dyn Interpolator3<fdt>,
        _stepper: DynStepper3<fdt>,
        verbosity: &Verbosity,
    ) -> io::Result<(Vec<P>, Self::AccelerationDataCollectionType)>
    where
        P: Propagator<Self::DistributionType>,
    {
        let seeder = detector.detect_reconnection_sites(snapshot, verbosity);
        let number_of_locations = seeder.number_of_indices();

        if verbosity.print_messages() {
            println!("Computing total beam powers");
        }
        let progress_bar = verbosity.create_progress_bar(number_of_locations);

        snapshot.cache_scalar_field("qjoule")?;
        let properties: Vec<_> = seeder
            .indices()
            .par_iter()
            .filter_map(|indices| {
                let property = if self.config.inclusion_probability < 1.0
                    && rand::thread_rng().gen::<feb>() >= self.config.inclusion_probability
                {
                    None
                } else {
                    let total_power_density = self.determine_total_power_density(snapshot, indices);
                    if total_power_density < self.config.min_total_power_density {
                        None
                    } else {
                        Some((indices.clone(), total_power_density))
                    }
                };
                progress_bar.inc();
                property
            })
            .collect();
        snapshot.drop_scalar_field("qjoule");

        if verbosity.print_messages() {
            println!("Computing magnetic and electric field directions");
        }
        let number_of_locations = properties.len();
        let progress_bar = verbosity.create_progress_bar(number_of_locations);

        snapshot.cache_vector_field("b")?;
        snapshot.cache_vector_field("e")?;
        let properties: Vec<_> = properties
            .into_par_iter()
            .filter_map(|(indices, total_power_density)| {
                let acceleration_position =
                    Self::determine_acceleration_position(snapshot, &indices);
                let properties =
                    match Self::determine_electric_field_strength_and_direction(snapshot, &indices)
                    {
                        Some((electric_field_strength, electric_field_direction)) => {
                            let (magnetic_field_strength, magnetic_field_direction) =
                                Self::determine_magnetic_field_strength_and_direction(
                                    snapshot,
                                    interpolator,
                                    &acceleration_position,
                                );
                            let electric_field_angle_cosine = self
                                .compute_electric_field_angle_cosine(
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
                                    electric_field_strength as feb,
                                    magnetic_field_strength as feb,
                                ))
                            }
                        }
                        None => None,
                    };
                progress_bar.inc();
                properties
            })
            .collect();

        if verbosity.print_messages() {
            println!("Computing lower cutoff energies and estimating stopping distances");
        }
        let number_of_locations = properties.len();
        let progress_bar = verbosity.create_progress_bar(number_of_locations);

        snapshot.cache_scalar_field("nel")?;
        snapshot.cache_scalar_field("r")?;
        snapshot.cache_scalar_field("tg")?;
        let propagators = properties
            .into_par_iter()
            .filter_map(
                |(
                    indices,
                    total_power_density,
                    acceleration_position,
                    partitioned_power_densities,
                    electric_field_angle_cosine,
                    electric_field_strength,
                    magnetic_field_strength,
                )| {
                    let ambient_electron_density =
                        Self::determine_electron_density(snapshot, &indices);
                    assert!(
                        ambient_electron_density > 0.0,
                        "Electron density must be larger than zero."
                    );

                    let ambient_temperature = Self::determine_temperature(snapshot, &indices);
                    assert!(
                        ambient_temperature > 0.0,
                        "Temperature must be larger than zero."
                    );

                    let ambient_mass_density = Self::determine_mass_density(snapshot, &indices);

                    let lower_cutoff_energy = match self.compute_lower_cutoff_energy(
                        ambient_temperature,
                        ambient_electron_density,
                        total_power_density,
                    ) {
                        Some(energy) => energy,
                        None => return None,
                    };

                    let initial_pitch_angle_cosine = match self.compute_initial_pitch_angle_cosine(
                        ambient_temperature,
                        self.config.power_law_delta,
                        lower_cutoff_energy,
                    ) {
                        Some(initial_pitch_angle_cosine) => initial_pitch_angle_cosine,
                        None => return None,
                    };

                    let acceleration_volume =
                        self.determine_acceleration_volume(snapshot, &indices);

                    let propagators = if ambient_temperature < self.config.min_temperature
                        && ambient_mass_density > self.config.max_mass_density
                    {
                        None
                    } else {
                        let mut propagators = Vec::with_capacity(2);
                        if self.config.tracing_sense == FieldLineTracingSense::opposite()
                            || self.config.tracing_sense == FieldLineTracingSense::Both
                        {
                            if let (Some(backward_power_density), _) = partitioned_power_densities {
                                let backward_power = PowerLawDistribution::compute_total_power(
                                    backward_power_density,
                                    acceleration_volume,
                                );
                                let distribution = PowerLawDistribution {
                                    delta: self.config.power_law_delta,
                                    initial_pitch_angle_cosine,
                                    total_power: backward_power,
                                    total_power_density: backward_power_density,
                                    lower_cutoff_energy,
                                    propagation_sense: SteppingSense::Opposite,
                                    electric_field_angle_cosine,
                                    acceleration_position: acceleration_position.clone(),
                                    acceleration_indices: indices.clone(),
                                    acceleration_volume,
                                    ambient_mass_density,
                                    ambient_electron_density,
                                    ambient_temperature,
                                    ambient_trajectory_aligned_electric_field:
                                        -electric_field_angle_cosine
                                            * electric_field_strength
                                            * (*U_EL),
                                    ambient_magnetic_field_strength: magnetic_field_strength
                                        * (*U_B),
                                };
                                if let Some(propagator) =
                                    P::new(propagator_config.clone(), distribution)
                                {
                                    propagators.push(propagator);
                                }
                            }
                        }
                        if self.config.tracing_sense == FieldLineTracingSense::same()
                            || self.config.tracing_sense == FieldLineTracingSense::Both
                        {
                            if let (_, Some(forward_power_density)) = partitioned_power_densities {
                                let forward_power = PowerLawDistribution::compute_total_power(
                                    forward_power_density,
                                    acceleration_volume,
                                );
                                let distribution = PowerLawDistribution {
                                    delta: self.config.power_law_delta,
                                    initial_pitch_angle_cosine,
                                    total_power: forward_power,
                                    total_power_density: forward_power_density,
                                    lower_cutoff_energy,
                                    propagation_sense: SteppingSense::Same,
                                    electric_field_angle_cosine,
                                    acceleration_position,
                                    acceleration_indices: indices,
                                    acceleration_volume,
                                    ambient_mass_density,
                                    ambient_electron_density,
                                    ambient_temperature,
                                    ambient_trajectory_aligned_electric_field:
                                        electric_field_angle_cosine
                                            * electric_field_strength
                                            * (*U_EL),
                                    ambient_magnetic_field_strength: magnetic_field_strength
                                        * (*U_B),
                                };
                                if let Some(propagator) =
                                    P::new(propagator_config.clone(), distribution)
                                {
                                    propagators.push(propagator);
                                }
                            }
                        }
                        if propagators.is_empty() {
                            None
                        } else {
                            Some(propagators)
                        }
                    };
                    progress_bar.inc();
                    propagators
                },
            )
            .flatten()
            .collect();

        Ok((propagators, ()))
    }
}

impl SimplePowerLawAccelerationConfig {
    pub const DEFAULT_ACCELERATION_DURATION: feb = 1.0; // [s]
    pub const DEFAULT_PARTICLE_ENERGY_FRACTION: feb = 0.2;
    pub const DEFAULT_POWER_LAW_DELTA: feb = 4.0;
    pub const DEFAULT_MIN_TOTAL_POWER_DENSITY: feb = 1e-2; // [erg/s/cm^3]
    pub const DEFAULT_MAX_PITCH_ANGLE: feb = 70.0; // [deg]
    pub const DEFAULT_MAX_ELECTRIC_FIELD_ANGLE: feb = 90.0; // [deg]
    pub const DEFAULT_MIN_TEMPERATURE: feb = 0.0; // [K]
    pub const DEFAULT_MAX_MASS_DENSITY: feb = INFINITY; // [g/cm^3]
    pub const DEFAULT_INCLUSION_PROBABILITY: feb = 1.0;
    pub const DEFAULT_INITIAL_CUTOFF_ENERGY_GUESS: feb = 2.0; // [keV]
    pub const DEFAULT_ACCEPTABLE_ROOT_FINDING_ERROR: feb = 1e-3;
    pub const DEFAULT_MAX_ROOT_FINDING_ITERATIONS: i32 = 100;
    pub const DEFAULT_TRACING_SENSE: FieldLineTracingSense = FieldLineTracingSense::Both;

    /// Creates a set of simple power law accelerator configuration parameters with
    /// values read from the specified parameter file when available, otherwise
    /// falling back to the hardcoded defaults.
    pub fn with_defaults_from_param_file(parameters: &dyn SnapshotParameters) -> Self {
        let acceleration_duration =
            snapshot::get_converted_numerical_param_or_fallback_to_default_with_warning(
                parameters,
                "acceleration_duration",
                "dt",
                &|dt: feb| dt * U_T,
                Self::DEFAULT_ACCELERATION_DURATION,
            );
        let particle_energy_fraction =
            snapshot::get_converted_numerical_param_or_fallback_to_default_with_warning(
                parameters,
                "particle_energy_fraction",
                "qjoule_acc_frac",
                &|qjoule_acc_frac: feb| qjoule_acc_frac,
                Self::DEFAULT_PARTICLE_ENERGY_FRACTION,
            );
        let power_law_delta =
            snapshot::get_converted_numerical_param_or_fallback_to_default_with_warning(
                parameters,
                "power_law_delta",
                "power_law_index",
                &|power_law_index: feb| power_law_index,
                Self::DEFAULT_POWER_LAW_DELTA,
            );
        let min_total_power_density =
            snapshot::get_converted_numerical_param_or_fallback_to_default_with_warning(
                parameters,
                "min_total_power_density",
                "min_beam_en",
                &|min_beam_en: feb| min_beam_en,
                Self::DEFAULT_MIN_TOTAL_POWER_DENSITY,
            );
        SimplePowerLawAccelerationConfig {
            acceleration_duration,
            particle_energy_fraction,
            power_law_delta,
            min_total_power_density,
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
            max_pitch_angle: Self::DEFAULT_MAX_PITCH_ANGLE,
            max_electric_field_angle: Self::DEFAULT_MAX_ELECTRIC_FIELD_ANGLE,
            min_temperature: Self::DEFAULT_MIN_TEMPERATURE,
            max_mass_density: Self::DEFAULT_MAX_MASS_DENSITY,
            inclusion_probability: Self::DEFAULT_INCLUSION_PROBABILITY,
            initial_cutoff_energy_guess: Self::DEFAULT_INITIAL_CUTOFF_ENERGY_GUESS,
            acceptable_root_finding_error: Self::DEFAULT_ACCEPTABLE_ROOT_FINDING_ERROR,
            max_root_finding_iterations: Self::DEFAULT_MAX_ROOT_FINDING_ITERATIONS,
            tracing_sense: Self::DEFAULT_TRACING_SENSE,
        }
    }
}
