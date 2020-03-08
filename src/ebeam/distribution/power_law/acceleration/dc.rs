//! Static electric field (direct current) model for acceleration of non-thermal
//! electron beams described by power-law distributions.

pub mod acceleration_region;

use super::super::super::super::accelerator::Accelerator;
use super::super::super::super::detection::ReconnectionSiteDetector;
use super::super::super::super::{feb, AccelerationDataCollection};
use super::super::{PowerLawDistribution, PowerLawDistributionConfig, PowerLawDistributionData};
use crate::constants::{KEV_TO_ERG, Q_ELECTRON};
use crate::geometry::{Idx3, Point3, Vec3};
use crate::grid::Grid3;
use crate::interpolation::Interpolator3;
use crate::io::snapshot::{fdt, SnapshotCacher3, SnapshotReader3};
use crate::io::Verbose;
use crate::plasma::ionization;
use crate::tracing::field_line::{FieldLineSet3, FieldLineSetProperties3, FieldLineTracer3};
use crate::tracing::stepping::{StepperFactory3, SteppingSense};
use crate::units::solar::{U_E, U_L, U_L3, U_R, U_T};
use acceleration_region::AccelerationRegionTracer;
use rayon::prelude::*;
use std::io;

/// Represents a set of acceleration regions in the DC power-law acceleration model
pub type DCAccelerationRegions = FieldLineSet3;

/// Configuration parameters for the DC power-law acceleration model.
#[derive(Clone, Debug)]
pub struct DCPowerLawAccelerationConfig {
    /// Duration of the acceleration events [s].
    pub acceleration_duration: feb,
    /// Fraction of the released reconnection energy going into acceleration of electrons.
    pub particle_energy_fraction: feb,
    /// Exponent of the inverse power-law describing the non-thermal electron distribution.
    pub power_law_delta: feb,
    /// Distributions with total power densities smaller than this value
    /// are discarded [erg/(cm^3 s)].
    pub min_total_power_density: feb,
    /// Distributions with an estimated depletion distance smaller than this value
    /// are discarded [Mm].
    pub min_depletion_distance: feb,
}

// Static electric field (direct current) acceleration process producing
// power-law distributions of non-thermal electrons.
#[derive(Clone, Debug)]
pub struct DCPowerLawAccelerator {
    distribution_config: PowerLawDistributionConfig,
    config: DCPowerLawAccelerationConfig,
    acceleration_region_tracer: AccelerationRegionTracer,
}

impl DCPowerLawAccelerator {
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

    fn compute_lower_cutoff_energy(
        average_parallel_electric_field_strength: feb,
        acceleration_distance: feb,
    ) -> feb {
        average_parallel_electric_field_strength * acceleration_distance * Q_ELECTRON / KEV_TO_ERG
    }

    fn compute_initial_pitch_angle_cosine(average_electric_magnetic_angle_cosine: feb) -> feb {
        // Electrons propagate in the opposite direction of the electric field.
        -average_electric_magnetic_angle_cosine
    }

    fn find_propagation_sense(pitch_angle_cosine: feb) -> SteppingSense {
        if pitch_angle_cosine >= 0.0 {
            SteppingSense::Same
        } else {
            SteppingSense::Opposite
        }
    }
}

impl DCPowerLawAccelerator {
    /// Creates a new DC power-law accelerator.
    ///
    /// # Parameters
    ///
    /// - `distribution_config`: Configuration parameters for the distribution.
    /// - `config`: Configuration parameters for the accelerator.
    /// - `acceleration_region_tracer`: Tracer for acceleration regions.
    ///
    /// # Returns
    ///
    /// A new `DCPowerLawAccelerator`.
    pub fn new(
        distribution_config: PowerLawDistributionConfig,
        config: DCPowerLawAccelerationConfig,
        acceleration_region_tracer: AccelerationRegionTracer,
    ) -> Self {
        distribution_config.validate();
        config.validate();
        DCPowerLawAccelerator {
            distribution_config,
            config,
            acceleration_region_tracer,
        }
    }
}

impl Accelerator for DCPowerLawAccelerator {
    type DistributionType = PowerLawDistribution;
    type AccelerationDataCollectionType = DCAccelerationRegions;

    fn generate_distributions<G, D, I, StF>(
        &self,
        snapshot: &mut SnapshotCacher3<G>,
        detector: D,
        interpolator: &I,
        stepper_factory: &StF,
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
            println!("Extracting reconnection site properties");
        }
        snapshot.cache_scalar_field("qjoule")?;
        snapshot.cache_scalar_field("nel")?;
        snapshot.cache_scalar_field("r")?;
        snapshot.cache_scalar_field("tg")?;
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
            println!("Tracing acceleration regions");
        }
        snapshot.cache_vector_field("b")?;
        snapshot.cache_vector_field("e")?;
        let properties: Vec<_> = properties
            .into_par_iter()
            .filter_map(|(indices, total_power_density)| {
                let start_position =
                    Point3::from(&snapshot.reader().grid().centers().point(&indices));
                if let Some(acceleration_region_data) = self.acceleration_region_tracer.trace(
                    "",
                    snapshot,
                    interpolator,
                    stepper_factory.produce(),
                    &start_position,
                ) {
                    Some((acceleration_region_data, indices, total_power_density))
                } else {
                    None
                }
            })
            .collect();
        snapshot.drop_vector_field("e");

        if verbose.is_yes() {
            println!("Computing lower cutoff energies and estimating depletion distances");
        }
        let (acceleration_region_properties, distributions): (FieldLineSetProperties3, Vec<_>) =
            properties
                .into_par_iter()
                .filter_map(|(acceleration_region_data, indices, total_power_density)| {
                    let electron_density = Self::determine_electron_density(snapshot, &indices);
                    assert!(
                        electron_density > 0.0,
                        "Electron density must be larger than zero."
                    );

                    let temperature = Self::determine_temperature(snapshot, &indices);
                    assert!(temperature > 0.0, "Temperature must be larger than zero.");

                    let lower_cutoff_energy = Self::compute_lower_cutoff_energy(
                        acceleration_region_data.average_parallel_electric_field_strength(),
                        acceleration_region_data.total_length() * U_L,
                    );
                    let mean_energy = PowerLawDistribution::compute_mean_energy(
                        self.config.power_law_delta,
                        lower_cutoff_energy,
                    );

                    let initial_pitch_angle_cosine = Self::compute_initial_pitch_angle_cosine(
                        acceleration_region_data.average_electric_magnetic_angle_cosine(),
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

                    let estimated_depletion_distance =
                        PowerLawDistribution::estimate_depletion_distance(
                            self.config.power_law_delta,
                            self.distribution_config.min_residual_factor,
                            total_hydrogen_density,
                            effective_coulomb_logarithm,
                            electron_coulomb_logarithm,
                            stopping_ionized_column_depth,
                        );

                    if estimated_depletion_distance < self.config.min_depletion_distance * U_L {
                        None
                    } else {
                        let acceleration_position = acceleration_region_data.exit_position();
                        let propagation_sense = Self::find_propagation_sense(
                            acceleration_region_data.average_parallel_electric_field_strength(),
                        );

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
                            estimated_depletion_distance,
                            electric_field_angle_cosine: unimplemented!(),
                        };
                        Some((
                            acceleration_region_data,
                            PowerLawDistribution::new(
                                self.distribution_config.clone(),
                                distribution_data,
                            ),
                        ))
                    }
                })
                .unzip();

        let mut acceleration_regions = DCAccelerationRegions::new(
            Vec3::from(snapshot.reader().grid().lower_bounds()),
            Vec3::from(snapshot.reader().grid().upper_bounds()),
            acceleration_region_properties,
            verbose,
        );

        for name in self.acceleration_region_tracer.extra_varying_scalar_names() {
            acceleration_regions.extract_varying_scalars(
                snapshot
                    .obtain_scalar_field(name)
                    .unwrap_or_else(|err| panic!("Could not read {} from snapshot: {}", name, err)),
                interpolator,
            );
            if !["r", "bx", "by", "bz", "tg", "nel"].contains(&name.as_str()) {
                snapshot.drop_scalar_field(name);
            }
        }
        for name in self.acceleration_region_tracer.extra_varying_vector_names() {
            acceleration_regions.extract_varying_vectors(
                snapshot
                    .obtain_vector_field(name)
                    .unwrap_or_else(|err| panic!("Could not read {} from snapshot: {}", name, err)),
                interpolator,
            );
            if name != "b" {
                snapshot.drop_vector_field(name);
            }
        }

        Ok((distributions, acceleration_regions))
    }
}

impl AccelerationDataCollection for DCAccelerationRegions {
    fn write<W: io::Write>(&self, format_hint: &str, writer: &mut W) -> io::Result<()> {
        match format_hint {
            "pickle" => self.write_as_combined_pickles(writer),
            "fl" => self.write_as_custom_binary(writer),
            _ => self.write_as_custom_binary(writer),
        }
    }

    fn write_into<W: io::Write>(self, format_hint: &str, writer: &mut W) -> io::Result<()> {
        match format_hint {
            "pickle" => self.write_as_combined_pickles(writer),
            "fl" => self.write_into_custom_binary(writer),
            _ => self.write_into_custom_binary(writer),
        }
    }
}

impl DCPowerLawAccelerationConfig {
    pub const DEFAULT_ACCELERATION_DURATION: feb = 1.0; // [s]
    pub const DEFAULT_PARTICLE_ENERGY_FRACTION: feb = 0.2;
    pub const DEFAULT_POWER_LAW_DELTA: feb = 4.0;
    pub const DEFAULT_MIN_TOTAL_POWER_DENSITY: feb = 1e-2; // [erg/(cm^3 s)]
    pub const DEFAULT_MIN_DEPLETION_DISTANCE: feb = 0.5; // [Mm]

    /// Creates a set of DC power law accelerator configuration parameters with
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
        DCPowerLawAccelerationConfig {
            acceleration_duration,
            particle_energy_fraction,
            power_law_delta,
            min_total_power_density,
            min_depletion_distance,
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
    }
}

impl Default for DCPowerLawAccelerationConfig {
    fn default() -> Self {
        DCPowerLawAccelerationConfig {
            acceleration_duration: Self::DEFAULT_ACCELERATION_DURATION,
            particle_energy_fraction: Self::DEFAULT_PARTICLE_ENERGY_FRACTION,
            power_law_delta: Self::DEFAULT_POWER_LAW_DELTA,
            min_total_power_density: Self::DEFAULT_MIN_TOTAL_POWER_DENSITY,
            min_depletion_distance: Self::DEFAULT_MIN_DEPLETION_DISTANCE,
        }
    }
}
