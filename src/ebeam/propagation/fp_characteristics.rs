//! Propagation of a power-law electron distribution by numerical
//! computation of the characteristics of the non-diffusive
//! Fokker-Planck equation.
//!
//! # Note
//! This module operates internally with energies given in erg,
//! not keV.

mod atmosphere;
mod transport;

use self::{
    atmosphere::{CoulombLogarithm, HybridCoulombLogarithm},
    transport::{AnalyticalTransporterConfig, Transporter},
};
use super::analytical::AnalyticalPropagator;
use crate::{
    constants::KEV_TO_ERG,
    ebeam::{
        distribution::power_law::PowerLawDistribution,
        feb,
        propagation::{DepletionStatus, PropagationResult, Propagator},
    },
    field::CachingScalarFieldProvider3,
    geometry::{
        Dim3::{X, Y, Z},
        Point3, Vec3,
    },
    grid::Grid3,
    interpolation::Interpolator3,
    io::snapshot::{self, fdt, SnapshotParameters},
    plasma::ionization,
    tracing::ftr,
    units::solar::{U_L, U_L3, U_R},
};
use ndarray::prelude::*;

/// Configuration parameters for the characteristics propagator.
#[derive(Clone, Debug)]
pub struct CharacteristicsPropagatorConfig {
    pub analytical_transporter_config: AnalyticalTransporterConfig,
    pub n_energies: usize,
    pub min_energy_relative_to_cutoff: feb,
    pub max_energy_relative_to_cutoff: feb,
    /// Distributions with an estimated depletion distance smaller than this value
    /// are discarded [Mm].
    pub min_depletion_distance: feb,
    /// Distributions are considered depleted when the residual energy factor has
    /// decreased below this limit, given that the deposited power per distance is
    /// smaller than its lower limit.
    pub min_residual_factor: feb,
    /// Distributions are considered depleted when the deposited power per distance
    /// [erg/s/cm] has decreased below this limit, given that the residual energy factor
    /// is smaller than its lower limit.
    pub min_deposited_power_per_distance: feb,
    /// Maximum distance the distribution can propagate before propagation should be terminated [Mm].
    pub max_propagation_distance: ftr,
    /// Maximum distance outside the initial extended acceleration region the
    /// distribution can propagate before energy deposition starts [Mm].
    pub outside_deposition_threshold: feb,
    /// Whether to keep propagating beams even after they are considered depleted.
    pub continue_depleted_beams: bool,
}

/// A propagator of a power-law electron distribution the computes
/// the characteristics of the non-diffusive Fokker-Planck equation.
#[derive(Clone, Debug)]
pub struct CharacteristicsPropagator {
    config: CharacteristicsPropagatorConfig,
    distribution: PowerLawDistribution,
    transporter: Transporter,
    coulomb_log: CoulombLogarithm,
    energies: Array1<feb>,
    initial_energies: Array1<feb>,
    pitch_angle_cosines: Array1<feb>,
    electron_numbers: Array1<feb>,
}

impl CharacteristicsPropagator {
    fn advance_distributions(
        &mut self,
        current_hybrid_coulomb_log: HybridCoulombLogarithm,
        current_total_hydrogen_density: feb,
        current_log_magnetic_field_col_depth_deriv: feb,
        current_electric_field: feb,
        col_depth_increase: feb,
    ) -> feb {
        todo!()
    }
}

impl Propagator<PowerLawDistribution> for CharacteristicsPropagator {
    type Config = CharacteristicsPropagatorConfig;

    fn new(config: Self::Config, distribution: PowerLawDistribution) -> Option<Self> {
        let mean_energy = PowerLawDistribution::compute_mean_energy(
            distribution.delta,
            distribution.lower_cutoff_energy,
        );

        let coulomb_logarithm_energy = feb::max(
            mean_energy,
            AnalyticalPropagator::MIN_COULOMB_LOG_MEAN_ENERGY,
        ) * KEV_TO_ERG;

        let ionization_fraction = ionization::compute_equilibrium_hydrogen_ionization_fraction(
            distribution.ambient_temperature,
            distribution.ambient_electron_density,
        );

        let total_hydrogen_density =
            AnalyticalPropagator::compute_total_hydrogen_density(distribution.ambient_mass_density);

        let coulomb_log = CoulombLogarithm::new(
            total_hydrogen_density,
            ionization_fraction,
            coulomb_logarithm_energy,
        );
        let hybrid_coulomb_log =
            HybridCoulombLogarithm::new(coulomb_log.clone(), ionization_fraction);

        let heating_scale = AnalyticalPropagator::compute_heating_scale(
            distribution.total_power,
            distribution.delta,
            distribution.initial_pitch_angle_cosine,
            distribution.lower_cutoff_energy,
        );

        let stopping_ionized_column_depth = AnalyticalPropagator::compute_stopping_column_depth(
            distribution.initial_pitch_angle_cosine,
            distribution.lower_cutoff_energy,
            coulomb_log.with_electrons_protons(),
        );

        let estimated_depletion_distance = AnalyticalPropagator::estimate_depletion_distance(
            distribution.delta,
            config.min_residual_factor,
            config.min_deposited_power_per_distance,
            total_hydrogen_density,
            hybrid_coulomb_log.for_energy(),
            coulomb_log.with_electrons_protons(),
            stopping_ionized_column_depth,
            heating_scale,
        );

        if estimated_depletion_distance >= config.min_depletion_distance * U_L {
            let transporter = Transporter::new(
                config.analytical_transporter_config.clone(),
                hybrid_coulomb_log,
                total_hydrogen_density,
                0.0,
                0.0,
            );

            let min_energy = config.min_energy_relative_to_cutoff
                * distribution.lower_cutoff_energy
                * KEV_TO_ERG;
            let max_energy = config.max_energy_relative_to_cutoff
                * distribution.lower_cutoff_energy
                * KEV_TO_ERG;

            let log10_min_energy = feb::log10(min_energy);
            let log10_max_energy = feb::log10(max_energy);

            let log10_energies = Array::linspace(min_energy, max_energy, config.n_energies);
            let energies = log10_energies.map(|&log10_energy| feb::powf(10.0, log10_energy));

            let lower_cutoff_energy = distribution.lower_cutoff_energy * KEV_TO_ERG;
            let electron_numbers = energies.map(|&energy| {
                if energy >= lower_cutoff_energy {
                    PowerLawDistribution::evaluate_electron_number(
                        distribution.total_power,
                        lower_cutoff_energy,
                        distribution.delta,
                        energy,
                    )
                } else {
                    0.0
                }
            });

            let pitch_angle_cosines =
                Array::from_elem(config.n_energies, distribution.initial_pitch_angle_cosine);

            let initial_energies = energies.clone();

            Some(Self {
                config,
                distribution,
                transporter,
                coulomb_log,
                energies,
                initial_energies,
                pitch_angle_cosines,
                electron_numbers,
            })
        } else {
            None
        }
    }

    fn distribution(&self) -> &PowerLawDistribution {
        &self.distribution
    }

    fn into_distribution(self) -> PowerLawDistribution {
        let Self {
            config: _,
            distribution,
            transporter: _,
            coulomb_log: _,
            energies: _,
            initial_energies: _,
            pitch_angle_cosines: _,
            electron_numbers: _,
        } = self;
        distribution
    }

    fn max_propagation_distance(&self) -> ftr {
        self.config.max_propagation_distance
    }

    fn propagate(
        &mut self,
        snapshot: &dyn CachingScalarFieldProvider3<fdt>,
        acceleration_map: &Array3<bool>,
        interpolator: &dyn Interpolator3<fdt>,
        displacement: &Vec3<ftr>,
        new_position: &Point3<ftr>,
    ) -> PropagationResult {
        let mut deposition_position = new_position - displacement * 0.5;

        let deposition_indices = snapshot
            .grid()
            .find_grid_cell(&Point3::from(&deposition_position))
            .unwrap_and_update_position(&mut deposition_position);

        let electron_density_field = snapshot.cached_scalar_field("nel");
        let mass_density_field = snapshot.cached_scalar_field("r");
        let temperature_field = snapshot.cached_scalar_field("tg");

        #[allow(clippy::useless_conversion)]
        let electron_density = feb::from(interpolator.interp_scalar_field_known_cell(
            electron_density_field,
            &Point3::from(&deposition_position),
            &deposition_indices,
        ));

        #[allow(clippy::useless_conversion)]
        let mass_density = feb::from(interpolator.interp_scalar_field_known_cell(
            mass_density_field,
            &Point3::from(&deposition_position),
            &deposition_indices,
        )) * U_R;

        #[allow(clippy::useless_conversion)]
        let temperature = feb::from(interpolator.interp_scalar_field_known_cell(
            temperature_field,
            &Point3::from(&deposition_position),
            &deposition_indices,
        ));

        let total_hydrogen_density =
            AnalyticalPropagator::compute_total_hydrogen_density(mass_density);

        let ionization_fraction = ionization::compute_equilibrium_hydrogen_ionization_fraction(
            temperature,
            electron_density,
        );

        let hybrid_coulomb_log =
            HybridCoulombLogarithm::new(self.coulomb_log.clone(), ionization_fraction);

        let step_length = displacement.length() * U_L; // [cm]
        let col_depth_increase = step_length * total_hydrogen_density;

        let deposited_power = self.advance_distributions(
            hybrid_coulomb_log,
            total_hydrogen_density,
            0.0,
            0.0,
            col_depth_increase,
        );

        let volume = snapshot.grid().grid_cell_volume(&deposition_indices) * U_L3;
        let deposited_power_density = deposited_power / volume;

        let depletion_status = if self.config.continue_depleted_beams
            || deposited_power / step_length >= self.config.min_deposited_power_per_distance
        {
            DepletionStatus::Undepleted
        } else {
            DepletionStatus::Depleted
        };

        PropagationResult {
            deposited_power,
            deposited_power_density,
            deposition_position,
            depletion_status,
        }
    }
}

impl CharacteristicsPropagatorConfig {
    pub const DEFAULT_N_ENERGIES: usize = 200;
    pub const DEFAULT_MIN_ENERGY_RELATIVE_TO_CUTOFF: feb = 1e-1;
    pub const DEFAULT_MAX_ENERGY_RELATIVE_TO_CUTOFF: feb = 1e2;
    pub const DEFAULT_MIN_DEPLETION_DISTANCE: feb = 0.5; // [Mm]
    pub const DEFAULT_MIN_RESIDUAL_FACTOR: feb = 1e-5;
    pub const DEFAULT_MIN_DEPOSITED_POWER_PER_DISTANCE: feb = 1e5; // [erg/s/cm]
    pub const DEFAULT_MAX_PROPAGATION_DISTANCE: ftr = 100.0; // [Mm]
    pub const DEFAULT_OUTSIDE_DEPOSITION_THRESHOLD: feb = 0.1; // [Mm]
    pub const DEFAULT_CONTINUE_DEPLETED_BEAMS: bool = false;

    /// Creates a set of analytical propagator configuration parameters with
    /// values read from the specified parameter file when available, otherwise
    /// falling back to the hardcoded defaults.
    pub fn with_defaults_from_param_file(parameters: &dyn SnapshotParameters) -> Self {
        let min_depletion_distance =
            snapshot::get_converted_numerical_param_or_fallback_to_default_with_warning(
                parameters,
                "min_depletion_distance",
                "min_stop_dist",
                &|min_stop_dist: feb| min_stop_dist,
                Self::DEFAULT_MIN_DEPLETION_DISTANCE,
            );
        let min_residual_factor =
            snapshot::get_converted_numerical_param_or_fallback_to_default_with_warning(
                parameters,
                "min_residual_factor",
                "min_residual",
                &|min_residual: feb| min_residual,
                Self::DEFAULT_MIN_RESIDUAL_FACTOR,
            );
        let min_deposited_power_per_distance =
            snapshot::get_converted_numerical_param_or_fallback_to_default_with_warning(
                parameters,
                "min_deposited_power_per_distance",
                "min_dep_en",
                &|min_dep_en: feb| min_dep_en,
                Self::DEFAULT_MIN_DEPOSITED_POWER_PER_DISTANCE,
            );
        let max_propagation_distance =
            snapshot::get_converted_numerical_param_or_fallback_to_default_with_warning(
                parameters,
                "max_propagation_distance",
                "max_dist",
                &|max_dist: feb| max_dist,
                Self::DEFAULT_MAX_PROPAGATION_DISTANCE,
            );
        let outside_deposition_threshold =
            snapshot::get_converted_numerical_param_or_fallback_to_default_with_warning(
                parameters,
                "outside_deposition_threshold",
                "out_dep_thresh",
                &|out_dep_thresh: feb| out_dep_thresh,
                Self::DEFAULT_OUTSIDE_DEPOSITION_THRESHOLD,
            );
        CharacteristicsPropagatorConfig {
            analytical_transporter_config: AnalyticalTransporterConfig::default(),
            n_energies: Self::DEFAULT_N_ENERGIES,
            min_energy_relative_to_cutoff: Self::DEFAULT_MIN_ENERGY_RELATIVE_TO_CUTOFF,
            max_energy_relative_to_cutoff: Self::DEFAULT_MAX_ENERGY_RELATIVE_TO_CUTOFF,
            min_depletion_distance,
            min_residual_factor,
            min_deposited_power_per_distance,
            max_propagation_distance,
            outside_deposition_threshold,
            continue_depleted_beams: Self::DEFAULT_CONTINUE_DEPLETED_BEAMS,
        }
    }

    /// Panics if any of the configuration parameter values are invalid.
    pub fn validate(&self) {
        assert!(
            self.min_energy_relative_to_cutoff > 0.0,
            "Minimum energy must be larger than zero"
        );
        assert!(
            self.max_energy_relative_to_cutoff > self.min_energy_relative_to_cutoff,
            "Maximum energy must be higher than minimum energy"
        );
        assert!(
            self.min_depletion_distance >= 0.0,
            "Minimum stopping distance must be larger than or equal to zero."
        );
        assert!(
            self.min_residual_factor >= 0.0,
            "Minimum residual factor must be larger than or equal to zero."
        );
        assert!(
            self.min_deposited_power_per_distance >= 0.0,
            "Minimum deposited power per distance must be larger than or equal to zero."
        );
        assert!(
            self.max_propagation_distance >= 0.0,
            "Maximum propagation distance must be larger than or equal to zero."
        );
        assert!(
            self.outside_deposition_threshold >= 0.0,
            "Outside deposition threshold must be larger than or equal to zero."
        );
    }
}

impl Default for CharacteristicsPropagatorConfig {
    fn default() -> Self {
        CharacteristicsPropagatorConfig {
            analytical_transporter_config: AnalyticalTransporterConfig::default(),
            n_energies: Self::DEFAULT_N_ENERGIES,
            min_energy_relative_to_cutoff: Self::DEFAULT_MIN_ENERGY_RELATIVE_TO_CUTOFF,
            max_energy_relative_to_cutoff: Self::DEFAULT_MAX_ENERGY_RELATIVE_TO_CUTOFF,
            min_depletion_distance: Self::DEFAULT_MIN_DEPLETION_DISTANCE,
            min_residual_factor: Self::DEFAULT_MIN_RESIDUAL_FACTOR,
            min_deposited_power_per_distance: Self::DEFAULT_MIN_DEPOSITED_POWER_PER_DISTANCE,
            max_propagation_distance: Self::DEFAULT_MAX_PROPAGATION_DISTANCE,
            outside_deposition_threshold: Self::DEFAULT_OUTSIDE_DEPOSITION_THRESHOLD,
            continue_depleted_beams: Self::DEFAULT_CONTINUE_DEPLETED_BEAMS,
        }
    }
}
