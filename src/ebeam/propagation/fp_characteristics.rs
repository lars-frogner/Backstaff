//! Propagation of a power-law electron distribution by numerical
//! computation of the characteristics of the non-diffusive
//! Fokker-Planck equation.
//!
//! # Note
//! This module operates internally with energies given in erg,
//! not keV.

mod atmosphere;
mod transport;

pub use transport::AnalyticalTransporterConfig;

use self::{
    atmosphere::{CoulombLogarithm, HybridCoulombLogarithm},
    transport::{TransportResult, Transporter},
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
    geometry::{Point3, Vec3},
    grid::{self, Grid3},
    interpolation::Interpolator3,
    io::snapshot::{self, fdt, SnapshotParameters},
    plasma::ionization,
    tracing::ftr,
    units::solar::{U_B, U_EL, U_L, U_L3, U_R},
};
use ndarray::prelude::*;
use std::mem;

/// Configuration parameters for the characteristics propagator.
#[derive(Clone, Debug)]
pub struct CharacteristicsPropagatorConfig {
    pub analytical_transporter_config: AnalyticalTransporterConfig,
    pub n_energies: usize,
    pub min_energy_relative_to_cutoff: feb,
    pub max_energy_relative_to_cutoff: feb,
    pub min_steps_to_initial_thermalization: usize,
    pub max_steps_to_initial_thermalization: usize,
    pub include_return_current: bool,
    pub include_magnetic_mirroring: bool,
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
    energies: Vec<feb>,
    log10_energies: Vec<feb>,
    initial_energies: Vec<feb>,
    pitch_angle_cosines: Vec<feb>,
    electron_numbers_per_dist: Vec<feb>,
    delta_log10_energy: feb,
    stepped_energies: Vec<feb>,
    log10_stepped_energies: Vec<feb>,
    stepped_pitch_angle_cosines: Vec<feb>,
    stepped_electron_numbers_per_dist: Vec<feb>,
    resampled_initial_energies: Vec<feb>,
}

impl CharacteristicsPropagator {
    fn n_substeps(&self, col_depth_increase: feb) -> usize {
        let lower_cutoff_energy = self.distribution.lower_cutoff_energy * KEV_TO_ERG;

        if self.energies[0] < lower_cutoff_energy {
            let n_steps_to_thermalization = self.transporter.n_steps_to_thermalization(
                lower_cutoff_energy,
                self.distribution.initial_pitch_angle_cosine,
                col_depth_increase,
            );
            usize::min(
                self.config.max_steps_to_initial_thermalization,
                feb::ceil(
                    self.config.min_steps_to_initial_thermalization as f64
                        / n_steps_to_thermalization,
                ) as usize,
            )
        } else {
            1
        }
    }

    fn advance_distributions(
        &mut self,
        current_hybrid_coulomb_log: HybridCoulombLogarithm,
        current_total_hydrogen_density: feb,
        current_electric_field_strength: feb,
        current_magnetic_field_strength: feb,
        col_depth_increase: feb,
    ) -> (feb, DepletionStatus) {
        let mut first_valid_stepped_idx = 0;

        for (idx, ((&energy, &pitch_angle_cos), &electron_number_per_dist)) in self
            .energies
            .iter()
            .zip(self.pitch_angle_cosines.iter())
            .zip(self.electron_numbers_per_dist.iter())
            .enumerate()
            .rev()
        {
            match self.transporter.advance_quantities(
                energy,
                pitch_angle_cos,
                electron_number_per_dist,
                col_depth_increase,
            ) {
                TransportResult::NewValues((
                    new_energy,
                    new_pitch_angle_cos,
                    new_electron_number_per_dist,
                )) => {
                    self.stepped_energies[idx] = new_energy;
                    self.stepped_pitch_angle_cosines[idx] = new_pitch_angle_cos;
                    self.stepped_electron_numbers_per_dist[idx] = new_electron_number_per_dist;
                }
                TransportResult::Thermalized => {
                    first_valid_stepped_idx = idx + 1;
                    break;
                }
            }
        }

        let all_thermalized_stepped = || ..first_valid_stepped_idx;
        let all_valid_stepped = || first_valid_stepped_idx..;

        let valid_stepped_energies = &self.stepped_energies[all_valid_stepped()];
        let valid_stepped_pitch_angle_cosines =
            &self.stepped_pitch_angle_cosines[all_valid_stepped()];
        let valid_stepped_electron_numbers_per_dist =
            &self.stepped_electron_numbers_per_dist[all_valid_stepped()];

        let log10_valid_stepped_energies = &mut self.log10_stepped_energies[all_valid_stepped()];
        valid_stepped_energies
            .iter()
            .zip(log10_valid_stepped_energies.iter_mut())
            .for_each(|(&energy, log10_energy)| {
                *log10_energy = feb::log10(energy);
            });

        let mut deposited_power_per_dist = 0.0;

        if first_valid_stepped_idx > 0 {
            // Add deposited power of all thermalized electrons
            deposited_power_per_dist += self.transporter.compute_deposited_power_density(
                &self.energies[all_thermalized_stepped()],
                &self.initial_energies[all_thermalized_stepped()],
                &self.pitch_angle_cosines[all_thermalized_stepped()],
                &self.electron_numbers_per_dist[all_thermalized_stepped()],
            );

            if first_valid_stepped_idx >= self.config.n_energies - 1 {
                return (deposited_power_per_dist, DepletionStatus::Depleted);
            }
        }

        let first_valid_idx = self
            .energies
            .iter()
            .enumerate()
            .find_map(|(idx, &energy)| {
                if energy > valid_stepped_energies[0] {
                    Some(idx)
                } else {
                    None
                }
            })
            .unwrap();

        let all_valid = || first_valid_idx..;
        let all_valid_downshifted = || ..(self.config.n_energies - first_valid_idx);

        Self::interpolate_to_grid(
            &self.distribution,
            &self.log10_energies[all_valid()],
            log10_valid_stepped_energies,
            valid_stepped_pitch_angle_cosines,
            valid_stepped_electron_numbers_per_dist,
            &self.initial_energies[all_valid_stepped()],
            &mut self.pitch_angle_cosines[all_valid_downshifted()],
            &mut self.electron_numbers_per_dist[all_valid_downshifted()],
            &mut self.resampled_initial_energies[all_valid_downshifted()],
        );
        mem::swap(
            &mut self.initial_energies,
            &mut self.resampled_initial_energies,
        );

        if first_valid_idx > 0 {
            self.shift_energies_down(first_valid_idx);
            self.fill_in_initial_conditions_after_shift(first_valid_idx);
        }

        self.transporter.update_conditions(
            current_hybrid_coulomb_log,
            current_total_hydrogen_density,
            current_electric_field_strength,
            current_magnetic_field_strength,
            col_depth_increase,
        );

        deposited_power_per_dist += self.transporter.compute_deposited_power_density(
            &self.energies,
            &self.initial_energies,
            &self.pitch_angle_cosines,
            &self.electron_numbers_per_dist,
        );

        (deposited_power_per_dist, DepletionStatus::Undepleted)
    }

    pub fn interpolate_to_grid(
        distribution: &PowerLawDistribution,
        log10_energies: &[feb],
        log10_stepped_energies: &[feb],
        stepped_pitch_angle_cosines: &[feb],
        stepped_electron_numbers_per_dist: &[feb],
        stepped_initial_energies: &[feb],
        pitch_angle_cosines: &mut [feb],
        electron_numbers_per_dist: &mut [feb],
        initial_energies: &mut [feb],
    ) {
        let n_data = log10_stepped_energies.len();
        assert!(n_data > 1);
        assert_eq!(n_data, stepped_pitch_angle_cosines.len());
        assert_eq!(n_data, stepped_electron_numbers_per_dist.len());
        assert_eq!(n_data, stepped_initial_energies.len());
        assert_eq!(pitch_angle_cosines.len(), log10_energies.len());
        assert_eq!(initial_energies.len(), log10_energies.len());

        fn lerp(x_data: &[feb], f_data: &[feb], idx: usize, x: feb) -> feb {
            f_data[idx]
                + (x - x_data[idx]) * (f_data[idx + 1] - f_data[idx])
                    / (x_data[idx + 1] - x_data[idx])
        }

        log10_energies
            .iter()
            .zip(pitch_angle_cosines.iter_mut())
            .zip(electron_numbers_per_dist.iter_mut())
            .zip(initial_energies.iter_mut())
            .for_each(
                |(
                    ((&log10_energy, pitch_angle_cosine), electron_number_per_dist),
                    initial_energy,
                )| {
                    // Extrapolate if out of bounds
                    let idx = usize::min(
                        n_data - 2,
                        grid::search_idx_of_coord(log10_stepped_energies, log10_energy)
                            .unwrap_or(0),
                    );

                    *pitch_angle_cosine = feb::min(
                        lerp(
                            log10_stepped_energies,
                            stepped_pitch_angle_cosines,
                            idx,
                            log10_energy,
                        ),
                        distribution.initial_pitch_angle_cosine,
                    );

                    *electron_number_per_dist = feb::max(
                        lerp(
                            log10_stepped_energies,
                            stepped_electron_numbers_per_dist,
                            idx,
                            log10_energy,
                        ),
                        0.0,
                    );

                    *initial_energy = lerp(
                        log10_stepped_energies,
                        stepped_initial_energies,
                        idx,
                        log10_energy,
                    );
                },
            );
    }

    fn shift_energies_down(&mut self, shift: usize) {
        assert!(shift > 0);
        assert!(shift < self.config.n_energies);

        self.log10_energies.rotate_left(shift);
        self.energies.rotate_left(shift);
    }

    fn fill_in_initial_conditions_after_shift(&mut self, shift: usize) {
        assert!(shift > 0);
        assert!(shift < self.config.n_energies);

        let first_idx_to_fill = self.config.n_energies - shift;
        let log10_upper_energy_before_shift = self.log10_energies[first_idx_to_fill - 1];

        self.log10_energies[first_idx_to_fill..]
            .iter_mut()
            .zip(self.energies[first_idx_to_fill..].iter_mut())
            .zip(self.initial_energies[first_idx_to_fill..].iter_mut())
            .zip(self.pitch_angle_cosines[first_idx_to_fill..].iter_mut())
            .zip(self.electron_numbers_per_dist[first_idx_to_fill..].iter_mut())
            .enumerate()
            .for_each(
                |(
                    idx,
                    (
                        (((log10_energy, energy), initial_energy), pitch_angle_cosine),
                        electron_number_per_dist,
                    ),
                )| {
                    *log10_energy = log10_upper_energy_before_shift
                        + ((idx + 1) as feb) * self.delta_log10_energy;

                    *energy = feb::powf(10.0, *log10_energy);

                    *initial_energy = self
                        .transporter
                        .energy_without_loss_to_electric_field(*energy);

                    *pitch_angle_cosine = self.transporter.high_energy_pitch_angle_cos();

                    *electron_number_per_dist =
                        PowerLawDistribution::evaluate_electron_number_per_dist(
                            self.distribution.total_power,
                            self.distribution.lower_cutoff_energy * KEV_TO_ERG,
                            self.distribution.delta,
                            *initial_energy,
                        );
                },
            );
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
            distribution.ambient_electron_density,
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
            let electric_field_strength = if config.include_return_current {
                distribution.ambient_electric_field_strength
            } else {
                0.0
            };

            let magnetic_field_strength = if config.include_magnetic_mirroring {
                distribution.ambient_magnetic_field_strength
            } else {
                0.0
            };

            let transporter = Transporter::new(
                config.analytical_transporter_config.clone(),
                distribution.initial_pitch_angle_cosine,
                hybrid_coulomb_log,
                total_hydrogen_density,
                electric_field_strength,
                magnetic_field_strength,
            );

            let min_energy = config.min_energy_relative_to_cutoff
                * distribution.lower_cutoff_energy
                * KEV_TO_ERG;
            let max_energy = config.max_energy_relative_to_cutoff
                * distribution.lower_cutoff_energy
                * KEV_TO_ERG;

            let log10_min_energy = feb::log10(min_energy);
            let log10_max_energy = feb::log10(max_energy);

            let delta_log10_energy =
                (log10_max_energy - log10_min_energy) / ((config.n_energies - 1) as feb);

            let log10_energies: Vec<_> = (0..config.n_energies)
                .map(|i| log10_min_energy + (i as feb) * delta_log10_energy)
                .collect();

            let energies: Vec<_> = log10_energies
                .iter()
                .map(|&log10_energy| feb::powf(10.0, log10_energy))
                .collect();

            let lower_cutoff_energy = distribution.lower_cutoff_energy * KEV_TO_ERG;
            let electron_numbers_per_dist = energies
                .iter()
                .map(|&energy| {
                    if energy >= lower_cutoff_energy {
                        PowerLawDistribution::evaluate_electron_number_per_dist(
                            distribution.total_power,
                            lower_cutoff_energy,
                            distribution.delta,
                            energy,
                        )
                    } else {
                        0.0
                    }
                })
                .collect();

            let pitch_angle_cosines =
                vec![distribution.initial_pitch_angle_cosine; config.n_energies];

            let initial_energies = energies.clone();

            let stepped_energies = vec![0.0; config.n_energies];
            let log10_stepped_energies = vec![0.0; config.n_energies];
            let stepped_pitch_angle_cosines = vec![0.0; config.n_energies];
            let stepped_electron_numbers_per_dist = vec![0.0; config.n_energies];
            let resampled_initial_energies = vec![0.0; config.n_energies];

            Some(Self {
                config,
                distribution,
                transporter,
                coulomb_log,
                energies,
                log10_energies,
                initial_energies,
                pitch_angle_cosines,
                electron_numbers_per_dist,
                delta_log10_energy,
                stepped_energies,
                log10_stepped_energies,
                stepped_pitch_angle_cosines,
                stepped_electron_numbers_per_dist,
                resampled_initial_energies,
            })
        } else {
            None
        }
    }

    fn distribution(&self) -> &PowerLawDistribution {
        &self.distribution
    }

    fn into_distribution(self) -> PowerLawDistribution {
        let Self { distribution, .. } = self;
        distribution
    }

    fn max_propagation_distance(&self) -> ftr {
        self.config.max_propagation_distance
    }

    fn propagate(
        &mut self,
        snapshot: &dyn CachingScalarFieldProvider3<fdt>,
        _acceleration_map: &Array3<bool>,
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

        #[allow(clippy::useless_conversion)]
        let electric_field_strength = if self.config.include_return_current {
            let electric_field = snapshot.cached_vector_field("e");
            feb::from(
                interpolator
                    .interp_vector_field_known_cell(
                        electric_field,
                        &Point3::from(&deposition_position),
                        &deposition_indices,
                    )
                    .length(),
            ) * (*U_EL)
        } else {
            0.0
        };

        #[allow(clippy::useless_conversion)]
        let magnetic_field_strength = if self.config.include_magnetic_mirroring {
            let magnetic_field = snapshot.cached_vector_field("b");
            feb::from(
                interpolator
                    .interp_vector_field_known_cell(
                        magnetic_field,
                        &Point3::from(&deposition_position),
                        &deposition_indices,
                    )
                    .length(),
            ) * (*U_B)
        } else {
            0.0
        };

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

        let mut deposited_power_per_dist = 0.0;
        let mut depletion_status = DepletionStatus::Undepleted;

        let n_substeps = self.n_substeps(col_depth_increase);
        let substep_col_depth_increase = col_depth_increase / (n_substeps as feb);
        for _ in 0..n_substeps {
            (deposited_power_per_dist, depletion_status) = self.advance_distributions(
                hybrid_coulomb_log.clone(),
                total_hydrogen_density,
                electric_field_strength,
                magnetic_field_strength,
                substep_col_depth_increase,
            );
            if depletion_status == DepletionStatus::Depleted {
                break;
            }
        }

        let deposited_power = deposited_power_per_dist * step_length;
        let volume = snapshot.grid().grid_cell_volume(&deposition_indices) * U_L3;
        let deposited_power_density = deposited_power / volume;

        let depletion_status = if (self.config.continue_depleted_beams
            || deposited_power / step_length >= self.config.min_deposited_power_per_distance)
            && depletion_status == DepletionStatus::Undepleted
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
    pub const DEFAULT_N_ENERGIES: usize = 40;
    pub const DEFAULT_MIN_ENERGY_RELATIVE_TO_CUTOFF: feb = 0.05;
    pub const DEFAULT_MAX_ENERGY_RELATIVE_TO_CUTOFF: feb = 120.0;
    pub const DEFAULT_MIN_STEPS_TO_INITIAL_THERMALIZATION: usize = 2;
    pub const DEFAULT_MAX_STEPS_TO_INITIAL_THERMALIZATION: usize = 10;
    pub const DEFAULT_INCLUDE_RETURN_CURRENT: bool = false;
    pub const DEFAULT_INCLUDE_MAGNETIC_MIRRORING: bool = false;
    pub const DEFAULT_MIN_DEPLETION_DISTANCE: feb = 0.5; // [Mm]
    pub const DEFAULT_MIN_RESIDUAL_FACTOR: feb = 1e-5;
    pub const DEFAULT_MIN_DEPOSITED_POWER_PER_DISTANCE: feb = 1e5; // [erg/s/cm]
    pub const DEFAULT_MAX_PROPAGATION_DISTANCE: ftr = 100.0; // [Mm]
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
        CharacteristicsPropagatorConfig {
            analytical_transporter_config: AnalyticalTransporterConfig::default(),
            n_energies: Self::DEFAULT_N_ENERGIES,
            min_energy_relative_to_cutoff: Self::DEFAULT_MIN_ENERGY_RELATIVE_TO_CUTOFF,
            max_energy_relative_to_cutoff: Self::DEFAULT_MAX_ENERGY_RELATIVE_TO_CUTOFF,
            min_steps_to_initial_thermalization: Self::DEFAULT_MIN_STEPS_TO_INITIAL_THERMALIZATION,
            max_steps_to_initial_thermalization: Self::DEFAULT_MAX_STEPS_TO_INITIAL_THERMALIZATION,
            include_return_current: Self::DEFAULT_INCLUDE_RETURN_CURRENT,
            include_magnetic_mirroring: Self::DEFAULT_INCLUDE_MAGNETIC_MIRRORING,
            min_depletion_distance,
            min_residual_factor,
            min_deposited_power_per_distance,
            max_propagation_distance,
            continue_depleted_beams: Self::DEFAULT_CONTINUE_DEPLETED_BEAMS,
        }
    }

    /// Panics if any of the configuration parameter values are invalid.
    pub fn validate(&self) {
        assert!(
            self.n_energies > 1,
            "Number of energies must be larger than one"
        );
        assert!(
            self.min_energy_relative_to_cutoff > 0.0,
            "Minimum energy must be larger than zero"
        );
        assert!(
            self.max_energy_relative_to_cutoff > self.min_energy_relative_to_cutoff,
            "Maximum energy must be higher than minimum energy"
        );
        assert!(
            self.min_steps_to_initial_thermalization > 0,
            "Minimum number of steps to initial thermalization must be larger than zero"
        );
        assert!(
            self.max_steps_to_initial_thermalization >= self.min_steps_to_initial_thermalization,
            "Maximum number of substeps to initial thermalization cannot be smaller than minimum number"
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
    }
}

impl Default for CharacteristicsPropagatorConfig {
    fn default() -> Self {
        CharacteristicsPropagatorConfig {
            analytical_transporter_config: AnalyticalTransporterConfig::default(),
            n_energies: Self::DEFAULT_N_ENERGIES,
            min_energy_relative_to_cutoff: Self::DEFAULT_MIN_ENERGY_RELATIVE_TO_CUTOFF,
            max_energy_relative_to_cutoff: Self::DEFAULT_MAX_ENERGY_RELATIVE_TO_CUTOFF,
            min_steps_to_initial_thermalization: Self::DEFAULT_MIN_STEPS_TO_INITIAL_THERMALIZATION,
            max_steps_to_initial_thermalization: Self::DEFAULT_MAX_STEPS_TO_INITIAL_THERMALIZATION,
            include_return_current: Self::DEFAULT_INCLUDE_RETURN_CURRENT,
            include_magnetic_mirroring: Self::DEFAULT_INCLUDE_MAGNETIC_MIRRORING,
            min_depletion_distance: Self::DEFAULT_MIN_DEPLETION_DISTANCE,
            min_residual_factor: Self::DEFAULT_MIN_RESIDUAL_FACTOR,
            min_deposited_power_per_distance: Self::DEFAULT_MIN_DEPOSITED_POWER_PER_DISTANCE,
            max_propagation_distance: Self::DEFAULT_MAX_PROPAGATION_DISTANCE,
            continue_depleted_beams: Self::DEFAULT_CONTINUE_DEPLETED_BEAMS,
        }
    }
}
