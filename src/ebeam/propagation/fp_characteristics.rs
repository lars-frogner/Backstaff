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
    exit_on_error,
    field::CachingScalarFieldProvider3,
    geometry::{Point3, Vec3},
    grid::{self, Grid3},
    interpolation::Interpolator3,
    io::{
        snapshot::{self, fdt, SnapshotParameters},
        utils::{self as io_utils, AtomicOutputFileMap},
    },
    plasma::ionization,
    tracing::{ftr, stepping::SteppingSense},
    units::solar::{U_B, U_EL, U_L, U_L3, U_R},
};
use ndarray::prelude::*;
use ndarray_npy::NpzWriter;
use std::{
    io, mem,
    path::PathBuf,
    sync::{Arc, Mutex},
};

/// Configuration parameters for the characteristics propagator.
#[derive(Clone, Debug)]
pub struct CharacteristicsPropagatorConfig {
    pub n_energies: usize,
    pub min_energy_relative_to_cutoff: feb,
    pub max_energy_relative_to_cutoff: feb,
    pub max_col_depth_increase: feb,
    pub max_substeps: usize,
    pub n_initial_steps_with_substeps: usize,
    pub n_substeps: usize,
    pub keep_initial_ionization_fraction: bool,
    pub assume_ambient_electrons_all_from_hydrogen: bool,
    pub include_ambient_electric_field: bool,
    pub include_return_current: bool,
    pub include_magnetic_mirroring: bool,
    pub enable_warm_target: bool,
    pub min_depletion_distance: feb,
    pub min_remaining_flux_fraction: feb,
    pub min_residual_factor: feb,
    pub min_deposited_power_per_distance: feb,
    pub max_propagation_distance: ftr,
    pub continue_depleted_beams: bool,
    pub detailed_output_config: Option<DetailedOutputConfig>,
}

/// A propagator of a power-law electron distribution the computes
/// the characteristics of the non-diffusive Fokker-Planck equation.
#[derive(Clone, Debug)]
pub struct CharacteristicsPropagator {
    id: i64,
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
    step_count: usize,
    prev_n_substeps: usize,
    initial_total_electron_flux_over_cross_section: feb,
    distance: feb,
    detailed_output: Option<DetailedOutput>,
}

#[derive(Clone, Debug)]
pub struct DetailedOutputConfig {
    pub detailed_output_dir: PathBuf,
    pub atomic_output_file_map: Arc<Mutex<AtomicOutputFileMap>>,
}

#[derive(Clone, Debug)]
struct DetailedOutput {
    distances: Vec<feb>,
    total_electron_fluxes_over_cross_section: Vec<feb>,
    energies: Array2<feb>,
    pitch_angle_cosines: Array2<feb>,
    electron_numbers_per_dist: Array2<feb>,
}

impl CharacteristicsPropagator {
    fn determine_n_substeps(&mut self, col_depth_increase: feb) -> usize {
        let n_substeps = if self.step_count < self.config.n_initial_steps_with_substeps {
            feb::ceil(col_depth_increase / self.config.max_col_depth_increase) as usize
        } else {
            1
        };

        // Never reduce number of substeps by more than half in one step
        let n_substeps = usize::max(self.prev_n_substeps / 2, n_substeps);

        let n_substeps = usize::min(self.config.max_substeps, n_substeps);

        self.prev_n_substeps = n_substeps;

        n_substeps
    }

    fn advance_distributions(
        &mut self,
        current_hybrid_coulomb_log: HybridCoulombLogarithm,
        current_total_hydrogen_density: feb,
        current_temperature: feb,
        current_ambient_trajectory_aligned_electric_field: feb,
        current_ambient_magnetic_field_strength: feb,
        beam_cross_sectional_area: feb,
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
            deposited_power_per_dist += self.transporter.compute_deposited_power_per_dist(
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
            current_temperature,
            current_ambient_trajectory_aligned_electric_field,
            current_ambient_magnetic_field_strength,
            &self.energies,
            &self.initial_energies,
            &self.pitch_angle_cosines,
            &self.electron_numbers_per_dist,
            beam_cross_sectional_area,
            col_depth_increase,
        );

        deposited_power_per_dist += self.transporter.compute_deposited_power_per_dist(
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

    fn new(config: Self::Config, distribution: PowerLawDistribution, id: i64) -> Option<Self> {
        let mean_energy = PowerLawDistribution::compute_mean_energy(
            distribution.delta,
            distribution.lower_cutoff_energy,
        );

        let coulomb_logarithm_energy = feb::max(
            mean_energy,
            AnalyticalPropagator::MIN_COULOMB_LOG_MEAN_ENERGY,
        ) * KEV_TO_ERG;

        let hydrogen_ionization_fraction =
            ionization::compute_equilibrium_hydrogen_ionization_fraction(
                distribution.ambient_temperature,
                distribution.ambient_electron_density,
            );

        let total_hydrogen_density =
            AnalyticalPropagator::compute_total_hydrogen_density(distribution.ambient_mass_density);

        let electron_to_hydrogen_ratio = if config.assume_ambient_electrons_all_from_hydrogen {
            hydrogen_ionization_fraction
        } else {
            distribution.ambient_electron_density / total_hydrogen_density
        };

        let coulomb_log = CoulombLogarithm::new(
            distribution.ambient_electron_density,
            coulomb_logarithm_energy,
        );
        let hybrid_coulomb_log = HybridCoulombLogarithm::new(
            config.enable_warm_target,
            coulomb_log.clone(),
            distribution.ambient_temperature,
            electron_to_hydrogen_ratio,
            hydrogen_ionization_fraction,
        );

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
            hybrid_coulomb_log.for_energy_cold_target(),
            coulomb_log.with_electrons_protons(),
            stopping_ionized_column_depth,
            heating_scale,
        );

        if estimated_depletion_distance >= config.min_depletion_distance * U_L {
            let ambient_trajectory_aligned_electric_field = if config.include_ambient_electric_field
            {
                distribution.ambient_trajectory_aligned_electric_field
            } else {
                0.0
            };

            let magnetic_field_strength = if config.include_magnetic_mirroring {
                distribution.ambient_magnetic_field_strength
            } else {
                0.0
            };

            let lower_cutoff_energy = distribution.lower_cutoff_energy * KEV_TO_ERG;

            let total_electron_flux_over_cross_section =
                PowerLawDistribution::compute_total_electron_flux_over_cross_section(
                    distribution.total_power,
                    lower_cutoff_energy,
                    distribution.delta,
                );

            let transporter = Transporter::new(
                config.include_ambient_electric_field,
                config.include_return_current,
                config.include_magnetic_mirroring,
                total_electron_flux_over_cross_section,
                distribution.initial_pitch_angle_cosine,
                hybrid_coulomb_log,
                total_hydrogen_density,
                distribution.ambient_temperature,
                ambient_trajectory_aligned_electric_field,
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

            let electron_numbers_per_dist: Vec<_> = energies
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

            let detailed_output = if config.detailed_output_config.is_some() {
                Some(DetailedOutput::new(
                    total_electron_flux_over_cross_section,
                    energies.clone(),
                    pitch_angle_cosines.clone(),
                    electron_numbers_per_dist.clone(),
                ))
            } else {
                None
            };

            Some(Self {
                id,
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
                initial_total_electron_flux_over_cross_section:
                    total_electron_flux_over_cross_section,
                distance: 0.0,
                step_count: 0,
                prev_n_substeps: 0,
                detailed_output,
            })
        } else {
            None
        }
    }

    fn id(&self) -> i64 {
        self.id
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

        let mut trajectory_aligned_electric_field = 0.0;
        let mut magnetic_field_strength = 0.0;

        #[allow(clippy::useless_conversion)]
        if self.config.include_magnetic_mirroring || self.config.include_ambient_electric_field {
            let magnetic_field = snapshot.cached_vector_field("b");
            let mut magnetic_field_direction = interpolator.interp_vector_field_known_cell(
                magnetic_field,
                &Point3::from(&deposition_position),
                &deposition_indices,
            );

            if self.config.include_magnetic_mirroring {
                magnetic_field_strength =
                    feb::from(magnetic_field_direction.normalize_and_get_length()) * (*U_B);
            } else {
                magnetic_field_direction.normalize();
            }

            if self.config.include_ambient_electric_field {
                let electric_field = snapshot.cached_vector_field("e");
                let electric_field_vector = interpolator.interp_vector_field_known_cell(
                    electric_field,
                    &Point3::from(&deposition_position),
                    &deposition_indices,
                );

                trajectory_aligned_electric_field =
                    feb::from(electric_field_vector.dot(&magnetic_field_direction)) * (*U_EL);

                if self.distribution.propagation_sense == SteppingSense::Opposite {
                    trajectory_aligned_electric_field = -trajectory_aligned_electric_field;
                }
            }
        }

        let hydrogen_ionization_fraction = if self.config.keep_initial_ionization_fraction {
            self.transporter
                .hybrid_coulomb_log()
                .hydrogen_ionization_fraction()
        } else {
            ionization::compute_equilibrium_hydrogen_ionization_fraction(
                temperature,
                electron_density,
            )
        };

        let total_hydrogen_density =
            AnalyticalPropagator::compute_total_hydrogen_density(mass_density);

        let electron_to_hydrogen_ratio = if self.config.assume_ambient_electrons_all_from_hydrogen {
            hydrogen_ionization_fraction
        } else {
            electron_density / total_hydrogen_density
        };

        let hybrid_coulomb_log = HybridCoulombLogarithm::new(
            self.config.enable_warm_target,
            self.coulomb_log.clone(),
            temperature,
            electron_to_hydrogen_ratio,
            hydrogen_ionization_fraction,
        );

        let step_length = displacement.length() * U_L; // [cm]
        let col_depth_increase = step_length * total_hydrogen_density;

        let grid_cell_volume = snapshot.grid().grid_cell_volume(&deposition_indices) * U_L3;
        let beam_cross_sectional_area = grid_cell_volume / step_length;

        let mut mean_deposited_power_per_dist = 0.0;
        let mut deposited_power_per_dist;
        let mut depletion_status = DepletionStatus::Undepleted;

        let n_substeps = self.determine_n_substeps(col_depth_increase);

        let substep_col_depth_increase = col_depth_increase / (n_substeps as feb);
        let substep_length = step_length / (n_substeps as feb);

        for _ in 0..n_substeps {
            (deposited_power_per_dist, depletion_status) = self.advance_distributions(
                hybrid_coulomb_log.clone(),
                total_hydrogen_density,
                temperature,
                trajectory_aligned_electric_field,
                magnetic_field_strength,
                beam_cross_sectional_area,
                substep_col_depth_increase,
            );
            self.distance += substep_length;

            mean_deposited_power_per_dist += deposited_power_per_dist;

            if depletion_status == DepletionStatus::Depleted {
                break;
            } else if let Some(detailed_output) = self.detailed_output.as_mut() {
                detailed_output.push_data(
                    self.distance,
                    self.transporter.total_electron_flux_over_cross_section(),
                    &self.energies,
                    &self.pitch_angle_cosines,
                    &self.electron_numbers_per_dist,
                );
            }
        }
        mean_deposited_power_per_dist /= n_substeps as feb;

        let deposited_power = mean_deposited_power_per_dist * step_length;
        let deposited_power_density = deposited_power / grid_cell_volume;

        let remaining_flux_fraction = self.transporter.total_electron_flux_over_cross_section()
            / self.initial_total_electron_flux_over_cross_section;

        self.step_count += 1;

        let depletion_status = if (self.config.continue_depleted_beams
            || remaining_flux_fraction >= self.config.min_remaining_flux_fraction
            || mean_deposited_power_per_dist >= self.config.min_deposited_power_per_distance)
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

    fn end_propagation(&self) {
        if let (Some(detailed_output_config), Some(detailed_output)) = (
            self.config.detailed_output_config.as_ref(),
            self.detailed_output.as_ref(),
        ) {
            let mut detailed_output_file_path = detailed_output_config.detailed_output_dir.clone();
            detailed_output_file_path.push(format!("{}.npz", self.id));

            exit_on_error!(
                detailed_output.write_as_npz(
                    detailed_output_file_path,
                    &detailed_output_config.atomic_output_file_map,
                ),
                "Error: Could not write detailed output file for beam: {}"
            );
        }
    }
}

impl DetailedOutput {
    fn new(
        total_electron_flux_over_cross_section: feb,
        energies: Vec<feb>,
        pitch_angle_cosines: Vec<feb>,
        electron_numbers_per_dist: Vec<feb>,
    ) -> Self {
        let n_electrons = energies.len();
        assert_eq!(pitch_angle_cosines.len(), n_electrons);
        assert_eq!(electron_numbers_per_dist.len(), n_electrons);

        let distances = vec![0.0];
        let total_electron_fluxes_over_cross_section = vec![total_electron_flux_over_cross_section];

        let energies = Array1::from_vec(energies)
            .into_shape((1, n_electrons))
            .unwrap();
        let pitch_angle_cosines = Array1::from_vec(pitch_angle_cosines)
            .into_shape((1, n_electrons))
            .unwrap();
        let electron_numbers_per_dist = Array1::from_vec(electron_numbers_per_dist)
            .into_shape((1, n_electrons))
            .unwrap();

        Self {
            distances,
            total_electron_fluxes_over_cross_section,
            energies,
            pitch_angle_cosines,
            electron_numbers_per_dist,
        }
    }

    fn push_data(
        &mut self,
        distance: feb,
        total_electron_flux_over_cross_section: feb,
        energies: &[feb],
        pitch_angle_cosines: &[feb],
        electron_numbers_per_dist: &[feb],
    ) {
        self.distances.push(distance);
        self.total_electron_fluxes_over_cross_section
            .push(total_electron_flux_over_cross_section);
        self.energies.push_row(ArrayView::from(energies)).unwrap();
        self.pitch_angle_cosines
            .push_row(ArrayView::from(pitch_angle_cosines))
            .unwrap();
        self.electron_numbers_per_dist
            .push_row(ArrayView::from(electron_numbers_per_dist))
            .unwrap();
    }

    fn write_as_npz(
        &self,
        file_path: PathBuf,
        atomic_output_file_map: &Mutex<AtomicOutputFileMap>,
    ) -> io::Result<()> {
        let atomic_output_file = atomic_output_file_map
            .lock()
            .unwrap()
            .register_output_path(file_path)?;

        let file =
            io_utils::create_file_and_required_directories(atomic_output_file.temporary_path())?;

        let mut writer = NpzWriter::new_compressed(file);

        let map_err = |err| {
            io::Error::new(
                io::ErrorKind::Other,
                format!("Failed to write .npz file with detailed beam data: {}", err),
            )
        };

        writer
            .add_array("distances", &ArrayView::from(&self.distances))
            .map_err(map_err)?;

        writer
            .add_array(
                "total_electron_fluxes_over_cross_section",
                &ArrayView::from(&self.total_electron_fluxes_over_cross_section),
            )
            .map_err(map_err)?;

        writer
            .add_array("energies", &self.energies)
            .map_err(map_err)?;

        writer
            .add_array("pitch_angle_cosines", &self.pitch_angle_cosines)
            .map_err(map_err)?;

        writer
            .add_array("electron_numbers_per_dist", &self.electron_numbers_per_dist)
            .map_err(map_err)?;

        if let Err(err) = writer.finish() {
            return Err(map_err(err));
        };

        atomic_output_file_map
            .lock()
            .unwrap()
            .move_to_target(atomic_output_file)
    }
}

impl CharacteristicsPropagatorConfig {
    pub const DEFAULT_N_ENERGIES: usize = 40;
    pub const DEFAULT_MIN_ENERGY_RELATIVE_TO_CUTOFF: feb = 0.05;
    pub const DEFAULT_MAX_ENERGY_RELATIVE_TO_CUTOFF: feb = 120.0;
    pub const DEFAULT_MAX_COL_DEPTH_INCREASE: feb = 2e14;
    pub const DEFAULT_MAX_SUBSTEPS: usize = 10000;
    pub const DEFAULT_N_INITIAL_STEPS_WITH_SUBSTEPS: usize = 0;
    pub const DEFAULT_N_SUBSTEPS: usize = 1;
    pub const DEFAULT_KEEP_INITIAL_IONIZATION_FRACTION: bool = false;
    pub const DEFAULT_ASSUME_AMBIENT_ELECTRONS_ALL_FROM_HYDROGEN: bool = false;
    pub const DEFAULT_AMBIENT_ELECTRIC_FIELD: bool = false;
    pub const DEFAULT_INCLUDE_RETURN_CURRENT: bool = false;
    pub const DEFAULT_INCLUDE_MAGNETIC_MIRRORING: bool = false;
    pub const DEFAULT_ENABLE_WARM_TARGET: bool = false;
    pub const DEFAULT_MIN_DEPLETION_DISTANCE: feb = 0.5; // [Mm]
    pub const DEFAULT_MIN_REMAINING_FLUX_FRACTION: feb = 1e-5;
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
            n_energies: Self::DEFAULT_N_ENERGIES,
            min_energy_relative_to_cutoff: Self::DEFAULT_MIN_ENERGY_RELATIVE_TO_CUTOFF,
            max_energy_relative_to_cutoff: Self::DEFAULT_MAX_ENERGY_RELATIVE_TO_CUTOFF,
            max_col_depth_increase: Self::DEFAULT_MAX_COL_DEPTH_INCREASE,
            max_substeps: Self::DEFAULT_MAX_SUBSTEPS,
            n_initial_steps_with_substeps: Self::DEFAULT_N_INITIAL_STEPS_WITH_SUBSTEPS,
            n_substeps: Self::DEFAULT_N_SUBSTEPS,
            keep_initial_ionization_fraction: Self::DEFAULT_KEEP_INITIAL_IONIZATION_FRACTION,
            assume_ambient_electrons_all_from_hydrogen:
                Self::DEFAULT_ASSUME_AMBIENT_ELECTRONS_ALL_FROM_HYDROGEN,
            include_ambient_electric_field: Self::DEFAULT_AMBIENT_ELECTRIC_FIELD,
            include_return_current: Self::DEFAULT_INCLUDE_RETURN_CURRENT,
            include_magnetic_mirroring: Self::DEFAULT_INCLUDE_MAGNETIC_MIRRORING,
            enable_warm_target: Self::DEFAULT_ENABLE_WARM_TARGET,
            min_depletion_distance,
            min_residual_factor,
            min_remaining_flux_fraction: Self::DEFAULT_MIN_REMAINING_FLUX_FRACTION,
            min_deposited_power_per_distance,
            max_propagation_distance,
            continue_depleted_beams: Self::DEFAULT_CONTINUE_DEPLETED_BEAMS,
            detailed_output_config: None,
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
            self.max_col_depth_increase > 0.0,
            "Maximum column depth increase must be larger than zero"
        );
        assert!(
            self.max_substeps > 0,
            "Maximum number of substeps must be larger than zero"
        );
        assert!(
            self.n_substeps > 0,
            "Number of substeps must be larger than zero."
        );
        assert!(
            self.min_depletion_distance >= 0.0,
            "Minimum stopping distance must be larger than or equal to zero."
        );
        assert!(
            self.min_remaining_flux_fraction >= 0.0,
            "Minimum remaining flux factor must be larger than or equal to zero."
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
            n_energies: Self::DEFAULT_N_ENERGIES,
            min_energy_relative_to_cutoff: Self::DEFAULT_MIN_ENERGY_RELATIVE_TO_CUTOFF,
            max_energy_relative_to_cutoff: Self::DEFAULT_MAX_ENERGY_RELATIVE_TO_CUTOFF,
            max_col_depth_increase: Self::DEFAULT_MAX_COL_DEPTH_INCREASE,
            max_substeps: Self::DEFAULT_MAX_SUBSTEPS,
            n_initial_steps_with_substeps: Self::DEFAULT_N_INITIAL_STEPS_WITH_SUBSTEPS,
            n_substeps: Self::DEFAULT_N_SUBSTEPS,
            keep_initial_ionization_fraction: Self::DEFAULT_KEEP_INITIAL_IONIZATION_FRACTION,
            assume_ambient_electrons_all_from_hydrogen:
                Self::DEFAULT_ASSUME_AMBIENT_ELECTRONS_ALL_FROM_HYDROGEN,
            include_ambient_electric_field: Self::DEFAULT_AMBIENT_ELECTRIC_FIELD,
            include_return_current: Self::DEFAULT_INCLUDE_RETURN_CURRENT,
            include_magnetic_mirroring: Self::DEFAULT_INCLUDE_MAGNETIC_MIRRORING,
            enable_warm_target: Self::DEFAULT_ENABLE_WARM_TARGET,
            min_depletion_distance: Self::DEFAULT_MIN_DEPLETION_DISTANCE,
            min_residual_factor: Self::DEFAULT_MIN_RESIDUAL_FACTOR,
            min_remaining_flux_fraction: Self::DEFAULT_MIN_REMAINING_FLUX_FRACTION,
            min_deposited_power_per_distance: Self::DEFAULT_MIN_DEPOSITED_POWER_PER_DISTANCE,
            max_propagation_distance: Self::DEFAULT_MAX_PROPAGATION_DISTANCE,
            continue_depleted_beams: Self::DEFAULT_CONTINUE_DEPLETED_BEAMS,
            detailed_output_config: None,
        }
    }
}
