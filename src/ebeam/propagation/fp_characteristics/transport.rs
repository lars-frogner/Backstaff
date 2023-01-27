//!

use super::atmosphere::{compute_parallel_resistivity, HybridCoulombLogarithm};
use crate::{
    constants::{M_ELECTRON, PI, Q_ELECTRON},
    ebeam::feb,
};
use roots::{self, SimpleConvergency};

const COLLISION_SCALE: feb = 2.0 * PI * Q_ELECTRON * Q_ELECTRON * Q_ELECTRON * Q_ELECTRON;

#[derive(Clone, Debug)]
pub struct Transporter {
    include_ambient_electric_field: bool,
    include_induced_electric_field: bool,
    include_magnetic_field: bool,
    analytical_transporter: AnalyticalTransporter,
    hybrid_coulomb_log: HybridCoulombLogarithm,
    total_hydrogen_density: feb,
    temperature: feb,
    ambient_trajectory_aligned_electric_field: feb,
    induced_trajectory_aligned_electric_field: feb,
    total_trajectory_aligned_electric_field: feb,
    magnetic_field_strength: feb,
    log_magnetic_field_col_depth_deriv: feb,
    energy_loss_to_electric_field: feb,
    high_energy_pitch_angle_cos: feb,
}

#[derive(Clone, Debug)]
struct AnalyticalTransporter {
    config: AnalyticalTransporterConfig,
    electric_field_scale: feb,
    magnetic_field_scale: feb,
}

#[derive(Clone, Debug)]
pub struct AnalyticalTransporterConfig {
    pub enabled: bool,
    pub max_electric_field_ratio: feb,
    pub max_pitch_angle_cos: feb,
    pub max_magnetic_field_factor: feb,
    pub max_electric_field_factor: feb,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum TransportResult {
    NewValues((feb, feb, feb)),
    Thermalized,
}

impl Transporter {
    const MIN_MAGNETIC_FIELD: feb = 0.1;

    pub fn new(
        config: AnalyticalTransporterConfig,
        include_ambient_electric_field: bool,
        include_induced_electric_field: bool,
        include_magnetic_field: bool,
        initial_pitch_angle_cos: feb,
        hybrid_coulomb_log: HybridCoulombLogarithm,
        total_hydrogen_density: feb,
        temperature: feb,
        ambient_trajectory_aligned_electric_field: feb,
        magnetic_field_strength: feb,
    ) -> Self {
        let magnetic_field_strength = if include_magnetic_field {
            magnetic_field_strength
        } else {
            0.0
        };
        let log_magnetic_field_col_depth_deriv = 0.0;

        let ambient_trajectory_aligned_electric_field = if include_ambient_electric_field {
            ambient_trajectory_aligned_electric_field
        } else {
            0.0
        };
        let induced_trajectory_aligned_electric_field = 0.0;
        let total_trajectory_aligned_electric_field =
            ambient_trajectory_aligned_electric_field + induced_trajectory_aligned_electric_field;

        let energy_loss_to_electric_field = 0.0;
        let high_energy_pitch_angle_cos = initial_pitch_angle_cos;

        let analytical_transporter = AnalyticalTransporter::new(
            config,
            &hybrid_coulomb_log,
            total_hydrogen_density,
            ambient_trajectory_aligned_electric_field,
            log_magnetic_field_col_depth_deriv,
        );
        Self {
            include_ambient_electric_field,
            include_induced_electric_field,
            include_magnetic_field,
            analytical_transporter,
            hybrid_coulomb_log,
            total_hydrogen_density,
            temperature,
            ambient_trajectory_aligned_electric_field,
            induced_trajectory_aligned_electric_field,
            total_trajectory_aligned_electric_field,
            magnetic_field_strength,
            log_magnetic_field_col_depth_deriv,
            energy_loss_to_electric_field,
            high_energy_pitch_angle_cos,
        }
    }

    pub fn energy_without_loss_to_electric_field(&self, energy: feb) -> feb {
        energy + self.energy_loss_to_electric_field
    }

    pub fn high_energy_pitch_angle_cos(&self) -> feb {
        self.high_energy_pitch_angle_cos
    }

    pub fn n_steps_to_thermalization(
        &self,
        initial_energy: feb,
        initial_pitch_angle_cos: feb,
        col_depth_increase: feb,
    ) -> feb {
        initial_energy
            / (-self.compute_energy_col_depth_deriv(initial_energy, initial_pitch_angle_cos)
                * col_depth_increase)
    }

    pub fn update_conditions(
        &mut self,
        hybrid_coulomb_log: HybridCoulombLogarithm,
        total_hydrogen_density: feb,
        temperature: feb,
        ambient_trajectory_aligned_electric_field: feb,
        magnetic_field_strength: feb,
        energies: &[feb],
        initial_energies: &[feb],
        pitch_angle_cosines: &[feb],
        electron_numbers_per_dist: &[feb],
        beam_cross_sectional_area: feb,
        col_depth_increase: feb,
    ) {
        self.hybrid_coulomb_log = hybrid_coulomb_log;
        self.total_hydrogen_density = total_hydrogen_density;
        self.temperature = temperature;

        if self.include_magnetic_field {
            self.log_magnetic_field_col_depth_deriv =
                Self::compute_log_magnetic_field_col_depth_deriv(
                    self.magnetic_field_strength,
                    magnetic_field_strength,
                    col_depth_increase,
                );

            self.magnetic_field_strength = magnetic_field_strength;

            Self::update_high_energy_pitch_angle_cos(
                &mut self.high_energy_pitch_angle_cos,
                self.log_magnetic_field_col_depth_deriv,
                col_depth_increase,
            );
        } else {
            self.magnetic_field_strength = 0.0;
            self.log_magnetic_field_col_depth_deriv = 0.0;
        }

        if self.include_ambient_electric_field {
            self.ambient_trajectory_aligned_electric_field =
                ambient_trajectory_aligned_electric_field;

            // Use old value for induced field when calculating total electric field for now
            self.total_trajectory_aligned_electric_field = self
                .ambient_trajectory_aligned_electric_field
                + self.induced_trajectory_aligned_electric_field;
        } else {
            self.ambient_trajectory_aligned_electric_field = 0.0;
        }

        if self.include_induced_electric_field {
            // Because the induced field depends on the flux, which depends
            // on the energy derivative, which depends on the electric field,
            // determining the induced field is strictly an implicit problem.
            // We handle this by keeping the old value for the electric field
            // when computing the flux, assuming that the delay of one step does
            // not make a significant difference
            self.induced_trajectory_aligned_electric_field = self
                .compute_induced_trajectory_aligned_electric_field(
                    temperature,
                    energies,
                    initial_energies,
                    pitch_angle_cosines,
                    electron_numbers_per_dist,
                    beam_cross_sectional_area,
                );
        } else {
            self.induced_trajectory_aligned_electric_field = 0.0;
        }

        if self.include_ambient_electric_field || self.include_induced_electric_field {
            self.total_trajectory_aligned_electric_field = self
                .ambient_trajectory_aligned_electric_field
                + self.induced_trajectory_aligned_electric_field;

            Self::update_energy_loss_to_electric_field(
                &mut self.energy_loss_to_electric_field,
                self.total_trajectory_aligned_electric_field,
                self.total_hydrogen_density,
                col_depth_increase,
            );
        } else {
            self.total_trajectory_aligned_electric_field = 0.0;
        }

        self.analytical_transporter.update_conditions(
            &self.hybrid_coulomb_log,
            self.total_hydrogen_density,
            self.total_trajectory_aligned_electric_field,
            self.log_magnetic_field_col_depth_deriv,
        );
    }

    pub fn advance_quantities(
        &self,
        initial_energy: feb,
        initial_pitch_angle_cos: feb,
        initial_number_density: feb,
        col_depth_increase: feb,
    ) -> TransportResult {
        if initial_pitch_angle_cos <= 0.0 {
            TransportResult::Thermalized
        } else {
            self.analytical_transporter
                .try_advancing_quantities(
                    &self.hybrid_coulomb_log,
                    initial_energy,
                    initial_pitch_angle_cos,
                    initial_number_density,
                    col_depth_increase,
                )
                .unwrap_or_else(|| {
                    self.advance_quantities_with_second_order_heun(
                        initial_energy,
                        initial_pitch_angle_cos,
                        initial_number_density,
                        col_depth_increase,
                    )
                })
        }
    }

    pub fn compute_deposited_power_density(
        &self,
        energies: &[feb],
        initial_energies: &[feb],
        pitch_angle_cosines: &[feb],
        number_densities: &[feb],
    ) -> feb {
        assert_eq!(initial_energies.len(), energies.len());
        assert_eq!(pitch_angle_cosines.len(), energies.len());
        assert_eq!(number_densities.len(), energies.len());

        let mut first_nonzero_idx = None;

        let deposited_power_initial_energy_derivs: Vec<_> = energies
            .iter()
            .zip(initial_energies.iter())
            .zip(pitch_angle_cosines.iter())
            .zip(number_densities.iter())
            .enumerate()
            .map(
                |(idx, (((&energy, &initial_energy), &pitch_angle_cos), &number_density))| {
                    if initial_energy <= 0.0 || pitch_angle_cos <= 0.0 || number_density <= 0.0 {
                        0.0
                    } else {
                        if first_nonzero_idx.is_none() {
                            first_nonzero_idx = Some(idx);
                        }
                        let energy_col_depth_deriv =
                            self.compute_energy_col_depth_deriv(energy, pitch_angle_cos);
                        let initial_energy_col_depth_deriv =
                            self.compute_energy_col_depth_deriv(initial_energy, pitch_angle_cos);

                        (energy_col_depth_deriv / initial_energy_col_depth_deriv)
                            * number_density
                            * (-energy_col_depth_deriv
                                * self.total_hydrogen_density
                                * pitch_angle_cos
                                * feb::sqrt(2.0 * energy / M_ELECTRON))
                    }
                },
            )
            .collect();

        let first_nonzero_idx = first_nonzero_idx.unwrap_or(0);

        let valid_initial_energies = &initial_energies[first_nonzero_idx..];
        let valid_deposited_power_initial_energy_derivs =
            &deposited_power_initial_energy_derivs[first_nonzero_idx..];

        // Integrate deposited power over initial energies using the trapezoidal method
        integrate_trapezoidal(
            valid_initial_energies,
            valid_deposited_power_initial_energy_derivs,
        )
    }

    fn compute_total_flux_density(
        &self,
        energies: &[feb],
        initial_energies: &[feb],
        pitch_angle_cosines: &[feb],
        number_densities: &[feb],
    ) -> feb {
        assert_eq!(initial_energies.len(), energies.len());
        assert_eq!(pitch_angle_cosines.len(), energies.len());
        assert_eq!(number_densities.len(), energies.len());

        let mut first_nonzero_idx = None;

        let flux_initial_energy_derivs: Vec<_> = energies
            .iter()
            .zip(initial_energies.iter())
            .zip(pitch_angle_cosines.iter())
            .zip(number_densities.iter())
            .enumerate()
            .map(
                |(idx, (((&energy, &initial_energy), &pitch_angle_cos), &number_density))| {
                    if initial_energy <= 0.0 || pitch_angle_cos <= 0.0 || number_density <= 0.0 {
                        0.0
                    } else {
                        if first_nonzero_idx.is_none() {
                            first_nonzero_idx = Some(idx);
                        }
                        let energy_col_depth_deriv =
                            self.compute_energy_col_depth_deriv(energy, pitch_angle_cos);
                        let initial_energy_col_depth_deriv =
                            self.compute_energy_col_depth_deriv(initial_energy, pitch_angle_cos);

                        (energy_col_depth_deriv / initial_energy_col_depth_deriv)
                            * number_density
                            * feb::sqrt(2.0 * energy / M_ELECTRON)
                    }
                },
            )
            .collect();

        let first_nonzero_idx = first_nonzero_idx.unwrap_or(0);

        let valid_initial_energies = &initial_energies[first_nonzero_idx..];
        let valid_flux_initial_energy_derivs = &flux_initial_energy_derivs[first_nonzero_idx..];

        integrate_trapezoidal(valid_initial_energies, valid_flux_initial_energy_derivs)
    }

    fn compute_induced_trajectory_aligned_electric_field(
        &self,
        temperature: feb,
        energies: &[feb],
        initial_energies: &[feb],
        pitch_angle_cosines: &[feb],
        electron_numbers_per_dist: &[feb],
        beam_cross_sectional_area: feb,
    ) -> feb {
        // Since we use electron count per distance and not
        // electron density, what we compute here is really
        // the flux density multiplied with area
        let total_flux_over_cross_section = self.compute_total_flux_density(
            energies,
            initial_energies,
            pitch_angle_cosines,
            electron_numbers_per_dist,
        );

        let total_flux_density = total_flux_over_cross_section / beam_cross_sectional_area;

        Q_ELECTRON
            * compute_parallel_resistivity(
                self.hybrid_coulomb_log.coulomb_log(),
                temperature,
                self.hybrid_coulomb_log.ionization_fraction(),
            )
            * total_flux_density
    }

    fn compute_log_magnetic_field_col_depth_deriv(
        prev_magnetic_field_strength: feb,
        magnetic_field_strength: feb,
        col_depth_increase: feb,
    ) -> feb {
        let mean_magnetic_field_strength =
            0.5 * (prev_magnetic_field_strength + magnetic_field_strength);
        if mean_magnetic_field_strength >= Self::MIN_MAGNETIC_FIELD {
            (magnetic_field_strength - prev_magnetic_field_strength)
                / (mean_magnetic_field_strength * col_depth_increase)
        } else {
            0.0
        }
    }

    fn update_energy_loss_to_electric_field(
        energy_loss_to_electric_field: &mut feb,
        total_trajectory_aligned_electric_field: feb,
        total_hydrogen_density: feb,
        col_depth_increase: feb,
    ) {
        *energy_loss_to_electric_field += (Q_ELECTRON * total_trajectory_aligned_electric_field
            / total_hydrogen_density)
            * col_depth_increase;
    }

    fn update_high_energy_pitch_angle_cos(
        high_energy_pitch_angle_cos: &mut feb,
        log_magnetic_field_col_depth_deriv: feb,
        col_depth_increase: feb,
    ) {
        *high_energy_pitch_angle_cos = feb::sqrt(feb::max(
            0.0,
            1.0 - (1.0 - (*high_energy_pitch_angle_cos) * (*high_energy_pitch_angle_cos))
                * feb::exp(log_magnetic_field_col_depth_deriv * col_depth_increase),
        ));
    }

    fn advance_quantities_with_second_order_heun(
        &self,
        initial_energy: feb,
        initial_pitch_angle_cos: feb,
        initial_number_density: feb,
        col_depth_increase: feb,
    ) -> TransportResult {
        let energy_col_depth_deriv_1 =
            self.compute_energy_col_depth_deriv(initial_energy, initial_pitch_angle_cos);
        let pitch_angle_cos_col_depth_deriv_1 =
            self.compute_pitch_angle_cos_col_depth_deriv(initial_energy, initial_pitch_angle_cos);
        let number_density_col_depth_deriv_1 = self.compute_number_density_col_depth_deriv(
            initial_energy,
            initial_pitch_angle_cos,
            initial_number_density,
        );

        let energy_1 = initial_energy + energy_col_depth_deriv_1 * col_depth_increase;
        let pitch_angle_cos_1 =
            initial_pitch_angle_cos + pitch_angle_cos_col_depth_deriv_1 * col_depth_increase;
        let number_density_1 =
            initial_number_density + number_density_col_depth_deriv_1 * col_depth_increase;

        if energy_1 <= 0.0 || pitch_angle_cos_1 <= 0.0 {
            return TransportResult::Thermalized;
        }

        let energy_col_depth_deriv_2 =
            self.compute_energy_col_depth_deriv(energy_1, pitch_angle_cos_1);
        let pitch_angle_cos_col_depth_deriv_2 =
            self.compute_pitch_angle_cos_col_depth_deriv(energy_1, pitch_angle_cos_1);
        let number_density_col_depth_deriv_2 = self.compute_number_density_col_depth_deriv(
            energy_1,
            pitch_angle_cos_1,
            number_density_1,
        );

        let energy = initial_energy
            + 0.5 * (energy_col_depth_deriv_1 + energy_col_depth_deriv_2) * col_depth_increase;
        let pitch_angle_cos = initial_pitch_angle_cos
            + 0.5
                * (pitch_angle_cos_col_depth_deriv_1 + pitch_angle_cos_col_depth_deriv_2)
                * col_depth_increase;
        let number_density = initial_number_density
            + 0.5
                * (number_density_col_depth_deriv_1 + number_density_col_depth_deriv_2)
                * col_depth_increase;

        if energy <= 0.0 || pitch_angle_cos <= 0.0 {
            TransportResult::Thermalized
        } else {
            TransportResult::NewValues((energy, pitch_angle_cos, feb::max(0.0, number_density)))
        }
    }

    #[allow(dead_code)]
    fn advance_quantities_with_third_order_heun(
        &self,
        initial_energy: feb,
        initial_pitch_angle_cos: feb,
        initial_number_density: feb,
        col_depth_increase: feb,
    ) -> TransportResult {
        let energy_col_depth_deriv_1 =
            self.compute_energy_col_depth_deriv(initial_energy, initial_pitch_angle_cos);
        let pitch_angle_cos_col_depth_deriv_1 =
            self.compute_pitch_angle_cos_col_depth_deriv(initial_energy, initial_pitch_angle_cos);
        let number_density_col_depth_deriv_1 = self.compute_number_density_col_depth_deriv(
            initial_energy,
            initial_pitch_angle_cos,
            initial_number_density,
        );

        let energy_1 = initial_energy + energy_col_depth_deriv_1 * col_depth_increase / 3.0;
        let pitch_angle_cos_1 =
            initial_pitch_angle_cos + pitch_angle_cos_col_depth_deriv_1 * col_depth_increase / 3.0;
        let number_density_1 =
            initial_number_density + number_density_col_depth_deriv_1 * col_depth_increase / 3.0;

        if energy_1 <= 0.0 || pitch_angle_cos_1 <= 0.0 {
            return TransportResult::Thermalized;
        }

        let energy_col_depth_deriv_2 =
            self.compute_energy_col_depth_deriv(energy_1, pitch_angle_cos_1);
        let pitch_angle_cos_col_depth_deriv_2 =
            self.compute_pitch_angle_cos_col_depth_deriv(energy_1, pitch_angle_cos_1);
        let number_density_col_depth_deriv_2 = self.compute_number_density_col_depth_deriv(
            energy_1,
            pitch_angle_cos_1,
            number_density_1,
        );

        let energy_2 = energy_1 + energy_col_depth_deriv_2 * col_depth_increase * 2.0 / 3.0;
        let pitch_angle_cos_2 =
            pitch_angle_cos_1 + pitch_angle_cos_col_depth_deriv_2 * col_depth_increase * 2.0 / 3.0;
        let number_density_2 =
            number_density_1 + number_density_col_depth_deriv_2 * col_depth_increase * 2.0 / 3.0;

        if energy_2 <= 0.0 || pitch_angle_cos_2 <= 0.0 {
            return TransportResult::Thermalized;
        }

        let energy_col_depth_deriv_3 =
            self.compute_energy_col_depth_deriv(energy_2, pitch_angle_cos_2);
        let pitch_angle_cos_col_depth_deriv_3 =
            self.compute_pitch_angle_cos_col_depth_deriv(energy_2, pitch_angle_cos_2);
        let number_density_col_depth_deriv_3 = self.compute_number_density_col_depth_deriv(
            energy_2,
            pitch_angle_cos_2,
            number_density_2,
        );

        let energy = initial_energy
            + (0.25 * energy_col_depth_deriv_1 + 0.75 * energy_col_depth_deriv_3)
                * col_depth_increase;
        let pitch_angle_cos = initial_pitch_angle_cos
            + (0.25 * pitch_angle_cos_col_depth_deriv_1 + 0.75 * pitch_angle_cos_col_depth_deriv_3)
                * col_depth_increase;
        let number_density = initial_number_density
            + (0.25 * number_density_col_depth_deriv_1 + 0.75 * number_density_col_depth_deriv_3)
                * col_depth_increase;

        if energy <= 0.0 || pitch_angle_cos <= 0.0 {
            TransportResult::Thermalized
        } else {
            TransportResult::NewValues((energy, pitch_angle_cos, feb::max(0.0, number_density)))
        }
    }

    fn compute_energy_col_depth_deriv(&self, energy: feb, pitch_angle_cos: feb) -> feb {
        -COLLISION_SCALE * self.hybrid_coulomb_log.for_energy() / (pitch_angle_cos * energy)
            - Q_ELECTRON * self.total_trajectory_aligned_electric_field
                / self.total_hydrogen_density
    }

    fn compute_pitch_angle_cos_col_depth_deriv(&self, energy: feb, pitch_angle_cos: feb) -> feb {
        -COLLISION_SCALE * self.hybrid_coulomb_log.for_pitch_angle() / (2.0 * energy * energy)
            - (self.log_magnetic_field_col_depth_deriv
                + Q_ELECTRON * self.total_trajectory_aligned_electric_field
                    / (self.total_hydrogen_density * energy))
                * (1.0 - pitch_angle_cos * pitch_angle_cos)
                / (2.0 * pitch_angle_cos)
    }

    fn compute_number_density_col_depth_deriv_per_number_density(
        &self,
        energy: feb,
        pitch_angle_cos: feb,
    ) -> feb {
        COLLISION_SCALE * self.hybrid_coulomb_log.for_number_density()
            / (2.0 * pitch_angle_cos * energy * energy)
    }

    fn compute_number_density_col_depth_deriv(
        &self,
        energy: feb,
        pitch_angle_cos: feb,
        number_density: feb,
    ) -> feb {
        self.compute_number_density_col_depth_deriv_per_number_density(energy, pitch_angle_cos)
            * number_density
    }
}

impl AnalyticalTransporter {
    const NEWTON_EPS: feb = 2e-5;
    const NEWTON_MAX_ITER: usize = 100;

    fn new(
        config: AnalyticalTransporterConfig,
        hybrid_coulomb_log: &HybridCoulombLogarithm,
        total_hydrogen_density: feb,
        trajectory_aligned_electric_field: feb,
        log_magnetic_field_col_depth_deriv: feb,
    ) -> Self {
        let electric_field_scale = Self::compute_electric_field_scale(
            hybrid_coulomb_log,
            total_hydrogen_density,
            trajectory_aligned_electric_field,
        );
        let magnetic_field_scale = Self::compute_magnetic_field_scale(
            hybrid_coulomb_log,
            log_magnetic_field_col_depth_deriv,
        );
        Self {
            config,
            electric_field_scale,
            magnetic_field_scale,
        }
    }

    fn try_advancing_quantities(
        &self,
        hybrid_coulomb_log: &HybridCoulombLogarithm,
        initial_energy: feb,
        initial_pitch_angle_cos: feb,
        initial_number_density: feb,
        col_depth_increase: feb,
    ) -> Option<TransportResult> {
        if !self.config.enabled {
            return None;
        }

        let electric_field_factor =
            self.electric_field_scale * initial_energy / initial_pitch_angle_cos;
        let magnetic_field_factor =
            self.magnetic_field_scale * initial_energy * initial_energy / initial_pitch_angle_cos;

        if (initial_pitch_angle_cos > self.config.max_pitch_angle_cos
            || initial_energy * initial_pitch_angle_cos * self.electric_field_scale
                > self.config.max_electric_field_ratio)
            && (feb::abs(magnetic_field_factor) > self.config.max_magnetic_field_factor
                || feb::abs(electric_field_factor) > self.config.max_electric_field_factor)
        {
            // The approximate analytical solution is not valid
            return None;
        }

        let beta = hybrid_coulomb_log.for_pitch_angle_for_energy_ratio();

        if 2.0 - beta < 0.05 {
            // The more efficient solution for ionized plasma (beta = 2)
            // can safely be used until beta deviates significantly from 2
            self.try_advancing_quantities_ionized(
                hybrid_coulomb_log,
                initial_energy,
                initial_pitch_angle_cos,
                initial_number_density,
                col_depth_increase,
                electric_field_factor,
                magnetic_field_factor,
            )
        } else {
            self.try_advancing_quantities_unionized(
                hybrid_coulomb_log,
                initial_energy,
                initial_pitch_angle_cos,
                initial_number_density,
                col_depth_increase,
                electric_field_factor,
                magnetic_field_factor,
            )
        }
    }

    fn try_advancing_quantities_ionized(
        &self,
        hybrid_coulomb_log: &HybridCoulombLogarithm,
        initial_energy: feb,
        initial_pitch_angle_cos: feb,
        initial_number_density: feb,
        col_depth_increase: feb,
        electric_field_factor: feb,
        magnetic_field_factor: feb,
    ) -> Option<TransportResult> {
        let offset = COLLISION_SCALE * hybrid_coulomb_log.for_energy() * col_depth_increase
            / (initial_pitch_angle_cos * initial_energy * initial_energy)
            - (1.0 - 0.125 * magnetic_field_factor - electric_field_factor / 6.0) / 3.0;

        if offset >= 0.0 {
            // The electron will thermalize
            return Some(TransportResult::Thermalized);
        }

        let f = |ln_x| {
            let x = feb::exp(ln_x);
            x * x
                * x
                * (1.0 - 0.5 * magnetic_field_factor
                    + electric_field_factor * (0.5 * ln_x - 1.0 / 6.0))
                / 3.0
                + x * x * x * x * magnetic_field_factor * 0.125
                + offset
        };

        let dfdlnx = |ln_x| {
            let x = feb::exp(ln_x);
            x * x
                * x
                * (1.0
                    + 0.5 * magnetic_field_factor * (x - 1.0)
                    + 0.5 * electric_field_factor * ln_x)
        };

        let mut convergency = SimpleConvergency {
            eps: Self::NEWTON_EPS,
            max_iter: Self::NEWTON_MAX_ITER,
        };
        match roots::find_root_newton_raphson(0.0, &f, &dfdlnx, &mut convergency) {
            Ok(ln_x) => {
                let x = feb::exp(ln_x);
                let energy = initial_energy * x;

                let pitch_angle_cos = initial_pitch_angle_cos * x
                    + 0.5 * self.magnetic_field_scale * energy * energy * (1.0 - 1.0 / x)
                    + self.electric_field_scale * energy * ln_x;

                let number_density = initial_number_density / feb::sqrt(x);

                Some(TransportResult::NewValues((
                    energy,
                    pitch_angle_cos,
                    feb::max(0.0, number_density),
                )))
            }
            Err(error) => {
                eprintln!(
                    "Convergence failed when advancing energy and pitch angle analytically: {}",
                    error
                );
                None
            }
        }
    }

    fn try_advancing_quantities_unionized(
        &self,
        hybrid_coulomb_log: &HybridCoulombLogarithm,
        initial_energy: feb,
        initial_pitch_angle_cos: feb,
        initial_number_density: feb,
        col_depth_increase: feb,
        electric_field_factor: feb,
        magnetic_field_factor: feb,
    ) -> Option<TransportResult> {
        let beta = hybrid_coulomb_log.for_pitch_angle_for_energy_ratio();

        let offset = COLLISION_SCALE * hybrid_coulomb_log.for_energy() * col_depth_increase
            / (initial_pitch_angle_cos * initial_energy * initial_energy)
            - (1.0 - 0.125 * magnetic_field_factor - electric_field_factor / 6.0)
                / (2.0 + 0.5 * beta);

        if offset >= 0.0 {
            // The electron will thermalize
            return Some(TransportResult::Thermalized);
        }

        let f = |ln_x| {
            let x = feb::exp(ln_x);
            x.powf(2.0 + 0.5 * beta)
                * (1.0
                    - magnetic_field_factor / (4.0 - beta)
                    - electric_field_factor / (2.0 - beta))
                / (2.0 + 0.5 * beta)
                + x * x * x * x * magnetic_field_factor / (4.0 * (4.0 - beta))
                + x * x * x * electric_field_factor / (3.0 * (2.0 - beta))
                + offset
        };

        let dfdlnx = |ln_x| {
            let x = feb::exp(ln_x);
            x * (x.powf(1.0 + 0.5 * beta)
                * (1.0
                    - magnetic_field_factor / (4.0 - beta)
                    - electric_field_factor / (2.0 - beta))
                + x * x * x * magnetic_field_factor / (4.0 - beta)
                + x * x * electric_field_factor / (2.0 - beta))
        };

        let mut convergency = SimpleConvergency {
            eps: Self::NEWTON_EPS,
            max_iter: Self::NEWTON_MAX_ITER,
        };
        match roots::find_root_newton_raphson(0.0, &f, &dfdlnx, &mut convergency) {
            Ok(ln_x) => {
                let x = feb::exp(ln_x);
                let energy = initial_energy * x;

                let pitch_angle_cos = x.powf(0.5 * beta)
                    * (initial_pitch_angle_cos
                        - initial_energy * initial_energy * self.magnetic_field_scale
                            / (4.0 - beta)
                        - initial_energy * self.electric_field_scale / (2.0 - beta))
                    + energy * energy * self.magnetic_field_scale / (4.0 - beta)
                    + energy * self.electric_field_scale / (2.0 - beta);

                let number_density = initial_number_density
                    * x.powf(-0.5 * hybrid_coulomb_log.for_number_density_for_energy_ratio());

                Some(TransportResult::NewValues((
                    energy,
                    pitch_angle_cos,
                    feb::max(0.0, number_density),
                )))
            }
            Err(error) => {
                eprintln!(
                    "Convergence failed when advancing energy and pitch angle analytically: {}",
                    error
                );
                None
            }
        }
    }

    fn update_conditions(
        &mut self,
        hybrid_coulomb_log: &HybridCoulombLogarithm,
        total_hydrogen_density: feb,
        trajectory_aligned_electric_field: feb,
        log_magnetic_field_col_depth_deriv: feb,
    ) {
        self.electric_field_scale = Self::compute_electric_field_scale(
            hybrid_coulomb_log,
            total_hydrogen_density,
            trajectory_aligned_electric_field,
        );
        self.magnetic_field_scale = Self::compute_magnetic_field_scale(
            hybrid_coulomb_log,
            log_magnetic_field_col_depth_deriv,
        );
    }

    fn compute_magnetic_field_scale(
        hybrid_coulomb_log: &HybridCoulombLogarithm,
        log_magnetic_field_col_depth_deriv: feb,
    ) -> feb {
        log_magnetic_field_col_depth_deriv / (COLLISION_SCALE * hybrid_coulomb_log.for_energy())
    }

    fn compute_electric_field_scale(
        hybrid_coulomb_log: &HybridCoulombLogarithm,
        total_hydrogen_density: feb,
        trajectory_aligned_electric_field: feb,
    ) -> feb {
        Q_ELECTRON * trajectory_aligned_electric_field
            / (COLLISION_SCALE * hybrid_coulomb_log.for_energy() * total_hydrogen_density)
    }
}

impl Default for AnalyticalTransporterConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_electric_field_ratio: 1e-3,
            max_pitch_angle_cos: 1e-1,
            max_magnetic_field_factor: 1e-2,
            max_electric_field_factor: 1e-2,
        }
    }
}

fn integrate_trapezoidal(x: &[feb], f: &[feb]) -> feb {
    0.5 * x
        .iter()
        .zip(x.iter().skip(1))
        .zip(f.iter().zip(f.iter().skip(1)))
        .fold(0.0, |acc, ((&x, &x_up), (&f, &f_up))| {
            acc + (f + f_up) * (x_up - x)
        })
}
