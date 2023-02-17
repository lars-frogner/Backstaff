//!

use super::atmosphere::{
    compute_parallel_resistivity, EvaluatedHydrogenCoulombLogarithms, HybridCoulombLogarithm,
};
use crate::{
    constants::{M_ELECTRON, PI, Q_ELECTRON},
    ebeam::feb,
};

const COLLISION_SCALE: feb = 2.0 * PI * Q_ELECTRON * Q_ELECTRON * Q_ELECTRON * Q_ELECTRON;

#[derive(Clone, Debug)]
pub struct Transporter {
    include_ambient_electric_field: bool,
    include_induced_electric_field: bool,
    include_magnetic_field: bool,
    hybrid_coulomb_log: HybridCoulombLogarithm,
    total_hydrogen_density: feb,
    temperature: feb,
    total_electron_flux_over_cross_section: feb,
    ambient_trajectory_aligned_electric_field: feb,
    induced_trajectory_aligned_electric_field: feb,
    total_trajectory_aligned_electric_field: feb,
    magnetic_field_strength: feb,
    log_magnetic_field_col_depth_deriv: feb,
    energy_loss_to_electric_field: feb,
    high_energy_pitch_angle_cos: feb,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum TransportResult {
    NewValues((feb, feb, feb)),
    Thermalized,
}

#[derive(Clone, Debug)]
struct ColumnDepthDerivatives {
    energy: feb,
    pitch_angle: feb,
    number_density: feb,
}

impl Transporter {
    const MIN_MAGNETIC_FIELD: feb = 0.1;

    pub fn new(
        include_ambient_electric_field: bool,
        include_induced_electric_field: bool,
        include_magnetic_field: bool,
        total_electron_flux_over_cross_section: feb,
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

        Self {
            include_ambient_electric_field,
            include_induced_electric_field,
            include_magnetic_field,
            hybrid_coulomb_log,
            total_hydrogen_density,
            temperature,
            total_electron_flux_over_cross_section,
            ambient_trajectory_aligned_electric_field,
            induced_trajectory_aligned_electric_field,
            total_trajectory_aligned_electric_field,
            magnetic_field_strength,
            log_magnetic_field_col_depth_deriv,
            energy_loss_to_electric_field,
            high_energy_pitch_angle_cos,
        }
    }

    pub fn total_electron_flux_over_cross_section(&self) -> feb {
        self.total_electron_flux_over_cross_section
    }

    pub fn hybrid_coulomb_log(&self) -> &HybridCoulombLogarithm {
        &self.hybrid_coulomb_log
    }

    pub fn energy_without_loss_to_electric_field(&self, energy: feb) -> feb {
        energy + self.energy_loss_to_electric_field
    }

    pub fn high_energy_pitch_angle_cos(&self) -> feb {
        self.high_energy_pitch_angle_cos
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

        self.total_electron_flux_over_cross_section = self
            .compute_total_electron_flux_over_cross_section(
                energies,
                initial_energies,
                pitch_angle_cosines,
                electron_numbers_per_dist,
            );

        if self.include_induced_electric_field {
            let total_electron_flux_density =
                self.total_electron_flux_over_cross_section / beam_cross_sectional_area;

            // Because the induced field depends on the flux, which depends
            // on the energy derivative, which depends on the electric field,
            // determining the induced field is strictly an implicit problem.
            // We handle this by keeping the old value for the electric field
            // when computing the flux, assuming that the delay of one step does
            // not make a significant difference
            self.induced_trajectory_aligned_electric_field = self
                .compute_induced_trajectory_aligned_electric_field(
                    temperature,
                    total_electron_flux_density,
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
            self.advance_quantities_with_third_order_heun(
                initial_energy,
                initial_pitch_angle_cos,
                initial_number_density,
                col_depth_increase,
            )
        }
    }

    pub fn compute_deposited_power_per_dist(
        &self,
        energies: &[feb],
        initial_energies: &[feb],
        pitch_angle_cosines: &[feb],
        electron_numbers_per_dist: &[feb],
    ) -> feb {
        assert_eq!(initial_energies.len(), energies.len());
        assert_eq!(pitch_angle_cosines.len(), energies.len());
        assert_eq!(electron_numbers_per_dist.len(), energies.len());

        let mut first_nonzero_idx = None;

        let deposited_power_initial_energy_derivs: Vec<_> = energies
            .iter()
            .zip(initial_energies.iter())
            .zip(pitch_angle_cosines.iter())
            .zip(electron_numbers_per_dist.iter())
            .enumerate()
            .map(
                |(
                    idx,
                    (((&energy, &initial_energy), &pitch_angle_cos), &electron_number_per_dist),
                )| {
                    if initial_energy <= 0.0
                        || pitch_angle_cos <= 0.0
                        || electron_number_per_dist <= 0.0
                    {
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
                            * electron_number_per_dist
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

        // Integrate deposited power per distance over initial energies using
        // the trapezoidal method
        integrate_trapezoidal(
            valid_initial_energies,
            valid_deposited_power_initial_energy_derivs,
        )
    }

    fn compute_total_electron_flux_over_cross_section(
        &self,
        energies: &[feb],
        initial_energies: &[feb],
        pitch_angle_cosines: &[feb],
        electron_numbers_per_dist: &[feb],
    ) -> feb {
        assert_eq!(initial_energies.len(), energies.len());
        assert_eq!(pitch_angle_cosines.len(), energies.len());
        assert_eq!(electron_numbers_per_dist.len(), energies.len());

        let mut first_nonzero_idx = None;

        let flux_initial_energy_derivs: Vec<_> = energies
            .iter()
            .zip(initial_energies.iter())
            .zip(pitch_angle_cosines.iter())
            .zip(electron_numbers_per_dist.iter())
            .enumerate()
            .map(
                |(
                    idx,
                    (((&energy, &initial_energy), &pitch_angle_cos), &electron_number_per_dist),
                )| {
                    if initial_energy <= 0.0
                        || pitch_angle_cos <= 0.0
                        || electron_number_per_dist <= 0.0
                    {
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
                            * electron_number_per_dist
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
        total_electron_flux_density: feb,
    ) -> feb {
        Q_ELECTRON
            * compute_parallel_resistivity(
                self.hybrid_coulomb_log.coulomb_log(),
                temperature,
                self.hybrid_coulomb_log.hydrogen_ionization_fraction(),
            )
            * total_electron_flux_density
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
        let ColumnDepthDerivatives {
            energy: energy_col_depth_deriv_1,
            pitch_angle: pitch_angle_cos_col_depth_deriv_1,
            number_density: number_density_col_depth_deriv_1,
        } = self.compute_col_depth_derivs(
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

        let ColumnDepthDerivatives {
            energy: energy_col_depth_deriv_2,
            pitch_angle: pitch_angle_cos_col_depth_deriv_2,
            number_density: number_density_col_depth_deriv_2,
        } = self.compute_col_depth_derivs(energy_1, pitch_angle_cos_1, number_density_1);

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

    fn advance_quantities_with_third_order_heun(
        &self,
        initial_energy: feb,
        initial_pitch_angle_cos: feb,
        initial_number_density: feb,
        col_depth_increase: feb,
    ) -> TransportResult {
        let ColumnDepthDerivatives {
            energy: energy_col_depth_deriv_1,
            pitch_angle: pitch_angle_cos_col_depth_deriv_1,
            number_density: number_density_col_depth_deriv_1,
        } = self.compute_col_depth_derivs(
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

        let ColumnDepthDerivatives {
            energy: energy_col_depth_deriv_2,
            pitch_angle: pitch_angle_cos_col_depth_deriv_2,
            number_density: number_density_col_depth_deriv_2,
        } = self.compute_col_depth_derivs(energy_1, pitch_angle_cos_1, number_density_1);

        let energy_2 = initial_energy + energy_col_depth_deriv_2 * col_depth_increase * 2.0 / 3.0;
        let pitch_angle_cos_2 = initial_pitch_angle_cos
            + pitch_angle_cos_col_depth_deriv_2 * col_depth_increase * 2.0 / 3.0;
        let number_density_2 = initial_number_density
            + number_density_col_depth_deriv_2 * col_depth_increase * 2.0 / 3.0;

        if energy_2 <= 0.0 || pitch_angle_cos_2 <= 0.0 {
            return TransportResult::Thermalized;
        }

        let ColumnDepthDerivatives {
            energy: energy_col_depth_deriv_3,
            pitch_angle: pitch_angle_cos_col_depth_deriv_3,
            number_density: number_density_col_depth_deriv_3,
        } = self.compute_col_depth_derivs(energy_2, pitch_angle_cos_2, number_density_2);

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

    fn compute_col_depth_derivs(
        &self,
        energy: feb,
        pitch_angle_cos: feb,
        number_density: feb,
    ) -> ColumnDepthDerivatives {
        let EvaluatedHydrogenCoulombLogarithms {
            for_energy: hybrid_coulomb_log_for_energy,
            for_pitch_angle: hybrid_coulomb_log_for_pitch_angle,
            for_number_density: hybrid_coulomb_log_for_number_density,
        } = self.hybrid_coulomb_log.evaluate(energy);

        ColumnDepthDerivatives {
            energy: self.compute_energy_col_depth_deriv_with_hybrid_coulomb_log(
                energy,
                pitch_angle_cos,
                hybrid_coulomb_log_for_energy,
            ),
            pitch_angle: self.compute_pitch_angle_cos_col_depth_deriv_with_hybrid_coulomb_log(
                energy,
                pitch_angle_cos,
                hybrid_coulomb_log_for_pitch_angle,
            ),
            number_density: self.compute_number_density_col_depth_deriv_with_hybrid_coulomb_log(
                energy,
                pitch_angle_cos,
                number_density,
                hybrid_coulomb_log_for_number_density,
            ),
        }
    }

    fn compute_energy_col_depth_deriv(&self, energy: feb, pitch_angle_cos: feb) -> feb {
        self.compute_energy_col_depth_deriv_with_hybrid_coulomb_log(
            energy,
            pitch_angle_cos,
            self.hybrid_coulomb_log.for_energy(energy),
        )
    }

    fn compute_energy_col_depth_deriv_with_hybrid_coulomb_log(
        &self,
        energy: feb,
        pitch_angle_cos: feb,
        hybrid_coulomb_log_for_energy: feb,
    ) -> feb {
        -COLLISION_SCALE * hybrid_coulomb_log_for_energy / (pitch_angle_cos * energy)
            - Q_ELECTRON * self.total_trajectory_aligned_electric_field
                / self.total_hydrogen_density
    }

    fn compute_pitch_angle_cos_col_depth_deriv_with_hybrid_coulomb_log(
        &self,
        energy: feb,
        pitch_angle_cos: feb,
        hybrid_coulomb_log_for_pitch_angle: feb,
    ) -> feb {
        -COLLISION_SCALE * hybrid_coulomb_log_for_pitch_angle / (2.0 * energy * energy)
            - (self.log_magnetic_field_col_depth_deriv
                + Q_ELECTRON * self.total_trajectory_aligned_electric_field
                    / (self.total_hydrogen_density * energy))
                * (1.0 - pitch_angle_cos * pitch_angle_cos)
                / (2.0 * pitch_angle_cos)
    }

    fn compute_number_density_col_depth_deriv_with_hybrid_coulomb_log(
        &self,
        energy: feb,
        pitch_angle_cos: feb,
        number_density: feb,
        hybrid_coulomb_log_for_number_density: feb,
    ) -> feb {
        -COLLISION_SCALE * hybrid_coulomb_log_for_number_density * number_density
            / (2.0 * pitch_angle_cos * energy * energy)
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
