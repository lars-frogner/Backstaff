//!

use super::atmosphere::HybridCoulombLogarithm;
use crate::{
    constants::{M_ELECTRON, PI, Q_ELECTRON},
    ebeam::feb,
};
use roots::{self, SimpleConvergency};

const COLLISION_SCALE: feb = 2.0 * PI * Q_ELECTRON * Q_ELECTRON * Q_ELECTRON * Q_ELECTRON;

#[derive(Clone, Debug)]
pub struct Transporter {
    analytical_transporter: AnalyticalTransporter,
    hybrid_coulomb_log: HybridCoulombLogarithm,
    total_hydrogen_density: feb,
    log_magnetic_field_col_depth_deriv: feb,
    electric_field: feb,
}

#[derive(Clone, Debug)]
struct AnalyticalTransporter {
    config: AnalyticalTransporterConfig,
    magnetic_field_scale: feb,
    electric_field_scale: feb,
}

#[derive(Clone, Debug)]
pub struct AnalyticalTransporterConfig {
    pub max_electric_field_ratio: feb,
    pub max_pitch_angle_cos: feb,
    pub max_magnetic_field_factor: feb,
    pub max_electric_field_factor: feb,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum TransportResult {
    NewEnergyAndPitchAngleCos((feb, feb)),
    Thermalized,
}

impl Transporter {
    pub fn new(
        config: AnalyticalTransporterConfig,
        hybrid_coulomb_log: HybridCoulombLogarithm,
        total_hydrogen_density: feb,
        log_magnetic_field_col_depth_deriv: feb,
        electric_field: feb,
    ) -> Self {
        let analytical_advancer = AnalyticalTransporter::new(
            config,
            &hybrid_coulomb_log,
            total_hydrogen_density,
            log_magnetic_field_col_depth_deriv,
            electric_field,
        );
        Self {
            analytical_transporter: analytical_advancer,
            hybrid_coulomb_log,
            total_hydrogen_density,
            log_magnetic_field_col_depth_deriv,
            electric_field,
        }
    }

    pub fn update_conditions(
        &mut self,
        hybrid_coulomb_log: HybridCoulombLogarithm,
        total_hydrogen_density: feb,
        log_magnetic_field_col_depth_deriv: feb,
        electric_field: feb,
    ) {
        self.analytical_transporter.update_conditions(
            &hybrid_coulomb_log,
            total_hydrogen_density,
            log_magnetic_field_col_depth_deriv,
            electric_field,
        );
        self.hybrid_coulomb_log = hybrid_coulomb_log;
        self.total_hydrogen_density = total_hydrogen_density;
        self.log_magnetic_field_col_depth_deriv = log_magnetic_field_col_depth_deriv;
        self.electric_field = electric_field;
    }

    pub fn advance_energy_and_pitch_angle_cos(
        &self,
        initial_energy: feb,
        initial_pitch_angle_cos: feb,
        col_depth_increase: feb,
    ) -> TransportResult {
        if initial_pitch_angle_cos <= 0.0 {
            TransportResult::Thermalized
        } else {
            self.analytical_transporter
                .try_advancing_energy_and_pitch_angle_cos(
                    &self.hybrid_coulomb_log,
                    initial_energy,
                    initial_pitch_angle_cos,
                    col_depth_increase,
                )
                .unwrap_or_else(|| {
                    self.advance_energy_and_pitch_angle_cos_with_second_order_heun(
                        initial_energy,
                        initial_pitch_angle_cos,
                        col_depth_increase,
                    )
                })
        }
    }

    pub fn advance_number_densities(
        &self,
        energies: &[feb],
        pitch_angle_cosines: &[feb],
        number_densities: &mut [feb],
        col_depth_increase: feb,
    ) {
        energies
            .iter()
            .zip(pitch_angle_cosines.iter())
            .zip(number_densities.iter_mut())
            .for_each(|((&energy, &pitch_angle_cos), number_density)| {
                let number_density_col_depth_deriv_per_number_density = self
                    .compute_number_density_col_depth_deriv_per_number_density(
                        energy,
                        pitch_angle_cos,
                    );
                *number_density *=
                    1.0 + number_density_col_depth_deriv_per_number_density * col_depth_increase;
            });
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

        let deposited_power_initial_energy_derivs: Vec<_> = energies
            .iter()
            .zip(initial_energies.iter())
            .zip(pitch_angle_cosines.iter())
            .zip(number_densities.iter())
            .map(
                |(((&energy, &initial_energy), &pitch_angle_cos), &number_density)| {
                    if initial_energy <= 0.0 || pitch_angle_cos <= 0.0 || number_density <= 0.0 {
                        0.0
                    } else {
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

        // Integrate deposited power over initial energies using the trapezoidal method
        let deposited_power = 0.5
            * initial_energies
                .iter()
                .zip(initial_energies.iter().skip(1))
                .zip(
                    deposited_power_initial_energy_derivs
                        .iter()
                        .zip(deposited_power_initial_energy_derivs.iter().skip(1)),
                )
                .fold(
                    0.0,
                    |acc, ((&initial_energy, &initial_energy_up), (&deriv, &deriv_up))| {
                        acc + (deriv + deriv_up) * (initial_energy_up - initial_energy)
                    },
                );

        deposited_power
    }

    fn advance_energy_and_pitch_angle_cos_with_second_order_heun(
        &self,
        initial_energy: feb,
        initial_pitch_angle_cos: feb,
        col_depth_increase: feb,
    ) -> TransportResult {
        let energy_col_depth_deriv_1 =
            self.compute_energy_col_depth_deriv(initial_energy, initial_pitch_angle_cos);
        let pitch_angle_cos_col_depth_deriv_1 =
            self.compute_pitch_angle_cos_col_depth_deriv(initial_energy, initial_pitch_angle_cos);

        let energy_1 = initial_energy + energy_col_depth_deriv_1 * col_depth_increase;
        let pitch_angle_cos_1 =
            initial_pitch_angle_cos + pitch_angle_cos_col_depth_deriv_1 * col_depth_increase;

        if energy_1 <= 0.0 || pitch_angle_cos_1 <= 0.0 {
            return TransportResult::Thermalized;
        }

        let energy_col_depth_deriv_2 =
            self.compute_energy_col_depth_deriv(energy_1, pitch_angle_cos_1);
        let pitch_angle_cos_col_depth_deriv_2 =
            self.compute_pitch_angle_cos_col_depth_deriv(energy_1, pitch_angle_cos_1);

        let energy = initial_energy
            + 0.5 * (energy_col_depth_deriv_1 + energy_col_depth_deriv_2) * col_depth_increase;
        let pitch_angle_cos = initial_pitch_angle_cos
            + 0.5
                * (pitch_angle_cos_col_depth_deriv_1 + pitch_angle_cos_col_depth_deriv_2)
                * col_depth_increase;

        if energy <= 0.0 || pitch_angle_cos <= 0.0 {
            TransportResult::Thermalized
        } else {
            TransportResult::NewEnergyAndPitchAngleCos((energy, pitch_angle_cos))
        }
    }

    fn compute_energy_col_depth_deriv(&self, energy: feb, pitch_angle_cos: feb) -> feb {
        -COLLISION_SCALE * self.hybrid_coulomb_log.for_energy() / (pitch_angle_cos * energy)
            - Q_ELECTRON * self.electric_field / self.total_hydrogen_density
    }

    fn compute_pitch_angle_cos_col_depth_deriv(&self, energy: feb, pitch_angle_cos: feb) -> feb {
        -COLLISION_SCALE * self.hybrid_coulomb_log.for_pitch_angle() / (2.0 * energy * energy)
            - (self.log_magnetic_field_col_depth_deriv
                + Q_ELECTRON * self.electric_field / (self.total_hydrogen_density * energy))
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
}

impl AnalyticalTransporter {
    fn new(
        config: AnalyticalTransporterConfig,
        hybrid_coulomb_log: &HybridCoulombLogarithm,
        total_hydrogen_density: feb,
        log_magnetic_field_col_depth_deriv: feb,
        electric_field: feb,
    ) -> Self {
        let magnetic_field_scale = Self::compute_magnetic_field_scale(
            hybrid_coulomb_log,
            log_magnetic_field_col_depth_deriv,
        );
        let electric_field_scale = Self::compute_electric_field_scale(
            hybrid_coulomb_log,
            total_hydrogen_density,
            electric_field,
        );
        Self {
            config,
            magnetic_field_scale,
            electric_field_scale,
        }
    }

    fn try_advancing_energy_and_pitch_angle_cos(
        &self,
        hybrid_coulomb_log: &HybridCoulombLogarithm,
        initial_energy: feb,
        initial_pitch_angle_cos: feb,
        col_depth_increase: feb,
    ) -> Option<TransportResult> {
        let magnetic_field_factor =
            self.magnetic_field_scale * initial_energy * initial_energy / initial_pitch_angle_cos;
        let electric_field_factor =
            self.electric_field_scale * initial_energy / initial_pitch_angle_cos;

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
            self.try_advancing_energy_and_pitch_angle_cos_ionized(
                hybrid_coulomb_log,
                initial_energy,
                initial_pitch_angle_cos,
                col_depth_increase,
                magnetic_field_factor,
                electric_field_factor,
            )
        } else {
            self.try_advancing_energy_and_pitch_angle_cos_unionized(
                hybrid_coulomb_log,
                initial_energy,
                initial_pitch_angle_cos,
                col_depth_increase,
                magnetic_field_factor,
                electric_field_factor,
            )
        }
    }

    fn try_advancing_energy_and_pitch_angle_cos_ionized(
        &self,
        hybrid_coulomb_log: &HybridCoulombLogarithm,
        initial_energy: feb,
        initial_pitch_angle_cos: feb,
        col_depth_increase: feb,
        magnetic_field_factor: feb,
        electric_field_factor: feb,
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
            eps: 3e-4,
            max_iter: 30,
        };
        match roots::find_root_newton_raphson(0.0, &f, &dfdlnx, &mut convergency) {
            Ok(ln_x) => {
                let x = feb::exp(ln_x);
                let energy = initial_energy * x;

                let pitch_angle_cos = initial_pitch_angle_cos * x
                    + 0.5 * self.magnetic_field_scale * energy * energy * (1.0 - 1.0 / x)
                    + self.electric_field_scale * energy * ln_x;

                Some(TransportResult::NewEnergyAndPitchAngleCos((
                    energy,
                    pitch_angle_cos,
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

    fn try_advancing_energy_and_pitch_angle_cos_unionized(
        &self,
        hybrid_coulomb_log: &HybridCoulombLogarithm,
        initial_energy: feb,
        initial_pitch_angle_cos: feb,
        col_depth_increase: feb,
        magnetic_field_factor: feb,
        electric_field_factor: feb,
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
            eps: 3e-4,
            max_iter: 30,
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

                Some(TransportResult::NewEnergyAndPitchAngleCos((
                    energy,
                    pitch_angle_cos,
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
        log_magnetic_field_col_depth_deriv: feb,
        electric_field: feb,
    ) {
        self.magnetic_field_scale = Self::compute_magnetic_field_scale(
            hybrid_coulomb_log,
            log_magnetic_field_col_depth_deriv,
        );
        self.electric_field_scale = Self::compute_electric_field_scale(
            hybrid_coulomb_log,
            total_hydrogen_density,
            electric_field,
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
        electric_field: feb,
    ) -> feb {
        Q_ELECTRON * electric_field
            / (COLLISION_SCALE * hybrid_coulomb_log.for_energy() * total_hydrogen_density)
    }
}

impl Default for AnalyticalTransporterConfig {
    fn default() -> Self {
        Self {
            max_electric_field_ratio: 1e-3,
            max_pitch_angle_cos: 1e-1,
            max_magnetic_field_factor: 1e-2,
            max_electric_field_factor: 1e-2,
        }
    }
}
