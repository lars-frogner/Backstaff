//!

use super::atmosphere::HybridCoulombLogarithm;
use crate::{
    constants::{PI, Q_ELECTRON},
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
    ) -> (feb, feb) {
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

    fn advance_energy_and_pitch_angle_cos_with_second_order_heun(
        &self,
        initial_energy: feb,
        initial_pitch_angle_cos: feb,
        col_depth_increase: feb,
    ) -> (feb, feb) {
        let energy_col_depth_deriv_1 =
            self.compute_energy_col_depth_deriv(initial_energy, initial_pitch_angle_cos);
        let pitch_angle_cos_col_depth_deriv_1 =
            self.compute_pitch_angle_cos_col_depth_deriv(initial_energy, initial_pitch_angle_cos);

        let energy_1 = initial_energy + energy_col_depth_deriv_1 * col_depth_increase;
        let pitch_angle_cos_1 =
            initial_pitch_angle_cos + pitch_angle_cos_col_depth_deriv_1 * col_depth_increase;

        if energy_1 <= 0.0 || pitch_angle_cos_1 <= 0.0 {
            return (0.0, 0.0);
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
            (0.0, 0.0)
        } else {
            (energy, pitch_angle_cos)
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
    ) -> Option<(feb, feb)> {
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
        let beta_not_2 = feb::min(1.9999, beta);

        let offset = COLLISION_SCALE * hybrid_coulomb_log.for_energy() * col_depth_increase
            / (initial_pitch_angle_cos * initial_energy * initial_energy)
            - (1.0 - 0.125 * magnetic_field_factor - electric_field_factor / 6.0)
                / (2.0 + 0.5 * beta);

        if offset >= 0.0 {
            // The electron will thermalize
            return Some((0.0, 0.0));
        }

        let f = |ln_x| {
            let x = feb::exp(ln_x);
            x.powf(2.0 + 0.5 * beta)
                * (1.0
                    - magnetic_field_factor / (4.0 - beta)
                    - electric_field_factor / (2.0 - beta_not_2))
                / (2.0 + 0.5 * beta)
                + x * x * x * x * magnetic_field_factor / (4.0 * (4.0 - beta))
                + x * x * x * electric_field_factor / (3.0 * (2.0 - beta_not_2))
                + offset
        };

        let dfdlnx = |ln_x| {
            let x = feb::exp(ln_x);
            x * (x.powf(1.0 + 0.5 * beta)
                * (1.0
                    - magnetic_field_factor / (4.0 - beta)
                    - electric_field_factor / (2.0 - beta_not_2))
                + x * x * x * magnetic_field_factor / (4.0 - beta)
                + x * x * electric_field_factor / (2.0 - beta_not_2))
        };

        let mut convergency = SimpleConvergency {
            eps: 1e-5,
            max_iter: 30,
        };
        match roots::find_root_newton_raphson(0.0, &f, &dfdlnx, &mut convergency) {
            Ok(ln_x) => {
                let energy = initial_energy * feb::exp(ln_x);

                let pitch_angle_cos = (energy / initial_energy).powf(0.5 * beta)
                    * (initial_pitch_angle_cos
                        - initial_energy * initial_energy * self.magnetic_field_scale
                            / (4.0 - beta)
                        - initial_energy * self.electric_field_scale / (2.0 - beta_not_2))
                    + energy * energy * self.magnetic_field_scale / (4.0 - beta)
                    + energy * self.electric_field_scale / (2.0 - beta_not_2);

                Some((energy, pitch_angle_cos))
            }
            Err(error) => {
                eprintln!(
                    "Convergence failed when advancing energy and pitch angle analytically: {}",
                    error
                );
                Some((0.0, 0.0))
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
            &hybrid_coulomb_log,
            log_magnetic_field_col_depth_deriv,
        );
        self.electric_field_scale = Self::compute_electric_field_scale(
            &hybrid_coulomb_log,
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
