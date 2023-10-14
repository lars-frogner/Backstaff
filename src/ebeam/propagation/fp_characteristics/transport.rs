//!

use super::atmosphere::{
    compute_parallel_resistivity, EvaluatedHydrogenCoulombLogarithms,
    EvaluatedHydrogenCoulombLogarithmsForEnergyAndPitchAngle, HybridCoulombLogarithm,
};
use crate::{
    constants::{CLIGHT, KEV_TO_ERG, M_ELECTRON, PI, Q_ELECTRON},
    ebeam::feb,
    plasma::ionization::Abundances,
};

const COLLISION_SCALE: feb = 2.0 * PI * Q_ELECTRON * Q_ELECTRON * Q_ELECTRON * Q_ELECTRON;

const GYROMAGNETIC_RADIATION_SCALE_E: feb = 2.0 * Q_ELECTRON * Q_ELECTRON * Q_ELECTRON * Q_ELECTRON
    / (3.0 * CLIGHT * CLIGHT * CLIGHT * CLIGHT * CLIGHT * M_ELECTRON * M_ELECTRON);
const GYROMAGNETIC_RADIATION_SCALE_MU: feb = GYROMAGNETIC_RADIATION_SCALE_E / M_ELECTRON;

const THERMALIZATION_ENERGY: feb = 0.0 * 0.01 * KEV_TO_ERG;
const THERMALIZATION_PITCH_ANGLE_COS: feb = 0.0 * 0.01;

#[derive(Clone, Debug)]
pub struct Transporter {
    include_ambient_electric_field: bool,
    include_induced_electric_field: bool,
    include_magnetic_field: bool,
    include_gyromagnetic_radiation: bool,
    hybrid_coulomb_log: HybridCoulombLogarithm,
    temperature: feb,
    parallel_electron_flux_over_cross_section: feb,
    ambient_trajectory_aligned_electric_field: feb,
    induced_trajectory_aligned_electric_field: feb,
    total_trajectory_aligned_electric_field: feb,
    magnetic_field_strength: feb,
    squared_magnetic_field_strength_for_radiation: feb,
    resistivity: feb,
    return_current_heating_power_per_dist: feb,
    log_magnetic_field_col_depth_deriv: feb,
    energy_loss_to_electric_field: feb,
    speed_loss_to_gyromagnetic_radiation: feb,
    high_energy_pitch_angle_cos: feb,
    high_energy_pitch_angle_cos_perturbed: feb,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum TransportResult {
    NewValues((feb, feb, feb)),
    Thermalized,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum TransportResultForEnergyAndPitchAngle {
    NewValues((feb, feb)),
    Thermalized,
}

#[derive(Clone, Debug)]
struct ColumnDepthDerivatives {
    energy: feb,
    pitch_angle: feb,
    flux_spectrum: feb,
}

#[derive(Clone, Debug)]
struct ColumnDepthDerivativesForEnergyAndPitchAngle {
    energy: feb,
    pitch_angle: feb,
}

impl Transporter {
    const MIN_MAGNETIC_FIELD: feb = 0.1;

    pub fn new(
        include_ambient_electric_field: bool,
        include_induced_electric_field: bool,
        include_magnetic_field: bool,
        include_gyromagnetic_radiation: bool,
        parallel_electron_flux_over_cross_section: feb,
        initial_pitch_angle_cos: feb,
        initial_pitch_angle_cos_perturbed: feb,
        hybrid_coulomb_log: HybridCoulombLogarithm,
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

        let squared_magnetic_field_strength_for_radiation = if include_gyromagnetic_radiation {
            magnetic_field_strength * magnetic_field_strength
        } else {
            0.0
        };

        let ambient_trajectory_aligned_electric_field = if include_ambient_electric_field {
            ambient_trajectory_aligned_electric_field
        } else {
            0.0
        };
        let induced_trajectory_aligned_electric_field = 0.0;
        let total_trajectory_aligned_electric_field =
            ambient_trajectory_aligned_electric_field + induced_trajectory_aligned_electric_field;

        let energy_loss_to_electric_field = 0.0;
        let speed_loss_to_gyromagnetic_radiation = 0.0;
        let high_energy_pitch_angle_cos = initial_pitch_angle_cos;
        let high_energy_pitch_angle_cos_perturbed = initial_pitch_angle_cos_perturbed;

        let resistivity =
            compute_parallel_resistivity(temperature, hybrid_coulomb_log.abundances());

        let return_current_heating_power_per_dist = 0.0;

        Self {
            include_ambient_electric_field,
            include_induced_electric_field,
            include_magnetic_field,
            include_gyromagnetic_radiation,
            hybrid_coulomb_log,
            temperature,
            parallel_electron_flux_over_cross_section,
            ambient_trajectory_aligned_electric_field,
            induced_trajectory_aligned_electric_field,
            total_trajectory_aligned_electric_field,
            magnetic_field_strength,
            squared_magnetic_field_strength_for_radiation,
            resistivity,
            return_current_heating_power_per_dist,
            log_magnetic_field_col_depth_deriv,
            energy_loss_to_electric_field,
            speed_loss_to_gyromagnetic_radiation,
            high_energy_pitch_angle_cos,
            high_energy_pitch_angle_cos_perturbed,
        }
    }

    pub fn parallel_electron_flux_over_cross_section(&self) -> feb {
        self.parallel_electron_flux_over_cross_section
    }

    pub fn induced_trajectory_aligned_electric_field(&self) -> feb {
        self.induced_trajectory_aligned_electric_field
    }

    pub fn return_current_heating_power_per_dist(&self) -> feb {
        self.return_current_heating_power_per_dist
    }

    pub fn hybrid_coulomb_log(&self) -> &HybridCoulombLogarithm {
        &self.hybrid_coulomb_log
    }

    pub fn abundances(&self) -> &Abundances {
        self.hybrid_coulomb_log.abundances()
    }

    pub fn log_magnetic_field_distance_deriv(&self) -> feb {
        self.log_magnetic_field_col_depth_deriv * self.abundances().total_hydrogen_density()
    }

    pub fn energy_without_loss_to_electric_field(&self, energy: feb) -> feb {
        energy + self.energy_loss_to_electric_field
    }

    pub fn energy_without_loss_to_gyromagnetic_radiation(&self, energy: feb) -> feb {
        0.5 * M_ELECTRON
            * (feb::sqrt(2.0 * energy / M_ELECTRON) + self.speed_loss_to_gyromagnetic_radiation)
                .powi(2)
    }

    pub fn high_energy_pitch_angle_cos(&self) -> feb {
        self.high_energy_pitch_angle_cos
    }

    pub fn high_energy_pitch_angle_cos_perturbed(&self) -> feb {
        self.high_energy_pitch_angle_cos_perturbed
    }

    pub fn update_conditions(
        &mut self,
        hybrid_coulomb_log: HybridCoulombLogarithm,
        temperature: feb,
        ambient_trajectory_aligned_electric_field: feb,
        magnetic_field_strength: feb,
        energies: &[feb],
        pitch_angle_cosines: &[feb],
        area_weighted_flux_spectrum: &[feb],
        jacobians: &[feb],
        beam_cross_sectional_area: feb,
        col_depth_increase: feb,
    ) {
        self.hybrid_coulomb_log = hybrid_coulomb_log;
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
            Self::update_high_energy_pitch_angle_cos(
                &mut self.high_energy_pitch_angle_cos_perturbed,
                self.log_magnetic_field_col_depth_deriv,
                col_depth_increase,
            );
        } else {
            self.magnetic_field_strength = 0.0;
            self.log_magnetic_field_col_depth_deriv = 0.0;
        }

        self.squared_magnetic_field_strength_for_radiation = if self.include_gyromagnetic_radiation
        {
            magnetic_field_strength * magnetic_field_strength
        } else {
            0.0
        };

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

        self.parallel_electron_flux_over_cross_section = self
            .compute_parallel_electron_flux_over_cross_section(
                energies,
                pitch_angle_cosines,
                area_weighted_flux_spectrum,
                jacobians,
            );

        if self.include_induced_electric_field {
            let parallel_electron_flux =
                self.parallel_electron_flux_over_cross_section / beam_cross_sectional_area;

            self.resistivity = compute_parallel_resistivity(temperature, self.abundances());

            // Because the induced field depends on the flux, which depends
            // on the energy derivative, which depends on the electric field,
            // determining the induced field is strictly an implicit problem.
            // We handle this by keeping the old value for the electric field
            // when computing the flux, assuming that the delay of one step does
            // not make a significant difference
            self.induced_trajectory_aligned_electric_field =
                Self::compute_induced_trajectory_aligned_electric_field(
                    self.resistivity,
                    parallel_electron_flux,
                );

            self.return_current_heating_power_per_dist =
                Self::compute_resistive_heating_power_density(
                    self.resistivity,
                    parallel_electron_flux,
                ) * beam_cross_sectional_area;
        } else {
            self.induced_trajectory_aligned_electric_field = 0.0;
            self.return_current_heating_power_per_dist = 0.0;
        }

        if self.include_ambient_electric_field || self.include_induced_electric_field {
            self.total_trajectory_aligned_electric_field = self
                .ambient_trajectory_aligned_electric_field
                + self.induced_trajectory_aligned_electric_field;

            Self::update_energy_loss_to_electric_field(
                &mut self.energy_loss_to_electric_field,
                self.total_trajectory_aligned_electric_field,
                self.hybrid_coulomb_log
                    .abundances()
                    .total_hydrogen_density(),
                col_depth_increase,
            );
        } else {
            self.total_trajectory_aligned_electric_field = 0.0;
        }

        if self.include_gyromagnetic_radiation {
            Self::update_speed_loss_to_gyromagnetic_radiation(
                &mut self.speed_loss_to_gyromagnetic_radiation,
                self.squared_magnetic_field_strength_for_radiation,
                self.high_energy_pitch_angle_cos,
                self.hybrid_coulomb_log
                    .abundances()
                    .total_hydrogen_density(),
                col_depth_increase,
            );
        }
    }

    pub fn advance_quantities(
        &self,
        initial_energy: feb,
        initial_pitch_angle_cos: feb,
        initial_electron_flux: feb,
        col_depth_increase: feb,
    ) -> TransportResult {
        if initial_pitch_angle_cos <= THERMALIZATION_PITCH_ANGLE_COS {
            TransportResult::Thermalized
        } else {
            self.advance_quantities_with_third_order_heun(
                initial_energy,
                initial_pitch_angle_cos,
                initial_electron_flux,
                col_depth_increase,
            )
        }
    }

    pub fn advance_energy_and_pitch_angle_cos(
        &self,
        initial_energy: feb,
        initial_pitch_angle_cos: feb,
        col_depth_increase: feb,
    ) -> TransportResultForEnergyAndPitchAngle {
        if initial_pitch_angle_cos <= THERMALIZATION_PITCH_ANGLE_COS {
            TransportResultForEnergyAndPitchAngle::Thermalized
        } else {
            self.advance_energy_and_pitch_angle_cos_with_third_order_heun(
                initial_energy,
                initial_pitch_angle_cos,
                col_depth_increase,
            )
        }
    }

    pub fn compute_deposited_power_per_dist(
        &self,
        energies: &[feb],
        pitch_angle_cosines: &[feb],
        area_weighted_flux_spectrum: &[feb],
        jacobians: &[feb],
    ) -> feb {
        assert_eq!(pitch_angle_cosines.len(), energies.len());
        assert_eq!(area_weighted_flux_spectrum.len(), energies.len());
        assert_eq!(jacobians.len(), energies.len());

        let mut first_nonzero_idx = None;

        let deposited_power_energy_derivs: Vec<_> = energies
            .iter()
            .zip(pitch_angle_cosines.iter())
            .zip(area_weighted_flux_spectrum.iter())
            .zip(jacobians.iter())
            .enumerate()
            .map(
                |(idx, (((&energy, &pitch_angle_cos), &area_weighted_flux), &jacobian))| {
                    if pitch_angle_cos <= THERMALIZATION_PITCH_ANGLE_COS
                        || area_weighted_flux <= 0.0
                    {
                        0.0
                    } else {
                        if first_nonzero_idx.is_none() {
                            first_nonzero_idx = Some(idx);
                        }

                        -self.compute_collisional_energy_time_deriv(energy)
                            * area_weighted_flux
                            * jacobian
                            / (pitch_angle_cos * feb::sqrt(2.0 * energy / M_ELECTRON))
                    }
                },
            )
            .collect();

        let first_nonzero_idx = first_nonzero_idx.unwrap_or(0);

        let valid_energies = &energies[first_nonzero_idx..];
        let valid_deposited_power_energy_derivs =
            &deposited_power_energy_derivs[first_nonzero_idx..];

        // Integrate deposited power per distance over energies using the
        // trapezoidal method
        let collisional_deposited_power_per_dist =
            integrate_trapezoidal(valid_energies, valid_deposited_power_energy_derivs);

        collisional_deposited_power_per_dist + self.return_current_heating_power_per_dist
    }

    fn compute_parallel_electron_flux_over_cross_section(
        &self,
        energies: &[feb],
        pitch_angle_cosines: &[feb],
        area_weighted_flux_spectrum: &[feb],
        jacobians: &[feb],
    ) -> feb {
        assert_eq!(pitch_angle_cosines.len(), energies.len());
        assert_eq!(area_weighted_flux_spectrum.len(), energies.len());
        assert_eq!(jacobians.len(), energies.len());

        let mut first_nonzero_idx = None;

        let flux_energy_derivs: Vec<_> = area_weighted_flux_spectrum
            .iter()
            .zip(jacobians.iter())
            .enumerate()
            .map(|(idx, (&area_weighted_flux, &jacobian))| {
                if area_weighted_flux <= 0.0 {
                    0.0
                } else {
                    if first_nonzero_idx.is_none() {
                        first_nonzero_idx = Some(idx);
                    }

                    area_weighted_flux * jacobian
                }
            })
            .collect();

        let first_nonzero_idx = first_nonzero_idx.unwrap_or(0);

        let valid_energies = &energies[first_nonzero_idx..];
        let valid_flux_energy_derivs = &flux_energy_derivs[first_nonzero_idx..];

        integrate_trapezoidal(valid_energies, valid_flux_energy_derivs)
    }

    fn compute_induced_trajectory_aligned_electric_field(
        resistivity: feb,
        parallel_electron_flux: feb,
    ) -> feb {
        Q_ELECTRON * resistivity * parallel_electron_flux
    }

    fn compute_resistive_heating_power_density(
        resistivity: feb,
        parallel_electron_flux: feb,
    ) -> feb {
        resistivity * (Q_ELECTRON * parallel_electron_flux).powi(2)
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

    fn update_speed_loss_to_gyromagnetic_radiation(
        speed_loss_to_gyromagnetic_radiation: &mut feb,
        squared_magnetic_field_strength: feb,
        pitch_angle_cos: feb,
        total_hydrogen_density: feb,
        col_depth_increase: feb,
    ) {
        *speed_loss_to_gyromagnetic_radiation += (GYROMAGNETIC_RADIATION_SCALE_MU
            * (1.0 - pitch_angle_cos * pitch_angle_cos)
            * squared_magnetic_field_strength
            / (pitch_angle_cos * total_hydrogen_density))
            * col_depth_increase;
    }

    fn update_high_energy_pitch_angle_cos(
        high_energy_pitch_angle_cos: &mut feb,
        log_magnetic_field_col_depth_deriv: feb,
        col_depth_increase: feb,
    ) {
        *high_energy_pitch_angle_cos = feb::sqrt(feb::max(
            THERMALIZATION_PITCH_ANGLE_COS,
            feb::min(
                1.0,
                1.0 - (1.0 - (*high_energy_pitch_angle_cos) * (*high_energy_pitch_angle_cos))
                    * feb::exp(log_magnetic_field_col_depth_deriv * col_depth_increase),
            ),
        ));
    }

    #[allow(dead_code)]
    fn advance_quantities_with_second_order_heun(
        &self,
        start_energy: feb,
        start_pitch_angle_cos: feb,
        start_electron_flux: feb,
        col_depth_increase: feb,
    ) -> TransportResult {
        let ColumnDepthDerivatives {
            energy: energy_col_depth_deriv_1,
            pitch_angle: pitch_angle_cos_col_depth_deriv_1,
            flux_spectrum: flux_spectrum_col_depth_deriv_1,
        } = self.compute_col_depth_derivs(start_energy, start_pitch_angle_cos, start_electron_flux);

        let energy_1 = start_energy + energy_col_depth_deriv_1 * col_depth_increase;
        let pitch_angle_cos_1 =
            start_pitch_angle_cos + pitch_angle_cos_col_depth_deriv_1 * col_depth_increase;
        let electron_flux_1 =
            start_electron_flux + flux_spectrum_col_depth_deriv_1 * col_depth_increase;

        if energy_1 <= THERMALIZATION_ENERGY || pitch_angle_cos_1 <= THERMALIZATION_PITCH_ANGLE_COS
        {
            return TransportResult::Thermalized;
        }

        let ColumnDepthDerivatives {
            energy: energy_col_depth_deriv_2,
            pitch_angle: pitch_angle_cos_col_depth_deriv_2,
            flux_spectrum: flux_spectrum_col_depth_deriv_2,
        } = self.compute_col_depth_derivs(energy_1, pitch_angle_cos_1, electron_flux_1);

        let energy = start_energy
            + 0.5 * (energy_col_depth_deriv_1 + energy_col_depth_deriv_2) * col_depth_increase;
        let pitch_angle_cos = start_pitch_angle_cos
            + 0.5
                * (pitch_angle_cos_col_depth_deriv_1 + pitch_angle_cos_col_depth_deriv_2)
                * col_depth_increase;
        let electron_flux = start_electron_flux
            + 0.5
                * (flux_spectrum_col_depth_deriv_1 + flux_spectrum_col_depth_deriv_2)
                * col_depth_increase;

        if energy <= THERMALIZATION_ENERGY || pitch_angle_cos <= THERMALIZATION_PITCH_ANGLE_COS {
            TransportResult::Thermalized
        } else {
            TransportResult::NewValues((energy, pitch_angle_cos, feb::max(0.0, electron_flux)))
        }
    }

    #[allow(dead_code)]
    fn advance_energy_and_pitch_angle_cos_with_second_order_heun(
        &self,
        start_energy: feb,
        start_pitch_angle_cos: feb,
        col_depth_increase: feb,
    ) -> TransportResultForEnergyAndPitchAngle {
        let ColumnDepthDerivativesForEnergyAndPitchAngle {
            energy: energy_col_depth_deriv_1,
            pitch_angle: pitch_angle_cos_col_depth_deriv_1,
        } = self.compute_col_depth_derivs_for_energy_and_pitch_angle_cos(
            start_energy,
            start_pitch_angle_cos,
        );

        let energy_1 = start_energy + energy_col_depth_deriv_1 * col_depth_increase;
        let pitch_angle_cos_1 =
            start_pitch_angle_cos + pitch_angle_cos_col_depth_deriv_1 * col_depth_increase;

        if energy_1 <= THERMALIZATION_ENERGY || pitch_angle_cos_1 <= THERMALIZATION_PITCH_ANGLE_COS
        {
            return TransportResultForEnergyAndPitchAngle::Thermalized;
        }

        let ColumnDepthDerivativesForEnergyAndPitchAngle {
            energy: energy_col_depth_deriv_2,
            pitch_angle: pitch_angle_cos_col_depth_deriv_2,
        } = self
            .compute_col_depth_derivs_for_energy_and_pitch_angle_cos(energy_1, pitch_angle_cos_1);

        let energy = start_energy
            + 0.5 * (energy_col_depth_deriv_1 + energy_col_depth_deriv_2) * col_depth_increase;
        let pitch_angle_cos = start_pitch_angle_cos
            + 0.5
                * (pitch_angle_cos_col_depth_deriv_1 + pitch_angle_cos_col_depth_deriv_2)
                * col_depth_increase;

        if energy <= THERMALIZATION_ENERGY || pitch_angle_cos <= THERMALIZATION_PITCH_ANGLE_COS {
            TransportResultForEnergyAndPitchAngle::Thermalized
        } else {
            TransportResultForEnergyAndPitchAngle::NewValues((energy, pitch_angle_cos))
        }
    }

    #[allow(dead_code)]
    fn advance_quantities_with_third_order_heun(
        &self,
        start_energy: feb,
        start_pitch_angle_cos: feb,
        start_electron_flux: feb,
        col_depth_increase: feb,
    ) -> TransportResult {
        let ColumnDepthDerivatives {
            energy: energy_col_depth_deriv_1,
            pitch_angle: pitch_angle_cos_col_depth_deriv_1,
            flux_spectrum: flux_spectrum_col_depth_deriv_1,
        } = self.compute_col_depth_derivs(start_energy, start_pitch_angle_cos, start_electron_flux);

        let energy_1 = start_energy + energy_col_depth_deriv_1 * col_depth_increase / 3.0;
        let pitch_angle_cos_1 =
            start_pitch_angle_cos + pitch_angle_cos_col_depth_deriv_1 * col_depth_increase / 3.0;
        let electron_flux_1 =
            start_electron_flux + flux_spectrum_col_depth_deriv_1 * col_depth_increase / 3.0;

        if energy_1 <= THERMALIZATION_ENERGY || pitch_angle_cos_1 <= THERMALIZATION_PITCH_ANGLE_COS
        {
            return TransportResult::Thermalized;
        }

        let ColumnDepthDerivatives {
            energy: energy_col_depth_deriv_2,
            pitch_angle: pitch_angle_cos_col_depth_deriv_2,
            flux_spectrum: flux_spectrum_col_depth_deriv_2,
        } = self.compute_col_depth_derivs(energy_1, pitch_angle_cos_1, electron_flux_1);

        let energy_2 = start_energy + energy_col_depth_deriv_2 * col_depth_increase * 2.0 / 3.0;
        let pitch_angle_cos_2 = start_pitch_angle_cos
            + pitch_angle_cos_col_depth_deriv_2 * col_depth_increase * 2.0 / 3.0;
        let electron_flux_2 =
            start_electron_flux + flux_spectrum_col_depth_deriv_2 * col_depth_increase * 2.0 / 3.0;

        if energy_2 <= THERMALIZATION_ENERGY || pitch_angle_cos_2 <= THERMALIZATION_PITCH_ANGLE_COS
        {
            return TransportResult::Thermalized;
        }

        let ColumnDepthDerivatives {
            energy: energy_col_depth_deriv_3,
            pitch_angle: pitch_angle_cos_col_depth_deriv_3,
            flux_spectrum: flux_spectrum_col_depth_deriv_3,
        } = self.compute_col_depth_derivs(energy_2, pitch_angle_cos_2, electron_flux_2);

        let energy = start_energy
            + (0.25 * energy_col_depth_deriv_1 + 0.75 * energy_col_depth_deriv_3)
                * col_depth_increase;
        let pitch_angle_cos = start_pitch_angle_cos
            + (0.25 * pitch_angle_cos_col_depth_deriv_1 + 0.75 * pitch_angle_cos_col_depth_deriv_3)
                * col_depth_increase;
        let electron_flux = start_electron_flux
            + (0.25 * flux_spectrum_col_depth_deriv_1 + 0.75 * flux_spectrum_col_depth_deriv_3)
                * col_depth_increase;

        if energy <= THERMALIZATION_ENERGY || pitch_angle_cos <= THERMALIZATION_PITCH_ANGLE_COS {
            TransportResult::Thermalized
        } else {
            TransportResult::NewValues((energy, pitch_angle_cos, feb::max(0.0, electron_flux)))
        }
    }

    #[allow(dead_code)]
    fn advance_energy_and_pitch_angle_cos_with_third_order_heun(
        &self,
        start_energy: feb,
        start_pitch_angle_cos: feb,
        col_depth_increase: feb,
    ) -> TransportResultForEnergyAndPitchAngle {
        let ColumnDepthDerivativesForEnergyAndPitchAngle {
            energy: energy_col_depth_deriv_1,
            pitch_angle: pitch_angle_cos_col_depth_deriv_1,
        } = self.compute_col_depth_derivs_for_energy_and_pitch_angle_cos(
            start_energy,
            start_pitch_angle_cos,
        );

        let energy_1 = start_energy + energy_col_depth_deriv_1 * col_depth_increase / 3.0;
        let pitch_angle_cos_1 =
            start_pitch_angle_cos + pitch_angle_cos_col_depth_deriv_1 * col_depth_increase / 3.0;

        if energy_1 <= THERMALIZATION_ENERGY || pitch_angle_cos_1 <= THERMALIZATION_PITCH_ANGLE_COS
        {
            return TransportResultForEnergyAndPitchAngle::Thermalized;
        }

        let ColumnDepthDerivativesForEnergyAndPitchAngle {
            energy: energy_col_depth_deriv_2,
            pitch_angle: pitch_angle_cos_col_depth_deriv_2,
        } = self
            .compute_col_depth_derivs_for_energy_and_pitch_angle_cos(energy_1, pitch_angle_cos_1);

        let energy_2 = start_energy + energy_col_depth_deriv_2 * col_depth_increase * 2.0 / 3.0;
        let pitch_angle_cos_2 = start_pitch_angle_cos
            + pitch_angle_cos_col_depth_deriv_2 * col_depth_increase * 2.0 / 3.0;

        if energy_2 <= THERMALIZATION_ENERGY || pitch_angle_cos_2 <= THERMALIZATION_PITCH_ANGLE_COS
        {
            return TransportResultForEnergyAndPitchAngle::Thermalized;
        }

        let ColumnDepthDerivativesForEnergyAndPitchAngle {
            energy: energy_col_depth_deriv_3,
            pitch_angle: pitch_angle_cos_col_depth_deriv_3,
        } = self
            .compute_col_depth_derivs_for_energy_and_pitch_angle_cos(energy_2, pitch_angle_cos_2);

        let energy = start_energy
            + (0.25 * energy_col_depth_deriv_1 + 0.75 * energy_col_depth_deriv_3)
                * col_depth_increase;
        let pitch_angle_cos = start_pitch_angle_cos
            + (0.25 * pitch_angle_cos_col_depth_deriv_1 + 0.75 * pitch_angle_cos_col_depth_deriv_3)
                * col_depth_increase;

        if energy <= THERMALIZATION_ENERGY || pitch_angle_cos <= THERMALIZATION_PITCH_ANGLE_COS {
            TransportResultForEnergyAndPitchAngle::Thermalized
        } else {
            TransportResultForEnergyAndPitchAngle::NewValues((energy, pitch_angle_cos))
        }
    }

    #[allow(dead_code)]
    fn advance_quantities_with_fourth_order_runge_kutta(
        &self,
        start_energy: feb,
        start_pitch_angle_cos: feb,
        start_electron_flux: feb,
        col_depth_increase: feb,
    ) -> TransportResult {
        let ColumnDepthDerivatives {
            energy: energy_col_depth_deriv_1,
            pitch_angle: pitch_angle_cos_col_depth_deriv_1,
            flux_spectrum: flux_spectrum_col_depth_deriv_1,
        } = self.compute_col_depth_derivs(start_energy, start_pitch_angle_cos, start_electron_flux);

        let energy_1 = start_energy + energy_col_depth_deriv_1 * col_depth_increase * 0.5;
        let pitch_angle_cos_1 =
            start_pitch_angle_cos + pitch_angle_cos_col_depth_deriv_1 * col_depth_increase * 0.5;
        let electron_flux_1 =
            start_electron_flux + flux_spectrum_col_depth_deriv_1 * col_depth_increase * 0.5;

        if energy_1 <= THERMALIZATION_ENERGY || pitch_angle_cos_1 <= THERMALIZATION_PITCH_ANGLE_COS
        {
            return TransportResult::Thermalized;
        }

        let ColumnDepthDerivatives {
            energy: energy_col_depth_deriv_2,
            pitch_angle: pitch_angle_cos_col_depth_deriv_2,
            flux_spectrum: flux_spectrum_col_depth_deriv_2,
        } = self.compute_col_depth_derivs(energy_1, pitch_angle_cos_1, electron_flux_1);

        let energy_2 = start_energy + energy_col_depth_deriv_2 * col_depth_increase * 0.5;
        let pitch_angle_cos_2 =
            start_pitch_angle_cos + pitch_angle_cos_col_depth_deriv_2 * col_depth_increase * 0.5;
        let electron_flux_2 =
            start_electron_flux + flux_spectrum_col_depth_deriv_2 * col_depth_increase * 0.5;

        if energy_2 <= THERMALIZATION_ENERGY || pitch_angle_cos_2 <= THERMALIZATION_PITCH_ANGLE_COS
        {
            return TransportResult::Thermalized;
        }

        let ColumnDepthDerivatives {
            energy: energy_col_depth_deriv_3,
            pitch_angle: pitch_angle_cos_col_depth_deriv_3,
            flux_spectrum: flux_spectrum_col_depth_deriv_3,
        } = self.compute_col_depth_derivs(energy_2, pitch_angle_cos_2, electron_flux_2);

        let energy_3 = start_energy + energy_col_depth_deriv_3 * col_depth_increase;
        let pitch_angle_cos_3 =
            start_pitch_angle_cos + pitch_angle_cos_col_depth_deriv_3 * col_depth_increase;
        let electron_flux_3 =
            start_electron_flux + flux_spectrum_col_depth_deriv_3 * col_depth_increase;

        if energy_3 <= THERMALIZATION_ENERGY || pitch_angle_cos_3 <= THERMALIZATION_PITCH_ANGLE_COS
        {
            return TransportResult::Thermalized;
        }

        let ColumnDepthDerivatives {
            energy: energy_col_depth_deriv_4,
            pitch_angle: pitch_angle_cos_col_depth_deriv_4,
            flux_spectrum: flux_spectrum_col_depth_deriv_4,
        } = self.compute_col_depth_derivs(energy_3, pitch_angle_cos_3, electron_flux_3);

        let energy = start_energy
            + (energy_col_depth_deriv_1
                + 2.0 * energy_col_depth_deriv_2
                + 2.0 * energy_col_depth_deriv_3
                + energy_col_depth_deriv_4)
                * col_depth_increase
                / 6.0;
        let pitch_angle_cos = start_pitch_angle_cos
            + (pitch_angle_cos_col_depth_deriv_1
                + 2.0 * pitch_angle_cos_col_depth_deriv_2
                + 2.0 * pitch_angle_cos_col_depth_deriv_3
                + pitch_angle_cos_col_depth_deriv_4)
                * col_depth_increase
                / 6.0;
        let electron_flux = start_electron_flux
            + (flux_spectrum_col_depth_deriv_1
                + 2.0 * flux_spectrum_col_depth_deriv_2
                + 2.0 * flux_spectrum_col_depth_deriv_3
                + flux_spectrum_col_depth_deriv_4)
                * col_depth_increase
                / 6.0;

        if energy <= THERMALIZATION_ENERGY || pitch_angle_cos <= THERMALIZATION_PITCH_ANGLE_COS {
            TransportResult::Thermalized
        } else {
            TransportResult::NewValues((energy, pitch_angle_cos, feb::max(0.0, electron_flux)))
        }
    }

    #[allow(dead_code)]
    fn advance_energy_and_pitch_angle_cos_with_fourth_order_runge_kutta(
        &self,
        start_energy: feb,
        start_pitch_angle_cos: feb,
        col_depth_increase: feb,
    ) -> TransportResultForEnergyAndPitchAngle {
        let ColumnDepthDerivativesForEnergyAndPitchAngle {
            energy: energy_col_depth_deriv_1,
            pitch_angle: pitch_angle_cos_col_depth_deriv_1,
        } = self.compute_col_depth_derivs_for_energy_and_pitch_angle_cos(
            start_energy,
            start_pitch_angle_cos,
        );

        let energy_1 = start_energy + energy_col_depth_deriv_1 * col_depth_increase * 0.5;
        let pitch_angle_cos_1 =
            start_pitch_angle_cos + pitch_angle_cos_col_depth_deriv_1 * col_depth_increase * 0.5;

        if energy_1 <= THERMALIZATION_ENERGY || pitch_angle_cos_1 <= THERMALIZATION_PITCH_ANGLE_COS
        {
            return TransportResultForEnergyAndPitchAngle::Thermalized;
        }

        let ColumnDepthDerivativesForEnergyAndPitchAngle {
            energy: energy_col_depth_deriv_2,
            pitch_angle: pitch_angle_cos_col_depth_deriv_2,
        } = self
            .compute_col_depth_derivs_for_energy_and_pitch_angle_cos(energy_1, pitch_angle_cos_1);

        let energy_2 = start_energy + energy_col_depth_deriv_2 * col_depth_increase * 0.5;
        let pitch_angle_cos_2 =
            start_pitch_angle_cos + pitch_angle_cos_col_depth_deriv_2 * col_depth_increase * 0.5;

        if energy_2 <= THERMALIZATION_ENERGY || pitch_angle_cos_2 <= THERMALIZATION_PITCH_ANGLE_COS
        {
            return TransportResultForEnergyAndPitchAngle::Thermalized;
        }

        let ColumnDepthDerivativesForEnergyAndPitchAngle {
            energy: energy_col_depth_deriv_3,
            pitch_angle: pitch_angle_cos_col_depth_deriv_3,
        } = self
            .compute_col_depth_derivs_for_energy_and_pitch_angle_cos(energy_2, pitch_angle_cos_2);

        let energy_3 = start_energy + energy_col_depth_deriv_3 * col_depth_increase;
        let pitch_angle_cos_3 =
            start_pitch_angle_cos + pitch_angle_cos_col_depth_deriv_3 * col_depth_increase;

        if energy_3 <= THERMALIZATION_ENERGY || pitch_angle_cos_3 <= THERMALIZATION_PITCH_ANGLE_COS
        {
            return TransportResultForEnergyAndPitchAngle::Thermalized;
        }

        let ColumnDepthDerivativesForEnergyAndPitchAngle {
            energy: energy_col_depth_deriv_4,
            pitch_angle: pitch_angle_cos_col_depth_deriv_4,
        } = self
            .compute_col_depth_derivs_for_energy_and_pitch_angle_cos(energy_3, pitch_angle_cos_3);

        let energy = start_energy
            + (energy_col_depth_deriv_1
                + 2.0 * energy_col_depth_deriv_2
                + 2.0 * energy_col_depth_deriv_3
                + energy_col_depth_deriv_4)
                * col_depth_increase
                / 6.0;
        let pitch_angle_cos = start_pitch_angle_cos
            + (pitch_angle_cos_col_depth_deriv_1
                + 2.0 * pitch_angle_cos_col_depth_deriv_2
                + 2.0 * pitch_angle_cos_col_depth_deriv_3
                + pitch_angle_cos_col_depth_deriv_4)
                * col_depth_increase
                / 6.0;

        if energy <= THERMALIZATION_ENERGY || pitch_angle_cos <= THERMALIZATION_PITCH_ANGLE_COS {
            TransportResultForEnergyAndPitchAngle::Thermalized
        } else {
            TransportResultForEnergyAndPitchAngle::NewValues((energy, pitch_angle_cos))
        }
    }

    fn compute_col_depth_derivs(
        &self,
        energy: feb,
        pitch_angle_cos: feb,
        electron_flux: feb,
    ) -> ColumnDepthDerivatives {
        let EvaluatedHydrogenCoulombLogarithms {
            for_energy: hybrid_coulomb_log_for_energy,
            for_pitch_angle: hybrid_coulomb_log_for_pitch_angle,
            for_flux_spectrum: hybrid_coulomb_log_for_flux_spectrum,
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
            flux_spectrum: self.compute_flux_spectrum_col_depth_deriv_with_hybrid_coulomb_log(
                energy,
                pitch_angle_cos,
                electron_flux,
                hybrid_coulomb_log_for_flux_spectrum,
            ),
        }
    }

    fn compute_col_depth_derivs_for_energy_and_pitch_angle_cos(
        &self,
        energy: feb,
        pitch_angle_cos: feb,
    ) -> ColumnDepthDerivativesForEnergyAndPitchAngle {
        let EvaluatedHydrogenCoulombLogarithmsForEnergyAndPitchAngle {
            for_energy: hybrid_coulomb_log_for_energy,
            for_pitch_angle: hybrid_coulomb_log_for_pitch_angle,
        } = self
            .hybrid_coulomb_log
            .evaluate_for_energy_and_pitch_angle(energy);

        ColumnDepthDerivativesForEnergyAndPitchAngle {
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
        }
    }

    fn compute_energy_col_depth_deriv(&self, energy: feb, pitch_angle_cos: feb) -> feb {
        self.compute_energy_col_depth_deriv_with_hybrid_coulomb_log(
            energy,
            pitch_angle_cos,
            self.hybrid_coulomb_log.for_energy(energy),
        )
    }

    pub fn compute_collisional_energy_time_deriv(&self, energy: feb) -> feb {
        self.compute_collisional_energy_time_deriv_with_hybrid_coulomb_log(
            energy,
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
            - (Q_ELECTRON * self.total_trajectory_aligned_electric_field
                + GYROMAGNETIC_RADIATION_SCALE_E
                    * (1.0 - pitch_angle_cos * pitch_angle_cos)
                    * feb::sqrt(2.0 * energy / M_ELECTRON)
                    * self.squared_magnetic_field_strength_for_radiation
                    / pitch_angle_cos)
                / self.abundances().total_hydrogen_density()
    }

    fn compute_collisional_energy_time_deriv_with_hybrid_coulomb_log(
        &self,
        energy: feb,
        hybrid_coulomb_log_for_energy: feb,
    ) -> feb {
        -COLLISION_SCALE
            * hybrid_coulomb_log_for_energy
            * self.abundances().total_hydrogen_density()
            * feb::sqrt(2.0 * energy / M_ELECTRON)
            / energy
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
                    / (self.abundances().total_hydrogen_density() * energy))
                * (1.0 - pitch_angle_cos * pitch_angle_cos)
                / (2.0 * pitch_angle_cos)
            + GYROMAGNETIC_RADIATION_SCALE_MU
                * (1.0 - pitch_angle_cos * pitch_angle_cos)
                * self.squared_magnetic_field_strength_for_radiation
                / (feb::sqrt(2.0 * energy / M_ELECTRON)
                    * self.abundances().total_hydrogen_density())
    }

    fn compute_flux_spectrum_col_depth_deriv_with_hybrid_coulomb_log(
        &self,
        energy: feb,
        pitch_angle_cos: feb,
        electron_flux: feb,
        hybrid_coulomb_log_for_flux_spectrum: feb,
    ) -> feb {
        -COLLISION_SCALE * hybrid_coulomb_log_for_flux_spectrum * electron_flux
            / (pitch_angle_cos * energy * energy)
            - ((1.0 + pitch_angle_cos * pitch_angle_cos)
                * Q_ELECTRON
                * self.total_trajectory_aligned_electric_field
                / (self.abundances().total_hydrogen_density() * energy)
                + (1.0 - pitch_angle_cos * pitch_angle_cos)
                    * self.log_magnetic_field_col_depth_deriv)
                * electron_flux
                / (2.0 * pitch_angle_cos * pitch_angle_cos)
            - GYROMAGNETIC_RADIATION_SCALE_MU
                * (1.0 - pitch_angle_cos * pitch_angle_cos)
                * self.squared_magnetic_field_strength_for_radiation
                * electron_flux
                / (pitch_angle_cos
                    * feb::sqrt(2.0 * energy / M_ELECTRON)
                    * self.abundances().total_hydrogen_density())
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
