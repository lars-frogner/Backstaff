//!

use crate::{
    constants::{KBOLTZMANN, KEV_TO_ERG, M_ELECTRON, PI, Q_ELECTRON},
    ebeam::feb,
};
use special::Error;

#[derive(Clone, Debug)]
pub struct CoulombLogarithm {
    with_electrons_protons: feb,
    with_neutral_hydrogen_for_energy: feb,
    with_neutral_hydrogen_for_pitch_angle: feb,
}

#[derive(Clone, Debug)]
pub struct HybridCoulombLogarithm {
    warm_target: Option<WarmTarget>,
    coulomb_log: CoulombLogarithm,
    electron_to_hydrogen_ratio: feb,
    hydrogen_ionization_fraction: feb,
    for_energy_cold_target: feb,
    for_pitch_angle_cold_target: feb,
    for_number_density_cold_target: feb,
}

#[derive(Clone, Debug)]
pub struct EvaluatedHydrogenCoulombLogarithms {
    pub for_energy: feb,
    pub for_pitch_angle: feb,
    pub for_number_density: feb,
}

#[derive(Clone, Debug)]
struct WarmTarget {
    energy_to_squared_dimensionless_speed_factor: feb,
}

#[derive(Clone, Debug)]
struct WarmTargetHybridCoulombLogFactors {
    for_energy: feb,
    for_pitch_angle: feb,
    for_number_density: feb,
}

impl CoulombLogarithm {
    /// `-1/2*ln(2*pi*(electron charge [esu])^6)`
    const COULOMB_LOG_WITH_ELECTRONS_PROTONS_OFFSET: feb = 63.4507;

    /// `ln(2/(1.105*(hydrogen ionization potential [erg])))`
    const COULOMB_LOG_WITH_NEUTRAL_HYDROGEN_FOR_ENERGY_OFFSET: feb = 25.1433;

    /// `ln(sqrt(2 / (electron mass [g])) / ((fine structure constant) * (light speed [cm/s])))`
    const COULOMB_LOG_WITH_NEUTRAL_HYDROGEN_FOR_PITCH_ANGLE_OFFSET: feb = 12.2746;

    /// Smallest mean electron energy that will be used to compute Coulomb logarithms [erg].
    const MIN_COULOMB_LOG_MEAN_ENERGY: feb = 1e-3 * KEV_TO_ERG;

    pub fn new(electron_density: feb, electron_energy: feb) -> Self {
        Self {
            with_electrons_protons: Self::compute_coulomb_log_with_electrons_protons(
                electron_density,
                electron_energy,
            ),
            with_neutral_hydrogen_for_energy:
                Self::compute_coulomb_log_with_neutral_hydrogen_for_energy(electron_energy),
            with_neutral_hydrogen_for_pitch_angle:
                Self::compute_coulomb_log_with_neutral_hydrogen_for_pitch_angle(electron_energy),
        }
    }

    pub fn with_electrons_protons(&self) -> feb {
        self.with_electrons_protons
    }

    pub fn with_neutral_hydrogen_for_energy(&self) -> feb {
        self.with_neutral_hydrogen_for_energy
    }

    pub fn with_neutral_hydrogen_for_pitch_angle(&self) -> feb {
        self.with_neutral_hydrogen_for_pitch_angle
    }

    fn compute_coulomb_log_with_electrons_protons(
        electron_density: feb,
        electron_energy: feb,
    ) -> feb {
        Self::COULOMB_LOG_WITH_ELECTRONS_PROTONS_OFFSET
            + 0.5
                * feb::ln(
                    feb::powi(
                        feb::max(electron_energy, Self::MIN_COULOMB_LOG_MEAN_ENERGY),
                        3,
                    ) / electron_density,
                )
    }

    fn compute_coulomb_log_with_neutral_hydrogen_for_energy(electron_energy: feb) -> feb {
        Self::COULOMB_LOG_WITH_NEUTRAL_HYDROGEN_FOR_ENERGY_OFFSET
            + feb::ln(feb::max(electron_energy, Self::MIN_COULOMB_LOG_MEAN_ENERGY))
    }

    fn compute_coulomb_log_with_neutral_hydrogen_for_pitch_angle(electron_energy: feb) -> feb {
        Self::COULOMB_LOG_WITH_NEUTRAL_HYDROGEN_FOR_PITCH_ANGLE_OFFSET
            + 0.5 * feb::ln(feb::max(electron_energy, Self::MIN_COULOMB_LOG_MEAN_ENERGY))
    }
}

impl HybridCoulombLogarithm {
    pub fn new(
        enable_warm_target: bool,
        coulomb_log: CoulombLogarithm,
        temperature: feb,
        electron_to_hydrogen_ratio: feb,
        hydrogen_ionization_fraction: feb,
    ) -> Self {
        let warm_target = if enable_warm_target {
            Some(WarmTarget::new(temperature, enable_warm_target))
        } else {
            None
        };

        let for_energy_cold_target = Self::compute_cold_target_hybrid_coulomb_log_for_energy(
            &coulomb_log,
            electron_to_hydrogen_ratio,
            hydrogen_ionization_fraction,
        );
        let for_pitch_angle_cold_target =
            Self::compute_cold_target_hybrid_coulomb_log_for_pitch_angle(
                &coulomb_log,
                electron_to_hydrogen_ratio,
                hydrogen_ionization_fraction,
            );
        let for_number_density_cold_target =
            Self::compute_cold_target_hybrid_coulomb_log_for_number_density(
                &coulomb_log,
                hydrogen_ionization_fraction,
            );

        Self {
            warm_target,
            coulomb_log,
            electron_to_hydrogen_ratio,
            hydrogen_ionization_fraction,
            for_energy_cold_target,
            for_pitch_angle_cold_target,
            for_number_density_cold_target,
        }
    }

    pub fn coulomb_log(&self) -> &CoulombLogarithm {
        &self.coulomb_log
    }

    pub fn hydrogen_ionization_fraction(&self) -> feb {
        self.hydrogen_ionization_fraction
    }

    pub fn for_energy_cold_target(&self) -> feb {
        self.for_energy_cold_target
    }

    pub fn for_pitch_angle_cold_target(&self) -> feb {
        self.for_pitch_angle_cold_target
    }

    pub fn for_number_density_cold_target(&self) -> feb {
        self.for_number_density_cold_target
    }

    pub fn for_energy(&self, energy: feb) -> feb {
        if let Some(warm_target) = &self.warm_target {
            let warm_target_factor_for_energy =
                warm_target.compute_hybrid_coulom_log_factor_for_energy(energy);
            Self::compute_hybrid_coulomb_log_for_energy(
                warm_target_factor_for_energy,
                &self.coulomb_log,
                self.electron_to_hydrogen_ratio,
                self.hydrogen_ionization_fraction,
            )
        } else {
            self.for_energy_cold_target
        }
    }

    pub fn evaluate(&self, energy: feb) -> EvaluatedHydrogenCoulombLogarithms {
        if let Some(warm_target) = &self.warm_target {
            let WarmTargetHybridCoulombLogFactors {
                for_energy: warm_target_factor_for_energy,
                for_pitch_angle: warm_target_factor_for_pitch_angle,
                for_number_density: warm_target_factor_for_number_density,
            } = warm_target.compute_hybrid_coulomb_log_factors(energy);

            EvaluatedHydrogenCoulombLogarithms {
                for_energy: Self::compute_hybrid_coulomb_log_for_energy(
                    warm_target_factor_for_energy,
                    &self.coulomb_log,
                    self.electron_to_hydrogen_ratio,
                    self.hydrogen_ionization_fraction,
                ),
                for_pitch_angle: Self::compute_hybrid_coulomb_log_for_pitch_angle(
                    warm_target_factor_for_pitch_angle,
                    &self.coulomb_log,
                    self.electron_to_hydrogen_ratio,
                    self.hydrogen_ionization_fraction,
                ),
                for_number_density: Self::compute_hybrid_coulomb_log_for_number_density(
                    warm_target_factor_for_number_density,
                    &self.coulomb_log,
                    self.electron_to_hydrogen_ratio,
                    self.hydrogen_ionization_fraction,
                ),
            }
        } else {
            EvaluatedHydrogenCoulombLogarithms {
                for_energy: self.for_energy_cold_target,
                for_pitch_angle: self.for_pitch_angle_cold_target,
                for_number_density: self.for_number_density_cold_target,
            }
        }
    }

    fn compute_cold_target_hybrid_coulomb_log_for_energy(
        coulomb_log: &CoulombLogarithm,
        electron_to_hydrogen_ratio: feb,
        hydrogen_ionization_fraction: feb,
    ) -> feb {
        electron_to_hydrogen_ratio * coulomb_log.with_electrons_protons()
            + (1.0 - hydrogen_ionization_fraction) * coulomb_log.with_neutral_hydrogen_for_energy()
    }

    fn compute_hybrid_coulomb_log_for_energy(
        warm_target_factor: feb,
        coulomb_log: &CoulombLogarithm,
        electron_to_hydrogen_ratio: feb,
        hydrogen_ionization_fraction: feb,
    ) -> feb {
        warm_target_factor * electron_to_hydrogen_ratio * coulomb_log.with_electrons_protons()
            + (1.0 - hydrogen_ionization_fraction) * coulomb_log.with_neutral_hydrogen_for_energy()
    }

    fn compute_cold_target_hybrid_coulomb_log_for_pitch_angle(
        coulomb_log: &CoulombLogarithm,
        electron_to_hydrogen_ratio: feb,
        hydrogen_ionization_fraction: feb,
    ) -> feb {
        (electron_to_hydrogen_ratio + hydrogen_ionization_fraction)
            * coulomb_log.with_electrons_protons()
            + (1.0 - hydrogen_ionization_fraction)
                * coulomb_log.with_neutral_hydrogen_for_pitch_angle()
    }

    fn compute_hybrid_coulomb_log_for_pitch_angle(
        warm_target_factor: feb,
        coulomb_log: &CoulombLogarithm,
        electron_to_hydrogen_ratio: feb,
        hydrogen_ionization_fraction: feb,
    ) -> feb {
        (warm_target_factor * electron_to_hydrogen_ratio + hydrogen_ionization_fraction)
            * coulomb_log.with_electrons_protons()
            + (1.0 - hydrogen_ionization_fraction)
                * coulomb_log.with_neutral_hydrogen_for_pitch_angle()
    }

    fn compute_cold_target_hybrid_coulomb_log_for_number_density(
        coulomb_log: &CoulombLogarithm,
        hydrogen_ionization_fraction: feb,
    ) -> feb {
        -hydrogen_ionization_fraction * coulomb_log.with_electrons_protons()
            + (1.0 - hydrogen_ionization_fraction)
                * (coulomb_log.with_neutral_hydrogen_for_energy()
                    - coulomb_log.with_neutral_hydrogen_for_pitch_angle())
    }

    fn compute_hybrid_coulomb_log_for_number_density(
        warm_target_factor: feb,
        coulomb_log: &CoulombLogarithm,
        electron_to_hydrogen_ratio: feb,
        hydrogen_ionization_fraction: feb,
    ) -> feb {
        (warm_target_factor * electron_to_hydrogen_ratio - hydrogen_ionization_fraction)
            * coulomb_log.with_electrons_protons()
            + (1.0 - hydrogen_ionization_fraction)
                * (coulomb_log.with_neutral_hydrogen_for_energy()
                    - coulomb_log.with_neutral_hydrogen_for_pitch_angle())
    }
}

impl WarmTarget {
    fn new(temperature: feb, enabled: bool) -> Self {
        Self {
            energy_to_squared_dimensionless_speed_factor:
                Self::compute_energy_to_squared_dimensionless_speed_factor(temperature),
        }
    }

    fn compute_hybrid_coulomb_log_factors(&self, energy: feb) -> WarmTargetHybridCoulombLogFactors {
        let u = self.compute_dimensionless_speed(energy);
        let erf = Self::compute_error_function(u);
        let u_times_erf_deriv = u * Self::compute_error_function_deriv(u);
        let G = Self::compute_chandrasekhar_function(u);

        WarmTargetHybridCoulombLogFactors {
            for_energy: erf + 2.0 * (u_times_erf_deriv + G),
            for_pitch_angle: erf - G,
            for_number_density: (4.0 * u * u - 3.0) * u_times_erf_deriv + 7.0 * G,
        }
    }

    fn compute_hybrid_coulom_log_factor_for_energy(&self, energy: feb) -> feb {
        let u = self.compute_dimensionless_speed(energy);
        let erf = Self::compute_error_function(u);
        let u_times_erf_deriv = u * Self::compute_error_function_deriv(u);
        let G = Self::compute_chandrasekhar_function(u);

        erf + 2.0 * (u_times_erf_deriv + G)
    }

    fn compute_dimensionless_speed(&self, energy: feb) -> feb {
        feb::sqrt(energy * self.energy_to_squared_dimensionless_speed_factor)
    }

    fn compute_error_function(dimensionless_speed: feb) -> feb {
        feb::error(dimensionless_speed)
    }

    fn compute_error_function_deriv(dimensionless_speed: feb) -> feb {
        (2.0 / feb::sqrt(PI)) * feb::exp(-dimensionless_speed * dimensionless_speed)
    }

    fn compute_chandrasekhar_function(dimensionless_speed: feb) -> feb {
        (Self::compute_error_function(dimensionless_speed)
            - dimensionless_speed * Self::compute_error_function_deriv(dimensionless_speed))
            / (2.0 * dimensionless_speed * dimensionless_speed)
    }

    fn compute_energy_to_squared_dimensionless_speed_factor(temperature: feb) -> feb {
        1.0 / (KBOLTZMANN * temperature)
    }
}

/// Uses the Spitzer resistivity formula to calculate resistivity
/// parallel to the magnetic field in cgs units.
pub fn compute_parallel_resistivity(
    coulomb_log: &CoulombLogarithm,
    temperature: feb,
    hydrogen_ionization_fraction: feb,
) -> feb {
    const PARALLEL_RESISTIVITY_FACTOR: feb = 0.51282;
    PARALLEL_RESISTIVITY_FACTOR
        * (4.0 / 3.0)
        * feb::sqrt(2.0 * PI)
        * Q_ELECTRON
        * Q_ELECTRON
        * feb::sqrt(M_ELECTRON)
        * coulomb_log.with_electrons_protons()
        / feb::sqrt(KBOLTZMANN * temperature).powi(3)
}
