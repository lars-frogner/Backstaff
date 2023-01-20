//!

use crate::{constants::KEV_TO_ERG, ebeam::feb};

#[derive(Clone, Debug)]
pub struct CoulombLogarithm {
    with_electrons_protons: feb,
    with_neutral_hydrogen_for_energy: feb,
    with_neutral_hydrogen_for_pitch_angle: feb,
}

#[derive(Clone, Debug)]
pub struct HybridCoulombLogarithm {
    for_energy: feb,
    for_pitch_angle: feb,
    for_number_density: feb,
    for_pitch_angle_for_energy_ratio: feb,
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
    pub fn new(coulomb_log: CoulombLogarithm, ionization_fraction: feb) -> Self {
        let for_energy =
            Self::compute_hybrid_coulomb_log_for_energy(&coulomb_log, ionization_fraction);
        let for_pitch_angle =
            Self::compute_hybrid_coulomb_log_for_pitch_angle(&coulomb_log, ionization_fraction);
        let for_number_density =
            Self::compute_hybrid_coulomb_log_for_number_density(&coulomb_log, ionization_fraction);
        let for_pitch_angle_for_energy_ratio = for_pitch_angle / for_energy;

        Self {
            for_energy,
            for_pitch_angle,
            for_number_density,
            for_pitch_angle_for_energy_ratio,
        }
    }

    pub fn for_energy(&self) -> feb {
        self.for_energy
    }

    pub fn for_pitch_angle(&self) -> feb {
        self.for_pitch_angle
    }

    pub fn for_number_density(&self) -> feb {
        self.for_number_density
    }

    pub fn for_pitch_angle_for_energy_ratio(&self) -> feb {
        self.for_pitch_angle_for_energy_ratio
    }

    fn compute_hybrid_coulomb_log_for_energy(
        coulomb_log: &CoulombLogarithm,
        ionization_fraction: feb,
    ) -> feb {
        ionization_fraction * coulomb_log.with_electrons_protons()
            + (1.0 - ionization_fraction) * coulomb_log.with_neutral_hydrogen_for_energy()
    }

    fn compute_hybrid_coulomb_log_for_pitch_angle(
        coulomb_log: &CoulombLogarithm,
        ionization_fraction: feb,
    ) -> feb {
        2.0 * ionization_fraction * coulomb_log.with_electrons_protons()
            + (1.0 - ionization_fraction) * coulomb_log.with_neutral_hydrogen_for_pitch_angle()
    }

    fn compute_hybrid_coulomb_log_for_number_density(
        coulomb_log: &CoulombLogarithm,
        ionization_fraction: feb,
    ) -> feb {
        ionization_fraction * coulomb_log.with_electrons_protons()
            + (1.0 - ionization_fraction)
                * (coulomb_log.with_neutral_hydrogen_for_pitch_angle()
                    - coulomb_log.with_neutral_hydrogen_for_energy())
    }
}
