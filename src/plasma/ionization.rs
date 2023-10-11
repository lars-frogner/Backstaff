//! Calculations of the ionization state of plasma particles.

use super::fpl;
use crate::constants::{
    EV_TO_ERG, HPLANCK, KBOLTZMANN, M_ELECTRON, M_H, M_HE, PI, XI_H, XI_HEI, XI_HEII,
};
use lazy_static::lazy_static;

lazy_static! {
    /// Constant factor in the Saha equations: (h^2/(2*pi*me*kB))^3/2
    static ref SAHA_SCALE: fpl = fpl::powf(HPLANCK*HPLANCK/(2.0*PI*M_ELECTRON*KBOLTZMANN), 1.5);
}

#[derive(Clone, Debug)]
pub struct Abundances {
    hydrogen_mass_fraction: fpl,
    helium_mass_fraction: fpl,
    electron_density: fpl,
    total_hydrogen_density: fpl,
    total_helium_density: fpl,
    electron_to_hydrogen_ratio: fpl,
    helium_to_hydrogen_ratio: fpl,
    hydrogen_ionization_fraction: fpl,
    helium_first_ionization_fraction: fpl,
    helium_second_ionization_fraction: fpl,
}

impl Abundances {
    pub fn new(
        hydrogen_mass_fraction: fpl,
        helium_mass_fraction: fpl,
        mass_density: fpl,
        temperature: fpl,
        electron_density: fpl,
    ) -> Self {
        let total_hydrogen_density = mass_density * hydrogen_mass_fraction / M_H;
        let total_helium_density = mass_density * helium_mass_fraction / M_HE;
        let electron_to_hydrogen_ratio = electron_density / total_hydrogen_density;
        let helium_to_hydrogen_ratio = total_helium_density / total_hydrogen_density;
        let hydrogen_ionization_fraction =
            compute_equilibrium_hydrogen_ionization_fraction(temperature, electron_density);
        let (helium_first_ionization_fraction, helium_second_ionization_fraction) =
            compute_equilibrium_helium_ionization_fractions(temperature, electron_density);

        Self {
            hydrogen_mass_fraction,
            helium_mass_fraction,
            electron_density,
            total_hydrogen_density,
            total_helium_density,
            electron_to_hydrogen_ratio,
            helium_to_hydrogen_ratio,
            hydrogen_ionization_fraction,
            helium_first_ionization_fraction,
            helium_second_ionization_fraction,
        }
    }

    pub fn set_electron_to_hydrogen_ratio(&mut self, electron_to_hydrogen_ratio: fpl) {
        self.electron_to_hydrogen_ratio = electron_to_hydrogen_ratio;
    }

    pub fn set_helium_to_hydrogen_ratio(&mut self, helium_to_hydrogen_ratio: fpl) {
        self.helium_to_hydrogen_ratio = helium_to_hydrogen_ratio;
    }

    pub fn set_hydrogen_ionization_fraction(&mut self, hydrogen_ionization_fraction: fpl) {
        self.hydrogen_ionization_fraction = hydrogen_ionization_fraction;
    }

    pub fn hydrogen_mass_fraction(&self) -> fpl {
        self.hydrogen_mass_fraction
    }

    pub fn helium_mass_fraction(&self) -> fpl {
        self.helium_mass_fraction
    }

    pub fn electron_to_hydrogen_ratio(&self) -> fpl {
        self.electron_to_hydrogen_ratio
    }

    pub fn helium_to_hydrogen_ratio(&self) -> fpl {
        self.helium_to_hydrogen_ratio
    }

    pub fn hydrogen_ionization_fraction(&self) -> fpl {
        self.hydrogen_ionization_fraction
    }

    pub fn helium_first_ionization_fraction(&self) -> fpl {
        self.helium_first_ionization_fraction
    }

    pub fn helium_second_ionization_fraction(&self) -> fpl {
        self.helium_second_ionization_fraction
    }

    pub fn total_hydrogen_density(&self) -> fpl {
        self.total_hydrogen_density
    }

    pub fn true_electron_density(&self) -> fpl {
        self.electron_density
    }

    pub fn proton_density(&self) -> fpl {
        self.hydrogen_ionization_fraction * self.total_hydrogen_density
    }

    pub fn neutral_hydrogen_density(&self) -> fpl {
        (1.0 - self.hydrogen_ionization_fraction) * self.total_hydrogen_density
    }
}

/// Computes the number density of neutral hydrogen, assuming thermal equilibrium.
///
/// The result is obtained by evaluating the Saha ionization equation.
pub fn compute_equilibrium_neutral_hydrogen_density(
    hydrogen_mass_fraction: fpl,
    mass_density: fpl,
    temperature: fpl,
    electron_density: fpl,
) -> fpl {
    let tmp = electron_density * (*SAHA_SCALE) / fpl::sqrt(temperature).powi(3);
    mass_density * hydrogen_mass_fraction * tmp
        / (M_H * (tmp + fpl::exp(-XI_H * EV_TO_ERG / (KBOLTZMANN * temperature))))
}

/// Computes the fraction of hydrogen that is ionized, assuming thermal equilibrium.
///
/// The result is obtained by evaluating the Saha ionization equation.
pub fn compute_equilibrium_hydrogen_ionization_fraction(
    temperature: fpl,
    electron_density: fpl,
) -> fpl {
    let tmp = electron_density * (*SAHA_SCALE) / fpl::sqrt(temperature).powi(3);
    1.0 / (1.0 + tmp * fpl::exp(XI_H * EV_TO_ERG / (KBOLTZMANN * temperature)))
}

/// Computes the fractions of helium that are singly and doubly ionized,
/// assuming thermal equilibrium.
///
/// The result is obtained by evaluating the Saha ionization equation.
pub fn compute_equilibrium_helium_ionization_fractions(
    temperature: fpl,
    electron_density: fpl,
) -> (fpl, fpl) {
    let tmp = fpl::sqrt(temperature).powi(3) / ((*SAHA_SCALE) * electron_density);
    let one_over_kb_temperature_ev = EV_TO_ERG / (KBOLTZMANN * temperature);

    let second_over_first_ionization_fraction =
        tmp * fpl::exp(-XI_HEII * one_over_kb_temperature_ev);

    let first_ionization_fraction = 1.0
        / (1.0
            + second_over_first_ionization_fraction
            + fpl::exp(XI_HEI * one_over_kb_temperature_ev) / (4.0 * tmp));

    let second_ionization_fraction =
        second_over_first_ionization_fraction * first_ionization_fraction;

    (first_ionization_fraction, second_ionization_fraction)
}
