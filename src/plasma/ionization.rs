//! Calculations of the ionization state of plasma particles.

use super::fpl;
use crate::constants::{EV_TO_ERG, HPLANCK, KBOLTZMANN, M_ELECTRON, M_H, PI, XI_H};
use lazy_static::lazy_static;

lazy_static! {
    /// Constant factor in the Saha equations: (h^2/(2*pi*me*kB))^3/2
    static ref SAHA_SCALE: fpl = fpl::powf(HPLANCK*HPLANCK/(2.0*PI*M_ELECTRON*KBOLTZMANN), 1.5);
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
    let tmp = electron_density * (*SAHA_SCALE) / fpl::powf(temperature, 1.5);
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
    let tmp = electron_density * (*SAHA_SCALE) / fpl::powf(temperature, 1.5);
    1.0 / (1.0 + tmp * fpl::exp(XI_H * EV_TO_ERG / (KBOLTZMANN * temperature)))
}
