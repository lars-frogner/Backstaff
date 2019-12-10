//! Physical and mathematical constants.

/// Floating-point precision to use for constants.
#[allow(non_camel_case_types)]
pub type fcn = f64;

// Mathematical constants

#[allow(clippy::approx_constant)]
pub const PI: fcn = 3.14159;

// Physical constants

/// Atomic mass unit [g].
pub const AMU: fcn = 1.660_540_2e-24;
/// Electron charge [esu].
pub const Q_ELECTRON: fcn = 4.80325e-10;
/// Electron mass [g].
pub const M_ELECTRON: fcn = 9.109_389_7e-28;
/// Hydrogen mass [g].
pub const M_H: fcn = 1.672_621_9e-24;
/// Helium mass [g].
pub const M_HE: fcn = 6.65e-24;
/// Ion mass [g].
pub const MION: fcn = M_H;
/// Speed of light in vacuum [cm/s].
pub const CLIGHT: fcn = 2.997_924_58e10;
/// Boltzmann constant [erg/K].
pub const KBOLTZMANN: fcn = 1.380_658e-16;
/// Planck constant [erg s].
pub const HPLANCK: fcn = 6.626_075_5e-27;
/// Stefan-Boltzmann constant [erg/(cm^2 s K^4)].
pub const STEFBOLTZ: fcn = 5.670_400e-5;
/// Bohr radius [cm].
pub const RBOHR: fcn = 5.291_773_49e-9;
/// Ionization potential of hydrogen [erg].
pub const E_RYDBERG: fcn = 2.179_874_1e-11;
/// H2 dissociation energy [eV].
pub const EH2DISS: fcn = 4.478;
/// pi e^2 / m_e c [cm^2 Hz].
pub const PIE2_MEC: fcn = 0.02654;
/// Electron rest energy [erg].
pub const MC2_ELECTRON: fcn = M_ELECTRON * CLIGHT * CLIGHT;
/// Ionization energy of an hydrogen atom [eV].
pub const XI_H: fcn = 13.595;

// Unit conversion factors

/// Conversion factor from electron volts to ergs.
pub const EV_TO_ERG: fcn = 1.602_177_33e-12;
/// Conversion factor from kilo electron volts to ergs.
pub const KEV_TO_ERG: fcn = EV_TO_ERG * 1e3;
/// Conversion factor from electron volts to kelvin (1/`KBOLTZMANN`).
pub const EV_TO_K: fcn = 11604.50520;
/// Conversion factor from electron volts to Joules.
pub const EV_TO_J: fcn = 1.602_177_33e-19;
/// Conversion factor from nanometers to meters.
pub const NM_TO_M: fcn = 1e-9;
/// Conversion factor from centimeters to meters.
pub const CM_TO_M: fcn = 1e-2;
/// Conversion factor from kilometers to meters.
pub const KM_TO_M: fcn = 1e3;
/// Conversion factor from ergs to Joules.
pub const ERG_TO_JOULE: fcn = 1e-7;
/// Conversion factor from grams to kilograms.
pub const G_TO_KG: fcn = 1e-3;
/// Conversion factor from microns to nanometers.
pub const MICRON_TO_NM: fcn = 1e3;
/// Conversion factor from megabarns to square meters.
pub const MEGABARN_TO_M2: fcn = 1e-22;
/// Conversion factor from atmospheres to Pascal.
pub const ATM_TO_PA: fcn = 1.0135e5;
/// Conversion factor from dynes per square centimeter to Pascal.
pub const DYNE_CM2_TO_PASCAL: fcn = 0.1;
