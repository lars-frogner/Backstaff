//! Power-law electron distribution.

use nrfind;
use crate::constants::{KBOLTZMANN, MC2_ELECTRON, KEV_TO_ERG};
use crate::num::BFloat;
use crate::geometry::Point3;
use crate::grid::Grid3;
use crate::field::ScalarField3;
use crate::interpolation::Interpolator3;
use super::Distribution;
use super::super::feb;

/// A non-thermal power-law distribution over electron energy,
/// parameterized by an exponent `delta`, a `total_power_density`
/// and a `lower_cutoff_energy`.
///
/// The probability density for an electron energy `E` is
/// `P(E) = (delta - 1)*lower_cutoff_energy^(delta - 1)*E^(-delta)`.
#[derive(Clone)]
pub struct PLDistribution {
    config: PLDistributionConfig,
    /// Total energy injected into the distribution per volume and time [erg/(cm^3 s)].
    total_power_density: feb,
    /// Lower cut-off energy [units of electron rest energy].
    lower_cutoff_energy: feb,
    /// Current collisional depth [dimensionless]. Increases as the distribution propagates.
    collisional_depth: feb,
    /// Current remaining energy per volume and time [erg/(cm^3 s)]. Decreases as the distribution propagates.
    remaining_power_density: feb
}

/// Configuration parameters for power-law distributions.
#[derive(Clone)]
pub struct PLDistributionConfig {
    /// Exponent of the inverse power-law.
    pub delta: feb,
    /// Initial guess to use when estimating lower cut-off energy [keV].
    pub initial_cutoff_energy_guess: feb,
    /// Target relative error when estimating lower cut-off energy.
    pub acceptable_root_finding_error: feb,
    /// Maximum number of iterations when estimating lower cut-off energy.
    pub max_root_finding_iterations: i32
}

impl PLDistribution {
/*     pub fn from_acceleration_event(config: PLDistributionConfig, event: &AccelerationEvent) -> Self {
        config.validate();
        PLDistribution{
            config,
            total_power_density,
            lower_cutoff_energy,
            collisional_depth: 0.0,
            remaining_power_density: total_power_density
        }
    } */

    pub fn collisional_depth(&self) -> feb { self.collisional_depth }

    pub fn remaining_power_density(&self) -> feb { self.remaining_power_density }

    pub fn propagate<F, G, I>(&mut self, electron_density: &ScalarField3<F, G>, interpolator: &I, new_position: &Point3<F>, step_size: feb) -> Option<feb>
    where F: BFloat,
          G: Grid3<F>,
          I: Interpolator3
    {
        None
    }

    /// Estimates the lower cut-off energy of the non-thermal distribution by
    /// determining where the power-law intersects the thermal Maxwell-Boltzmann
    /// distribution.
    ///
    /// More precisely, the method determines the cut-off energy `E_cut` such that
    /// `e_therm*P_MB(E_cut) = e_acc*P_PL(E_cut)`,
    /// where `e_therm = 3*ne/(2*beta)` is the energy density of thermal electrons,
    /// `e_acc` is the energy density of non-thermal electrons and `P_MB` and `P_PL`
    /// are respectively the Maxwell-Boltzmann and power-law probability distributions.
    fn estimate_cutoff_energy_from_thermal_intersection(config: &PLDistributionConfig, event: &AccelerationEvent) -> Option<feb> {
        if event.acc_energy_density < std::f64::EPSILON {
            return None
        }
        let beta = KEV_TO_ERG/(KBOLTZMANN*event.temperature);                              // [1/keV]
        let thermal_fraction = KEV_TO_ERG*(3.0/2.0)*event.electron_density*feb::sqrt(beta)
                               /(event.acc_energy_density*(config.delta - 1.0));           // [1/keV^(3/2)]

        let difference = |energy| thermal_fraction*energy*feb::exp(-beta*energy) - 1.0;
        let derivative = |energy| thermal_fraction*(1.0 - beta*energy)*feb::exp(-beta*energy);

        let intersection_energy = match nrfind::find_root(&difference,
                                                          &derivative,
                                                          config.initial_cutoff_energy_guess,
                                                          config.acceptable_root_finding_error,
                                                          config.max_root_finding_iterations)
        {
            Ok(energy) => energy,
            Err(err) => {
                println!("Cut-off energy estimation failed: {}", err);
                return None
            }
        };
        let lower_cutoff_energy = intersection_energy*KEV_TO_ERG*MC2_ELECTRON;
        Some(lower_cutoff_energy)
    }
}

struct AccelerationEvent {
    /// Energy per volume going into acceleration of non-thermal particles [erg/cm^3].
    acc_energy_density: feb,
    /// Temperature at the acceleration site [K].
    temperature: feb,
    /// Total electron number density at the acceleration site [electrons/cm^3].
    electron_density: feb
}

/* impl AccelerationEvent {
    pub fn new(temperature: &ScalarField3<F, G>, electron_density: &ScalarField3<F, G>, joule_heating_rate: &ScalarField3<F, G>, interpolator: &I, position: &Point3<F>) -> Self {
        assert!(acc_energy_density > 0.0, "Acceleration energy must be larger than zero.");
        assert!(temperature > 0.0, "Temperature must be larger than zero.");
        assert!(electron_density > 0.0, "Electron density must be larger than zero.");
        AccelerationEvent{ acc_energy_density, temperature, electron_density }
    }
} */

impl PLDistributionConfig {
    const DEFAULT_DELTA: feb = 4.0;
    const DEFAULT_INITIAL_CUTOFF_ENERGY_GUESS: feb = 4.0;
    const DEFAULT_ACCEPTABLE_ROOT_FINDING_ERROR: feb = 1e-3;
    const DEFAULT_MAX_ROOT_FINDING_ITERATIONS: i32 = 100;

    /// Creates a new configuration struct with the default values.
    pub fn default() -> Self {
        PLDistributionConfig {
            delta: Self::DEFAULT_DELTA,
            initial_cutoff_energy_guess: Self::DEFAULT_INITIAL_CUTOFF_ENERGY_GUESS,
            acceptable_root_finding_error: Self::DEFAULT_ACCEPTABLE_ROOT_FINDING_ERROR,
            max_root_finding_iterations: Self::DEFAULT_MAX_ROOT_FINDING_ITERATIONS
        }
    }

    fn validate(&self) {
        assert!(self.delta > 1.0, "Delta-exponent must be larger than one.");
        assert!(self.initial_cutoff_energy_guess > 0.0, "Initial cut-off energy guess must be larger than zero.");
        assert!(self.acceptable_root_finding_error > 0.0, "Acceptable root finding error must be larger than zero.");
        assert!(self.max_root_finding_iterations > 0, "Maximum number of root finding iterations must be larger than zero.");
    }
}

impl Distribution for PLDistribution {}
