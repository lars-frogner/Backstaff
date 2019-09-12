//! Power-law electron distribution.

use nrfind;
use crate::constants::{KBOLTZMANN, MC2_ELECTRON, KEV_TO_ERG};
use crate::units::solar::{U_L, U_L3};
use crate::io::snapshot::{fdt, SnapshotCacher3};
use crate::geometry::{Vec3, Point3};
use crate::grid::Grid3;
use crate::interpolation::Interpolator3;
use crate::tracing::stepping::SteppingSense;
use super::{PropagationResult, Distribution, DistributionGenerator};
use super::super::feb;
use super::super::acceleration::simple::SimpleAccelerationEvent;

/// An electron pitch-angle distribution which is either peaked or isotropic.
#[derive(Clone, Copy, Debug)]
pub enum PitchAngleDistribution {
    Peaked,
    Isotropic
}

/// Direction of propagation of the non-thermal electrons with respect to the magnetic field.
pub type PropagationSense = Option<SteppingSense>;

/// Configuration parameters for power-law distributions.
#[derive(Clone, Debug)]
pub struct PLDistributionConfig {
    /// Exponent of the inverse power-law.
    pub delta: feb,
    /// Type of pitch angle distribution of the non-thermal electrons.
    pub pitch_angle_distribution: PitchAngleDistribution,
    /// Distributions with total power densities smaller than this value are discarded [erg/(cm^3 s)].
    pub min_total_power_density: feb,
    /// Distributions with remaining power densities smaller than this value are discarded [erg/(cm^3 s)].
    pub min_remaining_power_density: feb,
    /// Distributions with an initial estimated depletion distance smaller than this value are discarded [cm].
    pub min_estimated_depletion_distance: feb,
    /// Distributions with an acceleration direction with a smaller angle
    /// to the magnetic field normal than this are discarded [deg].
    pub min_acceleration_angle: feb,
    /// Initial guess to use when estimating lower cut-off energy [keV].
    pub initial_cutoff_energy_guess: feb,
    /// Target relative error when estimating lower cut-off energy.
    pub acceptable_root_finding_error: feb,
    /// Maximum number of iterations when estimating lower cut-off energy.
    pub max_root_finding_iterations: i32
}

/// A non-thermal power-law distribution over electron energy,
/// parameterized by an exponent `delta`, a `total_power_density`
/// and a `lower_cutoff_energy`.
///
/// The probability density for an electron energy `E` is
/// `P(E) = (delta - 1)*lower_cutoff_energy^(delta - 1)*E^(-delta)`.
#[derive(Clone, Debug)]
pub struct PLDistribution {
    config: PLDistributionConfig,
    /// Factor which is 2 for a peaked and 4 for an isotropic pitch angle distribution.
    pitch_angle_factor: feb,
    /// Cosine of the minimum angle that the acceleration direction must be away from
    /// the normal of the magnetic field direction in order for the electrons to be propagated.
    acceleration_alignment_threshold: fdt,
    /// Total energy injected into the distribution per volume and time [erg/(cm^3 s)].
    total_power_density: feb,
    /// Lower cut-off energy [units of electron rest energy].
    lower_cutoff_energy: feb,
    /// Mean energy of the electrons in the distribution [units of electron rest energy].
    mean_energy: feb,
    /// Direction of acceleration of the electrons.
    acceleration_direction: Vec3<fdt>,
    /// Current position of the distribution [Mm].
    position: Point3<fdt>,
    /// Current collisional depth [dimensionless]. Increases as the distribution propagates.
    collisional_depth: feb,
    /// Current remaining energy per volume and time [erg/(cm^3 s)]. Decreases as the distribution propagates.
    remaining_power_density: feb
}

/// Generator for power-law distributions with the same configuration parameters.
#[derive(Clone, Debug)]
pub struct PLDistributionGenerator {
    config: PLDistributionConfig
}

impl PLDistribution {
    /// 2*pi*(classical electron radius)^2 [cm^2]
    const COLLISION_SCALE: feb = 4.989_344e-25;

    /// 1/2*ln( (2*pi*me*c/h)^3/(pi*alpha) [1/cm^3] )
    const COULOMB_OFFSET: feb = 37.853_791;

    fn from_acceleration_event(config: PLDistributionConfig, event: &SimpleAccelerationEvent) -> Option<Self> {
        config.validate();

        let pitch_angle_factor = match config.pitch_angle_distribution {
            PitchAngleDistribution::Peaked => 2.0,
            PitchAngleDistribution::Isotropic => 4.0
        };

        let acceleration_alignment_threshold = feb::cos(config.min_acceleration_angle.to_radians()) as fdt;

        let total_power_density = event.particle_power_density();
        if total_power_density < config.min_total_power_density {
            return None
        }

        let lower_cutoff_energy = match Self::estimate_cutoff_energy_from_thermal_intersection(&config, event) {
            Some(energy) => energy,
            None => return None
        };

        let mean_energy = lower_cutoff_energy*(config.delta - 1.0)/(config.delta - 2.0);

        let acceleration_direction = match event.acceleration_direction() {
            Some(direction) => direction.clone(),
            None => return None
        };

        let estimated_depletion_distance = Self::estimate_depletion_distance(
            event.electron_density(),
            config.delta,
            pitch_angle_factor,
            total_power_density,
            lower_cutoff_energy,
            mean_energy,
            config.min_remaining_power_density
        );

        if estimated_depletion_distance < config.min_estimated_depletion_distance {
            return None
        }

        Some(PLDistribution{
            config,
            pitch_angle_factor,
            acceleration_alignment_threshold,
            total_power_density,
            lower_cutoff_energy,
            mean_energy,
            acceleration_direction,
            position: event.position().clone(),
            collisional_depth: 0.0,
            remaining_power_density: total_power_density
        })
    }

    /// Returns the exponent of the inverse power-law.
    pub fn delta(&self) -> feb { self.config.delta }

    /// Returns the type of pitch angle distribution of the non-thermal electrons.
    pub fn pitch_angle_distribution(&self) -> PitchAngleDistribution { self.config.pitch_angle_distribution }

    /// Returns the total energy injected into the distribution per volume and time [erg/(cm^3 s)].
    pub fn total_power_density(&self) -> feb { self.total_power_density }

    /// Returns the lower cut-off energy [units of electron rest energy].
    pub fn lower_cutoff_energy(&self) -> feb { self.lower_cutoff_energy }

    /// Returns the mean energy of the electrons in the distribution [units of electron rest energy].
    pub fn mean_energy(&self) -> feb { self.mean_energy }

    /// Returns the current collisional depth.
    ///
    /// Increases as the distribution propagates.
    pub fn collisional_depth(&self) -> feb { self.collisional_depth }

    /// Returns the current remaining energy per volume and time [erg/(cm^3 s)].
    ///
    /// Decreases as the distribution propagates.
    pub fn remaining_power_density(&self) -> feb { self.remaining_power_density }

    /// Finds the direction that the distribution will move along the magnetic field, if any.
    pub fn determine_propagation_sense(&self, magnetic_field_direction: &Vec3<fdt>) -> PropagationSense {
        let aligment = self.acceleration_direction.dot(magnetic_field_direction);
        if fdt::abs(aligment) > self.acceleration_alignment_threshold {
            if aligment > 0.0 {
                Some(SteppingSense::Same)
            } else {
                Some(SteppingSense::Opposite)
            }
        } else {
            None
        }
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
    fn estimate_cutoff_energy_from_thermal_intersection(config: &PLDistributionConfig, event: &SimpleAccelerationEvent) -> Option<feb> {
        if event.particle_energy_density() < std::f64::EPSILON {
            return None
        }
        let beta = KEV_TO_ERG/(KBOLTZMANN*event.temperature());                              // [1/keV]
        let thermal_fraction = KEV_TO_ERG*(3.0/2.0)*event.electron_density()*feb::sqrt(beta)
                               /(event.particle_energy_density()*(config.delta - 1.0));      // [1/keV^(3/2)]

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

    pub fn estimate_depletion_distance(electron_density: feb, delta: feb, pitch_angle_factor: feb, total_power_density: feb, lower_cutoff_energy: feb, mean_energy: feb, depletion_power_density: feb) -> feb {
        feb::powi(lower_cutoff_energy, 2)*
        (feb::powf(depletion_power_density/total_power_density, 2.0/(2.0 - delta)) - 1.0)/
        (pitch_angle_factor*Self::compute_collisional_coef(electron_density, mean_energy))
    }

    fn compute_collisional_depth_increase(electron_density: feb, pitch_angle_factor: feb, mean_energy: feb, step_length: feb) -> feb {
        pitch_angle_factor*Self::compute_collisional_coef(electron_density, mean_energy)*step_length
    }

    fn compute_remaining_power_density(delta: feb, total_power_density: feb, lower_cutoff_energy: feb, collisional_depth: feb) -> feb {
        total_power_density*feb::powf(1.0 + collisional_depth/feb::powi(lower_cutoff_energy, 2), 1.0 - 0.5*delta)
    }

    fn compute_collisional_coef(electron_density: feb, electron_energy: feb) -> feb {
        Self::COLLISION_SCALE*electron_density*Self::compute_coulomb_logarithm(electron_density, electron_energy)
    }

    fn compute_coulomb_logarithm(particle_density: feb, electron_energy: feb) -> feb {
        Self::COULOMB_OFFSET + 0.5*feb::ln(feb::powi(electron_energy*(electron_energy + 2.0), 2)/particle_density)
    }
}

impl Distribution for PLDistribution {
    fn propagate<G, I>(&mut self, snapshot: &mut SnapshotCacher3<G>, interpolator: &I, displacement: &Vec3<fdt>) -> PropagationResult
    where G: Grid3<fdt>,
          I: Interpolator3
    {
        let midpoint = &self.position + displacement*0.5;
        let electron_density = interpolator.interp_scalar_field(snapshot.expect_scalar_field("ne"), &midpoint).expect_inside();

        let electron_density = feb::from(electron_density)/U_L3; // [electrons/cm^3]
        let step_length = feb::from(displacement.length())*U_L;  // [cm]

        let collisional_depth_increase = Self::compute_collisional_depth_increase(
            electron_density,
            self.pitch_angle_factor,
            self.mean_energy,
            step_length
        );
        let slope_correction_factor = 1.0;// + slope_corr*z_weight;
        let new_collisional_depth = self.collisional_depth + slope_correction_factor*collisional_depth_increase;

        let new_remaining_power_density = Self::compute_remaining_power_density(
            self.config.delta,
            self.total_power_density,
            self.lower_cutoff_energy,
            new_collisional_depth
        );

        let deposited_power_density = self.remaining_power_density - new_remaining_power_density;

        self.position = &self.position + displacement;
        self.collisional_depth = new_collisional_depth;
        self.remaining_power_density = new_remaining_power_density;

        if self.remaining_power_density >= self.config.min_remaining_power_density {
            PropagationResult::Ok(deposited_power_density)
        } else {
            PropagationResult::Depleted(deposited_power_density)
        }
    }
}

impl PLDistributionConfig {
    const DEFAULT_DELTA: feb = 4.0;
    const DEFAULT_PITCH_ANGLE_DISTRIBUTION: PitchAngleDistribution = PitchAngleDistribution::Peaked;
    const DEFAULT_MIN_TOTAL_POWER_DENSITY:          feb = 1e-2; // [erg/(cm^3 s)]
    const DEFAULT_MIN_REMAINING_POWER_DENSITY:      feb = 1e-6; // [erg/(cm^3 s)]
    const DEFAULT_MIN_ESTIMATED_DEPLETION_DISTANCE: feb = 3e7;  // [cm]
    const DEFAULT_MIN_ACCELERATION_ANGLE:           feb = 20.0; // [deg]
    const DEFAULT_INITIAL_CUTOFF_ENERGY_GUESS:      feb = 4.0;  // [keV]
    const DEFAULT_ACCEPTABLE_ROOT_FINDING_ERROR:    feb = 1e-3;
    const DEFAULT_MAX_ROOT_FINDING_ITERATIONS:      i32 = 100;

    /// Creates a new configuration struct with the default values.
    pub fn default() -> Self {
        PLDistributionConfig {
            delta: Self::DEFAULT_DELTA,
            pitch_angle_distribution: Self::DEFAULT_PITCH_ANGLE_DISTRIBUTION,
            min_total_power_density: Self::DEFAULT_MIN_TOTAL_POWER_DENSITY,
            min_remaining_power_density: Self::DEFAULT_MIN_REMAINING_POWER_DENSITY,
            min_estimated_depletion_distance: Self::DEFAULT_MIN_ESTIMATED_DEPLETION_DISTANCE,
            min_acceleration_angle: Self::DEFAULT_MIN_ACCELERATION_ANGLE,
            initial_cutoff_energy_guess: Self::DEFAULT_INITIAL_CUTOFF_ENERGY_GUESS,
            acceptable_root_finding_error: Self::DEFAULT_ACCEPTABLE_ROOT_FINDING_ERROR,
            max_root_finding_iterations: Self::DEFAULT_MAX_ROOT_FINDING_ITERATIONS
        }
    }

    fn validate(&self) {
        assert!(self.delta > 2.0, "Delta-exponent must be larger than two.");
        assert!(self.min_total_power_density >= 0.0, "Minimum total power density must be larger than or equal to zero.");
        assert!(self.min_remaining_power_density >= 0.0, "Minimum remaining power density must be larger than or equal to zero.");
        assert!(self.min_estimated_depletion_distance >= 0.0, "Minimum estimated depletion distance must be larger than or equal to zero.");
        assert!(self.min_acceleration_angle >= 0.0, "Minimum acceleration angle must be larger than or equal to zero.");
        assert!(self.initial_cutoff_energy_guess > 0.0, "Initial cut-off energy guess must be larger than zero.");
        assert!(self.acceptable_root_finding_error > 0.0, "Acceptable root finding error must be larger than zero.");
        assert!(self.max_root_finding_iterations > 0, "Maximum number of root finding iterations must be larger than zero.");
    }
}

impl PLDistributionGenerator {
    /// Creates a new generator for power-law distributions with the given
    /// configuration parameters.
    pub fn new(config: PLDistributionConfig) -> Self {
        PLDistributionGenerator{ config }
    }
}

impl DistributionGenerator for PLDistributionGenerator {
    type AccelerationEventType = SimpleAccelerationEvent;
    type DistributionType = PLDistribution;

    /// Generates a new power-law distribution from the given acceleration event.
    ///
    /// Returns `None` if the distribution could not be generated.
    fn generate_from_acceleration_event(&self, event: &Self::AccelerationEventType) -> Option<Self::DistributionType> {
        PLDistribution::from_acceleration_event(self.config.clone(), event)
    }
}
