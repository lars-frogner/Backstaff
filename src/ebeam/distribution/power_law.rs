//! Power-law electron distribution.

pub mod acceleration;

use std::collections::HashMap;
use crate::constants::{AMU, MC2_ELECTRON, KEV_TO_ERG};
use crate::units::solar::{U_L, U_R};
use crate::io::snapshot::{fdt, SnapshotCacher3};
use crate::geometry::{Vec3, Point3};
use crate::grid::Grid3;
use crate::interpolation::Interpolator3;
use crate::tracing::ftr;
use crate::tracing::stepping::SteppingSense;
use super::{PropagationSense, DepletionStatus, PropagationResult, Distribution};
use super::super::feb;

/// An electron pitch-angle distribution which is either peaked or isotropic.
#[derive(Clone, Copy, Debug)]
pub enum PitchAngleDistribution {
    Peaked,
    Isotropic
}

/// Configuration parameters for power-law distributions.
#[derive(Clone, Debug)]
pub struct PowerLawDistributionConfig {
    /// Distributions with an acceleration direction with a smaller angle
    /// to the magnetic field normal than this are discarded [deg].
    pub min_acceleration_angle: feb,
    /// Distributions with an initial estimated depletion distance smaller than this value are discarded [cm].
    pub min_estimated_depletion_distance: feb,
    /// Distributions with remaining power densities smaller than this value are discarded [erg/(cm^3 s)].
    pub min_remaining_power_density: feb
}

/// Properties of a power-law distribution of non-thermal electrons.
#[derive(Clone, Debug)]
pub struct PowerLawDistributionProperties {
    /// Exponent of the inverse power-law.
    delta: feb,
    /// Type of pitch angle distribution of the non-thermal electrons.
    pitch_angle_distribution: PitchAngleDistribution,
    /// Total energy injected into the distribution per volume and time [erg/(cm^3 s)].
    total_power_density: feb,
    /// Lower cut-off energy [units of electron rest energy].
    lower_cutoff_energy: feb,
    /// Position where the distribution originates [Mm].
    acceleration_position: Point3<fdt>,
    /// Direction of acceleration of the electrons.
    acceleration_direction: Vec3<fdt>,
    /// Total mass density at the acceleration site [g/cm^3].
    mass_density: feb
}

/// A non-thermal power-law distribution over electron energy,
/// parameterized by an exponent `delta`, a `total_power_density`
/// and a `lower_cutoff_energy`.
///
/// The probability density for an electron energy `E` is
/// `P(E) = (delta - 1)*lower_cutoff_energy^(delta - 1)*E^(-delta)`.
#[derive(Clone, Debug)]
pub struct PowerLawDistribution {
    config: PowerLawDistributionConfig,
    properties: PowerLawDistributionProperties,
    /// Factor which is 2 for a peaked and 4 for an isotropic pitch angle distribution.
    pitch_angle_factor: feb,
    /// Cosine of the minimum angle that the acceleration direction must be away from
    /// the normal of the magnetic field direction in order for the electrons to be propagated.
    acceleration_alignment_threshold: fdt,
    /// Mean energy of the electrons in the distribution [units of electron rest energy].
    mean_energy: feb,
    /// Current collisional depth [dimensionless]. Increases as the distribution propagates.
    collisional_depth: feb,
    /// Current remaining energy per volume and time [erg/(cm^3 s)]. Decreases as the distribution propagates.
    remaining_power_density: feb
}

impl PowerLawDistribution {
    /// Fraction of a mass of plasma assumed to be made up of hydrogen.
    const HYDROGEN_MASS_FRACTION: feb = 0.735;

    /// Conversion factor from mass density [g] to electron density [1/cm^3],
    /// assuming a fully ionized plasma with no metals and the hard-coded value for
    /// the hydrogen mass fraction.
    const MASS_DENSITY_TO_ELECTRON_DENSITY: feb = (1.0 + Self::HYDROGEN_MASS_FRACTION)/(2.0*AMU);

    /// 2*pi*(classical electron radius)^2 [cm^2]
    const COLLISION_SCALE: feb = 4.989_344e-25;

    /// 1/2*ln( (2*pi*me*c/h)^3/(pi*alpha) [1/cm^3] )
    const COULOMB_OFFSET: feb = 37.853_791;

    /// Returns the exponent of the inverse power-law.
    pub fn delta(&self) -> feb { self.properties.delta }

    /// Returns the type of pitch angle distribution of the non-thermal electrons.
    pub fn pitch_angle_distribution(&self) -> PitchAngleDistribution { self.properties.pitch_angle_distribution }

    /// Returns the total energy injected into the distribution per volume and time [erg/(cm^3 s)].
    pub fn total_power_density(&self) -> feb { self.properties.total_power_density }

    /// Returns the lower cut-off energy [units of electron rest energy].
    pub fn lower_cutoff_energy(&self) -> feb { self.properties.lower_cutoff_energy }

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

    fn new(config: PowerLawDistributionConfig, properties: PowerLawDistributionProperties) -> Option<Self> {
        let pitch_angle_factor = match properties.pitch_angle_distribution {
            PitchAngleDistribution::Peaked => 2.0,
            PitchAngleDistribution::Isotropic => 4.0
        };

        let acceleration_alignment_threshold = feb::cos(config.min_acceleration_angle.to_radians()) as fdt;
        let mean_energy = properties.lower_cutoff_energy*(properties.delta - 1.0)/(properties.delta - 2.0);
        let electron_density = Self::compute_electron_density(properties.mass_density);

        let estimated_depletion_distance = Self::estimate_depletion_distance(
            electron_density,
            properties.delta,
            pitch_angle_factor,
            properties.total_power_density,
            properties.lower_cutoff_energy,
            mean_energy,
            config.min_remaining_power_density
        );
        if estimated_depletion_distance < config.min_estimated_depletion_distance {
            return None
        }

        let collisional_depth = 0.0;
        let remaining_power_density = properties.total_power_density;

        Some(PowerLawDistribution{
            config,
            properties,
            pitch_angle_factor,
            acceleration_alignment_threshold,
            mean_energy,
            collisional_depth,
            remaining_power_density
        })
    }

    fn compute_electron_density(mass_density: feb) -> feb {
        mass_density*Self::MASS_DENSITY_TO_ELECTRON_DENSITY
    }

    fn estimate_depletion_distance(electron_density: feb, delta: feb, pitch_angle_factor: feb, total_power_density: feb, lower_cutoff_energy: feb, mean_energy: feb, depletion_power_density: feb) -> feb {
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

impl Distribution for PowerLawDistribution {
    fn acceleration_position(&self) -> &Point3<fdt> { &self.properties.acceleration_position }

    fn determine_propagation_sense(&self, magnetic_field_direction: &Vec3<fdt>) -> PropagationSense {
        let aligment = self.properties.acceleration_direction.dot(magnetic_field_direction);
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

    fn scalar_properties(&self) -> HashMap<String, feb> {
        let mut properties = HashMap::new();
        properties.insert("total_power_density".to_string(), self.properties.total_power_density);
        properties.insert("lower_cutoff_energy".to_string(), self.properties.lower_cutoff_energy*MC2_ELECTRON/KEV_TO_ERG); // [keV]
        properties
    }

    fn vector_properties(&self) -> HashMap<String, Vec3<feb>> {
        let mut properties = HashMap::new();
        properties.insert("acceleration_direction".to_string(), Vec3::from(&self.properties.acceleration_direction));
        properties
    }

    fn propagate<G, I>(&mut self, snapshot: &SnapshotCacher3<G>, interpolator: &I, displacement: &Vec3<ftr>, new_position: &Point3<ftr>) -> PropagationResult
    where G: Grid3<fdt>,
          I: Interpolator3
    {
        let mut deposition_position = new_position - displacement*0.5;
        let mass_density_field = snapshot.cached_scalar_field("r");

        let mass_density = interpolator.interp_scalar_field(mass_density_field, &Point3::from(&deposition_position))
                                       .unwrap_and_update_position(&mut deposition_position);

        let electron_density = Self::compute_electron_density(feb::from(mass_density)*U_R); // [electrons/cm^3]
        let step_length = displacement.length()*U_L;                                        // [cm]

        let collisional_depth_increase = Self::compute_collisional_depth_increase(
            electron_density,
            self.pitch_angle_factor,
            self.mean_energy,
            step_length
        );
        let slope_correction_factor = 1.0;// + slope_corr*z_weight;
        let new_collisional_depth = self.collisional_depth + slope_correction_factor*collisional_depth_increase;

        let new_remaining_power_density = Self::compute_remaining_power_density(
            self.properties.delta,
            self.properties.total_power_density,
            self.properties.lower_cutoff_energy,
            new_collisional_depth
        );

        let deposited_power_density = self.remaining_power_density - new_remaining_power_density;

        self.collisional_depth = new_collisional_depth;
        self.remaining_power_density = new_remaining_power_density;

        let depletion_status = if self.remaining_power_density >= self.config.min_remaining_power_density {
            DepletionStatus::Undepleted
        } else {
            DepletionStatus::Depleted
        };

        assert!(deposited_power_density.is_finite());

        PropagationResult{ deposited_power_density, deposition_position, depletion_status }
    }
}

impl PowerLawDistributionConfig {
    const DEFAULT_MIN_ACCELERATION_ANGLE:           feb = 20.0; // [deg]
    const DEFAULT_MIN_ESTIMATED_DEPLETION_DISTANCE: feb = 3e7;  // [cm]
    const DEFAULT_MIN_REMAINING_POWER_DENSITY:      feb = 1e-6; // [erg/(cm^3 s)]

    /// Creates a new configuration struct with the default values.
    pub fn default() -> Self {
        PowerLawDistributionConfig {
            min_acceleration_angle: Self::DEFAULT_MIN_ACCELERATION_ANGLE,
            min_estimated_depletion_distance: Self::DEFAULT_MIN_ESTIMATED_DEPLETION_DISTANCE,
            min_remaining_power_density: Self::DEFAULT_MIN_REMAINING_POWER_DENSITY
        }
    }

    /// Panics if any of the configuration parameter values are invalid.
    pub fn validate(&self) {
        assert!(self.min_acceleration_angle >= 0.0, "Minimum acceleration angle must be larger than or equal to zero.");
        assert!(self.min_estimated_depletion_distance >= 0.0, "Minimum estimated depletion distance must be larger than or equal to zero.");
        assert!(self.min_remaining_power_density >= 0.0, "Minimum remaining power density must be larger than or equal to zero.");
    }
}
