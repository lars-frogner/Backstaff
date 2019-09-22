//! Power-law electron distribution.

pub mod acceleration;

use crate::constants::{AMU, MC2_ELECTRON, KEV_TO_ERG};
use crate::units::solar::{U_L, U_R};
use crate::io::snapshot::{fdt, SnapshotCacher3};
use crate::geometry::{Vec3, Point3};
use crate::grid::Grid3;
use crate::interpolation::Interpolator3;
use crate::tracing::ftr;
use crate::tracing::stepping::SteppingSense;
use super::{DepletionStatus, PropagationResult, Distribution};
use super::super::{feb, FixedBeamScalarValues, FixedBeamVectorValues, ElectronBeamProperties, ElectronBeamMetadata};

/// An electron pitch-angle distribution which is either peaked or isotropic.
#[derive(Clone, Copy, Debug)]
pub enum PitchAngleDistribution {
    Peaked,
    Isotropic
}

/// Configuration parameters for power-law distributions.
#[derive(Clone, Debug)]
pub struct PowerLawDistributionConfig {
    /// Distributions with remaining power densities smaller than this value are discarded [erg/(cm^3 s)].
    pub min_remaining_power_density: feb
}

/// Data associated with a power-law distribution.
#[derive(Clone, Debug)]
pub struct PowerLawDistributionData<M: ElectronBeamMetadata> {
    /// Exponent of the inverse power-law.
    delta: feb,
    /// Factor which is 2 for a peaked and 4 for an isotropic pitch angle distribution.
    pitch_angle_factor: feb,
    /// Total energy injected into the distribution per volume and time [erg/(cm^3 s)].
    total_power_density: feb,
    /// Lower cut-off energy [units of electron rest energy].
    lower_cutoff_energy: feb,
    /// Mean energy of the electrons in the distribution [units of electron rest energy].
    mean_energy: feb,
    /// Estimated propagation distance until the remaining power density drops below the minimum value [cm].
    estimated_depletion_distance: feb,
    /// Position where the distribution originates [Mm].
    acceleration_position: Point3<fdt>,
    /// Direction of acceleration of the electrons.
    acceleration_direction: Vec3<fdt>,
    /// Direction of propagation of the electrons relative to the magnetic field direction.
    propagation_sense: SteppingSense,
    /// Total mass density at the acceleration site [g/cm^3].
    mass_density: feb,
    /// Electron beam metadata object for holding arbitrary information about the distribution.
    metadata: M
}

/// Exposed properties of a power-law distribution.
#[derive(Clone, Debug)]
pub struct PowerLawDistributionProperties {
    /// Total energy injected into the distribution per volume and time [erg/(cm^3 s)].
    total_power_density: feb,
    /// Lower cut-off energy [keV].
    lower_cutoff_energy: feb,
    /// Estimated propagation distance until the remaining power density drops below the minimum value [Mm].
    estimated_depletion_distance: feb,
    /// Direction of acceleration of the electrons.
    acceleration_direction: Vec3<feb>
}

/// A non-thermal power-law distribution over electron energy,
/// parameterized by an exponent `delta`, a `total_power_density`
/// and a `lower_cutoff_energy`.
///
/// The probability density for an electron energy `E` is
/// `P(E) = (delta - 1)*lower_cutoff_energy^(delta - 1)*E^(-delta)`.
#[derive(Clone, Debug)]
pub struct PowerLawDistribution<M: ElectronBeamMetadata> {
    config: PowerLawDistributionConfig,
    data: PowerLawDistributionData<M>,
    /// Current collisional depth [dimensionless]. Increases as the distribution propagates.
    collisional_depth: feb,
    /// Current remaining energy per volume and time [erg/(cm^3 s)]. Decreases as the distribution propagates.
    remaining_power_density: feb
}

impl<M: ElectronBeamMetadata> PowerLawDistribution<M> {
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
    pub fn delta(&self) -> feb { self.data.delta }

    /// Returns the total energy injected into the distribution per volume and time [erg/(cm^3 s)].
    pub fn total_power_density(&self) -> feb { self.data.total_power_density }

    /// Returns the lower cut-off energy [units of electron rest energy].
    pub fn lower_cutoff_energy(&self) -> feb { self.data.lower_cutoff_energy }

    /// Returns the mean energy of the electrons in the distribution [units of electron rest energy].
    pub fn mean_energy(&self) -> feb { self.data.mean_energy }

    /// Returns the estimated propagation distance until the remaining
    /// power density drops below the minimum value [cm].
    pub fn estimated_depletion_distance(&self) -> feb { self.data.estimated_depletion_distance }

    /// Returns the current collisional depth.
    ///
    /// Increases as the distribution propagates.
    pub fn collisional_depth(&self) -> feb { self.collisional_depth }

    /// Returns the current remaining energy per volume and time [erg/(cm^3 s)].
    ///
    /// Decreases as the distribution propagates.
    pub fn remaining_power_density(&self) -> feb { self.remaining_power_density }

    fn new(config: PowerLawDistributionConfig, data: PowerLawDistributionData<M>) -> Self {
        let collisional_depth = 0.0;
        let remaining_power_density = data.total_power_density;
        PowerLawDistribution{
            config,
            data,
            collisional_depth,
            remaining_power_density
        }
    }

    fn compute_electron_density(mass_density: feb) -> feb {
        mass_density*Self::MASS_DENSITY_TO_ELECTRON_DENSITY
    }

    fn estimate_depletion_distance(electron_density: feb, delta: feb, pitch_angle_factor: feb, total_power_density: feb, lower_cutoff_energy: feb, mean_energy: feb, depletion_power_density: feb) -> feb {
        feb::powi(lower_cutoff_energy, 2)*
        (feb::powf(depletion_power_density/total_power_density, 2.0/(2.0 - delta)) - 1.0)/
        (pitch_angle_factor*Self::compute_collisional_coef(electron_density, mean_energy))
    }

    fn compute_mean_energy(delta: feb, lower_cutoff_energy: feb) -> feb {
        lower_cutoff_energy*(delta - 1.0)/(delta - 2.0)
    }

    fn determine_pitch_angle_factor(pitch_angle_distribution: PitchAngleDistribution) -> feb {
        match pitch_angle_distribution {
            PitchAngleDistribution::Peaked => 2.0,
            PitchAngleDistribution::Isotropic => 4.0
        }
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

impl ElectronBeamProperties for PowerLawDistributionProperties {
    type NestedTuplesOfValues = (feb, (feb, (feb, Vec3<feb>)));
    #[allow(clippy::type_complexity)]
    type NestedTuplesOfVecs = (Vec<feb>, (Vec<feb>, (Vec<feb>, Vec<Vec3<feb>>)));

    fn into_nested_tuples_of_values(self) -> Self::NestedTuplesOfValues {
        (self.total_power_density, (self.lower_cutoff_energy, (self.estimated_depletion_distance, self.acceleration_direction)))
    }

    fn distribute_nested_tuples_of_vecs_into_maps(vecs: Self::NestedTuplesOfVecs, scalar_values: &mut FixedBeamScalarValues, vector_values: &mut FixedBeamVectorValues) {
        let (total_power_densities, (lower_cutoff_energies, (estimated_depletion_distances, acceleration_directions))) = vecs;
        scalar_values.insert("total_power_density".to_string(), total_power_densities);
        scalar_values.insert("lower_cutoff_energy".to_string(), lower_cutoff_energies);
        scalar_values.insert("estimated_depletion_distance".to_string(), estimated_depletion_distances);
        vector_values.insert("acceleration_direction".to_string(), acceleration_directions);
    }
}

impl<M: ElectronBeamMetadata> Distribution for PowerLawDistribution<M> {
    type PropertiesType = PowerLawDistributionProperties;
    type MetadataType = M;

    fn acceleration_position(&self) -> &Point3<fdt> { &self.data.acceleration_position }

    fn propagation_sense(&self) -> SteppingSense { self.data.propagation_sense }

    fn properties(&self) -> Self::PropertiesType {
        PowerLawDistributionProperties{
            total_power_density: self.data.total_power_density,
            lower_cutoff_energy: self.data.lower_cutoff_energy*MC2_ELECTRON/KEV_TO_ERG,
            estimated_depletion_distance: self.data.estimated_depletion_distance/U_L,
            acceleration_direction: Vec3::from(&self.data.acceleration_direction)
        }
    }

    fn metadata(&self) -> Self::MetadataType { self.data.metadata.clone() }

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

        let collisional_depth_increase = if electron_density > std::f64::EPSILON {
            Self::compute_collisional_depth_increase(
                electron_density,
                self.data.pitch_angle_factor,
                self.data.mean_energy,
                step_length
            )
        } else {
            0.0
        };

        let slope_correction_factor = 1.0;// + slope_corr*z_weight;
        let new_collisional_depth = self.collisional_depth + slope_correction_factor*collisional_depth_increase;

        let new_remaining_power_density = Self::compute_remaining_power_density(
            self.data.delta,
            self.data.total_power_density,
            self.data.lower_cutoff_energy,
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

        PropagationResult{ deposited_power_density, deposition_position, depletion_status }
    }
}

impl PowerLawDistributionConfig {
    const DEFAULT_MIN_REMAINING_POWER_DENSITY: feb = 1e-6; // [erg/(cm^3 s)]

    /// Panics if any of the configuration parameter values are invalid.
    pub fn validate(&self) {
        assert!(self.min_remaining_power_density >= 0.0, "Minimum remaining power density must be larger than or equal to zero.");
    }
}

impl Default for PowerLawDistributionConfig {
    fn default() -> Self {
        PowerLawDistributionConfig {
            min_remaining_power_density: Self::DEFAULT_MIN_REMAINING_POWER_DENSITY
        }
    }
}
