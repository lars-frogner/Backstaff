//! Simple model for acceleration of non-thermal electron beams described by power-law distributions.

use std::io;
use rayon::prelude::*;
use serde::Serialize;
use nrfind;
use crate::constants::{KBOLTZMANN, MC2_ELECTRON, KEV_TO_ERG};
use crate::units::solar::{U_T, U_E, U_R};
use crate::io::snapshot::{fdt, SnapshotCacher3};
use crate::geometry::{Dim3, Vec3, Point3, Idx3};
use crate::grid::Grid3;
use crate::interpolation::Interpolator3;
use super::super::{PitchAngleDistribution, PowerLawDistributionConfig, PowerLawDistributionProperties, PowerLawDistribution};
use super::super::super::Distribution;
use super::super::super::super::feb;
use super::super::super::super::accelerator::Accelerator;
use Dim3::{X, Y, Z};

/// Reason for rejecting a distribution in the simple power-law acceleration model.
#[derive(Clone, Debug, Serialize)]
pub struct PossibleRejectionCauses {
    none: u8,
    too_low_total_power_density: u8,
    too_short_depletion_distance: u8,
    too_perpendicular_direction: u8
}

/// Stores bitvectors encoding distribution rejection causes for a set of acceleration site points.
#[derive(Clone, Debug, Serialize)]
pub struct DistributionRejectionMap {
    positions: Vec<Point3<fdt>>,
    rejection_causes: Vec<u8>,
    possible_rejection_causes: PossibleRejectionCauses
}

/// Configuration parameters for the simple power-law acceleration model.
#[derive(Clone, Debug)]
pub struct SimplePowerLawAccelerationConfig {
    /// Distributions with total power densities smaller than this value are discarded [erg/(cm^3 s)].
    pub min_total_power_density: feb,
    /// Distributions with an initial estimated depletion distance smaller than this value are discarded [cm].
    pub min_estimated_depletion_distance: feb,
    /// Initial guess to use when estimating lower cut-off energy [keV].
    pub initial_cutoff_energy_guess: feb,
    /// Target relative error when estimating lower cut-off energy.
    pub acceptable_root_finding_error: feb,
    /// Maximum number of iterations when estimating lower cut-off energy.
    pub max_root_finding_iterations: i32
}

// Simple acceleration process producing power-law distributions of non-thermal electrons.
#[derive(Clone, Debug)]
pub struct SimplePowerLawAccelerator {
    distribution_config: PowerLawDistributionConfig,
    config: SimplePowerLawAccelerationConfig,
    duration: feb,
    particle_energy_fraction: feb,
    delta: feb,
    pitch_angle_distribution: PitchAngleDistribution
}

impl Default for PossibleRejectionCauses {
    fn default() -> Self {
        PossibleRejectionCauses{
            none:                         0b000,
            too_low_total_power_density:  0b001,
            too_short_depletion_distance: 0b010,
            too_perpendicular_direction:  0b100
        }
    }
}

impl FromParallelIterator<(Point3<fdt>, u8)> for DistributionRejectionMap {
    fn from_par_iter<I>(par_iter: I) -> Self
        where I: IntoParallelIterator<Item = (Point3<fdt>, u8)>
    {
        let (positions, rejection_causes) = par_iter.into_par_iter().unzip();
        DistributionRejectionMap{
            positions,
            rejection_causes,
            possible_rejection_causes: PossibleRejectionCauses::default()
        }
    }
}

impl SimplePowerLawAccelerator {
    /// How many adjacent grid cells in each direction to include when
    /// computing the average electric field around the acceleration site.
    const ELECTRIC_FIELD_PROBING_SPAN: isize = 2;

    /// Generates a new electron distribution, but instead of aborting when the first rejection
    /// condition is met, creates a bitmap encoding all encountered rejection causes.
    ///
    /// Returns `None` if the distribution was rejected due to an unrecoverable condition.
    pub fn detect_rejection_causes<G, I>(&self, snapshot: &SnapshotCacher3<G>, interpolator: &I, indices: &Idx3<usize>) -> Option<u8>
    where G: Grid3<fdt>,
          I: Interpolator3
    {
        let possible_rejection_causes = PossibleRejectionCauses::default();
        let mut rejection_causes = possible_rejection_causes.none;

        let total_power_density = self.compute_total_power_density(snapshot, indices);
        if total_power_density < self.config.min_total_power_density {
            rejection_causes |= possible_rejection_causes.too_low_total_power_density;
        }
        let total_energy_density = total_power_density*self.duration;

        let temperature = feb::from(snapshot.cached_scalar_field("tg").value(indices));
        assert!(temperature > 0.0, "Temperature must be larger than zero.");

        let mass_density = feb::from(snapshot.cached_scalar_field("r").value(indices))*U_R;  // [g/cm^3]
        let electron_density = PowerLawDistribution::compute_electron_density(mass_density); // [electrons/cm^3]
        assert!(electron_density > 0.0, "Electron density must be larger than zero.");

        let lower_cutoff_energy = match self.estimate_cutoff_energy_from_thermal_intersection(temperature, electron_density, total_energy_density) {
            Some(energy) => energy,
            None => return None
        };

        let acceleration_direction = match self.determine_acceleration_direction(snapshot, indices) {
            Some(direction) => direction,
            None => return None
        };

        let distribution_properties = PowerLawDistributionProperties{
            delta: self.delta,
            pitch_angle_distribution: self.pitch_angle_distribution,
            total_power_density,
            lower_cutoff_energy,
            acceleration_position: snapshot.reader().grid().centers().point(indices),
            acceleration_direction,
            mass_density
        };
        let distribution = PowerLawDistribution::new(self.distribution_config.clone(), distribution_properties);

        if distribution.estimated_depletion_distance() < self.config.min_estimated_depletion_distance {
            rejection_causes |= possible_rejection_causes.too_short_depletion_distance;
        };

        let magnetic_field = snapshot.cached_vector_field("b");
        let acceleration_position = distribution.acceleration_position();
        let mut magnetic_field_direction = interpolator.interp_vector_field(magnetic_field, acceleration_position).expect_inside();
        magnetic_field_direction.normalize();
        if distribution.determine_propagation_sense(&magnetic_field_direction).is_none() {
            rejection_causes |= possible_rejection_causes.too_perpendicular_direction;
        }

        Some(rejection_causes)
    }

    /// Makes sure the fields required to find rejection causes are cached in the snapshot cacher.
    pub fn prepare_snapshot_for_rejection_cause_detection<G: Grid3<fdt>>(snapshot: &mut SnapshotCacher3<G>) -> io::Result<()> {
        snapshot.cache_scalar_field("r")?;
        snapshot.cache_scalar_field("tg")?;
        snapshot.cache_scalar_field("qjoule")?;
        snapshot.cache_vector_field("e")?;
        snapshot.cache_vector_field("b")
    }

    fn compute_total_power_density<G>(&self, snapshot: &SnapshotCacher3<G>, indices: &Idx3<usize>) -> feb
    where G: Grid3<fdt>
    {
        let joule_heating_field = snapshot.cached_scalar_field("qjoule");
        let joule_heating = feb::from(joule_heating_field.value(indices));
        let joule_heating = feb::max(0.0, joule_heating*U_E/U_T); // [erg/(cm^3 s)]

        self.particle_energy_fraction*joule_heating
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
    fn estimate_cutoff_energy_from_thermal_intersection(&self, temperature: feb, electron_density: feb, total_energy_density: feb) -> Option<feb> {
        if total_energy_density < std::f64::EPSILON {
            return None
        }
        let beta = KEV_TO_ERG/(KBOLTZMANN*temperature);                              // [1/keV]
        let thermal_fraction = KEV_TO_ERG*(3.0/2.0)*electron_density*feb::sqrt(beta)
                               /(total_energy_density*(self.delta - 1.0));           // [1/keV^(3/2)]
        let ln_thermal_fraction = feb::ln(thermal_fraction);

        let difference = |energy| ln_thermal_fraction + feb::ln(energy) - beta*energy;
        let derivative = |energy| 1.0/energy - beta;

        // Make sure the initial guess never results in a positive initial derivative
        let initial_guess = feb::max(0.9/beta, self.config.initial_cutoff_energy_guess);

        let intersection_energy = match nrfind::find_root(&difference,
                                                          &derivative,
                                                          initial_guess,
                                                          self.config.acceptable_root_finding_error,
                                                          self.config.max_root_finding_iterations)
        {
            Ok(energy) => energy,
            Err(err) => {
                println!("Cut-off energy estimation failed: {}", err);
                return None
            }
        };
        let lower_cutoff_energy = intersection_energy*KEV_TO_ERG/MC2_ELECTRON;
        Some(lower_cutoff_energy)
    }

    fn determine_acceleration_direction<G>(&self, snapshot: &SnapshotCacher3<G>, indices: &Idx3<usize>) -> Option<Vec3<fdt>>
    where G: Grid3<fdt>
    {
        let electric_field = snapshot.cached_vector_field("e");
        let grid = electric_field.grid();

        let lower_indices = Idx3::new(
            indices[X] as isize - Self::ELECTRIC_FIELD_PROBING_SPAN,
            indices[Y] as isize - Self::ELECTRIC_FIELD_PROBING_SPAN,
            indices[Z] as isize - Self::ELECTRIC_FIELD_PROBING_SPAN
        );
        let upper_indices = Idx3::new(
            indices[X] as isize + Self::ELECTRIC_FIELD_PROBING_SPAN + 1,
            indices[Y] as isize + Self::ELECTRIC_FIELD_PROBING_SPAN + 1,
            indices[Z] as isize + Self::ELECTRIC_FIELD_PROBING_SPAN + 1
        );

        let mut total_electric_vector = Vec3::zero();

        for &k in grid.create_idx_range_list(Z, lower_indices[Z], upper_indices[Z]).iter() {
            for &j in grid.create_idx_range_list(Y, lower_indices[Y], upper_indices[Y]).iter() {
                for &i in grid.create_idx_range_list(X, lower_indices[X], upper_indices[X]).iter() {
                    total_electric_vector = total_electric_vector + electric_field.vector(&Idx3::new(i, j, k));
                }
            }
        }
        let squared_total_electric_vector = total_electric_vector.squared_length();

        if squared_total_electric_vector > std::f32::EPSILON {
            // Electrons are accelerated in the opposite direction as the electric field.
            Some(total_electric_vector/(-fdt::sqrt(squared_total_electric_vector)))
        } else {
            None
        }
    }
}

impl SimplePowerLawAccelerator {
    /// Creates a new simple power-law accelerator.
    ///
    /// # Parameters
    ///
    /// - `distribution_config`: Configuration parameters for the distribution.
    /// - `config`: Configuration parameters for the accelerator.
    /// - `duration`: Duration of the acceleration events [s].
    /// - `particle_energy_fraction`: Fraction of the total energy release going into acceleration of non-thermal particles.
    /// - `delta`: Exponent of the inverse power-law.
    /// - `pitch_angle_distribution`: Type of pitch angle distribution of the non-thermal electrons.
    ///
    /// # Returns
    ///
    /// A new `SimplePowerLawAccelerator`.
    pub fn new(distribution_config: PowerLawDistributionConfig, config: SimplePowerLawAccelerationConfig, duration: feb, particle_energy_fraction: feb, delta: feb, pitch_angle_distribution: PitchAngleDistribution) -> Self {
        distribution_config.validate();
        config.validate();

        assert!(duration >= 0.0, "Duration must be larger than or equal to zero.");
        assert!(particle_energy_fraction >= 0.0, "Particle energy fraction must be larger than or equal to zero.");
        assert!(delta > 2.0, "Power-law delta must be larger than two.");

        SimplePowerLawAccelerator{
            distribution_config,
            config,
            duration,
            particle_energy_fraction,
            delta,
            pitch_angle_distribution
        }
    }
}

impl Accelerator for SimplePowerLawAccelerator {
    type DistributionType = PowerLawDistribution;

    fn prepare_snapshot_for_generation<G: Grid3<fdt>>(snapshot: &mut SnapshotCacher3<G>) -> io::Result<()> {
        snapshot.cache_scalar_field("r")?;
        snapshot.cache_scalar_field("tg")?;
        snapshot.cache_scalar_field("qjoule")?;
        snapshot.cache_vector_field("e")
    }

    fn prepare_snapshot_for_propagation<G: Grid3<fdt>>(snapshot: &mut SnapshotCacher3<G>) -> io::Result<()> {
        snapshot.drop_scalar_field("tg");
        snapshot.drop_scalar_field("qjoule");
        snapshot.drop_vector_field("e");
        snapshot.cache_vector_field("b")
    }

    fn generate_distribution<G>(&self, snapshot: &SnapshotCacher3<G>, indices: &Idx3<usize>) -> Option<Self::DistributionType>
    where G: Grid3<fdt>
    {
        let total_power_density = self.compute_total_power_density(snapshot, indices);
        if total_power_density < self.config.min_total_power_density {
            return None
        }
        let total_energy_density = total_power_density*self.duration;

        let temperature = feb::from(snapshot.cached_scalar_field("tg").value(indices));
        assert!(temperature > 0.0, "Temperature must be larger than zero.");

        let mass_density = feb::from(snapshot.cached_scalar_field("r").value(indices))*U_R;  // [g/cm^3]
        let electron_density = PowerLawDistribution::compute_electron_density(mass_density); // [electrons/cm^3]
        assert!(electron_density > 0.0, "Electron density must be larger than zero.");

        let lower_cutoff_energy = match self.estimate_cutoff_energy_from_thermal_intersection(temperature, electron_density, total_energy_density) {
            Some(energy) => energy,
            None => return None
        };

        let acceleration_direction = match self.determine_acceleration_direction(snapshot, indices) {
            Some(direction) => direction,
            None => return None
        };

        let distribution_properties = PowerLawDistributionProperties{
            delta: self.delta,
            pitch_angle_distribution: self.pitch_angle_distribution,
            total_power_density,
            lower_cutoff_energy,
            acceleration_position: snapshot.reader().grid().centers().point(indices),
            acceleration_direction,
            mass_density
        };
        let distribution = PowerLawDistribution::new(self.distribution_config.clone(), distribution_properties);

        if distribution.estimated_depletion_distance() < self.config.min_estimated_depletion_distance {
            None
        } else {
            Some(distribution)
        }
    }
}

impl SimplePowerLawAccelerationConfig {
    const DEFAULT_MIN_TOTAL_POWER_DENSITY:          feb = 1e-2; // [erg/(cm^3 s)]
    const DEFAULT_MIN_ESTIMATED_DEPLETION_DISTANCE: feb = 3e7;  // [cm]
    const DEFAULT_INITIAL_CUTOFF_ENERGY_GUESS:      feb = 4.0;  // [keV]
    const DEFAULT_ACCEPTABLE_ROOT_FINDING_ERROR:    feb = 1e-3;
    const DEFAULT_MAX_ROOT_FINDING_ITERATIONS:      i32 = 100;

    /// Panics if any of the configuration parameter values are invalid.
    pub fn validate(&self) {
        assert!(self.min_total_power_density >= 0.0, "Minimum total power density must be larger than or equal to zero.");
        assert!(self.min_estimated_depletion_distance >= 0.0, "Minimum estimated depletion distance must be larger than or equal to zero.");
        assert!(self.initial_cutoff_energy_guess > 0.0, "Initial cut-off energy guess must be larger than zero.");
        assert!(self.acceptable_root_finding_error > 0.0, "Acceptable root finding error must be larger than zero.");
        assert!(self.max_root_finding_iterations > 0, "Maximum number of root finding iterations must be larger than zero.");
    }
}

impl Default for SimplePowerLawAccelerationConfig {
    fn default() -> Self {
        SimplePowerLawAccelerationConfig {
            min_total_power_density: Self::DEFAULT_MIN_TOTAL_POWER_DENSITY,
            min_estimated_depletion_distance: Self::DEFAULT_MIN_ESTIMATED_DEPLETION_DISTANCE,
            initial_cutoff_energy_guess: Self::DEFAULT_INITIAL_CUTOFF_ENERGY_GUESS,
            acceptable_root_finding_error: Self::DEFAULT_ACCEPTABLE_ROOT_FINDING_ERROR,
            max_root_finding_iterations: Self::DEFAULT_MAX_ROOT_FINDING_ITERATIONS
        }
    }
}
