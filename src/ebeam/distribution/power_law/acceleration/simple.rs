//! Simple model for acceleration of non-thermal electron beams described by power-law distributions.

use std::io;
use std::sync::Arc;
use nrfind;
use crate::constants::{KBOLTZMANN, MC2_ELECTRON, KEV_TO_ERG};
use crate::units::solar::{U_T, U_E, U_L3};
use crate::io::snapshot::{fdt, SnapshotCacher3};
use crate::geometry::{Dim3, In3D, Vec3, Point3};
use crate::grid::Grid3;
use crate::grid::regular::RegularGrid3;
use crate::interpolation::Interpolator3;
use super::super::{PitchAngleDistribution, PowerLawDistributionConfig, PowerLawDistributionProperties, PowerLawDistribution};
use super::super::super::super::feb;
use super::super::super::super::accelerator::Accelerator;
use Dim3::{X, Y, Z};

/// Configuration parameters for the simple power-law acceleration model.
#[derive(Clone, Debug)]
pub struct SimplePowerLawAccelerationConfig {
    /// Distributions with total power densities smaller than this value are discarded [erg/(cm^3 s)].
    pub min_total_power_density: feb,
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
    extent: fdt,
    duration: feb,
    particle_energy_fraction: feb,
    delta: feb,
    pitch_angle_distribution: PitchAngleDistribution
}

impl SimplePowerLawAccelerator {
    /// Spatial resolution of the acceleration event (number of points in each direction).
    const RESOLUTION: usize = 5;

    fn compute_total_power_density<G, I>(&self, snapshot: &SnapshotCacher3<G>, interpolator: &I, position: &Point3<fdt>) -> feb
    where G: Grid3<fdt>,
          I: Interpolator3
    {
        let joule_heating_field = snapshot.cached_scalar_field("qjoule");
        let joule_heating = feb::from(interpolator.interp_scalar_field(joule_heating_field, position).expect_inside());
        let joule_heating = joule_heating*U_E/U_T; // [erg/(cm^3 s)]

        assert!(joule_heating >= 0.0, "Joule heating must be larger than or equal to zero.");

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
                               /(total_energy_density*(self.delta - 1.0));         // [1/keV^(3/2)]

        let difference = |energy| thermal_fraction*energy*feb::exp(-beta*energy) - 1.0;
        let derivative = |energy| thermal_fraction*(1.0 - beta*energy)*feb::exp(-beta*energy);

        let intersection_energy = match nrfind::find_root(&difference,
                                                          &derivative,
                                                          self.config.initial_cutoff_energy_guess,
                                                          self.config.acceptable_root_finding_error,
                                                          self.config.max_root_finding_iterations)
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

    fn determine_acceleration_direction<G, I>(&self, snapshot: &SnapshotCacher3<G>, interpolator: &I, position: &Point3<fdt>) -> Option<Vec3<fdt>>
    where G: Grid3<fdt>,
          I: Interpolator3
    {
        let grid = Arc::new(self.construct_grid(position));
        let local_electric_field = snapshot.cached_vector_field("e").resampled_to_grid(grid, interpolator);

        let total_electric_vector = Vec3::new(
            local_electric_field.values(X).sum(),
            local_electric_field.values(Y).sum(),
            local_electric_field.values(Z).sum()
        );
        let squared_total_electric_vector = total_electric_vector.squared_length();

        if squared_total_electric_vector > std::f32::EPSILON {
            // Electrons are accelerated in the opposite direction as the electric field.
            Some(total_electric_vector/(-fdt::sqrt(squared_total_electric_vector)))
        } else {
            None
        }
    }

    fn construct_grid(&self, position: &Point3<fdt>) -> RegularGrid3<fdt> {
        let shape = In3D::same(Self::RESOLUTION);
        let extent_vec = Vec3::equal_components(0.5*self.extent);
        let lower_bounds = position - &extent_vec;
        let upper_bounds = position + &extent_vec;
        let is_periodic = In3D::same(false);
        RegularGrid3::from_bounds(shape, lower_bounds.to_vec3(), upper_bounds.to_vec3(), is_periodic)
    }
}

impl SimplePowerLawAccelerator {
    /// Creates a new simple power-law accelerator.
    ///
    /// # Parameters
    ///
    /// - `distribution_config`: Configuration parameters for the distribution.
    /// - `config`: Configuration parameters for the accelerator.
    /// - `extent`: Spatial extent of the acceleration event [cm].
    /// - `duration`: Duration of the acceleration events [s].
    /// - `particle_energy_fraction`: Fraction of the total energy release going into acceleration of non-thermal particles.
    /// - `delta`: Exponent of the inverse power-law.
    /// - `pitch_angle_distribution`: Type of pitch angle distribution of the non-thermal electrons.
    ///
    /// # Returns
    ///
    /// A new `SimplePowerLawAccelerator`.
    pub fn new(distribution_config: PowerLawDistributionConfig, config: SimplePowerLawAccelerationConfig, extent: fdt, duration: feb, particle_energy_fraction: feb, delta: feb, pitch_angle_distribution: PitchAngleDistribution) -> Self {
        distribution_config.validate();
        config.validate();

        assert!(extent > 0.0, "Extent must be larger than zero.");
        assert!(duration >= 0.0, "Duration must be larger than or equal to zero.");
        assert!(particle_energy_fraction >= 0.0, "Particle energy fraction must be larger than or equal to zero.");
        assert!(delta > 2.0, "Power-law delta must be larger than two.");

        SimplePowerLawAccelerator{
            distribution_config,
            config,
            extent,
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
        snapshot.cache_scalar_field("tg")?;
        snapshot.cache_scalar_field("ne")?;
        snapshot.cache_scalar_field("qjoule")?;
        snapshot.cache_vector_field("e")
    }

    fn prepare_snapshot_for_propagation<G: Grid3<fdt>>(snapshot: &mut SnapshotCacher3<G>) -> io::Result<()> {
        snapshot.drop_scalar_field("tg");
        snapshot.drop_scalar_field("qjoule");
        snapshot.drop_vector_field("e");
        snapshot.cache_vector_field("b")
    }

    fn generate_distribution<G, I>(&self, snapshot: &SnapshotCacher3<G>, interpolator: &I, position: &Point3<fdt>) -> Option<Self::DistributionType>
    where G: Grid3<fdt>,
          I: Interpolator3
    {
        let total_power_density = self.compute_total_power_density(snapshot, interpolator, &position);
        if total_power_density < self.config.min_total_power_density {
            return None
        }
        let total_energy_density = total_power_density*self.duration;

        let temperature = feb::from(interpolator.interp_scalar_field(snapshot.cached_scalar_field("tg"), position).expect_inside());
        assert!(temperature > 0.0, "Temperature must be larger than zero.");

        let electron_density = interpolator.interp_scalar_field(snapshot.cached_scalar_field("ne"), position).expect_inside();
        let electron_density = feb::from(electron_density)/U_L3; // [electrons/cm^3]
        assert!(electron_density > 0.0, "Electron density must be larger than zero.");

        let lower_cutoff_energy = match self.estimate_cutoff_energy_from_thermal_intersection(temperature, electron_density, total_energy_density) {
            Some(energy) => energy,
            None => return None
        };

        let acceleration_direction = match self.determine_acceleration_direction(snapshot, interpolator, position) {
            Some(direction) => direction,
            None => return None
        };

        let distribution_properties = PowerLawDistributionProperties{
            delta: self.delta,
            pitch_angle_distribution: self.pitch_angle_distribution,
            total_power_density,
            lower_cutoff_energy,
            acceleration_position: position.clone(),
            acceleration_direction,
            electron_density
        };
        PowerLawDistribution::new(self.distribution_config.clone(), distribution_properties)
    }
}

impl SimplePowerLawAccelerationConfig {
    const DEFAULT_MIN_TOTAL_POWER_DENSITY:       feb = 1e-2; // [erg/(cm^3 s)]
    const DEFAULT_INITIAL_CUTOFF_ENERGY_GUESS:   feb = 4.0;  // [keV]
    const DEFAULT_ACCEPTABLE_ROOT_FINDING_ERROR: feb = 1e-3;
    const DEFAULT_MAX_ROOT_FINDING_ITERATIONS:   i32 = 100;

    /// Creates a new configuration struct with the default values.
    pub fn default() -> Self {
        SimplePowerLawAccelerationConfig {
            min_total_power_density: Self::DEFAULT_MIN_TOTAL_POWER_DENSITY,
            initial_cutoff_energy_guess: Self::DEFAULT_INITIAL_CUTOFF_ENERGY_GUESS,
            acceptable_root_finding_error: Self::DEFAULT_ACCEPTABLE_ROOT_FINDING_ERROR,
            max_root_finding_iterations: Self::DEFAULT_MAX_ROOT_FINDING_ITERATIONS
        }
    }

    /// Panics if any of the configuration parameter values are invalid.
    pub fn validate(&self) {
        assert!(self.min_total_power_density >= 0.0, "Minimum total power density must be larger than or equal to zero.");
        assert!(self.initial_cutoff_energy_guess > 0.0, "Initial cut-off energy guess must be larger than zero.");
        assert!(self.acceptable_root_finding_error > 0.0, "Acceptable root finding error must be larger than zero.");
        assert!(self.max_root_finding_iterations > 0, "Maximum number of root finding iterations must be larger than zero.");
    }
}
