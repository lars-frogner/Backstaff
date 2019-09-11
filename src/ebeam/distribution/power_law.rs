//! Power-law electron distribution.

use nrfind;
use crate::constants::{KBOLTZMANN, MC2_ELECTRON, KEV_TO_ERG};
use crate::units::solar::{U_T, U_E, U_L3};
use crate::io::snapshot::{fdt, SnapshotCacher3};
use crate::geometry::Point3;
use crate::grid::Grid3;
use crate::interpolation::Interpolator3;
use crate::tracing::stepping::SteppingSense;
use super::Distribution;
use super::super::feb;

/// Direction of acceleration of non-thermal electrons relative to the magnetic field.
pub type AccelerationDirection = Option<SteppingSense>;

/// Acceleration event where particles are accelerated.
#[derive(Clone, Debug)]
pub struct AccelerationEvent {
    /// Duration of the acceleration event [s].
    duration: feb,
    /// Temperature at the acceleration site [K].
    temperature: feb,
    /// Total electron number density at the acceleration site [electrons/cm^3].
    electron_density: feb,
    /// Fraction of the total energy release going into acceleration of non-thermal particles.
    particle_energy_fraction: feb,
    /// Total energy per volume going into acceleration of non-thermal particles during the event [erg/cm^3].
    particle_energy_density: feb,
    /// Average energy per volume and time going into acceleration of non-thermal particles [erg/(cm^3 s)].
    particle_power_density: feb,
    /// Direction of acceleration relative to the magnetic field.
    direction: AccelerationDirection
}

/// An electron pitch-angle distribution which is either peaked or isotropic.
#[derive(Clone, Copy, Debug)]
pub enum PitchAngleDistribution {
    Peaked,
    Isotropic
}

/// Configuration parameters for power-law distributions.
#[derive(Clone, Debug)]
pub struct PLDistributionConfig {
    /// Exponent of the inverse power-law.
    pub delta: feb,
    /// Type of pitch angle distribution of the non-thermal electrons.
    pub pitch_angle_distribution: PitchAngleDistribution,
    /// Distributions with total power densities smaller than this value are discarded.
    pub min_total_power_density: feb,
    /// Distributions with remaining power densities smaller than this value are discarded.
    pub min_remaining_power_density: feb,
    /// Distributions with an initial estimated depletion distance smaller than this value are discarded.
    pub min_estimated_depletion_distance: feb,
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
    /// Total energy injected into the distribution per volume and time [erg/(cm^3 s)].
    total_power_density: feb,
    /// Lower cut-off energy [units of electron rest energy].
    lower_cutoff_energy: feb,
    /// Mean energy of the electrons in the distribution [units of electron rest energy].
    mean_energy: feb,
    /// Current collisional depth [dimensionless]. Increases as the distribution propagates.
    collisional_depth: feb,
    /// Current remaining energy per volume and time [erg/(cm^3 s)]. Decreases as the distribution propagates.
    remaining_power_density: feb
}

/// Holds the deposited power density after propagating the electron distribution
/// and additionally specifies whether the distribution is depleted.
pub enum PropagationResult {
    Ok(feb),
    Depleted(feb)
}

/// Generator for acceleration events with the same duration and particle energy fraction.
#[derive(Clone, Debug)]
pub struct AccelerationEventGenerator {
    /// Duration of the acceleration events [s].
    duration: feb,
    /// Fraction of the total energy release going into acceleration of non-thermal particles.
    particle_energy_fraction: feb
}

/// Generator for power-law distributions with the same configuration parameters.
#[derive(Clone, Debug)]
pub struct PLDistributionGenerator {
    config: PLDistributionConfig
}

impl AccelerationEvent {
    fn new<G, I>(snapshot: &mut SnapshotCacher3<G>, interpolator: &I, position: &Point3<fdt>, duration: feb, particle_energy_fraction: feb) -> Self
    where G: Grid3<fdt>,
          I: Interpolator3
    {
        assert!(duration >= 0.0, "Duration must be larger than or equal to zero.");
        assert!(particle_energy_fraction >= 0.0, "Particle energy fraction must be larger than or equal to zero.");

        let temperature = feb::from(interpolator.interp_scalar_field(snapshot.expect_scalar_field("tg"), position).expect_inside());
        let electron_density = feb::from(interpolator.interp_scalar_field(snapshot.expect_scalar_field("ne"), position).expect_inside());

        let electron_density = electron_density/U_L3; // [1/cm^3]

        assert!(temperature > 0.0, "Temperature must be larger than zero.");
        assert!(electron_density > 0.0, "Electron density must be larger than zero.");

        let particle_power_density = Self::compute_particle_power_density(snapshot, interpolator, position, particle_energy_fraction);
        let particle_energy_density = particle_power_density*duration;

        let direction = Self::determine_acceleration_direction(snapshot, interpolator, position);

        AccelerationEvent{
            duration,
            temperature,
            electron_density,
            particle_energy_fraction,
            particle_energy_density,
            particle_power_density,
            direction
        }
    }

    /// Returns the duration of the acceleration event [s].
    pub fn duration(&self) -> feb { self.duration }

    /// Returns the temperature at the acceleration site [K].
    pub fn temperature(&self) -> feb { self.temperature }

    /// Returns the total electron number density at the acceleration site [electrons/cm^3].
    pub fn electron_density(&self) -> feb { self.electron_density }

    /// Returns the fraction of the total energy release
    /// going into acceleration of non-thermal particles.
    pub fn particle_energy_fraction(&self) -> feb { self.particle_energy_fraction }

    /// Returns the total energy per volume going into acceleration
    /// of non-thermal particles during the event [erg/cm^3].
    pub fn particle_energy_density(&self) -> feb { self.particle_energy_density }

    /// Returns the average energy per volume and time going into
    /// acceleration of non-thermal particles [erg/(cm^3 s)].
    pub fn particle_power_density(&self) -> feb { self.particle_power_density }

    fn compute_particle_power_density<G, I>(snapshot: &mut SnapshotCacher3<G>, interpolator: &I, position: &Point3<fdt>, particle_energy_fraction: feb) -> feb
    where G: Grid3<fdt>,
          I: Interpolator3
    {
        let joule_heating_field = snapshot.expect_scalar_field("qjoule");
        let joule_heating = feb::from(interpolator.interp_scalar_field(joule_heating_field, position).expect_inside());
        let joule_heating = joule_heating*U_E/U_T; // [erg/(cm^3 s)]

        assert!(joule_heating >= 0.0, "Joule heating must be larger than or equal to zero.");

        particle_energy_fraction*joule_heating
    }

    fn determine_acceleration_direction<G, I>(snapshot: &mut SnapshotCacher3<G>, interpolator: &I, position: &Point3<fdt>) -> AccelerationDirection
    where G: Grid3<fdt>,
          I: Interpolator3
    {
        None
    }
}

impl PLDistribution {
    /// 10^40 * 2*pi*( (classical electron radius) [10^8 cm] )^2
    const COLLISION_SCALE: feb = 4.989_344e-1;
    /// 1/2*ln( (2*pi*me*c/h)^3/(pi*alpha) [(10^8 cm)^(-3)] )
    const COULOMB_OFFSET: feb = 65.4848;

     fn from_acceleration_event(config: PLDistributionConfig, event: &AccelerationEvent) -> Option<Self> {
        config.validate();

        let pitch_angle_factor = match config.pitch_angle_distribution {
            PitchAngleDistribution::Peaked => 2.0,
            PitchAngleDistribution::Isotropic => 4.0
        };
        let total_power_density = event.particle_power_density();
        if total_power_density < config.min_total_power_density {
            return None
        }
        let lower_cutoff_energy = match Self::estimate_cutoff_energy_from_thermal_intersection(&config, event) {
            Some(energy) => energy,
            None => return None
        };
        let mean_energy = lower_cutoff_energy*(config.delta - 1.0)/(config.delta - 2.0);

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
            total_power_density,
            lower_cutoff_energy,
            mean_energy,
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

    /// Propagates the electron distribution for a given distance through a region
    /// with the given average electron density and returns the power density deposited
    /// over the distance.
    pub fn propagate<G, I>(&mut self, average_electron_density: feb, distance: feb) -> PropagationResult
    where G: Grid3<fdt>,
          I: Interpolator3
    {
        let collisional_depth_increase = Self::compute_collisional_depth_increase(
            average_electron_density,
            self.pitch_angle_factor,
            self.mean_energy,
            distance
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

        self.collisional_depth = new_collisional_depth;
        self.remaining_power_density = new_remaining_power_density;

        if self.remaining_power_density >= self.config.min_remaining_power_density {
            PropagationResult::Ok(deposited_power_density)
        } else {
            PropagationResult::Depleted(deposited_power_density)
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
    fn estimate_cutoff_energy_from_thermal_intersection(config: &PLDistributionConfig, event: &AccelerationEvent) -> Option<feb> {
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

    fn compute_collisional_depth_increase(electron_density: feb, pitch_angle_factor: feb, mean_energy: feb, distance: feb) -> feb {
        pitch_angle_factor*Self::compute_collisional_coef(electron_density, mean_energy)*distance
    }

    fn compute_remaining_power_density(delta: feb, total_power_density: feb, lower_cutoff_energy: feb, collisional_depth: feb) -> feb {
        total_power_density*feb::powf(1.0 + collisional_depth/feb::powi(lower_cutoff_energy, 2), 1.0 - 0.5*delta)
    }

    fn compute_collisional_coef(electron_density: feb, electron_energy: feb) -> feb {
        Self::COLLISION_SCALE*electron_density*Self::compute_coulomb_logarithm(electron_density, electron_energy)
    }

    fn compute_coulomb_logarithm(particle_density: feb, electron_energy: feb) -> feb {
        Self::COULOMB_OFFSET + 0.5*(feb::ln(feb::powi(electron_energy*(electron_energy + 2.0), 2)/particle_density))
    }
}

impl PLDistributionConfig {
    const DEFAULT_DELTA: feb = 4.0;
    const DEFAULT_PITCH_ANGLE_DISTRIBUTION: PitchAngleDistribution = PitchAngleDistribution::Peaked;
    const DEFAULT_MIN_TOTAL_POWER_DENSITY: feb = 1e-5;
    const DEFAULT_MIN_REMAINING_POWER_DENSITY: feb = 1e-9;
    const DEFAULT_MIN_ESTIMATED_DEPLETION_DISTANCE: feb = 0.3;
    const DEFAULT_INITIAL_CUTOFF_ENERGY_GUESS: feb = 4.0;
    const DEFAULT_ACCEPTABLE_ROOT_FINDING_ERROR: feb = 1e-3;
    const DEFAULT_MAX_ROOT_FINDING_ITERATIONS: i32 = 100;

    /// Creates a new configuration struct with the default values.
    pub fn default() -> Self {
        PLDistributionConfig {
            delta: Self::DEFAULT_DELTA,
            pitch_angle_distribution: Self::DEFAULT_PITCH_ANGLE_DISTRIBUTION,
            min_total_power_density: Self::DEFAULT_MIN_TOTAL_POWER_DENSITY,
            min_remaining_power_density: Self::DEFAULT_MIN_REMAINING_POWER_DENSITY,
            min_estimated_depletion_distance: Self::DEFAULT_MIN_ESTIMATED_DEPLETION_DISTANCE,
            initial_cutoff_energy_guess: Self::DEFAULT_INITIAL_CUTOFF_ENERGY_GUESS,
            acceptable_root_finding_error: Self::DEFAULT_ACCEPTABLE_ROOT_FINDING_ERROR,
            max_root_finding_iterations: Self::DEFAULT_MAX_ROOT_FINDING_ITERATIONS
        }
    }

    fn validate(&self) {
        assert!(self.delta > 2.0, "Delta-exponent must be larger than two.");
        assert!(self.min_total_power_density >= 0.0, "Minimum total power density must be larger than or equal to zero.");
        assert!(self.initial_cutoff_energy_guess > 0.0, "Initial cut-off energy guess must be larger than zero.");
        assert!(self.acceptable_root_finding_error > 0.0, "Acceptable root finding error must be larger than zero.");
        assert!(self.max_root_finding_iterations > 0, "Maximum number of root finding iterations must be larger than zero.");
    }
}

impl AccelerationEventGenerator {
    /// Creates a new generator for acceleration events with the given
    /// duration and particle energy fraction.
    pub fn new(duration: feb, particle_energy_fraction: feb) -> Self {
        AccelerationEventGenerator{ duration, particle_energy_fraction }
    }

    /// Generates a new acceleration event at the given position in the given snapshot.
    pub fn generate<G, I>(&self, snapshot: &mut SnapshotCacher3<G>, interpolator: &I, position: &Point3<fdt>) -> AccelerationEvent
    where G: Grid3<fdt>,
        I: Interpolator3
    {
        AccelerationEvent::new(snapshot, interpolator, position, self.duration, self.particle_energy_fraction)
    }
}

impl PLDistributionGenerator {
    /// Creates a new generator for power-law distributions with the given
    /// configuration parameters.
    pub fn new(config: PLDistributionConfig) -> Self {
        PLDistributionGenerator{ config }
    }

    /// Generates a new power-law distribution from the given acceleration event.
    ///
    /// Returns `None` if the distribution could not be generated.
    pub fn generate_from_acceleration_event(&self, event: &AccelerationEvent) -> Option<PLDistribution> {
        PLDistribution::from_acceleration_event(self.config.clone(), event)
    }
}

impl Distribution for PLDistribution {}
