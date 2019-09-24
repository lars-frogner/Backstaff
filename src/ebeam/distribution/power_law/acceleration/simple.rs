//! Simple model for acceleration of non-thermal electron beams described by power-law distributions.

use std::io;
use rayon::prelude::*;
use serde::Serialize;
use serde::ser::{Serializer, SerializeTuple};
use nrfind;
use crate::constants::{KBOLTZMANN, MC2_ELECTRON, KEV_TO_ERG};
use crate::units::solar::{U_T, U_E, U_R};
use crate::io::snapshot::{fdt, SnapshotCacher3};
use crate::geometry::{Dim3, Vec3, Point3, Idx3};
use crate::grid::Grid3;
use crate::interpolation::Interpolator3;
use crate::tracing::stepping::SteppingSense;
use super::super::{PitchAngleDistribution, PowerLawDistributionConfig, PowerLawDistributionData, PowerLawDistribution};
use super::super::super::super::{feb, ElectronBeamMetadataCollection};
use super::super::super::super::accelerator::Accelerator;
use Dim3::{X, Y, Z};

/// Reason for rejecting a distribution in the simple power-law acceleration model.
#[allow(clippy::enum_variant_names)]
#[derive(Clone, Copy, Debug)]
enum RejectionCause {
    TooLowTotalPowerDensity   = 0b001,
    TooShortDepletionDistance = 0b010,
    TooPerpendicularDirection = 0b100
}

/// Encodes the rejection conditions satisfied when generating a distribution.
#[derive(Clone, Debug, Serialize)]
pub struct RejectionCauseCode(u8);

/// Holds a rejection cause code for each distribution.
#[derive(Clone, Default, Debug)]
pub struct RejectionCauseCodeCollection {
    rejection_cause_codes: Vec<RejectionCauseCode>,
}

type SimplePowerLawAcceleratorMetadataCollection = RejectionCauseCodeCollection;
type SimplePowerLawDistribution = PowerLawDistribution<SimplePowerLawAcceleratorMetadataCollection>;

/// Configuration parameters for the simple power-law acceleration model.
#[derive(Clone, Debug)]
pub struct SimplePowerLawAccelerationConfig {
    /// Whether to abort generation of a distribution whenever a rejection condition is met.
    pub enforce_rejection: bool,
    /// Distributions with total power densities smaller than this value are discarded [erg/(cm^3 s)].
    pub min_total_power_density: feb,
    /// Distributions with an initial estimated depletion distance smaller than this value are discarded [cm].
    pub min_estimated_depletion_distance: feb,
    /// The acceleration direction has to be at least this many degrees
    /// away from the perpendicular direction of the magnetic field in order
    /// for the distribution to be included.
    pub min_acceleration_angle: feb,
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
    /// Cosine of the minimum angle that the acceleration direction must be away from
    /// the normal of the magnetic field direction in order for the electrons to be propagated.
    acceleration_alignment_threshold: fdt,
    pitch_angle_factor: feb,
    duration: feb,
    particle_energy_fraction: feb,
    delta: feb
}

impl RejectionCauseCode {
    fn add_cause(&mut self, cause: RejectionCause) {
        self.0 |= cause as u8;
    }
}

impl Default for RejectionCauseCode {
    fn default() -> Self {
        RejectionCauseCode(0b000)
    }
}

impl ElectronBeamMetadataCollection for RejectionCauseCodeCollection {
    type Item = RejectionCauseCode;
}

impl ParallelExtend<RejectionCauseCode> for RejectionCauseCodeCollection {
    fn par_extend<I>(&mut self, par_iter: I)
    where
        I: IntoParallelIterator<Item=RejectionCauseCode>,
    {
        self.rejection_cause_codes.par_extend(par_iter);
    }
}

impl Serialize for RejectionCauseCodeCollection {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut tup = serializer.serialize_tuple(2)?;
        tup.serialize_element("rejection_cause_code")?;
        tup.serialize_element(&self.rejection_cause_codes)?;
        tup.end()
    }
}

impl SimplePowerLawAccelerator {
    /// How many adjacent grid cells in each direction to include when
    /// computing the average electric field around the acceleration site.
    const ELECTRIC_FIELD_PROBING_SPAN: isize = 2;

    fn determine_total_power_density<G>(&self, snapshot: &SnapshotCacher3<G>, indices: &Idx3<usize>) -> feb
    where G: Grid3<fdt>
    {
        let joule_heating_field = snapshot.cached_scalar_field("qjoule");
        let joule_heating = feb::from(joule_heating_field.value(indices));
        let joule_heating = feb::max(0.0, joule_heating*U_E/U_T); // [erg/(cm^3 s)]

        self.particle_energy_fraction*joule_heating
    }

    fn compute_total_energy_density(&self, total_power_density: feb) -> feb {
        total_power_density*self.duration
    }

    fn determine_temperature<G>(snapshot: &SnapshotCacher3<G>, indices: &Idx3<usize>) -> feb
    where G: Grid3<fdt>
    {
        let temperature = feb::from(snapshot.cached_scalar_field("tg").value(indices));
        assert!(temperature > 0.0, "Temperature must be larger than zero.");
        temperature
    }

    fn determine_mass_density<G>(snapshot: &SnapshotCacher3<G>, indices: &Idx3<usize>) -> feb
    where G: Grid3<fdt>
    {
        feb::from(snapshot.cached_scalar_field("r").value(indices))*U_R // [g/cm^3]
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
    fn compute_lower_cutoff_energy(&self, temperature: feb, electron_density: feb, total_energy_density: feb) -> Option<feb> {
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

    fn determine_acceleration_position<G>(snapshot: &SnapshotCacher3<G>, indices: &Idx3<usize>) -> Point3<fdt>
    where G: Grid3<fdt>
    {
        snapshot.reader().grid().centers().point(indices)
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

    fn determine_magnetic_field_direction<G, I>(snapshot: &SnapshotCacher3<G>, interpolator: &I, acceleration_position: &Point3<fdt>) -> Vec3<fdt>
    where G: Grid3<fdt>,
          I: Interpolator3
    {
        let magnetic_field = snapshot.cached_vector_field("b");
        let mut magnetic_field_direction = interpolator.interp_vector_field(magnetic_field, acceleration_position).expect_inside();
        magnetic_field_direction.normalize();
        magnetic_field_direction
    }

    fn compute_acceleration_alignment_factor(magnetic_field_direction: &Vec3<fdt>, acceleration_direction: &Vec3<fdt>) -> fdt {
        acceleration_direction.dot(magnetic_field_direction)
    }

    fn find_propagation_sense(acceleration_alignment_factor: fdt) -> SteppingSense {
        if acceleration_alignment_factor >= 0.0 {
            SteppingSense::Same
        } else {
            SteppingSense::Opposite
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

        let pitch_angle_factor = SimplePowerLawDistribution::determine_pitch_angle_factor(pitch_angle_distribution);
        let acceleration_alignment_threshold = feb::cos(config.min_acceleration_angle.to_radians()) as fdt;

        assert!(duration >= 0.0, "Duration must be larger than or equal to zero.");
        assert!(particle_energy_fraction >= 0.0, "Particle energy fraction must be larger than or equal to zero.");
        assert!(delta > 2.0, "Power-law delta must be larger than two.");

        SimplePowerLawAccelerator{
            distribution_config,
            config,
            acceleration_alignment_threshold,
            pitch_angle_factor,
            duration,
            particle_energy_fraction,
            delta
        }
    }
}

impl Accelerator for SimplePowerLawAccelerator {
    type DistributionType = SimplePowerLawDistribution;

    fn prepare_snapshot_for_generation<G: Grid3<fdt>>(snapshot: &mut SnapshotCacher3<G>) -> io::Result<()> {
        snapshot.cache_scalar_field("r")?;
        snapshot.cache_scalar_field("tg")?;
        snapshot.cache_scalar_field("qjoule")?;
        snapshot.cache_vector_field("e")?;
        snapshot.cache_vector_field("b")
    }

    fn prepare_snapshot_for_propagation<G: Grid3<fdt>>(snapshot: &mut SnapshotCacher3<G>) -> io::Result<()> {
        snapshot.drop_scalar_field("tg");
        snapshot.drop_scalar_field("qjoule");
        snapshot.drop_vector_field("e");
        Ok(())
    }

    fn generate_distribution<G, I>(&self, snapshot: &SnapshotCacher3<G>, interpolator: &I, indices: &Idx3<usize>) -> Option<Self::DistributionType>
    where G: Grid3<fdt>,
          I: Interpolator3
    {
        let mut rejection_cause_code = RejectionCauseCode::default();

        let total_power_density = self.determine_total_power_density(snapshot, indices);
        if total_power_density < self.config.min_total_power_density {
            if self.config.enforce_rejection {
                return None
            } else {
                rejection_cause_code.add_cause(RejectionCause::TooLowTotalPowerDensity);
            }
        }

        let total_energy_density = self.compute_total_energy_density(total_power_density);

        let temperature = Self::determine_temperature(snapshot, indices);
        assert!(temperature > 0.0, "Temperature must be larger than zero.");

        let mass_density = Self::determine_mass_density(snapshot, indices);
        let electron_density = SimplePowerLawDistribution::compute_electron_density(mass_density);
        assert!(electron_density > 0.0, "Electron density must be larger than zero.");

        let lower_cutoff_energy = match self.compute_lower_cutoff_energy(temperature, electron_density, total_energy_density) {
            Some(energy) => energy,
            None => return None
        };
        let mean_energy = SimplePowerLawDistribution::compute_mean_energy(self.delta, lower_cutoff_energy);

        let estimated_depletion_distance = SimplePowerLawDistribution::estimate_depletion_distance(
            electron_density,
            self.delta,
            self.pitch_angle_factor,
            total_power_density,
            lower_cutoff_energy,
            mean_energy,
            self.distribution_config.min_remaining_power_density
        );

        if estimated_depletion_distance < self.config.min_estimated_depletion_distance {
            if self.config.enforce_rejection {
                return None
            } else {
                rejection_cause_code.add_cause(RejectionCause::TooShortDepletionDistance);
            }
        }

        let acceleration_position = Self::determine_acceleration_position(snapshot, indices);

        let acceleration_direction = match self.determine_acceleration_direction(snapshot, indices) {
            Some(direction) => direction,
            None => return None
        };

        let magnetic_field_direction = Self::determine_magnetic_field_direction(snapshot, interpolator, &acceleration_position);
        let acceleration_alignment_factor = Self::compute_acceleration_alignment_factor(&magnetic_field_direction, &acceleration_direction);
        if fdt::abs(acceleration_alignment_factor) < self.acceleration_alignment_threshold {
            if self.config.enforce_rejection {
                return None
            } else {
                rejection_cause_code.add_cause(RejectionCause::TooPerpendicularDirection);
            }
        }

        let propagation_sense = Self::find_propagation_sense(acceleration_alignment_factor);

        let distribution_data = PowerLawDistributionData{
            delta: self.delta,
            pitch_angle_factor: self.pitch_angle_factor,
            total_power_density,
            lower_cutoff_energy,
            mean_energy,
            estimated_depletion_distance,
            acceleration_position,
            acceleration_direction,
            propagation_sense,
            mass_density,
            metadata: rejection_cause_code
        };
        Some(SimplePowerLawDistribution::new(self.distribution_config.clone(), distribution_data))
    }
}

impl SimplePowerLawAccelerationConfig {
    const DEFAULT_ENFORCE_REJECTION:                bool = true;
    const DEFAULT_MIN_TOTAL_POWER_DENSITY:          feb = 1e-2; // [erg/(cm^3 s)]
    const DEFAULT_MIN_ESTIMATED_DEPLETION_DISTANCE: feb = 3e7;  // [cm]
    const DEFAULT_MIN_ACCELERATION_ANGLE:           feb = 20.0; // [deg]
    const DEFAULT_INITIAL_CUTOFF_ENERGY_GUESS:      feb = 4.0;  // [keV]
    const DEFAULT_ACCEPTABLE_ROOT_FINDING_ERROR:    feb = 1e-3;
    const DEFAULT_MAX_ROOT_FINDING_ITERATIONS:      i32 = 100;

    /// Panics if any of the configuration parameter values are invalid.
    pub fn validate(&self) {
        assert!(self.min_total_power_density >= 0.0, "Minimum total power density must be larger than or equal to zero.");
        assert!(self.min_estimated_depletion_distance >= 0.0, "Minimum estimated depletion distance must be larger than or equal to zero.");
        assert!(self.min_acceleration_angle >= 0.0, "Minimum acceleration angle must be larger than or equal to zero.");
        assert!(self.initial_cutoff_energy_guess > 0.0, "Initial cut-off energy guess must be larger than zero.");
        assert!(self.acceptable_root_finding_error > 0.0, "Acceptable root finding error must be larger than zero.");
        assert!(self.max_root_finding_iterations > 0, "Maximum number of root finding iterations must be larger than zero.");
    }
}

impl Default for SimplePowerLawAccelerationConfig {
    fn default() -> Self {
        SimplePowerLawAccelerationConfig {
            enforce_rejection: Self::DEFAULT_ENFORCE_REJECTION,
            min_total_power_density: Self::DEFAULT_MIN_TOTAL_POWER_DENSITY,
            min_estimated_depletion_distance: Self::DEFAULT_MIN_ESTIMATED_DEPLETION_DISTANCE,
            min_acceleration_angle: Self::DEFAULT_MIN_ACCELERATION_ANGLE,
            initial_cutoff_energy_guess: Self::DEFAULT_INITIAL_CUTOFF_ENERGY_GUESS,
            acceptable_root_finding_error: Self::DEFAULT_ACCEPTABLE_ROOT_FINDING_ERROR,
            max_root_finding_iterations: Self::DEFAULT_MAX_ROOT_FINDING_ITERATIONS
        }
    }
}
