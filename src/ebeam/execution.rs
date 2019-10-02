//! Execeution of electron beam simulations.

use super::distribution::power_law::acceleration::simple::{
    SimplePowerLawAccelerationConfig, SimplePowerLawAccelerator,
};
use super::distribution::power_law::{PitchAngleDistribution, PowerLawDistributionConfig};
use super::{feb, ElectronBeamSwarm};
use crate::geometry::Dim3;
use crate::grid::hor_regular::HorRegularGrid3;
use crate::grid::Grid3;
use crate::interpolation::poly_fit::{PolyFitInterpolator3, PolyFitInterpolatorConfig};
use crate::io::snapshot::{fdt, SnapshotCacher3, SnapshotReader3};
use crate::io::{Endianness, Verbose};
use crate::tracing::ftr;
use crate::tracing::seeding::criterion::CriterionSeeder3;
use crate::tracing::seeding::IndexSeeder3;
use crate::tracing::stepping::rkf::rkf23::RKF23StepperFactory3;
use crate::tracing::stepping::rkf::rkf45::RKF45StepperFactory3;
use crate::tracing::stepping::rkf::{RKFStepperConfig, RKFStepperType};
use crate::units::solar::{U_E, U_L, U_T};
use std::path;

/// Convenience object for running offline electron beam simulations.
pub struct ElectronBeamSimulator {
    param_file_path: path::PathBuf,
    /// Whether to use a normalized version of the reconnection factor when seeding.
    pub use_normalized_reconnection_factor: bool,
    /// Beams will be generated where the reconnection factor value is larger than this.
    pub reconnection_factor_threshold: fdt,
    /// Smallest depth at which electrons will be accelerated [Mm].
    pub min_acceleration_depth: fdt,
    /// Largest depth at which electrons will be accelerated [Mm].
    pub max_acceleration_depth: fdt,
    /// Configuration parameters for the acceleration model.
    pub accelerator_config: SimplePowerLawAccelerationConfig,
    /// Duration of the acceleration events [s].
    pub acceleration_duration: feb,
    /// Fraction of the released reconnection energy going into acceleration of electrons.
    pub particle_energy_fraction: feb,
    /// Exponent of the inverse power-law describing the non-thermal electron distribution.
    pub power_law_delta: feb,
    /// Type of pitch angle distribution of the non-thermal electrons.
    pub pitch_angle_distribution: PitchAngleDistribution,
    /// Configuration parameters for the electron distribution model.
    pub distribution_config: PowerLawDistributionConfig,
    /// Configuration parameters for the interpolator.
    pub interpolator_config: PolyFitInterpolatorConfig,
    /// Type of stepper to use.
    pub rkf_stepper_type: RKFStepperType,
    /// Configuration parameters for the stepper.
    pub rkf_stepper_config: RKFStepperConfig,
}

impl ElectronBeamSimulator {
    pub const DEFAULT_IGNORE_REJECTION: bool = false;
    pub const DEFAULT_MAX_ACCELERATION_ANGLE: feb = 70.0;
    pub const DEFAULT_INITIAL_CUTOFF_ENERGY_GUESS: feb = 4.0;
    pub const DEFAULT_ACCEPTABLE_ROOT_FINDING_ERROR: feb = 1e-3;
    pub const DEFAULT_MAX_ROOT_FINDING_ITERATIONS: i32 = 100;
    pub const DEFAULT_PITCH_ANGLE_DISTRIBUTION: PitchAngleDistribution =
        PitchAngleDistribution::Peaked;
    pub const DEFAULT_INTERPOLATION_ORDER: usize = 3;
    pub const DEFAULT_VARIATION_THRESHOLD_FOR_LINEAR_INTERPOLATION: f64 = 0.3;
    pub const DEFAULT_RKF_STEPPER_TYPE: RKFStepperType = RKFStepperType::RKF45;
    pub const DEFAULT_MAX_STEP_ATTEMPTS: u32 = 16;
    pub const DEFAULT_ABSOLUTE_TOLERANCE: ftr = 1e-6;
    pub const DEFAULT_RELATIVE_TOLERANCE: ftr = 1e-6;
    pub const DEFAULT_SAFETY_FACTOR: ftr = 0.9;
    pub const DEFAULT_MIN_STEP_SCALE: ftr = 0.2;
    pub const DEFAULT_MAX_STEP_SCALE: ftr = 10.0;
    pub const DEFAULT_INITIAL_ERROR: ftr = 1e-4;
    pub const DEFAULT_INITIAL_STEP_LENGTH: ftr = 1e-4;
    pub const DEFAULT_SUDDEN_REVERSALS_FOR_SINK: u32 = 3;

    /// Creates a new electron beam generator with parameters read from
    /// the given parameter (.idl) file.
    pub fn from_param_file<P: AsRef<path::Path>>(param_file_path: P) -> Self {
        let param_file_path = param_file_path.as_ref().to_path_buf();
        let reader = Self::create_reader(&param_file_path.as_path());

        let use_normalized_reconnection_factor =
            Self::read_use_normalized_reconnection_factor(&reader);
        let reconnection_factor_threshold = Self::read_reconnection_factor_threshold(&reader);
        let min_acceleration_depth = Self::read_min_acceleration_depth(&reader);
        let max_acceleration_depth = Self::read_max_acceleration_depth(&reader);
        let accelerator_config = Self::read_accelerator_config(&reader);
        let acceleration_duration = Self::read_acceleration_duration(&reader);
        let particle_energy_fraction = Self::read_particle_energy_fraction(&reader);
        let power_law_delta = Self::read_power_law_delta(&reader);
        let pitch_angle_distribution = Self::read_pitch_angle_distribution(&reader);
        let distribution_config = Self::read_distribution_config(&reader);
        let interpolator_config = Self::read_interpolator_config(&reader);
        let rkf_stepper_type = Self::read_rkf_stepper_type(&reader);
        let rkf_stepper_config = Self::read_rkf_stepper_config(&reader);

        ElectronBeamSimulator {
            param_file_path,
            use_normalized_reconnection_factor,
            reconnection_factor_threshold,
            min_acceleration_depth,
            max_acceleration_depth,
            accelerator_config,
            acceleration_duration,
            particle_energy_fraction,
            power_law_delta,
            pitch_angle_distribution,
            distribution_config,
            interpolator_config,
            rkf_stepper_type,
            rkf_stepper_config,
        }
    }

    /// Generates a new set of electron beams using the current parameter values.
    pub fn generate_beams(
        &self,
        generate_only: bool,
        extra_fixed_scalars: Option<&Vec<&str>>,
        extra_varying_scalars: Option<&Vec<&str>>,
        verbose: Verbose,
    ) -> ElectronBeamSwarm<SimplePowerLawAccelerator> {
        let mut snapshot = self.create_cacher();
        let seeder = self.create_seeder(&mut snapshot);
        let accelerator = self.create_accelerator();
        let interpolator = self.create_interpolator();
        let mut beams = if generate_only {
            ElectronBeamSwarm::generate_unpropagated(
                seeder,
                &mut snapshot,
                accelerator,
                &interpolator,
                verbose,
            )
        } else {
            match self.rkf_stepper_type {
                RKFStepperType::RKF23 => {
                    let stepper_factory = self.create_rkf23_stepper_factory();
                    ElectronBeamSwarm::generate_propagated(
                        seeder,
                        &mut snapshot,
                        accelerator,
                        &interpolator,
                        stepper_factory,
                        verbose,
                    )
                }
                RKFStepperType::RKF45 => {
                    let stepper_factory = self.create_rkf45_stepper_factory();
                    ElectronBeamSwarm::generate_propagated(
                        seeder,
                        &mut snapshot,
                        accelerator,
                        &interpolator,
                        stepper_factory,
                        verbose,
                    )
                }
            }
        };
        snapshot.drop_all_fields();

        if let Some(extra_fixed_scalars) = extra_fixed_scalars {
            for name in extra_fixed_scalars {
                beams.extract_fixed_scalars(
                    snapshot.obtain_scalar_field(name).unwrap_or_else(|err| {
                        panic!("Could not read {} from snapshot: {}", name, err)
                    }),
                    &interpolator,
                );
                snapshot.drop_scalar_field(name);
            }
        }
        if let Some(extra_varying_scalars) = extra_varying_scalars {
            for name in extra_varying_scalars {
                beams.extract_varying_scalars(
                    snapshot.obtain_scalar_field(name).unwrap_or_else(|err| {
                        panic!("Could not read {} from snapshot: {}", name, err)
                    }),
                    &interpolator,
                );
                snapshot.drop_scalar_field(name);
            }
        }
        beams
    }

    fn read_use_normalized_reconnection_factor<G: Grid3<fdt>>(reader: &SnapshotReader3<G>) -> bool {
        let use_normalized_reconnection_factor: u8 = reader
            .get_numerical_param("norm_krec")
            .unwrap_or_else(|err| panic!("{}", err));
        use_normalized_reconnection_factor > 0
    }

    fn read_reconnection_factor_threshold<G: Grid3<fdt>>(reader: &SnapshotReader3<G>) -> fdt {
        reader
            .get_numerical_param("krec_lim")
            .unwrap_or_else(|err| panic!("{}", err))
    }

    fn read_min_acceleration_depth<G: Grid3<fdt>>(reader: &SnapshotReader3<G>) -> fdt {
        reader
            .get_numerical_param("z_rec_ulim")
            .unwrap_or_else(|err| panic!("{}", err))
    }

    fn read_max_acceleration_depth<G: Grid3<fdt>>(reader: &SnapshotReader3<G>) -> fdt {
        reader
            .get_numerical_param("z_rec_llim")
            .unwrap_or_else(|err| panic!("{}", err))
    }

    fn read_accelerator_config<G: Grid3<fdt>>(
        reader: &SnapshotReader3<G>,
    ) -> SimplePowerLawAccelerationConfig {
        let min_total_power_density = reader
            .get_numerical_param::<feb>("min_beam_en")
            .unwrap_or_else(|err| panic!("{}", err))
            * U_E
            / U_T;

        let min_estimated_depletion_distance = reader
            .get_numerical_param::<feb>("min_stop_dist")
            .unwrap_or_else(|err| panic!("{}", err))
            * U_L;

        SimplePowerLawAccelerationConfig {
            ignore_rejection: Self::DEFAULT_IGNORE_REJECTION,
            min_total_power_density,
            min_estimated_depletion_distance,
            max_acceleration_angle: Self::DEFAULT_MAX_ACCELERATION_ANGLE,
            initial_cutoff_energy_guess: Self::DEFAULT_INITIAL_CUTOFF_ENERGY_GUESS,
            acceptable_root_finding_error: Self::DEFAULT_ACCEPTABLE_ROOT_FINDING_ERROR,
            max_root_finding_iterations: Self::DEFAULT_MAX_ROOT_FINDING_ITERATIONS,
        }
    }

    fn read_acceleration_duration<G: Grid3<fdt>>(reader: &SnapshotReader3<G>) -> feb {
        reader
            .get_numerical_param::<feb>("dt")
            .unwrap_or_else(|err| panic!("{}", err))
            * U_T
    }

    fn read_particle_energy_fraction<G: Grid3<fdt>>(reader: &SnapshotReader3<G>) -> feb {
        reader
            .get_numerical_param("qjoule_acc_frac")
            .unwrap_or_else(|err| panic!("{}", err))
    }

    fn read_power_law_delta<G: Grid3<fdt>>(reader: &SnapshotReader3<G>) -> feb {
        reader
            .get_numerical_param("power_law_index")
            .unwrap_or_else(|err| panic!("{}", err))
    }

    fn read_pitch_angle_distribution<G: Grid3<fdt>>(
        _reader: &SnapshotReader3<G>,
    ) -> PitchAngleDistribution {
        Self::DEFAULT_PITCH_ANGLE_DISTRIBUTION
    }

    fn read_distribution_config<G: Grid3<fdt>>(
        reader: &SnapshotReader3<G>,
    ) -> PowerLawDistributionConfig {
        let min_remaining_power_density = reader
            .get_numerical_param::<feb>("min_stop_en")
            .unwrap_or_else(|err| panic!("{}", err))
            * U_E
            / U_T;

        PowerLawDistributionConfig {
            min_remaining_power_density,
        }
    }

    fn read_interpolator_config<G: Grid3<fdt>>(
        _reader: &SnapshotReader3<G>,
    ) -> PolyFitInterpolatorConfig {
        PolyFitInterpolatorConfig {
            order: Self::DEFAULT_INTERPOLATION_ORDER,
            variation_threshold_for_linear:
                Self::DEFAULT_VARIATION_THRESHOLD_FOR_LINEAR_INTERPOLATION,
        }
    }

    fn read_rkf_stepper_type<G: Grid3<fdt>>(_reader: &SnapshotReader3<G>) -> RKFStepperType {
        Self::DEFAULT_RKF_STEPPER_TYPE
    }

    fn read_rkf_stepper_config<G: Grid3<fdt>>(reader: &SnapshotReader3<G>) -> RKFStepperConfig {
        let dense_step_length = reader
            .get_numerical_param("ds_out")
            .unwrap_or_else(|err| panic!("{}", err));

        let use_pi_control: u8 = reader
            .get_numerical_param("use_pi_ctrl")
            .unwrap_or_else(|err| panic!("{}", err));
        let use_pi_control = use_pi_control > 0;

        RKFStepperConfig {
            dense_step_length,
            max_step_attempts: Self::DEFAULT_MAX_STEP_ATTEMPTS,
            absolute_tolerance: Self::DEFAULT_ABSOLUTE_TOLERANCE,
            relative_tolerance: Self::DEFAULT_RELATIVE_TOLERANCE,
            safety_factor: Self::DEFAULT_SAFETY_FACTOR,
            min_step_scale: Self::DEFAULT_MIN_STEP_SCALE,
            max_step_scale: Self::DEFAULT_MAX_STEP_SCALE,
            initial_error: Self::DEFAULT_INITIAL_ERROR,
            initial_step_length: Self::DEFAULT_INITIAL_STEP_LENGTH,
            sudden_reversals_for_sink: Self::DEFAULT_SUDDEN_REVERSALS_FOR_SINK,
            use_pi_control,
        }
    }

    fn create_reader(param_file_path: &path::Path) -> SnapshotReader3<HorRegularGrid3<fdt>> {
        SnapshotReader3::new(param_file_path, Endianness::Little)
            .unwrap_or_else(|err| panic!("Could not create snapshot reader: {}", err))
    }

    fn create_cacher(&self) -> SnapshotCacher3<HorRegularGrid3<fdt>> {
        Self::create_reader(&self.param_file_path.as_path()).into_cacher()
    }

    fn create_seeder<G: Grid3<fdt>>(&self, snapshot: &mut SnapshotCacher3<G>) -> CriterionSeeder3 {
        let reconnection_factor_variable = if self.use_normalized_reconnection_factor {
            "krec_norm"
        } else {
            "krec"
        };
        let reconnection_factor_field = snapshot
            .obtain_scalar_field(reconnection_factor_variable)
            .unwrap_or_else(|err| panic!("Could not obtain reconnection factor field: {}", err));

        let mut seeder = CriterionSeeder3::on_scalar_field_values(
            reconnection_factor_field,
            &|reconnection_factor| reconnection_factor >= self.reconnection_factor_threshold,
        );

        snapshot.drop_scalar_field(reconnection_factor_variable);

        let z_coordinates = &snapshot.reader().grid().centers()[Dim3::Z];
        seeder.retain_indices(|indices| {
            z_coordinates[indices[Dim3::Z]] >= self.min_acceleration_depth
                && z_coordinates[indices[Dim3::Z]] <= self.max_acceleration_depth
        });
        seeder
    }

    fn create_accelerator(&self) -> SimplePowerLawAccelerator {
        SimplePowerLawAccelerator::new(
            self.distribution_config.clone(),
            self.accelerator_config.clone(),
            self.acceleration_duration,
            self.particle_energy_fraction,
            self.power_law_delta,
            self.pitch_angle_distribution,
        )
    }

    fn create_interpolator(&self) -> PolyFitInterpolator3 {
        PolyFitInterpolator3::new(self.interpolator_config.clone())
    }

    fn create_rkf23_stepper_factory(&self) -> RKF23StepperFactory3 {
        RKF23StepperFactory3::new(self.rkf_stepper_config.clone())
    }

    fn create_rkf45_stepper_factory(&self) -> RKF45StepperFactory3 {
        RKF45StepperFactory3::new(self.rkf_stepper_config.clone())
    }
}
