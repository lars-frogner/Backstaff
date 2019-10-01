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
use crate::tracing::seeding::criterion::CriterionSeeder3;
use crate::tracing::seeding::IndexSeeder3;
use crate::tracing::stepping::rkf::rkf23::RKF23StepperFactory3;
use crate::tracing::stepping::rkf::rkf45::RKF45StepperFactory3;
use crate::tracing::stepping::rkf::{RKFStepperConfig, RKFStepperType};
use crate::units::solar::{U_E, U_L, U_T};
use clap::{App, Arg, ArgMatches, SubCommand};
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

        // Online version always uses 70 degrees
        let max_acceleration_angle = 70.0;

        // Online version always uses 4 keV
        let initial_cutoff_energy_guess = 4.0;

        let acceptable_root_finding_error = 1e-3;
        let max_root_finding_iterations = 100;

        SimplePowerLawAccelerationConfig {
            ignore_rejection: false,
            min_total_power_density,
            min_estimated_depletion_distance,
            max_acceleration_angle,
            initial_cutoff_energy_guess,
            acceptable_root_finding_error,
            max_root_finding_iterations,
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
        // Online version always uses a peaked distribution
        PitchAngleDistribution::Peaked
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
        // Online version always 3'rd order
        let order = 3;
        PolyFitInterpolatorConfig {
            order,
            ..PolyFitInterpolatorConfig::default()
        }
    }

    fn read_rkf_stepper_type<G: Grid3<fdt>>(_reader: &SnapshotReader3<G>) -> RKFStepperType {
        // Online version typically uses 5th order stepper
        RKFStepperType::RKF45
    }

    fn read_rkf_stepper_config<G: Grid3<fdt>>(reader: &SnapshotReader3<G>) -> RKFStepperConfig {
        let dense_step_length = reader
            .get_numerical_param("ds_out")
            .unwrap_or_else(|err| panic!("{}", err));

        // The following values are always used in the online version
        let max_step_attempts = 16;
        let absolute_tolerance = 1e-6;
        let relative_tolerance = 1e-6;
        let safety_factor = 0.9;
        let min_step_scale = 0.2;
        let max_step_scale = 10.0;
        let initial_step_length = 1e-4;
        let initial_error = 1e-4;
        let sudden_reversals_for_sink = 3;

        let use_pi_control: u8 = reader
            .get_numerical_param("use_pi_ctrl")
            .unwrap_or_else(|err| panic!("{}", err));
        let use_pi_control = use_pi_control > 0;

        RKFStepperConfig {
            dense_step_length,
            max_step_attempts,
            absolute_tolerance,
            relative_tolerance,
            safety_factor,
            min_step_scale,
            max_step_scale,
            initial_step_length,
            initial_error,
            sudden_reversals_for_sink,
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

    fn add_arguments_to_subcommand<'a, 'b>(app: App<'a, 'b>) -> App<'a, 'b> {
        app.arg(
            Arg::with_name("use-normalized-reconnection-factor")
                .long("use-normalized-reconnection-factor")
                .help("Use a normalized version of the reconnection factor when seeding"),
        )
        .arg(
            Arg::with_name("reconnection-factor-threshold")
                .long("reconnection-factor-threshold")
                .value_name("VALUE")
                .help("Beams will be generated where the reconnection factor value is larger than this")
                .next_line_help(true)
                .takes_value(true),
        )
        .arg(
            Arg::with_name("min-acceleration-depth")
                .long("min-acceleration-depth")
                .value_name("VALUE")
                .help("Smallest depth at which electrons will be accelerated [Mm]")
                .next_line_help(true)
                .takes_value(true),
        )
        .arg(
            Arg::with_name("max-acceleration-depth")
                .long("max-acceleration-depth")
                .value_name("VALUE")
                .help("Largest depth at which electrons will be accelerated [Mm]")
                .next_line_help(true)
                .takes_value(true),
        )
        .arg(
            Arg::with_name("ignore-rejection")
                .long("ignore-rejection")
                .help("Generate beams even when they meet a rejection condition"),
        )
        .arg(
            Arg::with_name("min-total-power-density")
                .long("min-total-power-density")
                .value_name("VALUE")
                .help("Distributions with total power densities smaller than this value are discarded [erg/(cm^3 s)]")
                .next_line_help(true)
                .takes_value(true),
        )
        .arg(
            Arg::with_name("min-estimated-depletion-distance")
                .long("min-estimated-depletion-distance")
                .value_name("VALUE")
                .help("Distributions with an initial estimated depletion distance smaller than this value are discarded [cm]")
                .next_line_help(true)
                .takes_value(true),
        )
        .arg(
            Arg::with_name("max-acceleration-angle")
                .long("max-acceleration-angle")
                .value_name("VALUE")
                .help("Distributions with acceleration directions angled more than this away from the magnetic field axis are discarded [deg]")
                .next_line_help(true)
                .takes_value(true),
        )
        .arg(
            Arg::with_name("acceleration-duration")
                .long("acceleration-duration")
                .value_name("VALUE")
                .help("Duration of the acceleration events [s]")
                .next_line_help(true)
                .takes_value(true),
        )
        .arg(
            Arg::with_name("particle-energy-fraction")
                .long("particle-energy-fraction")
                .value_name("VALUE")
                .help("Fraction of the released reconnection energy going into acceleration of electrons")
                .next_line_help(true)
                .takes_value(true),
        )
        .arg(
            Arg::with_name("power-law-delta")
                .long("power-law-delta")
                .value_name("VALUE")
                .help("Exponent of the inverse power-law describing the non-thermal electron distribution")
                .next_line_help(true)
                .takes_value(true),
        )
        .arg(
            Arg::with_name("min-remaining-power-density")
                .long("min-remaining-power-density")
                .value_name("VALUE")
                .help("Distributions with remaining power densities smaller than this value are discarded [erg/(cm^3 s)]")
                .next_line_help(true)
                .takes_value(true),
        )
    }

    fn configure_from_arguments(&mut self, arguments: &ArgMatches) {
        self.use_normalized_reconnection_factor =
            arguments.is_present("use-normalized-reconnection-factor");

        if let Some(reconnection_factor_threshold) =
            arguments.value_of("reconnection-factor-threshold")
        {
            self.reconnection_factor_threshold =
                reconnection_factor_threshold.parse().unwrap_or_else(|err| {
                    panic!(
                        "Could not parse value of reconnection-factor-threshold: {}",
                        err
                    )
                });
        }

        if let Some(min_acceleration_depth) = arguments.value_of("min-acceleration-depth") {
            self.min_acceleration_depth = min_acceleration_depth.parse().unwrap_or_else(|err| {
                panic!("Could not parse value of min-acceleration-depth: {}", err)
            });
        }

        if let Some(max_acceleration_depth) = arguments.value_of("max-acceleration-depth") {
            self.max_acceleration_depth = max_acceleration_depth.parse().unwrap_or_else(|err| {
                panic!("Could not parse value of max-acceleration-depth: {}", err)
            });
        }

        self.accelerator_config.ignore_rejection = arguments.is_present("ignore-rejection");

        if let Some(min_total_power_density) = arguments.value_of("min-total-power-density") {
            self.accelerator_config.min_total_power_density =
                min_total_power_density.parse().unwrap_or_else(|err| {
                    panic!("Could not parse value of min-total-power-density: {}", err)
                });
        }

        if let Some(min_estimated_depletion_distance) =
            arguments.value_of("min-estimated-depletion-distance")
        {
            self.accelerator_config.min_estimated_depletion_distance =
                min_estimated_depletion_distance
                    .parse()
                    .unwrap_or_else(|err| {
                        panic!(
                            "Could not parse value of min-estimated-depletion-distance: {}",
                            err
                        )
                    });
        }

        if let Some(max_acceleration_angle) = arguments.value_of("max-acceleration-angle") {
            self.accelerator_config.max_acceleration_angle =
                max_acceleration_angle.parse().unwrap_or_else(|err| {
                    panic!("Could not parse value of max-acceleration-angle: {}", err)
                });
        }

        if let Some(acceleration_duration) = arguments.value_of("acceleration-duration") {
            self.acceleration_duration = acceleration_duration.parse().unwrap_or_else(|err| {
                panic!("Could not parse value of acceleration-duration: {}", err)
            });
        }

        if let Some(particle_energy_fraction) = arguments.value_of("particle-energy-fraction") {
            self.particle_energy_fraction =
                particle_energy_fraction.parse().unwrap_or_else(|err| {
                    panic!("Could not parse value of particle-energy-fraction: {}", err)
                });
        }

        if let Some(power_law_delta) = arguments.value_of("power-law-delta") {
            self.power_law_delta = power_law_delta
                .parse()
                .unwrap_or_else(|err| panic!("Could not parse value of power-law-delta: {}", err));
        }

        if let Some(min_remaining_power_density) = arguments.value_of("min-remaining-power-density")
        {
            self.distribution_config.min_remaining_power_density =
                min_remaining_power_density.parse().unwrap_or_else(|err| {
                    panic!(
                        "Could not parse value of min-remaining-power-density: {}",
                        err
                    )
                });
        }
    }
}

/// Builds a representation of the `ebeam-simulate` command line subcommand.
pub fn build_subcommand_simulate<'a, 'b>() -> App<'a, 'b> {
    let app = SubCommand::with_name("simulate")
        .about("Simulates electron beams in a Bifrost snapshot")
        .arg(
            Arg::with_name("PARAM_PATH")
                .help("Path to the parameter (.idl) file for the snapshot")
                .required(true)
                .index(1),
        )
        .arg(
            Arg::with_name("output-path")
                .short("o")
                .long("output-path")
                .value_name("PATH")
                .help("Path where the beam data should be saved")
                .next_line_help(true)
                .takes_value(true),
        )
        .arg(
            Arg::with_name("generate-only")
                .short("g")
                .long("generate-only")
                .help("Do not propagate the generated beams"),
        )
        .arg(
            Arg::with_name("extra-fixed-scalars")
                .long("extra-fixed-scalars")
                .value_name("NAMES")
                .help("List of scalar fields to extract at acceleration sites")
                .next_line_help(true)
                .takes_value(true)
                .multiple(true),
        )
        .arg(
            Arg::with_name("extra-varying-scalars")
                .long("extra-varying-scalars")
                .value_name("NAMES")
                .help("List of scalar fields to extract along beam trajectories")
                .next_line_help(true)
                .takes_value(true)
                .multiple(true),
        )
        .arg(
            Arg::with_name("verbose")
                .short("v")
                .long("verbose")
                .help("Print status messages"),
        );

    ElectronBeamSimulator::add_arguments_to_subcommand(app)
}

/// Runs the actions for the `ebeam-simulate` subcommand using the given arguments.
pub fn run_subcommand_simulate(arguments: &ArgMatches) {
    let param_file_path = arguments
        .value_of("PARAM_PATH")
        .expect("Required argument not present.");

    let possible_output_path = arguments.value_of("output-path");

    let generate_only = arguments.is_present("generate-only");

    let extra_fixed_scalars = arguments
        .values_of("extra-fixed-scalars")
        .map(|values| values.collect());
    let extra_varying_scalars = arguments
        .values_of("extra-varying-scalars")
        .map(|values| values.collect());

    let verbose = arguments.is_present("verbose").into();

    let mut simulator = ElectronBeamSimulator::from_param_file(param_file_path);

    simulator.configure_from_arguments(&arguments);

    let beams = simulator.generate_beams(
        generate_only,
        extra_fixed_scalars.as_ref(),
        extra_varying_scalars.as_ref(),
        verbose,
    );

    if let Some(output_path) = possible_output_path {
        beams
            .save_as_combined_pickles(output_path)
            .unwrap_or_else(|err| panic!("Could not save output data: {}", err));
    }
}
