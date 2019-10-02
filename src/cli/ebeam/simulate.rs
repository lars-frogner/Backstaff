//! Command line interface for simulating electron beams.

use crate::cli;
use crate::ebeam::execution::ElectronBeamSimulator;
use crate::tracing::stepping::rkf::RKFStepperType;
use clap::{App, Arg, ArgMatches, SubCommand};

/// Builds a representation of the `ebeam-simulate` command line subcommand.
pub fn build_subcommand_simulate<'a, 'b>() -> App<'a, 'b> {
    let app = SubCommand::with_name("simulate")
        .about("Simulates electron beams in a Bifrost snapshot")
        .arg(
            Arg::with_name("PARAM_PATH")
                .help("Path to the parameter (.idl) file for the snapshot")
                .required(true)
                .takes_value(true)
                .index(1),
        )
        .arg(
            Arg::with_name("output-path")
                .short("o")
                .long("output-path")
                .value_name("PATH")
                .help("Path where the beam data should be saved (no output if absent)")
                .next_line_help(true)
                .takes_value(true),
        )
        .arg(
            Arg::with_name("output-format")
                .short("f")
                .long("output-format")
                .value_name("FORMAT")
                .help("Format to use for saving beam data")
                .next_line_help(true)
                .takes_value(true)
                .possible_values(&["pickle", "json"])
                .default_value("pickle"),
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

    add_simulator_arguments_to_subcommand(app)
}

/// Runs the actions for the `ebeam-simulate` subcommand using the given arguments.
pub fn run_subcommand_simulate(arguments: &ArgMatches) {
    let param_file_path = arguments
        .value_of("PARAM_PATH")
        .expect("Required argument not present.");

    let possible_output_path = arguments.value_of("output-path");

    let output_format = arguments
        .value_of("output-format")
        .expect("No value for argument with default.");

    let generate_only = arguments.is_present("generate-only");

    let extra_fixed_scalars = arguments
        .values_of("extra-fixed-scalars")
        .map(|values| values.collect());
    let extra_varying_scalars = arguments
        .values_of("extra-varying-scalars")
        .map(|values| values.collect());

    let verbose = arguments.is_present("verbose").into();

    let mut simulator = ElectronBeamSimulator::from_param_file(param_file_path);

    configure_simulator_from_arguments(&mut simulator, &arguments);

    let beams = simulator.generate_beams(
        generate_only,
        extra_fixed_scalars.as_ref(),
        extra_varying_scalars.as_ref(),
        verbose,
    );

    if let Some(output_path) = possible_output_path {
        if output_format == "pickle" {
            beams
                .save_as_combined_pickles(output_path)
                .unwrap_or_else(|err| panic!("Could not save output data: {}", err));
        } else if output_format == "json" {
            beams
                .save_as_json(output_path)
                .unwrap_or_else(|err| panic!("Could not save output data: {}", err));
        } else {
            panic!("Invalid output format {}.", output_format)
        }
    }
}

fn add_simulator_arguments_to_subcommand<'a, 'b>(app: App<'a, 'b>) -> App<'a, 'b> {
    app.arg(
        Arg::with_name("reconnection-factor-type")
            .long("reconnection-factor-type")
            .value_name("TYPE")
            .help("Which version of the reconnection factor to use for seeding [default: from param file]")
            .next_line_help(true)
            .takes_value(true)
            .possible_values(&["standard", "normalized"]),
    )
    .arg(
        Arg::with_name("reconnection-factor-threshold")
            .long("reconnection-factor-threshold")
            .value_name("VALUE")
            .help("Beams will be generated where the reconnection factor value is larger than this [default: from param file]")
            .next_line_help(true)
            .takes_value(true),
    )
    .arg(
        Arg::with_name("min-acceleration-depth")
            .long("min-acceleration-depth")
            .value_name("VALUE")
            .help("Smallest depth at which electrons will be accelerated [Mm] [default: from param file]")
            .next_line_help(true)
            .takes_value(true),
    )
    .arg(
        Arg::with_name("max-acceleration-depth")
            .long("max-acceleration-depth")
            .value_name("VALUE")
            .help("Largest depth at which electrons will be accelerated [Mm] [default: from param file]")
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
            .help("Distributions with total power densities smaller than this value are discarded [erg/(cm^3 s)] [default: from param file]")
            .next_line_help(true)
            .takes_value(true),
    )
    .arg(
        Arg::with_name("min-estimated-depletion-distance")
            .long("min-estimated-depletion-distance")
            .value_name("VALUE")
            .help("Distributions with an initial estimated depletion distance smaller than this value are discarded [cm] [default: from param file]")
            .next_line_help(true)
            .takes_value(true),
    )
    .arg(
        Arg::with_name("max-acceleration-angle")
            .long("max-acceleration-angle")
            .value_name("VALUE")
            .help("Distributions with acceleration directions angled more than this away from the magnetic field axis are discarded [deg]")
            .next_line_help(true)
            .takes_value(true)
            .default_value("70.0"),
    )
    .arg(
        Arg::with_name("acceleration-duration")
            .long("acceleration-duration")
            .value_name("VALUE")
            .help("Duration of the acceleration events [s] [default: from param file]")
            .next_line_help(true)
            .takes_value(true),
    )
    .arg(
        Arg::with_name("particle-energy-fraction")
            .long("particle-energy-fraction")
            .value_name("VALUE")
            .help("Fraction of the released reconnection energy going into acceleration of electrons [default: from param file]")
            .next_line_help(true)
            .takes_value(true),
    )
    .arg(
        Arg::with_name("power-law-delta")
            .long("power-law-delta")
            .value_name("VALUE")
            .help("Exponent of the inverse power-law describing the non-thermal electron distribution [default: from param file]")
            .next_line_help(true)
            .takes_value(true),
    )
    .arg(
        Arg::with_name("min-remaining-power-density")
            .long("min-remaining-power-density")
            .value_name("VALUE")
            .help("Distributions with remaining power densities smaller than this value are discarded [erg/(cm^3 s)] [default: from param file]")
            .next_line_help(true)
            .takes_value(true),
    )
    .arg(
        Arg::with_name("initial-cutoff-energy-guess")
            .long("initial-cutoff-energy-guess")
            .value_name("VALUE")
            .help("Initial guess to use when estimating lower cut-off energy [keV]")
            .next_line_help(true)
            .takes_value(true)
            .default_value("4.0"),
    )
    .arg(
        Arg::with_name("acceptable-root-finding-error")
            .long("acceptable-root-finding-error")
            .value_name("VALUE")
            .help("Target relative error when estimating lower cut-off energy")
            .next_line_help(true)
            .takes_value(true)
            .default_value("1e-3"),
    )
    .arg(
        Arg::with_name("max-root-finding-iterations")
            .long("max-root-finding-iterations")
            .value_name("NUMBER")
            .help("Maximum number of iterations when estimating lower cut-off energy")
            .next_line_help(true)
            .takes_value(true)
            .default_value("100"),
    )
    .arg(
        Arg::with_name("interpolation-order")
            .long("interpolation-order")
            .value_name("ORDER")
            .help("Order of the polynomials to fit when interpolating field values")
            .next_line_help(true)
            .takes_value(true)
            .possible_values(&["1", "2", "3", "4", "5"])
            .default_value("3"),
    )
    .arg(
        Arg::with_name("variation-threshold-for-linear-interpolation")
            .long("variation-threshold-for-linear-interpolation")
            .value_name("VALUE")
            .help("Linear interpolation is used when a normalized variance of the values surrounding the interpolation point exceeds this")
            .next_line_help(true)
            .takes_value(true)
            .default_value("0.3"),
    )
    .arg(
        Arg::with_name("stepping-scheme")
            .long("stepping-scheme")
            .value_name("NAME")
            .help("Which stepping scheme to use for tracing beam trajectories")
            .next_line_help(true)
            .takes_value(true)
            .default_value("rkf45"),
    )
    .arg(
        Arg::with_name("dense-step-length")
            .long("dense-step-length")
            .value_name("VALUE")
            .help("Step length to use for dense (uniform) output positions [Mm] [default: from param file]")
            .next_line_help(true)
            .takes_value(true),
    )
    .arg(
        Arg::with_name("max-step-attempts")
            .long("max-step-attempts")
            .value_name("NUMBER")
            .help("Maximum number of step attempts before terminating")
            .next_line_help(true)
            .takes_value(true)
            .default_value("16"),
    )
    .arg(
        Arg::with_name("stepping-absolute-tolerance")
            .long("stepping-absolute-tolerance")
            .value_name("VALUE")
            .help("Absolute error tolerance for stepping")
            .next_line_help(true)
            .takes_value(true)
            .default_value("1e-6"),
    )
    .arg(
        Arg::with_name("stepping-relative-tolerance")
            .long("stepping-relative-tolerance")
            .value_name("VALUE")
            .help("Relative error tolerance for stepping")
            .next_line_help(true)
            .takes_value(true)
            .default_value("1e-6"),
    )
    .arg(
        Arg::with_name("stepping-safety-factor")
            .long("stepping-safety-factor")
            .value_name("VALUE")
            .help("Scaling factor for the error to reduce step length oscillations")
            .next_line_help(true)
            .takes_value(true)
            .default_value("0.9"),
    )
    .arg(
        Arg::with_name("min-step-scale")
            .long("min-step-scale")
            .value_name("VALUE")
            .help("Smallest allowed scaling of the step size in one step")
            .next_line_help(true)
            .takes_value(true)
            .default_value("0.2"),
    )
    .arg(
        Arg::with_name("max-step-scale")
            .long("max-step-scale")
            .value_name("VALUE")
            .help("Largest allowed scaling of the step size in one step")
            .next_line_help(true)
            .takes_value(true)
            .default_value("10.0"),
    )
    .arg(
        Arg::with_name("stepping-initial-error")
            .long("stepping-initial-error")
            .value_name("VALUE")
            .help("Start value for stepping error")
            .next_line_help(true)
            .takes_value(true)
            .default_value("1e-4"),
    )
    .arg(
        Arg::with_name("initial-step-length")
            .long("stepping-initial-step-length")
            .value_name("VALUE")
            .help("Initial step size")
            .next_line_help(true)
            .takes_value(true)
            .default_value("1e-4"),
    )
    .arg(
        Arg::with_name("sudden-reversals-for-sink")
            .long("sudden-reversals-for-sink")
            .value_name("NUMBER")
            .help("Number of sudden direction reversals before the area is considered a sink")
            .next_line_help(true)
            .takes_value(true)
            .default_value("3"),
    ).arg(
        Arg::with_name("pi-control")
            .long("pi-control")
            .value_name("STATE")
            .help("Whether to use Proportional Integral (PI) control for stabilizing the stepping [default: from param file]")
            .next_line_help(true)
            .takes_value(true)
            .possible_values(&["off", "on"]),
    )
}

fn configure_simulator_from_arguments(
    simulator: &mut ElectronBeamSimulator,
    arguments: &ArgMatches,
) {
    cli::assign_value_from_selected_argument(
        &mut simulator.use_normalized_reconnection_factor,
        arguments,
        "reconnection-factor-type",
        &["standard", "normalized"],
        &[false, true],
    );
    cli::assign_value_from_parseable_argument(
        &mut simulator.reconnection_factor_threshold,
        arguments,
        "reconnection-factor-threshold",
    );
    cli::assign_value_from_parseable_argument(
        &mut simulator.min_acceleration_depth,
        arguments,
        "min-acceleration-depth",
    );
    cli::assign_value_from_parseable_argument(
        &mut simulator.max_acceleration_depth,
        arguments,
        "max-acceleration-depth",
    );
    cli::assign_bool_value_from_flag_presence(
        &mut simulator.accelerator_config.ignore_rejection,
        arguments,
        "ignore-rejection",
    );
    cli::assign_value_from_parseable_argument(
        &mut simulator.accelerator_config.min_total_power_density,
        arguments,
        "min-total-power-density",
    );
    cli::assign_value_from_parseable_argument(
        &mut simulator
            .accelerator_config
            .min_estimated_depletion_distance,
        arguments,
        "min-estimated-depletion-distance",
    );
    cli::assign_value_from_parseable_argument(
        &mut simulator.accelerator_config.max_acceleration_angle,
        arguments,
        "max-acceleration-angle",
    );
    cli::assign_value_from_parseable_argument(
        &mut simulator.acceleration_duration,
        arguments,
        "acceleration-duration",
    );
    cli::assign_value_from_parseable_argument(
        &mut simulator.particle_energy_fraction,
        arguments,
        "particle-energy-fraction",
    );
    cli::assign_value_from_parseable_argument(
        &mut simulator.power_law_delta,
        arguments,
        "power-law-delta",
    );
    cli::assign_value_from_parseable_argument(
        &mut simulator.distribution_config.min_remaining_power_density,
        arguments,
        "min-remaining-power-density",
    );
    cli::assign_value_from_parseable_argument(
        &mut simulator.accelerator_config.initial_cutoff_energy_guess,
        arguments,
        "initial-cutoff-energy-guess",
    );
    cli::assign_value_from_parseable_argument(
        &mut simulator.accelerator_config.acceptable_root_finding_error,
        arguments,
        "acceptable-root-finding-error",
    );
    cli::assign_value_from_parseable_argument(
        &mut simulator.accelerator_config.max_root_finding_iterations,
        arguments,
        "max-root-finding-iterations",
    );
    cli::assign_value_from_parseable_argument(
        &mut simulator.interpolator_config.order,
        arguments,
        "interpolation-order",
    );
    cli::assign_value_from_parseable_argument(
        &mut simulator.interpolator_config.variation_threshold_for_linear,
        arguments,
        "variation-threshold-for-linear-interpolation",
    );
    cli::assign_value_from_selected_argument(
        &mut simulator.rkf_stepper_type,
        arguments,
        "stepping-scheme",
        &["rkf23", "rkf45"],
        &[RKFStepperType::RKF23, RKFStepperType::RKF45],
    );
    cli::assign_value_from_parseable_argument(
        &mut simulator.rkf_stepper_config.dense_step_length,
        arguments,
        "dense-step-length",
    );
    cli::assign_value_from_parseable_argument(
        &mut simulator.rkf_stepper_config.max_step_attempts,
        arguments,
        "max-step-attempts",
    );
    cli::assign_value_from_parseable_argument(
        &mut simulator.rkf_stepper_config.absolute_tolerance,
        arguments,
        "stepping-absolute-tolerance",
    );
    cli::assign_value_from_parseable_argument(
        &mut simulator.rkf_stepper_config.relative_tolerance,
        arguments,
        "stepping-relative-tolerance",
    );
    cli::assign_value_from_parseable_argument(
        &mut simulator.rkf_stepper_config.safety_factor,
        arguments,
        "stepping-safety-factor",
    );
    cli::assign_value_from_parseable_argument(
        &mut simulator.rkf_stepper_config.min_step_scale,
        arguments,
        "min-step-scale",
    );
    cli::assign_value_from_parseable_argument(
        &mut simulator.rkf_stepper_config.max_step_scale,
        arguments,
        "max-step-scale",
    );
    cli::assign_value_from_parseable_argument(
        &mut simulator.rkf_stepper_config.initial_error,
        arguments,
        "stepping-initial-error",
    );
    cli::assign_value_from_parseable_argument(
        &mut simulator.rkf_stepper_config.initial_step_length,
        arguments,
        "initial-step-length",
    );
    cli::assign_value_from_parseable_argument(
        &mut simulator.rkf_stepper_config.sudden_reversals_for_sink,
        arguments,
        "sudden-reversals-for-sink",
    );
    cli::assign_value_from_selected_argument(
        &mut simulator.rkf_stepper_config.use_pi_control,
        arguments,
        "pi-control",
        &["off", "on"],
        &[false, true],
    );
}
