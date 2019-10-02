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
            Arg::with_name("OUTPUT_PATH")
                .help("Path where the beam data should be saved")
                .required(true)
                .takes_value(true)
                .index(2),
        )
        .arg(
            Arg::with_name("output-format")
                .short("f")
                .long("output-format")
                .value_name("FORMAT")
                .long_help("Format to use for saving beam data")
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
                .long_help("List of scalar fields to extract at acceleration sites")
                .takes_value(true)
                .multiple(true),
        )
        .arg(
            Arg::with_name("extra-varying-scalars")
                .long("extra-varying-scalars")
                .value_name("NAMES")
                .long_help("List of scalar fields to extract along beam trajectories")
                .takes_value(true)
                .multiple(true),
        )
        .arg(
            Arg::with_name("verbose")
                .short("v")
                .long("verbose")
                .help("Print status messages"),
        );

    let app = add_electron_beam_simulator_options_to_subcommand(app);
    let app = super::accelerator::simple_power_law::add_simple_power_law_accelerator_options_to_subcommand(app);
    let app = super::distribution::power_law::add_power_law_distribution_options_to_subcommand(app);
    let app = cli::interpolation::poly_fit::add_poly_fit_interpolator_options_to_subcommand(app);
    cli::tracing::stepping::rkf::add_rkf_stepper_options_to_subcommand(app)
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

    configure_electron_beam_simulator_from_options(&mut simulator, &arguments);

    super::accelerator::simple_power_law::configure_simple_power_law_accelerator_from_options(
        &mut simulator.accelerator_config,
        &arguments,
    );
    super::distribution::power_law::configure_power_law_distribution_from_options(
        &mut simulator.distribution_config,
        &arguments,
    );
    cli::interpolation::poly_fit::configure_poly_fit_interpolator_from_options(
        &mut simulator.interpolator_config,
        &arguments,
    );
    cli::tracing::stepping::rkf::configure_rkf_stepper_from_options(
        &mut simulator.stepper_config,
        &arguments,
    );

    let beams = simulator.generate_beams(
        generate_only,
        extra_fixed_scalars.as_ref(),
        extra_varying_scalars.as_ref(),
        verbose,
    );

    if let Some(output_path) = possible_output_path {
        match output_format {
            "pickle" => {
                beams
                    .save_as_combined_pickles(output_path)
                    .unwrap_or_else(|err| panic!("Could not save output data: {}", err));
            }
            "json" => {
                beams
                    .save_as_json(output_path)
                    .unwrap_or_else(|err| panic!("Could not save output data: {}", err));
            }
            invalid => panic!("Invalid output format {}.", invalid),
        }
    }
}

/// Adds arguments for parameters used by the electron beam simulator.
fn add_electron_beam_simulator_options_to_subcommand<'a, 'b>(app: App<'a, 'b>) -> App<'a, 'b> {
    app.arg(
        Arg::with_name("reconnection-factor-type")
            .long("reconnection-factor-type")
            .value_name("TYPE")
            .long_help(
                "Which version of the reconnection factor to use for seeding\n\
                 [default: from param file]",
            )
            .takes_value(true)
            .possible_values(&["standard", "normalized"]),
    )
    .arg(
        Arg::with_name("reconnection-factor-threshold")
            .long("reconnection-factor-threshold")
            .value_name("VALUE")
            .long_help(
                "Beams will be generated where the reconnection factor value is larger than this\n\
                 [default: from param file]",
            )
            .takes_value(true),
    )
    .arg(
        Arg::with_name("min-acceleration-depth")
            .long("min-acceleration-depth")
            .value_name("VALUE")
            .long_help(
                "Smallest depth at which electrons will be accelerated [Mm]\n\
                 [default: from param file]",
            )
            .takes_value(true),
    )
    .arg(
        Arg::with_name("max-acceleration-depth")
            .long("max-acceleration-depth")
            .value_name("VALUE")
            .long_help(
                "Largest depth at which electrons will be accelerated [Mm]\n\
                 [default: from param file]",
            )
            .takes_value(true),
    )
    .arg(
        Arg::with_name("acceleration-duration")
            .long("acceleration-duration")
            .value_name("VALUE")
            .long_help("Duration of the acceleration events [s] [default: from param file]")
            .takes_value(true),
    )
    .arg(
        Arg::with_name("particle-energy-fraction")
            .long("particle-energy-fraction")
            .value_name("VALUE")
            .long_help(
                "Fraction of the released reconnection energy going into acceleration of\n\
                 electrons [default: from param file]",
            )
            .takes_value(true),
    )
    .arg(
        Arg::with_name("power-law-delta")
            .long("power-law-delta")
            .value_name("VALUE")
            .long_help(
                "Exponent of the inverse power-law describing the non-thermal electron\n\
                 distribution [default: from param file]",
            )
            .takes_value(true),
    )
    .arg(
        Arg::with_name("stepping-scheme")
            .long("stepping-scheme")
            .value_name("NAME")
            .long_help("Which stepping scheme to use for tracing beam trajectories")
            .takes_value(true)
            .default_value("rkf45"),
    )
}

/// Sets electron beam simulator parameters based on present arguments.
fn configure_electron_beam_simulator_from_options(
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
    cli::assign_value_from_selected_argument(
        &mut simulator.stepper_type,
        arguments,
        "stepping-scheme",
        &["rkf23", "rkf45"],
        &[RKFStepperType::RKF23, RKFStepperType::RKF45],
    );
}
