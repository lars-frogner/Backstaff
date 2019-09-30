//! Command line interface for simulation of electron beams.

use super::simulation::ElectronBeamSimulator;
use clap::{App, Arg, ArgMatches, SubCommand};

/// Builds a representation of the `ebeam` command line subcommand.
pub fn build_subcommand<'a, 'b>() -> App<'a, 'b> {
    let app = SubCommand::with_name("ebeam")
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

    add_simulator_parameter_args(app)
}

/// Runs the actions for the `ebeam` subcommand using the given arguments.
pub fn run(arguments: &ArgMatches) {
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

    configure_simulator_parameters(&mut simulator, &arguments);

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

fn add_simulator_parameter_args<'a, 'b>(app: App<'a, 'b>) -> App<'a, 'b> {
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

fn configure_simulator_parameters(simulator: &mut ElectronBeamSimulator, arguments: &ArgMatches) {
    simulator.use_normalized_reconnection_factor =
        arguments.is_present("use-normalized-reconnection-factor");

    if let Some(reconnection_factor_threshold) = arguments.value_of("reconnection-factor-threshold")
    {
        simulator.reconnection_factor_threshold =
            reconnection_factor_threshold.parse().unwrap_or_else(|err| {
                panic!(
                    "Could not parse value of reconnection-factor-threshold: {}",
                    err
                )
            });
    }

    if let Some(min_acceleration_depth) = arguments.value_of("min-acceleration-depth") {
        simulator.min_acceleration_depth = min_acceleration_depth.parse().unwrap_or_else(|err| {
            panic!("Could not parse value of min-acceleration-depth: {}", err)
        });
    }

    if let Some(max_acceleration_depth) = arguments.value_of("max-acceleration-depth") {
        simulator.max_acceleration_depth = max_acceleration_depth.parse().unwrap_or_else(|err| {
            panic!("Could not parse value of max-acceleration-depth: {}", err)
        });
    }

    simulator.accelerator_config.ignore_rejection = arguments.is_present("ignore-rejection");

    if let Some(min_total_power_density) = arguments.value_of("min-total-power-density") {
        simulator.accelerator_config.min_total_power_density =
            min_total_power_density.parse().unwrap_or_else(|err| {
                panic!("Could not parse value of min-total-power-density: {}", err)
            });
    }

    if let Some(min_estimated_depletion_distance) =
        arguments.value_of("min-estimated-depletion-distance")
    {
        simulator
            .accelerator_config
            .min_estimated_depletion_distance = min_estimated_depletion_distance
            .parse()
            .unwrap_or_else(|err| {
                panic!(
                    "Could not parse value of min-estimated-depletion-distance: {}",
                    err
                )
            });
    }

    if let Some(max_acceleration_angle) = arguments.value_of("max-acceleration-angle") {
        simulator.accelerator_config.max_acceleration_angle =
            max_acceleration_angle.parse().unwrap_or_else(|err| {
                panic!("Could not parse value of max-acceleration-angle: {}", err)
            });
    }

    if let Some(acceleration_duration) = arguments.value_of("acceleration-duration") {
        simulator.acceleration_duration = acceleration_duration.parse().unwrap_or_else(|err| {
            panic!("Could not parse value of acceleration-duration: {}", err)
        });
    }

    if let Some(particle_energy_fraction) = arguments.value_of("particle-energy-fraction") {
        simulator.particle_energy_fraction =
            particle_energy_fraction.parse().unwrap_or_else(|err| {
                panic!("Could not parse value of particle-energy-fraction: {}", err)
            });
    }

    if let Some(power_law_delta) = arguments.value_of("power-law-delta") {
        simulator.power_law_delta = power_law_delta
            .parse()
            .unwrap_or_else(|err| panic!("Could not parse value of power-law-delta: {}", err));
    }

    if let Some(min_remaining_power_density) = arguments.value_of("min-remaining-power-density") {
        simulator.distribution_config.min_remaining_power_density =
            min_remaining_power_density.parse().unwrap_or_else(|err| {
                panic!(
                    "Could not parse value of min-remaining-power-density: {}",
                    err
                )
            });
    }
}
