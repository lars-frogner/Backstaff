//! Command line interface for reconnection site detector reading positions from an input file.

use crate::{
    add_subcommand_combinations,
    cli::{
        ebeam::{
            accelerator::simple_power_law::create_simple_power_law_accelerator_subcommand,
            distribution::power_law::create_power_law_distribution_subcommand,
            propagator::{
                analytical::create_analytical_propagator_subcommand,
                fp_characteristics::create_characteristics_propagator_subcommand,
            },
        },
        interpolation::poly_fit::create_poly_fit_interpolator_subcommand,
        tracing::stepping::rkf::create_rkf_stepper_subcommand,
    },
    ebeam::detection::manual::ManualReconnectionSiteDetector,
    exit_on_error, update_command_graph,
};
use clap::{Arg, ArgMatches, Command, ValueHint};
use std::{path::PathBuf, str::FromStr};

/// Creates a subcommand for using the manual reconnection site detector.
pub fn create_manual_reconnection_site_detector_subcommand(
    _parent_command_name: &'static str,
) -> Command<'static> {
    let command_name = "manual_detector";

    update_command_graph!(_parent_command_name, command_name);

    let command = Command::new(command_name)
        .about("Read reconnection site positions from input file")
        .long_about(
            "Read reconnection site positions from input file.\n\
             The input file is assumed to be in CSV format, with each line consisting\n\
             of the three comma-separated coordinates of a single position.",
        )
        .arg(
            Arg::new("input-file")
                .short('i')
                .long("input-file")
                .require_equals(true)
                .value_name("FILE")
                .help("Path to the text file containing the reconnection site positions")
                .required(true)
                .takes_value(true)
                .value_hint(ValueHint::FilePath),
        )
        .subcommand(create_power_law_distribution_subcommand(command_name))
        .subcommand(create_simple_power_law_accelerator_subcommand(command_name))
        .subcommand(create_analytical_propagator_subcommand(command_name))
        .subcommand(create_characteristics_propagator_subcommand(command_name));

    add_subcommand_combinations!(command, command_name, false; poly_fit_interpolator, rkf_stepper)
}

/// Creates a manual reconnection site detector from the provided options.
pub fn construct_manual_reconnection_site_detector_from_options(
    arguments: &ArgMatches,
) -> ManualReconnectionSiteDetector {
    let input_file_path = exit_on_error!(
        PathBuf::from_str(
            arguments
                .value_of("input-file")
                .expect("No value for required argument"),
        ),
        "Error: Could not interpret path to input file: {}"
    );
    exit_on_error!(
        ManualReconnectionSiteDetector::new(&input_file_path),
        "Error: Could not create manual reconnection site detector: {}"
    )
}
