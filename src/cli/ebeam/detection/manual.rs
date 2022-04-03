//! Command line interface for reconnection site detector reading positions from an input file.

use super::super::distribution::power_law::create_power_law_distribution_subcommand;
use crate::{ebeam::detection::manual::ManualReconnectionSiteDetector, exit_on_error};
use clap::{Arg, ArgMatches, Command};
use std::{path::PathBuf, str::FromStr};

/// Creates a subcommand for using the manual reconnection site detector.
pub fn create_manual_reconnection_site_detector_subcommand(
    parent_command_name: &'static str,
) -> Command<'static> {
    let command_name = "manual_detector";

    crate::cli::command_graph::insert_command_graph_edge(parent_command_name, command_name);

    Command::new(command_name)
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
                .takes_value(true),
        )
        .subcommand(create_power_law_distribution_subcommand(command_name))
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
        ManualReconnectionSiteDetector::new(input_file_path),
        "Error: Could not create manual reconnection site detector: {}"
    )
}
