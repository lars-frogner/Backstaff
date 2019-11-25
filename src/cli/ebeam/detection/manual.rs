//! Command line interface for reconnection site detector reading positions from an input file.

use crate::ebeam::detection::manual::ManualReconnectionSiteDetector;
use clap::{App, Arg, ArgMatches, SubCommand};
use std::path::PathBuf;
use std::str::FromStr;

/// Creates a subcommand for using the manual reconnection site detector.
pub fn create_manual_reconnection_site_detector_subcommand<'a, 'b>() -> App<'a, 'b> {
    SubCommand::with_name("manual_detector")
        .about("Read reconnection site positions from input file")
        .long_about(
            "Read reconnection site positions from input file.\n\
             The input file is assumed to be in CSV format, with each line consisting\n\
             of the three comma-separated coordinates of a single position.",
        )
        .help_message("Print help information")
        .arg(
            Arg::with_name("input-path")
                .short("i")
                .long("input-path")
                .require_equals(true)
                .value_name("PATH")
                .help("Path to the text file containing the reconnection site positions")
                .required(true)
                .takes_value(true),
        )
}

/// Creates a manual reconnection site detector from the provided options.
pub fn construct_manual_reconnection_site_detector_from_options(
    arguments: &ArgMatches,
) -> ManualReconnectionSiteDetector {
    let input_file_path = PathBuf::from_str(
        arguments
            .value_of("input-path")
            .expect("No value for required argument."),
    )
    .unwrap_or_else(|err| panic!("Could not interpret input-path: {}", err));
    ManualReconnectionSiteDetector::new(input_file_path).unwrap_or_else(|err| {
        panic!(
            "Could not create manual reconnection site detector: {}",
            err
        )
    })
}
