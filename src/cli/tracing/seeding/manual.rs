//! Command line interface for reading seed points from an input file.

use crate::tracing::seeding::manual::ManualSeeder3;
use clap::{App, Arg, ArgMatches, SubCommand};
use std::path::PathBuf;
use std::str::FromStr;

/// Creates a subcommand for using a manual seeder.
pub fn create_manual_seeder_subcommand<'a, 'b>() -> App<'a, 'b> {
    SubCommand::with_name("manual_seeder")
        .about("Read seed point from input file")
        .long_about(
            "Read seed point from input file.\n\
             The input file is assumed to be in CSV format, with each line consisting\n\
             of the three comma-separated coordinates of a single seed point.",
        )
        .help_message("Print help information")
        .arg(
            Arg::with_name("input-path")
                .short("i")
                .long("input-path")
                .require_equals(true)
                .value_name("PATH")
                .help("Path to the text file containing the seed points")
                .required(true)
                .takes_value(true),
        )
}

/// Creates a manual seeder based on the provided arguments.
pub fn create_manual_seeder_from_arguments(arguments: &ArgMatches) -> ManualSeeder3 {
    let input_file_path = PathBuf::from_str(
        arguments
            .value_of("input-path")
            .expect("No value for required argument."),
    )
    .unwrap_or_else(|err| panic!("Could not interpret input-path: {}", err));

    ManualSeeder3::new(input_file_path)
        .unwrap_or_else(|err| panic!("Could not create manual seeder: {}", err))
}
