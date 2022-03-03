//! Command line interface for reading seed points from an input file.

use crate::{exit_on_error, seeding::manual::ManualSeeder3};
use clap::{Arg, ArgMatches, Command, ValueHint};
use std::{path::PathBuf, str::FromStr};

/// Creates a subcommand for using a manual seeder.
pub fn create_manual_seeder_subcommand() -> Command<'static> {
    Command::new("manual_seeder")
        .about("Read seed point from input file")
        .long_about(
            "Read seed point from input file.\n\
             The input file is assumed to be in CSV format, with each line consisting\n\
             of the three comma-separated coordinates of a single seed point.",
        )
        .arg(
            Arg::new("input-file")
                .short('i')
                .long("input-file")
                .require_equals(true)
                .value_name("FILE")
                .help("Path to the text file containing the seed points")
                .required(true)
                .takes_value(true)
                .value_hint(ValueHint::FilePath),
        )
}

/// Creates a manual seeder based on the provided arguments.
pub fn create_manual_seeder_from_arguments(arguments: &ArgMatches) -> ManualSeeder3 {
    let input_file_path = exit_on_error!(
        PathBuf::from_str(
            arguments
                .value_of("input-file")
                .expect("No value for required argument"),
        ),
        "Error: Could not interpret path to input file: {}"
    );

    exit_on_error!(
        ManualSeeder3::new(input_file_path),
        "Error: Could not create manual seeder: {}"
    )
}
