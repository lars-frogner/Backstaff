//! Command line interface for printing statistics for quantities in a snapshot.

use crate::grid::Grid3;
use crate::io::snapshot::{fdt, SnapshotReader3};
use clap::{App, Arg, ArgMatches, SubCommand};

/// Builds a representation of the `snapshot-inspect-statistics` command line subcommand.
pub fn build_subcommand_statistics<'a, 'b>() -> App<'a, 'b> {
    SubCommand::with_name("statistics")
        .about("Prints statistics from quantities in a snapshot")
        .arg(
            Arg::with_name("QUANTITIES")
                .help("List of quantities to print statistics for")
                .required(true)
                .takes_value(true)
                .index(1)
                .multiple(true)
                .min_values(1),
        )
}

/// Runs the actions for the `snapshot-inspect-statistics` subcommand using the given arguments.
pub fn run_subcommand_statistics<G: Grid3<fdt>>(
    arguments: &ArgMatches,
    reader: SnapshotReader3<G>,
) {
    for quantity in arguments
        .values_of("QUANTITIES")
        .expect("No values for required argument")
    {
        println!("{}", quantity)
    }
}
