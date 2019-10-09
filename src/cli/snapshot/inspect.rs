//! Command line interface for actions related to inspection of snapshots.

mod statistics;

use crate::grid::Grid3;
use crate::io::snapshot::{fdt, SnapshotCacher3};
use clap::{App, AppSettings, ArgMatches, SubCommand};

/// Builds a representation of the `snapshot-inspect` command line subcommand.
pub fn create_inspect_subcommand<'a, 'b>() -> App<'a, 'b> {
    SubCommand::with_name("inspect")
        .about("Inspect properties of the snapshot")
        .setting(AppSettings::SubcommandRequiredElseHelp)
        .subcommand(statistics::create_statistics_subcommand())
}

/// Runs the actions for the `snapshot-inspect` subcommand using the given arguments.
pub fn run_inspect_subcommand<G: Grid3<fdt>>(
    arguments: &ArgMatches,
    cacher: &mut SnapshotCacher3<G>,
) {
    if let Some(statistics_arguments) = arguments.subcommand_matches("statistics") {
        statistics::run_statistics_subcommand(statistics_arguments, cacher);
    }
}
