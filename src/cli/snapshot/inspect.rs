//! Command line interface for actions related to inspection of snapshots.

mod statistics;

use self::statistics::{create_statistics_subcommand, run_statistics_subcommand};
use crate::{
    create_subcommand,
    grid::Grid3,
    io::snapshot::{fdt, SnapshotCacher3, SnapshotReader3},
};
use clap::{App, AppSettings, ArgMatches, SubCommand};

/// Builds a representation of the `snapshot-inspect` command line subcommand.
pub fn create_inspect_subcommand<'a, 'b>() -> App<'a, 'b> {
    SubCommand::with_name("inspect")
        .about("Inspect properties of the snapshot")
        .help_message("Print help information")
        .setting(AppSettings::SubcommandRequired)
        .subcommand(create_subcommand!(inspect, statistics))
}

/// Runs the actions for the `snapshot-inspect` subcommand using the given arguments.
pub fn run_inspect_subcommand<G, R>(arguments: &ArgMatches, snapshot: &mut SnapshotCacher3<G, R>)
where
    G: Grid3<fdt>,
    R: SnapshotReader3<G>,
{
    if let Some(statistics_arguments) = arguments.subcommand_matches("statistics") {
        run_statistics_subcommand(statistics_arguments, snapshot);
    }
}
