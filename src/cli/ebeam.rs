//! Command line interface for actions related to electron beams.

pub mod accelerator;
pub mod detection;
pub mod distribution;
pub mod simulate;

use self::simulate::{create_simulate_subcommand, run_simulate_subcommand};
use crate::{
    cli::snapshot::SnapNumInRange,
    create_subcommand,
    grid::Grid3,
    io::snapshot::{fdt, SnapshotCacher3, SnapshotReader3},
};
use clap::{App, AppSettings, ArgMatches, SubCommand};

/// Builds a representation of the `ebeam` command line subcommand.
pub fn create_ebeam_subcommand<'a, 'b>() -> App<'a, 'b> {
    SubCommand::with_name("ebeam")
        .about("Perform actions related to electron beams in the snapshot")
        .help_message("Print help information")
        .setting(AppSettings::SubcommandRequired)
        .subcommand(create_subcommand!(ebeam, simulate))
}

/// Runs the actions for the `ebeam` subcommand using the given arguments.
pub fn run_ebeam_subcommand<G, R>(
    arguments: &ArgMatches,
    snapshot: &mut SnapshotCacher3<G, R>,
    snap_num_in_range: &Option<SnapNumInRange>,
    protected_file_types: &[&str],
) where
    G: Grid3<fdt>,
    R: SnapshotReader3<G> + Sync,
{
    if let Some(simulate_arguments) = arguments.subcommand_matches("simulate") {
        run_simulate_subcommand(
            simulate_arguments,
            snapshot,
            snap_num_in_range,
            protected_file_types,
        );
    }
}
