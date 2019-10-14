//! Command line interface for actions related to electron beams.

pub mod accelerator;
pub mod detection;
pub mod distribution;
pub mod simulate;

use crate::grid::Grid3;
use crate::io::snapshot::{fdt, SnapshotCacher3};
use clap::{App, AppSettings, ArgMatches, SubCommand};

/// Builds a representation of the `ebeam` command line subcommand.
pub fn create_ebeam_subcommand<'a, 'b>() -> App<'a, 'b> {
    SubCommand::with_name("ebeam")
        .about("Perform actions related to electron beams in the snapshot")
        .help_message("Print help information")
        .setting(AppSettings::SubcommandRequiredElseHelp)
        .subcommand(simulate::create_simulate_subcommand())
}

/// Runs the actions for the `ebeam` subcommand using the given arguments.
pub fn run_ebeam_subcommand<G: Grid3<fdt>>(
    arguments: &ArgMatches,
    snapshot: &mut SnapshotCacher3<G>,
) {
    if let Some(simulate_arguments) = arguments.subcommand_matches("simulate") {
        simulate::run_simulate_subcommand(simulate_arguments, snapshot);
    }
}
