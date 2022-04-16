//! Command line interface for actions related to electron beams.

pub mod accelerator;
pub mod detection;
pub mod distribution;
pub mod simulate;

use self::simulate::{create_simulate_subcommand, run_simulate_subcommand};
use crate::{
    cli::snapshot::SnapNumInRange,
    grid::Grid3,
    io::snapshot::{fdt, SnapshotProvider3},
    update_command_graph,
};
use clap::{ArgMatches, Command};

/// Builds a representation of the `ebeam` command line subcommand.
pub fn create_ebeam_subcommand(_parent_command_name: &'static str) -> Command<'static> {
    let command_name = "ebeam";

    update_command_graph!(_parent_command_name, command_name);

    Command::new(command_name)
        .about("Perform actions related to electron beams in the snapshot")
        .subcommand_required(true)
        .subcommand(create_simulate_subcommand(command_name))
}

/// Runs the actions for the `ebeam` subcommand using the given arguments.
pub fn run_ebeam_subcommand<G, P>(
    arguments: &ArgMatches,
    provider: P,
    snap_num_in_range: &Option<SnapNumInRange>,
    protected_file_types: &[&str],
) where
    G: Grid3<fdt>,
    P: SnapshotProvider3<G>,
{
    if let Some(simulate_arguments) = arguments.subcommand_matches("simulate") {
        run_simulate_subcommand(
            simulate_arguments,
            provider,
            snap_num_in_range,
            protected_file_types,
        );
    }
}
