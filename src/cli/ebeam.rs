//! Command line interface for actions related to electron beams.

pub mod accelerator;
pub mod detection;
pub mod distribution;
pub mod propagator;
pub mod simulate;

use self::simulate::{create_simulate_subcommand, run_simulate_subcommand};
use crate::{
    field::DynScalarFieldProvider3,
    io::{
        snapshot::{fdt, SnapshotMetadata},
        utils::IOContext,
    },
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
pub fn run_ebeam_subcommand(
    arguments: &ArgMatches,
    metadata: &dyn SnapshotMetadata,
    provider: DynScalarFieldProvider3<fdt>,
    io_context: &mut IOContext,
) {
    if let Some(simulate_arguments) = arguments.subcommand_matches("simulate") {
        run_simulate_subcommand(simulate_arguments, metadata, provider, io_context);
    }
}
