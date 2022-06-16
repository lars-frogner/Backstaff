//! Command line interface for resampling a snapshot using weighted cell averaging.

use crate::{
    add_subcommand_combinations,
    cli::snapshot::{inspect::create_inspect_subcommand, write::create_write_subcommand},
    update_command_graph,
};
use clap::Command;

#[cfg(feature = "derivation")]
use crate::cli::snapshot::derive::create_derive_subcommand;

#[cfg(feature = "synthesis")]
use crate::cli::snapshot::synthesize::create_synthesize_subcommand;

/// Builds a representation of the `snapshot-resample-cell_averaging` command line subcommand.
pub fn create_cell_averaging_subcommand(_parent_command_name: &'static str) -> Command<'static> {
    let command_name = "cell_averaging";

    update_command_graph!(_parent_command_name, command_name);

    let command = Command::new(command_name)
        .about("Use the weighted cell averaging method")
        .long_about(
            "Use the weighted cell averaging method.\n\
             For each new grid cell, the values of all overlapped original grid cells are\n\
             averaged with weights according to the intersected volumes.\n\
             This method is suited for downsampling. It is faster than weighted sample\n\
             averaging, but slightly less accurate.",
        );
    add_subcommand_combinations!(
        command, command_name, true;
        derive if "derivation",
        synthesize if "synthesis",
        (write, inspect)
    )
}
