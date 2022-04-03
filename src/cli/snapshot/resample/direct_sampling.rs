//! Command line interface for resampling a snapshot using direct sampling.

use crate::{
    add_subcommand_combinations,
    cli::{
        interpolation::poly_fit::create_poly_fit_interpolator_subcommand,
        snapshot::write::create_write_subcommand,
    },
};
use clap::Command;

/// Builds a representation of the `snapshot-resample-direct_sampling` command line subcommand.
pub fn create_direct_sampling_subcommand(parent_command_name: &'static str) -> Command<'static> {
    let command_name = "direct_sampling";

    crate::cli::command_graph::insert_command_graph_edge(parent_command_name, command_name);

    let command = Command::new(command_name)
        .about("Use the direct sampling method")
        .long_about(
            "Use the direct sampling method.\n\
             Each value on the new grid is found by interpolation of the values on the old\n\
             grid at the new coordinate location.\n\
             This is the preferred method for upsampling. For heavy downsampling it yields a\n\
             more noisy result than weighted averaging.",
        )
        .after_help(
            "You can use a subcommand to configure the interpolator. If left unspecified,\n\
             the default interpolator implementation and parameters are used.",
        );

    add_subcommand_combinations!(command, command_name, true; poly_fit_interpolator, write)
}
