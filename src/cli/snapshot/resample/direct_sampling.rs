//! Command line interface for resampling a snapshot using direct sampling.

use crate::{
    add_subcommand_combinations,
    cli::{
        interpolation::poly_fit::create_poly_fit_interpolator_subcommand,
        snapshot::{
            derive::create_derive_subcommand, inspect::create_inspect_subcommand,
            write::create_write_subcommand,
        },
    },
    update_command_graph,
};
use clap::Command;

#[cfg(feature = "synthesis")]
use crate::cli::snapshot::synthesize::create_synthesize_subcommand;

/// Builds a representation of the `snapshot-resample-direct_sampling` command line subcommand.
pub fn create_direct_sampling_subcommand(_parent_command_name: &'static str) -> Command<'static> {
    let command_name = "direct_sampling";

    update_command_graph!(_parent_command_name, command_name);

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

    #[cfg(feature = "synthesis")]
    let command = add_subcommand_combinations!(command, command_name, true; poly_fit_interpolator, derive, synthesize, (write, inspect));
    #[cfg(not(feature = "synthesis"))]
    let command = add_subcommand_combinations!(command, command_name, true; poly_fit_interpolator, derive, (write, inspect));

    command
}
