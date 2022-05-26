//! Command line interface for resampling a snapshot using weighted sample averaging.

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

/// Builds a representation of the `snapshot-resample-sample_averaging` command line subcommand.
pub fn create_sample_averaging_subcommand(_parent_command_name: &'static str) -> Command<'static> {
    let command_name = "sample_averaging";

    update_command_graph!(_parent_command_name, command_name);

    let command = Command::new(command_name)
        .about("Use the weighted sample averaging method")
        .long_about(
            "Use the weighted sample averaging method.\n\
             For each new grid cell, values are interpolated from all overlapped original\n\
             grid cells and averaged with weights according to the intersected volumes.\n\
             If the new grid cell is contained within an original grid cell, this reduces\n\
             to a single interpolation.\n\
             This method gives robust results for arbitrary resampling grids, but is slower\n\
             than direct sampling or weighted cell averaging.",
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
