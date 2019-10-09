//! Command line interface for general resampling of a snapshot.

use crate::cli;
use clap::{App, SubCommand};

/// Builds a representation of the `snapshot-resample-general` command line subcommand.
pub fn create_general_subcommand<'a, 'b>() -> App<'a, 'b> {
    SubCommand::with_name("general")
        .about("Use a general resampling method")
        .long_about(
            "Use a general resampling method.\n\
             This method gives robust results for arbitrary resampling grids, but is slower\n\
             than the dedicated down- and upsamling methods.\n\
             For each new grid cell, values are interpolated from all overlapped original\n\
             grid cells and averaged with weights according to the intersected volumes. If\n\
             the new grid cell is contained within an original grid cell, this reduces to a\n\
             single interpolation.",
        )
        .after_help(
            "You can use a subcommand to configure the interpolator. If left unspecified,\n\
             the default interpolator implementation and parameters are used.",
        )
        .subcommand(cli::interpolation::poly_fit::create_poly_fit_interpolator_subcommand())
}
