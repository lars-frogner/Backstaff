//! Command line interface for resampling a snapshot using weighted sample averaging.

use crate::cli;
use clap::{App, SubCommand};

/// Builds a representation of the `snapshot-resample-weighted_sample_averaging` command line subcommand.
pub fn create_weighted_sample_averaging_subcommand<'a, 'b>() -> App<'a, 'b> {
    SubCommand::with_name("weighted_sample_averaging")
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
        )
        .subcommand(cli::interpolation::poly_fit::create_poly_fit_interpolator_subcommand())
}
