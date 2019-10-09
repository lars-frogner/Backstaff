//! Command line interface for upsampling a snapshot.

use crate::cli;
use clap::{App, SubCommand};

/// Builds a representation of the `snapshot-resample-upsampling` command line subcommand.
pub fn create_upsampling_subcommand<'a, 'b>() -> App<'a, 'b> {
    SubCommand::with_name("upsampling")
        .about("Use a dedicated upsampling method")
        .long_about(
            "Use a dedicated upsampling method.\n\
             Each value on the new grid is found by interpolation of the values on the old\n\
             grid at the new coordinate location. While the given grid does not have to be\n\
             finer than the original grid, this form of resampling can lead to artifacts if\n\
             the values are resampled onto a coarser grid.",
        )
        .after_help(
            "You can use a subcommand to configure the interpolator. If left unspecified,\n\
             the default interpolator implementation and parameters are used.",
        )
        .subcommand(cli::interpolation::poly_fit::create_poly_fit_interpolator_subcommand())
}
