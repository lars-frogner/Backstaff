//! Command line interface for resampling a snapshot using direct sampling.

use crate::{
    cli::{
        interpolation::poly_fit::create_poly_fit_interpolator_subcommand,
        snapshot::write::create_write_subcommand,
    },
    create_subcommand,
};
use clap::Command;

/// Builds a representation of the `snapshot-resample-direct_sampling` command line subcommand.
pub fn create_direct_sampling_subcommand() -> Command<'static> {
    Command::new("direct_sampling")
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
        )
        .subcommand_required(true)
        .subcommand(
            create_subcommand!(direct_sampling, poly_fit_interpolator)
                .subcommand_required(true)
                .subcommand(create_subcommand!(poly_fit_interpolator, write)),
        )
        .subcommand(create_subcommand!(direct_sampling, write))
}
