//! Command line interface for resampling a snapshot using weighted cell averaging.

use clap::{App, SubCommand};

/// Builds a representation of the `snapshot-resample-weighted_cell_averaging` command line subcommand.
pub fn create_weighted_cell_averaging_subcommand<'a, 'b>() -> App<'a, 'b> {
    SubCommand::with_name("weighted_cell_averaging")
        .about("Use the weighted cell averaging method")
        .long_about(
            "Use the weighted cell averaging method.\n\
             For each new grid cell, the values of all overlapped original grid cells are\n\
             averaged with weights according to the intersected volumes.\n\
             This method is faster than weighted sample averaging, but does not preserve\n\
             contrast as well.",
        )
}
