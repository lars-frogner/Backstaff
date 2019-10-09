//! Command line interface for downsampling a snapshot.

use clap::{App, SubCommand};

/// Builds a representation of the `snapshot-resample-downsampling` command line subcommand.
pub fn create_downsampling_subcommand<'a, 'b>() -> App<'a, 'b> {
    SubCommand::with_name("downsampling")
        .about("Use a dedicated downsampling method")
        .long_about(
            "Use a dedicated downsampling method.\n\
             For each new grid cell, the values in all overlapped original grid cells are\n\
             averaged with weights according to the intersected volumes.",
        )
}
