//! Command line interface for resampling a snapshot using weighted cell averaging.

use crate::{cli::snapshot::write::create_write_subcommand, create_subcommand};
use clap::{App, AppSettings, SubCommand};

/// Builds a representation of the `snapshot-resample-weighted_cell_averaging` command line subcommand.
pub fn create_weighted_cell_averaging_subcommand<'a, 'b>() -> App<'a, 'b> {
    SubCommand::with_name("weighted_cell_averaging")
        .about("Use the weighted cell averaging method")
        .long_about(
            "Use the weighted cell averaging method.\n\
             For each new grid cell, the values of all overlapped original grid cells are\n\
             averaged with weights according to the intersected volumes.\n\
             This method is suited for downsampling. It is faster than weighted sample\n\
             averaging, but slightly less accurate.",
        )
        .help_message("Print help information")
        .setting(AppSettings::SubcommandRequired)
        .subcommand(create_subcommand!(weighted_cell_averaging, write))
}
