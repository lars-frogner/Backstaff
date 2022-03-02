//! Command line interface for printing derivable quantities.

use crate::field::quantities;
use clap::{ArgMatches, Command};

/// Creates a subcommand for printing derivable quantities.
pub fn create_derivable_quantities_subcommand() -> Command<'static> {
    Command::new("derivable_quantities")
        .about("Print an overview of available derivable quantities")
        .long_about(
            "Print an overview of available derivable quantities.\n\
             Such quantities can be included in output snapshots given that the\n\
             quantities they depend on are present in the original snapshot.",
        )
}

pub fn run_derivable_quantities_subcommand(_arguments: &ArgMatches) {
    quantities::print_available_quantities();
}
