//! Command line interface for printing derivable quantities.

use crate::field::quantities;
use clap::{ArgMatches, Command};

/// Creates a subcommand for printing derivable quantities.
pub fn create_derivable_quantities_subcommand(
    parent_command_name: &'static str,
) -> Command<'static> {
    let command_name = "derivable_quantities";

    crate::cli::command_graph::insert_command_graph_edge(parent_command_name, command_name);

    Command::new(command_name)
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
