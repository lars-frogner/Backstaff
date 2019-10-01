//! Command line interface for actions related to electron beams.

pub mod simulate;

use clap::{App, ArgMatches, SubCommand};

/// Builds a representation of the `ebeam` command line subcommand.
pub fn build_subcommand_ebeam<'a, 'b>() -> App<'a, 'b> {
    SubCommand::with_name("ebeam")
        .about("Performs actions related to electron beams")
        .subcommand(simulate::build_subcommand_simulate())
}

/// Runs the actions for the `ebeam` subcommand using the given arguments.
pub fn run_subcommand_ebeam(arguments: &ArgMatches) {
    if let Some(simulate_arguments) = arguments.subcommand_matches("simulate") {
        simulate::run_subcommand_simulate(simulate_arguments);
    }
}
