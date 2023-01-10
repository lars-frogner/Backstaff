//! Command line interface for the power-law electron distribution.

use super::super::accelerator::simple_power_law::create_simple_power_law_accelerator_subcommand;
use crate::update_command_graph;
use clap::Command;

/// Creates a subcommand for using the power-law distribution.
pub fn create_power_law_distribution_subcommand(
    _parent_command_name: &'static str,
) -> Command<'static> {
    let command_name = "power_law_distribution";

    update_command_graph!(_parent_command_name, command_name);

    Command::new(command_name)
        .about("Use the power-law distribution")
        .long_about(
            "Use the power-law distribution.\n\
             The distribution of non-thermal electrons is assumed to follow a power-law\n\
             described by a total power density, lower cut-off energy and a power-law\n\
             index.",
        )
        .subcommand(create_simple_power_law_accelerator_subcommand(command_name))
}
