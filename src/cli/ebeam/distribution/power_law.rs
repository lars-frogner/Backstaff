//! Command line interface for the power-law electron distribution.

use crate::{
    add_subcommand_combinations,
    cli::{
        ebeam::{
            accelerator::simple_power_law::create_simple_power_law_accelerator_subcommand,
            propagator::analytical::create_analytical_propagator_subcommand,
        },
        interpolation::poly_fit::create_poly_fit_interpolator_subcommand,
        tracing::stepping::rkf::create_rkf_stepper_subcommand,
    },
    update_command_graph,
};
use clap::Command;

/// Creates a subcommand for using the power-law distribution.
pub fn create_power_law_distribution_subcommand(
    _parent_command_name: &'static str,
) -> Command<'static> {
    let command_name = "power_law_distribution";

    update_command_graph!(_parent_command_name, command_name);

    let command = Command::new(command_name)
        .about("Use the power-law distribution")
        .long_about(
            "Use the power-law distribution.\n\
             The distribution of non-thermal electrons is assumed to follow a power-law\n\
             described by a total power density, lower cut-off energy and a power-law\n\
             index.",
        )
        .subcommand(create_simple_power_law_accelerator_subcommand(command_name))
        .subcommand(create_analytical_propagator_subcommand(command_name));

    add_subcommand_combinations!(command, command_name, false; poly_fit_interpolator, rkf_stepper)
}
