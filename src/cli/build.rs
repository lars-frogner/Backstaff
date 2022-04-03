//! Function for building the command line hierarchy.

use super::{
    command_graph::create_command_graph_subcommand, completions::create_completions_subcommand,
    mesh::create_create_mesh_subcommand, quantities::create_derivable_quantities_subcommand,
    snapshot::create_snapshot_subcommand,
};
use clap::{self, AppSettings, Arg, Command};

/// Build the `backstaff` command line hierarchy.
pub fn build() -> Command<'static> {
    let command_name = "backstaff";

    Command::new(clap::crate_name!())
        .version(clap::crate_version!())
        .author(clap::crate_authors!())
        .about(clap::crate_description!())

        .propagate_version(false)
        .disable_help_subcommand(true)
        .global_setting(AppSettings::DeriveDisplayOrder)
        .subcommand_required(true)
        .arg_required_else_help(true)
        .arg(
            Arg::new("timing")
                .short('t')
                .long("timing")
                .help("Display elapsed time when done"),
        )
        .arg(
            Arg::new("protected-file-types")
                .long("protected-file-types")
                .require_equals(true)
                .use_value_delimiter(true)
                .require_value_delimiter(true)
                .value_name("EXTENSIONS")
                .help(
                    "List of extensions for file types that never should be overwritten automatically\n\
                    (comma-separated)",
                )
                .takes_value(true)
                .multiple_values(true)
                .default_value("idl,snap,aux"),
        )
        .subcommand(create_snapshot_subcommand(command_name))
        .subcommand(create_create_mesh_subcommand(command_name))
        .subcommand(create_derivable_quantities_subcommand(command_name))
        .subcommand(create_command_graph_subcommand())
        .subcommand(create_completions_subcommand())
}
