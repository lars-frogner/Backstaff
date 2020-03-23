//! Function for building the command line hierarchy.

use super::{
    command_graph::create_command_graph_subcommand, completions::create_completions_subcommand,
    mesh::create_create_mesh_subcommand, quantities::create_derivable_quantities_subcommand,
    snapshot::create_snapshot_subcommand,
};
use crate::create_subcommand;
use clap::{self, App, AppSettings, Arg};

/// Build the `backstaff` command line hierarchy.
pub fn build<'a, 'b>() -> App<'a, 'b> {
    App::new(clap::crate_name!())
        .version(clap::crate_version!())
        .author(clap::crate_authors!())
        .about(clap::crate_description!())
        .help_message("Print help information")
        .global_setting(AppSettings::VersionlessSubcommands)
        .global_setting(AppSettings::DisableHelpSubcommand)
        .global_setting(AppSettings::DeriveDisplayOrder)
        .setting(AppSettings::SubcommandRequiredElseHelp)
        .arg(
            Arg::with_name("timing")
                .short("t")
                .long("timing")
                .help("Display elapsed time when done"),
        )
        .subcommand(create_subcommand!(backstaff, snapshot))
        .subcommand(create_subcommand!(backstaff, create_mesh))
        .subcommand(create_subcommand!(backstaff, derivable_quantities))
        .subcommand(create_subcommand!(backstaff, command_graph))
        .subcommand(create_subcommand!(backstaff, completions))
}
