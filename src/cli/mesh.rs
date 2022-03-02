//! Command line interface for creating Bifrost mesh files.

mod horizontally_regular;
mod regular;

use self::{
    horizontally_regular::{
        create_horizontally_regular_mesh_subcommand, run_horizontally_regular_subcommand,
    },
    regular::{create_regular_mesh_subcommand, run_regular_subcommand},
};
use crate::{
    cli::utils as cli_utils,
    create_subcommand, exit_on_error, exit_with_error,
    grid::Grid3,
    io::{
        snapshot::{fdt, native},
        utils::AtomicOutputPath,
    },
};
use clap::{App, AppSettings, Arg, ArgMatches, SubCommand};
use std::{path::PathBuf, str::FromStr};

/// Creates a subcommand for generating a Bifrost mesh file.
pub fn create_create_mesh_subcommand<'a, 'b>() -> App<'a, 'b> {
    SubCommand::with_name("create_mesh")
        .about("Create a Bifrost mesh file")
        .help_message("Print help information")
        .setting(AppSettings::SubcommandRequired)
        .arg(
            Arg::with_name("output-file")
                .value_name("OUTPUT_FILE")
                .help("Path where the mesh file should be created")
                .required(true)
                .takes_value(true),
        )
        .arg(
            Arg::with_name("overwrite")
                .long("overwrite")
                .help("Automatically overwrite any existing files (unless listed as protected)")
                .conflicts_with("no-overwrite"),
        )
        .arg(
            Arg::with_name("no-overwrite")
                .long("no-overwrite")
                .help("Do not overwrite any existing files")
                .conflicts_with("overwrite"),
        )
        .subcommand(create_subcommand!(create_mesh, regular_mesh))
        .subcommand(create_subcommand!(create_mesh, horizontally_regular_mesh))
}

/// Runs the actions for the `create_mesh` subcommand using the given arguments.
pub fn run_create_mesh_subcommand(arguments: &ArgMatches, protected_file_types: &[&str]) {
    if let Some(regular_arguments) = arguments.subcommand_matches("regular") {
        run_regular_subcommand(arguments, regular_arguments, protected_file_types);
    } else if let Some(horizontally_regular_arguments) =
        arguments.subcommand_matches("horizontally_regular")
    {
        run_horizontally_regular_subcommand(
            arguments,
            horizontally_regular_arguments,
            protected_file_types,
        );
    } else {
        exit_with_error!("Error: No resampling mode specified");
    };
}

fn write_mesh_file<G: Grid3<fdt>>(
    root_arguments: &ArgMatches,
    grid: G,
    protected_file_types: &[&str],
) {
    let mut output_file_path = exit_on_error!(
        PathBuf::from_str(
            root_arguments
                .value_of("output-file")
                .expect("No value for required argument"),
        ),
        "Error: Could not interpret path to output file: {}"
    );

    if output_file_path.extension().is_none() {
        output_file_path.set_extension("mesh");
    }

    let overwrite_mode = cli_utils::overwrite_mode_from_arguments(root_arguments);

    let atomic_output_path = exit_on_error!(
        AtomicOutputPath::new(output_file_path),
        "Error: Could not create temporary output file: {}"
    );

    if !atomic_output_path.check_if_write_allowed(overwrite_mode, protected_file_types) {
        return;
    }

    exit_on_error!(
        native::write_mesh_file_from_grid(&grid, atomic_output_path.temporary_path()),
        "Error: Could not write mesh file: {}"
    );

    exit_on_error!(
        atomic_output_path.perform_replace(),
        "Error: Could not move temporary output file to target path: {}"
    );
}
