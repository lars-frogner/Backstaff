//! Command line interface for creating Bifrost mesh files.

mod regular;

use self::regular::{create_regular_mesh_subcommand, run_regular_subcommand};
use crate::{
    create_subcommand, exit_on_error, exit_with_error,
    grid::Grid3,
    io::snapshot::{fdt, native},
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
                .help("Automatically overwrite any existing file"),
        )
        .subcommand(create_subcommand!(create_mesh, regular_mesh))
}

/// Runs the actions for the `create_mesh` subcommand using the given arguments.
pub fn run_create_mesh_subcommand(arguments: &ArgMatches) {
    if let Some(regular_arguments) = arguments.subcommand_matches("regular") {
        run_regular_subcommand(arguments, regular_arguments);
    } else {
        exit_with_error!("Error: No resampling mode specified");
    };
}

fn write_mesh_file<G: Grid3<fdt>>(root_arguments: &ArgMatches, grid: G) {
    let mut output_file_path = exit_on_error!(
        PathBuf::from_str(
            root_arguments
                .value_of("output-file")
                .expect("No value for required argument."),
        ),
        "Error: Could not interpret path to output file: {}"
    );

    if output_file_path.extension().is_none() {
        output_file_path.set_extension("mesh");
    }

    let force_overwrite = root_arguments.is_present("overwrite");

    exit_on_error!(
        native::write_mesh_file_from_grid(&grid, output_file_path, force_overwrite),
        "Error: Could not write mesh file: {}"
    );
}
