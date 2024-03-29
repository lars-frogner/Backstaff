//! Command line interface for creating Bifrost mesh files.

mod horizontally_extended;
mod horizontally_regular;
mod regular;

use self::{
    horizontally_extended::{
        create_horizontally_extended_subcommand, run_horizontally_extended_subcommand,
    },
    horizontally_regular::{
        create_horizontally_regular_subcommand, run_horizontally_regular_subcommand,
    },
    regular::{create_regular_subcommand, run_regular_subcommand},
};
use crate::{
    cli::utils as cli_utils,
    exit_on_error, exit_with_error,
    grid::{fgr, Grid3},
    io::{snapshot::native, utils::IOContext},
    update_command_graph,
};
use clap::{Arg, ArgMatches, Command};
use std::{path::PathBuf, str::FromStr};

/// Creates a subcommand for generating a Bifrost mesh file.
pub fn create_create_mesh_subcommand(_parent_command_name: &'static str) -> Command<'static> {
    let command_name = "create_mesh";

    update_command_graph!(_parent_command_name, command_name);

    Command::new(command_name)
        .about("Create a Bifrost mesh file")
        .subcommand_required(true)
        .arg(
            Arg::new("output-file")
                .value_name("OUTPUT_FILE")
                .help("Path where the mesh file should be created")
                .required(true)
                .takes_value(true),
        )
        .arg(
            Arg::new("overwrite")
                .long("overwrite")
                .help("Automatically overwrite any existing files (unless listed as protected)")
                .conflicts_with("no-overwrite"),
        )
        .arg(
            Arg::new("no-overwrite")
                .long("no-overwrite")
                .help("Do not overwrite any existing files")
                .conflicts_with("overwrite"),
        )
        .arg(
            Arg::new("verbose")
                .short('v')
                .long("verbose")
                .help("Print status messages related to mesh file creation"),
        )
        .subcommand(create_regular_subcommand(command_name))
        .subcommand(create_horizontally_regular_subcommand(command_name))
        .subcommand(create_horizontally_extended_subcommand(command_name))
}

/// Runs the actions for the `create_mesh` subcommand using the given arguments.
pub fn run_create_mesh_subcommand(arguments: &ArgMatches, io_context: &mut IOContext) {
    if let Some(regular_arguments) = arguments.subcommand_matches("regular") {
        run_regular_subcommand(arguments, regular_arguments, io_context);
    } else if let Some(horizontally_regular_arguments) =
        arguments.subcommand_matches("horizontally_regular")
    {
        run_horizontally_regular_subcommand(arguments, horizontally_regular_arguments, io_context);
    } else if let Some(horizontally_extended_arguments) =
        arguments.subcommand_matches("horizontally_extended")
    {
        run_horizontally_extended_subcommand(
            arguments,
            horizontally_extended_arguments,
            io_context,
        );
    } else {
        exit_with_error!("Error: No resampling mode specified");
    };
}

fn write_mesh_file<G: Grid3<fgr>>(
    root_arguments: &ArgMatches,
    grid: G,
    io_context: &mut IOContext,
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
    let verbosity = cli_utils::parse_verbosity(root_arguments, false);

    io_context.set_overwrite_mode(overwrite_mode);

    let atomic_output_file = exit_on_error!(
        io_context.create_atomic_output_file(output_file_path),
        "Error: Could not create temporary output file: {}"
    );

    if !atomic_output_file.check_if_write_allowed(io_context, &verbosity) {
        return;
    }

    exit_on_error!(
        native::write_mesh_file_from_grid(&grid, atomic_output_file.temporary_path()),
        "Error: Could not write mesh file: {}"
    );

    exit_on_error!(
        io_context.close_atomic_output_file(atomic_output_file),
        "Error: Could not move temporary output file to target path: {}"
    );
}
