//! Command line interface for creating or removing auxiliary variables for a snapshot.

use crate::grid::Grid3;
use crate::io::snapshot::{fdt, SnapshotReader3};
use chrono::Local;
use clap::{App, Arg, ArgMatches, SubCommand};
use std::fs;

mod compute;
mod remove;

/// Builds a representation of the `snapshot-modify_aux` command line subcommand.
pub fn create_modify_aux_subcommand<'a, 'b>() -> App<'a, 'b> {
    SubCommand::with_name("modify_aux")
        .about("Creates or removes auxiliary quantities associated with the snapshot")
        .long_about(
            "Creates or removes auxiliary quantities associated with the snapshot.\n\
             Backups of the relevant .idl and .aux files will be created before they are\n\
             modified.",
        )
        .arg(
            Arg::with_name("no-backups").long("no-backups").help(
                "Do not create backups of the .idl and .aux files associated with the snapshot",
            ),
        )
        .arg(
            Arg::with_name("verbose")
                .short("v")
                .long("verbose")
                .help("Print status messages"),
        )
        .help_message("Print help information")
        .subcommand(compute::create_compute_subcommand())
        .subcommand(remove::create_remove_subcommand())
}

/// Runs the actions for the `snapshot-modify_aux` subcommand using the given arguments.
pub fn run_modify_aux_subcommand<G: Grid3<fdt>>(
    arguments: &ArgMatches,
    reader: &mut SnapshotReader3<G>,
) {
    let no_backups = arguments.is_present("no-backups");
    let is_verbose = arguments.is_present("verbose");

    let parameter_file_path = reader.parameter_file_path();
    let auxiliary_variable_file_path = reader.auxiliary_variable_file_path();

    let datetime_string = Local::now().to_rfc3339();

    // Create temporary file names for new idl and aux files
    let temp_parameter_file_path = parameter_file_path.with_extension(format!(
        "{}tmp_{}",
        parameter_file_path
            .extension()
            .map_or(String::from(""), |ext| format!(
                "{}.",
                ext.to_string_lossy()
            )),
        datetime_string
    ));
    let temp_auxiliary_variable_file_path = auxiliary_variable_file_path.with_extension(format!(
        "{}tmp_{}",
        auxiliary_variable_file_path
            .extension()
            .map_or(String::from(""), |ext| format!(
                "{}.",
                ext.to_string_lossy()
            )),
        datetime_string
    ));

    let auxiliary_quantity_names =
        if let Some(compute_arguments) = arguments.subcommand_matches("compute") {
            compute::generate_auxiliary_variable_file(
                compute_arguments,
                reader,
                &temp_auxiliary_variable_file_path,
                is_verbose,
            )
        } else if let Some(remove_arguments) = arguments.subcommand_matches("remove") {
            remove::generate_auxiliary_variable_file(
                remove_arguments,
                reader,
                &temp_auxiliary_variable_file_path,
                is_verbose,
            )
        } else {
            panic!("No modification mode specified.")
        };

    // Update aux quantity list in parameter file
    let mut parameter_file = reader.parameter_file().clone();
    parameter_file.replace_parameter_value(
        "aux",
        &format!("\"{}\"", auxiliary_quantity_names.join(" ")),
    );
    if is_verbose {
        println!(
            "Writing updated parameter file: {}",
            temp_parameter_file_path.display()
        );
    }
    parameter_file
        .write(&temp_parameter_file_path)
        .unwrap_or_else(|err| panic!("Could not update parameter file: {}", err));

    // Create backups of idl and aux file
    if !no_backups {
        let backup_parameter_file_path = parameter_file_path.with_extension(format!(
            "{}backup_{}",
            parameter_file_path
                .extension()
                .map_or(String::from(""), |ext| format!(
                    "{}.",
                    ext.to_string_lossy()
                )),
            datetime_string
        ));
        let backup_auxiliary_variable_file_path =
            auxiliary_variable_file_path.with_extension(format!(
                "{}backup_{}",
                auxiliary_variable_file_path
                    .extension()
                    .map_or(String::from(""), |ext| format!(
                        "{}.",
                        ext.to_string_lossy()
                    )),
                datetime_string
            ));
        if is_verbose {
            println!(
                "Creating backup parameter file: {}",
                backup_parameter_file_path.display()
            );
        }
        fs::rename(&parameter_file_path, &backup_parameter_file_path)
            .unwrap_or_else(|err| panic!("Could not back up parameter file: {}", err));
        if is_verbose {
            println!(
                "Creating backup auxiliary variable file: {}",
                backup_auxiliary_variable_file_path.display()
            );
        }
        fs::rename(
            &auxiliary_variable_file_path,
            &backup_auxiliary_variable_file_path,
        )
        .unwrap_or_else(|err| panic!("Could not back up auxiliary variable file: {}", err));
    }

    // Overwrite original idl and aux file
    if is_verbose {
        println!(
            "Overwriting original parameter file: {}",
            parameter_file_path.display()
        );
    }
    fs::rename(&temp_parameter_file_path, &parameter_file_path)
        .unwrap_or_else(|err| panic!("Could not overwrite original parameter file: {}", err));

    if is_verbose {
        println!(
            "Overwriting original auxiliary variable file: {}",
            auxiliary_variable_file_path.display()
        );
    }
    fs::rename(
        &temp_auxiliary_variable_file_path,
        &auxiliary_variable_file_path,
    )
    .unwrap_or_else(|err| {
        panic!(
            "Could not overwrite original auxiliary variable file: {}",
            err
        )
    });

    reader
        .reread()
        .unwrap_or_else(|err| panic!("Could not sync reader object with updated files: {}", err));
}
