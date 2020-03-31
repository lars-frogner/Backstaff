//! Function for running the command line program.

use super::{
    build, command_graph::run_command_graph_subcommand, completions::run_completions_subcommand,
    mesh::run_create_mesh_subcommand, quantities::run_derivable_quantities_subcommand,
    snapshot::run_snapshot_subcommand,
};
use std::time::Instant;

/// Runs the `backstaff` command line program.
pub fn run() {
    let app = build::build();

    let arguments = app.get_matches();

    let protected_file_types: Vec<_> = arguments
        .values_of("protected-file-types")
        .expect("No value for argument with default")
        .collect();

    let start_instant = Instant::now();

    if let Some(snapshot_arguments) = arguments.subcommand_matches("snapshot") {
        run_snapshot_subcommand(snapshot_arguments, &protected_file_types);
    }
    if let Some(create_mesh_arguments) = arguments.subcommand_matches("create_mesh") {
        run_create_mesh_subcommand(create_mesh_arguments, &protected_file_types);
    }
    if let Some(derivable_quantities_arguments) =
        arguments.subcommand_matches("derivable_quantities")
    {
        run_derivable_quantities_subcommand(derivable_quantities_arguments);
    }
    if let Some(command_graph_arguments) = arguments.subcommand_matches("command_graph") {
        run_command_graph_subcommand(command_graph_arguments, &protected_file_types);
    }
    if let Some(completions_arguments) = arguments.subcommand_matches("completions") {
        run_completions_subcommand(completions_arguments);
    }

    if arguments.is_present("timing") {
        println!("Elapsed time: {} s", start_instant.elapsed().as_secs_f64());
    }
}
