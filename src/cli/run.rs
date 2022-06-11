//! Function for running the command line program.

use crate::io::utils::IOContext;

use super::{
    build, completions::run_completions_subcommand, mesh::run_create_mesh_subcommand,
    snapshot::run_snapshot_subcommand,
};
use clap::ArgMatches;
use std::time::Instant;

/// Sets up a context for I/O for the running program.
pub fn setup_io() -> IOContext {
    let io_context = IOContext::new();

    let atomic_output_file_map = io_context.obtain_atomic_file_map_handle();
    ctrlc::set_handler(move || {
        atomic_output_file_map.lock().unwrap().clear();
    })
    .expect("Error setting Ctrl-C handler");

    io_context
}

/// Runs the `backstaff` command line program.
pub fn run() {
    let io_context = setup_io();
    let command = build::build();
    run_with_args(command.get_matches(), io_context);
}

/// Runs the `backstaff` command line program with the given arguments.
pub fn run_with_args(arguments: ArgMatches, mut io_context: IOContext) {
    let protected_file_types = if arguments.is_present("no-protected-files") {
        Vec::new()
    } else {
        arguments
            .values_of("protected-file-types")
            .expect("No value for argument with default")
            .map(String::from)
            .collect()
    };

    io_context.set_protected_file_types(protected_file_types);

    let start_instant = Instant::now();

    if let Some(snapshot_arguments) = arguments.subcommand_matches("snapshot") {
        run_snapshot_subcommand(snapshot_arguments, &mut io_context);
    } else if let Some(create_mesh_arguments) = arguments.subcommand_matches("create_mesh") {
        run_create_mesh_subcommand(create_mesh_arguments, &mut io_context);
    } else if let Some(completions_arguments) = arguments.subcommand_matches("completions") {
        run_completions_subcommand(completions_arguments);
    } else {
        #[cfg(feature = "command-graph")]
        if let Some(command_graph_arguments) = arguments.subcommand_matches("command_graph") {
            super::command_graph::run_command_graph_subcommand(
                command_graph_arguments,
                &mut io_context,
            );
        }
    }

    if arguments.is_present("timing") {
        println!("Elapsed time: {} s", start_instant.elapsed().as_secs_f64());
    }
}
