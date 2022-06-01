//! Function for running the command line program.

use super::{
    build, completions::run_completions_subcommand, mesh::run_create_mesh_subcommand,
    snapshot::run_snapshot_subcommand,
};
use clap::ArgMatches;
use std::time::Instant;

/// Runs the `backstaff` command line program.
pub fn run() {
    let command = build::build();
    run_with_args(command.get_matches());
}

/// Runs the `backstaff` command line program with the given arguments.
pub fn run_with_args(arguments: ArgMatches) {
    let protected_file_types: Vec<_> = if arguments.is_present("no-protected-files") {
        Vec::new()
    } else {
        arguments
            .values_of("protected-file-types")
            .expect("No value for argument with default")
            .collect()
    };

    let start_instant = Instant::now();

    if let Some(snapshot_arguments) = arguments.subcommand_matches("snapshot") {
        run_snapshot_subcommand(snapshot_arguments, &protected_file_types);
    }
    if let Some(create_mesh_arguments) = arguments.subcommand_matches("create_mesh") {
        run_create_mesh_subcommand(create_mesh_arguments, &protected_file_types);
    }
    if let Some(completions_arguments) = arguments.subcommand_matches("completions") {
        run_completions_subcommand(completions_arguments);
    }

    #[cfg(feature = "command-graph")]
    if let Some(command_graph_arguments) = arguments.subcommand_matches("command_graph") {
        super::command_graph::run_command_graph_subcommand(
            command_graph_arguments,
            &protected_file_types,
        );
    }

    if arguments.is_present("timing") {
        println!("Elapsed time: {} s", start_instant.elapsed().as_secs_f64());
    }
}
