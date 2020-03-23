//! Command line interface for creating a graph of the command hierarchy.

use crate::{exit_on_error, io::utils};
use clap::{self, App, AppSettings, Arg, ArgMatches, SubCommand};
use lazy_static::lazy_static;
use petgraph::{
    dot::{Config, Dot},
    graph::NodeIndex,
    Directed, Graph,
};
use std::{
    collections::HashMap,
    path::PathBuf,
    str::{self, FromStr},
    sync::Mutex,
};

/// Builds a representation of the `command_graph` command line subcommand.
pub fn create_command_graph_subcommand<'a, 'b>() -> App<'a, 'b> {
    SubCommand::with_name("command_graph")
        .about("Output a graph of the command hierarchy as a DOT file")
        .setting(AppSettings::Hidden)
        .long_about(
            "Output a graph of the command hierarchy as a DOT file.\n\
             The DOT (.gv) file can be rendered using Graphviz:\n\n    dot -Tpdf -O <DOT file>",
        )
        .arg(
            Arg::with_name("output-file")
                .value_name("OUTPUT_FILE")
                .help("Path of the output DOT file to produce.")
                .takes_value(true)
                .default_value("command_graph.gv"),
        )
        .arg(
            Arg::with_name("overwrite")
                .long("overwrite")
                .help("Automatically overwrite any existing file"),
        )
        .help_message("Print help information")
}

/// Runs the actions for the `command_graph` subcommand using the given arguments.
pub fn run_command_graph_subcommand(arguments: &ArgMatches) {
    let output_file_path = exit_on_error!(
        PathBuf::from_str(
            arguments
                .value_of("output-file")
                .expect("Required argument not present."),
        ),
        "Error: Could not interpret path to output file: {}"
    );

    let force_overwrite = arguments.is_present("overwrite");

    let dot_text = format!(
        "{}",
        Dot::with_config(
            &COMMAND_GRAPH.lock().unwrap().clone(),
            &[Config::EdgeNoLabel]
        )
    );
    exit_on_error!(
        utils::write_text_file(&dot_text, output_file_path, force_overwrite),
        "Error: Could not write command graph: {}"
    );
}

lazy_static! {
    static ref COMMAND_NODES: Mutex<HashMap<&'static str, NodeIndex<u32>>> =
        Mutex::new(HashMap::new());
    static ref COMMAND_GRAPH: Mutex<Graph<&'static str, f32, Directed>> = Mutex::new(Graph::new());
}

/// Adds a parent-child command relationship to the global command graph.
pub fn insert_command_graph_edge(
    parent_command_name: &'static str,
    child_command_name: &'static str,
    weight: f32,
) {
    let parent_index = *COMMAND_NODES
        .lock()
        .unwrap()
        .entry(parent_command_name)
        .or_insert_with(|| COMMAND_GRAPH.lock().unwrap().add_node(parent_command_name));
    let child_index = *COMMAND_NODES
        .lock()
        .unwrap()
        .entry(child_command_name)
        .or_insert_with(|| COMMAND_GRAPH.lock().unwrap().add_node(child_command_name));
    COMMAND_GRAPH
        .lock()
        .unwrap()
        .add_edge(parent_index, child_index, weight);
}
