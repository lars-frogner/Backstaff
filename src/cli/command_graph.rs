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

pub const CONFIG_COMMANDS: [&'static str; 14] = [
    "weighted_sample_averaging",
    "weighted_cell_averaging",
    "direct_sampling",
    "poly_fit_interpolator",
    "basic_field_line_tracer",
    "slice_seeder",
    "manual_seeder",
    "rkf_stepper",
    "power_law_distribution",
    "manual_detector",
    "simple_detector",
    "simple_power_law_accelerator",
    "manual_reconnection_site_detector",
    "simple_reconnection_site_detector",
];

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
            Arg::with_name("include-configuration")
                .long("include-configuration")
                .help("Include optional configuration commands"),
        )
        .arg(
            Arg::with_name("overwrite")
                .long("overwrite")
                .help("Automatically overwrite any existing file (unless listed as protected)"),
        )
        .help_message("Print help information")
}

/// Runs the actions for the `command_graph` subcommand using the given arguments.
pub fn run_command_graph_subcommand(arguments: &ArgMatches, protected_file_types: &[&str]) {
    let output_file_path = exit_on_error!(
        PathBuf::from_str(
            arguments
                .value_of("output-file")
                .expect("No value for required argument"),
        ),
        "Error: Could not interpret path to output file: {}"
    );

    let include_configuration = arguments.is_present("include-configuration");
    let automatic_overwrite = arguments.is_present("overwrite");

    let mut command_graph = COMMAND_GRAPH.lock().unwrap().clone();

    if !include_configuration {
        command_graph.retain_edges(|graph, edge_index| !graph.edge_weight(edge_index).unwrap());
        command_graph
            .retain_nodes(|graph, node_index| graph.neighbors_undirected(node_index).count() > 0);
    };

    let dot_text = format!(
        "{}",
        Dot::with_config(&command_graph, &[Config::EdgeNoLabel])
    );

    if utils::write_should_be_skipped(&output_file_path, automatic_overwrite, protected_file_types)
    {
        return;
    }

    exit_on_error!(
        utils::write_text_file(&dot_text, output_file_path),
        "Error: Could not write command graph: {}"
    );
}

lazy_static! {
    static ref COMMAND_NODES: Mutex<HashMap<&'static str, NodeIndex<u32>>> =
        Mutex::new(HashMap::new());
    static ref COMMAND_GRAPH: Mutex<Graph<&'static str, bool, Directed>> = Mutex::new(Graph::new());
}

/// Adds a parent-child command relationship to the global command graph.
pub fn insert_command_graph_edge(
    parent_command_name: &'static str,
    child_command_name: &'static str,
) {
    let is_config = CONFIG_COMMANDS.contains(&parent_command_name)
        || CONFIG_COMMANDS.contains(&child_command_name);
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
        .add_edge(parent_index, child_index, is_config);
}
