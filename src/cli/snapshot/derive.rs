//! Command line interface for computing derived quantities for a snapshot.

use std::process;

use crate::{
    field::quantities::{DerivedSnapshotProvider3, AVAILABLE_QUANTITY_TABLE_STRING},
    grid::Grid3,
    io::{
        snapshot::{fdt, SnapshotProvider3},
        utils as io_utils,
    },
};
use clap::{Arg, ArgMatches, Command};

/// Builds a representation of the `snapshot-derive` command line subcommand.
pub fn create_derive_subcommand(parent_command_name: &'static str) -> Command<'static> {
    let command_name = "derive";

    crate::cli::command_graph::insert_command_graph_edge(parent_command_name, command_name);

    Command::new(command_name)
        .about("Computes derived quantities for the snapshot")
        .arg(
            Arg::new("quantities")
                .short('q')
                .long("quantities")
                .require_equals(true)
                .use_value_delimiter(true)
                .require_value_delimiter(true)
                .value_name("NAMES")
                .help(
                    "List of derived quantities to explicitly compute\n\
                     (comma-separated) [default: none]",
                )
                .takes_value(true)
                .multiple_values(true),
        )
        .arg(
            Arg::new("ignore-warnings")
                .long("ignore-warnings")
                .help("Automatically continue on warnings"),
        )
        .arg(
            Arg::new("verbose")
                .short('v')
                .long("verbose")
                .help("Print status messages related to computation of derived quantities"),
        )
        .after_help(&**AVAILABLE_QUANTITY_TABLE_STRING)
}

/// Creates a `DerivedQuantityComputer3` for the given arguments and snapshot provider.
pub fn create_derive_provider<G, P>(
    arguments: &ArgMatches,
    provider: P,
) -> DerivedSnapshotProvider3<G, P>
where
    G: Grid3<fdt>,
    P: SnapshotProvider3<G>,
{
    let continue_on_warnings = arguments.is_present("ignore-warnings");
    let verbose = arguments.is_present("verbose").into();

    let derived_quantity_names = arguments
        .values_of("quantities")
        .map(|values| values.collect::<Vec<_>>())
        .unwrap_or(Vec::new())
        .into_iter()
        .filter_map(|name| {
            if name.is_empty() {
                None
            } else {
                Some(String::from(name))
            }
        })
        .collect();

    DerivedSnapshotProvider3::new(provider, verbose).with_derived_quantities(
        derived_quantity_names,
        |quantity_name, missing_dependencies| {
            if let Some(missing_dependencies) = missing_dependencies {
                eprintln!(
                    "Warning: Missing following dependencies for derived quantity {}: {}",
                    quantity_name,
                    missing_dependencies.join(", ")
                );
                if !continue_on_warnings && !io_utils::user_says_yes("Still continue?", true) {
                    process::exit(1);
                }
            } else {
                eprintln!("Warning: Derived quantity {} not supported", quantity_name);
                if !continue_on_warnings && !io_utils::user_says_yes("Still continue?", true) {
                    process::exit(1);
                }
            }
        },
    )
}
