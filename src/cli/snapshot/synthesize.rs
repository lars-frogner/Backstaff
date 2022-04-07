//! Command line interface for computing synthesized quantities for a snapshot.

use crate::{
    field::quantities::DerivedSnapshotProvider3,
    grid::Grid3,
    io::snapshot::{fdt, SnapshotProvider3},
};
use clap::{Arg, ArgMatches, Command};

/// Builds a representation of the `snapshot-synthesize` command line subcommand.
pub fn _create_synthesize_subcommand(parent_command_name: &'static str) -> Command<'static> {
    let command_name = "synthesize";

    crate::cli::command_graph::insert_command_graph_edge(parent_command_name, command_name);

    Command::new(command_name)
        .about("Computes synthetic quantities for the snapshot")
        .long_about("Computes synthetic quantities for the snapshot.")
        .arg(
            Arg::new("verbose")
                .short('v')
                .long("verbose")
                .help("Print status messages related to computation of synthetic quantities"),
        )
}

pub fn create_synthesize_provider<G, P>(
    _arguments: &ArgMatches,
    _provider: P,
) -> DerivedSnapshotProvider3<G, P>
where
    G: Grid3<fdt>,
    P: SnapshotProvider3<G>,
{
    unimplemented!()
}
