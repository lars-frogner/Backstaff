//! Command line interface for actions related to inspection of snapshots.

#[cfg(feature = "statistics")]
mod statistics;

use crate::{
    cli::utils as cli_utils,
    grid::{fgr, Grid3},
    io::{snapshot::SnapshotProvider3, utils::IOContext},
    update_command_graph,
};
use clap::{Arg, ArgMatches, Command};

#[cfg(feature = "statistics")]
use self::statistics::{create_statistics_subcommand, run_statistics_subcommand};

/// Builds a representation of the `snapshot-inspect` command line subcommand.
pub fn create_inspect_subcommand(_parent_command_name: &'static str) -> Command<'static> {
    let command_name = "inspect";

    update_command_graph!(_parent_command_name, command_name);

    let command = Command::new(command_name)
        .about("Inspect properties of the snapshot")
        .arg(
            Arg::new("included-quantities")
                .long("included-quantities")
                .short('I')
                .require_equals(true)
                .use_value_delimiter(true)
                .require_value_delimiter(true)
                .value_name("NAMES")
                .help("List of the original quantities to inspect (comma-separated)")
                .takes_value(true)
                .multiple_values(true)
                .default_value("all")
                .conflicts_with_all(&["excluded-quantities"]),
        )
        .arg(
            Arg::new("excluded-quantities")
                .long("excluded-quantities")
                .short('E')
                .require_equals(true)
                .use_value_delimiter(true)
                .require_value_delimiter(true)
                .value_name("NAMES")
                .help("List of original quantities to ignore (comma-separated)")
                .takes_value(true)
                .multiple_values(true)
                .conflicts_with_all(&["included-quantities"]),
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
                .help("Print status messages related to inspection"),
        );

    #[cfg(feature = "statistics")]
    let command = command
        .subcommand_required(true)
        .subcommand(create_statistics_subcommand(command_name));

    command
}

/// Runs the actions for the `snapshot-inspect` subcommand using the given arguments.
pub fn run_inspect_subcommand<G, P>(arguments: &ArgMatches, provider: P, io_context: &mut IOContext)
where
    G: Grid3<fgr>,
    P: SnapshotProvider3<G>,
{
    let continue_on_warnings = arguments.is_present("ignore-warnings");
    let verbosity = cli_utils::parse_verbosity(arguments, false);

    let quantity_names =
        super::parse_included_quantity_list(arguments, &provider, continue_on_warnings);

    if quantity_names.is_empty() {
        exit_with_error!("Aborted: No quantities to inspect");
    }

    #[cfg(feature = "statistics")]
    if let Some(statistics_arguments) = arguments.subcommand_matches("statistics") {
        run_statistics_subcommand(
            statistics_arguments,
            provider,
            quantity_names,
            io_context,
            &verbosity,
        );
    }
    #[cfg(not(feature = "statistics"))]
    exit_with_error!(
        "Error: Compile with statistics feature in order to inspect snapshot statistics\n\
         Tip: Use cargo flag --features=statistics"
    );
}
