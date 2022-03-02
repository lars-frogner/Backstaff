//! Command line interface for actions related to inspection of snapshots.

mod statistics;

use self::statistics::{create_statistics_subcommand, run_statistics_subcommand};
use crate::{
    create_subcommand,
    field::quantities,
    grid::Grid3,
    io::snapshot::{fdt, SnapshotReader3},
};
use clap::{Arg, ArgMatches, Command};

/// Builds a representation of the `snapshot-inspect` command line subcommand.
pub fn create_inspect_subcommand() -> Command<'static> {
    Command::new("inspect")
        .about("Inspect properties of the snapshot")
        .subcommand_required(true)
        .arg(
            Arg::new("all-quantities")
                .long("all-quantities")
                .help("Include all original quantities in the output snapshot")
                .conflicts_with_all(&["included-quantities", "excluded-quantities"]),
        )
        .arg(
            Arg::new("included-quantities")
                .long("included-quantities")
                .require_equals(true)
                .use_value_delimiter(true)
                .require_value_delimiter(true)
                .value_name("NAMES")
                .help(
                    "List of all the original quantities to inspect (comma-separated)\n\
                     [default: none]",
                )
                .takes_value(true)
                .multiple_values(true)
                .conflicts_with_all(&["all-quantities", "excluded-quantities"]),
        )
        .arg(
            Arg::new("excluded-quantities")
                .long("excluded-quantities")
                .require_equals(true)
                .use_value_delimiter(true)
                .require_value_delimiter(true)
                .value_name("NAMES")
                .help("List of original quantities to ignore (comma-separated) [default: none]")
                .takes_value(true)
                .multiple_values(true)
                .conflicts_with_all(&["all-quantities", "included-quantities"]),
        )
        .arg(
            Arg::new("derived-quantities")
                .long("derived-quantities")
                .require_equals(true)
                .use_value_delimiter(true)
                .require_value_delimiter(true)
                .value_name("NAMES")
                .help(
                    "List of derived quantities to compute and inspect (comma-separated)\n\
                     [default: none]",
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
                .help("Print status messages related to inspection"),
        )
        .subcommand(create_subcommand!(inspect, statistics))
}

/// Runs the actions for the `snapshot-inspect` subcommand using the given arguments.
pub fn run_inspect_subcommand<G, R>(arguments: &ArgMatches, reader: &R)
where
    G: Grid3<fdt>,
    R: SnapshotReader3<G>,
{
    let continue_on_warnings = arguments.is_present("ignore-warnings");
    let verbose = arguments.is_present("verbose").into();

    let (included_quantities, derived_quantities) =
        super::parse_quantity_lists(arguments, reader, continue_on_warnings);

    let quantity_names: Vec<_> = included_quantities
        .iter()
        .cloned()
        .chain(derived_quantities.iter().cloned())
        .collect();

    if quantity_names.is_empty() {
        exit_with_error!("Aborted: No quantities to write");
    }

    if let Some(statistics_arguments) = arguments.subcommand_matches("statistics") {
        run_statistics_subcommand(
            statistics_arguments,
            reader,
            |name| match if included_quantities.contains(&name) {
                reader.read_scalar_field(name)
            } else if derived_quantities.contains(&name) {
                quantities::compute_quantity(reader, name, verbose)
            } else {
                unreachable!()
            } {
                Ok(field) => Ok(field),
                Err(err) => Err(err),
            },
            quantity_names,
        );
    }
}
