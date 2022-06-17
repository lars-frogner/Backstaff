//! Command line interface for computing derived quantities for a snapshot.

use crate::{
    cli::utils as cli_utils,
    exit_on_error, exit_with_error,
    field::{
        quantities::{DerivedSnapshotProvider3, AVAILABLE_QUANTITY_TABLE_STRING},
        ScalarFieldCacher3,
    },
    io::snapshot::{fdt, SnapshotProvider3},
    update_command_graph,
};
use clap::{Arg, ArgMatches, Command};

/// Builds a representation of the `snapshot-derive` command line subcommand.
pub fn create_derive_subcommand(_parent_command_name: &'static str) -> Command<'static> {
    let command_name = "derive";

    update_command_graph!(_parent_command_name, command_name);

    Command::new(command_name)
        .about("Compute derived quantities for the snapshot")
        .arg(
            Arg::new("quantities")
                .short('Q')
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
            Arg::new("max-memory-usage")
                .short('m')
                .long("max-memory-usage")
                .require_equals(true)
                .allow_hyphen_values(false)
                .value_name("PERCENTAGE")
                .help("Input fields are cached unless total memory usage exceeds this percentage")
                .takes_value(true)
                .default_value("50"),
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
        .arg(
            Arg::new("progress")
                .short('p')
                .long("progress")
                .help("Show progress bar for computation (also implies `verbose`)"),
        )
        .after_help(&**AVAILABLE_QUANTITY_TABLE_STRING)
}

/// Creates a `DerivedSnapshotProvider3` for the given arguments and snapshot provider.
pub fn create_derive_provider<P>(
    arguments: &ArgMatches,
    provider: P,
) -> DerivedSnapshotProvider3<ScalarFieldCacher3<fdt, P>>
where
    P: SnapshotProvider3,
{
    let derived_quantity_names = arguments
        .values_of("quantities")
        .map(|values| values.collect::<Vec<_>>())
        .unwrap_or_default()
        .into_iter()
        .filter_map(|name| {
            if name.is_empty() {
                None
            } else {
                Some(name.to_lowercase())
            }
        })
        .collect();

    let max_memory_usage = exit_on_error!(
        arguments
            .value_of("max-memory-usage")
            .expect("No value for argument with default")
            .trim()
            .parse::<f32>(),
        "Error: Could not parse value of max-memory-usage: {}"
    );
    if max_memory_usage < 0.0 {
        exit_with_error!("Error: max-memory-usage can not be negative");
    }

    let continue_on_warnings = arguments.is_present("ignore-warnings");
    let verbosity = cli_utils::parse_verbosity(arguments, true);

    let cached_provider =
        ScalarFieldCacher3::new_automatic_cacher(provider, max_memory_usage, verbosity.clone());

    DerivedSnapshotProvider3::new(
        cached_provider,
        derived_quantity_names,
        &|quantity_name, missing_dependencies| {
            if let Some(missing_dependencies) = missing_dependencies {
                eprintln!(
                    "Warning: Missing following dependencies for derived quantity {}: {}",
                    quantity_name,
                    missing_dependencies.join(", ")
                );
                if !continue_on_warnings {
                    cli_utils::verify_user_will_continue_or_abort()
                }
            } else {
                eprintln!("Warning: Derived quantity {} not supported", quantity_name);
                if !continue_on_warnings {
                    cli_utils::verify_user_will_continue_or_abort()
                }
            }
        },
        verbosity,
    )
}
