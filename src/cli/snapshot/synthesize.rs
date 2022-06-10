//! Command line interface for computing synthesized quantities for a snapshot.

use crate::{
    cli::utils as cli_utils,
    field::{
        synthesis::{EmissivitySnapshotProvider3, SYNTHESIZABLE_QUANTITY_TABLE_STRING},
        ScalarFieldCacher3,
    },
    grid::{fgr, Grid3},
    interpolation::poly_fit::{PolyFitInterpolator2, PolyFitInterpolatorConfig},
    io::snapshot::{fdt, CachingSnapshotProvider3, SnapshotProvider3},
    update_command_graph,
};
use clap::{Arg, ArgMatches, Command};
use std::{env, path::PathBuf};

/// Builds a representation of the `snapshot-synthesize` command line subcommand.
pub fn create_synthesize_subcommand(_parent_command_name: &'static str) -> Command<'static> {
    let command_name = "synthesize";

    update_command_graph!(_parent_command_name, command_name);

    Command::new(command_name)
        .about("Compute synthetic quantities for the snapshot")
        .long_about("Compute synthetic quantities for the snapshot.")
        .arg(
            Arg::new("spectral-lines")
                .short('L')
                .long("spectral-lines")
                .require_equals(true)
                .use_value_delimiter(true)
                .require_value_delimiter(true)
                .value_name("LINES")
                .help(
                    "List of spectral lines to synthesize, in format <ion>_<wavelength in Å>\n\
                     (e.g. si_4_1393.755) (comma-separated) [default: none]",
                )
                .takes_value(true)
                .multiple_values(true),
        )
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
                     (comma-separated)",
                )
                .takes_value(true)
                .multiple_values(true)
                .default_values(&["emis"]),
        )
        .arg(
            Arg::new("n-table-temperatures")
                .short('n')
                .long("n-table-temperatures")
                .require_equals(true)
                .value_name("NUMBER")
                .allow_hyphen_values(false)
                .help("Number of temperatures to use in emissivity tables")
                .takes_value(true)
                .default_value("200"),
        )
        .arg(
            Arg::new("n-table-electron-densities")
                .short('m')
                .long("n-table-electron-densities")
                .require_equals(true)
                .value_name("NUMBER")
                .allow_hyphen_values(false)
                .help("Number of electron densities to use in emissivity tables")
                .takes_value(true)
                .default_value("80"),
        )
        .arg(
            Arg::new("table-temperature-limits")
                .short('t')
                .long("table-temperature-limits")
                .require_equals(true)
                .use_value_delimiter(true)
                .require_value_delimiter(true)
                .allow_hyphen_values(true)
                .value_names(&["LOWER", "UPPER"])
                .help(
                    "Limits for temperature to use in emissivity tables\n\
                     (temperatures outside the limits will give zero emissivity)\n\
                     (in log₁₀ of K)",
                )
                .takes_value(true)
                .number_of_values(2)
                .default_value("4,7"),
        )
        .arg(
            Arg::new("table-electron-density-limits")
                .short('e')
                .long("table-electron-density-limits")
                .require_equals(true)
                .use_value_delimiter(true)
                .require_value_delimiter(true)
                .allow_hyphen_values(true)
                .value_names(&["LOWER", "UPPER"])
                .help(
                    "Limits for electron density to use in emissivity tables\n\
                     (electron densities outside the limits will give zero emissivity)\n\
                     (in log₁₀ of cm⁻³)",
                )
                .takes_value(true)
                .number_of_values(2)
                .default_value("6,13"),
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
                .help("Print status messages related to computation of synthetic quantities"),
        )
        .after_help(&**SYNTHESIZABLE_QUANTITY_TABLE_STRING)
}

/// Creates an `EmissivitySnapshotProvider3` for the given arguments and snapshot provider.
pub fn create_synthesize_provider<G, P>(
    arguments: &ArgMatches,
    provider: P,
) -> EmissivitySnapshotProvider3<G, P, PolyFitInterpolator2>
where
    G: Grid3<fgr>,
    P: CachingSnapshotProvider3<G>,
{
    match env::var_os("XUVTOP") {
        Some(ref path) => {
            if !PathBuf::from(path).is_dir() {
                exit_with_error!(
                    "Error: The XUVTOP environment variable is set but points to {}, \
                            which is not an existing directory",
                    path.to_string_lossy()
                )
            }
        }
        None => {
            exit_with_error!(
                "Error: The XUVTOP environment variable is not set, \
                        please set it to the directory of the CHIANTI database"
            )
        }
    }

    let line_names = arguments
        .values_of("spectral-lines")
        .map(|values| values.map(|name| name.to_lowercase()).collect::<Vec<_>>())
        .unwrap_or_default();

    let quantity_names: Vec<_> = arguments
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

    let n_temperature_points =
        cli_utils::get_value_from_required_parseable_argument(arguments, "n-table-temperatures");

    let n_electron_density_points = cli_utils::get_value_from_required_parseable_argument(
        arguments,
        "n-table-electron-densities",
    );

    let log_temperature_limits = cli_utils::parse_limits(
        arguments,
        "table-temperature-limits",
        cli_utils::AllowSameValue::No,
        cli_utils::AllowInfinity::No,
        None,
    );
    let log_electron_density_limits = cli_utils::parse_limits(
        arguments,
        "table-electron-density-limits",
        cli_utils::AllowSameValue::No,
        cli_utils::AllowInfinity::No,
        None,
    );

    let continue_on_warnings = arguments.is_present("ignore-warnings");
    let verbose = arguments.is_present("verbose").into();

    let interpolator = PolyFitInterpolator2::new(PolyFitInterpolatorConfig {
        order: 1,
        ..PolyFitInterpolatorConfig::default()
    });

    EmissivitySnapshotProvider3::new(
        provider,
        interpolator,
        &line_names,
        &quantity_names,
        n_temperature_points,
        n_electron_density_points,
        log_temperature_limits,
        log_electron_density_limits,
        &|quantity_name, missing_dependencies| {
            if let Some(missing_dependencies) = missing_dependencies {
                eprintln!(
                    "Warning: Missing following dependencies for synthesized quantity {}: {}",
                    quantity_name,
                    missing_dependencies.join(", ")
                );

                if !continue_on_warnings {
                    cli_utils::verify_user_will_continue_or_abort()
                }
            } else {
                eprintln!(
                    "Warning: Synthesized quantity {} not supported",
                    quantity_name
                );

                if !continue_on_warnings {
                    cli_utils::verify_user_will_continue_or_abort()
                }
            }
        },
        verbose,
    )
}

pub fn create_synthesize_provider_added_caching<G, P>(
    arguments: &ArgMatches,
    provider: P,
) -> EmissivitySnapshotProvider3<G, ScalarFieldCacher3<fdt, G, P>, PolyFitInterpolator2>
where
    G: Grid3<fgr>,
    P: SnapshotProvider3<G>,
{
    let verbose = arguments.is_present("verbose").into();
    let cached_provider = ScalarFieldCacher3::new_manual_cacher(provider, verbose);
    create_synthesize_provider(arguments, cached_provider)
}
