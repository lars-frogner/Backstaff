//! Command line interface for computing synthesized quantities for a snapshot.

use crate::{
    cli::{
        interpolation::poly_fit::{
            construct_poly_fit_interpolator_config_from_options,
            create_poly_fit_interpolator_subcommand,
        },
        utils,
    },
    field::{
        synthesis::{EmissivitySnapshotProvider3, LineProfileMoment},
        ScalarFieldCacher3,
    },
    grid::Grid3,
    interpolation::poly_fit::{PolyFitInterpolator2, PolyFitInterpolatorConfig},
    io::snapshot::{fdt, SnapshotProvider3},
};
use clap::{Arg, ArgMatches, Command};

/// Builds a representation of the `snapshot-synthesize` command line subcommand.
pub fn create_synthesize_subcommand(parent_command_name: &'static str) -> Command<'static> {
    let command_name = "synthesize";

    crate::cli::command_graph::insert_command_graph_edge(parent_command_name, command_name);

    Command::new(command_name)
        .about("Computes synthetic quantities for the snapshot")
        .long_about("Computes synthetic quantities for the snapshot.")
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
            Arg::new("highest-moment")
                .short('M')
                .long("highest-moment")
                .require_equals(true)
                .value_name("MOMENT")
                .help(
                    "Highest moment of the spectral line profile to compute contributions to:\n\
                     0 => Compute emis_<line> (emissivity [erg/s/sr/cm³]) from tg and nel\n\
                     1 => Compute 0 and shift_<line> (mean of local doppler shift (in z) [cm] times emissivity) from tg, nel and uz\n\
                     2 => Compute 0, 1 and var_<line> (variance of local doppler shift (in z) [cm2] times emissivity) from tg, nel and uz",
                )
                .takes_value(true)
                .possible_values(["0", "1", "2"])
                .default_value("0"),
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
                .default_value("100"),
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
                .default_value("100"),
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
                     (in log₁₀ of K)",
                )
                .takes_value(true)
                .default_value("3,7"),
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
                     (in log₁₀ of cm⁻³)",
                )
                .takes_value(true)
                .default_value("8,13"),
        )
        .arg(
            Arg::new("verbose")
                .short('v')
                .long("verbose")
                .help("Print status messages related to computation of synthetic quantities"),
        )
        .subcommand(create_poly_fit_interpolator_subcommand(command_name))
}

#[cfg(feature = "synthesis")]
pub fn create_synthesize_provider<G, P>(
    arguments: &ArgMatches,
    provider: P,
) -> EmissivitySnapshotProvider3<G, P, PolyFitInterpolator2>
where
    G: Grid3<fdt>,
    P: SnapshotProvider3<G>,
{
    let line_names = arguments
        .values_of("spectral-lines")
        .map(|values| values.map(String::from).collect::<Vec<_>>())
        .unwrap_or(Vec::new());

    let highest_moment = match arguments
        .value_of("highest-moment")
        .expect("No value for argument with default")
    {
        "0" => LineProfileMoment::Zeroth,
        "1" => LineProfileMoment::First,
        "2" => LineProfileMoment::Second,
        invalid => exit_with_error!("Error: Invalid highest moment {}", invalid),
    };

    let n_temperature_points =
        utils::get_value_from_required_parseable_argument(arguments, "n-table-temperatures");

    let n_electron_density_points =
        utils::get_value_from_required_parseable_argument(arguments, "n-table-electron-densities");

    let log_temperature_limits = utils::parse_limits(arguments, "table-temperature-limits", false);
    let log_electron_density_limits =
        utils::parse_limits(arguments, "table-electron-density-limits", false);

    let verbose = arguments.is_present("verbose").into();

    let interpolator_config = if let Some(interpolator_arguments) =
        arguments.subcommand_matches("poly_fit_interpolator")
    {
        construct_poly_fit_interpolator_config_from_options(interpolator_arguments)
    } else {
        PolyFitInterpolatorConfig::default()
    };
    let interpolator = PolyFitInterpolator2::new(interpolator_config);

    if !line_names.is_empty() {
        for variable_name in ["tg", "nel"] {
            exit_on_false!(
                provider.has_variable(variable_name),
                "Error: Missing variable {} required for synthesizing emissivities",
                variable_name
            );
        }
        exit_on_false!(
            highest_moment == LineProfileMoment::Zeroth || provider.has_variable("uz"),
            "Error: Missing variable uz required for computing doppler shifts"
        );
    }

    EmissivitySnapshotProvider3::new(
        provider,
        interpolator,
        &line_names,
        highest_moment,
        n_temperature_points,
        n_electron_density_points,
        log_temperature_limits,
        log_electron_density_limits,
        verbose,
    )
}
