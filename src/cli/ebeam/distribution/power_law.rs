//! Command line interface for the power-law electron distribution.

use crate::cli;
use crate::ebeam::distribution::power_law::PowerLawDistributionConfig;
use crate::grid::Grid3;
use crate::io::snapshot::{fdt, SnapshotReader3};
use clap::{App, Arg, ArgMatches, SubCommand};

/// Creates a subcommand for using the power-law distribution.
pub fn create_power_law_distribution_subcommand<'a, 'b>() -> App<'a, 'b> {
    SubCommand::with_name("power_law_distribution")
        .about("Use the power-law distribution")
        .long_about(
            "Use the power-law distribution.\n\
             The distribution of non-thermal electrons is assumed to follow a power-law\n\
             described by a total power density, lower cut-off energy and a power-law\n\
             index. The transport method follows Hawley & Fisher (1994).",
        )
        .help_message("Print help information")
        .arg(
            Arg::with_name("min-residual-factor")
                .long("min-residual-factor")
                .require_equals(true)
                .value_name("VALUE")
                .help(
                    "Distributions are considered thermalized when the heating has been reduced to\n\
                     this fraction of the initial heating.\n\
                     [default: from param file]",
                )
                .takes_value(true),
        )
        .arg(
            Arg::with_name("max-propagation-distance")
                .long("max-propagation-distance")
                .require_equals(true)
                .value_name("VALUE")
                .help(
                    "Maximum distance the distribution can propagate before propagation\n\
                     should be terminated [Mm] [default: from param file]",
                )
                .takes_value(true),
        )
        .arg(
            Arg::with_name("continue-thermalized-beams")
                .long("continue-thermalized-beams")
                .help("Keep propagating beams even after they are considered thermalized"),
        )
}

/// Determines power-law distribution parameters based on
/// provided options and values in parameter file.
pub fn construct_power_law_distribution_config_from_options<G: Grid3<fdt>>(
    arguments: &ArgMatches,
    reader: &SnapshotReader3<G>,
) -> PowerLawDistributionConfig {
    let min_residual_factor = cli::get_value_from_param_file_argument_with_default(
        reader,
        arguments,
        "min-residual-factor",
        "min_heat_frac",
        &|min_heat_frac| min_heat_frac,
        PowerLawDistributionConfig::DEFAULT_MIN_RESIDUAL_FACTOR,
    );
    let max_propagation_distance = cli::get_value_from_param_file_argument_with_default(
        reader,
        arguments,
        "max-propagation-distance",
        "max_dist",
        &|max_dist| max_dist,
        PowerLawDistributionConfig::DEFAULT_MAX_PROPAGATION_DISTANCE,
    );
    let continue_thermalized_beams = arguments.is_present("continue-thermalized-beams");

    PowerLawDistributionConfig {
        min_residual_factor,
        max_propagation_distance,
        continue_thermalized_beams,
    }
}
