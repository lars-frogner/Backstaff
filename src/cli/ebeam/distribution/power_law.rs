//! Command line interface for the power-law electron distribution.

use crate::cli;
use crate::ebeam::distribution::power_law::PowerLawDistributionConfig;
use crate::grid::Grid3;
use crate::io::snapshot::{fdt, SnapshotReader3};
use crate::units::solar::{U_E, U_T};
use clap::{App, Arg, ArgMatches, SubCommand};

/// Creates a subcommand for using the power-law distribution.
pub fn create_power_law_distribution_subcommand<'a, 'b>() -> App<'a, 'b> {
    SubCommand::with_name("power_law_distribution")
        .about("Use the power-law distribution")
        .long_about(
            "Use the power-law distribution.\n\
             The distribution of non-thermal electrons is assumed to follow a power-law\n\
             described by a total power density, lower cut-off energy and a power-law\n\
             index.",
        )
        .help_message("Print help information")
        .arg(
            Arg::with_name("min-remaining-power-density")
                .long("min-remaining-power-density")
                .require_equals(true)
                .value_name("VALUE")
                .help(
                    "Distributions with remaining power densities smaller than this\n\
                     value are discarded [erg/(cm^3 s)] [default: from param file]",
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
}

/// Determines power-law distribution parameters based on
/// provided options and values in parameter file.
pub fn construct_power_law_distribution_config_from_options<G: Grid3<fdt>>(
    arguments: &ArgMatches,
    reader: &SnapshotReader3<G>,
) -> PowerLawDistributionConfig {
    let min_remaining_power_density = cli::get_value_from_param_file_argument_with_default(
        reader,
        arguments,
        "min-remaining-power-density",
        "min_stop_en",
        &|min_stop_en| min_stop_en * U_E / U_T,
        PowerLawDistributionConfig::DEFAULT_MIN_REMAINING_POWER_DENSITY,
    );
    let max_propagation_distance = cli::get_value_from_param_file_argument_with_default(
        reader,
        arguments,
        "max-propagation-distance",
        "max_dist",
        &|max_dist| max_dist,
        PowerLawDistributionConfig::DEFAULT_MAX_PROPAGATION_DISTANCE,
    );
    PowerLawDistributionConfig {
        min_remaining_power_density,
        max_propagation_distance,
    }
}
