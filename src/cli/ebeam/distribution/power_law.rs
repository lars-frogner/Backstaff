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
                    "Distributions are considered depleted when the residual energy factor has\n\
                     decreased below this limit, given that the deposited power density is smaller\n\
                     than its lower limit [default: from param file]",
                )
                .takes_value(true),
        )
        .arg(
            Arg::with_name("min-deposited-power-density")
                .long("min-deposited-power-density")
                .require_equals(true)
                .value_name("VALUE")
                .help(
                    "Distributions are considered depleted when the deposited power density has\n\
                     decreased below this limit, given that the residual energy factor is smaller\n\
                     than its lower limit [default: from param file]",
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
            Arg::with_name("continue-depleted-beams")
                .long("continue-depleted-beams")
                .help("Keep propagating beams even after they are considered depleted"),
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
        "min_residual",
        &|min_residual| min_residual,
        PowerLawDistributionConfig::DEFAULT_MIN_RESIDUAL_FACTOR,
    );
    let min_deposited_power_density = cli::get_value_from_param_file_argument_with_default(
        reader,
        arguments,
        "min-deposited-power-density",
        "min_qbeam",
        &|min_qbeam| min_qbeam,
        PowerLawDistributionConfig::DEFAULT_MIN_DEPOSITED_POWER_DENSITY,
    );
    let max_propagation_distance = cli::get_value_from_param_file_argument_with_default(
        reader,
        arguments,
        "max-propagation-distance",
        "max_dist",
        &|max_dist| max_dist,
        PowerLawDistributionConfig::DEFAULT_MAX_PROPAGATION_DISTANCE,
    );
    let continue_depleted_beams = arguments.is_present("continue-depleted-beams");

    PowerLawDistributionConfig {
        min_residual_factor,
        min_deposited_power_density,
        max_propagation_distance,
        continue_depleted_beams,
    }
}
