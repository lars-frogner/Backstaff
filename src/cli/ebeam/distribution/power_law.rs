//! Command line interface for the power-law electron distribution.

use crate::{
    cli::utils,
    ebeam::distribution::power_law::PowerLawDistributionConfig,
    grid::Grid3,
    io::snapshot::{fdt, SnapshotReader3},
};
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
                     decreased below this limit, given that the deposited power per distance is\n\
                     smaller than its lower limit [default: from param file]",
                )
                .takes_value(true),
        )
        .arg(
            Arg::with_name("min-deposited-power-per-distance")
                .long("min-deposited-power-per-distance")
                .require_equals(true)
                .value_name("VALUE")
                .help(
                    "Distributions are considered depleted when the deposited power per distance\n\
                     [erg/s/cm] has decreased below this limit, given that the residual energy factor\n\
                     is smaller than its lower limit [default: from param file]",
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
            Arg::with_name("outside-deposition-threshold")
                .long("outside-deposition-threshold")
                .require_equals(true)
                .value_name("VALUE")
                .help(
                    "Maximum distance outside the initial extended acceleration region the\n\
                     distribution can propagate before energy deposition starts [Mm]\n\
                     [default: from param file]",
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
pub fn construct_power_law_distribution_config_from_options<G, R>(
    arguments: &ArgMatches,
    reader: &R,
) -> PowerLawDistributionConfig
where
    G: Grid3<fdt>,
    R: SnapshotReader3<G>,
{
    let min_residual_factor = utils::get_value_from_param_file_argument_with_default(
        reader,
        arguments,
        "min-residual-factor",
        "min_residual",
        &|min_residual| min_residual,
        PowerLawDistributionConfig::DEFAULT_MIN_RESIDUAL_FACTOR,
    );
    let min_deposited_power_per_distance = utils::get_value_from_param_file_argument_with_default(
        reader,
        arguments,
        "min-deposited-power-per-distance",
        "min_dep_en",
        &|min_dep_en| min_dep_en,
        PowerLawDistributionConfig::DEFAULT_MIN_DEPOSITED_POWER_PER_DISTANCE,
    );
    let max_propagation_distance = utils::get_value_from_param_file_argument_with_default(
        reader,
        arguments,
        "max-propagation-distance",
        "max_dist",
        &|max_dist| max_dist,
        PowerLawDistributionConfig::DEFAULT_MAX_PROPAGATION_DISTANCE,
    );
    let outside_deposition_threshold = utils::get_value_from_param_file_argument_with_default(
        reader,
        arguments,
        "outside-deposition-threshold",
        "out_dep_thresh",
        &|out_dep_thresh| out_dep_thresh,
        PowerLawDistributionConfig::DEFAULT_OUTSIDE_DEPOSITION_THRESHOLD,
    );
    let continue_depleted_beams = arguments.is_present("continue-depleted-beams");

    PowerLawDistributionConfig {
        min_residual_factor,
        min_deposited_power_per_distance,
        max_propagation_distance,
        outside_deposition_threshold,
        continue_depleted_beams,
    }
}
