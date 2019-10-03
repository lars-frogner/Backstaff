//! Command line interface for the simple reconnection site detector.

use crate::cli;
use crate::ebeam::detection::simple::SimpleReconnectionSiteDetectorConfig;
use clap::{App, Arg, ArgMatches};

/// Adds arguments for parameters used by the simple reconnection site detector.
pub fn add_simple_power_law_accelerator_options_to_subcommand<'a, 'b>(
    app: App<'a, 'b>,
) -> App<'a, 'b> {
    app.arg(
        Arg::with_name("reconnection-factor-type")
            .long("reconnection-factor-type")
            .value_name("TYPE")
            .long_help(
                "Which version of the reconnection factor to use for seeding\n\
                 [default: from param file]",
            )
            .takes_value(true)
            .possible_values(&["standard", "normalized"]),
    )
    .arg(
        Arg::with_name("reconnection-factor-threshold")
            .long("reconnection-factor-threshold")
            .value_name("VALUE")
            .long_help(
                "Reconnection sites will be detected where the reconnection factor value is larger than this\n\
                 [default: from param file]",
            )
            .takes_value(true),
    )
    .arg(
        Arg::with_name("min-reconnection-detection-depth")
            .long("min-reconnection-detection-depth")
            .value_name("VALUE")
            .long_help(
                "Smallest depth at which reconnection sites will be detected [Mm]\n\
                 [default: from param file]",
            )
            .takes_value(true),
    )
    .arg(
        Arg::with_name("max-reconnection-detection-depth")
            .long("max-reconnection-detection-depth")
            .value_name("VALUE")
            .long_help(
                "Largest depth at which reconnection sites will be detected [Mm]\n\
                 [default: from param file]",
            )
            .takes_value(true),
    )
}

/// Sets simple reconnection site detector parameters based on present arguments.
pub fn configure_simple_power_law_accelerator_from_options(
    config: &mut SimpleReconnectionSiteDetectorConfig,
    arguments: &ArgMatches,
) {
    cli::assign_value_from_selected_argument(
        &mut config.use_normalized_reconnection_factor,
        arguments,
        "reconnection-factor-type",
        &["standard", "normalized"],
        &[false, true],
    );
    cli::assign_value_from_parseable_argument(
        &mut config.reconnection_factor_threshold,
        arguments,
        "reconnection-factor-threshold",
    );
    cli::assign_value_from_parseable_argument(
        &mut config.min_detection_depth,
        arguments,
        "min-reconnection-detection-depth",
    );
    cli::assign_value_from_parseable_argument(
        &mut config.max_detection_depth,
        arguments,
        "max-reconnection-detection-depth",
    );
}
