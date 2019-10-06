//! Command line interface for the simple reconnection site detector.

use crate::cli;
use crate::ebeam::detection::simple::{
    ReconnectionFactorType, SimpleReconnectionSiteDetectorConfig,
};
use crate::grid::Grid3;
use crate::io::snapshot::{fdt, SnapshotReader3};
use clap::{App, Arg, ArgMatches, SubCommand};

/// Creates a subcommand for using the simple reconnection site detector.
pub fn create_simple_reconnection_site_detector_subcommand<'a, 'b>() -> App<'a, 'b> {
    let app = SubCommand::with_name("simple_detector")
        .about("Use the simple reconnection site detection method")
        .long_about(
            "Use the simple reconnection site detection method.\n\
             Evaluates a reconnection factor in each grid cell and considers a reconnection\n\
             site to be where the factor exceeds a given threshold. The reconnection factor\n\
             indicates changes in the magnetic topology and is described by Biskamp (2005)",
        );
    add_simple_reconnection_site_detector_options_to_subcommand(app)
}

/// Adds arguments for parameters used by the simple reconnection site detector.
pub fn add_simple_reconnection_site_detector_options_to_subcommand<'a, 'b>(
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
            .next_line_help(true)
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
            .next_line_help(true)
            .takes_value(true),
    )
    .arg(
        Arg::with_name("detection-depth-limits")
            .long("detection-depth-limits")
            .value_names(&["MIN, MAX"])
            .long_help(
                "Smallest and largest depth at which reconnection sites will be detected [Mm] [default: from param file]",
            )
            .next_line_help(true)
            .takes_value(true),
    )
}

/// Determines simple reconnection site detector parameters
/// based on provided options and values in parameter file.
pub fn construct_simple_reconnection_site_detector_config_from_options<G: Grid3<fdt>>(
    arguments: &ArgMatches,
    reader: &SnapshotReader3<G>,
) -> SimpleReconnectionSiteDetectorConfig {
    let reconnection_factor_type = match arguments.value_of("reconnection-factor-type") {
        Some("normalized") => ReconnectionFactorType::Normalized,
        Some("standard") => ReconnectionFactorType::Standard,
        None => reader.get_converted_numerical_param_or_fallback_to_default_with_warning(
            "reconnection-factor-type",
            "norm_krec",
            &|norm_krec: u8| {
                if norm_krec > 0 {
                    ReconnectionFactorType::Normalized
                } else {
                    ReconnectionFactorType::Standard
                }
            },
            SimpleReconnectionSiteDetectorConfig::DEFAULT_RECONNECTION_FACTOR_TYPE,
        ),
        Some(invalid) => panic!("Invalid reconnection-factor-type: {}", invalid),
    };
    let reconnection_factor_threshold = cli::get_value_from_param_file_argument_with_default(
        reader,
        arguments,
        "reconnection-factor-threshold",
        "krec_lim",
        &|krec_lim| krec_lim,
        SimpleReconnectionSiteDetectorConfig::DEFAULT_RECONNECTION_FACTOR_THRESHOLD,
    );
    let detection_depth_limits = cli::get_values_from_param_file_argument_with_defaults(
        reader,
        arguments,
        "detection-depth-limits",
        &["z_rec_ulim", "z_rec_llim"],
        &|lim| lim,
        &[
            SimpleReconnectionSiteDetectorConfig::DEFAULT_MIN_DETECTION_DEPTH,
            SimpleReconnectionSiteDetectorConfig::DEFAULT_MAX_DETECTION_DEPTH,
        ],
    );
    SimpleReconnectionSiteDetectorConfig {
        reconnection_factor_type,
        reconnection_factor_threshold,
        min_detection_depth: detection_depth_limits[0],
        max_detection_depth: detection_depth_limits[1],
    }
}
