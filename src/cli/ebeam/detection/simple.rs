//! Command line interface for the simple reconnection site detector.

use crate::cli;
use crate::ebeam::detection::simple::SimpleReconnectionSiteDetectorConfig;
use crate::grid::Grid3;
use crate::io::snapshot::{fdt, SnapshotReader3};
use clap::{App, Arg, ArgMatches, SubCommand};

/// Creates a subcommand for using the simple reconnection site detector.
pub fn create_simple_reconnection_site_detector_subcommand<'a, 'b>() -> App<'a, 'b> {
    SubCommand::with_name("simple_detector")
        .about("Use the simple reconnection site detection method")
        .long_about(
            "Use the simple reconnection site detection method.\n\
             Evaluates a reconnection factor in each grid cell and considers a reconnection\n\
             site to be where the factor exceeds a given threshold. The reconnection factor\n\
             indicates changes in the magnetic topology and is described by Biskamp (2005).",
        )
        .help_message("Print help information")
        .arg(
            Arg::with_name("reconnection-factor-threshold")
                .long("reconnection-factor-threshold")
                .require_equals(true)
                .value_name("VALUE")
                .help(
                    "Reconnection sites will be detected where the reconnection factor\n\
                     value is larger than this [default: from param file]",
                )
                .takes_value(true),
        )
        .arg(
            Arg::with_name("detection-depth-limits")
                .long("detection-depth-limits")
                .require_equals(true)
                .require_delimiter(true)
                .allow_hyphen_values(true)
                .value_names(&["MIN", "MAX"])
                .help(
                    "Smallest and largest depth at which reconnection sites will be \n\
                     detected [Mm] [default: from param file]",
                )
                .takes_value(true),
        )
}

/// Determines simple reconnection site detector parameters
/// based on provided options and values in parameter file.
pub fn construct_simple_reconnection_site_detector_config_from_options<G: Grid3<fdt>>(
    arguments: &ArgMatches,
    reader: &SnapshotReader3<G>,
) -> SimpleReconnectionSiteDetectorConfig {
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
        reconnection_factor_threshold,
        min_detection_depth: detection_depth_limits[0],
        max_detection_depth: detection_depth_limits[1],
    }
}
