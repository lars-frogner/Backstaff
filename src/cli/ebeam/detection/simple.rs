//! Command line interface for the simple reconnection site detector.

use crate::{
    add_subcommand_combinations,
    cli::{
        ebeam::{
            accelerator::simple_power_law::create_simple_power_law_accelerator_subcommand,
            distribution::power_law::create_power_law_distribution_subcommand,
            propagator::{
                analytical::create_analytical_propagator_subcommand,
                fp_characteristics::create_characteristics_propagator_subcommand,
            },
        },
        interpolation::poly_fit::create_poly_fit_interpolator_subcommand,
        tracing::stepping::rkf::create_rkf_stepper_subcommand,
        utils,
    },
    ebeam::detection::simple::SimpleReconnectionSiteDetectorConfig,
    io::snapshot::{fpa, SnapshotParameters},
    update_command_graph,
};
use clap::{Arg, ArgMatches, Command};

/// Creates a subcommand for using the simple reconnection site detector.
pub fn create_simple_reconnection_site_detector_subcommand(
    _parent_command_name: &'static str,
) -> Command<'static> {
    let command_name = "simple_detector";

    update_command_graph!(_parent_command_name, command_name);

    let command = Command::new(command_name)
        .about("Use the simple reconnection site detection method")
        .long_about(
            "Use the simple reconnection site detection method.\n\
             Evaluates a reconnection factor in each grid cell and considers a reconnection\n\
             site to be where the factor exceeds a given threshold. The reconnection factor\n\
             indicates changes in the magnetic topology and is described by Biskamp (2005).",
        )
        .arg(
            Arg::new("reconnection-factor-threshold")
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
            Arg::new("detection-depth-limits")
                .long("detection-depth-limits")
                .require_equals(true)
                .use_value_delimiter(true)
                .require_value_delimiter(true)
                .allow_hyphen_values(true)
                .value_names(&["MIN", "MAX"])
                .help(
                    "Smallest and largest depth at which reconnection sites will be \n\
                     detected [Mm] [default: from param file]",
                )
                .takes_value(true)
                .number_of_values(2),
        )
        .subcommand(create_power_law_distribution_subcommand(command_name))
        .subcommand(create_simple_power_law_accelerator_subcommand(command_name))
        .subcommand(create_analytical_propagator_subcommand(command_name))
        .subcommand(create_characteristics_propagator_subcommand(command_name));

    add_subcommand_combinations!(command, command_name, false; poly_fit_interpolator, rkf_stepper)
}

/// Determines simple reconnection site detector parameters
/// based on provided options and values in parameter file.
pub fn construct_simple_reconnection_site_detector_config_from_options(
    arguments: &ArgMatches,
    parameters: &dyn SnapshotParameters,
) -> SimpleReconnectionSiteDetectorConfig {
    let reconnection_factor_threshold = utils::get_value_from_param_file_argument_with_default(
        parameters,
        arguments,
        "reconnection-factor-threshold",
        "krec_lim",
        &|krec_lim: fpa| krec_lim,
        SimpleReconnectionSiteDetectorConfig::DEFAULT_RECONNECTION_FACTOR_THRESHOLD,
    );
    let detection_depth_limits = utils::get_values_from_param_file_argument_with_defaults(
        parameters,
        arguments,
        "detection-depth-limits",
        &["z_rec_ulim", "z_rec_llim"],
        &|lim: fpa| lim,
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
