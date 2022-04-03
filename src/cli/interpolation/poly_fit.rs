//! Command line interface for interpolation by polynomial fitting.

use crate::{cli::utils, interpolation::poly_fit::PolyFitInterpolatorConfig};
use clap::{Arg, ArgMatches, Command};

/// Creates a subcommand for using the polynomial fitting interpolator.
pub fn create_poly_fit_interpolator_subcommand(
    parent_command_name: &'static str,
) -> Command<'static> {
    let command_name = "poly_fit_interpolator";

    crate::cli::command_graph::insert_command_graph_edge(parent_command_name, command_name);

    Command::new(command_name)
        .about("Use the polynomial fitting interpolator")
        .long_about(
            "Use the polynomial fitting interpolator.\n\
             An interpolated value is found by evaluating a polynomial of a certain order\n\
             fitted to the surrounding points. To reduce overshoot, linear interpolation\n\
             can be engaged automatically in regions with high variance.",
        )
        .arg(
            Arg::new("order")
                .long("order")
                .require_equals(true)
                .value_name("NUMBER")
                .help("Order of the polynomials to fit when interpolating field values\n")
                .takes_value(true)
                .possible_values(&["1", "2", "3", "4", "5"])
                .default_value("3"),
        )
        .arg(
            Arg::new("variation-threshold")
                .long("variation-threshold")
                .require_equals(true)
                .value_name("VALUE")
                .help(
                    "Linear interpolation is used when a normalized variance of the values\n\
                     surrounding the interpolation point exceeds this",
                )
                .takes_value(true)
                .default_value("0.3"),
        )
}

/// Determines polynomial fitting interpolator parameters based on
/// provided options.
pub fn construct_poly_fit_interpolator_config_from_options(
    arguments: &ArgMatches,
) -> PolyFitInterpolatorConfig {
    let order = utils::get_value_from_required_parseable_argument(arguments, "order");
    let variation_threshold_for_linear =
        utils::get_value_from_required_parseable_argument(arguments, "variation-threshold");
    PolyFitInterpolatorConfig {
        order,
        variation_threshold_for_linear,
    }
}
