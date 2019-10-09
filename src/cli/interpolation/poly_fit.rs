//! Command line interface for interpolation by polynomial fitting.

use crate::cli;
use crate::interpolation::poly_fit::PolyFitInterpolatorConfig;
use clap::{App, Arg, ArgMatches, SubCommand};

/// Creates a subcommand for using the polynomial fitting interpolator.
pub fn create_poly_fit_interpolator_subcommand<'a, 'b>() -> App<'a, 'b> {
    SubCommand::with_name("poly_fit_interpolator")
        .about("Use the polynomial fitting interpolator")
        .long_about(
            "Use the polynomial fitting interpolator.\n\
             An interpolated value is found by evaluating a polynomial of a certain order\n\
             fitted to the surrounding points. To reduce overshoot, linear interpolation\n\
             can be engaged automatically in regions with high variance.",
        )
        .arg(
            Arg::with_name("order")
                .long("order")
                .require_equals(true)
                .value_name("NUMBER")
                .long_help("Order of the polynomials to fit when interpolating field values\n")
                .next_line_help(true)
                .takes_value(true)
                .possible_values(&["1", "2", "3", "4", "5"])
                .default_value("3"),
        )
        .arg(
            Arg::with_name("variation-threshold")
                .long("variation-threshold")
                .require_equals(true)
                .value_name("VALUE")
                .long_help(
                    "Linear interpolation is used when a normalized variance of the values\n\
                     surrounding the interpolation point exceeds this",
                )
                .next_line_help(true)
                .takes_value(true)
                .default_value("0.3"),
        )
}

/// Determines polynomial fitting interpolator parameters based on
/// provided options.
pub fn construct_poly_fit_interpolator_config_from_options(
    arguments: &ArgMatches,
) -> PolyFitInterpolatorConfig {
    let order = cli::get_value_from_required_parseable_argument(arguments, "order");
    let variation_threshold_for_linear =
        cli::get_value_from_required_parseable_argument(arguments, "variation-threshold");
    PolyFitInterpolatorConfig {
        order,
        variation_threshold_for_linear,
    }
}
