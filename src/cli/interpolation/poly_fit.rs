//! Command line interface for interpolation by polynomial fitting.

use crate::cli;
use crate::interpolation::poly_fit::PolyFitInterpolatorConfig;
use clap::{App, Arg, ArgMatches};

/// Adds arguments for parameters used by the polynomial fitting interpolator.
pub fn add_poly_fit_interpolator_options_to_subcommand<'a, 'b>(app: App<'a, 'b>) -> App<'a, 'b> {
    app
    .arg(
        Arg::with_name("interpolation-order")
            .long("interpolation-order")
            .value_name("ORDER")
            .long_help("Order of the polynomials to fit when interpolating field values")
            .next_line_help(true)
            .takes_value(true)
            .possible_values(&["1", "2", "3", "4", "5"])
            .default_value("3"),
    )
    .arg(
        Arg::with_name("variation-threshold-for-linear-interpolation")
            .long("variation-threshold-for-linear-interpolation")
            .value_name("VALUE")
            .long_help("Linear interpolation is used when a normalized variance of the values surrounding the interpolation point exceeds this")
            .next_line_help(true)
            .takes_value(true)
            .default_value("0.3"),
    )
}

/// Sets polynomial fitting interpolator parameters based on present arguments.
pub fn configure_poly_fit_interpolator_from_options(
    config: &mut PolyFitInterpolatorConfig,
    arguments: &ArgMatches,
) {
    cli::assign_value_from_parseable_argument(&mut config.order, arguments, "interpolation-order");
    cli::assign_value_from_parseable_argument(
        &mut config.variation_threshold_for_linear,
        arguments,
        "variation-threshold-for-linear-interpolation",
    );
}
