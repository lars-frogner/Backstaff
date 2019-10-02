//! Command line interface for the power-law electron distribution.

use crate::cli;
use crate::ebeam::distribution::power_law::PowerLawDistributionConfig;
use clap::{App, Arg, ArgMatches};

/// Adds arguments for parameters used by the power-law distribution.
pub fn add_power_law_distribution_options_to_subcommand<'a, 'b>(app: App<'a, 'b>) -> App<'a, 'b> {
    app
    .arg(
        Arg::with_name("min-remaining-power-density")
            .long("min-remaining-power-density")
            .value_name("VALUE")
            .long_help("Distributions with remaining power densities smaller than this value are discarded [erg/(cm^3 s)] [default: from param file]")
            .next_line_help(true)
            .takes_value(true),
    )
}

/// Sets power-law distribution parameters based on present arguments.
pub fn configure_power_law_distribution_from_options(
    config: &mut PowerLawDistributionConfig,
    arguments: &ArgMatches,
) {
    cli::assign_value_from_parseable_argument(
        &mut config.min_remaining_power_density,
        arguments,
        "min-remaining-power-density",
    );
}
