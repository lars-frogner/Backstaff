//! Command line interface for the simple power-law distribution accelerator.

use crate::cli;
use crate::ebeam::distribution::power_law::acceleration::simple::SimplePowerLawAccelerationConfig;
use clap::{App, Arg, ArgMatches};

/// Adds arguments for parameters used by the simple power-law distribution accelerator.
pub fn add_simple_power_law_accelerator_options_to_subcommand<'a, 'b>(
    app: App<'a, 'b>,
) -> App<'a, 'b> {
    app.arg(
        Arg::with_name("ignore-rejection")
            .long("ignore-rejection")
            .long_help("Generate beams even when they meet a rejection condition"),
    )
    .arg(
        Arg::with_name("min-total-power-density")
            .long("min-total-power-density")
            .value_name("VALUE")
            .long_help("Distributions with total power densities smaller than this value are discarded [erg/(cm^3 s)] [default: from param file]")
            .next_line_help(true)
            .takes_value(true),
    )
    .arg(
        Arg::with_name("min-estimated-depletion-distance")
            .long("min-estimated-depletion-distance")
            .value_name("VALUE")
            .long_help("Distributions with an initial estimated depletion distance smaller than this value are discarded [cm] [default: from param file]")
            .next_line_help(true)
            .takes_value(true),
    )
    .arg(
        Arg::with_name("max-acceleration-angle")
            .long("max-acceleration-angle")
            .value_name("VALUE")
            .long_help("Distributions with acceleration directions angled more than this away from the magnetic field axis are discarded [deg]")
            .next_line_help(true)
            .takes_value(true)
            .default_value("70.0"),
    ).arg(
        Arg::with_name("initial-cutoff-energy-guess")
            .long("initial-cutoff-energy-guess")
            .value_name("VALUE")
            .long_help("Initial guess to use when estimating lower cut-off energy [keV]")
            .next_line_help(true)
            .takes_value(true)
            .default_value("4.0"),
    )
    .arg(
        Arg::with_name("acceptable-root-finding-error")
            .long("acceptable-root-finding-error")
            .value_name("VALUE")
            .long_help("Target relative error when estimating lower cut-off energy")
            .next_line_help(true)
            .takes_value(true)
            .default_value("1e-3"),
    )
    .arg(
        Arg::with_name("max-root-finding-iterations")
            .long("max-root-finding-iterations")
            .value_name("NUMBER")
            .long_help("Maximum number of iterations when estimating lower cut-off energy")
            .next_line_help(true)
            .takes_value(true)
            .default_value("100"),
    )
}

/// Sets simple power-law distribution accelerator parameters based on present arguments.
pub fn configure_simple_power_law_accelerator_from_options(
    config: &mut SimplePowerLawAccelerationConfig,
    arguments: &ArgMatches,
) {
    cli::assign_bool_value_from_flag_presence(
        &mut config.ignore_rejection,
        arguments,
        "ignore-rejection",
    );
    cli::assign_value_from_parseable_argument(
        &mut config.min_total_power_density,
        arguments,
        "min-total-power-density",
    );
    cli::assign_value_from_parseable_argument(
        &mut config.min_estimated_depletion_distance,
        arguments,
        "min-estimated-depletion-distance",
    );
    cli::assign_value_from_parseable_argument(
        &mut config.max_acceleration_angle,
        arguments,
        "max-acceleration-angle",
    );
    cli::assign_value_from_parseable_argument(
        &mut config.initial_cutoff_energy_guess,
        arguments,
        "initial-cutoff-energy-guess",
    );
    cli::assign_value_from_parseable_argument(
        &mut config.acceptable_root_finding_error,
        arguments,
        "acceptable-root-finding-error",
    );
    cli::assign_value_from_parseable_argument(
        &mut config.max_root_finding_iterations,
        arguments,
        "max-root-finding-iterations",
    );
}
